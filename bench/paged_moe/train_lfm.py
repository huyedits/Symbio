#!/usr/bin/env python3
"""Paged-streaming LoRA trainer for lfm2_moe (LFM2.5-24B-A2B) — 16 GB tier.

Trains LoRA on the top attention layer (38: q/v_proj) plus the in-graph tail
(layer 39 short-conv + embedding_norm + tied head) while ALL weights stream
from SSD through an mmap LRU hot set — the 12.5 GB 4-bit base never needs to
be fully wired, so the ~10.2 GB default Metal working-set cap is not a blocker
and no sysctl is required.

Design mirrors mlx_lm/models/lfm2_moe.py + lfm2.py exactly:
  - routing: fp32 softmax -> += expert_bias -> argpartition top-k ->
    norm_topk_prob divide (+1e-20); weights include the bias, matching source
  - experts: fused switch_mlp gate/up/down [64, out, in] via mx.gather_qmm,
    weights are graph CONSTANTS (never parameters, never dequantized to bf16)
  - attention (layers i in layer_types full_attention): GQA 32/8, head_dim 64,
    rope base 1e6 traditional=False, q/k layernorms on head_dim
  - short conv: in_proj -> split(B,C,x) -> Bx=B*x -> left pad L-1 ->
    depthwise CxK conv -> C*conv -> out_proj
  - layers 0,1 use dense MLP (gate/up/down), rest are MoE
  - tied head: embedding (6-bit) doubles as lm_head via embedding_norm

Trained params: explicit P dict (LoRA A/B on layer-38 q/v_proj only).
Lower layers 0..37 run eagerly with per-layer mx.eval and are freed.

Adapters save in mlx_lm layout (model.layers.38.self_attn.{q,v}_proj.
lora_{a,b}) so mlx_lm.fuse and the Symbio adapter pipeline apply directly.

Run from repo root:
  python3 bench/paged_moe/train_lfm.py --structure-check \
      --model ~/.local/share/mlx-models/LFM2.5-24B-A2B-MLX-4bit
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx  # must precede safetensors (import-order quirk)
import mlx.nn as nn
import numpy as np

from train_paged import Bank, Cfg, ShardReader, linear as _qlinear


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_cfg(model_dir: Path) -> Cfg:
    c = json.loads((model_dir / "config.json").read_text())
    q = c.get("quantization", {})
    return Cfg(
        hidden=c["hidden_size"], heads=c["num_attention_heads"],
        kv_heads=c["num_key_value_heads"],
        head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        experts=c["num_experts"], per_tok=c["num_experts_per_tok"],
        mi=c["moe_intermediate_size"], dense_inter=c["intermediate_size"],
        nlayers=c["num_hidden_layers"],
        layer_types=c["layer_types"],
        num_dense=c.get("num_dense_layers", 0),
        rope_theta=c.get("rope_theta", 1000000.0),
        norm_topk_prob=c.get("norm_topk_prob", False),
        use_expert_bias=c.get("use_expert_bias", False),
        routed_scaling=c.get("routed_scaling_factor", 1.0),
        tie=c.get("tie_word_embeddings", True),
        eps=c.get("norm_eps", 1e-5),
        conv_L=c.get("conv_L_cache", 3),
        conv_bias=c.get("conv_bias", False),
        vocab=c["vocab_size"],
        gs=q["group_size"], bits=q["bits"],
        per_tensor={k: v for k, v in q.items() if isinstance(v, dict)},
        quant_map=q,
    )


def shard_map(model_dir: Path) -> dict:
    idx = model_dir / "model.safetensors.index.json"
    m = {}
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        for name, shard in wm.items():
            m.setdefault(name, str(model_dir / shard))
    else:
        for p in sorted(model_dir.glob("*.safetensors")):
            with open(p, "rb") as fh:
                (hl,) = struct.unpack("<Q", fh.read(8))
                hdr = json.loads(fh.read(hl))
            for k in hdr:
                if k != "__metadata__":
                    m.setdefault(k, str(p))
    import struct  # noqa
    return m


# --------------------------------------------------------------------------
# Quant helpers (lfm2 tensor names)
# --------------------------------------------------------------------------

def qfetch(bank, base):
    w = bank.get(base + ".weight", resident=False)
    s = bank.get(base + ".scales", resident=False) \
        if base + ".scales" in bank.names else None
    b = bank.get(base + ".biases", resident=False) \
        if base + ".biases" in bank.names else None
    return w, s, b


def qlin(x, base, bank, cfg):
    """Quantized linear honoring per-tensor bits (embed is 6-bit here)."""
    o = cfg.per_tensor.get(base, {})
    gs, bits = o.get("group_size", cfg.gs), o.get("bits", cfg.bits)
    w, s, b = qfetch(bank, base)
    if s is None:  # unquantized (bf16) tensor
        return x @ w.T
    return mx.quantized_matmul(x, w, s, b, group_size=gs, bits=bits,
                               transpose=True)


def silu(x):
    return x * mx.sigmoid(x)


def rms(x, w, eps):
    x32 = x.astype(mx.float32)
    v = mx.mean(x32 * x32, axis=-1, keepdims=True)
    return ((x32 / mx.sqrt(v + eps)) * w).astype(x.dtype)


def rope_tables(T, head_dim, theta):
    inv = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    t = mx.arange(T, dtype=mx.float32)
    f = mx.outer(t, inv)
    return mx.cos(f), mx.sin(f)


def rope_apply(x, cos, sin):
    d2 = x.shape[-1] // 2
    x0, x1 = x[..., :d2], x[..., d2:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return mx.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)


# --------------------------------------------------------------------------
# Forward pieces
# --------------------------------------------------------------------------

def attn(x, i, bank, cfg, cos, sin, lora=None):
    """lfm2 attention; lora applies on q/v when given (grad region only)."""
    B, T, _ = x.shape
    h, hd = cfg.heads, cfg.head_dim
    base = f"model.layers.{i}.self_attn"

    def proj(src, name, key=None):
        y = qlin(src, f"{base}.{name}", bank, cfg)
        if key is not None and lora is not None:
            y = y + ((src @ lora[key + ".lora_a"])
                     @ lora[key + ".lora_b"]) * lora["_scale"]
        return y

    q = proj(x, "q_proj", f"{base}.q_proj").reshape(B, T, h, hd)
    k = proj(x, "k_proj").reshape(B, T, cfg.kv_heads, hd)
    v = proj(x, "v_proj", f"{base}.v_proj").reshape(B, T, cfg.kv_heads, hd)
    q = rms(q, bank.get(f"{base}.q_layernorm.weight", resident=True), cfg.eps)
    k = rms(k, bank.get(f"{base}.k_layernorm.weight", resident=True), cfg.eps)
    q = rope_apply(q.transpose(0, 2, 1, 3), cos, sin)
    k = rope_apply(k.transpose(0, 2, 1, 3), cos, sin)
    v = v.transpose(0, 2, 1, 3)
    groups = h // cfg.kv_heads
    k = mx.repeat(k, groups, axis=1)
    v = mx.repeat(v, groups, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
    mask = mx.triu(mx.full((T, T), -mx.inf), k=1).astype(scores.dtype)
    ctx = mx.softmax(scores + mask, axis=-1).astype(x.dtype) @ v
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, h * hd)
    return qlin(ctx, f"{base}.out_proj", bank, cfg)


def short_conv(x, i, bank, cfg):
    """lfm2 ShortConv, graph-safe explicit depthwise 3-tap (causal)."""
    B, T, H = x.shape
    L = cfg.conv_L
    base = f"model.layers.{i}.conv"
    Bcx = qlin(x, f"{base}.in_proj", bank, cfg).reshape(B, T, 3, H)
    Bg, Cg, xg = Bcx[..., 0, :], Bcx[..., 1, :], Bcx[..., 2, :]
    bx = Bg * xg                                       # [B, T, H]
    bx = mx.pad(bx, [(0, 0), (L - 1, 0), (0, 0)])      # causal left pad
    w = bank.get(f"{base}.conv.weight", resident=True)  # base ends in .conv
    wt = w.reshape(w.shape[0], -1)                     # [H, L]
    if wt.shape[1] != L:
        wt = wt.T                                      # ensure [H, L]
    acc = mx.zeros((B, T, H), dtype=bx.dtype)
    for kk in range(L):
        win = bx[:, kk:kk + T, :]
        acc = acc + win * wt[:, kk][None, None, :]     # tap kk per channel
    return qlin(Cg * acc, f"{base}.out_proj", bank, cfg)


def moe(x, i, bank, cfg, p=None):
    """lfm2 MoE: gate bf16 + expert_bias; fused experts via gather_qmm.

    With p and cfg.spawn: N delta-expert slots ride alongside the 64 base
    experts — trainable router rows extend the gate logits, and each spawned
    slot contributes (x @ lora_a) @ lora_b weighted by its routing score.
    """
    base = f"model.layers.{i}.feed_forward"
    B, T, H = x.shape
    flat = x.reshape(B * T, H)
    g = bank.get(f"{base}.gate.weight", resident=True)
    logits = (flat @ g.T).astype(mx.float32)
    if cfg.use_expert_bias:
        logits = logits + bank.get(f"{base}.expert_bias", resident=True)
    n_spawn = cfg.spawn if p is not None else 0
    if n_spawn:
        r = p[f"model.layers.{i}.spawn_router"]          # [n_spawn, H]
        logits = mx.concatenate(
            [logits, (flat @ r.T).astype(mx.float32)], axis=-1)
    k = cfg.per_tok
    gates = mx.softmax(logits, axis=-1)
    inds = mx.stop_gradient(
        mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if cfg.norm_topk_prob:
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)

    xe = flat[:, None, None, :]
    wname = f"{base}.switch_mlp"

    # base experts: gather with clamped indices; zero out spawned-slot rows
    def slot(part, src):
        w, sb, bb = qfetch(bank, f"{wname}.{part}")
        y = mx.gather_qmm(src, w, sb, bb,
                          rhs_indices=mx.minimum(inds, cfg.experts - 1).astype(mx.int32),
                          transpose=True, group_size=cfg.gs, bits=cfg.bits)
        return y[..., 0, :]

    base_keep = (inds < cfg.experts).astype(x.dtype)          # [N, k]
    gi = slot("gate_proj", xe)
    ui = slot("up_proj", xe)
    dn = slot("down_proj", (silu(gi) * ui)[:, :, None, :])
    y = ((dn * scores[..., None]) * base_keep[..., None]).sum(-2)

    # spawned delta experts
    for s in range(n_spawn):
        A = p[f"model.layers.{i}.spawn.{s}.lora_a"]           # [H, r]
        Bl = p[f"model.layers.{i}.spawn.{s}.lora_b"]          # [r, H]
        w_s = ((inds == (cfg.experts + s)) * scores).sum(-1)  # [N]
        y = y + (((flat @ A) @ Bl) * w_s[:, None])
    return y.reshape(x.shape)


def dense_mlp(x, i, bank, cfg):
    base = f"model.layers.{i}.feed_forward"
    flat = x.reshape(x.shape[0] * x.shape[1], cfg.hidden)
    g = qlin(flat, f"{base}.gate_proj", bank, cfg)
    u = qlin(flat, f"{base}.up_proj", bank, cfg)
    d = qlin(silu(g) * u, f"{base}.down_proj", bank, cfg)
    return d.reshape(x.shape)


def layer_fwd_exact(x, i, bank, cfg, cos, sin, lora=None):
    nx = rms(x, bank.get(f"model.layers.{i}.operator_norm.weight",
                         resident=True), cfg.eps)
    r = (attn(nx, i, bank, cfg, cos, sin, lora)
         if cfg.layer_types[i] == "full_attention"
         else short_conv(nx, i, bank, cfg))
    h = x + r
    h = rms(h, bank.get(f"model.layers.{i}.ffn_norm.weight",
                        resident=True), cfg.eps)
    ff = (dense_mlp(h, i, bank, cfg) if i < cfg.num_dense
          else moe(h, i, bank, cfg, p=lora))
    return h + ff


def embed_rows(bank, ids, cfg):
    o = cfg.per_tensor.get("model.embed_tokens", {})
    gs, bits = o.get("group_size", cfg.gs), o.get("bits", cfg.bits)
    w = bank.get("model.embed_tokens.weight")
    s = bank.get("model.embed_tokens.scales")
    b = bank.get("model.embed_tokens.biases") \
        if "model.embed_tokens.biases" in bank.names else None
    idx = mx.array(ids).astype(mx.int32)
    pw, ps = mx.take(w, idx, axis=0), mx.take(s, idx, axis=0)
    pb = mx.take(b, idx, axis=0) if b is not None else None
    return mx.dequantize(pw, ps, pb, group_size=gs, bits=bits)


# --------------------------------------------------------------------------
# Seam probes
# --------------------------------------------------------------------------

def structure_check(cfg: Cfg, bank: Bank):
    ok = True

    def probe(name, fn):
        nonlocal ok
        try:
            fn()
            print(f"  [ok]   {name}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:160]}")

    def t_conv():
        """explicit 3-tap depthwise vs nn.Conv1d reference."""
        Hn, T_, L_ = 8, 6, cfg.conv_L
        x = mx.random.normal((2, T_ + L_ - 1, Hn))
        w = mx.random.normal((Hn, L_, 1))     # [out, kernel, in/groups]
        ref = nn.Conv1d(Hn, Hn, kernel_size=L_, groups=Hn, bias=False)
        ref.weight = w
        mine = mx.zeros((2, T_, Hn))
        for kk in range(L_):
            mine = mine + x[:, kk:kk + T_, :] * w[:, kk, 0][None, None, :]
        got = ref(x)
        err = float(mx.abs(mine.astype(mx.float32)
                           - got.astype(mx.float32)).max())
        assert err < 1e-3, f"conv mismatch {err}"
    probe("depthwise conv vs nn.Conv1d", t_conv)

    def t_gather():
        base = f"model.layers.38.feed_forward.switch_mlp.down_proj"
        w, s, b = qfetch(bank, base)
        ref = mx.dequantize(w, s, b, group_size=cfg.gs, bits=cfg.bits)
        x = mx.random.normal((2, 1, 1, ref.shape[2])).astype(mx.bfloat16)
        idx = mx.array([[0, 41], [7, 63]]).astype(mx.int32)
        got = mx.gather_qmm(x, w, s, b, rhs_indices=idx, transpose=True,
                            group_size=cfg.gs, bits=cfg.bits)
        for bi in range(2):
            for si in range(2):
                e = int(idx[bi, si])
                refv = x[bi, 0, 0].astype(mx.float32) @ ref[e].astype(mx.float32).T
                err = float(mx.abs(got[bi, si, 0].astype(mx.float32) - refv).max())
                assert err < 1.0, f"row {bi},{si} err {err}"
    probe("gather_qmm parity", t_gather)

    def t_gate_unquant():
        g = bank.get("model.layers.38.feed_forward.gate.weight")
        assert g.shape == (cfg.experts, cfg.hidden), f"gate {g.shape}"
    probe("router weights unquantized", t_gate_unquant)

    def t_coverage():
        need = ["model.embed_tokens.weight", "model.embed_tokens.scales",
                "model.layers.38.self_attn.q_proj.weight",
                "model.layers.38.self_attn.q_layernorm.weight",
                "model.layers.38.feed_forward.expert_bias",
                "model.layers.39.conv.conv.weight",
                "model.layers.39.conv.in_proj.scales",
                "model.layers.38.operator_norm.weight",
                "model.layers.38.ffn_norm.weight"]
        missing = [n for n in need if n not in bank.names]
        assert not missing, f"missing: {missing[:6]}"
    probe("tensor coverage", t_coverage)

    print(f"  {bank.stats()}")
    return ok


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

def pin_dense(bank, cfg):
    for i in range(cfg.nlayers):
        base = f"model.layers.{i}"
        for n in (f"{base}.operator_norm.weight", f"{base}.ffn_norm.weight"):
            bank.get(n, resident=True)
        if cfg.layer_types[i] == "full_attention":
            for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
                for sfx in (".weight", ".scales", ".biases"):
                    n = f"{base}.self_attn.{p}{sfx}"
                    if n in bank.names:
                        bank.get(n, resident=True)
            for n in (f"{base}.self_attn.q_layernorm.weight",
                      f"{base}.self_attn.k_layernorm.weight"):
                bank.get(n, resident=True)
        else:
            for p in ("in_proj", "out_proj"):
                for sfx in (".weight", ".scales", ".biases"):
                    n = f"{base}.conv.{p}{sfx}"
                    if n in bank.names:
                        bank.get(n, resident=True)
            bank.get(f"{base}.conv.conv.weight", resident=True)
        if f"{base}.feed_forward.gate.weight" in bank.names:  # MoE layers only
            bank.get(f"{base}.feed_forward.gate.weight", resident=True)
        if f"{base}.feed_forward.expert_bias" in bank.names:
            bank.get(f"{base}.feed_forward.expert_bias", resident=True)


def main():
    import struct  # needed by shard_map fallback

    ap = argparse.ArgumentParser(
        description="Paged-streaming lfm2_moe LoRA trainer (16 GB)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", help="dir with train.jsonl/valid.jsonl")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--hot-budget-mb", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--out", default="adapters_lfm_paged")
    ap.add_argument("--spawn", type=int, default=0,
                    help="number of new delta-expert slots to grow per MoE layer")
    ap.add_argument("--resume-adapter", default=None,
                    help="load existing adapters and continue (with --start-iter)")
    ap.add_argument("--start-iter", type=int, default=0)
    ap.add_argument("--structure-check", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    cfg = load_cfg(model_dir)
    bank = Bank(shard_map(model_dir), budget_mb=args.hot_budget_mb)
    attn_idx = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    cfg.nlayers_attn = attn_idx
    print(f"model: {model_dir.name} layers={cfg.nlayers} "
          f"experts={cfg.experts}x{cfg.per_tok} bits={cfg.bits} "
          f"attn@{attn_idx} dense={cfg.num_dense}")

    if args.structure_check:
        print("probing seams:")
        good = structure_check(cfg, bank)
        print("RESULT:", "OK" if good else "FAILED")
        return 0 if good else 2

    pin_dense(bank, cfg)
    for p in ("model.embed_tokens.weight", "model.embed_tokens.scales",
              "model.embed_tokens.biases"):
        if p in bank.names:
            bank.get(p, resident=True)
    print(f"pinned dense: {bank.stats()}")

    split_attn = attn_idx[-1]          # 38: last attention layer -> grad region
    grad_layers = [split_attn]         # attention layer with LoRA
    conv_tail = list(range(split_attn + 1, cfg.nlayers))  # conv layers 39
    H, rank = cfg.hidden, args.rank
    hd = cfg.head_dim
    key = mx.random.key(11)
    P = {}
    P_SCALE = args.scale
    for i in grad_layers:
        base = f"model.layers.{i}.self_attn"
        for proj, D in (("q_proj", cfg.heads * hd),
                        ("v_proj", cfg.kv_heads * hd)):
            key, sub = mx.random.split(key)
            P[f"{base}.{proj}.lora_a"] = mx.random.normal((H, rank), key=sub) * 0.01
            P[f"{base}.{proj}.lora_b"] = mx.zeros((rank, D))
    trainable = sum(math.prod(v.shape) for v in P.values())
    print(f"trainable: {trainable / 1e6:.2f}M params (rank {rank}, "
          f"layer {grad_layers})")

    # ---- spawn delta experts: router rows near frozen-gate mean, A/B pairs --
    cfg.spawn = args.spawn
    spawned_layers = []
    if cfg.spawn:
        n_sp = cfg.spawn
        for i in grad_layers:
            if i < cfg.num_dense:
                continue
            base = f"model.layers.{i}.feed_forward"
            if f"{base}.gate.weight" not in bank.names:
                continue
            g = bank.get(f"{base}.gate.weight", resident=True)  # [E, H]
            gmu, gsd = g.mean(0), g.std(0)
            key, sub = mx.random.split(key)
            P[f"model.layers.{i}.spawn_router"] = (
                mx.repeat(gmu[None, :], n_sp, 0)
                + 0.15 * gsd * mx.random.normal((n_sp, H), key=sub))
            for s in range(n_sp):
                key, sub = mx.random.split(key)
                P[f"model.layers.{i}.spawn.{s}.lora_a"] = \
                    mx.random.normal((H, rank), key=sub) * 0.01
                P[f"model.layers.{i}.spawn.{s}.lora_b"] = \
                    mx.zeros((rank, H))
            spawned_layers.append(i)
        trainable = sum(math.prod(v.shape) for v in P.values())
        print(f"spawned {n_sp} delta-experts on layers {spawned_layers} "
              f"(router rows seeded near gate mean)")
        print(f"trainable: {trainable / 1e6:.2f}M params total")

    if args.resume_adapter:
        from safetensors import safe_open
        adp = Path(args.resume_adapter)
        with safe_open(str(adp), framework="mlx") as s:
            for k in s.keys():
                if k in P or k.startswith(("model.layers.", "spawn.")):
                    P[k] = s.get_tensor(k)
        print(f"resumed adapter state from {adp.name} (from iter "
              f"{args.start_iter})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    if args.data:
        rows = [json.loads(ln) for ln in
                (Path(args.data) / "train.jsonl").read_text().splitlines()
                if ln.strip()]
    else:
        rows = [{"messages": [
            {"role": "user", "content": "reply with exactly: SMOKE OK"},
            {"role": "assistant", "content": "SMOKE OK " + "ok " * 30}]}]
        print("no --data: smoke example")
    if not rows:
        sys.exit("no training rows")

    def to_ids(row):
        msgs = row.get("messages") or [{"role": "user", "content": row["text"]}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        ids = [int(t) for t in tok(text)["input_ids"]]
        last_a = next((i for i, m in enumerate(msgs)
                       if m.get("role") == "assistant"), len(msgs))
        prefix = [int(t) for t in tok(tok.apply_chat_template(
            msgs[:last_a], tokenize=False,
            add_generation_prompt=True))["input_ids"]]
        n = min(len(prefix), len(ids))
        # Left-truncate: keep the tail, so the target grid always survives.
        if len(ids) > args.max_len:
            drop = len(ids) - args.max_len
            ids = ids[drop:]
            n = max(0, n - drop)
        labels = ([-100] * n) + ids[n:]
        return ids, labels[: args.max_len]

    def compute_stash(ids):
        """Frozen region 0..grad-1, streamed + evaluated layer by layer so
        no two layers' intermediates are ever alive at once."""
        T = len(ids)
        cos, sin = rope_tables(T, cfg.head_dim, cfg.rope_theta)
        x = embed_rows(bank, ids, cfg).astype(mx.bfloat16)[None, :, :]
        for i in range(grad_layers[0]):
            x = layer_fwd_exact(x, i, bank, cfg, cos, sin)
            mx.eval(x)     # free THIS layer before the next streams in
        return x           # [1, T, H] bf16

    def loss_fn(p, stash, ids, labels):
        T = stash.shape[1]
        cos, sin = rope_tables(T, cfg.head_dim, cfg.rope_theta)
        lora = dict(p)
        lora["_scale"] = P_SCALE
        x = stash
        for i in grad_layers:            # grad region: attention + LoRA + MoE
            x = layer_fwd_exact(x, i, bank, cfg, cos, sin, lora=lora)
        for i in conv_tail:              # conv tail, in-graph (grad flows)
            x = layer_fwd_exact(x, i, bank, cfg, cos, sin)
        x = rms(x[0], bank.get("model.embedding_norm.weight",
                               resident=True), cfg.eps)
        logits = qlin(x, "model.embed_tokens", bank, cfg)  # tied head
        tgt = mx.array(labels[1:] + [-100]).astype(mx.int32)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), tgt,
                                     reduction="none")
        valid = (tgt != -100).astype(mx.float32)
        return (ce * valid).sum() / mx.maximum(valid.sum(), 1.0)

    out = Path(args.out)
    t0 = time.time()
    stash_cache = {}
    for it in range(args.start_iter + 1, args.iters + 1):
        ex_idx = (it - 1) % len(rows)
        ids, labels = to_ids(rows[ex_idx])
        if ex_idx not in stash_cache:
            stash_cache[ex_idx] = compute_stash(ids)
        stash = stash_cache[ex_idx]
        lv, grads = mx.value_and_grad(
            lambda p: loss_fn(p, stash, ids, labels))(P)
        for k, g in grads.items():
            P[k] = P[k] - args.lr * g
        lv = float(lv)
        peak = ""
        if hasattr(mx, "get_peak_memory"):
            peak = f" peak {mx.get_peak_memory() / (1 << 20):.0f}MB"
            mx.reset_peak_memory()
        # Harness telemetry contract: training_live.line() parses exactly
        # "Iter N: Train loss X" — the pet and the TUI watch these.
        print(f"Iter {it}: Train loss {lv:.4f} |{peak} | {bank.stats()} | "
              f"{time.time() - t0:6.1f}s", flush=True)
        if it % args.save_every == 0 or it == args.iters:
            out.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(out / "adapters.safetensors"),
                                {k: v.astype(mx.float32) for k, v in P.items()})
            (out / "adapter_config.json").write_text(json.dumps({
                "model": str(model_dir), "rank": args.rank,
                "scale": args.scale, "layers": grad_layers,
                "paged": True, "arch": "lfm2_moe"}))
            print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())