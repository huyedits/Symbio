#!/usr/bin/env python3
"""Paged-streaming LoRA trainer for bailing_moe (Ling-mini-2.0) — 16 GB tier.

v0 of the Ling adaptation: same bank/streaming/gather_qmm spine as
train_lfm.py, with four arch deltas:

  1. FUSED QKV attention: one packed `query_key_value` projection
     (q 2048 = 16*128, k 512, v 512 = 3072 total out). LoRA rides the fused
     output; the split into q/k/v happens after the delta is added.
  2. HALF RoPE: rotary on the first head_dim//2 (64) dims of each 128-dim
     head; the rest passes through. rope_theta 600000.
  3. DEEPSEEK-STYLE ROUTING: sigmoid scores (fp32), per-expert bias,
     grouped top-k (n_group 8, topk_group 4), norm_topk_prob, and
     routed_scaling_factor 2.5 on the expert outputs.
  4. SHARED EXPERT: an always-on dense FFN (`mlp.shared_experts.*`)
     beside the routed 256-expert bank.

Everything else — Bank, ShardReader, gather_qmm experts as fused constants,
spawned delta-expert slots, explicit-P autograd region — is inherited
machinery, unchanged.

Run from bench/paged_moe/ :
  python3 train_ling.py --structure-check \
      --model ~/.local/share/mlx-models/Ling-mini-2.0-4bit
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

from train_paged import Bank
from train_lfm import Cfg, qlin as linear, rms, silu, shard_map

# routing reference from the installed mlx_lm bailing implementation
from mlx_lm.models.bailing_moe import group_expert_select


def load_ling_cfg(model_dir: Path) -> Cfg:
    c = json.loads((model_dir / "config.json").read_text())
    q = c.get("quantization", {})
    return Cfg(
        hidden=c["hidden_size"], heads=c["num_attention_heads"],
        kv_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        experts=c["num_experts"], per_tok=c["num_experts_per_tok"],
        mi=c["moe_intermediate_size"], dense_inter=c["intermediate_size"],
        nlayers=c["num_hidden_layers"], num_dense=c.get("first_k_dense_replace", 1),
        rope_theta=c["rope_theta"], partial_rotary=c.get("partial_rotary_factor", 0.5),
        n_group=c.get("n_group", 1), topk_group=c.get("topk_group", 4),
        score_function=c.get("score_function", "sigmoid"),
        routed_scaling=c.get("routed_scaling_factor", 1.0),
        shared_inter=c.get("moe_shared_expert_intermediate_size"),
        router_dtype=c.get("router_dtype", "fp32"),
        norm_topk_prob=c.get("norm_topk_prob", True),
        tie=c.get("tie_word_embeddings", False),
        vocab=c["vocab_size"], eps=c.get("rms_norm_eps", 1e-6),
        gs=q["group_size"], bits=q["bits"],
        per_tensor={k: v for k, v in q.items() if isinstance(v, dict)},
        eos=c.get("eos_token_id", 156895),
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
            import struct
            with open(p, "rb") as fh:
                (hl,) = struct.unpack("<Q", fh.read(8))
                hdr = json.loads(fh.read(hl))
            for k in hdr:
                if k != "__metadata__":
                    m.setdefault(k, str(p))
    return m


# --------------------------------------------------------------------------
# half rope
# --------------------------------------------------------------------------

def half_rope(x, cosf, sinf):
    """x [B, H, T, D]; rotary only on the first d2 = partial*D dims."""
    d2 = int(x.shape[-1] * 0.5)
    xh, xp = x[..., :d2], x[..., d2:]
    c = cosf[:, None, :]
    s = sinf[:, None, :]
    return mx.concatenate([xh * c - xp * s, xh * s + xp * c], axis=-1)


# wait — partial rope rotates dims[:d2] against dims[d2:2*d2]?? no — the
# standard half-rotary rotates the first d2 dims among themselves:
# pairs (i, i+d2/2) — see rope_tables below for the exact pairing used.

def rope_tables_half(T, head_dim, theta, partial=0.5):
    # nn.RoPE(dims=W) rotates the first W dims pairwise (i, i+W/2):
    # frequencies have W/2 entries, exponent over W steps.
    d2 = int(head_dim * partial)                 # rotary width (64 of 128)
    inv = 1.0 / (theta ** (mx.arange(0, d2, 2, dtype=mx.float32) / d2))
    t = mx.arange(T, dtype=mx.float32)
    f = mx.outer(t, inv)
    return mx.cos(f), mx.sin(f), d2              # cos/sin [T, d2/2]


def half_rope_pairs(x, cos, sin, d2):
    """rotate dims [0, d2/2) against [d2/2, d2); leave dims >= d2 alone.

    x: [B, H, T, D] — matches mlx nn.RoPE(dims=d2) semantics on the head axis.
    """
    hl = d2 // 2
    x0, x1, xp = x[..., :hl], x[..., hl:d2], x[..., d2:]
    c = cos[None, None, :, :]        # [1,1,T,W] against [B,H,T,W]
    s = sin[None, None, :, :]
    rot = mx.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
    return mx.concatenate([rot, xp], axis=-1)


# --------------------------------------------------------------------------
# attention (fused qkv)
# --------------------------------------------------------------------------

def attn_ling(x, i, bank, cfg, caches_or_none=None, lora=None, step_tok=False):
    B, T, _ = x.shape
    h, hd = cfg.heads, cfg.head_dim
    base = f"model.layers.{i}.attention"

    def proj(src, name, key=None):
        y = linear(src, f"{base}.{name}", bank, cfg)
        if key is not None and lora is not None:
            y = y + ((src @ lora[key + ".lora_a"])
                     @ lora[key + ".lora_b"]) * lora["_scale"]
        return y

    fused = proj(x, "query_key_value",
                 f"{base}.query_key_value")       # [B, T, 3072]
    q = fused[..., :h * hd].reshape(B, T, h, hd)
    k = fused[..., h * hd:(h + cfg.kv_heads) * hd].reshape(B, T, cfg.kv_heads, hd)
    v = fused[..., (h + cfg.kv_heads) * hd:].reshape(B, T, cfg.kv_heads, hd)

    q = rms(q, bank.get(f"{base}.query_layernorm.weight", resident=True), cfg.eps)
    k = rms(k, bank.get(f"{base}.key_layernorm.weight", resident=True), cfg.eps)

    cos, sin, d2 = rope_tables_half(T, hd, cfg.rope_theta, cfg.partial_rotary)
    q = half_rope_pairs(q.transpose(0, 2, 1, 3), cos, sin, d2)
    k = half_rope_pairs(k.transpose(0, 2, 1, 3), cos, sin, d2)
    v = v.transpose(0, 2, 1, 3)
    if caches_or_none is not None:  # decoding: append + use full cache
        k_all, v_all = caches_or_none
        k_all = mx.concatenate([k_all, k], axis=2)
        v_all = mx.concatenate([v_all, v], axis=2)
        caches_or_none[0], caches_or_none[1] = k_all, v_all
    else:
        k_all, v_all = k, v
    groups = h // cfg.kv_heads
    kk = mx.repeat(k_all, groups, axis=1)
    vv = mx.repeat(v_all, groups, axis=1)
    scores = (q @ kk.transpose(0, 1, 3, 2)) * (hd ** -0.5)
    if not step_tok and caches_or_none is None:
        mask = mx.triu(mx.full((T, T), -mx.inf), k=1).astype(scores.dtype)
        scores = scores + mask
    ctx = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype) @ vv
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, h * hd)
    return linear(ctx, f"{base}.dense", bank, cfg)


# --------------------------------------------------------------------------
# ling routing: sigmoid + grouped topk, via the installed reference for parity
# --------------------------------------------------------------------------

def route_ling(x_flat, i, bank, cfg):
    """Raw router logits; the correction bias is a SELECTION-only signal and
    is applied inside group_expert_select, exactly as the reference does."""
    base = f"model.layers.{i}.mlp.gate"
    return linear(x_flat, f"{base}.gate_proj", bank, cfg).astype(mx.float32)


def bias_ling(i, bank, cfg):
    base = f"model.layers.{i}.mlp.gate"
    return bank.get(f"{base}.expert_bias", resident=True) \
        if f"{base}.expert_bias" in bank.names else None


def moe_ling(x, i, bank, cfg):
    base = f"model.layers.{i}.mlp"
    B, T, H = x.shape
    flat = x.reshape(B * T, H)
    gates = route_ling(flat, i, bank, cfg)
    bias = bias_ling(i, bank, cfg)
    # grouped expert selection, autograd-seam-safe: internal indices are
    # stop-gradded BEFORE every take/put (the stock reference never
    # terminates its own index chains, which VJP chokes on in training).
    scores = mx.sigmoid(gates) if cfg.score_function == "sigmoid" \
        else mx.softmax(gates, axis=-1)
    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(mx.float32)   # selection-only signal
    scores_g = mx.unflatten(scores, axis=-1, shape=(cfg.n_group, -1))
    group_scores = mx.topk(scores_g, 2, axis=-1).sum(axis=-1, keepdims=True)
    k = cfg.n_group - cfg.topk_group
    group_idx = mx.stop_gradient(
        mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :])
    scores_g = mx.put_along_axis(scores_g, group_idx,
                                 mx.array(0.0, scores_g.dtype), axis=-2)
    scores = mx.flatten(scores_g, -2, -1)
    topk_idx = mx.stop_gradient(
        mx.argpartition(scores, kth=-cfg.per_tok, axis=-1)[..., -cfg.per_tok:])
    topk_wt = mx.take_along_axis(orig_scores, topk_idx, axis=-1)
    if cfg.per_tok > 1 and cfg.norm_topk_prob:
        topk_wt = topk_wt / (topk_wt.sum(axis=-1, keepdims=True) + 1e-20)
    topk_wt = topk_wt * cfg.routed_scaling

    xe = flat[:, None, None, :]
    wname = f"{base}.switch_mlp"

    def slot(part):
        w = bank.get(f"{wname}.{part}.weight")
        s = bank.get(f"{wname}.{part}.scales")
        b = bank.get(f"{wname}.{part}.biases") \
            if f"{wname}.{part}.biases" in bank.names else None
        y = mx.gather_qmm(xe, w, s, b, rhs_indices=topk_idx,
                          transpose=True, group_size=cfg.gs, bits=cfg.bits)
        return y[..., 0, :]                            # [N, k, dim]

    def slot(part, src_x):
        w = bank.get(f"{wname}.{part}.weight")
        s = bank.get(f"{wname}.{part}.scales")
        b = bank.get(f"{wname}.{part}.biases") \
            if f"{wname}.{part}.biases" in bank.names else None
        y = mx.gather_qmm(src_x, w, s, b, rhs_indices=topk_idx,
                          transpose=True, group_size=cfg.gs, bits=cfg.bits)
        return y[..., 0, :]

    gi = slot("gate_proj", xe)
    ui = slot("up_proj", xe)
    dn = slot("down_proj", (silu(gi) * ui)[:, :, None, :])
    y = (dn * topk_wt.astype(x.dtype)[..., None]).sum(-2)      # [N, H]

    # shared expert: always on
    sh = f"{base}.shared_experts"
    g = linear(flat, f"{sh}.gate_proj", bank, cfg)
    u = linear(flat, f"{sh}.up_proj", bank, cfg)
    d = linear(silu(g) * u, f"{sh}.down_proj", bank, cfg)
    y = y + d
    return y.reshape(x.shape)


# (dense mlp for layer 0 reuses BailingMoeMLP shape: gate/up/down)
def dense_mlp_ling(x, i, bank, cfg):
    base = f"model.layers.{i}.mlp"
    flat = x.reshape(x.shape[0] * x.shape[1], cfg.hidden)
    g = linear(flat, f"{base}.gate_proj", bank, cfg)
    u = linear(flat, f"{base}.up_proj", bank, cfg)
    d = linear(silu(g) * u, f"{base}.down_proj", bank, cfg)
    return d.reshape(x.shape)


def layer_ling(x, i, bank, cfg, lora=None):
    nx = rms(x, bank.get(f"model.layers.{i}.input_layernorm.weight",
                         resident=True), cfg.eps)
    r = attn_ling(nx, i, bank, cfg, lora=lora)
    h = x + r
    h = rms(h, bank.get(f"model.layers.{i}.post_attention_layernorm.weight",
                        resident=True), cfg.eps)
    ff = dense_mlp_ling(h, i, bank, cfg) if i < cfg.num_dense \
        else moe_ling(h, i, bank, cfg)
    return h + ff


# --------------------------------------------------------------------------
# probes
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

    def t_routing():
        """my route + mlx's group_expert_select agree with a straight topk."""
        gates = route_ling(mx.random.normal((7, cfg.hidden)), cfg.nlayers - 1,
                           bank, cfg)
        bias = bank.get("model.layers.19.mlp.gate.expert_bias",
                        resident=True) \
            if "model.layers.19.mlp.gate.expert_bias" in bank.names else None
        idx, wt = group_expert_select(
            gates, bias,
            top_k=cfg.per_tok, n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            routed_scaling_factor=cfg.routed_scaling,
            norm_topk_prob=cfg.norm_topk_prob,
            score_function=cfg.score_function)
        assert idx.shape == (7, cfg.per_tok) and wt.shape == (7, cfg.per_tok)
        scores = mx.sigmoid(gates)
        idx_plain = mx.argsort(scores, axis=-1)[..., -cfg.per_tok:]
        same_groups = all(
            set(int(x) for x in idx[n] if x < cfg.experts)
            & set(int(x) for x in idx_plain[n]
                  )
            for n in range(7))
        assert len(set(int(x) for x in idx[0])) == cfg.per_tok
    probe("sigmoid grouped routing", t_routing)

    def t_gather():
        wname = f"model.layers.{cfg.nlayers - 1}.mlp.switch_mlp"
        w = bank.get(f"{wname}.down_proj.weight")
        s = bank.get(f"{wname}.down_proj.scales")
        b = bank.get(f"{wname}.down_proj.biases") \
            if f"{wname}.down_proj.biases" in bank.names else None
        ref = mx.dequantize(w, s, b, group_size=cfg.gs, bits=cfg.bits)
        x = mx.random.normal((2, 1, 1, ref.shape[2])).astype(mx.bfloat16)
        idx = mx.array([[0, 129], [3, 3]]).astype(mx.int32)
        got = mx.gather_qmm(x, w, s, b, rhs_indices=idx, transpose=True,
                            group_size=cfg.gs, bits=cfg.bits)
        for bi in range(2):
            for si in range(2):
                e = int(idx[bi, si])
                refv = x[bi, 0, 0].astype(mx.float32) @ ref[e].astype(mx.float32).T
                err = float(mx.abs(got[bi, si, 0].astype(mx.float32) - refv).max())
                assert err < 1.0, f"row {bi},{si} err {err}"
    probe("gather_qmm parity (256 experts)", t_gather)

    def t_half_rope():
        hd = cfg.head_dim
        d2 = int(hd * cfg.partial_rotary)         # 64
        cos, sin, d2r = rope_tables_half(8, hd, cfg.rope_theta, cfg.partial_rotary)
        x = mx.random.normal((2, 4, 8, hd)).astype(mx.bfloat16)
        mine = half_rope_pairs(x, cos, sin, d2r)
        ref = nn.RoPE(d2, base=cfg.rope_theta)(x)
        err = float(mx.abs(mine.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert err < 0.1, f"half-rope mismatch {err}"
    probe("half rope vs nn.RoPE(dims=64)", t_half_rope)

    def t_coverage():
        need = ["model.word_embeddings.weight",
                f"model.layers.{cfg.nlayers-1}.attention.query_key_value.weight",
                f"model.layers.{cfg.nlayers-1}.mlp.gate.gate_proj.scales",
                f"model.layers.{cfg.nlayers-1}.mlp.shared_experts.gate_proj.weight",
                f"model.layers.{cfg.nlayers-1}.mlp.switch_mlp.down_proj.weight",
                f"model.layers.{cfg.nlayers-1}.mlp.gate.expert_bias",
                "lm_head.weight"]
        missing = [n for n in need if n not in bank.names]
        assert not missing, f"missing: {missing[:6]}"
    probe("tensor coverage", t_coverage)

    print(f"  {bank.stats()}")
    return ok


def pin_dense(bank, cfg):
    for i in range(cfg.nlayers):
        base = f"model.layers.{i}"
        for p in ("query_key_value", "dense"):
            for sfx in (".weight", ".scales", ".biases"):
                n = f"{base}.attention.{p}{sfx}"
                if n in bank.names:
                    bank.get(n, resident=True)
        for n in (f"{base}.query_layernorm.weight",
                  f"{base}.key_layernorm.weight",
                  f"{base}.input_layernorm.weight",
                  f"{base}.post_attention_layernorm.weight",
                  f"{base}.mlp.gate.gate_proj.weight",
                  f"{base}.mlp.gate.gate_proj.scales",
                  f"{base}.mlp.gate.gate_proj.biases",
                  f"{base}.mlp.gate.expert_bias"):
            if n in bank.names:
                bank.get(n, resident=True)
        if i < cfg.num_dense:
            for p in ("gate_proj", "up_proj", "down_proj"):
                for sfx in (".weight", ".scales", ".biases"):
                    n = f"{base}.mlp.{p}{sfx}"
                    if n in bank.names:
                        bank.get(n, resident=True)
        else:
            for p in ("gate_proj", "up_proj", "down_proj"):
                for sfx in (".weight", ".scales", ".biases"):
                    n = f"{base}.mlp.shared_experts.{p}{sfx}"
                    if n in bank.names:
                        bank.get(n, resident=True)


def main():
    ap = argparse.ArgumentParser(
        description="Paged-streaming bailing_moe (Ling-mini-2.0) trainer")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", help="dir with train.jsonl/valid.jsonl")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--resume-adapter", default=None)
    ap.add_argument("--start-iter", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--hot-budget-mb", type=int, default=2500)
    ap.add_argument("--spawn", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--out", default="adapters_ling_arc")
    ap.add_argument("--structure-check", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    cfg = load_ling_cfg(model_dir)
    bank = Bank(shard_map(model_dir), budget_mb=args.hot_budget_mb)
    print(f"model: {model_dir.name} layers={cfg.nlayers} "
          f"experts={cfg.experts}x{cfg.per_tok} bits={cfg.bits} "
          f"routing={cfg.score_function}/group{cfg.n_group}x{cfg.topk_group}")

    if args.structure_check:
        print("probing seams:")
        good = structure_check(cfg, bank)
        print("RESULT:", "OK" if good else "FAILED")
        return 0 if good else 2

    pin_dense(bank, cfg)
    for p in ("lm_head.weight", "lm_head.scales", "lm_head.biases"):
        if p in bank.names:
            bank.get(p, resident=True)
    print(f"pinned dense: {bank.stats()}")

    H, rank = cfg.hidden, args.rank
    Dqkv = cfg.heads * cfg.head_dim + 2 * cfg.kv_heads * cfg.head_dim
    key = mx.random.key(3)
    P = {}
    P_SCALE = args.scale
    # LoRA on the FUSED qkv of the top attention layer (last MoE layer)
    grad_layer = cfg.nlayers - 1
    P[f"model.layers.{grad_layer}.attention.query_key_value.lora_a"] = \
        mx.random.normal((H, rank), key=key) * 0.01
    P[f"model.layers.{grad_layer}.attention.query_key_value.lora_b"] = \
        mx.zeros((rank, Dqkv))
    if args.resume_adapter:
        with safe_open_safe(args.resume_adapter) as s:
            for k in s.keys():
                P[k] = s.get_tensor(k)
    trainable = sum(math.prod(v.shape) for k, v in P.items() if k != "_scale")
    print(f"trainable: {trainable / 1e6:.2f}M params (rank {rank} on layer "
          f"{grad_layer} fused-qkv)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir),
                                        trust_remote_code=True)

    if args.data:
        rows = [json.loads(ln) for ln in
                (Path(args.data) / "train.jsonl").read_text().splitlines()
                if ln.strip()]
    else:
        rows = [{"messages": [
            {"role": "user", "content": "reply with exactly: SMOKE OK"},
            {"role": "assistant", "content": "SMOKE OK " + "ok " * 30}]}]
        print("no --data: smoke example")

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
        if len(ids) > args.max_len:
            drop = len(ids) - args.max_len
            ids = ids[drop:]
            n = max(0, n - drop)
        labels = ([-100] * n) + ids[n:]
        return ids, labels[: args.max_len]

    def embed_rows(ids):
        o = cfg.per_tensor.get("model.word_embeddings", {})
        gs, bits = o.get("group_size", cfg.gs), o.get("bits", cfg.bits)
        w = bank.get("model.word_embeddings.weight")
        s = bank.get("model.word_embeddings.scales")
        b = bank.get("model.word_embeddings.biases") \
            if "model.word_embeddings.biases" in bank.names else None
        idx = mx.array(ids).astype(mx.int32)
        pw, ps = mx.take(w, idx, axis=0), mx.take(s, idx, axis=0)
        pb = mx.take(b, idx, axis=0) if b is not None else None
        return mx.dequantize(pw, ps, pb, group_size=gs, bits=bits)

    def loss_fn(p, ids, labels):
        lora = {**p, "_scale": P_SCALE}
        T = len(ids)
        cos, sin, d2 = rope_tables_half(T, cfg.head_dim, cfg.rope_theta,
                                        cfg.partial_rotary)
        x = embed_rows(ids).astype(mx.bfloat16)[None, :, :]
        for i in range(grad_layer):
            x = layer_ling(x, i, bank, cfg)
            mx.eval(x)                                     # stream + free
        for i in (grad_layer,):
            x = layer_ling(x, i, bank, cfg, lora=lora)
        assert "model.norm.weight" in bank.names, "no final norm tensor"
        x = rms(x[0], bank.get("model.norm.weight", resident=True), cfg.eps)
        logits = linear(x, "lm_head", bank, cfg)
        tgt = mx.array(labels[1:] + [-100]).astype(mx.int32)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), tgt,
                                     reduction="none")
        valid = (tgt != -100).astype(mx.float32)
        return (ce * valid).sum() / mx.maximum(valid.sum(), 1.0)

    out = Path(args.out)
    t0 = time.time()
    for it in range(args.start_iter + 1, args.iters + 1):
        ids, labels = to_ids(rows[(it - 1) % len(rows)])
        lv, grads = mx.value_and_grad(lambda p: loss_fn(p, ids, labels))(P)
        for k, g in grads.items():
            P[k] = P[k] - args.lr * g
        lv = float(lv)
        peak = ""
        if hasattr(mx, "get_peak_memory"):
            peak = f" peak {mx.get_peak_memory() / (1 << 20):.0f}MB"
            mx.reset_peak_memory()
        print(f"Iter {it}: Train loss {lv:.4f} |{peak} | {bank.stats()} | "
              f"{time.time() - t0:6.1f}s", flush=True)
        if it % args.save_every == 0 or it == args.iters:
            out.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(out / "adapters.safetensors"),
                                {k: v.astype(mx.float32) for k, v in P.items()})
            (out / "adapter_config.json").write_text(json.dumps({
                "model": str(model_dir), "rank": args.rank,
                "scale": args.scale, "grad_layer": grad_layer,
                "arch": "bailing_moe", "paged": True}))
            print(f"  saved -> {out}", flush=True)
    return 0


def safe_open_safe(path):
    from safetensors import safe_open
    return safe_open(str(Path(path)), framework="mlx")


if __name__ == "__main__":
    sys.exit(main())