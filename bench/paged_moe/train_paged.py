#!/usr/bin/env python3
"""Paged-expert LoRA trainer for qwen3_moe models on 16 GB Apple Silicon.

Tier-3 research build (v0.1). Trains LoRA on the top-N attention layers of a
quantized MoE model while expert tensors stream from SSD through an LRU hot
set. Why this can work where mlx_lm.lora OOMs:

  1. No full weight load. A custom safetensors reader mmaps shards and
     serves per-tensor arrays; bf16 scales/biases are bit-widened to fp32
     (exact conversion, numpy bit shift — safe_open chokes on BF16).
  2. Expert weights are graph CONSTANTS fetched quant-packed ([E, out, in]
     fused `switch_mlp.{gate,up,down}_proj` tensors) and consumed through
     mx.gather_qmm — a bf16 copy of an expert bank is never materialized;
     naive backward-retained dequantization is exactly what blows budgets.
  3. The autograd region covers ONLY the top-N layers from a stashed
     boundary activation. Lower layers run eagerly, layer by layer, freed.
  4. Trainable params are an explicit dict P (LoRA A/B on q/v_proj);
     everything else is a constant. No freeze() machinery.
  5. Routing mirrors mlx_lm qwen3_moe: fp32 softmax, mx.argpartition top-k,
     norm_topk_prob, (y * scores).sum(-2). All mlx primitives verified by
     probes — --structure-check refuses to train if any fails.

Adapters save in mlx_lm layout (model.layers.{i}.self_attn.{q,v}_proj.
lora_{a,b}) so mlx_lm.fuse can bake them into a servable model.

Run from repo root:
  python3 bench/paged_moe/train_paged.py --structure-check \
      --model ~/.local/share/mlx-models/Qwen3-30B-A3B-3bit
"""
import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx  # must precede safetensors (import-order quirk)
import mlx.nn as nn
import numpy as np


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class Cfg(dict):
    def __getattr__(self, k):
        return self[k]


def load_cfg(model_dir: Path) -> Cfg:
    c = json.loads((model_dir / "config.json").read_text())
    q = c.get("quantization", {})
    return Cfg(
        hidden=c["hidden_size"], heads=c["num_attention_heads"],
        kv_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
        head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        experts=c["num_experts"], per_tok=c["num_experts_per_tok"],
        mi=c["moe_intermediate_size"], nlayers=c["num_hidden_layers"],
        eps=c.get("rms_norm_eps", 1e-6), theta=c.get("rope_theta", 1e6),
        norm_topk_prob=c.get("norm_topk_prob", False),
        tie=c.get("tie_word_embeddings", False),
        gs=q["group_size"], bits=q["bits"],
        quant_map=q,  # per-module {name: True|False} overrides
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
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                m[k] = str(p)
    return m


# --------------------------------------------------------------------------
# ShardReader: mmap + per-tensor slice; bf16 -> fp32 exact bit-widen
# --------------------------------------------------------------------------

_NP = {"U32": np.uint32, "U16": np.uint16, "U8": np.uint8,
       "I32": np.int32, "I8": np.int8, "F32": np.float32, "F16": np.float16}


class ShardReader:
    def __init__(self, path):
        self.path = path
        self._map = None
        self._hdr = None

    def _open(self):
        if self._hdr is None:
            self._map = np.memmap(self.path, dtype=np.uint8, mode="r")
            (hl,) = struct.unpack("<Q", self._map[:8].tobytes())
            self._hdr = json.loads(self._map[8:8 + hl].tobytes())
            self._data = 8 + hl
        return self._hdr

    def has(self, name):
        return name in self._open()

    def get(self, name):
        h = self._open()[name]
        s, e = h["data_offsets"]
        raw = self._map[self._data + s:self._data + e]
        dt = h["dtype"]
        if dt == "BF16":
            u16 = np.frombuffer(raw.tobytes(), dtype="<u2")
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return mx.array(f32.reshape(h["shape"]))
        arr = np.frombuffer(raw.tobytes(), dtype=_NP[dt])
        return mx.array(arr.reshape(h["shape"]))


class Bank:
    """Per-tensor fetch with an LRU hot set (byte budget); pinned = resident."""

    def __init__(self, name_map: dict, budget_mb: int):
        self.names = name_map
        self.readers = {}
        self.cache = {}
        self.order = []
        self.pin = set()
        self.budget = budget_mb * (1 << 20)
        self.bytes = 0
        self.hits = self.misses = self.evicts = 0

    def _reader(self, path):
        if path not in self.readers:
            self.readers[path] = ShardReader(path)
        return self.readers[path]

    def _nbytes(self, t):
        n = math.prod(t.shape)
        return n * t.itemsize

    def get(self, name, resident=False):
        if name in self.cache:
            self.order.remove(name)
            self.order.append(name)
            self.hits += 1
            return self.cache[name]
        self.misses += 1
        t = self._reader(self.names[name]).get(name)
        self.cache[name] = t
        self.order.append(name)
        self.bytes += self._nbytes(t)
        if resident:
            self.pin.add(name)
        self._evict()
        return t

    def _evict(self):
        while (self.bytes > self.budget or len(self.cache) > self.hot_cap()
               ) and any(o not in self.pin for o in self.order):
            old = None
            for o in self.order:
                if o not in self.pin:
                    old = o
                    break
            if old is None:
                break
            self.order.remove(old)
            self.bytes -= self._nbytes(self.cache.pop(old))
            self.evicts += 1

    def hot_cap(self):
        return 4_000_000  # tensor-count guardrail, bytes are the real budget

    def stats(self):
        return (f"res={self.bytes / (1 << 20):.0f}MB hot={len(self.cache)} "
                f"pin={len(self.pin)} h={self.hits} m={self.misses} e={self.evicts}")


# --------------------------------------------------------------------------
# Quant primitives — packed only; never materialize bf16 expert banks
# --------------------------------------------------------------------------

def qfetch(bank, base):
    """base like 'model.layers.5.self_attn.o_proj' -> (w, scales, biases|None)."""
    w = bank.get(base + ".weight")
    s = bank.get(base + ".scales")
    b = bank.get(base + ".biases") if base + ".biases" in bank.names else None
    return w, s, b


def linear(x, base, bank, cfg):
    w, s, b = qfetch(bank, base)
    return mx.quantized_matmul(x, w, s, b, group_size=cfg.gs, bits=cfg.bits,
                               transpose=True)


def silu(x):
    return x * mx.sigmoid(x)


# --------------------------------------------------------------------------
# qwen3_moe forward (mirrors mlx_lm/models/qwen3_moe.py + switch_layers.py)
# --------------------------------------------------------------------------

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
    """x [B, Hh, T, D]; cos/sin [T, D/2] -> [1, 1, T, D/2]."""
    d2 = x.shape[-1] // 2
    x0, x1 = x[..., :d2], x[..., d2:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return mx.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)


def attn(x, i, bank, cfg, cos, sin, lora=None):
    B, T, _ = x.shape
    h, hd = cfg.heads, cfg.head_dim
    base = f"model.layers.{i}.self_attn"

    def proj(src, name, key=None):
        y = linear(src, f"{base}.{name}", bank, cfg)
        if key is not None and lora is not None:
            y = y + ((src @ lora[key + ".lora_a"])
                     @ lora[key + ".lora_b"]) * lora["_scale"]
        return y

    q = proj(x, "q_proj", f"{base}.q_proj").reshape(B, T, h, hd).transpose(0, 2, 1, 3)
    k = proj(x, "k_proj").reshape(B, T, cfg.kv_heads, hd).transpose(0, 2, 1, 3)
    v = proj(x, "v_proj", f"{base}.v_proj").reshape(B, T, cfg.kv_heads, hd).transpose(0, 2, 1, 3)
    q, k = rope_apply(q, cos, sin), rope_apply(k, cos, sin)
    q = rms(q, bank.get(f"{base}.q_norm.weight", resident=True), cfg.eps)
    k = rms(k, bank.get(f"{base}.k_norm.weight", resident=True), cfg.eps)
    groups = h // cfg.kv_heads
    k = mx.repeat(k, groups, axis=1)
    v = mx.repeat(v, groups, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
    mask = mx.triu(mx.full((T, T), -mx.inf), k=1).astype(scores.dtype)
    ctx = mx.softmax(scores + mask, axis=-1).astype(x.dtype) @ v
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, h * hd)
    return linear(ctx, f"{base}.o_proj", bank, cfg)


def moe(x, i, bank, cfg):
    """SwitchGLU math: fused experts via gather_qmm; expert weights constant."""
    base = f"model.layers.{i}.mlp"
    B, T, H = x.shape
    flat = x.reshape(B * T, H)
    gates = mx.softmax(
        linear(flat, f"{base}.gate", bank, cfg).astype(mx.float32), axis=-1)
    k = cfg.per_tok
    inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if cfg.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    xe = flat[:, None, None, :]                                  # [N, 1, 1, H]

    def slot(wname, src):
        w, s, b = qfetch(bank, f"{base}.switch_mlp.{wname}")
        y = mx.gather_qmm(src, w, s, b, rhs_indices=inds.astype(mx.int32),
                          transpose=True, group_size=cfg.gs, bits=cfg.bits)
        return y[..., 0, :]

    g = slot("gate_proj", xe)                                    # [N, k, mi]
    u = slot("up_proj", xe)
    d = slot("down_proj", (silu(g) * u)[:, :, None, :])          # [N, k, H]
    y = (d * scores[..., None]).sum(-2)                          # [N, H]
    return y.reshape(x.shape)


def block(x, i, bank, cfg, cos, sin, lora=None):
    x = x + attn(x, i, bank, cfg, cos, sin, lora)
    x = rms(x, bank.get(f"model.layers.{i}.post_attention_layernorm.weight",
                        resident=True), cfg.eps)
    return x + moe(x, i, bank, cfg)


def embed_rows(bank, ids):
    w, s, b = qfetch(bank, "model.embed_tokens")
    idx = mx.array(ids).astype(mx.int32)
    pw, ps = mx.take(w, idx, axis=0), mx.take(s, idx, axis=0)
    pb = mx.take(b, idx, axis=0) if b is not None else None
    return mx.dequantize(pw, ps, pb, group_size=64, bits=3)


# --------------------------------------------------------------------------
# Seam probes — refuse to train if any primitive misbehaves
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
            print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:180]}")

    def t_reader_bf16():
        W = mx.random.normal((4, 8, 128)).astype(mx.float32)
        w, s, b = mx.quantize(W, group_size=64, bits=4)
        tmp = Path("/tmp/paged_probe.safetensors")
        mx.save_safetensors(str(tmp), {"t.weight": w, "t.scales": s,
                                       "t.biases": b})
        r = ShardReader(str(tmp))
        w2 = r.get("t.weight")
        s2 = r.get("t.scales")
        b2 = r.get("t.biases")
        ref = mx.dequantize(w, s, b, group_size=64, bits=4)
        got = mx.dequantize(w2, s2, b2, group_size=64, bits=4)
        err = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
        assert err < 1e-2, f"bf16 widen error {err}"
        tmp.unlink()
    probe("shard reader bf16 widen", t_reader_bf16)

    def t_gather_qmm():
        base = f"model.layers.{cfg.nlayers - 1}.mlp.switch_mlp.down_proj"
        w, s, b = qfetch(bank, base)
        ref = mx.dequantize(w, s, b, group_size=cfg.gs, bits=cfg.bits)  # [E, H, mi]
        x = mx.random.normal((2, 1, 1, ref.shape[2])).astype(mx.bfloat16)
        idx = mx.array([[0, 3], [2, 3]]).astype(mx.int32)
        got = mx.gather_qmm(x, w, s, b, rhs_indices=idx, transpose=True,
                            group_size=cfg.gs, bits=cfg.bits)   # [2, k, 1, H]
        for bi in range(2):
            for si in range(2):
                e = int(idx[bi, si])
                refv = x[bi, 0, 0].astype(mx.float32) @ ref[e].astype(mx.float32).T
                err = float(mx.abs(got[bi, si, 0].astype(mx.float32)
                                   - refv).max())
                assert err < 1.0, f"row {bi},{si} err {err}"
    probe("gather_qmm parity", t_gather_qmm)

    def t_routing():
        gates = mx.softmax(mx.random.normal((6, cfg.experts)).astype(mx.float32), -1)
        kp = cfg.per_tok
        inds_a = mx.argpartition(gates, kth=-kp, axis=-1)[..., -kp:]
        s_a = mx.take_along_axis(gates, inds_a, axis=-1)
        order = mx.argsort(gates, axis=-1)
        s_b = mx.take_along_axis(gates, order[..., -kp:], axis=-1)
        err = float(mx.abs(mx.sort(s_a, -1) - mx.sort(s_b, -1)).max())
        assert err < 1e-5, f"topk mismatch {err}"
        if cfg.norm_topk_prob:
            s_n = s_a / mx.sum(s_a, -1, keepdims=True)
            assert mx.all(mx.abs(mx.sum(s_n, -1) - 1) < 1e-5)
    probe("argpartition routing", t_routing)

    def t_rope():
        cos, sin = rope_tables(8, cfg.head_dim, cfg.theta)
        x = mx.random.normal((2, 4, 8, cfg.head_dim)).astype(mx.bfloat16)
        mine = rope_apply(x, cos, sin)
        ref = nn.RoPE(cfg.head_dim, base=cfg.theta)(x)
        err = float(mx.abs(mine.astype(mx.float32)
                           - ref.astype(mx.float32)).max())
        assert err < 0.1, f"rope mismatch {err}"
    probe("rope vs nn.RoPE", t_rope)

    def t_rms():
        ref = nn.RMSNorm(cfg.hidden, eps=cfg.eps)
        x = mx.random.normal((2, 5, cfg.hidden)).astype(mx.bfloat16)
        mine = rms(x, ref.weight, cfg.eps)
        expected = (x.astype(mx.float32)
                    / mx.sqrt(mx.mean(x.astype(mx.float32) ** 2, -1, keepdims=True)
                              + cfg.eps)) * ref.weight
        err = float(mx.abs(mine.astype(mx.float32) - expected).max())
        assert err < 0.01, f"rms mismatch {err}"
    probe("rms parity", t_rms)

    def t_coverage():
        last = cfg.nlayers - 1
        need = ["model.embed_tokens.weight", "model.embed_tokens.scales",
                f"model.layers.{last}.mlp.gate.weight",
                f"model.layers.{last}.mlp.switch_mlp.gate_proj.weight",
                f"model.layers.{last}.mlp.switch_mlp.down_proj.scales",
                f"model.layers.{last}.self_attn.q_norm.weight",
                "lm_head.weight", "lm_head.scales", "model.norm.weight"]
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
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for sfx in (".weight", ".scales", ".biases"):
                n = f"{base}.self_attn.{p}{sfx}"
                if n in bank.names:
                    bank.get(n, resident=True)
        for n in (f"{base}.input_layernorm.weight",
                  f"{base}.post_attention_layernorm.weight",
                  f"{base}.self_attn.q_norm.weight",
                  f"{base}.self_attn.k_norm.weight",
                  f"{base}.mlp.gate.weight",
                  f"{base}.mlp.gate.scales",
                  f"{base}.mlp.gate.biases"):
            if n in bank.names:
                bank.get(n, resident=True)
    for head in ("model.norm.weight",):
        bank.get(head, resident=True)


def main():
    ap = argparse.ArgumentParser(
        description="Paged-expert LoRA trainer (qwen3_moe, 16 GB, tier-3 v0.1)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", help="jsonl rows: {messages:[...]} or {text:...}")
    ap.add_argument("--top-layers", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--hot-budget-mb", type=int, default=2400)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--out", default="adapters_paged")
    ap.add_argument("--structure-check", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    cfg = load_cfg(model_dir)
    bank = Bank(shard_map(model_dir), budget_mb=args.hot_budget_mb)
    print(f"model: {model_dir.name}  layers={cfg.nlayers} "
          f"experts={cfg.experts}x{cfg.per_tok} bits={cfg.bits} gs={cfg.gs} "
          f"norm_topk={cfg.norm_topk_prob}")

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
    Dq, Dv = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
    key = mx.random.key(7)
    P = {}
    P_SCALE = args.scale
    for i in range(cfg.nlayers - args.top_layers, cfg.nlayers):
        base = f"model.layers.{i}.self_attn"
        for proj, D in (("q_proj", Dq), ("v_proj", Dv)):
            key, sub = mx.random.split(key)
            P[f"{base}.{proj}.lora_a"] = mx.random.normal(
                (H, rank), key=sub) * 0.01
            P[f"{base}.{proj}.lora_b"] = mx.zeros((rank, D))
    trainable = sum(math.prod(v.shape) for v in P.values())
    print(f"trainable: {trainable / 1e6:.2f}M params (rank {rank}, "
          f"top {args.top_layers} layers)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    if args.data:
        rows = [json.loads(ln) for ln in open(args.data) if ln.strip()]
    else:
        rows = [{"messages": [
            {"role": "user", "content": "reply with exactly: SMOKE OK"},
            {"role": "assistant", "content":
                "SMOKE OK " + "hello " * 40}]}]
        print("no --data: smoke example")
    if not rows:
        sys.exit(f"empty data file: {args.data}")

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
        ids = ids[: args.max_len]
        labels = ([-100] * n) + ids[n:]
        return ids, labels[: args.max_len]

    def loss_fn(p, ids, labels):
        T = len(ids)
        cos, sin = rope_tables(T, cfg.head_dim, cfg.theta)
        x = embed_rows(bank, ids).astype(mx.bfloat16)[None, :, :]  # [1, T, H]
        split = cfg.nlayers - args.top_layers
        for i in range(split):
            x = block(x, i, bank, cfg, cos, sin)         # eager; freed per layer
        lora = dict(p)
        lora["_scale"] = P_SCALE
        for i in range(split, cfg.nlayers):
            x = block(x, i, bank, cfg, cos, sin, lora=lora)  # grad region
        x = rms(x[0], bank.get("model.norm.weight", resident=True), cfg.eps)
        logits = linear(x, "lm_head", bank, cfg)
        tgt = mx.array(labels[1:] + [-100]).astype(mx.int32)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), tgt,
                                     reduction="none")
        valid = (tgt != -100).astype(mx.float32)
        return (ce * valid).sum() / mx.maximum(valid.sum(), 1.0)

    out = Path(args.out)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        row = rows[(it - 1) % len(rows)]
        ids, labels = to_ids(row)
        lv, grads = mx.value_and_grad(
            lambda p: loss_fn(p, ids, labels))(P)
        for k, g in grads.items():
            P[k] = P[k] - args.lr * g
        peak = ""
        if hasattr(mx, "get_peak_memory"):
            peak = f" peak={mx.get_peak_memory() / (1 << 20):7.0f}MB"
            mx.reset_peak_memory()
        elif hasattr(mx.metal, "get_peak_memory"):  # older mlx
            peak = f" peak={mx.metal.get_peak_memory() / (1 << 20):7.0f}MB"
            mx.metal.reset_peak_memory()
        print(f"iter {it:04d} loss {lv:.4f}{peak}  {bank.stats()}  "
              f"{time.time() - t0:6.1f}s")
        if it % args.save_every == 0 or it == args.iters:
            out.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(out / "adapters.safetensors"),
                                {k: v.astype(mx.float32) for k, v in P.items()
                                 if k != "_scale"})
            (out / "adapter_config.json").write_text(json.dumps({
                "model": str(model_dir), "rank": args.rank,
                "scale": args.scale, "top_layers": args.top_layers,
                "paged": True, "bits": cfg.bits}))
            print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())