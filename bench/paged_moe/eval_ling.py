#!/usr/bin/env python3
"""Paired base-vs-grown eval for Ling-mini-2.0 (bailing_moe) on held-out
ARC-AGI-2 evaluation tasks — streamed, teacher-forced, same protocol as
eval_arc but for the fused-qkv/ling arch. Writes eval_results_ling.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from train_paged import Bank
from train_ling import layer_ling, load_ling_cfg, shard_map, rms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", default="eval_results_ling.json")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    cfg = load_ling_cfg(model_dir)
    bank = Bank(shard_map(model_dir), budget_mb=2200)
    pin_dense_ling(bank, cfg)
    for p in ("lm_head.weight", "lm_head.scales", "lm_head.biases"):
        if p in bank.names:
            bank.get(p, resident=True)

    P = {}
    with safe_open(str(Path(args.adapters)), framework="mlx") as s:
        for k in s.keys():
            P[k] = s.get_tensor(k)
    P_SCALE = 16.0
    try:
        cpath = Path(args.adapters).parent / "adapter_config.json"
        if cpath.exists():
            P_SCALE = float(json.loads(cpath.read_text()).get("scale", 16.0))
    except Exception:
        pass

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    rows = [json.loads(ln) for ln in
            Path(args.data).read_text().splitlines() if ln.strip()][: args.count]

    def to_ids(row):
        msgs = row["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        ids = [int(t) for t in tok(text)["input_ids"]]
        last_a = next((i for i, m in enumerate(msgs)
                       if m.get("role") == "assistant"), len(msgs))
        prefix = [int(t) for t in tok(tok.apply_chat_template(
            msgs[:last_a], tokenize=False, add_generation_prompt=True))["input_ids"]]
        n = min(len(prefix), len(ids))
        if len(ids) > args.max_len:
            drop = len(ids) - args.max_len
            ids = ids[drop:]
            n = max(0, n - drop)
        labels = ([-100] * n) + ids[n:]
        return ids, labels[: args.max_len]

    results = []
    t0 = time.time()
    for n_done, row in enumerate(rows, 1):
        try:
            ids, labels = to_ids(row)
        except Exception:
            continue
        if sum(1 for t in labels if t != -100) < 8:
            continue
        out = {"task_id": row.get("task_id", "")}

        # ── streamed stash: frozen layers 0..18 (evaluated per layer)
        x = _embed(bank, cfg, ids).astype(mx.bfloat16)[None, :, :]
        for i in range(19):
            x = layer_ling(x, i, bank, cfg)
            mx.eval(x)
        stash = x

        for tag, lora in (("base", None), ("adapted", dict(P, _scale=P_SCALE))):
            h = layer_ling(stash, 19, bank, cfg, lora=lora)
            logits = _embed_head(bank, cfg, rms(h[0], bank.get("model.norm.weight", resident=True), cfg.eps))
            tgt = mx.array(labels[1:] + [-100]).astype(mx.int32)
            valid = (tgt != -100).astype(mx.float32)
            ce = nn.losses.cross_entropy(logits.astype(mx.float32), tgt,
                                         reduction="none")
            nt = float(mx.maximum(valid.sum(), 1.0))
            out[f"{tag}_loss"] = float((ce * valid).sum()) / nt
            amax = mx.argmax(logits, axis=-1).astype(mx.int32)
            out[f"{tag}_cell"] = float(
                ((amax == tgt).astype(mx.float32) * valid).sum()) / nt
            out[f"{tag}_exact"] = bool(float(
                (amax == tgt).astype(mx.float32) * valid.sum()) >= nt - 0.5)
        out["loss_delta"] = out["adapted_loss"] - out["base_loss"]
        out["cell_delta"] = out["adapted_cell"] - out["base_cell"]
        results.append(out)
        mx.eval(list(P.values()))
        if n_done % 10 == 0:
            print(f"  scored {n_done}/{len(rows)} ({time.time()-t0:.0f}s)",
                  flush=True)

    n = len(results)
    summary = {
        "n": n,
        "base_loss": sum(r["base_loss"] for r in results) / max(n, 1),
        "adapted_loss": sum(r["adapted_loss"] for r in results) / max(n, 1),
        "wins": sum(1 for r in results if r["loss_delta"] < 0),
        "losses": sum(1 for r in results if r["loss_delta"] > 0),
        "ties": sum(1 for r in results if r["loss_delta"] == 0),
        "base_cell_acc": sum(r["base_cell"] for r in results) / max(n, 1),
        "adapted_cell_acc": sum(r["adapted_cell"] for r in results) / max(n, 1),
    }
    summary["cell_delta"] = summary["adapted_cell_acc"] - summary["base_cell_acc"]
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "per_task": results}, indent=1))
    print(json.dumps({"summary": summary}, indent=1))
    return 0


def _embed_head(bank, cfg, x):
    w = bank.get("lm_head.weight")
    s = bank.get("lm_head.scales")
    b = bank.get("lm_head.biases") \
        if "lm_head.biases" in bank.names else None
    return mx.quantized_matmul(x, w, s, b, group_size=cfg.gs, bits=cfg.bits,
                               transpose=True)


if __name__ == "__main__":
    sys.exit(main())

def _embed(bank, cfg, ids):
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


def pin_dense_ling(bank, cfg):
    from train_ling import pin_dense_ling as _p
    return _p(bank, cfg)
