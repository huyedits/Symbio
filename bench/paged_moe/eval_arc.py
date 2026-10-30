#!/usr/bin/env python3
"""Paired base-vs-grown evaluation on held-out ARC-AGI-2 evaluation tasks.

Teacher-forced scoring inside the streaming trainer's own machinery — no
serving, no Docker, no wired-limit bump. For each task the frozen stack
(layers 0..37) is stashed once, then the tail region scores twice:
  base    — p=None (spawn machinery off, no LoRA)
  grown   — p=adapters (layer-38 LoRA + spawned delta-expert slots)
Metrics per task: CE nats/target-token, argmax cell accuracy, full-grid exact
(top-1 at every masked position). Paired deltas + summary written to JSON.

IMPORTANT HONESTY NOTE: argmax-under-teacher-forcing is a cell-level proxy,
not free-running generation; full-run exact match is reported as such.

Run from bench/paged_moe/ with PYTHONPATH=. (arc_lab sets both).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from train_paged import Bank
from train_lfm import (Cfg, layer_fwd_exact, load_cfg, pin_dense,
                       rope_tables, shard_map, embed_rows, rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="eval.jsonl path")
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--spawn", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    cfg = load_cfg(model_dir)
    cfg.spawn = args.spawn
    bank = Bank(shard_map(model_dir), budget_mb=3500)
    attn_idx = [i for i, t in enumerate(cfg.layer_types)
                if t == "full_attention"]
    grad_layers = [attn_idx[-1]]
    conv_tail = list(range(grad_layers[0] + 1, cfg.nlayers))
    pin_dense(bank, cfg)
    for p in ("model.embed_tokens.weight", "model.embed_tokens.scales",
              "model.embed_tokens.biases"):
        if p in bank.names:
            bank.get(p, resident=True)
    bank.get("model.embedding_norm.weight", resident=True)

    # ---- P from adapters (spawn layout, same keys as training saved) -------
    P = {}
    with safe_open(str(Path(args.adapters)), framework="mlx") as s:
        for k in s.keys():
            P[k] = s.get_tensor(k)
    # The saved A/B were trained under lora scale S — replay the same scale,
    # otherwise the grown brain is silently 1/S of its trained strength.
    cfg_path = Path(args.adapters).parent / "adapter_config.json"
    P_SCALE = 8.0
    if cfg_path.exists():
        try:
            P_SCALE = float(json.loads(cfg_path.read_text()).get("scale", 8.0))
        except Exception:
            pass

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    rows = [json.loads(ln) for ln in
            Path(args.data).read_text().splitlines() if ln.strip()][: args.count]

    left_trunc = getattr(sys.modules[__name__], "left_truncate", None)

    def truncate(msgs):
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        ids = [int(t) for t in tok(text)["input_ids"]]
        # mask: everything before the final assistant turn
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

    def score(p, stash, ids, labels):
        T = len(ids)
        cos, sin = rope_tables(T, cfg.head_dim, cfg.rope_theta)
        lora = None
        if p is not None:
            lora = dict(p)
            lora["_scale"] = P_SCALE
        x = stash
        for i in grad_layers:
            x = layer_fwd_exact(x, i, bank, cfg, cos, sin, lora=lora)
        for i in conv_tail:
            x = layer_fwd_exact(x, i, bank, cfg, cos, sin)
        x = rms(x[0], bank.get("model.embedding_norm.weight",
                               resident=True), cfg.eps)
        logits = qlin_head(x)                    # [T, V] tied head
        tgt = mx.array(labels[1:] + [-100]).astype(mx.int32)
        valid = (tgt != -100).astype(mx.float32)
        ce_all = nn.losses.cross_entropy(logits.astype(mx.float32), tgt,
                                         reduction="none")
        n_tok = float(mx.maximum(valid.sum(), 1.0))
        loss = float((ce_all * valid).sum()) / n_tok
        amax = mx.argmax(logits, axis=-1).astype(mx.int32)
        hit = ((amax == tgt).astype(mx.float32) * valid).sum() / n_tok
        exact = bool(float(((amax == tgt).astype(mx.float32) * valid).sum())
                     >= n_tok - 0.5)
        return loss, float(hit), exact, int(float(valid.sum()))

    def qlin_head(x):
        o = cfg.per_tensor.get("model.embed_tokens", {})
        gs, bits = o.get("group_size", cfg.gs), o.get("bits", cfg.bits)
        w = bank.get("model.embed_tokens.weight")
        s = bank.get("model.embed_tokens.scales")
        b = bank.get("model.embed_tokens.biases") \
            if "model.embed_tokens.biases" in bank.names else None
        return mx.quantized_matmul(x, w, s, b, group_size=gs, bits=bits,
                                   transpose=True)

    def stash_of(ids):
        T = len(ids)
        cos, sin = rope_tables(T, cfg.head_dim, cfg.rope_theta)
        x = embed_rows(bank, ids, cfg).astype(mx.bfloat16)[None, :, :]
        for i in range(grad_layers[0]):
            x = layer_fwd_exact(x, i, bank, cfg, cos, sin)
            mx.eval(x)          # free this layer before the next streams
        return x

    results = []
    t0 = time.time()
    for n_done, row in enumerate(rows, 1):
        try:
            ids, labels = truncate(row["messages"])
        except Exception as e:
            continue
        if sum(1 for t in labels if t != -100) < 8:
            continue
        st = stash_of(ids)
        bl, bcell, bex, ntok = score(None, st, ids, labels)
        al, acell, aex, _ = score(P, st, ids, labels)
        results.append({"task_id": row.get("task_id", ""),
                        "n_target_tokens": ntok,
                        "base_loss": bl, "adapted_loss": al,
                        "loss_delta": al - bl,
                        "base_cell": bcell, "adapted_cell": acell,
                        "cell_delta": acell - bcell,
                        "base_exact": bex, "adapted_exact": aex})
        mx.eval(list(P.values()))
        if n_done % 10 == 0:
            print(f"  scored {n_done}/{len(rows)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    n = len(results)
    summary = {
        "n": n,
        "base_loss": sum(r["base_loss"] for r in results) / max(n, 1),
        "adapted_loss": sum(r["adapted_loss"] for r in results) / max(n, 1),
        "base_cell_acc": sum(r["base_cell"] for r in results) / max(n, 1),
        "adapted_cell_acc": sum(r["adapted_cell"] for r in results) / max(n, 1),
        "wins": sum(1 for r in results if r["loss_delta"] < 0),
        "losses": sum(1 for r in results if r["loss_delta"] > 0),
        "ties": sum(1 for r in results if r["loss_delta"] == 0),
        "base_exact_count": sum(r["base_exact"] for r in results),
        "adapted_exact_count": sum(r["adapted_exact"] for r in results),
    }
    summary["cell_delta"] = summary["adapted_cell_acc"] - summary["base_cell_acc"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "per_task": results}, indent=1))
    print(json.dumps({"summary": summary}, indent=1))
    print(f"eval done: {n} tasks in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())