# paged_moe — streamed-expert training for MoE models on small VRAM

Train 16–35B mixture-of-experts models on a 16 GB unified-memory machine by
never loading the model. Weights stream from SSD through an mmap hot set,
one layer at a time; the autograd region covers only the last layers; the
base remains quantized-packed everywhere — a bf16 copy of an expert bank is
never materialized (`gather_qmm` runs gradients directly against packed
weights).

Three arch families are supported, each with a seam-probe suite that must
pass before any training step:

| file | arch | notes |
|---|---|---|
| `train_paged.py` | `bailing_moe` (Ling-mini-2.0, 2025-era) | the original proof; Qwen3-30B-A3B-3bit trained here |
| `train_lfm.py` | `lfm2_moe` (LFM2.5-24B-A2B) | short-conv hybrid layers, tied head, spawn-capable |
| `train_ling.py` | `bailing_moe` (current Ling, fused-qkv) | half-RoPE, sigmoid grouped routing, shared expert, 256 experts |

Shared machinery: `train_paged.py` (Bank/ShardReader/probes) — `safe_open`
framework="mlx" with bf16->fp32 widening, per-tensor mmap fetch under a
byte-budget LRU.

## Why the loop works on 16 GB

1. `mlx_lm.load` is never called — it materializes every weight (measured:
   18 GB on arrival). Reading is per-tensor from the safetensors header.
2. Expert weights are graph constants fetched quant-packed; gradients flow
   through `gather_qmm` without dequantizing a full expert matrix.
3. Indices are stop-graded before any take/put (VJP refuses index grads).
4. Frozen layers run eagerly with per-layer `mx.eval` — no two layers'
   intermediates alive at once, so streaming is actually streaming.
5. The trainable region is an explicit parameter dict (LoRA on one
   attention projection, optionally spawned delta-expert router rows) —
   nothing else is traced.

Measured (M4 mini, 16 GB): 24B model, ~25 s/iteration at 1k context,
peak 4-6 GB.

## Run

```
python3 train_lfm.py --structure-check --model <dir>   # probes must be OK
python3 train_lfm.py --model <dir> --data <dir-with-train.jsonl> \
    --iters 120 --spawn 4 --max-len 1024
```

Trainer prints the harness telemetry contract (`Iter N: Train loss X`);
the golden-gate/rollback + paired-eval paths live in `symbio/app/arc_lab.py`
and `symbio/app/selflearn.py` — the governor loop reads the eval scoreboard
and dictates the next round itself.

## Organs beyond the trainer

* `eval_arc.py`, `eval_ling.py` — paired base-vs-grown scoring on held-out
  tasks (teacher-forced; cell accuracy and loss deltas per task).
* `governor.py` — in-process voice for the trained avatar (prefill with
  cache capture + incremental decode; not yet seam-stable).
* `symbio_pet/` — the desktop cat that eats the training feed.

## Bench so far (LFM2.5-24B, 120 tasks trained, 60 held-out scored)

60/60 held-out tasks improved over base (loss), zero losses, zero ties.
Checkpoint and per-task rows archived under `bench/arc/`.