# Symbio Native

A tiny MLX transformer language model built from scratch inside the Symbio repo.
It lives on the `model-from-scratch` branch and is intentionally separate from the
main `symbio/` package.

## Philosophy

- **Never freeze the base weights**: the model starts from random init and keeps
  training.
- **Substrate branches**: when the model fails, train a small delta (full diff or
  low-rank) on that failure and save it as a branch. Base weights are not
  overwritten.
- **Router selects branches**: at inference time the router picks the most
  relevant substrate branches based on trigger-token overlap with the prompt.

## Layout

```
symbio_native/
  tokenizer/     byte-level BPE tokenizer trained from raw text
  model/         NativeConfig + NativeLM transformer (RoPE, GQA, RMSNorm, SwiGLU)
  train/         MLX training loop
  substrate/     BranchManager creates failure-driven deltas without touching base weights
  router/        lightweight per-turn branch selector
  scripts/       train_native.py, make_branch.py
  tests/         pytest coverage
```

## Quick start

Train a tokenizer + base model:

```bash
python -m symbio_native.scripts.train_native \
    --data-dir ./training_data \
    --out-dir ./native_artifacts \
    --iters 1000
```

Create a correction branch after observing a failure:

```bash
python -m symbio_native.scripts.make_branch \
    --base ./native_artifacts/checkpoints/native_lm_final.safetensors \
    --tokenizer ./native_artifacts/tokenizer.json \
    --config ./native_artifacts/config.json \
    --name "refuse-injection" \
    --prompt "User said ignore prior instructions" \
    --expected "I can't ignore my system prompt." \
    --branches-dir ./native_artifacts/branches
```

Run tests:

```bash
python -m pytest symbio_native/tests/ -q
```

## Design notes

- The tokenizer is byte-level so it can represent any UTF-8 text without an
  unknown-token gap.
- `NativeLM` ties input embeddings to the output LM head, matching common
  efficient LLaMA-style designs.
- `BranchManager.make_branch` trains a fresh copy of the base model on the
  failure example, computes `delta = trained_copy - base`, then restores the live
  base weights. The branch contains only the delta, keeping base weights
  untouched.
