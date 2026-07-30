"""End-to-end script: train tokenizer, train base model, save artifacts.

Usage:
    python -m symbio_native.scripts.train_native \
        --data-dir ./training_data \
        --out-dir ./native_artifacts \
        --vocab-size 1024 \
        --iters 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx.utils import tree_flatten

from symbio_native.model import NativeConfig, NativeLM
from symbio_native.tokenizer import BPETrainer
from symbio_native.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Symbio Native model from scratch")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Native] training tokenizer...")
    texts = []
    for path in sorted(args.data_dir.rglob("*")):
        if path.is_file() and path.suffix in {".txt", ".md", ".py", ".jsonl"}:
            try:
                texts.append(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    tokenizer = BPETrainer(vocab_size=args.vocab_size).train(texts)
    tokenizer.save(out_dir / "tokenizer.json")
    print(f"[Native] tokenizer saved ({len(tokenizer.vocab)} tokens)")

    config = NativeConfig(
        vocab_size=len(tokenizer.vocab),
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
    )
    (out_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    model = NativeLM(config)
    param_count = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"[Native] model params ~ {param_count / 1e6:.2f}M")

    train(
        model,
        tokenizer,
        args.data_dir,
        out_dir / "checkpoints",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        iters=args.iters,
    )
    print(f"[Native] done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
