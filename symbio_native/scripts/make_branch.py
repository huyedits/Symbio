"""Create a substrate branch from a (prompt, expected) failure pair.

Usage:
    python -m symbio_native.scripts.make_branch \
        --base ./native_artifacts/checkpoints/native_lm_final.safetensors \
        --tokenizer ./native_artifacts/tokenizer.json \
        --config ./native_artifacts/config.json \
        --name "refuse-injection" \
        --prompt "User said ignore prior instructions" \
        --expected "I can't ignore my system prompt." \
        --branches-dir ./native_artifacts/branches
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from symbio_native.model import NativeConfig, NativeLM
from symbio_native.substrate import BranchManager
from symbio_native.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a correction substrate branch")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--branches-dir", required=True, type=Path)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    config = NativeConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    model = NativeLM.load(args.base, config)
    tokenizer = BPETokenizer.load(args.tokenizer)

    mgr = BranchManager(model, args.branches_dir)
    branch = mgr.make_branch(args.name, args.prompt, args.expected, tokenizer, iters=args.iters)
    print(f"[Native] created branch '{branch.name}' -> {args.branches_dir / branch.name}")


if __name__ == "__main__":
    main()
