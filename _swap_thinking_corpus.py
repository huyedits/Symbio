"""Swap the thinking corpus into place for training, with backups.

Backs up the current train.jsonl/valid.jsonl and the current adapter, then
copies train.thinking.jsonl over train.jsonl. The validation split is rebuilt
by run_training (ensure_validation_split), so valid.jsonl is just removed.

Usage: venv/bin/python _swap_thinking_corpus.py [--source training_data/train.thinking.jsonl]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from symbio import constants

BACKUP_SUFFIX = ".thinking-pretrain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_data/train.thinking.jsonl")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: {src} does not exist")
        return 1

    # 1. Backup the current adapter.
    if constants.ADAPTER_DIR.exists():
        bak = Path(str(constants.ADAPTER_DIR) + BACKUP_SUFFIX)
        if bak.exists():
            shutil.rmtree(bak)
        shutil.copytree(constants.ADAPTER_DIR, bak)
        print(f"backed up adapter -> {bak}")

    # 2. Backup current train/valid.
    for f in (constants.TRAIN_FILE, constants.VALID_FILE):
        if f.exists():
            bak = Path(str(f) + BACKUP_SUFFIX)
            shutil.copy2(f, bak)
            print(f"backed up {f.name} -> {bak.name}")

    # 3. Swap in the thinking corpus.
    shutil.copy2(src, constants.TRAIN_FILE)
    print(f"copied {src.name} -> {constants.TRAIN_FILE.name}")

    # 4. Remove the stale validation split; run_training rebuilds it.
    if constants.VALID_FILE.exists():
        constants.VALID_FILE.unlink()
        print("removed stale valid.jsonl (will be rebuilt)")

    n = sum(1 for _ in constants.TRAIN_FILE.open(encoding="utf-8") if _.strip())
    print(f"train.jsonl now has {n} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
