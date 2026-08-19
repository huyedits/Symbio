#!/usr/bin/env python3
"""Collect every adapter into one readable folder, with its data and metadata.

    python3 archive_adapters.py                 # build Adapter_skills/
    python3 archive_adapters.py --dry-run       # show what it would write
    python3 archive_adapters.py --list          # what is in there now
    python3 archive_adapters.py --restore NAME  # rebuild a loadable adapter dir

Layout produced:

    Adapter_skills/
      HEADMASTER_600_HEADMASTER/
        HEADMASTER_600.safetensors      the weights
        adapter_config.json             metadata, as trained
        training_progress.json
        last_used.json
        manifest.json                   where it came from, how to restore
        training_data/train.jsonl       the data it was trained on
        training_data/valid.jsonl
      brew_loose_leaf_tea_150_WORKER/
        brew_loose_leaf_tea_150.safetensors
        ...

WHY THIS IS AN ARCHIVE AND NOT THE LIVE LAYOUT

mlx_lm hardcodes the filename it loads:

    mlx_lm/tuner/utils.py: load_adapters()
        model.load_weights(str(adapter_path / "adapters.safetensors"))

An adapter file named anything else cannot be loaded — including at boot. So
this copies rather than moves: adapters/ keeps working exactly as it does now,
and Adapter_skills/ is the organised, browsable, portable version, named the
way you asked. --restore turns any archived folder back into a directory
mlx_lm will load, so nothing here is a one-way trip.

Iterations come from adapter_config.json["iters"], falling back to
training_progress.json["total_iters"], falling back to 0.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTERS = WORKER_ADAPTERS = TRAINING = WORKER_TRAINING = OUT = Path()


def set_root(root: Path) -> None:
    """Point every path at `root`.

    The default is the directory this script sits in, which is the project.
    --root exists so it can be pointed at a checkout it does not live in —
    which is also how it gets tested without a copy of the adapters.
    """
    global ROOT, ADAPTERS, WORKER_ADAPTERS, TRAINING, WORKER_TRAINING, OUT
    ROOT = root.resolve()
    ADAPTERS = ROOT / "adapters"
    WORKER_ADAPTERS = ADAPTERS / "workers"
    TRAINING = ROOT / "training_data"
    WORKER_TRAINING = TRAINING / "workers"
    OUT = ROOT / "Adapter_skills"


set_root(Path(__file__).resolve().parent)

WEIGHTS = "adapters.safetensors"
META_FILES = ("adapter_config.json", "training_progress.json", "last_used.json")
DATA_FILES = ("train.jsonl", "valid.jsonl")


def _iters(adapter_dir: Path) -> int:
    """How many iterations this adapter was trained for."""
    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        try:
            n = json.loads(cfg.read_text(encoding="utf-8")).get("iters")
            if isinstance(n, int) and n > 0:
                return n
        except (json.JSONDecodeError, OSError):
            pass
    prog = adapter_dir / "training_progress.json"
    if prog.exists():
        try:
            n = json.loads(prog.read_text(encoding="utf-8")).get("total_iters")
            if isinstance(n, int) and n > 0:
                return n
        except (json.JSONDecodeError, OSError):
            pass
    return 0


def _sources() -> list[tuple[str, str, Path, Path]]:
    """(skill, kind, adapter_dir, training_dir) for everything worth archiving."""
    found: list[tuple[str, str, Path, Path]] = []
    if (ADAPTERS / WEIGHTS).exists():
        found.append(("HEADMASTER", "HEADMASTER", ADAPTERS, TRAINING))
    if WORKER_ADAPTERS.is_dir():
        for d in sorted(p for p in WORKER_ADAPTERS.iterdir() if p.is_dir()):
            if (d / WEIGHTS).exists():
                found.append((d.name, "WORKER", d, WORKER_TRAINING / d.name))
    return found


def build(dry_run: bool = False) -> int:
    sources = _sources()
    if not sources:
        print(f"No adapters found under {ADAPTERS}")
        return 1

    print(f"{'Would archive' if dry_run else 'Archiving'} {len(sources)} adapter(s) "
          f"into {OUT}\n")
    total_bytes = 0

    for skill, kind, adapter_dir, training_dir in sources:
        iters = _iters(adapter_dir)
        folder = OUT / f"{skill}_{iters}_{kind}"
        weights_name = f"{skill}_{iters}.safetensors"
        size = (adapter_dir / WEIGHTS).stat().st_size
        total_bytes += size

        data_files = [f for f in DATA_FILES if (training_dir / f).exists()]
        print(f"  {folder.name}")
        print(f"      {weights_name}  ({size / 1e6:.1f} MB)")
        print(f"      metadata: {', '.join(f for f in META_FILES if (adapter_dir / f).exists())}")
        print(f"      training_data: {', '.join(data_files) or '(none found)'}")

        if dry_run:
            continue

        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_dir / WEIGHTS, folder / weights_name)
        for f in META_FILES:
            if (adapter_dir / f).exists():
                shutil.copy2(adapter_dir / f, folder / f)
        if data_files:
            (folder / "training_data").mkdir(exist_ok=True)
            for f in data_files:
                shutil.copy2(training_dir / f, folder / "training_data" / f)

        # Enough to put it back, and to know what it was without guessing from
        # the folder name.
        (folder / "manifest.json").write_text(json.dumps({
            "skill": skill,
            "kind": kind,
            "iters": iters,
            "weights": weights_name,
            "archived": datetime.now().isoformat(timespec="seconds"),
            "source_adapter_dir": str(adapter_dir),
            "source_training_dir": str(training_dir),
            "restore_hint": (
                "Copy the .safetensors back as 'adapters.safetensors' beside "
                "adapter_config.json — mlx_lm loads that exact filename. "
                f"Or: python3 archive_adapters.py --restore {folder.name}"),
        }, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'Would copy' if dry_run else 'Copied'} {total_bytes / 1e6:.0f} MB. "
          f"adapters/ is untouched and still what the app loads.")
    return 0


def list_archive() -> int:
    if not OUT.is_dir():
        print(f"No archive yet. Run: python3 {Path(__file__).name}")
        return 1
    rows = []
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        man = d / "manifest.json"
        info = {}
        if man.exists():
            try:
                info = json.loads(man.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        weights = d / info.get("weights", "")
        size = weights.stat().st_size / 1e6 if weights.exists() else 0.0
        rows.append((d.name, info.get("kind", "?"), info.get("iters", "?"), size))
    width = max((len(r[0]) for r in rows), default=10)
    for name, kind, iters, size in rows:
        print(f"  {name:{width}}  {kind:10}  {iters:>6} iters  {size:6.1f} MB")
    print(f"\n  {len(rows)} archived under {OUT}")
    return 0


def restore(name: str) -> int:
    folder = OUT / name
    man = folder / "manifest.json"
    if not man.exists():
        print(f"No archive named {name} (looked for {man})")
        return 1
    info = json.loads(man.read_text(encoding="utf-8"))
    weights = folder / info["weights"]
    if not weights.exists():
        print(f"Archive is missing its weights: {weights}")
        return 1

    dest = ROOT / f"restored_{name}"
    dest.mkdir(parents=True, exist_ok=True)
    # The whole point: back to the filename mlx_lm insists on.
    shutil.copy2(weights, dest / WEIGHTS)
    for f in META_FILES:
        if (folder / f).exists():
            shutil.copy2(folder / f, dest / f)

    print(f"Restored to {dest}")
    print(f"  {WEIGHTS} + {', '.join(f for f in META_FILES if (dest / f).exists())}")
    print("\nThis directory is loadable as-is:")
    print(f"  load(model_name, adapter_path='{dest}')")
    print("\nIt was NOT copied over adapters/ — moving a live adapter into place "
          "is your call, and worth backing up first.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="show, do not write")
    ap.add_argument("--list", action="store_true", help="list what is archived")
    ap.add_argument("--restore", metavar="NAME", help="rebuild a loadable dir")
    ap.add_argument("--root", metavar="PATH",
                    help="project directory (default: where this script lives)")
    args = ap.parse_args()

    if args.root:
        set_root(Path(args.root))

    if args.list:
        return list_archive()
    if args.restore:
        return restore(args.restore)
    return build(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
