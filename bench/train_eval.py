"""Fine-tune, then measure whether it actually helped.

    python bench/train_eval.py                     # baseline -> train -> re-measure
    python bench/train_eval.py --iters 60          # shorter pass
    python bench/train_eval.py --skip-baseline     # reuse the last baseline

A training run that is never graded is a training run that might have made the
model worse. This does the same eval before and after, on the same items, and
reports the difference with the honest error bars — because on 120 items a
gap under about ten points is indistinguishable from noise, and reporting one
as an improvement is how a regression ships.

The adapter is backed up before training and restored if the score drops, so
a bad run costs time rather than the adapter.
"""
import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import harness  # noqa: E402
from symbio import constants  # noqa: E402
from symbio.app import config as app_config  # noqa: E402
from symbio.app import training  # noqa: E402

# Fine-tuning a model this big does not fit in 16 GB — mlx_lm's LoRA still has
# to hold the base weights, the adapter, gradients and optimiser state at once.
# Refusing up front beats discovering it after an hour and an OOM kill.
TRAINABLE_CEILING_GB = 5.0


def _model_size_gb(name):
    hub = Path.home() / ".cache/huggingface/hub"
    d = hub / ("models--" + name.replace("/", "--"))
    if not d.exists():
        return None
    return sum(f.stat().st_size for f in d.rglob("*.safetensors")) / 1e9


def _score(label, suite, model, adapter, limit, out):
    print(f"\n{'='*66}\n  {label}\n{'='*66}", flush=True)
    return harness.run(suite, model, limit=limit, max_tokens=640,
                       out=out, adapter_path=adapter)["accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="bfcl_live", choices=list(harness.SUITES))
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="train even if the model looks too big for this machine")
    a = ap.parse_args()

    cfg = app_config.load_config()
    model = cfg["model_name"]
    adapter_dir = constants.ADAPTER_DIR

    size = _model_size_gb(model)
    print(f"model   : {model}" + (f"  ({size:.2f} GB)" if size else ""))
    print(f"suite   : {a.suite}   items: {a.limit}")
    print(f"adapter : {adapter_dir}")

    if size and size > TRAINABLE_CEILING_GB and not a.force:
        print(f"\nREFUSING: {size:.1f} GB base is beyond what LoRA fits in 16 GB "
              f"(ceiling {TRAINABLE_CEILING_GB} GB).\n"
              f"Switch model_name to a smaller base — Qwen/Qwen3-8B-MLX-4bit is "
              f"4.35 GB and is what the existing adapter was trained against — "
              f"or pass --force to try anyway.")
        return 1

    baseline_file = ROOT / "bench" / f"traineval_before_{a.suite}.json"
    if a.skip_baseline and baseline_file.exists():
        before = json.load(open(baseline_file))["accuracy"]
        print(f"\nreusing baseline: {before:.1f}%")
    else:
        before = _score("BASELINE — base weights, no adapter", a.suite, model,
                        None, a.limit, baseline_file)

    # Back the adapter up before training overwrites it.
    backup = None
    if adapter_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = adapter_dir.parent / f"adapters.pre_traineval_{stamp}"
        shutil.copytree(adapter_dir, backup)
        print(f"\nadapter backed up -> {backup.name}")

    print(f"\n{'='*66}\n  TRAINING\n{'='*66}", flush=True)
    t0 = time.perf_counter()
    ok = training.run_training(cfg, iters=a.iters)
    mins = (time.perf_counter() - t0) / 60
    if not ok:
        print(f"\nTraining did not complete ({mins:.1f} min). Nothing re-measured.")
        return 1
    print(f"\ntraining finished in {mins:.1f} min")

    after = _score("AFTER — same weights, with the new adapter", a.suite, model,
                   str(adapter_dir), a.limit,
                   ROOT / "bench" / f"traineval_after_{a.suite}.json")

    n = a.limit
    pa, pb = after / 100, before / 100
    se = math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n) * 100
    diff = after - before
    z = diff / se if se else 0.0

    print(f"\n{'='*66}\n  RESULT\n{'='*66}")
    print(f"  before : {before:5.1f}%")
    print(f"  after  : {after:5.1f}%")
    print(f"  change : {diff:+5.1f} pts   (SE {se:.1f}, z {z:+.2f}, n={n})")
    if abs(z) < 1.96:
        print(f"  verdict: NOT significant — indistinguishable from no change.\n"
              f"           A gap under ~{1.96*se:.0f} pts cannot be resolved at n={n}.")
    elif diff > 0:
        print("  verdict: a real improvement.")
    else:
        print("  verdict: a real REGRESSION — training made it worse.")

    if diff < 0 and abs(z) >= 1.96 and backup:
        shutil.rmtree(adapter_dir)
        shutil.copytree(backup, adapter_dir)
        print(f"\n  adapter rolled back from {backup.name} — the trained one scored worse.")
    elif backup:
        print(f"\n  trained adapter kept. Previous one is at {backup.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
