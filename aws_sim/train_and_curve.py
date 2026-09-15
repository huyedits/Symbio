#!/usr/bin/env python3
"""Train on what the model earned, then score every checkpoint.

The question is "what does 1000 iterations across 50 checkpoints actually do",
and the only honest way to answer it is to grade every checkpoint on a held-out
battery and print the curve. A single before/after number cannot tell learning
from luck, and cannot show the point where more training starts making it
worse — which is the thing worth knowing.

Checkpoints are evaluated by loading each one into the SAME resident model, not
by reloading the 14B fifty times. mlx_lm keeps LoRA weights as live attributes,
so swapping a checkpoint in is a weight assignment.

Nothing here writes a training example. The corpus is whatever self_teach.py
earned by execution.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import eval_aws

CORPUS = HERE / "self_taught.jsonl"
DATA_DIR = HERE / "data"
ADAPTER_DIR = HERE / "adapters_aws"
CURVE = HERE / "curve.json"


def split_corpus(valid_fraction: float = 0.15) -> tuple[int, int]:
    rows = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no earned samples — self_teach.py produced nothing to train on")
    random.Random(3).shuffle(rows)
    cut = max(1, int(len(rows) * valid_fraction))
    valid, train = rows[:cut], rows[cut:]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("valid", valid)):
        with (DATA_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(train), len(valid)


def train(model_name: str, iters: int, save_every: int) -> float:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--train",
        "--data", str(DATA_DIR),
        "--adapter-path", str(ADAPTER_DIR),
        "--iters", str(iters),
        "--save-every", str(save_every),
        "--steps-per-eval", str(max(save_every, iters // 10)),
        "--batch-size", "1",
        "--num-layers", "8",
        "--max-seq-length", "1024",
        "--learning-rate", "1e-4",
        "--grad-checkpoint",
    ]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(HERE.parent))
    took = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"training failed with exit {proc.returncode}")
    return took


def checkpoints() -> list[tuple[int, Path]]:
    found = []
    for path in ADAPTER_DIR.glob("*_adapters.safetensors"):
        stem = path.name.split("_")[0]
        if stem.isdigit():
            found.append((int(stem), path))
    return sorted(found)


def _live_module_paths(model) -> list[str]:
    return [name for name, mod in model.named_modules()
            if hasattr(mod, "lora_a") and hasattr(mod, "lora_b")]


def _unloadable_keys(path: Path, live: set[str]) -> int:
    """How many of a checkpoint's tensors have no module to land on.

    The whole point is that this is never allowed to be silent: mlx's
    load_weights(strict=False) drops unknown keys without a word, which is the
    right behaviour for a loader and a disaster for an experiment.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        keys = list(handle.keys())
    return sum(1 for key in keys if key.rsplit(".", 1)[0] not in live)


def score_curve(model_name: str) -> list[dict]:
    """Grade every checkpoint on the held-out battery, in one resident model."""
    from mlx_lm import generate as generate_fn
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    print(f"loading {model_name.split('/')[-1]} once for the whole curve ...",
          flush=True)
    model, tokenizer = load(model_name)

    base, base_rows = eval_aws.run_battery(model, tokenizer, generate_fn,
                                           verbose=False)
    print(f"  iter      0 (base, no adapter): {base}/{len(base_rows)}", flush=True)
    curve = [{"iter": 0, "score": base, "of": len(base_rows),
              "failing": [cid for cid, ok, _ in base_rows if not ok],
              "emitted": {cid: d for cid, ok, d in base_rows if not ok}}]

    # Attach LoRA exactly as TRAINING attached it, by reading the config the
    # training run wrote rather than assuming. The first version of this
    # assumed q_proj/v_proj; mlx_lm's `lora` CLI has no --keys flag and used
    # its defaults, so the checkpoints carry q/k/v/o_proj AND all three MLP
    # projections — 112 tensors against the 32 I had attached. load_weights
    # with strict=False then loaded the 32 it recognised and dropped the other
    # 80 WITHOUT A WORD, so every checkpoint scored 0/13 while its training
    # loss said 0.055. A silent partial load looks exactly like a model that
    # learned nothing.
    trained_cfg = json.loads(
        (ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    lora_cfg = dict(trained_cfg.get("lora_parameters") or {})
    num_layers = int(trained_cfg.get("num_layers", 8))
    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_cfg)
    model.eval()

    live = set(_live_module_paths(model))
    for iteration, path in checkpoints():
        missing = _unloadable_keys(path, live)
        if missing:
            raise SystemExit(
                f"{path.name} has {missing} tensor(s) with nowhere to load: the "
                f"eval attaches LoRA differently from training, and strict=False "
                f"would hide it. Fix the attach, do not score this.")
        model.load_weights(str(path), strict=False)
        model.eval()
        score, rows = eval_aws.run_battery(model, tokenizer, generate_fn,
                                           verbose=False)
        # Keep WHAT IT EMITTED, not just which cases failed. The per-case
        # failing set already makes acquisition distinct rather than inferred
        # — and reading it showed acquisition is not monotonic: drop_container
        # was acquired, lost, reacquired, lost and reacquired again across one
        # run while the aggregate score rose the whole time. But a boolean
        # cannot say WHY a case was lost, and for a chain it cannot even say
        # whether step one was right. The emission can.
        curve.append({"iter": iteration, "score": score,
                      "of": len(rows),
                      "failing": [cid for cid, ok, _ in rows if not ok],
                      "emitted": {cid: detail for cid, ok, detail in rows
                                  if not ok}})
        print(f"  iter {iteration:6}: {score}/{len(rows)}", flush=True)
    return curve


def main():
    from symbio.app import config as app_config

    # Positional arguments are read AFTER the flags are taken out; reading
    # argv[1] as an int first meant --score-only crashed on its own flag.
    numbers = [a for a in sys.argv[1:] if not a.startswith("--")]
    iters = int(numbers[0]) if numbers else 1000
    save_every = int(numbers[1]) if len(numbers) > 1 else 20
    cfg = app_config.load_config()
    model_name = cfg["model_name"]

    if "--score-only" in sys.argv:
        print(f"scoring {len(checkpoints())} existing checkpoints\n")
        curve = score_curve(model_name)
        CURVE.write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")
        best = max(curve, key=lambda row: row["score"])
        print(f"\n=== CURVE ===\nbase:  {curve[0]['score']}/{curve[0]['of']}")
        print(f"best:  {best['score']}/{best['of']} at iteration {best['iter']}")
        print(f"final: {curve[-1]['score']}/{curve[-1]['of']}")
        return

    n_train, n_valid = split_corpus()
    print(f"corpus the model earned: {n_train} train / {n_valid} valid")
    print(f"plan: {iters} iterations, a checkpoint every {save_every} "
          f"({iters // save_every} checkpoints)\n")

    took = train(model_name, iters, save_every)
    print(f"\ntraining took {took / 60:.1f} min "
          f"({took / max(1, iters):.1f}s per iteration)")

    print(f"\n--- scoring {len(checkpoints())} checkpoints on the held-out "
          f"battery ---", flush=True)
    curve = score_curve(model_name)
    CURVE.write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")

    best = max(curve, key=lambda row: row["score"])
    print(f"\n=== CURVE ===")
    print(f"base:  {curve[0]['score']}/{curve[0]['of']}")
    print(f"best:  {best['score']}/{best['of']} at iteration {best['iter']}")
    print(f"final: {curve[-1]['score']}/{curve[-1]['of']} at "
          f"iteration {curve[-1]['iter']}")
    if best["iter"] not in (0, curve[-1]["iter"]):
        print(f"NOTE: it peaked at {best['iter']} and was WORSE by the end — "
              f"more training made it worse, which is the thing a single "
              f"before/after number hides.")
    print(f"curve -> {CURVE}")


if __name__ == "__main__":
    main()
