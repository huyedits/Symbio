#!/usr/bin/env python3
"""Does N examples of ONE verb teach it, with no sibling grammar to lean on?

The full run measured "~3 examples per verb" — but those three arrived inside a
60-example corpus that had already taught the shared grammar: --region on every
command, the --option value shape, Key:Value labels. Three examples of
`compute resume` only had to teach WHICH VERB, not the syntax around it.

That distinction decides how the loop feels in daily use. If three is three
flat, ordinary corrections teach new tools continuously. If the FIRST
capability in a domain costs fifteen and the rest cost three, it learns tools
cheaply and domains expensively.

So: train on one verb's examples alone, nothing else, and score the battery.
Base is 0/13, so anything that passes was taught by those few examples and
nothing else.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import eval_aws

CORPUS = HERE / "self_taught.jsonl"
WORK = HERE / "cold"

# (verb prefix in the corpus, the battery case it should light up)
TRIALS = [
    ("store duplicate", "duplicate"),      # 4 flags + region: the hardest shape
    ("compute resume", "resume_node"),     # 1 flag + region: the simplest
]


def samples_for(verb: str) -> list[dict]:
    rows = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        command = row["messages"][2]["content"]
        if " ".join(command.split()[1:3]) == verb:
            rows.append(row)
    return rows


def train_on(rows: list[dict], adapter_dir: Path, model_name: str,
             iters: int) -> None:
    data = adapter_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid"):
        with (data / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", model_name,
           "--train", "--data", str(data), "--adapter-path", str(adapter_dir),
           "--iters", str(iters), "--save-every", str(iters),
           "--steps-per-eval", str(iters), "--batch-size", "1",
           "--num-layers", "8", "--max-seq-length", "1024",
           "--learning-rate", "1e-4", "--grad-checkpoint"]
    proc = subprocess.run(cmd, cwd=str(HERE.parent),
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"training failed ({proc.returncode})")


def score(model_name: str, adapter_dir: Path, target: str) -> tuple[int, bool, str]:
    from mlx_lm import generate as generate_fn
    from mlx_lm import load

    model, tokenizer = load(model_name, adapter_path=str(adapter_dir))
    total, rows = eval_aws.run_battery(model, tokenizer, generate_fn,
                                       verbose=False)
    hit = next((r for r in rows if r[0] == target), None)
    del model
    return total, bool(hit and hit[1]), (hit[2] if hit else "")


def main():
    from symbio.app import config as app_config

    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    model_name = app_config.load_config()["model_name"]
    print(f"cold-start trials, {iters} iterations each\n")

    for verb, case_id in TRIALS:
        rows = samples_for(verb)
        if not rows:
            print(f"  {verb}: no samples in the corpus, skipping")
            continue
        adapter_dir = WORK / verb.replace(" ", "_")
        if adapter_dir.exists():
            import shutil
            shutil.rmtree(adapter_dir)
        print(f"--- {verb}: {len(rows)} example(s), nothing else ---", flush=True)
        for row in rows:
            print(f"      {row['messages'][2]['content'][:96]}")
        t0 = time.time()
        train_on(rows, adapter_dir, model_name, iters)
        total, passed, detail = score(model_name, adapter_dir, case_id)
        print(f"    {case_id}: {'PASS' if passed else 'FAIL'}   "
              f"(whole battery {total}/13, {time.time() - t0:.0f}s)")
        print(f"    emitted: {detail[:110]}\n", flush=True)

    print("Base model scores 0/13, so anything passing here was taught by "
          "those few examples alone.")


if __name__ == "__main__":
    main()
