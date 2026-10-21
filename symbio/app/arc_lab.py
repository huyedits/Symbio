"""ARC-AGI-2 laboratory: the fluid-learning experiment, run by the harness.

What this module exists for: the harness learns. Here the lesson material is
ARC-AGI-2's public training set, the skill is the puzzle grammar itself, and
the question is whether a grown expert — spawned delta-expert slots plus a
LoRA on the last attention layer — can wrap its head around transformations
the frozen base only stumbles through in-context.

Everything runs through the harness's own telemetry: the trainer subprocess
prints the "Iter N: Train loss X" lines training_live.line() was built to
parse, so the pet eats its droplets and the TUI shows the curve while the
run happens. The conversion and the evaluation are subprocesses too; the chat
narrates. Nothing here imports mlx — the chat session stays light.

Commands (via the /arc chat command):
  /arc train [n_tasks] [spawn N]  — convert + train on the public training set
  /arc eval  [n_eval]             — paired base-vs-adapted scoring on public
                                    evaluation tasks (never trained on)
  /arc status                     — what runs exist, where their adapters are

Artifacts live under bench/arc/: train.jsonl / valid.jsonl (converted tasks),
eval.jsonl (held-out), adapters_lfm_arc/ (trainer output), eval_results.json.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
PAGED_DIR = PROJECT / "bench" / "paged_moe"
ARC_DIR = PROJECT / "bench" / "arc"
TRAIN_LFM = PAGED_DIR / "train_lfm.py"
EVAL_LFM = PAGED_DIR / "eval_arc.py"

MODEL_REL = "~/.local/share/mlx-models/LFM2.5-24B-A2B-MLX-4bit"
ARC_DATA = Path("~/.local/share/ARC-AGI-2/data").expanduser()
TRAIN_SCRIPT_SPEC = (
    "You are an ARC puzzle solver. Each task shows example input/output grid "
    "pairs that reveal one transformation rule; apply that same rule to the "
    "test input. Grids are rows of digits 0-9, one row per line, spaces "
    "between cells. Reply with ONLY the test output grid, nothing else."
)

_SYS_RE = re.compile(r"Iter\s+(\d+):\s+Train\s+loss\s+([-+\d.eE]+)")


def _venv_python() -> str:
    return str(PROJECT / ".trial-venv" / "bin" / "python3")


def _render_task(task: dict) -> dict | None:
    """ARC JSON -> {"messages":[...]} in this model's chat format."""
    try:
        train_pairs = task["train"]
        test_in = task["test"][0]["input"]
        test_out = task["test"][0].get("output")
    except (KeyError, IndexError):
        return None

    def grid(g: list[list[int]]) -> str:
        return "\n".join(" ".join(str(c) for c in row) for row in g)

    parts = []
    for n, pair in enumerate(train_pairs, 1):
        parts.append(f"EXAMPLE {n}\nINPUT:\n{grid(pair['input'])}\nOUTPUT:"
                     f"\n{grid(pair['output'])}")
    parts.append(f"TEST INPUT:\n{grid(test_in)}")
    user = "\n\n".join(parts)

    msgs = [{"role": "system", "content": TRAIN_SCRIPT_SPEC},
            {"role": "user", "content": user}]
    if test_out is not None:
        msgs.append({"role": "assistant", "content": grid(test_out)})
    return {"messages": msgs, "task_id": task.get("task_id", "")}


def convert(train_count: int = 300) -> dict:
    """Shuffle-free split: first `train_count` tasks by name into train,
    all evaluation tasks into eval (held-out by construction)."""
    ARC_DIR.mkdir(parents=True, exist_ok=True)
    train_files = sorted((ARC_DATA / "training").glob("*.json"))
    eval_files = sorted((ARC_DATA / "evaluation").glob("*.json"))
    out = {"train_rows": 0, "eval_rows": 0}

    for split, files, cap in (("train", train_files[:train_count], train_count),
                              ("valid", train_files[:1], 1),
                              ("eval", eval_files, None)):
        rows = []
        for f in files:
            if cap is not None and len(rows) >= cap:
                break
            task = json.loads(f.read_text())
            task["task_id"] = f.stem
            rendered = _render_task(task)
            if rendered is not None:
                rows.append(rendered)
        dest = ARC_DIR / f"{split}.jsonl"
        dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out[f"{split}_rows"] = len(rows)
    return out


def _tee(stream, output_fn, on_line):
    for raw in iter(stream.readline, ""):
        line = raw.rstrip("\n")
        if output_fn:
            try:
                output_fn(f"    {line}")
            except Exception:
                pass
        if on_line:
            on_line(line)
    stream.close()


def run_training(count: int = 300, spawn: int = 4, iters: int | None = None,
                 output_fn=None) -> str:
    """Convert + train through a subprocess, narrated through training_live."""
    from symbio.app import training_live  # harness's own telemetry writer

    conv = convert(train_count=count) if not (
        ARC_DIR / "train.jsonl").exists() else {"train_rows": "cached"}
    output_fn(f"  [ARC] train rows: {conv.get('train_rows')}")

    iters = iters or count
    model = str(Path(MODEL_REL).expanduser())
    cmd = [_venv_python(), TRAIN_LFM.name,
           "--model", model,
           "--data", str(ARC_DIR),
           "--iters", str(iters),
           "--max-len", "1536",
           "--spawn", str(spawn),
           "--hot-budget-mb", "3500",
           "--rank", "8",
           "--lr", "1e-4",
           "--save-every", str(max(10, iters // 4)),
           "--out", "adapters_lfm_arc",
           ]
    training_live.begin("arc", iters)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(PAGED_DIR), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONPATH": str(PAGED_DIR)})
    losses: list[float] = []

    def on_line(line: str):
        m = _SYS_RE.search(line)
        if m:
            losses.append(float(m.group(2)))

    tee_out = threading.Thread(target=_tee, args=(proc.stdout, output_fn, on_line), daemon=True)
    tee_out.start()
    code = proc.wait()
    tee_out.join(timeout=5)
    dur = time.time() - t0
    ok = code == 0 and losses
    training_live.end("trained" if ok else "failed",
                      f"exit {code}" if code else "", iter=len(losses),
                      total_iters=iters)
    if losses:
        out_path = ARC_DIR / "adapter_out"
        summary = (f"  [ARC] trained {len(losses)}/{iters} iters in {dur / 60:.1f} min; "
                   f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    else:
        summary = f"  [ARC] training failed (exit {code}) after {dur / 60:.1f} min"
    return summary


def run_eval(count: int = 120, output_fn=None) -> str:
    """Paired base-vs-adapted scoring on held-out evaluation tasks."""
    model = str(Path(MODEL_REL).expanduser())
    adapters = PAGED_DIR / "adapters_lfm_arc" / "adapters.safetensors"
    if not adapters.exists():
        return "  [ARC] no trained adapters yet — run /arc train first"
    cmd = [_venv_python(), EVAL_LFM.name,
           "--model", model,
           "--data", str(ARC_DIR / "eval.jsonl"),
           "--adapters", str(adapters),
           "--count", str(count),
           "--spawn", "4",
           "--out", str(ARC_DIR / "eval_results.json")]
    output_fn("  [ARC] scoring held-out evaluation tasks (base vs grown)...")
    proc = subprocess.run(cmd, cwd=str(PAGED_DIR), capture_output=True,
                          text=True,
                          env={**os.environ, "PYTHONPATH": str(PAGED_DIR)})
    tail = proc.stdout.strip().splitlines()[-3:] if proc.stdout.strip() else []
    for line in tail:
        output_fn(f"    {line}")
    if proc.returncode != 0:
        return f"  [ARC] eval failed (exit {proc.returncode})"
    try:
        res = json.loads((ARC_DIR / "eval_results.json").read_text())["summary"]
        return (f"  [ARC] held-out eval (n={res['n']}): "
                f"base cell-acc {res['base_cell_acc']:.1%} vs "
                f"grown {res['adapted_cell_acc']:.1%} "
                f"(delta {res['cell_delta']:+.1%}); "
                f"base loss {res['base_loss']:.3f} vs "
                f"grown {res['adapted_loss']:.3f}")
    except Exception as e:
        return f"  [ARC] eval ran but summary parse failed: {e}"


def handle(session, arg: str) -> None:
    """The /arc chat command: dispatches subcommands, narrates in chat."""
    output_fn = session.output_fn
    parts = (arg or "").split()
    sub = parts[0] if parts else "status"
    try:
        if sub == "train":
            count = int(parts[1]) if len(parts) > 1 else 300
            spawn = int(parts[2]) if len(parts) > 2 else 4
            iters = int(parts[3]) if len(parts) > 3 else count
            output_fn(f"  [ARC] growing an expert on {count} ARC-AGI-2 "
                      f"training tasks, {spawn} delta slots, {iters} iters...")
            output_fn(run_training(count, spawn, iters, output_fn))
        elif sub == "eval":
            n = int(parts[1]) if len(parts) > 1 else 120
            output_fn(run_eval(n, output_fn))
        elif sub == "status":
            ad = PAGED_DIR / "adapters_lfm_arc" / "adapters.safetensors"
            res = ARC_DIR / "eval_results.json"
            output_fn(f"  [ARC] adapters: {'present' if ad.exists() else 'none'}"
                      f" ({PAGED_DIR / 'adapters_lfm_arc'})")
            if res.exists():
                output_fn(f"  [ARC] last eval: {res.read_text()[:400]}")
        else:
            output_fn("  Usage: /arc train [n_tasks] [spawn N] [iters] | "
                      "/arc eval [n] | /arc status")
    except Exception as e:
        output_fn(f"  [ARC] error: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    class _Stub:
        output_fn = staticmethod(print)
    _Stub and handle(_Stub(), " ".join(sys.argv[1:]) or "status")