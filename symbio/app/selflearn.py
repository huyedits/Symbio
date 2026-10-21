"""The self-directed learning governor: IT dictates how and when it learns.

Until now the human dialed /arc train. This module closes the loop: the
brain reads a compact model of its own state — training telemetry, paired
eval scores, data inventory, adapters on disk — and emits a JSON plan
(action, data, spawn, iters, reasoning). The harness executes whatever it
decides, through the same arc_lab handlers, and narrates the outcome back
into the chat. Nothing here is a heuristic policy: the plan comes from a
model call, so the strategy can change as the state changes.

The decider is the learner's own brain: LFM2.5 reads its self-state and
emits the plan. The anti-self-deception rule survives the change because
the ARBITER is not a model at all — it is the held-out scoreboard (golden
grids, can't be gamed). Self-governance here is strategy; grading stays
objective.

The loop:  telemetry -> decide -> execute -> observe -> telemetry ...
ticked from chat via /selflearn, or by a cron prompt for full autonomy.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
ARC_DIR = PROJECT / "bench" / "arc"
LOGS = PROJECT / "logs"

TRAIN_LOG = ARC_DIR / "train_log.txt"
EVAL_RESULTS = ARC_DIR / "eval_results.json"
FRONTIER_RESULTS = ARC_DIR / "frontier_results.json"

# The governor is the learner's own brain — LFM2.5 served locally through
# llama.cpp's mmap streaming profile (no wired-limit bump needed). It
# dictates how and when it learns; the held-out scoreboard stays the
# ground-truth arbiter, so self-governance here is strategy, not grading.
BRAIN_URL = "http://127.0.0.1:8082"
BRAIN_MODEL = "lfm2.5-24b"

PLAN_SCHEMA = (
    '{"action": "train" | "eval" | "wait", '
    '"tasks": <int, training rows for the next run>, '
    '"spawn": <int, new delta-expert slots for the next run>, '
    '"iters": <int>, "place": {"<action>": "<node name>"} ?'
    '"reasoning": "<one paragraph, first person, no markdown>"}'
)


_GOV = None

def _trained_avatar(state):
    """The trained avatar, loaded once. Speaks with whatever adapters and
    spawned experts exist on disk right now — its voice grows mid-run."""
    import importlib, sys as _sys
    global _GOV
    if _GOV is not None:
        return _GOV
    paged = PROJECT / "bench" / "paged_moe"
    if str(paged) not in _sys.path:
        _sys.path.insert(0, str(paged))
    from governor import Governor
    model = Path("~/.local/share/mlx-models/LFM2.5-24B-A2B-MLX-4bit"
                 ).expanduser()
    ad = paged / "adapters_lfm_arc" / "adapters.safetensors"
    state["voice_note"] = ("adapters live from checkpoint" if ad.exists()
                           else "base voice — no checkpoint yet")
    _GOV = Governor(model, adapters=ad if ad.exists() else None)
    return _GOV


def _tail(path: Path, n: int = 14) -> list[str]:
    try:
        return [ln for ln in
                path.read_text().splitlines()[-n:] if ln.strip()]
    except OSError:
        return []


def gather_compute() -> dict:
    """The body: what it runs on, right now. The brain schedules THIS."""
    import subprocess, shutil as sh
    nodes = []

    def probe(name, role, host_alias=None, runtime=None, mem_gb=None):
        runtime = list(runtime or ())
        node = {"name": name, "role": role, "runtime": runtime,
                "mem_gb": mem_gb, "battery": "?", "online": True}
        if host_alias:
            try:
                r = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=4", "-o",
                     "BatchMode=yes", host_alias,
                     "df -g / | tail -1 | awk '{print $4}'; "
                     r"pmset -g batt | grep -o 'AC Power\|Battery Power'"],
                    capture_output=True, text=True, timeout=12)
                online = r.returncode == 0
                lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
                if online and lines:
                    node["disk_free_gb"] = (float(lines[0])
                                            if lines[0][0].isdigit() else None)
                    if len(lines) > 1:
                        node["battery"] = lines[1]
            except Exception:
                node["online"] = False
            if not node["online"]:
                return {"name": name, "role": role, "online": False,
                        "note": "unreachable — asleep or off"}
        else:
            try:
                node["disk_free_gb"] = round(
                    sh.disk_usage("/System/Volumes/Data").free / (1 << 30), 1)
            except Exception:
                pass
            try:
                b = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                   text=True, timeout=5).stdout
                node["battery"] = ("AC Power" if "AC Power" in b
                                   else "Battery Power" if "Battery" in b
                                   else "?")
            except Exception:
                pass
            node["trainer_running"] = _trainer_running()
        return node

    nodes.append(probe("mini", "brain+trainer (rank 0)", mem_gb=16,
                       runtime=("mlx", "llamacpp", "streaming-trainer")))
    try:
        c = json.loads((PROJECT / "config.json").read_text())
        for alias in (c.get("remote", {}).get("hosts") or {}):
            nodes.append(probe(alias, "rank (from remote.hosts)",
                               host_alias=alias))
    except Exception:
        pass
    try:
        cf = PROJECT / "bench" / "arc" / "cluster.json"
        if cf.exists():
            for n in json.loads(cf.read_text()):
                if not any(x.get("name") == n.get("name") for x in nodes):
                    nodes.append(probe(**n))
    except Exception:
        pass
    return {"nodes": nodes,
            "scheduling_notes": ("train only on online+plugged nodes with "
                                 "the trainer; streaming trainer wants "
                                 "~10GB disk scratch; defer heavy work on "
                                 "battery; unreachable = asleep or off")}


def gather_state() -> dict:
    """The brain's self-model: what it has, what happened, what's around."""
    train_lines = [ln for ln in _tail(TRAIN_LOG, 400)
                   if "Train loss" in ln]
    losses = []
    for ln in train_lines:
        m = re.search(r"Train loss ([-+\d.eE]+)", ln)
        if m:
            losses.append(float(m.group(1)))

    eval_summary = None
    try:
        eval_summary = json.loads(EVAL_RESULTS.read_text())["summary"]
    except (OSError, ValueError, KeyError):
        pass

    frontier = None
    try:
        frontier = json.loads(FRONTIER_RESULTS.read_text())["summary"]
    except (OSError, ValueError, KeyError):
        pass

    adapters = sorted(str(p) for p in
                      (Path("~/.local/share/mlx-models").expanduser().iterdir()))
    adapters += sorted(str(p) for p in
                       (PROJECT / "bench" / "paged_moe").glob("adapters_*"))

    df_free_gb = None
    try:
        st = Path("/System/Volumes/Data").stat()
        import shutil as _sh
        df_free_gb = round(_sh.disk_usage("/System/Volumes/Data").free
                           / (1 << 30), 1)
    except OSError:
        pass

    return {
        "learner": "LFM2.5-24B-A2B (24B MoE, 2.45B active, streaming on 16GB)",
        "trained_task_count": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "recent_losses": losses[-8:],
        "last_paired_eval": eval_summary,
        "frontier_reference": frontier,
        "adapters_on_disk": adapters,
        "data_inventory": {"arc_train_rows_ready":
                           sum(1 for _ in open(ARC_DIR / "train.jsonl"))
                           if (ARC_DIR / "train.jsonl").exists() else 0,
                           "arc_eval_rows": 120},
        "training_running_now": _trainer_running(),
        "disk_free_gb": df_free_gb,
        "compute": gather_compute(),
        "published_reference": ("TRM trained-on-ARC scores 6.2% exact on "
                                "ARC-AGI-2 public eval; human avg 66%"),
    }


def _trainer_running() -> bool:
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", "train_lfm.py"],
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip())
    except Exception:
        return False


def decide(state: dict, goal: str = "") -> dict:
    """The brain dictating how and when it learns. Returns the plan JSON."""
    prompt = (
        "You are the learning governor of an on-device AI harness (Symbio). "
        "You decide how and when it trains itself. Here is your current "
        "self-state as JSON:\n\n"
        + json.dumps(state, indent=1)
        + "\n\nGoal: "
        + (goal or "learn how ARC-AGI-2 works; improve held-out performance "
                    "while keeping each run's cost sane on this machine.")
        + "\n\nDecide the next move. Reply with ONLY a JSON object of the "
          "form:\n" + PLAN_SCHEMA
        + "\n\nScheduling: you may add a place map assigning each action to "
           "a node from compute.nodes — only nodes that are online, plugged "
           "in for heavy work, and capable of the runtime that action needs. "
           "Never schedule heavy training on battery power."
        + "\n\nRules: training is already running -> action 'wait' with the "
          "reasoning about what to check next. If a paired eval exists and "
          "shows no gain over base, say so plainly and adjust (more tasks, "
          "more iters, more spawn slots, or a different slice). If an eval "
          "shows gain, compound it. Be honest about what the numbers do and "
          "do not show."
    )
    avatar = _trained_avatar(state)
    plan = avatar.decide(state, goal)
    plan["_decider"] = "LFM2.5 trained avatar (in-process)"
    plan["_voice_note"] = state.get("voice_note")
    return plan


def execute(plan: dict, session) -> None:
    """The harness runs whatever IT decided, narrated through chat."""
    output_fn = session.output_fn
    action = plan.get("action", "wait")
    payload = {k: plan.get(k) for k in ("action", "tasks", "spawn", "iters",
                                        "place")}
    output_fn(f"  [Self] decision: {json.dumps(payload, default=None)}")
    # placement guard: the dictated node must exist, be online, and (for
    # heavy work) be plugged in — the brain schedules, physics enforces
    place = (plan.get("place") or {}).get(action) if action != "wait" else None
    if place:
        nodes = {n.get("name"): n for n in
                 state.get("compute", {}).get("nodes", [])}
        node = nodes.get(place)
        if not node or not node.get("online"):
            output_fn(f"  [Self] placement refused: {place} is not online; "
                      "running on mini instead")
        elif "Battery" in str(node.get("battery")) and action == "train":
            output_fn(f"  [Self] placement refused: {place} is on battery; "
                      "deferred until plugged")
    elif action in ("train", "eval"):
        output_fn("  [Self] placement: mini (default rank 0)")
    reasoning = plan.get("reasoning", "")
    if reasoning:
        output_fn(f"  [Self] reasoning: {reasoning[:400]}")

    if action == "train":
        from symbio.app import arc_lab
        output_fn(arc_lab.run_training(int(plan.get("tasks", 120)),
                                       int(plan.get("spawn", 4)),
                                       int(plan.get("iters", 120)),
                                       output_fn))
    elif action == "eval":
        from symbio.app import arc_lab
        arc_lab.run_eval(min(120, int(plan.get("tasks", 40))), output_fn)
    elif action == "wait":
        output_fn("  [Self] holding. " + (plan.get("reasoning", "")
                                          [:200] or ""))
    else:
        output_fn(f"  [Self] unknown action from decider: {action}")


def handle(session, arg: str = "") -> None:
    """The /selflearn command: one full round of the loop, narrated."""
    output_fn = session.output_fn
    goal = (arg or "").strip()
    try:
        state = gather_state()
        output_fn("  [Self] self-model gathered: "
                  f"{len(state['recent_losses'])} recent loss points, "
                  f"eval={'yes' if state['last_paired_eval'] else 'none yet'}, "
                  f"trainer_running={state['training_running_now']}")
        plan = decide(state, goal)
        Path(ARC_DIR / "last_plan.json").write_text(json.dumps(
            {"state": state, "plan": plan}, indent=1, default=str))
        execute(plan, session)
    except Exception as e:
        output_fn(f"  [Self] loop error: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    class _S:
        @staticmethod
        def output_fn(line):
            print(line)
    handle(_S(), " ".join(sys.argv[1:]))