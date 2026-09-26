"""The fine-tune in progress, as one small file another process can watch.

A retrain's only live account was the trainer's own per-iteration lines, and
they went to a terminal or into the daemon's log. symbio_pet draws the run as
it happens — a droplet eaten per step, a wobble that follows the loss, the
adapter's core growing inside the slime — and it must not import this package
(symbio/__init__.py is ~105 MB of agent stack; see symbio_desktop/server.py).
So the account is kept on disk, in LOG_DIR/training_live.json, rewritten whole
through a rename so a reader never sees half of one.

Who writes what:

* _run_training opens a run (begin) just before the trainer starts, and
  closes it (end) once the outcome is known: trained, failed or stopped.
* Both trainer loops, early stop and plain, pass every line to line(), which
  keeps the train and validation losses it reports.
* The golden gate rules on a finished run through its two exits:
  restore_adapter marks it rolled back, discard_adapter_backup marks it kept.
  backup_adapter arms the gate first, so a run nobody is going to judge
  (`symb train`, a first worker adapter with nothing to roll back to) is kept
  the moment it finishes instead of waiting on a verdict that never comes.

Nothing here may fail a training run. Every write swallows its own errors: a
pet that cannot see the run is a cosmetic loss, a retrain that dies because
logs/ was read-only is not.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from symbio import constants

# How long an armed gate waits for its run. Every gated path calls
# backup_adapter right before the trainer starts; a snapshot taken for any
# other reason must not leave a later, unrelated run waiting on a verdict.
GATE_WINDOW_S = 600.0

# Points kept per curve. mlx_lm reports every 10 steps, so this holds a
# 2,400-step run whole; past that the oldest points go. The sparkline that
# draws them is about 130 pixels wide.
MAX_POINTS = 240

_NUMBER = r"([-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))"
# Case-insensitive: symbio/cuda_lora.py prints "train loss", mlx_lm "Train loss".
_TRAIN_RE = re.compile(r"Iter\s+(\d+):\s+Train\s+loss\s+" + _NUMBER, re.IGNORECASE)
_VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val\s+loss\s+" + _NUMBER, re.IGNORECASE)

_lock = threading.Lock()
# The run this process is writing. TRAINER_LOCK already allows one trainer
# per process, so one slot is enough.
_state: dict[str, Any] | None = None
# role -> when backup_adapter armed a gate for its next run.
_gate_armed: dict[str | None, float] = {}


def live_file() -> Path:
    """Resolved on every call, not bound at import: the suite points
    constants.LOG_DIR at a scratch directory, and a test run must not show up
    on the user's real pet."""
    return constants.LOG_DIR / constants.TRAINING_LIVE_NAME


def read() -> dict[str, Any] | None:
    try:
        state = json.loads(live_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def _write(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    path = live_file()
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(state), encoding="utf-8")
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def arm_gate(role: str | None = None) -> None:
    """A golden gate will judge `role`'s next run (called by backup_adapter)."""
    with _lock:
        _gate_armed[role] = time.time()


def begin(role: str | None, iters: int | None, prior_iters: int = 0) -> None:
    """A trainer is about to start.

    `prior_iters` is what the adapter already had when the run resumes it, and
    0 for a fresh run: LoRA's B matrices start at zero, so a new adapter
    begins as an exact no-op and the pet's core starts from nothing.
    """
    global _state
    now = time.time()
    with _lock:
        armed_at = _gate_armed.pop(role, None)
        # A run that starts while this process's last run still waits on its
        # gate is that gate's follow-up (the golden remedy), judged with it.
        # Seen live: the arm was spent on the first run, so the remedy ended
        # "kept" by itself and the gate then rolled both back.
        previous = _state
        follows = (previous["run"] if previous is not None and previous.get("gated")
                   and previous.get("phase") == "trained" and not previous.get("verdict")
                   and previous.get("role") == role else None)
        _state = {
            "version": 1,
            "run": f"{os.getpid()}-{now:.3f}",
            "role": role,
            "phase": "training",
            "gated": bool(follows) or (armed_at is not None and now - armed_at < GATE_WINDOW_S),
            "follows": follows,
            "pid": os.getpid(),
            "iters": int(iters) if iters else None,
            "prior_iters": max(0, int(prior_iters or 0)),
            "iter": 0,
            "train": [],
            "val": [],
            "started_at": now,
            "ended_at": None,
            "total_iters": None,
            "reason": "",
            "verdict": None,
            "verdict_at": None,
            "verdict_role": None,
        }
        _write(_state)


def line(text: str) -> None:
    """One line of trainer output. Keeps the losses it reports, if any."""
    with _lock:
        if _state is None or _state.get("phase") != "training":
            return
        changed = False
        for pattern, key in ((_TRAIN_RE, "train"), (_VAL_RE, "val")):
            match = pattern.search(text)
            if not match:
                continue
            step = int(match.group(1))
            try:
                loss = float(match.group(2))
            except ValueError:
                continue
            _state["iter"] = max(int(_state.get("iter") or 0), step)
            changed = True
            if not math.isfinite(loss):
                # Kept out of the curve (JSON has no nan) but not out of
                # sight: the early-stop monitor aborts on exactly this.
                _state["reason"] = f"{key} loss {loss} at iter {step}"
                continue
            curve = _state[key]
            curve.append([step, round(loss, 4)])
            del curve[:-MAX_POINTS]
        if changed:
            _write(_state)


def end(phase: str, reason: str = "", iter: int | None = None,
        total_iters: int | None = None) -> None:
    """The trainer is done. The first ending recorded wins, so the precise
    one ("stopped", from a Ctrl-C handler) is not overwritten by the generic
    "failed" _run_training records on its way out."""
    with _lock:
        if _state is None or _state.get("phase") != "training":
            return
        _state["phase"] = phase
        _state["ended_at"] = time.time()
        if reason:
            _state["reason"] = reason
        if iter is not None:
            _state["iter"] = int(iter)
        if total_iters is not None:
            _state["total_iters"] = int(total_iters)
        if phase == "trained" and not _state.get("gated"):
            # Nobody is going to judge this one; it stands as trained.
            _state.update(verdict="kept", verdict_at=_state["ended_at"],
                          verdict_role=_state.get("role"))
        _write(_state)


def _record_verdict(state: dict[str, Any], verdict: str, role: str | None) -> None:
    state.update(verdict=verdict, verdict_at=time.time(), verdict_role=role)
    _write(state)
    # Keep this process's copy in step, in case the same run is written again.
    if _state is not None and _state is not state and _state.get("run") == state.get("run"):
        _state.update(verdict=state["verdict"], verdict_at=state["verdict_at"],
                      verdict_role=role)


def mark_kept() -> None:
    """The gate kept the latest run (called by discard_adapter_backup).

    Only a finished, trained run that is still waiting can be kept: the same
    call runs in the `finally` of every gated flow, including the ones that
    failed or have already rolled back.
    """
    with _lock:
        state = read()
        if not state or state.get("phase") != "trained" or state.get("verdict"):
            return
        _record_verdict(state, "kept", state.get("role"))


def mark_rolled_back(role: str | None = None) -> None:
    """An adapter was put back from a backup (called by restore_adapter).

    Recorded whatever the latest run's phase: the golden gate's rollback, the
    rollback of a failed run, the crash recovery at start-up and a manual
    restore all put an older adapter back in place of the one that was there.
    """
    with _lock:
        _gate_armed.pop(role, None)
        state = read() or {"version": 1, "run": None, "role": role,
                           "phase": "restored"}
        _record_verdict(state, "rolled_back", role)
