"""Laya: a small typed-decision model for cheap YES/NO and choice judgments.

Laya (convaiinnovations/laya) is a 421M ModernBERT-plus-head classifier that
answers typed questions in one forward pass — ~40-260ms on the Apple GPU —
instead of the multi-second generation the 14B headmaster spends on the same
judgment. It never generates text: options are scored at  positions and
softmaxed, so an answer is a probability distribution over options the caller
names, with nothing to parse and nothing to hallucinate.

What it is NOT: a general brain. Measured on this machine (2026-09-26), the
shipped `typed-decisions` checkpoint is strong on its home workflows (billing
triage 0.81, urgency 0.75) but near chance on Symbio's own tool-and-safety
judgments — margins like 0.37 vs 0.21. So every call site treats its answer
as a PRIOR under a confidence floor, never a verdict: below the floor, or on
any failure, the caller falls back to whatever it did before this existed.

RAM: ~1.2GB resident while loaded, beside the daemon's ~9GB. That is too much
to hold idle on 16GB, so the module loads on first use, answers, and frees —
the whole point is that a 40ms judgment does not have to be a resident
feature. Two calls in a row reload twice; batching beats residency here.

The checkpoint resolves from the local HF cache first (HF_HOME/hub), then the
hub. `SYMBIO_NO_LAYA=1` turns the whole module off.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Below this confidence the answer is not the model's to make. Measured
# margins on real judgments ran 0.37 vs 0.21 — real signal, small margin —
# while a wrong "general" label is cheap and a wrong confident one is not.
CONFIDENCE_FLOOR = 0.40

_LOCK: Any = None
_AGENT: Any = None
_LOCAL_SNAPSHOT: Path | None = None


def _lazy_imports():
    global _LOCK, Path
    import threading

    if _LOCK is None:
        _LOCK = threading.Lock()
    return _LOCK


def _checkpoint() -> str:
    """The laya-typed-decisions checkpoint: local snapshot when cached."""
    global _LOCAL_SNAPSHOT
    if _LOCAL_SNAPSHOT is not None:
        return str(_LOCAL_SNAPSHOT)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        snap = (Path(hf_home) / "hub"
                / "models--convaiinnovations--laya" / "snapshots")
        if snap.is_dir():
            for child in sorted(snap.iterdir()):
                if (child / "typed-decisions" / "model.safetensors").is_file():
                    _LOCAL_SNAPSHOT = child
                    return str(child)
    _LOCAL_SNAPSHOT = Path("convaiinnovations/laya")
    return str(_LOCAL_SNAPSHOT)


def available() -> bool:
    """Cheap gate: off-switch, the package, and the checkpoint on disk.

    Does not load weights — a caller can offer the fast path without paying
    for it."""
    if os.environ.get("SYMBIO_NO_LAYA") == "1":
        return False
    try:
        import laya  # noqa: F401
    except Exception:
        return False
    path = _checkpoint()
    return path.endswith("/typed-decisions/model.safetensors") or "convaiinnovations" in path


def _agent():
    """The loaded agent, one at a time. Caller frees it — see forget()."""
    _lazy_imports()
    global _AGENT
    with _LOCK:
        if _AGENT is None:
            os.environ.setdefault("USE_TF", "0")
            import laya

            _AGENT = laya.load(_checkpoint(), subfolder="typed-decisions")
        return _AGENT


def forget() -> None:
    """Release the ~1.2GB a loaded checkpoint holds.

    Called by every wrapper after its decision: a router that stays resident
    is a second model on a 16GB machine, which is the residency this module
    exists to avoid."""
    _lazy_imports()
    global _AGENT
    with _LOCK:
        if _AGENT is None:
            return
        try:
            unload = getattr(_AGENT, "unload", None)
            if callable(unload):
                unload()
        except Exception:
            pass
        _AGENT = None


def decide(state: str, question: str, criteria: dict[str, str],
           floor: float | None = None) -> dict | None:
    """One typed choice, or None when the model will not commit.

    `criteria` maps option names to one-line descriptions. Returns
    {"choice", "probabilities", "confidence"} — confidence is the winning
    option's share, the honest number to floor on — or None when laya is
    off, fails, or lands under the floor. Never raises. A None floor reads
    the module-level CONFIDENCE_FLOOR at call time, so tests and callers can
    retune the dial without re-binding a default."""
    if floor is None:
        floor = CONFIDENCE_FLOOR
    if not criteria:
        return None
    try:
        agent = _agent()
        answer = agent.predict(state, {
            "q": {"type": "choice", "instructions": question,
                  "criteria": criteria},
        })["answers"]["q"]
        probs = {str(k): float(v) for k, v in
                 (answer.get("probabilities") or {}).items()}
        choice = str(answer.get("choice", ""))
        confidence = float(answer.get("answer_confidence")
                           or probs.get(choice, 0.0) or 0.0)
        if not choice or choice not in criteria or confidence < floor:
            return None
        return {"choice": choice, "probabilities": probs,
                "confidence": confidence}
    except Exception:
        return None
    finally:
        forget()


def yes_no(state: str, question: str,
           floor: float | None = None) -> bool | None:
    """One yes/no judgment as a two-option choice with neutral keys.

    `noul` is avoided on purpose: the shipped English checkpoint follows its
    own false:/true: option labels on noul questions (laya#156), returning a
    confident "no" for clearly positive input. Two neutral criteria keyed by
    meaning are the documented workaround."""
    if floor is None:
        floor = CONFIDENCE_FLOOR
    result = decide(state, question, {
        "yes": "yes",
        "no": "no",
    }, floor=floor)
    return None if result is None else result["choice"] == "yes"