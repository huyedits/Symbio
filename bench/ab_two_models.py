"""A/B: Gemma-12B-coder vs FableVibes-14B-A3B on the same three probes.

Each model gets, fresh-loaded, single resident, exact measurements printed:

  1. arithmetic       'What is 17 times 23? Just the number.'  — the number
     from this week's logs, both models' speed floor
  2. tool turn        real ChatSession tool loop — 'count notes' style
  3. learn turn       the correction capture (does the machinery fire identically?)

Run twice with SYMBIO_A_MODEL / SYMBIO_B_MODEL env, results keyed to bench/.
Nothing here trusts a claim: every number is read off the run.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = os.environ.get("SYMBIO_A_MODEL") or os.environ.get("SYMBIO_B_MODEL", "")
TAG = os.environ.get("SYMBIO_TAG", "a")

from symbio.app.config import load_config          # noqa: E402
from symbio.app.chat import ChatSession            # noqa: E402


def main() -> int:
    cfg = load_config()
    cfg["model_name"] = MODEL
    cfg["memory"]["auto_compact_enabled"] = False   # measurement hygiene
    results = {}

    # --- 1. arithmetic, cold, greedy via sampler the session already uses---
    s = ChatSession(cfg)
    t0 = time.perf_counter()
    out = s._agent_turn("What is 17 times 23? Just the number.")
    results["arith_s"] = round(time.perf_counter() - t0, 1)
    # _agent_turn returns None — the reply lives in history[-1] (the final
    # assistant turn), which is the session's own convention.
    def last_reply() -> str:
        import json as _json
        for t in reversed(s.history):
            if t.get("role") == "assistant":
                return str(t.get("content", ""))[-140:]
        return ""

    results["arith_reply"] = last_reply()

    # --- 2. tool turn ---
    t0 = time.perf_counter()
    s._agent_turn("How many notes are in my notes directory? Use a tool. Short answer.")
    results["tool_s"] = round(time.perf_counter() - t0, 1)
    results["tool_reply"] = last_reply()

    # --- 3. correction capture fires? ---
    s._agent_turn("no wait, the deployment password is TESTPASS-4242.")
    results["correction_reply"] = last_reply()
    from symbio.app import learn
    results["mistake_notes_pending"] = learn.mistake_note_count()

    print(f"\n=== A/B {TAG}: {MODEL}")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())