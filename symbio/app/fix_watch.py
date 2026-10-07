"""The harness self-check: operate normally; notice when the suite goes
red; repair without being told.

This is the "spot it" half of the self-repair claim (measured companion to
fix_loop, which repairs a failure it is POINTED at). A turn that ends
normally is followed — on the background thread, never in the user's way —
by a suite probe. When the probe reads red and the top failure is new (not
the one this watcher just tried), repair_failure runs with the session's
own generate as the patch-writer. The user sees one line per step, or
nothing at all when the suite stays green.

Discipline inherited from the measured failures this week:
- the probe targets ONLY test files (the fast slice) on normal turns, and
  the full suite when a code edit happened during the turn;
- repairs are bounded (fix_loop's own cycle budget) and never re-enter
  while a previous repair is running (a lock);
- the watcher's verdict is the suite's, never the model's.
"""

import threading

from symbio import constants
from symbio.app import devtools, fix_loop

# One watcher at a time — the suite is a GPU-adjacent process and two would
# race the model for the machine.
_LOCK = threading.Lock()

# Do not re-probe on every turn: the suite takes ~3 minutes, and a chat that
# is doing nothing to code does not need it each time. N = turns.
_TURNS_BETWEEN_PROBES = 4

_state = {"turns_since": 0, "last_repaired": None}


def _top_failure(report: str) -> str | None:
    for line in (report or "").splitlines():
        if line.startswith("FAILED"):
            return line.split(" - ")[0].replace("FAILED ", "").strip()
    return None


def _slice_source_for(test_id: str) -> str:
    """Which source file a failing test most plausibly repaired — the test
    module's sibling-under-symbio mapping is not derivable in general, so
    the watcher takes it from the FAILURE's import path in the report, or
    falls back to the test's own module name."""
    import re

    # "FAILED tests/test_custom_commands.py::test_x" → the file the test
    # lives in maps to a same-named module under symbio/ when one exists.
    m = re.match(r"(?:FAILED\s+)?([\w./-]+\.py)::", test_id)
    if not m:
        return "symbio/app/chat.py"   # the loop's home; a sane default target
    leaf = m.group(1).rsplit("/", 1)[-1]          # test_custom_commands.py
    stem = leaf.removeprefix("test_")             # custom_commands.py
    for candidate in (f"symbio/app/{stem}", f"symbio/{stem}",
                      f"symbio/app/{stem.replace('.py', '.py')}"):
        if (constants.PROJECT_DIR / candidate).exists():
            return candidate
    return "symbio/app/chat.py"


def watch_turn(edit_done: bool = False, generate=None,
               log=None) -> None:
    """Post-turn hook: maybe probe the suite; maybe repair. Never blocks
    the interactive turn's caller when called on a background thread.

    `generate(prompt) -> reply` is the model's one-patch-per-cycle writer
    (ChatSession._fix_generate). `log(line)` is where the watcher's one-line
    announcements go."""
    if generate is None:
        return
    say = log or (lambda s: None)
    if not _LOCK.acquire(blocking=False):
        return
    try:
        _state["turns_since"] = (0 if edit_done else
                                 _state["turns_since"] + 1)
        if not edit_done and _state["turns_since"] < _TURNS_BETWEEN_PROBES:
            return
        ok, report = devtools.run_tests(["tests", "bench"])
        if ok:
            return
        failure = _top_failure(report)
        if not failure or failure == _state.get("last_repaired"):
            return
        say(f"  [Watch] The suite is red ({failure}) — repairing without "
            f"being told.")
        source = _slice_source_for(failure)
        _state["last_repaired"] = failure
        rec = fix_loop.repair_failure(
            failure, source, generate=generate,
            verify=lambda _t: devtools.run_tests(["tests"]), log=say)
        verdict = "fixed" if rec["fixed"] else (
            "reverted" if rec["reverted"] else "could not fix — the "
            "failure needs a designer, not a patch")
        say(f"  [Watch] {failure}: {verdict}.")
    finally:
        _LOCK.release()