"""Which lines did each test run? A pytest plugin, loaded only on request.

The repair loop's hardest step is not the patch, it is WHERE: a wrong value
fails in the test's own assert, the traceback names only the test file, and
the code that produced the value can be any callee of any callee. Guessing
the file from the test's imports handed the model the wrong file on 2 of the
first 4 injected bugs (2026-10-08), and no model can patch a line it was
never shown.

Spectrum-based fault localisation answers it mechanically: run the failing
test's file once with line tracing, and a line executed by the failing tests
but rarely by the passing ones is the likely culprit (Ochiai score). This
module is the tracing half; devtools.locate_fault runs it and ranks.

    pytest -p symbio.app.fault_trace tests/test_x.py   # SYMBIO_FAULT_OUT=path.json

Only files under the project's symbio/ package are recorded: the suite's own
lines and the standard library are noise for a repair that may only touch
the package.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

_ROOT = str((Path(__file__).resolve().parent.parent)) + os.sep   # .../symbio/
_PROJECT = Path(__file__).resolve().parent.parent.parent
_SELF = str(Path(__file__).resolve())

_results: dict[str, dict] = {}
_current: dict[str, set[int]] | None = None


def _tracer(frame, event, arg):
    filename = frame.f_code.co_filename
    if not filename.startswith(_ROOT) or filename == _SELF:
        return None             # do not trace into this frame's lines
    if frame.f_code.co_name == "<module>":
        # A module body runs once, under whichever test imported it first;
        # charging those lines to that test made a whole unrelated module
        # the top suspect whenever the importer happened to fail (measured
        # 2026-10-08: two fix_loop.py bugs located to chat_tools.py).
        return None
    if event in ("call", "line") and _current is not None:
        _current.setdefault(filename, set()).add(frame.f_lineno)
    return _tracer


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _current
    _current = {}
    outcome = {"failed": False}
    item._fault_trace_outcome = outcome
    sys.settrace(_tracer)
    threading.settrace(_tracer)
    try:
        yield
    finally:
        sys.settrace(None)
        threading.settrace(None)
        lines = {}
        for f, ls in _current.items():
            try:
                rel = str(Path(f).resolve().relative_to(_PROJECT))
            except ValueError:
                continue
            lines[rel] = sorted(ls)
        _results[item.nodeid] = {"failed": outcome["failed"], "lines": lines}
        _current = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    rep = (yield).get_result()
    if rep.failed and hasattr(item, "_fault_trace_outcome"):
        item._fault_trace_outcome["failed"] = True


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("SYMBIO_FAULT_OUT")
    if out:
        Path(out).write_text(json.dumps(_results), encoding="utf-8")
