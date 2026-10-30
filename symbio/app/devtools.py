"""Run the project's own test suite, and turn its output into something a
small model can act on.

This is the missing half of "the assistant repairs its own harness": it can
already read and edit its source (read_file / edit_file) and already fine-tune
on corrections (train_adapter, behind a user approval and a golden-set gate),
but a fix without a way to verify a fix is a guess with a backup file.

The terminal tool cannot be that verifier. python/python3 are on the sandbox
denylist (the wrapper scan would stop `python -m pytest` dead, or stop to ask),
`execute_code` forbids subprocess entirely, and the generic sandbox timeout is
30 s — this suite takes minutes. So run_tests runs pytest directly:

  venv/bin/python -m pytest <targets> -q

via subprocess with the project as cwd, no shell, a separate long timeout, and
a trimmed report. The venv interpreter is resolved once per call from
constants.PROJECT_DIR (venv/bin/python on macOS/Linux, venv/Scripts/python.exe
on Windows), so the sandbox cwd rules never apply: tests read the whole
project, which is the point, and are gated instead by the tool gates — group
"code" in tools.enabled_groups, the same risk scorer as every other tool, and
 LOCAL_TRUSTED (ask by name only on remote front-ends).

The parser is mechanical, not a judgement: the last "--- N failed / M passed"
line in pytest's short summary decides ok, the FAILED lines name the tests.
Anything the regexes cannot read is reported unparsed rather than guessed.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from symbio import constants

# Long on purpose, and deliberately NOT agent.sandbox_timeout (30 s): a full
# suite run of this project takes minutes. The tool's own ceiling keeps a hung
# collection from stalling a turn forever.
TEST_TIMEOUT_SECONDS = 900

# The report keeps the tail of pytest's output — the summary and the failures
# — not the whole wall of dots. A 2.7k-test run prints a hundred lines of
# progress dots that read as noise and push the failures out of the window.
_REPORT_CHARS = 6000

# Targets the tool accepts, against the two suites this project actually has.
_KNOWN_TARGETS = {
    "tests": Path("tests"),
    "bench": Path("bench/test_bench_harness.py"),
}


def _venv_python() -> Path:
    """The interpreter the project's own test suite runs under.

    A sibling venv is the environment pytest imports the project from; the
    running assistant's sys.executable is the same interpreter in the normal
    install (constants.PROJECT_DIR venv), and NOT the same when the app was
    launched from a system python — where the venv is still the right thing to
    use, because it is where pytest is installed.
    """
    candidates = [
        constants.PROJECT_DIR / "venv" / "bin" / "python",
        constants.PROJECT_DIR / "venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(sys.executable)


def _summary_line(output: str) -> str | None:
    """pytest's final counts line, e.g. '38 passed in 26.05s' — the truth."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        # `N failed, M passed`, `N passed`, `no tests ran`, `N errors`...
        if re.fullmatch(
                r"(?:\d+ \w+(?:, )?)+ in [\d.]+s(?: .*)?|no tests ran", line):
            return line
        # collection errors replace the summary with "ERROR ..."? Handled by
        # the caller: an unparsable tail is reported unparsed, never guessed.
    return None


_SUMMARY_ITEM_RE = re.compile(r"(\d+) (\w+)")
_FAILED_LINE_RE = re.compile(
    r"^(?:FAILED|ERROR) (?:tests/|bench/)?[\w./:-]+?::[\w.:-]+", re.MULTILINE)


def parse_pytest_output(output: str) -> dict[str, Any]:
    """Extract (ok, counts, failing test ids) from pytest -q output.

    Mechanical only: the summary line decides the verdict, and any summary
    carrying "failed", "error" or a non-zero exit with nothing parsed is a
    failure. Returns {}, not a guess, when the output cannot be read.
    """
    summary = _summary_line(output)
    if not summary:
        return {}
    counts: dict[str, int] = {}
    for n, word in _SUMMARY_ITEM_RE.findall(summary):
        counts[word] = int(n)
    failed = _FAILED_LINE_RE.findall(output)
    bad = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    return {"summary": summary, "counts": counts,
            "bad": bad, "failing": failed[:40]}


def trace_test(test_id: str, timeout: int = 300) -> str:
    """pytest's own traceback for ONE test id, short form, untruncated where
    it matters: the repair loop reads which source frame the failure surfaced
    in from here (the whole-suite report keeps only its tail). Returns "" when
    the id is not a test path or the run cannot start."""
    if not re.fullmatch(r"(?:tests|bench)/[\w./-]+\.py::[\w\[\].:-]+", test_id or ""):
        return ""
    cmd = [str(_venv_python()), "-m", "pytest", test_id, "-q", "--no-header",
           "--tb=short", "-p", "no:cacheprovider"]
    try:
        result = _run_pytest(cmd, timeout)
    except Exception:
        return ""
    return (result.stdout or "") + (result.stderr or "")


def locate_fault(test_id: str, timeout: int = 300,
                 top: int = 12) -> list[tuple[str, int, float]]:
    """Package lines ranked by how strongly they point at this failure:
    [(project-relative path, line, Ochiai score)], best first.

    Runs the failing test's whole FILE once under symbio.app.fault_trace, so
    the file's passing tests are the contrast: a line every passing test also
    runs is not much of a suspect. Ochiai = f / sqrt(F * (f + p)), with f and
    p the failing and passing tests that ran the line and F all failing tests.
    Ties keep the order the failing test first ran them in reverse — the
    deepest, latest line before the assert is the usual culprit.
    Returns [] when the run cannot start or nothing failed under tracing.
    """
    import json
    import math
    import tempfile

    if not re.fullmatch(r"(?:tests|bench)/[\w./-]+\.py::[\w\[\].:-]+", test_id or ""):
        return []
    test_file = test_id.split("::")[0]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "spectrum.json"
        cmd = [str(_venv_python()), "-m", "pytest", test_file, "-q", "--no-header",
               "--tb=no", "-p", "no:cacheprovider", "-p", "symbio.app.fault_trace"]
        try:
            _run_pytest(cmd, timeout, env={"SYMBIO_FAULT_OUT": str(out)})
            spectrum = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            return []
    failed = {n for n, r in spectrum.items() if r.get("failed")}
    if not failed:
        return []
    hits: dict[tuple[str, int], list[int]] = {}
    for node, r in spectrum.items():
        for rel, lines in r.get("lines", {}).items():
            for ln in lines:
                slot = hits.setdefault((rel, ln), [0, 0])
                slot[0 if node in failed else 1] += 1
    ranked = []
    for (rel, ln), (f, p) in hits.items():
        if f:
            ranked.append((rel, ln, round(f / math.sqrt(len(failed) * (f + p)), 4)))
    ranked.sort(key=lambda r: (-r[2], r[0], -r[1]))
    return ranked[:top]


def _run_pytest(cmd: list[str], timeout: int,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run pytest, and on timeout stop it the way Ctrl-C does before killing.

    subprocess.run's timeout sends SIGKILL. A killed pytest never reaches the
    `finally` of tests/test_utils.preserve_training_state, so the user's real
    adapters/ was left holding test junk and the backup orphaned as
    adapters.testbak.* (measured 2026-10-07: the HEADMASTER adapter and four
    skill adapters gone after one chat turn's suite run). SIGINT first lets
    pytest unwind its fixtures; SIGKILL only if it will not.
    """
    import os
    import signal

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=constants.PROJECT_DIR,
                            env={**os.environ, **env} if env else None)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def suite_running() -> bool:
    """True while another pytest run of this project holds the suite lock
    (tests/conftest.py takes it for the whole session). The suite's own
    process holds the lock and is not "another" run."""
    import os

    if os.environ.get("SYMBIO_SUITE_LOCK_PID") == str(os.getpid()):
        return False
    try:
        import fcntl
    except ImportError:          # Windows: no flock; no guard
        return False

    lock = constants.PROJECT_DIR / ".pytest-suite.lock"
    try:
        fd = open(lock, "a+")
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        fd.close()


def run_tests(targets: list[str] | None = None,
              max_output_chars: int = _REPORT_CHARS) -> tuple[bool, str]:
    """Run the project's pytest suite. Returns (ok, report-for-the-model).

    Unknown target names are refused — the fallback would otherwise run the
    WHOLE suite twice when the model meant the same run once, and the loop's
    budget is minutes, not seconds.
    """
    if targets:
        unknown = [t for t in targets if t not in _KNOWN_TARGETS]
        if unknown:
            known = ", ".join(sorted(_KNOWN_TARGETS))
            return False, (f"Unknown test target(s): {', '.join(unknown)}. "
                           f"Known targets: {known}.")
        paths = [_KNOWN_TARGETS[t] for t in targets]
    else:
        paths = [_KNOWN_TARGETS["tests"]]

    if suite_running():
        # Two suites at once race on the real adapters/ directory (each backs
        # it up, writes test adapters into it, restores) — the second run's
        # "backup" is the first run's junk. Refuse rather than queue: the
        # caller's verdict would be minutes stale anyway.
        return False, ("Another test run of this project is in progress; "
                       "wait for it to finish, then run the tests again.")
    cmd = [str(_venv_python()), "-m", "pytest", *[str(p) for p in paths],
           "-q", "--no-header", "-p", "no:cacheprovider"]
    try:
        result = _run_pytest(cmd, TEST_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return False, f"Test run timed out after {TEST_TIMEOUT_SECONDS}s."
    except FileNotFoundError:
        return False, ("pytest could not run: the project venv has no "
                       "interpreter at venv/bin/python.")
    except Exception as e:
        return False, f"Test run failed to start: {e}"

    output = (result.stdout or "") + (result.stderr or "")
    parsed = parse_pytest_output(output)
    if not parsed:
        # Never grade an unparsable run as ok. Exit code plus the raw tail,
        # so the model sees what pytest actually printed.
        return False, (f"Test run exited {result.returncode} but the output "
                       f"could not be parsed:\n{output[-max_output_chars:]}")

    ok = parsed["bad"] == 0 and result.returncode == 0
    if len(output) > max_output_chars:
        output = output[-max_output_chars:]
    header = (f"pytest targets={[str(p) for p in paths]} exit={result.returncode}"
              f" — {parsed['summary']}")
    report = f"{header}\n{output}".strip()
    if parsed["failing"]:
        # The failing ids again, on top: a wall of pytest output and a small
        # context budget are how the ids get lost.
        report += "\nFailing: " + ", ".join(parsed["failing"])
    return ok, report