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

    cmd = [str(_venv_python()), "-m", "pytest", *[str(p) for p in paths],
           "-q", "--no-header", "-p", "no:cacheprovider"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TEST_TIMEOUT_SECONDS, cwd=constants.PROJECT_DIR)
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