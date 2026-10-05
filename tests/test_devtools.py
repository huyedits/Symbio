"""The run_tests tool: the half of self-repair that makes a fix verifiable.

The assistant can already read and edit its own source, and fine-tune behind a
user approval. What it could not do is run the project's own test suite —
python is denylisted in the sandbox, execute_code forbids subprocess, and the
generic timeout is 30 s against a suite that takes minutes. So every "fixed
it" was unverifiable, and the repair loop that is the point of the feature
could not close.

These tests hold the tool to its own contract: the verdict comes from pytest's
summary line, an unparsable run is a failure rather than a guess, and unknown
targets are refused rather than silently widening the run.
"""

import subprocess
from unittest.mock import patch

import pytest

from symbio.app import chat_tools, devtools


# ---- the parser: mechanical, and honest when it cannot read ----

def test_a_passing_summary_parses_green():
    out = "12 passed in 3.42s"
    parsed = devtools.parse_pytest_output(out)
    assert parsed["bad"] == 0
    assert parsed["counts"]["passed"] == 12


def test_a_failed_summary_names_its_tests():
    out = ("FAILED tests/test_x.py::test_a - AssertionError: boom\n"
           "FAILED tests/test_y.py::test_b - ValueError: nope\n"
           "2 failed, 30 passed in 8.01s")
    parsed = devtools.parse_pytest_output(out)
    assert parsed["bad"] == 2
    assert len(parsed["failing"]) == 2
    assert any("test_a" in f for f in parsed["failing"])


def test_errors_count_as_bad():
    out = "1 error in 0.42s"
    parsed = devtools.parse_pytest_output(out)
    assert parsed["bad"] >= 1


@pytest.mark.parametrize("garbage", [
    "",  # no output at all
    "Traceback (most recent call last):\n  ...boom",  # died before the summary
    "collecting 2755 items (this is not a summary line at all)",
])
def test_unparsable_output_is_not_graded_ok(garbage):
    """parse_pytest_output returns {} for anything without a summary line, and
    run_tests turns that into a FAILURE. An unparsable run must never read as
    green — the repair loop would conclude a broken state was fixed."""
    assert devtools.parse_pytest_output(garbage) == {}
    with patch.object(devtools.subprocess, "run",
                      return_value=FakeResult(0, garbage)):
        ok, report = devtools.run_tests([])
    assert ok is False
    assert "could not be parsed" in report


class FakeResult:
    def __init__(self, returncode, output):
        self.returncode = returncode
        self.stdout = output
        self.stderr = ""


# ---- the runner: right interpreter, right cwd, honest verdict ----

def test_run_tests_reports_green_suite_green():
    with patch.object(devtools.subprocess, "run",
                      return_value=FakeResult(0, "5 passed in 1.00s")) as run:
        ok, report = devtools.run_tests(["tests"])
    assert ok is True
    assert "5 passed" in report
    cmd = run.call_args.args[0]
    assert cmd[1:3] == ["-m", "pytest"]
    assert str(devtools._KNOWN_TARGETS["tests"]) in cmd


def test_exit_code_zero_but_failed_tests_is_a_failure():
    """pytest exit 3 does not happen here, but the verdict is computed from
    the summary AND the exit code together; a green summary beside a non-zero
    exit is a failure, not a win."""
    with patch.object(devtools.subprocess, "run",
                      return_value=FakeResult(2, "5 passed in 1.00s")):
        ok, report = devtools.run_tests(["tests"])
    assert ok is False


def test_unknown_target_is_refused_not_run():
    """The fallback for a typo'd target would run the WHOLE suite — minutes of
    budget on a call that could not have done what the model meant."""
    ok, report = devtools.run_tests(["test"])
    assert ok is False
    assert "Known targets" in report
    assert "tests" in report and "bench" in report


def test_a_timed_out_run_is_a_failure_named_as_one():
    with patch.object(devtools.subprocess, "run",
                      side_effect=subprocess.TimeoutExpired(cmd="pytest",
                                                            timeout=900)):
        ok, report = devtools.run_tests(["tests"])
    assert ok is False
    assert "timed out" in report.lower()


def test_targets_default_to_the_main_suite():
    with patch.object(devtools.subprocess, "run",
                      return_value=FakeResult(0, "5 passed in 1.00s")) as run:
        ok, _ = devtools.run_tests(None)
    assert ok
    assert str(devtools._KNOWN_TARGETS["tests"]) in run.call_args.args[0]


def test_string_target_is_accepted():
    """The model writes targets as a bare string often enough that refusing it
    would cost a round for nothing."""
    with patch.object(devtools.subprocess, "run",
                      return_value=FakeResult(0, "5 passed in 1.00s")):
        ok, _ = devtools.run_tests(["bench"])
    assert ok


# ---- the gates: how the tool is approved and catalogued ----

def test_run_tests_is_locally_trusted_and_scored_moderate():
    from symbio.app.chat_constants import _LOCAL_TRUSTED_TOOLS
    from symbio import safety

    assert "run_tests" in _LOCAL_TRUSTED_TOOLS
    assert "run_tests" not in _ALWAYS_CONFIRM
    risk = safety.assess_tool_risk("run_tests", {"targets": ["tests"]}, {})
    assert risk["risk_score"] == 1
    assert "runs_project_code" in risk["flags"]


from symbio.app.chat_constants import _ALWAYS_CONFIRM_TOOLS as _ALWAYS_CONFIRM  # noqa: E402


def test_run_tests_is_in_the_catalog_with_a_group():
    from symbio.app import tooling

    tooling.sync_tool_files(seed=False)
    schema = next(t for t in tooling._TOOLS if t["name"] == "run_tests")
    assert "pytest" in schema["description"].lower()
    assert tooling._TOOL_GROUPS.get("run_tests") == "code"
    assert tooling._TOOL_FAMILIES.get("run_tests") == "code"


def test_the_dispatcher_has_a_branch_for_run_tests():
    """A catalog entry with no dispatch branch is the orphan-tool failure seen
    with run_script: advertised to the model, then 'Unknown tool'."""
    import inspect

    src = inspect.getsource(chat_tools.ToolsMixin._dispatch_tool)
    assert 'name == "run_tests"' in src


def test_orphaned_tool_docs_are_gone():
    """The catalog must not advertise a tool the dispatcher refuses: the
    run_tests failure mode was described-but-'Unknown tool'. Main merged the
    scheduled-tasks scripts feature, so save_script/run_script et al. are
    real now; the check survives as the general contract — every tool file
    on disk maps to a name the dispatcher knows, from the catalog, from an
    MCP prefix, or from a tool docs entry."""
    from symbio.app import tool_docs, tooling
    import inspect

    names = {p["name"] for p in tool_docs.load_tool_files()}
    assert names, "the tools directory is the catalog's source of truth"
    dispatchable = {t["name"] for t in tooling._TOOLS}
    src = inspect.getsource(chat_tools.ToolsMixin._dispatch_tool)
    with_branch = {n for n in names if 'name == "%s"' % n in src}
    from_catalog = {n for n in names if n in dispatchable}
    unimplemented = names - with_branch - from_catalog
    assert not unimplemented, (
        f"advertised but not dispatchable: {sorted(unimplemented)}")