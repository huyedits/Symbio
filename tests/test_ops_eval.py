"""Grading ops answers by running them.

Built after a string-matching rubric passed `sed -i 's/old/new/g' config.txt`
— which on BSD userland reads the expression as a backup suffix, errors, and
leaves the file untouched. It contains sed, names the right file, has the right
substitution, and does nothing. Only execution catches that.
"""
import sys

import pytest

from symbio.app import ops_eval


# ---- the gate ----

# Tasks whose reference solution or check is macOS userland on purpose (BSD
# sed wants `-i ''`, which GNU sed reads as the script; BSD `stat -f '%Lp'`
# is GNU's `stat -c`). The eval grades the shell the agent actually runs in,
# which is a Mac's; off one they cannot pass.
_BSD_ONLY = {"config_edit_in_place", "restrict_permissions"}


def test_every_shipped_task_passes_the_gate():
    """Each task's check must FAIL on the untouched state and PASS after the
    reference solution. A task that cannot do both grades nothing."""
    for task in ops_eval.TASKS:
        if task["id"] in _BSD_ONLY and sys.platform != "darwin":
            continue
        ok, why = ops_eval.validate(task)
        assert ok, f"{task['id']}: {why}"


def test_a_self_fulfilling_check_is_rejected():
    """The failure that taught this. A model-authored check ran
    `kill -9 $(pgrep sleep)` itself, so it passed whatever answer it was
    given."""
    ok, why = ops_eval.validate({
        "setup": "touch marker",
        "task": "delete marker",
        "solution": "rm marker",
        "check": "rm -f marker; test ! -e marker",   # does the work itself
    })

    assert not ok
    assert "grades nothing" in why


def test_an_unsatisfiable_check_is_rejected():
    """Two authored tasks could never pass at all, so both arms failed and the
    scoreboard read like a model failure when it was an authoring failure."""
    ok, why = ops_eval.validate({
        "setup": "touch a",
        "task": "do the impossible",
        "solution": "true",
        "check": "test -e never_created_by_anything",
    })

    assert not ok
    assert "never passes" in why


def test_a_task_needing_privileges_is_rejected():
    ok, why = ops_eval.validate({
        "setup": "sudo useradd bob", "task": "x", "solution": "true", "check": "false"})

    assert not ok
    assert "outside the sandbox" in why


# ---- grading ----

def test_a_working_answer_passes():
    task = next(t for t in ops_eval.TASKS if t["id"] == "count_errors_in_log")

    ok, why = ops_eval.grade(task, "```bash\ngrep -c ERROR app.log > error_count.txt\n```")

    assert ok, why


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="the task grades BSD sed; GNU sed inverts it")
def test_the_bsd_trap_is_caught_by_running_it():
    """The whole reason this file exists: the GNU form looks right and changes
    nothing on this machine."""
    task = next(t for t in ops_eval.TASKS if t["id"] == "config_edit_in_place")

    gnu, _ = ops_eval.grade(
        task, "```bash\nsed -i 's/old.example.com/new.example.com/' app.conf\n```")
    bsd, _ = ops_eval.grade(
        task, "```bash\nsed -i '' 's/old.example.com/new.example.com/' app.conf\n```")

    assert gnu is False
    assert bsd is True


def test_an_answer_with_no_command_is_not_a_pass():
    task = next(t for t in ops_eval.TASKS if t["id"] == "count_errors_in_log")

    ok, why = ops_eval.grade(task, "You could use grep for this.")

    assert not ok
    assert "no runnable command" in why


def test_an_answer_reaching_outside_the_sandbox_is_refused():
    task = next(t for t in ops_eval.TASKS if t["id"] == "count_errors_in_log")

    ok, why = ops_eval.grade(task, "```bash\nsudo rm -rf /\n```")

    assert not ok
    assert "outside the sandbox" in why


def test_grading_leaves_nothing_behind(tmp_path):
    """Each run gets a fresh sandbox that is removed afterwards, so one task
    cannot see another's files."""
    import tempfile
    task = next(t for t in ops_eval.TASKS if t["id"] == "archive_then_verify")
    before = set(p.name for p in __import__("pathlib").Path(tempfile.gettempdir()).glob("opsrun-*"))

    ops_eval.grade(task, "```bash\ntar -czf site.tar.gz site\n```")

    after = set(p.name for p in __import__("pathlib").Path(tempfile.gettempdir()).glob("opsrun-*"))
    assert after == before


# ---- the model marks first, the shell has the final say ----

def test_the_shell_decides_and_the_self_mark_sits_beside_it():
    """Recorded, not averaged in: a disagreement is the interesting part and
    an aggregate would hide it."""
    task = next(t for t in ops_eval.TASKS if t["id"] == "count_errors_in_log")
    good = "```bash\ngrep -c ERROR app.log > error_count.txt\n```"

    result = ops_eval.assess(task, good, generate_fn=lambda _p: "FAIL")

    assert result["passed"] is True          # the shell ran it
    assert result["self_mark"] is False      # the model doubted it
    assert result["agreed"] is False         # and that is visible


def test_no_marker_means_no_opinion_not_a_failure():
    task = next(t for t in ops_eval.TASKS if t["id"] == "count_errors_in_log")

    result = ops_eval.assess(task, "```bash\ngrep -c ERROR app.log > error_count.txt\n```")

    assert result["passed"] is True
    assert result["self_mark"] is None
    assert result["agreed"] is None


def test_a_marker_that_raises_does_not_take_the_grade_with_it():
    """The shell verdict is the one that matters; a broken marker must not
    turn a graded answer into no answer."""
    task = next(t for t in ops_eval.TASKS if t["id"] == "count_errors_in_log")

    def explode(_prompt):
        raise RuntimeError("model unloaded")

    result = ops_eval.assess(task, "```bash\ngrep -c ERROR app.log > error_count.txt\n```",
                             generate_fn=explode)

    assert result["passed"] is True
    assert result["self_mark"] is None


def test_the_marker_sees_the_commands_not_the_prose():
    """It judges what would run, so it is shown what would run."""
    task = next(t for t in ops_eval.TASKS if t["id"] == "count_errors_in_log")
    seen = {}

    def capture(prompt):
        seen["prompt"] = prompt
        return "PASS"

    ops_eval.assess(task, "Sure, here you go:\n```bash\ngrep -c ERROR app.log\n```",
                    generate_fn=capture)

    assert "grep -c ERROR app.log" in seen["prompt"]
    assert "Sure, here you go" not in seen["prompt"]
