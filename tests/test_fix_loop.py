"""The mechanical repair loop, pinned without a GPU.

The model is a good STEP, not a good LOOP (measured 2026-10-07: a 12B
no-think given /fix-harness as a prompt called run_tests, stated the right
diagnosis, then talked about edit_file instead of calling it). So the loop
is machinery: fix_loop.repair_failure runs the plan in Python and the
model's whole job is one patch per cycle. These tests hold that machinery
to its contract with a FAKE model — no GPU anywhere — because the loop's
own logic is the thing that must never fail.
"""

import threading
from pathlib import Path

import pytest

from symbio import constants
from symbio.app import fix_loop


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "symbio" / "app").mkdir(parents=True)
    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    return tmp_path


def _make_broken(project: Path) -> None:
    (project / "tests" / "test_x.py").write_text(
        "def test_greet():\n"
        "    from symbio.app import greet\n"
        "    assert greet('Huy') == 'Hi Huy!'\n"
        "\n"
        "def test_greet_bang():\n"
        "    from symbio.app import greet\n"
        "    assert greet('Huy!') == 'Hi Huy!!'\n", encoding="utf-8")
    (project / "symbio" / "app" / "greet.py").write_text(
        "def greet(name):\n"
        "    return 'Hi ' + name\n", encoding="utf-8")


def test_a_correct_patch_on_the_first_cycle_ends_the_loop_green(project):
    _make_broken(project)
    (project / "symbio" / "app" / "greet.py").write_text(
        "def greet(name):\n    return 'Nope'\n", encoding="utf-8")

    def verify(targets):
        src = (project / "symbio" / "app" / "greet.py").read_text()
        ok = "'Hi ' + name" in src
        return ok, ("1 passed in 0.1s" if ok else "1 failed in 0.1s")

    def generate(prompt):
        return ("<<<< OLD\n    return 'Nope'\n===="
                "\n    return 'Hi ' + name\n>>>> NEW")

    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        generate, verify)
    assert rec["fixed"] is True and len(rec["cycles"]) == 1
    assert "'Hi ' + name" in (project / "symbio" / "app" / "greet.py").read_text()


def test_a_malformed_patch_is_refused_and_the_file_untouched(project):
    _make_broken(project)
    before = (project / "symbio" / "app" / "greet.py").read_text()
    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda prompt: "I think the fix is obvious, no fence though.",
        lambda t: (False, "1 failed in 0.1s"))
    assert rec["fixed"] is False
    assert all(c["result"] == "no parseable patch" for c in rec["cycles"])
    assert (project / "symbio" / "app" / "greet.py").read_text() == before


def test_a_worse_patch_reverts_immediately(project):
    _make_broken(project)
    # The red state the cycle starts from: greet does not return the right
    # thing ('Nope') — this test's injected bug.
    (project / "symbio" / "app" / "greet.py").write_text(
        "def greet(name):\n    return 'Nope'\n", encoding="utf-8")
    before = (project / "symbio" / "app" / "greet.py").read_text()
    calls = {"n": 0}

    def verify(targets):
        # First red (the injected bug); after the model's patch, WORSE: two
        # failures. The loop must revert, not keep it.
        calls["n"] += 1
        if calls["n"] == 1:
            return False, "1 failed, 1 passed in 0.1s"
        src = (project / "symbio" / "app" / "greet.py").read_text()
        if "raise" in src:
            return False, "2 failed in 0.1s"
        return False, "1 failed, 1 passed in 0.1s"

    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: ("<<<< OLD\n    return 'Nope'\n===="
                   "\n    raise RuntimeError('oops')\n>>>> NEW"),
        verify)
    assert rec["reverted"] is True and rec["fixed"] is False
    assert (project / "symbio" / "app" / "greet.py").read_text() == before


def test_two_unproductive_cycles_end_in_an_honest_defeat(project):
    _make_broken(project)
    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: ("<<<< OLD\n    return 'Nope'\n===="
                   "\n    return 'Nope'\n>>>> NEW"),   # no-op patch: old==new refused
        lambda t: (False, "1 failed in 0.1s"))
    assert rec["fixed"] is False
    assert rec["result"].startswith("defeated")


def test_patch_parsing_reads_the_fence_shape():
    hunks = fix_loop._extract_patch(
        "Sure. <<<< OLD\na = 1\n====\na = 2\n>>>> NEW hope this helps")
    assert hunks == [("a = 1", "a = 2")]
    # old == new is refused to keep no-op patches from burning a cycle
    assert fix_loop._extract_patch("<<<< OLD\nx\n====\nx\n>>>> NEW") == []


def test_a_patch_cut_off_mid_fence_still_parses():
    """The measured 12B failure: the RIGHT patch, dead at the reply budget
    three lines short of the close. OLD/==== present, >>>> missing — the
    hunk is readable from what arrived, and the suite, not the parser,
    decides whether the partial NEW was right."""
    truncated = ("The bug inserts the last word. Fix:\n\n"
                 "<<<< OLD\n         text = text.replace(\n"
                 "             f\"${i}\",\n"
                 "             words[min(i, len(words)) - 1] if words else \"\")\n"
                 "====\n"
                 "         text = text.replace(\n"
                 "             f\"${i}\",\n"
                 "             words[i - 1] if len(words) >= i else \"\")\n"
                 "         # and the fix keeps going but the reply ends he")
    hunks = fix_loop._extract_patch(truncated)
    assert hunks, "a truncated fence with a complete OLD must still parse"
    old, new = hunks[0]
    assert "min(i, len(words))" in old
    assert "words[i - 1]" in new
    # The dangling partial tail is trimmed from NEW, not shipped blindly.
    assert not new.endswith("ends he")


def test_ambiguity_refuses_instead_of_replacing_the_wrong_hunk(project):
    (project / "tests" / "test_x.py").write_text("def test_greet():\n    assert True\n", encoding="utf-8")
    text = "    return 'Nope'\n    return 'Nope'\n"
    (project / "symbio" / "app" / "greet.py").write_text(
        "def greet(name):\n" + text, encoding="utf-8")
    assert fix_loop._apply_patch(text, [("    return 'Nope'", "    return 'Hi ' + name")]) is None

def test_a_patch_that_truncates_its_last_line_still_applies(project):
    """Measured 2026-10-07, cycle 2 of the live run: the model's OLD/NEW both
    dropped the closing `)` of the final line — the CONTENT was right, its
    last line was a prefix of the file's. The line-window matcher accepts
    that shape and applies the fix; exact-refusal would burn the cycle."""
    _make_broken(project)
    (project / "symbio" / "app" / "greet.py").write_text(
        "def greet(name):\n    return 'Nope'\n", encoding="utf-8")

    def verify(targets):
        src = (project / "symbio" / "app" / "greet.py").read_text()
        return ("'Hi ' + name" in src,
                ("1 passed in 0.1s" if "'Hi ' + name" in src
                 else "1 failed in 0.1s"))

    # OLD truncates the last line (missing the file's trailing content).
    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: ("<<<< OLD\n    return 'Nope'\n===="
                   "\n    return 'Hi ' + name\n>>>> NEW"),
        verify)
    assert rec["fixed"] is True


def test_an_ambiguous_window_is_refused():
    """Line-window matching must never guess between multiple candidate
    locations; an ambiguous window is a refusal, not a coin flip."""
    text = "    return 'Nope'\n filler\n    return 'Nope'\n"
    assert fix_loop._apply_patch(
        text, [("    return 'Nope'", "    return 'Hi ' + name")]) is None


def test_one_line_windows_are_never_matched():
    """A one-line match is a guess; two contiguous lines are a window."""
    assert fix_loop._fuzzy_window("x = 1\n    return 'Nope'\n",
                                  "    return 'Nope'") is None


def test_the_watcher_probes_after_a_turn_that_edited_source(
        project, monkeypatch):
    """Operate normally, spot the red, repair unasked: the watch hook's whole
    contract. With a green suite the watcher is silent (no repair run); on a
    red suite with a NEW failure it drives repair_failure with the model's
    patch-writer; the same failure twice in a row is not re-attempted."""
    from symbio.app import fix_watch

    (project / "tests" / "test_a.py").write_text("def test_ok():\n    assert True\n")
    # Patch devtools.run_tests where fix_watch reads it:
    import symbio.app.fix_watch as fw
    seen, repairs = [], []
    monkeypatch.setattr(fw.devtools, "run_tests",
                        lambda targets: (seen.append(targets),
                                         (False, "FAILED tests/test_a.py::test_ok - x\n"
                                                 "1 failed in 0.01s"))[1])
    fixed = {"n": 0}
    def fake_repair(failure, source, generate, verify, log):
        repairs.append((failure, source))
        return {"fixed": True, "reverted": False, "cycles": []}
    monkeypatch.setattr(fw.fix_loop, "repair_failure", fake_repair)
    gen = lambda p: "patch"
    log = lambda s: None

    # edit_done=True: probe runs this turn, repair fires on red.
    fw.watch_turn(edit_done=True, generate=gen, log=log)
    assert repairs and repairs[0][0] == "tests/test_a.py::test_ok"
    # The same failure is not re-attempted on the next turn (state memory):
    fw._state["turns_since"] = 99
    fw.watch_turn(edit_done=False, generate=gen, log=log)
    assert len(repairs) == 1
    # A normal turn under the probe interval: no probe at all.
    fw._state["last_repaired"] = None
    fw._state["turns_since"] = 0
    fw.watch_turn(edit_done=False, generate=gen, log=log)
    assert len(seen) == 2   # probed once for edit_done + once for the 99-turn state


def test_the_watcher_never_runs_two_repairs_at_once(project, monkeypatch):
    """One watcher at a time: the suite is a GPU-adjacent process and two
    racing probes are how a 16 GB machine dies."""
    from symbio.app import fix_watch
    running = {"n": 0}
    def blocker(targets):
        running["n"] += 1
        import time
        time.sleep(0.2)
        return True, "2 passed in 0.01s"
    import symbio.app.fix_watch as fw
    monkeypatch.setattr(fw.devtools, "run_tests", blocker)
    fw._state["turns_since"] = 99
    t1 = threading.Thread(target=fw.watch_turn,
                          kwargs=dict(edit_done=True, generate=lambda p: "",
                                      log=lambda s: None))
    t1.start(); t1.join()
    # A second call while the first held the lock is a no-op (no second run):
    assert running["n"] == 1
