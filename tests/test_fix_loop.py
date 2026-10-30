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
    monkeypatch.setattr(fw.devtools, "trace_test", lambda test_id: "")
    monkeypatch.setattr(fw.devtools, "locate_fault", lambda test_id: [])
    def fake_repair(failure, source, generate, verify, log, trace=None,
                    suspects=None):
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


# ---- the inputs the model is handed: right file, right lines ----

_REPORT_WITH_TRACE = """\
pytest targets=['tests'] exit=1 — 2 failed, 40 passed in 3.00s
_________________________ test_other _________________________
symbio/app/elsewhere.py:9: in other
    raise ValueError
E   ValueError
_________________________ test_render_fills_slots _________________________
tests/test_x.py:4: in test_render_fills_slots
    assert render("a b") == "a b"
symbio/app/commands.py:12: in render
    return helper(text)
symbio/app/helpers.py:3: in helper
    raise KeyError(text)
E   KeyError: 'a b'
=========================== short test summary info ===========================
FAILED tests/test_x.py::test_other - ValueError
FAILED tests/test_x.py::test_render_fills_slots - KeyError
Failing: FAILED tests/test_x.py::test_other, FAILED tests/test_x.py::test_render_fills_slots
"""


def test_the_source_comes_from_the_failing_tests_own_traceback(project):
    """The watcher used to map test_custom_commands.py to chat.py by name;
    the traceback names the real file. Deepest project frame of THIS test's
    section wins — not a frame from a neighbouring failure."""
    for rel in ("symbio/app/commands.py", "symbio/app/helpers.py",
                "symbio/app/elsewhere.py"):
        (project / rel).write_text("x = 1\n")
    assert fix_loop.source_from_report(
        _REPORT_WITH_TRACE, "tests/test_x.py::test_render_fills_slots") == (
        "symbio/app/helpers.py", 3)
    from symbio.app import fix_watch
    assert fix_watch._slice_source_for(
        "tests/test_x.py::test_render_fills_slots",
        _REPORT_WITH_TRACE) == "symbio/app/helpers.py"


def test_a_long_source_is_sliced_around_the_failing_line(project):
    """The test's name never appears in the source; slicing by it handed the
    model the top of the file. The traceback's line is the anchor."""
    _make_broken(project)
    body = "".join(f"x{i} = {i}\n" for i in range(400))
    body += "def greet(name):\n    return 'BUG HERE'\n"
    (project / "symbio" / "app" / "greet.py").write_text(body)
    out = fix_loop._slice_for_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py", source_line=402)
    assert "BUG HERE" in out["source"]
    assert "x0 = 0" not in out["source"]


def test_a_repair_is_judged_by_its_own_test_not_by_unrelated_red(project):
    """A suite with a failure the patch never touched (a test that needs
    mlx-vlm on this box) used to make every repair a defeat."""
    _make_broken(project)
    unrelated = "FAILED tests/test_vlm.py::test_needs_vlm - ImportError"

    def verify(targets):
        src = (project / "symbio" / "app" / "greet.py").read_text()
        if "'Hi ' + name" in src:
            return False, f"1 failed, 9 passed in 0.1s\n{unrelated}"
        return False, ("2 failed, 8 passed in 0.1s\n"
                       f"{unrelated}\nFAILED tests/test_x.py::test_greet - x")

    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: ("<<<< OLD\n    return 'Hi ' + name\n===="
                   "\n    return 'Hi ' + name + '!'\n>>>> NEW"),
        verify)
    # The patched file still holds 'Hi ' + name, so the target passes and
    # only the unrelated failure is left.
    assert rec["fixed"] is True
    assert not list((project / "symbio" / "app").glob("*.repairbak"))


def test_a_patch_that_will_not_parse_never_touches_the_file(project):
    _make_broken(project)
    before = (project / "symbio" / "app" / "greet.py").read_text()
    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: ("<<<< OLD\n    return 'Hi ' + name\n===="
                   "\n    return 'Hi ' + (name\n>>>> NEW"),
        lambda t: (False, "1 failed in 0.1s"))
    assert rec["fixed"] is False
    assert rec["cycles"][0]["result"].startswith("refused: patch does not parse")
    assert (project / "symbio" / "app" / "greet.py").read_text() == before


def test_no_backup_is_left_behind_after_a_two_cycle_fix(project):
    """A fix that landed on cycle 2 used to leave `<file>.1.repairbak` in the
    tree (only the last cycle's backup was removed)."""
    _make_broken(project)
    (project / "symbio" / "app" / "greet.py").write_text(
        "def greet(name):\n    return 'A'\n")
    replies = iter([
        "<<<< OLD\n    return 'A'\n====\n    return 'B'\n>>>> NEW",
        "<<<< OLD\n    return 'B'\n====\n    return 'Hi ' + name\n>>>> NEW",
    ])

    def verify(targets):
        ok = "'Hi ' + name" in (project / "symbio" / "app" / "greet.py").read_text()
        return ok, "1 passed in 0.1s" if ok else "1 failed in 0.1s"

    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: next(replies), verify)
    assert rec["fixed"] is True and len(rec["cycles"]) == 2
    assert not list((project / "symbio" / "app").glob("*.repairbak"))


def test_an_unfixed_patch_left_in_place_keeps_one_named_backup(project):
    _make_broken(project)
    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: ("<<<< OLD\n    return 'Hi ' + name\n===="
                   "\n    return 'Hello ' + name\n>>>> NEW"),
        lambda t: (False, "1 failed in 0.1s"), budget_cycles=1)
    assert rec["fixed"] is False
    assert rec["backup"] == "symbio/app/greet.py.repairbak"
    assert "'Hi ' + name" in (project / rec["backup"]).read_text()


def test_a_bare_close_fence_parses():
    """Gemma-12B closed its patch with `>>>>` and no NEW (measured live)."""
    assert fix_loop._extract_patch(
        "<<<< OLD\na = 1\n====\na = 2\n>>>>\n") == [("a = 1", "a = 2")]


def test_a_whole_function_quote_lands_as_the_one_line_it_changes():
    """The live miss, three cycles running: the right one-line fix quoted
    inside a whole function, one blank line short of the file. Only the
    changed line is applied; the file's own blank lines stay."""
    text = ("def render(x):\n"
            "    y = x\n"
            "    return y + 1\n"
            "\n"
            "\n"
            "def suggest():\n"
            "    pass\n")
    old = ("def render(x):\n    y = x\n    return y + 1\n\n"
           "def suggest():\n    pass")
    new = ("def render(x):\n    y = x\n    return y\n\n"
           "def suggest():\n    pass")
    out = fix_loop._apply_patch(text, [(old, new)])
    assert out == text.replace("return y + 1", "return y")


def test_a_rejected_patch_is_shown_back_so_the_next_cycle_differs(project):
    """Greedy decoding answers the same prompt with the same patch. Measured
    2026-10-08: the 14B sent one identical no-op patch three cycles running.
    The second prompt must carry what the first cycle tried."""
    _make_broken(project)
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return "<<<< OLD\n    return 'Hi ' + name\n====\n    return 'Hi ' + name\n>>>> NEW"

    fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py", generate,
        lambda t: (False, "1 failed in 0.1s"), budget_cycles=2)
    assert len(prompts) == 2
    assert "Already tried" not in prompts[0]
    assert "Already tried" in prompts[1]


def test_a_worse_patch_is_reverted_and_the_next_cycle_still_runs(project):
    _make_broken(project)
    greet = project / "symbio" / "app" / "greet.py"
    replies = iter([
        "<<<< OLD\n    return 'Hi ' + name\n====\n    raise RuntimeError\n>>>> NEW",
        "<<<< OLD\n    return 'Hi ' + name\n====\n    return 'Hi ' + name + '!'\n>>>> NEW",
    ])
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return next(replies)

    def verify(_t):
        src = greet.read_text()
        if "raise" in src:
            return False, ("FAILED tests/test_x.py::test_greet\n"
                           "FAILED tests/test_x.py::test_greet_bang")
        if "+ '!'" in src:
            return True, "2 passed in 0.1s"
        return False, "FAILED tests/test_x.py::test_greet"

    rec = fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py", generate, verify,
        budget_cycles=3)
    assert rec["fixed"] is True and rec["reverted"] is True
    assert "broke other tests" in prompts[1]
    assert "raise" not in greet.read_text()


def test_the_suspects_put_a_far_away_culprit_on_screen(project):
    _make_broken(project)
    filler = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(400))
    (project / "symbio" / "app" / "greet.py").write_text(
        "def greet(name):\n    return helper(name)\n\n" + filler
        + "def helper(name):\n    return 'Nope ' + name\n", encoding="utf-8")
    culprit = (project / "symbio" / "app" / "greet.py").read_text().splitlines().index(
        "    return 'Nope ' + name") + 1
    prompts = []
    fix_loop.repair_failure(
        "tests/test_x.py::test_greet", "symbio/app/greet.py",
        lambda p: prompts.append(p) or "no patch",
        lambda t: (False, "1 failed in 0.1s"), budget_cycles=1,
        suspects=[("symbio/app/greet.py", culprit, 1.0)])
    assert "return 'Nope ' + name" in prompts[0]


def test_the_top_suspect_overrides_a_caller_frame():
    suspects = [("symbio/app/belief.py", 480, 1.0), ("symbio/app/learn.py", 12, 0.7)]
    assert fix_loop.pick_source(suspects, "symbio/app/learn.py") == "symbio/app/belief.py"
    assert fix_loop.pick_source(suspects, "symbio/app/belief.py") == "symbio/app/belief.py"
    assert fix_loop.pick_source([], "symbio/app/learn.py") == "symbio/app/learn.py"


def test_a_whole_function_in_a_code_block_is_a_patch():
    """MiMo-9B answers with the fixed function in a ```python block, not the
    fence; that is a usable patch anchored on the function's name."""
    text = "import x\n\n\ndef greet(name):\n    return 'Nope'\n\n\ndef other():\n    return 1\n"
    reply = "The fix:\n```python\ndef greet(name):\n    return 'Hi ' + name\n```\n"
    hunks = fix_loop._block_patch(reply, text)
    assert fix_loop._apply_patch(text, hunks) == text.replace("'Nope'", "'Hi ' + name")


def test_a_code_block_snippet_replaces_the_span_it_names():
    text = "def f(a):\n    if a:\n        return 1\n    return 2\n"
    # The edited line comes first, so it matches nowhere: the block is
    # anchored by the window it nearly equals, and a stray fence divider
    # (sent by Nemotron-Elastic-12B) is dropped.
    reply = "```\n====\n    if not a:\n        return 1\n```"
    hunks = fix_loop._block_patch(reply, text)
    assert fix_loop._apply_patch(text, hunks) == "def f(a):\n    if not a:\n        return 1\n    return 2\n"
    reply = "```\n    if a:\n        return 3\n```"
    hunks = fix_loop._block_patch(reply, text)
    assert fix_loop._apply_patch(text, hunks) == "def f(a):\n    if a:\n        return 3\n    return 2\n"


def test_a_code_block_naming_nothing_in_the_file_is_no_patch():
    text = "def f():\n    return 1\n"
    assert fix_loop._block_patch("```python\ndef g():\n    return 2\n```", text) == []
    assert fix_loop._block_patch("no code here", text) == []


def test_an_edit_tool_call_is_read_as_a_patch():
    reply = ("The fix.<tool_call><function=patch><parameter=path>/x.py</parameter>"
             "<parameter=old_string>    return 'Nope'</parameter>"
             "<parameter=new_string>    return 'Hi ' + name</parameter></function></tool_call>")
    assert fix_loop._extract_patch(reply) == [("    return 'Nope'", "    return 'Hi ' + name")]


def test_a_block_that_nearly_matches_two_places_is_no_patch():
    text = "def f(a):\n    if a:\n        return 1\n\ndef g(a):\n    if a:\n        return 1\n"
    assert fix_loop._block_patch("```\n    if not a:\n        return 1\n```", text) == []


def test_a_short_open_fence_is_still_a_patch():
    reply = "<<\nOLD\n    if planned:\n====\n    if not planned:\n>>>> NEW"
    assert fix_loop._extract_patch(reply) == [("    if planned:", "    if not planned:")]


def test_a_chat_turn_starts_no_watcher_when_switched_off(monkeypatch):
    """The suite runs with SYMBIO_FIX_WATCH=0 (conftest): a scripted turn must
    not launch a real suite probe from inside the suite."""
    import threading
    from tests.test_main_loop import ScriptedSession
    started = []
    real = threading.Thread.start

    def spy(self):
        if getattr(self, "_target", None) and self._target.__name__ == "_watch":
            started.append(self)
        return real(self)
    monkeypatch.setattr(threading.Thread, "start", spy)
    ScriptedSession(user_inputs=["hi", "/quit", "n"], model_replies=["Hello."]).run()
    assert started == []
