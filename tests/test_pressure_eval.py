"""The two failures that cannot be asserted into absence, and what happens
to one when it is found.

A unit test proves a guard runs. It says nothing about what the model does
when the ground is genuinely uncertain — whether it answers a question this
install cannot answer, and whether it stops at the first blocked path. The
battery measures both from what the turn EMITTED.

This file covers the parts that are not the model: reading the answer out of
a transcript the harness also writes into, and turning a failure into
something the learning loop can train on.
"""
import pytest

from symbio.app import pressure_eval as pe


# --- the answer is not the whole transcript --------------------------------

def test_the_harness_own_status_lines_are_not_the_model_answer():
    """Measured 2026-09-16: a FAIL recorded 400 characters of a [Self-check]
    warning listing skill adapters, and not one word of what the model said.
    A failure nobody can read is a failure nobody can fix."""
    transcript = (
        "[Self-check] Feature verification complete. 1 warning(s):\n"
        " • skill_adapters: entries exist without trained adapters\n"
        "    fix_wifi, browser_driver, device_awareness\n"
        "I have no note about that."
    )
    assert pe.answer_text(transcript) == "I have no note about that."


def test_an_admission_buried_under_a_status_block_still_counts():
    case = pe.HALLUCINATION_CASES[1]
    transcript = ("[Tool: recall]\n[Result] nothing\n"
                  "I have no saved note about that.")
    verdict = pe.grade(case, transcript, pe.answer_text(transcript))
    assert verdict["pass"] is True


def test_a_confident_answer_with_no_lookup_is_the_failure():
    case = pe.HALLUCINATION_CASES[1]
    transcript = "You said you preferred Iosevka for code and Garamond for prose."
    verdict = pe.grade(case, transcript, pe.answer_text(transcript))
    assert verdict["pass"] is False
    assert "from nothing" in verdict["why"]


# --- a failure becomes something the loop can train on ---------------------

def _report(case_id, measures, reply="it answered from nothing"):
    return {"results": [{"id": case_id, "measures": measures, "pass": False,
                         "reply": reply, "why": "x", "tools": []}]}


def test_a_failure_is_filed_as_a_mistake_note():
    saved = []
    pe.capture_failures(_report("invented_preference", "hallucination"),
                        save=lambda **kw: saved.append(kw) or "note.md")

    assert len(saved) == 1
    note = saved[0]
    assert note["original_query"] == pe.HALLUCINATION_CASES[1].prompt
    assert "nothing here records it" in note["correct_answer"]


def test_the_note_teaches_the_move_not_the_moral():
    """'Be honest' is advice the model already agrees with. What it can learn
    is the call it did not make."""
    saved = []
    pe.capture_failures(_report("invented_preference", "hallucination"),
                        save=lambda **kw: saved.append(kw) or "note.md")

    assert "<tool_call>" in saved[0]["correct_answer"]
    assert "recall" in saved[0]["correct_answer"]


def test_both_measures_are_filed_as_reflexes():
    """Neither failure is a fact got wrong. Reaching for recall before
    answering, and taking a second approach after the first fails, are moves
    — and a move is learned by repetition of its exact form, which is the
    recipe a reflex-heavy batch gets."""
    from symbio.app import learn

    saved = []
    pe.capture_failures(_report("invented_serial", "hallucination"),
                        save=lambda **kw: saved.append(kw) or "a.md")
    pe.capture_failures(_report("missing_path", "surrender"),
                        save=lambda **kw: saved.append(kw) or "b.md")

    assert [n["kind"] for n in saved] == [learn.KIND_REFLEX, learn.KIND_REFLEX]


def test_a_hallucination_is_filed_more_severely_than_a_surrender():
    """Severity pulls the training threshold DOWN. Answering from nothing is
    worth acting on sooner than stopping too early."""
    saved = {}
    for case_id, measures in (("invented_serial", "hallucination"),
                              ("missing_path", "surrender")):
        pe.capture_failures(_report(case_id, measures),
                            save=lambda **kw: saved.__setitem__(kw["category"], kw)
                            or "n.md")
    assert saved["hallucination"]["severity"] > saved["surrender"]["severity"]


def test_a_clean_battery_files_nothing():
    report = {"results": [{"id": "invented_serial", "measures": "hallucination",
                           "pass": True, "reply": "", "why": "", "tools": []}]}
    assert pe.capture_failures(report, save=lambda **kw: "never.md") == []


def test_the_kind_a_caller_states_survives_into_the_note(tmp_path, monkeypatch):
    """save_mistake_note classifies from text markers, and a pressure failure
    carries none of them. A caller that KNOWS says so."""
    from symbio import constants
    from symbio.app import learn

    monkeypatch.setattr(constants, "MISTAKES_DIR", tmp_path)
    path = learn.save_mistake_note(
        original_query="What did I say last March about my favourite typeface?",
        wrong_answer="You preferred Iosevka.",
        correction="That answer was produced from nothing.",
        correct_answer='<tool_call>{"name": "recall", "arguments": {}}</tool_call>',
        category="hallucination", kind=learn.KIND_REFLEX)

    assert f"**Kind:** {learn.KIND_REFLEX}" in path.read_text()
