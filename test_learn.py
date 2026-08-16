#!/usr/bin/env python3
"""Unit tests for the /learn correction miner."""
import json
from pathlib import Path

from symbio import DEFAULT_CONFIG, _find_correction_sample, _looks_like_correction


def make_history(turns: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in turns]


def test_auto_correction_phrase_detection():
    history = make_history([
        ("user", "What is my name?"),
        ("assistant", "Your name is Bob."),
    ])
    ok, reason = _looks_like_correction("No, I'm Alice.", history, DEFAULT_CONFIG)
    assert ok and reason == "correction phrase"


def test_auto_no_correction():
    history = make_history([
        ("user", "What is my name?"),
        ("assistant", "Your name is Alice."),
    ])
    ok, _ = _looks_like_correction("Thanks.", history, DEFAULT_CONFIG)
    assert not ok


def test_auto_repeated_question_detection():
    history = make_history([
        ("user", "What is my name?"),
        ("assistant", "Your name is Bob."),
    ])
    ok, reason = _looks_like_correction("What is my name?", history, DEFAULT_CONFIG)
    assert ok and reason == "repeated question"


def test_typical_correction_pattern():
    history = make_history([
        ("user", "What is my name?"),
        ("assistant", "Your name is Bob."),
        ("user", "No, I'm Alice."),
        ("assistant", "Your name is Alice."),
    ])
    sample = _find_correction_sample(history, DEFAULT_CONFIG)
    assert sample is not None
    query, answer = sample
    assert query == "What is my name?"
    assert answer == "Your name is Alice."


def test_no_correction_detected():
    history = make_history([
        ("user", "What is my name?"),
        ("assistant", "Your name is Alice."),
    ])
    assert _find_correction_sample(history, DEFAULT_CONFIG) is None


def test_correction_with_tool_observation():
    history = make_history([
        ("user", "What is in the project directory?"),
        ("assistant", "It contains only config.json."),
        ("user", "No, list the files with ls."),
        ("tool", "terminal: ..."),
        ("assistant", '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls -la"}}</tool_call>Here is the listing.'),
    ])
    sample = _find_correction_sample(history, DEFAULT_CONFIG)
    assert sample is not None
    query, answer = sample
    assert query == "What is in the project directory?"
    assert "terminal" not in answer


def test_no_correction_phrase_means_no_sample():
    history = make_history([
        ("user", "What is my name?"),
        ("assistant", "Your name is Alice."),
        ("user", "Thanks."),
        ("assistant", "You're welcome."),
    ])
    assert _find_correction_sample(history, DEFAULT_CONFIG) is None


if __name__ == "__main__":
    test_auto_correction_phrase_detection()
    test_auto_no_correction()
    test_auto_repeated_question_detection()
    test_typical_correction_pattern()
    test_no_correction_detected()
    test_correction_with_tool_observation()
    test_no_correction_phrase_means_no_sample()
    print("All /learn miner tests passed.")


from symbio.app import learn  # the app-level miner, not the legacy one

# ---- a self-teaching system must not trust its own restatement ----
#
# The "correct answer" in a mistake note is the model's own reply after being
# corrected, and it goes into the corpus as ground truth. Measured live: told
# plainly "I use Helix", it wrote back "You use Helix, which is a terminal
# multiplexer and text editor combination" — Helix is not a terminal
# multiplexer — and that was stored as fact, ready to be trained in.

def test_an_invented_gloss_is_cut_from_the_correct_answer():
    out = learn.ground_corrected_answer(
        "You use Helix, which is a terminal multiplexer and text editor "
        "combination. To use it, you can run `helix` in the terminal.",
        "No, that's wrong. I use Helix.",
        "What text editor do I use?")
    assert out == "You use Helix."


def test_a_grounded_list_keeps_its_punctuation():
    """Truncating the original beats splitting and rejoining, which turned
    'Helix, Vim, and Emacs' into 'Helix Vim and Emacs'."""
    out = learn.ground_corrected_answer(
        "You use Helix, Vim, and Emacs.",
        "No, I use Helix, Vim and Emacs.",
        "Which editors do I use?")
    assert out == "You use Helix, Vim, and Emacs."


def test_a_wholly_invented_answer_is_dropped():
    """Better to lose the sample than to mint a false one."""
    assert learn.ground_corrected_answer(
        "Your car is a Volvo built in Gothenburg.",
        "No, my car is a Saab.", "What car do I drive?") == ""


def test_an_already_grounded_answer_is_untouched():
    for answer, correction, question in [
        ("You use Helix.", "No, I use Helix.", "What editor do I use?"),
        ("Your cat is called Mochi.", "No, my cat is Mochi.", "What is my cat called?"),
    ]:
        assert learn.ground_corrected_answer(answer, correction, question) == answer


def test_the_correction_sample_carries_the_grounded_answer(monkeypatch):
    """End to end through find_correction_sample, which is what actually
    writes the note."""
    history = [
        {"role": "user", "content": "What text editor do I use?"},
        {"role": "assistant", "content": "The assistant doesn't use a text editor."},
        {"role": "user", "content": "No, that's wrong. I use Helix."},
        {"role": "assistant",
         "content": "You use Helix, which is a terminal multiplexer and text editor combination."},
    ]
    from symbio.app import config as _config
    sample = learn.find_correction_sample(history, _config.load_config())
    assert sample is not None
    assert sample[3] == "You use Helix.", sample[3]


# ---- a correction belongs to the skill it is about ----
#
# Corrections were filed against every skill note *retrieved* during the
# session, cumulatively. Measured live: a correction about which text editor
# the user prefers was written into the health sidecars of Device Awareness,
# folding a fitted sheet, and descaling a kettle. Those sidecars feed skill
# retraining, so it would have been trained into unrelated adapters — through
# the per-skill isolation that exists to stop exactly that.

def _skill_note(tmp_path, title, body="1. Do the thing."):
    p = tmp_path / f"Skill__{title.replace(' ', '_')}.md"
    p.write_text(f"# Skill: {title}\n\n{body}\n", encoding="utf-8")
    return p


_EDITOR_CORRECTION = ("Original question: What text editor do I use? "
                      "Wrong answer: The assistant does not use one. "
                      "Correction: I use Helix. Correct answer: You use Helix.")


def test_an_unrelated_skill_does_not_collect_the_correction(tmp_path):
    for title in ["descaling a kettle", "Fold a fitted sheet", "Device Awareness"]:
        note = _skill_note(tmp_path, title)
        assert not learn.correction_concerns_skill(_EDITOR_CORRECTION, note), title


def test_the_skill_it_is_about_does_collect_it(tmp_path):
    note = _skill_note(tmp_path, "descaling a kettle")
    correction = ("Original question: How do I descale a kettle? "
                  "Correction: Use vinegar. Correct answer: Fill the kettle with vinegar.")
    assert learn.correction_concerns_skill(correction, note)


def test_a_slug_title_still_matches(tmp_path):
    """Titles arrive as "Fix wifi" from a human and "descaling_a_kettle" when
    the model emits the slug; the slug tokenises as one word."""
    note = _skill_note(tmp_path, "descaling_a_kettle")
    correction = ("Original question: How do I descale a kettle? "
                  "Correct answer: Fill the kettle with vinegar.")
    assert learn.correction_concerns_skill(correction, note)


def test_scaffold_labels_are_not_content(tmp_path):
    """"Wrong answer:"/"Correct answer:" matched every skill note containing
    the word "answer", which is most of them."""
    note = _skill_note(tmp_path, "descaling a kettle",
                       body="1. Answer the user. 2. Give the correct answer.")
    assert not learn.correction_concerns_skill(_EDITOR_CORRECTION, note)


def test_a_missing_note_is_not_a_match(tmp_path):
    assert not learn.correction_concerns_skill(_EDITOR_CORRECTION, tmp_path / "gone.md")


def test_an_acknowledgement_is_not_a_corrected_answer():
    """Trimming can leave "Okay." — safe, and worthless. Observed when the
    reply was the model's own reasoning rather than an answer, which the
    thinking adapter does on some prompts. Training on it teaches the model
    to answer a question with a nod."""
    assert learn.ground_corrected_answer(
        'Okay, the user is asking, "What text editor do I use?" Let me check.',
        "No, that's wrong. I use Helix.", "What text editor do I use?") == ""


def test_a_reasoning_leak_does_not_become_a_training_sample():
    """The whole sample is dropped, not stored half-formed."""
    from symbio.app import config as _config
    history = [
        {"role": "user", "content": "What text editor do I use?"},
        {"role": "assistant", "content": "The assistant does not use one."},
        {"role": "user", "content": "No, that's wrong. I use Helix."},
        {"role": "assistant",
         "content": "Okay, the user is asking about editors. Let me check the history."},
    ]
    assert learn.find_correction_sample(history, _config.load_config()) is None
