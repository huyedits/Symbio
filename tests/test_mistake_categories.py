"""The model's own labelling of its mistakes, and the counter that shows it.

A mistake note carries a "**Category:**" line the model wrote itself, and the
boot banner turns the pending notes into '3/5 mistakes to next tune (2
tool_error, 1 wrong_tool)' — a line that says what the next tune will be
trained to fix, not just how many corrections are stacked up. These tests pin
the slug normalisation, the header read, and when the breakdown earns its
parentheses.
"""
import pytest

from symbio import constants
from symbio.app import chat_ui, learn


@pytest.fixture
def mistakes(tmp_path, monkeypatch):
    d = tmp_path / "mistakes"
    d.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_DIR", d)
    return d


def write_note(mistakes, name: str, category: str | None, body: str = ""):
    """A mistake note in save_mistake_note's shape, with or without the field."""
    head = f"# Correction: {name}\n\n"
    if category is not None:
        head += f"**Category:** {category}\n\n"
    head += "**Severity:** 1\n\n**Original question:** q\n\n"
    p = mistakes / f"{name}.md"
    p.write_text(head + body, encoding="utf-8")
    return p


# ---- the slug ----

@pytest.mark.parametrize("raw, expected", [
    ("tool_error", "tool_error"),
    ("Tool Error", "tool_error"),
    ("tool-error", "tool_error"),
    ("  TOOL ERROR  ", "tool_error"),
])
def test_spellings_of_one_label_land_in_one_bucket(raw, expected):
    assert learn._slug_category(raw) == expected


def test_the_echoed_prompt_word_is_not_part_of_the_label():
    """The prompt ends on the literal word "Category:" and the model echoes it.
    Left in, the two-word trim turns this into the bucket 'category_tool'."""
    assert learn._slug_category("Category: tool_error") == "tool_error"


def test_a_longer_phrase_is_trimmed_to_two_words():
    assert learn._slug_category("wrong tool for the job") == "wrong_tool"


@pytest.mark.parametrize("raw", ["", "   ", "!!!", None])
def test_an_unusable_label_falls_back_to_general(raw):
    assert learn._slug_category(raw) == "general"


# ---- the classifier ----

def test_the_model_names_the_category():
    got = learn.classify_mistake_category(
        "what time is it", "3pm", "5pm", lambda prompt: "wrong_answer")
    assert got == "wrong_answer"


def test_categories_already_in_use_are_offered_back_to_the_model():
    seen = {}

    def classify(prompt):
        seen["prompt"] = prompt
        return "tool_error"

    learn.classify_mistake_category("q", "w", "c", classify,
                                    known_categories=("tool_error", "wrong_tool"))
    assert "tool_error, wrong_tool" in seen["prompt"]


def test_no_model_means_general_rather_than_a_guess():
    assert learn.classify_mistake_category("q", "w", "c", None) == "general"


def test_a_failing_generation_never_costs_the_capture():
    """The label annotates a mistake note; it must not be able to prevent one."""
    def boom(prompt):
        raise RuntimeError("model unloaded")

    assert learn.classify_mistake_category("q", "w", "c", boom) == "general"


def test_a_chatty_reply_is_reduced_to_its_first_line():
    got = learn.classify_mistake_category(
        "q", "w", "c", lambda p: "tool_error\nBecause the tool was misused.")
    assert got == "tool_error"


# ---- the counts ----

def test_notes_are_counted_by_category_dominant_first(mistakes):
    write_note(mistakes, "a", "tool_error")
    write_note(mistakes, "b", "tool_error")
    write_note(mistakes, "c", "wrong_tool")

    assert list(learn.mistake_category_counts().items()) == [
        ("tool_error", 2), ("wrong_tool", 1)]


def test_a_note_written_before_the_field_existed_counts_as_general(mistakes):
    write_note(mistakes, "old", None)

    assert learn.mistake_category_counts() == {"general": 1}


def test_the_breakdown_sums_to_the_counter_beside_it(mistakes):
    """The banner prints mistake_note_count() and this breakdown side by side,
    so every note the counter walks has to land in some bucket here."""
    write_note(mistakes, "a", "tool_error")
    write_note(mistakes, "b", None)
    write_note(mistakes, "c", "wrong_tool")

    counts = learn.mistake_category_counts()

    assert sum(counts.values()) == learn.mistake_note_count() == 3


def test_the_field_is_read_from_the_header_not_the_body(mistakes):
    """The body can run to a whole tool observation, and a note quoting the
    field inside it must not be read as declaring one."""
    write_note(mistakes, "quoting", "tool_error",
               body="**Correct answer:** the header said **Category:** wrong_tool\n")

    assert learn.mistake_category_counts() == {"tool_error": 1}


def test_no_directory_is_an_empty_breakdown(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "MISTAKES_DIR", tmp_path / "missing")
    assert learn.mistake_category_counts() == {}


# ---- the banner line ----

def _config(**learn_cfg):
    """A pinned threshold, so these tests are about the BREAKDOWN text.

    The banner now reads the same dynamic_mistake_threshold the gate fires on,
    and that number scales with the size of train.jsonl — which is the real
    one here, 882 rows on the box this was written on. Left scaling, every
    assertion below would also be an assertion about how much training data
    the machine running the suite happens to have. The scaling itself is
    tested underneath, with the corpus passed in.
    """
    return {"learn": {"enabled": True, "mistake_threshold": 5,
                      "scale_threshold_with_corpus": False,
                      "auto_train": True, **learn_cfg}}


def test_the_line_names_the_kinds_that_are_stacking_up(mistakes):
    write_note(mistakes, "a", "tool_error")
    write_note(mistakes, "b", "tool_error")
    write_note(mistakes, "c", "wrong_tool")

    assert chat_ui.learn_progress_line(_config()) == (
        "3/5 mistakes to next tune (2 tool_error, 1 wrong_tool)")


def test_a_lone_general_bucket_earns_no_parentheses(mistakes):
    """'general' is what a note says when nothing has classified it. Printing
    '(3 general)' beside '3/5 mistakes' names nothing the other half didn't."""
    write_note(mistakes, "a", None)
    write_note(mistakes, "b", None)
    write_note(mistakes, "c", "general")

    assert chat_ui.learn_progress_line(_config()) == "3/5 mistakes to next tune"


def test_general_is_still_shown_alongside_a_real_category(mistakes):
    write_note(mistakes, "a", None)
    write_note(mistakes, "b", "tool_error")

    assert chat_ui.learn_progress_line(_config()).endswith(
        "(1 general, 1 tool_error)")


def test_the_breakdown_survives_the_threshold_being_reached(mistakes):
    for i in range(5):
        write_note(mistakes, f"n{i}", "tool_error")

    assert chat_ui.learn_progress_line(_config()) == (
        "5/5 mistakes — tuning due (5 tool_error)")


def test_the_breakdown_follows_the_auto_train_suffix(mistakes):
    write_note(mistakes, "a", "tool_error")

    assert chat_ui.learn_progress_line(_config(auto_train=False)) == (
        "1/5 mistakes to next tune (auto-train off) (1 tool_error)")


def test_a_disabled_loop_says_nothing_about_categories(mistakes):
    write_note(mistakes, "a", "tool_error")

    assert chat_ui.learn_progress_line(_config(enabled=False)) == "learn: off"


# ---- the banner counts to the bar that actually fires ----

def test_the_banner_follows_the_dynamic_threshold_not_the_config(mistakes):
    """It read learn.mistake_threshold straight out of config while
    maybe_train_on_mistakes decided on dynamic_mistake_threshold. On this
    install that is 5 against 6: /status said "3/5 to next tune" and nothing
    happened at 5, because the counter the user watches was not the counter
    that fires."""
    write_note(mistakes, "a", "tool_error")
    config = {"learn": {"enabled": True, "mistake_threshold": 5,
                        "auto_train": True}}

    threshold = learn.dynamic_mistake_threshold(
        config, severity_total=learn.pending_severity_total())
    line = chat_ui.learn_progress_line(config)

    assert line.startswith(f"1/{threshold} mistakes")


def test_a_bigger_corpus_raises_the_bar_the_banner_shows(mistakes, monkeypatch):
    """Dilution: five boosted notes are most of an epoch against 50 samples
    and a few percent of one against 882."""
    write_note(mistakes, "a", "tool_error")
    config = {"learn": {"enabled": True, "mistake_threshold": 5,
                        "auto_train": True}}

    monkeypatch.setattr(learn, "_training_sample_count", lambda: 50)
    small = chat_ui.learn_progress_line(config)
    monkeypatch.setattr(learn, "_training_sample_count", lambda: 5000)
    large = chat_ui.learn_progress_line(config)

    assert small == "1/3 mistakes to next tune (1 tool_error)"
    assert large == "1/10 mistakes to next tune (1 tool_error)"


def test_switching_the_scaling_off_shows_the_configured_number(mistakes,
                                                               monkeypatch):
    write_note(mistakes, "a", "tool_error")
    monkeypatch.setattr(learn, "_training_sample_count", lambda: 5000)

    assert chat_ui.learn_progress_line(_config()).startswith("1/5 mistakes")
