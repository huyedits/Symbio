"""What a mistake asks the weights to change, and how much evidence to wait for.

Two things were fixed at once, both of them formulas that had been constants.

A correction is not one kind of thing. "Your name is Bob" / "No, I'm Alice" is
a FACT the model must recall. `<tool_call>{"name": "chrome"}` followed by
`open -a 'Google Chrome'` is a SHAPE it must reach for automatically. Trained
identically, each gets the wrong treatment: a fact repeated four times in one
epoch is still one fact, and an action learned once is not a reflex.

And the trigger was a fixed five notes, which was right at 50 training samples
and wrong at 882: the same five boosted notes are most of an epoch in the
first case and a rounding error in the second.
"""
import json

import pytest

from symbio import constants
from symbio.app import learn


# --- what kind of mistake is this ------------------------------------------


@pytest.mark.parametrize("case", [
    # The automatic capture: a tool failed, the next one worked. Nothing was
    # asserted and corrected — an action was reached for and missed.
    dict(original_query="open chrome",
         wrong_answer='<tool_call>{"name": "chrome"}</tool_call>',
         correction=learn.AUTO_TOOL_CORRECTION,
         correct_answer='<tool_call>{"name": "terminal", "arguments": {"cmd": "open -a \'Google Chrome\'"}}</tool_call>'),
    dict(original_query="edit the config",
         wrong_answer="sed -i 's/a/b/' config.json",
         correction="that silently does nothing on macOS",
         correct_answer="sed -i '' 's/a/b/' config.json"),
    dict(original_query="post it",
         wrong_answer="typed into the window",
         correction="Refused to type: the focused control is a Button",
         correct_answer="desktop_type with element 2"),
])
def test_a_wrong_shape_is_a_reflex(case):
    assert learn.classify_mistake_kind(**case) == learn.KIND_REFLEX


@pytest.mark.parametrize("case", [
    dict(original_query="what is my name?", wrong_answer="Your name is Bob.",
         correction="No, I'm Alice.", correct_answer="Your name is Alice."),
    dict(original_query="which model is the headmaster?",
         wrong_answer="The 8B.", correction="that's wrong, it is the 14B",
         correct_answer="Qwen3-14B at 3-bit."),
])
def test_a_wrong_statement_is_knowledge(case):
    assert learn.classify_mistake_kind(**case) == learn.KIND_KNOWLEDGE


def test_an_unreadable_mistake_falls_back_to_knowledge():
    """Ties go to knowledge: repeating a fact a few extra times is harmless,
    while drilling a wrong action makes it automatic."""
    assert learn.classify_mistake_kind() == learn.KIND_KNOWLEDGE


# --- the recipe that follows from it ---------------------------------------


def test_a_batch_of_reflexes_repeats_harder_and_runs_shorter():
    reflexes = learn.training_recipe({"reflex": 5, "knowledge": 0}, 25, 3)
    facts = learn.training_recipe({"reflex": 0, "knowledge": 5}, 25, 3)

    assert reflexes["boost"] > facts["boost"]
    assert reflexes["iters"] < facts["iters"]


def test_a_batch_of_facts_is_the_old_recipe_unchanged():
    """The knowledge-only batch is the baseline the severity scaling was
    tuned against; kinds must move off it, not move it."""
    assert learn.training_recipe({"reflex": 0, "knowledge": 5}, 25, 3)["iters"] == 25
    assert learn.training_recipe({"reflex": 0, "knowledge": 5}, 25, 3)["boost"] == 3


def test_a_mixed_batch_lands_between_the_two():
    """One batch, one run: this machine trains one model at a time, so the
    mix moves the dials rather than splitting into two passes."""
    mixed = learn.training_recipe({"reflex": 2, "knowledge": 2}, 25, 3)
    reflexes = learn.training_recipe({"reflex": 4, "knowledge": 0}, 25, 3)
    facts = learn.training_recipe({"reflex": 0, "knowledge": 4}, 25, 3)

    assert facts["iters"] > mixed["iters"] > reflexes["iters"]
    assert mixed["reflex_share"] == 0.5


def test_the_recipe_leaves_severity_to_its_caller():
    """maybe_train_on_mistakes already scales by severity. Doing it here too
    double-counted the same backlog: a severity-5 batch of two notes asked for
    55 iterations where the contract says 40."""
    import inspect

    source = inspect.getsource(learn.training_recipe)
    assert "severity_total" not in source.split('"""')[-1]


def test_an_empty_batch_asks_for_nothing_unusual():
    assert learn.training_recipe({}, 25, 3)["iters"] == 25


# --- how much evidence to wait for -----------------------------------------


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    def _set(samples: int, pending: int = 0, severity: int = 1):
        train = tmp_path / "train.jsonl"
        train.write_text("\n".join(json.dumps({"text": f"sample {i}"})
                                   for i in range(samples)), encoding="utf-8")
        mistakes = tmp_path / "mistakes"
        mistakes.mkdir(exist_ok=True)
        for old in mistakes.glob("*.md"):
            old.unlink()
        for i in range(pending):
            (mistakes / f"m{i}.md").write_text(
                f"# Correction\n\n**Severity:** {severity}\n", encoding="utf-8")
        monkeypatch.setattr(constants, "TRAIN_FILE", train)
        monkeypatch.setattr(constants, "MISTAKES_DIR", mistakes)
    return _set


def test_a_bigger_corpus_asks_for_more_evidence(corpus):
    """Every sample in train.jsonl competes with the new ones."""
    config = {"learn": {"mistake_threshold": 5}}

    corpus(50)
    small = learn.dynamic_mistake_threshold(config)
    corpus(5000)
    large = learn.dynamic_mistake_threshold(config)

    assert small < large


def test_the_bar_grows_by_log_not_by_multiple(corpus):
    """A corpus ten times bigger does not need ten times the evidence."""
    config = {"learn": {"mistake_threshold": 5}}

    corpus(500)
    at_reference = learn.dynamic_mistake_threshold(config)
    corpus(5000)
    ten_times = learn.dynamic_mistake_threshold(config)

    assert ten_times < at_reference * 3


def test_severe_mistakes_pull_the_bar_back_down(corpus):
    config = {"learn": {"mistake_threshold": 5}}

    corpus(5000, pending=4, severity=1)
    mild = learn.dynamic_mistake_threshold(config, severity_total=4)
    corpus(5000, pending=4, severity=3)
    harsh = learn.dynamic_mistake_threshold(config, severity_total=12)

    assert harsh < mild


def test_the_bar_is_clamped_at_both_ends(corpus):
    """Below 2 the agent retrains on noise; above 20 a real regression sits
    uncorrected for a week."""
    config = {"learn": {"mistake_threshold": 5}}

    corpus(1)
    assert learn.dynamic_mistake_threshold(config) >= 2
    corpus(10_000_000)
    assert learn.dynamic_mistake_threshold(config) <= 20


def test_the_scaling_can_be_switched_off(corpus):
    corpus(100000)
    config = {"learn": {"mistake_threshold": 5, "scale_threshold_with_corpus": False}}

    assert learn.dynamic_mistake_threshold(config) == 5


# --- the note carries its kind ---------------------------------------------


def test_a_saved_note_records_its_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "MISTAKES_DIR", tmp_path)

    path = learn.save_mistake_note(
        "open chrome", '<tool_call>{"name": "chrome"}</tool_call>',
        learn.AUTO_TOOL_CORRECTION, "open -a 'Google Chrome'")

    assert "**Kind:** reflex" in path.read_text(encoding="utf-8")


def test_notes_written_before_kinds_existed_are_classified_on_read(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "MISTAKES_DIR", tmp_path)
    (tmp_path / "old.md").write_text(
        "# Correction\n\n**Severity:** 1\n\n"
        "**Wrong answer:** unknown tool: browser_read\n", encoding="utf-8")

    assert learn.pending_mistake_kinds()[learn.KIND_REFLEX] == 1
