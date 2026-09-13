"""The store of what the machine takes itself to be.

Two sections, written differently: what it is being USED AS, and the operating
values it has read off how the user works. Neither fits agent_memory.md
(facts) or user_profile.md (who they are) — "wants the least possible
friction" and "wants to approve everything" are opposite instructions, and
which one is true decides how a turn should go.
"""
import pytest

from symbio import constants
from symbio.app import soul


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "SOUL_FILE", tmp_path / "soul.md")
    return tmp_path / "soul.md"


CFG = {"user_name": "Huy", "memory": {}}
_ROLE = "Used as a hands-on engineer: asked to fix, test and push."
_VALUE = "Wants the fewest confirmations (told me twice to just do it)."


# ---- what it will and will not write down ----

def test_an_observation_is_recorded_under_its_section(store):
    assert soul.record(soul.ROLE, _ROLE, CFG)
    assert soul.record(soul.VALUE, _VALUE, CFG)

    parsed = soul.sections()
    assert parsed[soul.ROLE] == [_ROLE]
    assert parsed[soul.VALUE] == [_VALUE]


def test_the_same_read_is_not_written_twice(store):
    """A pass that runs after every turn produces the same read repeatedly. A
    store that appends each time says one sentence forty times and evicts
    everything else to make room."""
    soul.record(soul.ROLE, _ROLE, CFG)

    assert soul.record(soul.ROLE, "Used as a hands-on engineer: asked to fix, "
                                  "test and push without checking in.", CFG) is False
    assert len(soul.sections()[soul.ROLE]) == 1


def test_a_character_judgement_is_refused(store):
    """The line sits in every prompt from then on, and the person never gets
    to answer it. Describe the behaviour or write nothing."""
    assert soul.record(soul.VALUE, "Is impatient (rushes me).", CFG) is False
    assert soul.sections()[soul.VALUE] == []


def test_a_value_without_the_behaviour_that_showed_it_is_refused(store):
    """The evidence is what makes it revisable later. "Wants speed" cannot be
    argued with; "wants speed (asked me to skip the summary twice)" can."""
    assert soul.record(soul.VALUE, "Wants speed.", CFG) is False


def test_a_role_line_needs_no_bracket(store):
    """It is the machine's read of its own place, not a claim about someone."""
    assert soul.record(soul.ROLE, "Used as a companion more than a tool.", CFG)


def test_a_secret_never_reaches_the_store(store):
    """This file goes into every prompt and is a training input like any other
    store, so a credential that landed here once would keep arriving."""
    soul.record(soul.ROLE, "Used to deploy with token ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.", CFG)

    assert "ghp_aaaa" not in store.read_text()


def test_the_empty_placeholder_is_not_read_back_as_an_entry(store):
    """It rendered as "- (nothing observed yet)" and parsed back as a real
    line, so the first real observation left the store saying both."""
    soul.record(soul.ROLE, _ROLE, CFG)          # VALUE renders the placeholder
    soul.record(soul.VALUE, _VALUE, CFG)

    assert soul.sections()[soul.VALUE] == [_VALUE]
    assert store.read_text().count("nothing observed yet") == 0


def test_the_store_stays_small_enough_to_sit_in_a_prompt(store):
    """It is a reading, not a diary. Over the cap, the OLDEST line goes: a
    superseded read of the relationship is exactly the one worth losing."""
    cfg = {"user_name": "Huy", "memory": {"soul_char_limit": 400}}
    for i in range(12):
        soul.record(soul.VALUE, f"Wants thing {i} (did behaviour {i}).", cfg)

    text = store.read_text()
    assert len(text) <= 400
    assert "thing 11" in text            # the newest survived
    assert "thing 0" not in text         # the oldest did not


# ---- how it reaches the model ----

def test_an_empty_store_sends_nothing(store):
    """Context spent telling the model it knows nothing about itself."""
    assert soul.soul_block(CFG) == ""


def test_what_it_has_observed_is_wrapped_as_untrusted(store):
    """Every line was derived from conversation, and conversation includes
    whatever a web page said. A page must not be able to tell the assistant
    what it is for by having its words written down and read back."""
    soul.record(soul.ROLE, _ROLE, CFG)

    block = soul.soul_block(CFG)

    assert "Begin untrusted" in block
    assert _ROLE in block


def test_it_can_be_switched_off(store):
    soul.record(soul.ROLE, _ROLE, CFG)

    assert soul.soul_block({"user_name": "Huy",
                            "memory": {"soul_enabled": False}}) == ""


# ---- when it looks ----

def test_a_correction_is_drastic():
    """Corrections, refusals and failures are the turns that REVISE a read
    rather than confirm it. A turn that went fine says little that was not
    already known."""
    history = [{"role": "user", "content": "what is the capital of France?"},
               {"role": "assistant", "content": "Berlin."}]
    # The real phrase list, not an empty one: looks_like_correction matches
    # against config, so a bare {} makes every correction invisible and the
    # test would be passing on a code path nobody runs.
    config = {"learn": {"correction_phrases": ["no,", "that's wrong", "wrong"]},
              "agent": {}}

    assert soul.is_drastic("no, it's Paris", "", history, config) is True


def test_a_refused_tool_is_drastic():
    assert soul.is_drastic(
        "do it", "Tool 'run_command' was not approved (risk score 3/3: shell).",
        [], {"learn": {}, "agent": {}}) is True


def test_an_ordinary_turn_is_not():
    assert soul.is_drastic(
        "thanks", "Saved note.", [],
        {"learn": {"correction_phrases": ["no,", "wrong"]}, "agent": {}}) is False


# ---- the reflection pass itself ----

def test_a_reply_is_parsed_into_observations():
    """The shape the 14B actually returns, verified live against this
    conversation: 6.9s, and it arrived at "minimal friction" on its own."""
    reply = ("ROLE: Used as a problem solver expected to act quickly.\n"
             "VALUE: Wants minimal friction (insists on fixing without extra steps).")

    assert soul.parse(reply) == [
        (soul.ROLE, "Used as a problem solver expected to act quickly."),
        (soul.VALUE, "Wants minimal friction (insists on fixing without extra steps)."),
    ]


def test_none_writes_nothing():
    assert soul.parse("NONE") == []


def test_a_pass_that_fails_costs_the_turn_nothing(store):
    """It runs on a background thread beside the note indexer. A model error
    there must not surface as anything at all."""
    def explode(_prompt):
        raise RuntimeError("Metal OOM")

    assert soul.reflect([{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "hello"}],
                        CFG, explode) == []


def test_the_prompt_carries_the_recent_conversation(store):
    history = [{"role": "user", "content": "fix the click bug and push it"},
               {"role": "assistant", "content": "Pushed as 6952588."}]

    prompt = soul.build_prompt(history, CFG)

    assert "fix the click bug" in prompt
    assert "Huy" in prompt
