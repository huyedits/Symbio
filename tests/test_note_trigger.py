"""Tests for the signal-based note-making trigger.

looks_durable_fact is the pure heuristic; _nudge_block is the thin shell that
turns a hit into an immediate <note> nudge instead of waiting for the interval
tick. Both are pinned here.
"""

from types import SimpleNamespace

import pytest

from symbio.app.chat_text import looks_durable_fact


@pytest.mark.parametrize("text", [
    "my name is Huy",
    "I prefer dark mode",
    "I work at Acme Corp",
    "I live in London",
    "remember that my birthday is in June",
    "from now on, keep replies short",
    "I'm a software engineer",
    "my favorite editor is vim",
    "I decided to switch to the new API",
    "I'm working on a new project",
    "I use vim for everything",
])
def test_durable_fact_signals_are_detected(text):
    assert looks_durable_fact(text)


@pytest.mark.parametrize("text", [
    "what is the weather today",
    "search for the price of a mac mini",
    "run df -h",
    "how do I make coffee",
    "tell me a joke",
    "I'm not sure about that",
    "I'm going to the store",
    "",
    "   ",
    "/notes",
])
def test_non_durable_inputs_are_not_detected(text):
    assert not looks_durable_fact(text)


def _session(user_turns, nudge_interval=5):
    return SimpleNamespace(
        config={
            "memory": {"enabled": True, "nudge_interval": nudge_interval},
            "user_name": "Huy",
        },
        user_turns=user_turns,
    )


def test_nudge_fires_immediately_on_a_durable_fact():
    from symbio.app import chat

    nudge = chat.ChatSession._nudge_block(_session(1), "I prefer dark mode")

    assert "<note>" in nudge
    assert "Huy" in nudge


def test_nudge_falls_back_to_the_interval_reminder():
    from symbio.app import chat

    nudge = chat.ChatSession._nudge_block(_session(5), "what is the weather")

    assert "<memory>" in nudge
    assert "<note>" not in nudge


def test_nudge_off_interval_and_no_fact_returns_empty():
    from symbio.app import chat

    assert chat.ChatSession._nudge_block(_session(2), "what is the weather") == ""


def test_nudge_disabled_when_memory_is_off():
    from symbio.app import chat

    session = _session(1)
    session.config["memory"]["enabled"] = False

    assert chat.ChatSession._nudge_block(session, "I prefer dark mode") == ""
