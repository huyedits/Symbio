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
