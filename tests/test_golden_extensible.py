"""Tests for the user-extensible golden set."""

import json

import pytest

from symbio import constants
from symbio.app import golden


@pytest.fixture
def golden_file(tmp_path, monkeypatch):
    path = tmp_path / "golden_cases.json"
    monkeypatch.setattr(constants, "GOLDEN_CASES_FILE", path)
    return path


def test_load_user_golden_cases_empty(golden_file):
    assert golden.load_user_golden_cases() == []
    assert len(golden.all_golden_cases()) == len(golden.GOLDEN_CASES)


def test_load_user_golden_cases_basic(golden_file):
    golden_file.write_text(json.dumps({
        "test_case": {
            "description": "A test extension",
            "prompt": "Say hello to ASSISTANT_NAME.",
            "requirements": [{"kind": "contains", "text": "hello"}],
            "ideal_reply": "hello there",
        }
    }), encoding="utf-8")

    cases = golden.all_golden_cases()
    ids = [c.id for c in cases]
    assert "test_case" in ids
    # Built-ins still present.
    assert "greeting" in ids

    # Check resolves correctly.
    case = next(c for c in cases if c.id == "test_case")
    assert case.description == "A test extension"
    assert case.prompt_fn({"assistant_name": "Caine", "user_name": "Huy"}) == "Say hello to Caine."
    assert case.check("hello world", [], {}) is True
    assert case.check("goodbye", [], {}) is False


def test_user_ideal_reply_used_for_remedy(golden_file):
    golden_file.write_text(json.dumps({
        "test_case": {
            "description": "A test extension",
            "prompt": "Say hello.",
            "requirements": [{"kind": "contains", "text": "hello"}],
            "ideal_reply": "hello there",
        }
    }), encoding="utf-8")

    reply = golden._user_ideal_reply("test_case", {"assistant_name": "Caine", "user_name": "Huy"})
    assert reply == "hello there"


def test_builtin_cases_take_precedence(golden_file):
    golden_file.write_text(json.dumps({
        "greeting": {
            "description": "Overridden",
            "prompt": "Overridden prompt",
            "requirements": [{"kind": "sane_reply"}],
        }
    }), encoding="utf-8")

    cases = golden.all_golden_cases()
    greeting = next(c for c in cases if c.id == "greeting")
    assert greeting.prompt_fn({"assistant_name": "Caine", "user_name": "Huy"}) == "Hey there!"
