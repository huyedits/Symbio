#!/usr/bin/env python3
"""Test deferred correction learning: mistake notes accumulate, training only runs at threshold."""
import shutil
from pathlib import Path

from mlx_lm import load

from symbio import (
    ADAPTER_DIR,
    AIAgent,
    DEFAULT_CONFIG,
    MISTAKES_ARCHIVE_DIR,
    MISTAKES_DIR,
    _archive_mistake_notes,
    _digest_mistakes_to_training,
    _mistake_note_count,
    _safe_mistake_filename,
    _save_mistake_note,
    learn_from_last_correction,
    load_config,
    maybe_train_on_mistakes,
)


def reset_mistakes():
    if MISTAKES_DIR.exists():
        for f in MISTAKES_DIR.glob("*.md"):
            f.unlink()
    if MISTAKES_ARCHIVE_DIR.exists():
        for f in MISTAKES_ARCHIVE_DIR.glob("*.md"):
            f.unlink()


def test_mistake_filename_and_save():
    reset_mistakes()
    path = _save_mistake_note("What is my name?", "Bob", "No, I'm Alice.", "Alice")
    assert path.exists()
    assert path.parent == MISTAKES_DIR
    content = path.read_text(encoding="utf-8")
    assert "**Original question:** What is my name?" in content
    assert "**Wrong answer:** Bob" in content
    assert "**Correct answer:** Alice" in content
    assert _mistake_note_count() == 1


def test_digest_mistakes_and_archive():
    reset_mistakes()
    _save_mistake_note("Q1", "A1", "C1", "A1c")
    _save_mistake_note("Q2", "A2", "C2", "A2c")
    assert _mistake_note_count() == 2

    config = load_config()
    _, tokenizer = load(config["model_name"])
    system_prompt = f"You are {config['assistant_name']}. User is {config['user_name']}."
    digested = _digest_mistakes_to_training(tokenizer, system_prompt, boost=1)
    assert digested == 2
    assert _mistake_note_count() == 0
    assert len(list(MISTAKES_ARCHIVE_DIR.glob("*.md"))) == 2


def test_threshold_training_does_not_run_under_threshold():
    reset_mistakes()
    _save_mistake_note("Q", "A", "C", "Ac")

    config = load_config()
    config["learn"]["mistake_threshold"] = 5
    _, tokenizer = load(config["model_name"])
    system_prompt = f"You are {config['assistant_name']}. User is {config['user_name']}."

    # Build a minimal agent-like object for maybe_train_on_mistakes.
    class DummyAgent:
        def __init__(self, config, tokenizer, system_prompt):
            self.config = config
            self.tokenizer = tokenizer
            self.system_prompt = system_prompt
            self.planner = None

    agent = DummyAgent(config, tokenizer, system_prompt)
    trained = maybe_train_on_mistakes(config, tokenizer, system_prompt, agent)
    assert not trained
    assert _mistake_note_count() == 1


def test_learn_from_history_creates_note():
    reset_mistakes()
    config = load_config()
    config["learn"]["mistake_threshold"] = 5
    print(f"Loading {config['model_name']}...")
    model, tokenizer = load(config["model_name"])
    agent = AIAgent(config, model, tokenizer, False)

    # Simulate wrong answer then correction.
    agent.run("What is my name?")
    agent.run("No, I'm Alice.")

    path = learn_from_last_correction(agent)
    assert path is not None
    assert path.exists()
    assert _mistake_note_count() == 1
    content = path.read_text(encoding="utf-8")
    assert "No, I'm Alice." in content or "Alice" in content


if __name__ == "__main__":
    from test_utils import preserve_training_state

    with preserve_training_state():
        test_mistake_filename_and_save()
        print("test_mistake_filename_and_save passed")
        test_digest_mistakes_and_archive()
        print("test_digest_mistakes_and_archive passed")
        test_threshold_training_does_not_run_under_threshold()
        print("test_threshold_training_does_not_run_under_threshold passed")
        test_learn_from_history_creates_note()
        print("test_learn_from_history_creates_note passed")
        reset_mistakes()
    print("All deferred-learning tests passed.")
