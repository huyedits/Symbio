"""Tests for the interactive setup wizard."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from symbio import constants
from symbio.app import setup
from symbio.app.config import DEFAULT_CONFIG


@pytest.fixture
def isolated_setup(tmp_path, monkeypatch):
    """Point config.json into tmp_path so wizard tests don't touch the repo."""
    monkeypatch.setattr(constants, "CONFIG_FILE", tmp_path / "config.json")


def test_setup_wizard_writes_config_and_sets_names(isolated_setup):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    inputs = iter(["", "Alice", "Friday", "", "fast", "n", "y", "y", "n", "y", ""])
    result = setup.run_setup_wizard(
        cfg,
        input_fn=lambda prompt="": next(inputs),
        output_fn=lambda s: None,
    )
    assert result["assistant_name"] == "Friday"
    assert result["user_name"] == "Alice"
    assert result["agent"]["speed_mode"] == "fast"
    assert constants.CONFIG_FILE.exists()
    saved = json.loads(constants.CONFIG_FILE.read_text())
    assert saved["assistant_name"] == "Friday"
    assert saved["user_name"] == "Alice"
    assert saved["first_run"] is False


def test_setup_wizard_skip_leaves_defaults(isolated_setup):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    inputs = iter(["s"])
    result = setup.run_setup_wizard(
        cfg,
        input_fn=lambda prompt="": next(inputs),
        output_fn=lambda s: None,
    )
    assert result["assistant_name"] == DEFAULT_CONFIG["assistant_name"]
    assert constants.CONFIG_FILE.exists()
    saved = json.loads(constants.CONFIG_FILE.read_text())
    assert saved["first_run"] is False


def test_setup_wizard_parses_multiple_telegram_chat_ids(isolated_setup):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    inputs = iter([
        "", "Bob", "Jarvis", "", "balanced", "n", "n", "n", "y",
        "my-token", "123456789, 987654321 ,bad, 111",
        "y", "y",
    ])
    result = setup.run_setup_wizard(
        cfg,
        input_fn=lambda prompt="": next(inputs),
        output_fn=lambda s: None,
    )
    assert result["telegram"]["enabled"] is True
    assert result["telegram"]["bot_token"] == "my-token"
    assert result["telegram"]["allowed_chat_ids"] == [123456789, 987654321, 111]


def test_setup_wizard_custom_model_repo(isolated_setup):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    inputs = iter(["", "User", "Bot", "0", "custom/model-7b", "balanced", "n", "y", "n", "n", "y", ""])
    result = setup.run_setup_wizard(
        cfg,
        input_fn=lambda prompt="": next(inputs),
        output_fn=lambda s: None,
    )
    assert result["model_name"] == "custom/model-7b"


def test_is_first_run_missing_config(isolated_setup):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    assert setup.is_first_run(cfg) is True


def test_is_first_run_after_setup(isolated_setup):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["assistant_name"] = "A"
    cfg["user_name"] = "U"
    cfg["first_run"] = False
    constants.CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    assert setup.is_first_run(cfg) is False
