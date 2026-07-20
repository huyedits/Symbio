"""Tests for the symbio/symb command-line interface."""

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from symbio import constants
from symbio.app.cli import main


def _default_config():
    from symbio.app.config import DEFAULT_CONFIG
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["telegram"]["bot_token"] = "secret-token"
    return config


@pytest.fixture
def isolated_cli(tmp_path, monkeypatch):
    """Point CLI file paths into tmp_path so tests do not touch the real repo."""
    monkeypatch.setattr(constants, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(constants, "GATEWAY_PID_FILE", tmp_path / "gateway.pid")

    # Patch load_config inside cli.py so it does not read the real config.json.
    import symbio.app.cli as cli_module
    monkeypatch.setattr(cli_module, "load_config", _default_config)


def test_cli_config_show(isolated_cli, capsys):
    main(["config"])
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-0.6B" in out
    assert "***REDACTED***" in out  # telegram token is redacted even when empty


def test_cli_config_get(isolated_cli, capsys):
    main(["config", "get", "agent.temperature"])
    out = capsys.readouterr().out.strip()
    assert out == "0.7"


def test_cli_config_get_unknown(isolated_cli, capsys):
    main(["config", "get", "agent.nonexistent"])
    out = capsys.readouterr().out.strip()
    assert "Unknown config key" in out


def test_cli_config_set(isolated_cli, capsys):
    main(["config", "set", "agent.temperature", "0.9"])
    out = capsys.readouterr().out.strip()
    assert "Set agent.temperature = 0.9" in out

    # Value should be persisted into the temp config.json.
    saved = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
    assert saved["agent"]["temperature"] == 0.9


def test_cli_gateway_status(isolated_cli, capsys):
    main(["gateway", "status"])
    out = capsys.readouterr().out
    assert "Gateway running: no" in out
    assert "Bot token configured: yes" in out
    assert "Allowed chat IDs: 0" in out
    assert "Model:" in out


def test_cli_legacy_telegram_flag(isolated_cli, monkeypatch, capsys):
    mock_bot_class = MagicMock()
    mock_bot = mock_bot_class.return_value

    monkeypatch.setattr("symbio.app.telegram.TelegramBot", mock_bot_class)
    monkeypatch.setattr(
        "symbio.app.cli.get_telegram_token",
        lambda config: "test-token",
    )

    main(["--telegram"])

    mock_bot_class.assert_called_once()
    mock_bot.run.assert_called_once()


def test_cli_legacy_train_flag(isolated_cli, monkeypatch, capsys):
    mock_train = MagicMock()
    monkeypatch.setattr("symbio.app.cli.run_training", mock_train)

    main(["--train"])

    mock_train.assert_called_once()
