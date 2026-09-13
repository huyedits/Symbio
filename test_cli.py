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
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", tmp_path / "worker_models.json")

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


def test_cli_skill_list_empty(isolated_cli, monkeypatch, capsys):
    monkeypatch.setattr("symbio.app.cli.load_config", _default_config)
    main(["skill"])
    out = capsys.readouterr().out.strip()
    assert "No skill adapters active" in out


def test_cli_archive_dry_run(isolated_cli, monkeypatch, capsys):
    monkeypatch.setattr("symbio.app.cli.load_config", _default_config)
    main(["archive", "--dry-run"])
    out = capsys.readouterr().out
    assert "Archive thresholds" in out
    assert "Would archive" in out


def test_cli_archive_list_empty(isolated_cli, monkeypatch, capsys):
    monkeypatch.setattr("symbio.app.cli.load_config", _default_config)
    main(["archive", "--list-archived"])
    out = capsys.readouterr().out
    assert "Archived notes: 0" in out
    assert "Archived adapters: 0" in out


# ---- `symb train <skill>` must use the same guarded path as everything else ----
#
# It called run_training directly, so a worker trained from the CLI got no
# golden baseline, no rollback on regression, and no journal update. Observed:
# a skill trained this way finished on disk while still listed as owed.

def test_symb_train_skill_goes_through_the_guarded_path(monkeypatch, tmp_path):
    from symbio import constants
    from symbio.app import cli as _cli
    from symbio.app import dispatch as _dispatch
    from symbio.app import config as _config

    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    catalog = tmp_path / "worker_models.json"
    catalog.write_text('{"s": {"model_name": "m/s", "role": "fix_wifi", "is_skill": true}}')
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", catalog)

    guarded, raw = [], []
    monkeypatch.setattr(_dispatch, "guarded_train_worker",
                        lambda role, cfg, **kw: (guarded.append((role, kw)), (True, "ok"))[1])
    monkeypatch.setattr(_cli, "run_training", lambda *a, **k: raw.append(k) or True)

    rc = _cli._cmd_train(_config.load_config(), skill="fix wifi", iters=7)

    assert rc == 0
    assert guarded == [("fix_wifi", {"iters": 7, "resume": False})]
    assert raw == [], "the unguarded trainer must not be called for a worker"


def test_symb_train_without_a_skill_still_trains_the_headmaster(monkeypatch, tmp_path):
    from symbio.app import cli as _cli
    from symbio.app import dispatch as _dispatch
    from symbio.app import config as _config

    guarded, raw = [], []
    monkeypatch.setattr(_dispatch, "guarded_train_worker",
                        lambda *a, **k: guarded.append(a) or (True, "ok"))
    monkeypatch.setattr(_cli, "run_training", lambda *a, **k: raw.append(k) or True)

    assert _cli._cmd_train(_config.load_config()) == 0
    assert raw and guarded == [], "the headmaster has its own path"
