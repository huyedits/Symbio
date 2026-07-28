"""Tests for the health verification layer in symbio/app/health.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from symbio import constants
from symbio.app import health


@pytest.fixture
def isolated_health_env(tmp_path, monkeypatch):
    """Point health.py file paths into a temporary tree."""
    notes_dir = tmp_path / "notes"
    notes_archive_dir = notes_dir / "archive"
    adapter_dir = tmp_path / "adapters"
    data_dir = tmp_path / "training_data"
    log_dir = tmp_path / "logs"
    sandbox_dir = tmp_path / "sandbox"
    sessions_dir = tmp_path / "sessions"
    screenshots_dir = tmp_path / "screenshots"
    mistakes_dir = notes_dir / "mistakes"
    mistakes_archive_dir = mistakes_dir / "archive"
    prompt_file = tmp_path / "prompt.md"
    prompt_default_file = tmp_path / "prompt.md.default"
    cron_file = tmp_path / "cron_jobs.json"
    worker_models_file = tmp_path / "worker_models.json"

    for d in (
        notes_dir,
        notes_archive_dir,
        adapter_dir,
        data_dir,
        log_dir,
        sandbox_dir,
        sessions_dir,
        screenshots_dir,
        mistakes_dir,
        mistakes_archive_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(constants, "NOTES_ARCHIVE_DIR", notes_archive_dir)
    monkeypatch.setattr(constants, "ADAPTER_DIR", adapter_dir)
    monkeypatch.setattr(constants, "ADAPTER_ARCHIVE_DIR", tmp_path / "adapters_archive")
    monkeypatch.setattr(constants, "DATA_DIR", data_dir)
    monkeypatch.setattr(constants, "TRAIN_FILE", data_dir / "train.jsonl")
    monkeypatch.setattr(constants, "VALID_FILE", data_dir / "valid.jsonl")
    monkeypatch.setattr(constants, "LOG_DIR", log_dir)
    monkeypatch.setattr(constants, "SANDBOX_DIR", sandbox_dir)
    monkeypatch.setattr(constants, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(constants, "SCREENSHOTS_DIR", screenshots_dir)
    monkeypatch.setattr(constants, "MISTAKES_DIR", mistakes_dir)
    monkeypatch.setattr(constants, "MISTAKES_ARCHIVE_DIR", mistakes_archive_dir)
    monkeypatch.setattr(constants, "PROMPT_FILE", prompt_file)
    monkeypatch.setattr(constants, "PROMPT_DEFAULT_FILE", prompt_default_file)
    monkeypatch.setattr(constants, "CRON_FILE", cron_file)
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", worker_models_file)

    yield tmp_path


def _minimal_config():
    return {
        "model_name": "dummy-model",
        "assistant_name": "Symbio",
        "user_name": "Tester",
        "agent": {
            "temperature": 0.1,
            "top_p": 0.9,
            "tool_use_temperature": 0.05,
            "max_tool_rounds": 3,
            "history_limit": 40,
            "max_history_chars": 20000,
            "cron_poll_seconds": 60,
            "auto_archive_poll_seconds": 300,
        },
        "rag": {"enabled": True, "sources": ["notes"], "top_k": 2, "max_context_tokens": 100},
        "memory": {"enabled": True, "nudge_interval": 0, "memory_char_limit": 10000, "profile_char_limit": 10000},
        "learn": {"enabled": True},
        "tools": {"enabled_groups": ["memory"]},
    }


def test_required_dirs_auto_created(isolated_health_env):
    # Pick a directory that should be recreated and remove it.
    target = constants.SCREENSHOTS_DIR
    import shutil
    shutil.rmtree(target)
    config = _minimal_config()
    result = health._check_required_dirs(config)
    assert result.ok
    assert result.auto_fixed
    assert target.exists()


def test_training_data_auto_creates_empty_jsonl(isolated_health_env):
    config = _minimal_config()
    result = health._check_training_data(config)
    assert result.ok
    assert result.auto_fixed
    assert constants.TRAIN_FILE.exists()
    assert constants.VALID_FILE.exists()


def test_memory_seeds_identity_notes_when_empty(isolated_health_env):
    config = _minimal_config()
    result = health._check_memory(config)
    assert result.ok
    assert result.auto_fixed
    notes = list(constants.NOTES_DIR.glob("*.md"))
    assert len(notes) == 2
    bodies = " ".join(n.read_text() for n in notes)
    assert "Symbio" in bodies and "Tester" in bodies


def test_cron_creates_missing_jobs_file(isolated_health_env):
    config = _minimal_config()
    config["tools"]["enabled_groups"] = ["cron"]
    result = health._check_cron(config)
    assert result.ok
    assert result.auto_fixed
    assert constants.CRON_FILE.exists()
    assert json.loads(constants.CRON_FILE.read_text(encoding="utf-8")) == []


def test_cron_repairs_invalid_json(isolated_health_env):
    config = _minimal_config()
    config["tools"]["enabled_groups"] = ["cron"]
    constants.CRON_FILE.write_text("not json", encoding="utf-8")
    result = health._check_cron(config)
    assert result.ok
    assert result.auto_fixed
    assert json.loads(constants.CRON_FILE.read_text(encoding="utf-8")) == []


def test_disk_space_error_below_1gb(isolated_health_env, monkeypatch):
    class FakeUsage:
        free = 512 * (1024 ** 2)  # 512 MB
        total = 1024 * (1024 ** 3)

    monkeypatch.setattr("shutil.disk_usage", lambda p: FakeUsage())
    result = health._check_disk(_minimal_config())
    assert not result.ok
    assert result.severity == "error"


def test_disk_space_warning_below_5gb(isolated_health_env, monkeypatch):
    class FakeUsage:
        free = 3 * (1024 ** 3)
        total = 1024 * (1024 ** 3)

    monkeypatch.setattr("shutil.disk_usage", lambda p: FakeUsage())
    result = health._check_disk(_minimal_config())
    assert result.ok
    assert result.severity == "warning"


def test_disk_space_healthy_above_5gb(isolated_health_env, monkeypatch):
    class FakeUsage:
        free = 10 * (1024 ** 3)
        total = 1024 * (1024 ** 3)

    monkeypatch.setattr("shutil.disk_usage", lambda p: FakeUsage())
    result = health._check_disk(_minimal_config())
    assert result.ok
    assert result.severity != "warning"


def test_verify_report_structure(isolated_health_env):
    config = _minimal_config()
    # Patch model load so the full verification doesn't require MLX.
    with patch.object(health, "_check_model_load", return_value=health._CheckResult("model_load", True, message="ok")):
        report = health.verify_enabled_features(config, verbose=False, skip_model_load=True)

    assert "healthy" in report
    assert "all_ok" in report
    assert "auto_fixed_count" in report
    assert "errors_count" in report
    assert "warnings_count" in report
    assert isinstance(report["checks"], list)
    assert any(c["name"] == "required_dirs" for c in report["checks"])


def test_summary_for_agent_all_ok():
    report = {"all_ok": True, "errors": [], "warnings": []}
    assert "passed" in health.summary_for_agent(report)


def test_summary_for_agent_with_error():
    report = {"all_ok": False, "errors": [{"name": "disk"}], "warnings": []}
    summary = health.summary_for_agent(report)
    assert "disk" in summary
    assert "human attention" in summary


def test_chat_session_persists_health_report(isolated_health_env, monkeypatch):
    from symbio.app import chat

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False, enable_thinking=False):
            return " ".join(f"{m['role']}: {m['content']}" for m in messages)
        def encode(self, text, add_special_tokens=True):
            return text.split(" ")

    chat.load = lambda *a, **k: (object(), FakeTokenizer())
    monkeypatch.setattr(chat.health, "verify_enabled_features",
                        lambda *a, **k: {"healthy": True, "all_ok": True,
                                         "auto_fixed_count": 0,
                                         "errors_count": 0, "warnings_count": 0,
                                         "errors": [], "warnings": [],
                                         "checks": [], "auto_fixed": []})
    monkeypatch.setattr(chat.training, "seed_training_data",
                        lambda tokenizer, system_prompt, config: None)
    monkeypatch.setattr(chat.training, "clean_training_duplicates",
                        lambda max_copies=3: None)
    try:
        session = chat.ChatSession(
            _minimal_config(),
            model=object(),
            tokenizer=FakeTokenizer(),
            adapter_loaded=False,
            input_fn=lambda p: "",
            output_fn=lambda t: None,
            generate_fn=lambda *a, **k: "ok",
        )
        per_session = constants.SESSIONS_DIR / f"{session.session_id}_health.json"
        latest = constants.SESSIONS_DIR / "latest_health.json"
        assert per_session.exists()
        assert latest.exists()
        data = json.loads(per_session.read_text(encoding="utf-8"))
        assert data["healthy"] is True
    finally:
        chat.load = chat.__dict__.get("load")
