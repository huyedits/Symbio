"""Tests for idle-adapter tracking: a saved LoRA adapter that exists on disk
but wasn't loaded into the current session (e.g. after switching to an
incompatible model) gets a removal reminder once it has sat unused past
learn.adapter_idle_days. Declining or asking to keep it both just reset the
grace period — nothing is ever deleted without an explicit yes."""

import json
from datetime import datetime, timedelta

from symbio import constants
from symbio.app import chat, training
from symbio.app import config as app_config


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        return "prompt"


def _write_adapter():
    constants.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    (constants.ADAPTER_DIR / "adapter_config.json").write_text("{}")
    (constants.ADAPTER_DIR / "adapters.safetensors").write_bytes(b"weights")


def _set_last_used(days_ago: int):
    path = constants.ADAPTER_DIR / "last_used.json"
    when = datetime.now() - timedelta(days=days_ago)
    path.write_text(json.dumps({"last_used": when.isoformat()}), encoding="utf-8")


def _make_session(config, monkeypatch, adapter_loaded, input_fn, output_calls):
    monkeypatch.setattr(chat, "load", lambda *a, **k: (object(), FakeTokenizer()))
    return chat.ChatSession(
        config, model=object(), tokenizer=FakeTokenizer(), adapter_loaded=adapter_loaded,
        input_fn=input_fn, output_fn=lambda t: output_calls.append(t),
        generate_fn=lambda *a, **k: "unused",
    )


# ---- training.py helpers ----

def test_adapter_last_used_none_when_untracked(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    assert training.adapter_last_used() is None


def test_mark_and_read_adapter_last_used(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()
    before = datetime.now()
    training.mark_adapter_used()
    after = training.adapter_last_used()
    assert after is not None
    assert before <= after <= datetime.now() + timedelta(seconds=1)


def test_remove_adapter_clears_files(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()
    training.mark_adapter_used()
    training.remove_adapter()
    assert constants.ADAPTER_DIR.exists()  # recreated empty
    assert not (constants.ADAPTER_DIR / "adapter_config.json").exists()
    assert not (constants.ADAPTER_DIR / "last_used.json").exists()


# ---- ChatSession._check_idle_adapter ----

def test_active_session_marks_adapter_used_without_prompting(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()

    config = app_config.load_config()
    output_calls: list[str] = []

    def fail_input(prompt=""):
        raise AssertionError("should not prompt while the adapter is actively in use")

    _make_session(config, monkeypatch, adapter_loaded=True,
                  input_fn=fail_input, output_calls=output_calls)

    assert training.adapter_last_used() is not None


def test_stale_untracked_adapter_starts_tracking_without_prompting(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()  # present, but never marked used, and not loaded this session

    config = app_config.load_config()
    output_calls: list[str] = []

    def fail_input(prompt=""):
        raise AssertionError("first sighting should just start tracking, not prompt")

    _make_session(config, monkeypatch, adapter_loaded=False,
                  input_fn=fail_input, output_calls=output_calls)

    assert training.adapter_last_used() is not None


def test_idle_below_threshold_does_not_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()
    _set_last_used(5)

    config = app_config.load_config()
    config["learn"]["adapter_idle_days"] = 30
    output_calls: list[str] = []

    def fail_input(prompt=""):
        raise AssertionError("should not prompt below the idle threshold")

    _make_session(config, monkeypatch, adapter_loaded=False,
                  input_fn=fail_input, output_calls=output_calls)


def test_idle_past_threshold_prompts_and_removes_on_yes(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()
    _set_last_used(45)

    config = app_config.load_config()
    config["learn"]["adapter_idle_days"] = 30
    output_calls: list[str] = []
    prompts_seen = []

    def fake_input(prompt=""):
        prompts_seen.append(prompt)
        return "y"

    session = _make_session(config, monkeypatch, adapter_loaded=False,
                            input_fn=fake_input, output_calls=output_calls)

    assert len(prompts_seen) == 1
    assert "45 day" in prompts_seen[0]
    assert not (constants.ADAPTER_DIR / "adapter_config.json").exists()
    assert any("Removed the unused adapter" in t for t in output_calls)


def test_idle_past_threshold_keeps_on_decline_and_resets_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()
    _set_last_used(45)

    config = app_config.load_config()
    config["learn"]["adapter_idle_days"] = 30
    output_calls: list[str] = []

    session = _make_session(config, monkeypatch, adapter_loaded=False,
                            input_fn=lambda prompt="": "n", output_calls=output_calls)

    assert (constants.ADAPTER_DIR / "adapter_config.json").exists()
    assert any("Keeping the adapter" in t for t in output_calls)
    idle_days = (datetime.now() - training.adapter_last_used()).days
    assert idle_days == 0


def test_idle_past_threshold_keeps_when_user_says_keep(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()
    _set_last_used(45)

    config = app_config.load_config()
    config["learn"]["adapter_idle_days"] = 30
    output_calls: list[str] = []

    _make_session(config, monkeypatch, adapter_loaded=False,
                  input_fn=lambda prompt="": "keep", output_calls=output_calls)

    assert (constants.ADAPTER_DIR / "adapter_config.json").exists()
    assert any("Keeping the adapter" in t for t in output_calls)


def test_reminder_disabled_never_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter()
    _set_last_used(90)

    config = app_config.load_config()
    config["learn"]["adapter_idle_reminder_enabled"] = False
    output_calls: list[str] = []

    def fail_input(prompt=""):
        raise AssertionError("reminder is disabled; must not prompt")

    _make_session(config, monkeypatch, adapter_loaded=False,
                  input_fn=fail_input, output_calls=output_calls)
