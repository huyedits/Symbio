"""Tests for the Telegram bot front-end."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from symbio.app import config as app_config
from symbio.app.chat import ChatSession
from symbio.app.telegram import (
    TelegramBot,
    _chat_tool_groups,
    _chat_tool_lock,
    _prepare_for_telegram,
    _split_message,
)


@pytest.fixture
def base_config(tmp_path, monkeypatch):
    """Isolated config and paths for Telegram tests."""
    from symbio import constants
    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "training_data" / "train.jsonl")
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "training_data" / "valid.jsonl")
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(constants, "MISTAKES_DIR", tmp_path / "notes" / "mistakes")
    monkeypatch.setattr(constants, "MISTAKES_ARCHIVE_DIR", tmp_path / "notes" / "mistakes" / "archive")
    monkeypatch.setattr(constants, "SANDBOX_DIR", tmp_path / "sandbox")
    monkeypatch.setattr(constants, "SCREENSHOTS_DIR", tmp_path / "screenshots")
    monkeypatch.setattr(constants, "DIGEST_MANIFEST", tmp_path / "training_data" / "digest_manifest.json")
    monkeypatch.setattr(constants, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(constants, "PROMPT_FILE", tmp_path / "prompt.md")
    monkeypatch.setattr(constants, "CRON_FILE", tmp_path / "cron_jobs.json")
    monkeypatch.setattr(constants, "MEMORY_FILE", tmp_path / "agent_memory.md")
    monkeypatch.setattr(constants, "PROFILE_FILE", tmp_path / "user_profile.md")
    monkeypatch.setattr(constants, "SESSIONS_DIR", tmp_path / "sessions")

    config = app_config.load_config()
    config["telegram"] = {
        "enabled": True,
        "bot_token": "fake-token",
        "allowed_chat_ids": [123456],
        "confirm_dangerous": True,
    }
    return config


@pytest.fixture(autouse=True)
def _clear_chat_tool_groups():
    with _chat_tool_lock:
        _chat_tool_groups.clear()
    yield
    with _chat_tool_lock:
        _chat_tool_groups.clear()


class FakeApplication:
    """Minimal stand-in for telegram.ext.Application."""

    def __init__(self):
        self.bot = MagicMock()
        self.bot.send_message = AsyncMock()
        self.updater = MagicMock()
        self.updater.start_polling = AsyncMock()
        self.updater.stop = AsyncMock()
        self._handlers = []

    async def initialize(self):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    async def shutdown(self):
        pass

    def add_handler(self, handler):
        self._handlers.append(handler)


def test_allowlist_allows_configured_chat(base_config):
    bot = TelegramBot(base_config)
    assert bot._is_allowed(123456)
    assert not bot._is_allowed(999999)


@pytest.mark.asyncio
async def test_start_replies_to_unauthorized_chat(base_config):
    bot = TelegramBot(base_config)
    update = MagicMock()
    update.effective_chat.id = 999999
    update.effective_user.first_name = "Stranger"
    update.message.reply_text = AsyncMock()
    await bot._cmd_start(update, None)
    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.await_args[0][0]
    assert "private" in text.lower()
    assert "999999" in text


@pytest.mark.asyncio
async def test_start_welcomes_authorized_chat(base_config):
    bot = TelegramBot(base_config)
    update = MagicMock()
    update.effective_chat.id = 123456
    update.effective_user.first_name = "Owner"
    update.message.reply_text = AsyncMock()
    await bot._cmd_start(update, None)
    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.await_args[0][0]
    assert "Hi Owner" in text


def test_prepare_for_telegram_strips_ansi():
    raw = "\x1b[32mHello\x1b[0m \x1b[1mworld\x1b[0m"
    assert _prepare_for_telegram(raw) == "Hello world"


def test_split_message_respects_limit():
    text = "\n".join(f"line {i}" for i in range(1000))
    chunks = _split_message(text, limit=200)
    for chunk in chunks[:-1]:
        assert len(chunk) <= 200
    assert "".join(chunks) == text


def test_config_show_redacts_telegram_token(base_config):
    shown = app_config.config_show(base_config)
    assert "fake-token" not in shown
    assert "***REDACTED***" in shown


def test_chat_session_telegram_confirmation_prompts(base_config):
    """Tools in _TELEGRAM_CONFIRM_TOOLS ask for approval when confirm_fn is set."""
    calls = []

    def fake_confirm(prompt):
        calls.append(prompt)
        return False

    # We don't need a real model/tokenizer for this test.
    session = ChatSession.__new__(ChatSession)
    session.confirm_fn = fake_confirm
    session.config = base_config

    result = session._execute_tool("execute_code", {"code": "print(1+1)"})
    assert len(calls) == 1
    assert "Python code" in calls[0]
    assert "not approved" in result

    result = session._execute_tool("run_command", {"cmd": "ssh root@209.38.82.54"})
    assert len(calls) == 2
    assert "shell command" in calls[1]
    assert "ssh root@209.38.82.54" in calls[1]
    assert "not approved" in result

    result = session._execute_tool("digest_notes", {})
    assert len(calls) == 3
    assert "notes" in calls[2]
    assert "not approved" in result


def test_tools_menu_defaults(base_config):
    base_config["tools"] = {"enabled_groups": ["memory", "notes", "terminal"]}
    bot = TelegramBot(base_config)
    text, keyboard = bot._tools_menu(123456)
    assert text == "Tap a group to toggle it:"
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "☑ Memory & Profile" in labels
    assert "☑ Notes & Skills" in labels
    assert "☑ Terminal" in labels
    assert "☐ Browser" in labels


def test_tools_menu_toggle(base_config):
    base_config["tools"] = {"enabled_groups": ["memory", "notes", "terminal"]}
    bot = TelegramBot(base_config)
    bot._tools_menu(123456)
    with _chat_tool_lock:
        assert _chat_tool_groups[123456] == {"memory", "notes", "terminal"}

    with _chat_tool_lock:
        _chat_tool_groups[123456].discard("terminal")

    text, keyboard = bot._tools_menu(123456)
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "☐ Terminal" in labels
    assert "☑ Memory & Profile" in labels


@pytest.mark.asyncio
async def test_handle_tools_callback_toggles(base_config):
    base_config["tools"] = {"enabled_groups": ["memory", "notes", "terminal"]}
    bot = TelegramBot(base_config)
    bot._tools_menu(123456)

    query = MagicMock()
    query.message.chat.id = 123456
    query.edit_message_text = AsyncMock()

    await bot._handle_tools_callback(query, "tools:terminal")
    with _chat_tool_lock:
        assert "terminal" not in _chat_tool_groups[123456]
    query.edit_message_text.assert_awaited_once()

    query.edit_message_text.reset_mock()
    await bot._handle_tools_callback(query, "tools:terminal")
    with _chat_tool_lock:
        assert "terminal" in _chat_tool_groups[123456]


@pytest.mark.asyncio
async def test_handle_tools_callback_close(base_config):
    bot = TelegramBot(base_config)
    query = MagicMock()
    query.message.chat.id = 123456
    query.edit_message_text = AsyncMock()

    await bot._handle_tools_callback(query, "tools:close")
    text = query.edit_message_text.await_args[0][0]
    assert text == "Tool settings closed."
