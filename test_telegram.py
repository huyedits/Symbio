"""Tests for the Telegram bot front-end."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from symbio.app import config as app_config, tooling
from symbio.app.chat import ChatSession
from symbio.app.telegram import (
    TelegramBot,
    _StreamSender,
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


def _run_loop_in_thread(loop: asyncio.AbstractEventLoop) -> threading.Thread:
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return t


@pytest.fixture
def stream_loop():
    loop = asyncio.new_event_loop()
    t = _run_loop_in_thread(loop)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)
    if not loop.is_closed():
        loop.close()


def test_stream_sender_sends_first_visible_chunk_immediately(stream_loop):
    """The first safe text is sent as a new message before generation ends."""
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    bot.edit_message_text = AsyncMock()

    sender = _StreamSender(123, bot, stream_loop, first_chunk_timeout_ms=5000)
    sender.feed("Hello")
    sender.feed(" world")
    sender.finish()

    assert bot.send_message.await_count == 1
    assert "Hello" in bot.send_message.await_args.kwargs["text"]
    # The second chunk should edit the same message.
    assert bot.edit_message_text.await_count >= 1
    assert "Hello world" in bot.edit_message_text.await_args.kwargs["text"]


def test_stream_sender_sends_placeholder_on_slow_first_chunk(stream_loop):
    """If no visible text arrives within the timeout, a placeholder is shown."""
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=8))
    bot.edit_message_text = AsyncMock()

    sender = _StreamSender(123, bot, stream_loop, first_chunk_timeout_ms=50)
    time.sleep(0.15)  # Wait longer than the placeholder timeout.
    assert bot.send_message.await_count == 1
    assert bot.send_message.await_args.kwargs["text"] == _StreamSender._PLACEHOLDER

    # Real text then replaces the placeholder.
    sender.feed("Here is the answer")
    sender.finish()
    assert bot.edit_message_text.await_count >= 1
    assert "Here is the answer" in bot.edit_message_text.await_args.kwargs["text"]


def test_stream_sender_strips_tool_tags_from_stream(stream_loop):
    """Tool tags are held back and never shown as raw markup."""
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=9))
    bot.edit_message_text = AsyncMock()

    sender = _StreamSender(123, bot, stream_loop, first_chunk_timeout_ms=5000)
    sender.feed("Checking <cmd>date</cmd> now.")
    sender.finish()

    text = bot.send_message.await_args.kwargs["text"]
    assert "Checking" in text
    assert "<cmd>" not in text
    assert "date" not in text


def test_parse_tools_open_chrome_command():
    """The agent can emit a macOS open-app command when asked to launch Chrome."""
    reply = "<cmd>open -a 'Google Chrome'</cmd> Opening Chrome for you, Huy."
    tools = tooling.parse_tools(reply)
    assert len(tools) == 1
    name, params = tools[0]
    assert name == "run_command"
    assert params["cmd"] == "open -a 'Google Chrome'"


def test_parse_tools_browser_press():
    """The agent can press a key in the browser instead of inventing a shell command."""
    reply = "<press>down</press> Pressing the down arrow key."
    tools = tooling.parse_tools(reply)
    assert len(tools) == 1
    name, params = tools[0]
    assert name == "browser_press"
    assert params["key"] == "down"
