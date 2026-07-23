"""Telegram bot front-end for Symbio.

Runs the same ChatSession agent loop, but bridges input/output and
yes/no confirmations to Telegram messages and inline keyboards.

MLX is stream/thread-sensitive: the model must be loaded and used on the
same thread. TelegramBot therefore keeps a dedicated MLX inference thread
that owns the model. Each ChatSession runs in its own worker thread and
delegates generate() calls to the inference thread.
"""

import asyncio
import concurrent.futures
import queue
import re
import threading
import time
import uuid
from typing import Any

from mlx_lm import generate, load

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from symbio import constants
from symbio.app.chat import ChatSession
from symbio.app.config import get_telegram_token
from symbio.app import tooling

# Per-chat state shared between the async Telegram handlers and the
# synchronous agent worker threads.
_sessions: dict[int, ChatSession] = {}
_session_lock = threading.Lock()

# Per-chat tool-group overrides (independent copies of config["tools"]["enabled_groups"]).
_chat_tool_groups: dict[int, set[str]] = {}
_chat_tool_lock = threading.Lock()

# Tool groups offered in the /tools menu.
_TOOL_MENU_GROUPS = [
    ("memory", "Memory & Profile"),
    ("notes", "Notes & Skills"),
    ("terminal", "Terminal"),
    ("code", "Python Code"),
    ("web_search", "Web Search"),
    ("browser", "Browser"),
    ("digest", "Digest Notes"),
    ("train", "Train Adapter"),
    ("cron", "Cron Jobs"),
    ("config", "Config Changes"),
]

# Pending requests from the agent thread back to Telegram.
_pending_inputs: dict[int, "_InputRequest"] = {}
_pending_confirmations: dict[str, "_ConfirmRequest"] = {}
_pending_lock = threading.Lock()


class _InputRequest:
    def __init__(self):
        self.event = threading.Event()
        self.value: str = ""


class _ConfirmRequest:
    def __init__(self):
        self.event = threading.Event()
        self.value: bool = False


class _StreamSender:
    """Deliver a streaming assistant reply to Telegram with throttled edits.

    The model emits text token-by-token. Known tool tags are stripped live,
    and safe text is sent as soon as possible. If no visible text arrives
    within `first_chunk_timeout_ms`, a "thinking…" placeholder is sent so the
    user knows the bot is alive; real text then replaces it.
    """

    _PLACEHOLDER = "thinking…"
    _EDIT_INTERVAL = 0.6
    _BATCH_SIZE = 300
    _MESSAGE_LIMIT = 3800
    _SENTENCE_END = frozenset(".!?\n")

    def __init__(self, chat_id: int, bot, loop: asyncio.AbstractEventLoop,
                 first_chunk_timeout_ms: float = 1500.0):
        self.chat_id = chat_id
        self.bot = bot
        self.loop = loop
        self.timeout_ms = first_chunk_timeout_ms
        self.stripper = tooling.StreamingStripper()
        self.buffer = ""
        self.sent_text = ""
        self.message_id: int | None = None
        self._placeholder_timer: threading.Timer | None = None
        self._last_edit_time = 0.0
        self._lock = threading.Lock()
        self._maybe_start_placeholder()

    def feed(self, text: str) -> None:
        """Consume a chunk of generated text."""
        if not text:
            return
        with self._lock:
            safe = self.stripper.feed(text)
            if not safe:
                self._maybe_start_placeholder()
                return
            self.buffer += safe
            self._cancel_placeholder()
            self._ensure_message()
            self._flush_if_ready()

    def finish(self) -> None:
        """Finalize the streamed message. Call after generation ends."""
        with self._lock:
            self._cancel_placeholder()
            tail = self.stripper.finish()
            if tail:
                self.buffer += tail
            self._flush(force=True)
            # Reset so a later tool round starts a fresh message.
            self.message_id = None
            self.sent_text = ""

    def _maybe_start_placeholder(self) -> None:
        if self.message_id is not None or self._placeholder_timer is not None:
            return
        self._placeholder_timer = threading.Timer(
            self.timeout_ms / 1000.0, self._send_placeholder)
        self._placeholder_timer.daemon = True
        self._placeholder_timer.start()

    def _send_placeholder(self) -> None:
        with self._lock:
            if self.message_id is not None:
                return
            self._placeholder_timer = None
        self._send_message(self._PLACEHOLDER)

    def _cancel_placeholder(self) -> None:
        timer = self._placeholder_timer
        if timer is not None:
            self._placeholder_timer = None
            try:
                timer.cancel()
            except Exception:
                pass

    def _ensure_message(self) -> None:
        if self.message_id is None:
            text = self.buffer
            self.buffer = ""
            self._send_message(text)

    def _flush_if_ready(self) -> None:
        now = time.monotonic()
        if len(self.buffer) > self._BATCH_SIZE:
            self._flush(force=True)
        elif self.buffer and (now - self._last_edit_time) > self._EDIT_INTERVAL:
            if len(self.buffer) > 80 or self.buffer[-1] in self._SENTENCE_END:
                self._flush(force=False)

    def _flush(self, force: bool) -> None:
        if not self.buffer:
            return
        text = self.sent_text + self.buffer
        if self.message_id is None:
            self._send_message(text)
        else:
            self._edit_message(text)
        self.sent_text = text
        self.buffer = ""
        self._last_edit_time = time.monotonic()

    def _send_message(self, text: str) -> None:
        text = _prepare_for_telegram(text)
        if not text:
            return
        if self.loop is None or self.loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.bot.send_message(chat_id=self.chat_id, text=text[:4096]),
                self.loop,
            )
            msg = future.result(timeout=10)
            self.message_id = msg.message_id
            self.sent_text = text
        except Exception as e:
            print(f"[Telegram] Failed to send stream message to {self.chat_id}: {e}")

    def _edit_message(self, text: str) -> None:
        if self.message_id is None:
            return
        text = _prepare_for_telegram(text)
        if not text:
            return
        if self.loop is None or self.loop.is_closed():
            return
        # If the message has grown very long, start a new one instead of
        # editing a giant block.
        if len(text) > self._MESSAGE_LIMIT and len(self.sent_text) < self._MESSAGE_LIMIT:
            overflow = text[self._MESSAGE_LIMIT:]
            text = text[:self._MESSAGE_LIMIT]
            self._send_message(overflow)
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.message_id,
                    text=text[:4096],
                ),
                self.loop,
            )
            future.result(timeout=10)
        except Exception as e:
            print(f"[Telegram] Failed to edit stream message {self.message_id}: {e}")


class TelegramBot:
    """Telegram long-polling wrapper around a shared ChatSession model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self.adapter_loaded: bool = False
        self.application: Application | None = None
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._typing_tasks: dict[int, asyncio.Future] = {}

        # MLX inference thread: loads the model once and services generate() calls.
        self._infer_queue: queue.Queue = queue.Queue()
        self._infer_thread: threading.Thread | None = None
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _allowed_chat_ids(self) -> set[int]:
        return set(self.config.get("telegram", {}).get("allowed_chat_ids", []))

    def _is_allowed(self, chat_id: int) -> bool:
        return chat_id in self._allowed_chat_ids()

    def _load_model(self) -> None:
        """Load the model on the inference thread and cache it there."""
        try:
            self.model, self.tokenizer = load(
                self.config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
            )
            self.adapter_loaded = True
        except Exception as e:
            print(f"Could not load adapter: {e}")
            print("Falling back to base model...")
            self.model, self.tokenizer = load(self.config["model_name"])
            self.adapter_loaded = False

    def _inference_loop(self) -> None:
        """Background thread that owns the MLX model and runs generate()."""
        self._load_model()
        while True:
            item = self._infer_queue.get()
            if item is None:
                break
            future, args, kwargs = item
            try:
                result = generate(*args, **kwargs)
            except Exception as e:
                future.set_exception(e)
            else:
                future.set_result(result)

    def _generate_on_infer_thread(self, *args, **kwargs):
        """Callable passed to ChatSession; blocks the chat thread until generate finishes."""
        future = concurrent.futures.Future()
        self._infer_queue.put((future, args, kwargs))
        return future.result()

    def _chat_config(self, chat_id: int) -> dict[str, Any]:
        """Return a shallow config copy with per-chat tool groups applied.

        Telegram replies are bounded more tightly than CLI replies so the
        worst-case time-to-first-byte stays under the sub-2-second target.
        """
        cfg = dict(self.config)
        cfg["agent"] = {**cfg.get("agent", {}), "max_reply_tokens": 256}
        with _chat_tool_lock:
            groups = _chat_tool_groups.get(chat_id)
        if groups is not None:
            cfg["tools"] = {**cfg.get("tools", {}), "enabled_groups": sorted(groups)}
        return cfg

    def _get_or_create_session(self, chat_id: int) -> ChatSession:
        with _session_lock:
            if chat_id not in _sessions:
                _sessions[chat_id] = ChatSession(
                    self._chat_config(chat_id),
                    model=self.model,
                    tokenizer=self.tokenizer,
                    adapter_loaded=self.adapter_loaded,
                    input_fn=lambda prompt="": self._telegram_input(chat_id, prompt),
                    output_fn=lambda text: self._telegram_output(chat_id, text),
                    confirm_fn=lambda prompt: self._telegram_confirm(chat_id, prompt),
                    generate_fn=self._generate_on_infer_thread,
                    stream_prefix=False,
                )
            return _sessions[chat_id]

    def _refresh_session_tool_groups(self, chat_id: int) -> None:
        """Update an existing session's enabled_groups from the per-chat override."""
        with _session_lock:
            session = _sessions.get(chat_id)
        if session is None:
            return
        with _chat_tool_lock:
            groups = _chat_tool_groups.get(chat_id)
        if groups is not None:
            session.enabled_groups = set(groups)

    def _telegram_input(self, chat_id: int, prompt: str) -> str:
        """Replacement for builtins.input() in a Telegram chat.

        Sends the prompt and blocks the agent thread until the user replies.
        """
        if prompt:
            self._send_text(chat_id, prompt)
        request = _InputRequest()
        with _pending_lock:
            _pending_inputs[chat_id] = request
        # Wait up to 10 minutes for a reply.
        if not request.event.wait(timeout=600):
            with _pending_lock:
                _pending_inputs.pop(chat_id, None)
            raise EOFError("No reply from Telegram user")
        with _pending_lock:
            _pending_inputs.pop(chat_id, None)
        return request.value

    def _telegram_output(self, chat_id: int, text: str) -> None:
        """Replacement for print() in a Telegram chat."""
        self._send_text(chat_id, text)

    def _telegram_confirm(self, chat_id: int, prompt: str) -> bool:
        """Replacement for terminal yes/no prompts via inline keyboard."""
        confirm_id = str(uuid.uuid4())
        request = _ConfirmRequest()
        with _pending_lock:
            _pending_confirmations[confirm_id] = request
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Yes", callback_data=f"confirm:{confirm_id}:yes"),
                InlineKeyboardButton("No", callback_data=f"confirm:{confirm_id}:no"),
            ]
        ])
        self._send_text(chat_id, f"{prompt}\n\nApprove this action?", reply_markup=keyboard)
        if not request.event.wait(timeout=300):
            with _pending_lock:
                _pending_confirmations.pop(confirm_id, None)
            return False
        with _pending_lock:
            _pending_confirmations.pop(confirm_id, None)
        return request.value

    def _send_text(self, chat_id: int, text: str, reply_markup=None):
        """Send a text message from a worker thread back to Telegram."""
        text = _prepare_for_telegram(text)
        if not text:
            return
        if self._loop is None or self._loop.is_closed():
            return
        # Split long messages to stay under Telegram's 4096-char limit.
        # Only attach the reply markup to the last chunk.
        chunks = _split_message(text)
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.application.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        reply_markup=reply_markup if last else None,
                    ),
                    self._loop,
                )
                # Surface send errors so they don't vanish silently.
                future.result(timeout=10)
            except Exception as e:
                print(f"[Telegram] Failed to send message to {chat_id}: {e}")

    async def _send_typing(self, chat_id: int):
        """Keep sending typing indicators until cancelled."""
        try:
            while True:
                if self._loop is None or self._loop.is_closed():
                    break
                await self.application.bot.send_chat_action(
                    chat_id=chat_id, action="typing"
                )
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def _start_typing(self, chat_id: int) -> None:
        """Start the typing indicator for a chat."""
        if self._loop is None or self._loop.is_closed():
            return
        self._stop_typing(chat_id)
        task = asyncio.run_coroutine_threadsafe(
            self._send_typing(chat_id), self._loop
        )
        self._typing_tasks[chat_id] = task

    def _stop_typing(self, chat_id: int) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            await update.message.reply_text(
                "This bot is private.\n"
                f"Add your chat ID {chat_id} to telegram.allowed_chat_ids in config.json."
            )
            return
        name = update.effective_user.first_name or "there"
        await update.message.reply_text(
            f"Hi {name}!\n"
            "Message me to chat. Use /cancel to stop a long-running turn."
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        await update.message.reply_text(
            "Symbio Telegram bot\n"
            "Just send text to chat. The same tools as the CLI work here, "
            "but shell commands, Python code, browser domains, and other "
            "dangerous actions ask for approval first.\n\n"
            "/start — show welcome\n"
            "/help — show this help\n"
            "/ping — last-turn latency breakdown\n"
            "/tools — enable or disable tool groups\n"
            "/cancel — stop the current turn"
        )

    async def _cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        session = self._get_or_create_session(chat_id)
        timings = getattr(session, "last_turn_timings", {}) or {}
        if not timings.get("total_ms"):
            await update.message.reply_text("No turns measured yet.")
            return
        lines = ["Last turn latency:"]
        for key in ("rag_ms", "prompt_ms", "ttft_ms", "gen_ms", "tools_ms", "total_ms"):
            val = timings.get(key)
            label = key.replace("_ms", "").upper()
            lines.append(f"  {label}: {val:.0f}ms" if val is not None else f"  {label}: —")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_tools(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        text, keyboard = self._tools_menu(chat_id)
        await update.message.reply_text(text, reply_markup=keyboard)

    def _tools_menu(self, chat_id: int) -> tuple[str, InlineKeyboardMarkup]:
        with _chat_tool_lock:
            if chat_id not in _chat_tool_groups:
                defaults = set(self.config.get("tools", {}).get("enabled_groups", []))
                _chat_tool_groups[chat_id] = defaults
            enabled = set(_chat_tool_groups[chat_id])

        rows = []
        for group, label in _TOOL_MENU_GROUPS:
            mark = "☑" if group in enabled else "☐"
            rows.append([InlineKeyboardButton(
                f"{mark} {label}", callback_data=f"tools:{group}"
            )])
        rows.append([InlineKeyboardButton("Close", callback_data="tools:close")])
        return "Tap a group to toggle it:", InlineKeyboardMarkup(rows)

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        try:
            await query.answer()
            data = query.data or ""
            if data.startswith("confirm:"):
                await self._handle_confirm_callback(query, data)
                return
            if data.startswith("tools:"):
                await self._handle_tools_callback(query, data)
                return
            print(f"[Telegram] Unknown callback data: {data!r}")
        except Exception as e:
            print(f"[Telegram] Callback handler error: {e}")
            try:
                await query.answer(text="Something went wrong. Try again.", show_alert=True)
            except Exception:
                pass

    async def _handle_confirm_callback(self, query, data: str):
        parts = data.split(":", 2)
        if len(parts) != 3:
            print(f"[Telegram] Malformed confirm callback: {data!r}")
            return
        _, confirm_id, answer = parts
        with _pending_lock:
            request = _pending_confirmations.get(confirm_id)
            if request is None:
                print(f"[Telegram] Confirm request {confirm_id} not found (expired or already answered)")
                await query.edit_message_text("This confirmation request expired or was already answered.")
                return
            request.value = answer == "yes"
            request.event.set()
        try:
            await query.edit_message_text(
                f"{query.message.text}\n\nAnswered: {answer.upper()}"
            )
        except Exception as e:
            print(f"[Telegram] Failed to edit confirmation message: {e}")

    async def _handle_tools_callback(self, query, data: str):
        chat_id = query.message.chat.id
        group = data.split(":", 1)[1]
        if group == "close":
            await query.edit_message_text("Tool settings closed.")
            return

        with _chat_tool_lock:
            enabled = set(_chat_tool_groups.get(chat_id, set()))
            if group in enabled:
                enabled.discard(group)
            else:
                enabled.add(group)
            _chat_tool_groups[chat_id] = enabled

        self._refresh_session_tool_groups(chat_id)
        text, keyboard = self._tools_menu(chat_id)
        await query.edit_message_text(text, reply_markup=keyboard)

    async def _cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        self._stop_typing(chat_id)
        # There is no clean way to interrupt a running MLX generation from
        # another thread; we record the request and the next turn will start
        # fresh with a cleared session.
        with _session_lock:
            if chat_id in _sessions:
                del _sessions[chat_id]
        await update.message.reply_text("Session cleared. Send a new message to start again.")

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            await update.message.reply_text(
                "This bot is private.\n"
                f"Add your chat ID {chat_id} to telegram.allowed_chat_ids in config.json."
            )
            return

        # If the agent thread is waiting for input, deliver the reply and stop.
        with _pending_lock:
            pending = _pending_inputs.get(chat_id)
        if pending is not None:
            pending.value = update.message.text or ""
            pending.event.set()
            return

        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            self._start_typing(chat_id)
            try:
                await asyncio.to_thread(self._process_message, chat_id, update.message.text or "")
            finally:
                self._stop_typing(chat_id)

    def _process_message(self, chat_id: int, text: str):
        session = self._get_or_create_session(chat_id)
        # Only normal chat turns stream; slash commands return synchronously
        # and should never show a "thinking…" placeholder.
        sender = None
        if not text.startswith("/"):
            sender = _StreamSender(
                chat_id,
                self.application.bot,
                self._loop,
                first_chunk_timeout_ms=session.config["agent"].get("first_chunk_timeout_ms", 1500),
            )
            session.stream_chunk_fn = sender.feed
        try:
            if text.startswith("/"):
                result = session._handle_command(text)
                if result == session._QUIT:  # /quit from Telegram clears the session
                    with _session_lock:
                        _sessions.pop(chat_id, None)
                    self._send_text(chat_id, "Session ended. Send /start or any message to begin again.")
                return
            session._agent_turn(text)
        except Exception as e:
            self._send_text(chat_id, f"Error: {e}")
        finally:
            if sender is not None:
                sender.finish()

    def run(self):
        """Start long-polling. Blocks until the process is interrupted."""
        token = get_telegram_token(self.config)
        if not token:
            raise ValueError(
                "No Telegram bot token found. Set SYMBIO_TELEGRAM_TOKEN or run "
                "`python main.py --telegram` and enter one when prompted."
            )
        allowed = self._allowed_chat_ids()
        if not allowed:
            print(
                "[Telegram] Warning: telegram.allowed_chat_ids is empty. "
                "The bot will only reply with setup instructions until you add a chat ID."
            )

        # Start the MLX inference thread before launching Telegram handlers.
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

        self.application = Application.builder().token(token).build()
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("ping", self._cmd_ping))
        self.application.add_handler(CommandHandler("tools", self._cmd_tools))
        self.application.add_handler(CommandHandler("cancel", self._cmd_cancel))
        self.application.add_handler(CallbackQueryHandler(self._on_callback))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        try:
            asyncio.run(self._run_async())
        finally:
            self._infer_queue.put(None)
            if self._infer_thread is not None:
                self._infer_thread.join(timeout=5)
            self._infer_executor.shutdown(wait=False)

    async def _run_async(self):
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        self._loop = asyncio.get_running_loop()
        try:
            # Run until interrupted.
            await asyncio.Future()
        finally:
            for task in list(self._typing_tasks.values()):
                try:
                    task.cancel()
                except Exception:
                    pass
            self._typing_tasks.clear()
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()


def _prepare_for_telegram(text: str) -> str:
    """Strip terminal formatting from agent output for plain-text Telegram."""
    if not text:
        return ""
    # Strip ANSI color codes if any.
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return text.strip("\n")


def _split_message(text: str, limit: int = 4000) -> list[str]:
    """Split text into Telegram-safe chunks at line boundaries when possible."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:]
    return chunks
