"""Symbio behind an OpenAI-compatible API, for Open WebUI and anything else
that speaks /v1/chat/completions.

Like the ACP agent and the desktop window, this is a bridge to `symb daemon`:
the model, the tools, memory and the fine-tune loop all stay in the daemon's
ChatSession, and this process only translates.

    GET  /v1/models              one model, "symbio"
    POST /v1/chat/completions    one turn; streamed (SSE) or not

What does not map one-to-one, and what was chosen instead:

* **History.** An OpenAI client resends the whole conversation every turn;
  the daemon keeps its own. Only the newest user message is sent. A request
  with no assistant message in it is a new chat, and gets a new session.
* **Approvals.** The API has no dialog. A tool that needs approval ends the
  reply with the approval card and "reply yes or no"; the next message is
  the answer, and the turn carries on from where it stopped.
* **Activity.** Symbio's status lines ("[Tool] ...") go out as
  `reasoning_content`, which Open WebUI folds into its thinking section.
* **Background prompts.** Open WebUI asks the same model for chat titles,
  tags and follow-ups ("### Task: ..."). Sent to the agent, they would be a
  turn in the user's conversation with the whole tool registry behind them,
  so they are answered here with an empty result and never reach the daemon.
* **Who may connect.** 127.0.0.1 only, and a bearer key: the daemon session
  behind this can run commands, and a web page in the same browser can POST
  to localhost. The key lives in openai_api.key (0600) next to config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import secrets
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from symbio_desktop.server import DaemonBridge, _config_summary, constants, wake_daemon

MODEL_ID = "symbio"
SESSION_READY_S = 180.0
# A turn on the 14B with tools can run for minutes; nothing in it is idle.
TURN_EVENT_WAIT_S = 1800.0
_TASK_PROMPT = re.compile(r"^\s*###\s*Task:", re.IGNORECASE)
_YES = re.compile(r"^\s*(y|yes|yep|yeah|ok|okay|approve|approved|allow|go|sure|do it)\b", re.IGNORECASE)
_NO = re.compile(r"^\s*(n|no|nope|deny|denied|reject|stop|cancel|don'?t)\b", re.IGNORECASE)


def log(*parts: Any) -> None:
    print("[symbio-openai]", *parts, file=sys.stderr, flush=True)


def key_file():
    return constants.PROJECT_DIR / "openai_api.key"


def api_key() -> str:
    """The bearer key, made on first use. Owner-only from the moment it exists."""
    path = key_file()
    try:
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    except OSError:
        pass
    key = "sk-symbio-" + secrets.token_urlsafe(24)
    previous = os.umask(0o177)
    try:
        path.write_text(key + "\n", encoding="utf-8")
    finally:
        os.umask(previous)
    return key


def message_text(content: Any) -> str:
    """OpenAI message content as plain text: a string, or the text parts of a list."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "\n\n".join(p for p in parts if p)


def task_reply(text: str) -> str:
    """An empty answer of the shape Open WebUI's background task expects."""
    lowered = text.lower()
    if "title" in lowered:
        return json.dumps({"title": ""})
    if "tags" in lowered:
        return json.dumps({"tags": []})
    if "follow" in lowered:
        return json.dumps({"follow_ups": []})
    if "queries" in lowered or "search" in lowered:
        return json.dumps({"queries": []})
    return ""


class Conversation:
    """The one daemon session this API drives. The daemon serves one
    conversation at a time, so there is one of these, and one turn at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: queue.Queue = queue.Queue()
        self.bridge: DaemonBridge | None = None
        # The approval card the daemon is blocked on, between two requests.
        self.awaiting_confirm = False

    def _open(self) -> tuple[bool, str]:
        ok, why = wake_daemon(lambda text: log(text))
        if not ok:
            return False, why
        self.events = queue.Queue()
        self.bridge = DaemonBridge(self.events.put, _config_summary()["assistant_name"])
        ok, why = self.bridge.connect()
        if not ok:
            return False, why
        deadline = time.monotonic() + SESSION_READY_S
        while not self.bridge.ready and time.monotonic() < deadline:
            time.sleep(0.1)
        if not self.bridge.ready:
            return False, ("Connected, but the resident model has not started a "
                           "conversation. It serves one client at a time; another "
                           "window or `symb chat` may be holding it.")
        self._discard_pending()
        return True, ""

    def _discard_pending(self) -> None:
        """The banner and anything else said before the first prompt."""
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        if self.bridge is not None:
            if self.awaiting_confirm:
                self.bridge.confirm(False)
            self.bridge.close()
        self.bridge = None
        self.awaiting_confirm = False

    def turn(self, text: str, new_chat: bool):
        """Yield ("content" | "reasoning", text) for one turn."""
        if new_chat or self.bridge is None or not (self.bridge.alive() or self.bridge.connecting()):
            self.close()
            # A cold model is minutes of loading and prompt warm-up. Said as
            # it goes, or the chat window sits blank and reads as broken.
            result: list = []
            opener = threading.Thread(target=lambda: result.append(self._open()), daemon=True)
            opener.start()
            opener.join(1.0)
            started = time.monotonic()
            if opener.is_alive():
                yield "reasoning", "Waking Symbio: loading the model and its prompt…\n"
            while opener.is_alive():
                opener.join(10.0)
                if opener.is_alive():
                    yield "reasoning", f"  still loading… {int(time.monotonic() - started)}s\n"
            ok, why = result[0] if result else (False, "Symbio failed to start.")
            if not ok:
                yield "content", why
                return
        if self.awaiting_confirm:
            self.awaiting_confirm = False
            approved = bool(_YES.match(text)) and not _NO.match(text)
            yield "reasoning", ("Approved.\n" if approved else "Not approved.\n")
            self.bridge.confirm(approved)
        else:
            self.bridge.say(text)
        command = text.lstrip().startswith("/")
        while True:
            try:
                event = self.events.get(timeout=TURN_EVENT_WAIT_S)
            except queue.Empty:
                yield "content", "\n\n[No answer from Symbio for 30 minutes; the turn was abandoned.]"
                self.close()
                return
            kind = event.get("type")
            if kind == "token":
                yield "content", event.get("text", "")
            elif kind == "system":
                line = (event.get("text") or "").rstrip()
                if line:
                    # A command answers in output lines: those ARE its reply.
                    yield ("content" if command else "reasoning"), line + "\n"
            elif kind == "confirm":
                self.awaiting_confirm = True
                card = (event.get("prompt") or "Symbio wants to act.").strip()
                yield "content", (f"\n\n**Approval needed**\n\n```\n{card}\n```\n\n"
                                  "Reply **yes** to allow it, or **no** to refuse.")
                return
            elif kind == "done":
                return
            elif kind == "quit":
                self.close()
                return


CONVERSATION = Conversation()


class Handler(BaseHTTPRequestHandler):
    server_version = "symbio-openai"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:
        log(fmt % args)

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def _authorized(self) -> bool:
        given = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        if secrets.compare_digest(given.encode(), self.server.key.encode()):
            return True
        self._error(401, "missing or wrong API key (see openai_api.key)")
        return False

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/health", ""):
            return self._json(200, {"status": "ok"})
        if not self._authorized():
            return
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            name = _config_summary().get("assistant_name") or "Symbio"
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "symbio", "name": name}]})
        self._error(404, f"not found: {self.path}")

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._error(404, f"not found: {self.path}")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            messages = body["messages"]
            user = [m for m in messages if m.get("role") == "user"]
            if not user:
                raise ValueError("no user message")
            text = message_text(user[-1].get("content"))
        except (ValueError, KeyError, TypeError) as e:
            return self._error(400, f"bad request: {e}")
        stream = bool(body.get("stream"))
        new_chat = not any(m.get("role") == "assistant" for m in messages)

        if _TASK_PROMPT.match(text):
            return self._reply(task_reply(text), stream, events=None)
        if not text.strip():
            return self._reply("", stream, events=None)
        if not CONVERSATION.lock.acquire(timeout=5):
            return self._error(429, "Symbio is still answering the last message.")
        try:
            self._reply(None, stream, events=CONVERSATION.turn(text, new_chat))
        finally:
            CONVERSATION.lock.release()

    def _reply(self, fixed: str | None, stream: bool, events) -> None:
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if events is None:
            events = iter([("content", fixed or "")])
        if not stream:
            content, reasoning = [], []
            for kind, text in events:
                (content if kind == "content" else reasoning).append(text)
            message = {"role": "assistant", "content": "".join(content).strip()}
            if reasoning:
                message["reasoning_content"] = "".join(reasoning)
            return self._json(200, {
                "id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def chunk(delta: dict, finish: str | None = None) -> None:
            frame = {"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        try:
            chunk({"role": "assistant", "content": ""})
            for kind, text in events:
                if text:
                    chunk({"content": text} if kind == "content" else {"reasoning_content": text})
            chunk({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The tab stopped the reply. The daemon cannot be interrupted
            # mid-generation, so the rest of the turn is read and dropped.
            for _ in events:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="symb openai")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--show-key", action="store_true", help="print the API key and exit")
    args = parser.parse_args(argv)
    key = api_key()
    if args.show_key:
        print(key)
        return 0
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.key = key
    print(f"Symbio OpenAI API on http://127.0.0.1:{args.port}/v1  model: {MODEL_ID}\n"
          f"API key: {key_file()}  (symb openai --show-key)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CONVERSATION.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
