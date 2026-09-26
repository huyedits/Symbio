"""Symbio for MCP hosts such as Claude Desktop: its resident model as tools.

Claude Desktop hosts MCP servers, not ACP agents, so this is the way in there:
a stdio MCP server (JSON-RPC, one object per line, no dependencies) with

    ask_symbio      one turn with Symbio's resident model, through the daemon
    symbio_status   whether the model is up, what it is, and what training
                    is doing right now

The daemon serves one conversation at a time. The bridge keeps its session
between calls, so Symbio remembers the thread, and lets it go after five idle
minutes so `symb chat` and the pet's window are not locked out. Symbio's
approval prompts cannot be answered from inside a tool call, so they are
declined, and the reply says what was asked.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from typing import Any

from symbio_desktop.acp import Session, ensure_daemon, log
from symbio_desktop.server import DaemonBridge, _config_summary, constants

SERVER_INFO = {"name": "symbio", "version": "0.2.0"}
IDLE_S = 300.0
TURN_S = 900.0

TOOLS = [
    {
        "name": "ask_symbio",
        "description": (
            "Send a message to Symbio, the user's own self-fine-tuning local model "
            "running on this Mac, and return its reply. It keeps the conversation "
            "between calls. Use it when the user wants Symbio's answer, or wants "
            "Symbio to do something with its own tools, notes and memory."),
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string",
                                       "description": "What to say to Symbio."}},
            "required": ["message"],
        },
    },
    {
        "name": "symbio_status",
        "description": ("Whether Symbio's resident model is running, which model it "
                        "is, how much its adapter has been trained, and what any "
                        "fine-tune in progress is doing."),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Bridge:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.write_lock = threading.Lock()
        self.session: Session | None = None
        self.used = 0.0
        self.turn_lock = threading.Lock()
        threading.Thread(target=self._release_when_idle, daemon=True).start()

    def send(self, message: dict) -> None:
        with self.write_lock:
            self.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.stdout.flush()

    def serve(self) -> int:
        for raw in self.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if "id" not in message or "method" not in message:
                continue                      # notifications, stray responses
            threading.Thread(target=self.handle, args=(message,), daemon=True).start()
        if self.session is not None:
            self.session.bridge.close()
        return 0

    def handle(self, message: dict) -> None:
        method, params = message["method"], message.get("params") or {}
        try:
            if method == "initialize":
                result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": SERVER_INFO}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self.call_tool(params.get("name"), params.get("arguments") or {})
            else:
                self.send({"jsonrpc": "2.0", "id": message["id"],
                           "error": {"code": -32601, "message": f"Method not found: {method}"}})
                return
        except Exception as e:
            log(f"{method} failed: {type(e).__name__}: {e}")
            result = {"content": [{"type": "text", "text": f"Symbio bridge error: {e}"}],
                      "isError": True}
        self.send({"jsonrpc": "2.0", "id": message["id"], "result": result})

    # ── the tools ────────────────────────────────────────────────────

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name == "symbio_status":
            return self._text(json.dumps(status(), indent=2))
        if name == "ask_symbio":
            message = str(arguments.get("message", "")).strip()
            if not message:
                return self._text("Nothing to send: `message` was empty.", error=True)
            return self.ask(message)
        return self._text(f"Unknown tool: {name}", error=True)

    @staticmethod
    def _text(text: str, error: bool = False) -> dict:
        return {"content": [{"type": "text", "text": text}], "isError": error}

    def ask(self, message: str) -> dict:
        with self.turn_lock:
            if self.session is None or not self.session.bridge.alive():
                ok, why = ensure_daemon()
                if not ok:
                    return self._text(why, error=True)
                session = Session("mcp")
                ok, why = session.open()
                if not ok:
                    session.bridge.close()
                    return self._text(why, error=True)
                self.session = session
            session = self.session
            session.bridge.say(message)
            reply, notes = [], []
            deadline = time.monotonic() + TURN_S
            while time.monotonic() < deadline:
                try:
                    event = session.events.get(timeout=1.0)
                except queue.Empty:
                    continue
                kind = event.get("type")
                if kind == "token":
                    reply.append(event["text"])
                elif kind == "confirm":
                    # Nobody can answer this from inside a tool call.
                    session.bridge.confirm(False)
                    notes.append("Symbio asked for approval and was declined here: "
                                 + (event.get("prompt") or "").strip().splitlines()[0][:200]
                                 + " (approve it in `symb chat` if you want it done).")
                elif kind in ("done", "quit"):
                    if kind == "quit":
                        self.session = None
                    break
            else:
                notes.append(f"Symbio was still answering after {TURN_S:.0f}s.")
            self.used = time.monotonic()
            text = "".join(reply).strip() or "(Symbio returned no text.)"
            if notes:
                text += "\n\n" + "\n".join(notes)
            return self._text(text)

    def _release_when_idle(self) -> None:
        while True:
            time.sleep(15)
            with self.turn_lock:
                if self.session is not None and time.monotonic() - self.used > IDLE_S:
                    self.session.bridge.close()
                    self.session = None


def status() -> dict[str, Any]:
    config = _config_summary()
    state: dict[str, Any] = {"model_running": DaemonBridge.daemon_ready(),
                             "model": config.get("model_name"),
                             "assistant_name": config.get("assistant_name"),
                             "home": str(constants.PROJECT_DIR)}
    try:
        progress = json.loads((constants.ADAPTER_DIR / "training_progress.json")
                              .read_text(encoding="utf-8"))
        state["adapter_steps"] = progress.get("total_iters")
    except (OSError, ValueError):
        state["adapter_steps"] = 0
    live_name = getattr(constants, "TRAINING_LIVE_NAME", "training_live.json")
    try:
        live = json.loads((constants.LOG_DIR / live_name).read_text(encoding="utf-8"))
        state["last_training"] = {k: live.get(k) for k in (
            "role", "phase", "iter", "iters", "verdict", "reason")}
        if live.get("train"):
            state["last_training"]["loss"] = live["train"][-1][1]
    except (OSError, ValueError):
        pass
    return state


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    return Bridge().serve()


if __name__ == "__main__":
    sys.exit(main())
