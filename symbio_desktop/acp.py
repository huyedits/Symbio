"""Symbio as an ACP agent: the Agent Client Protocol on stdin/stdout.

Any ACP host — Zed, the VS Code and JetBrains ACP plugins, Toad, and Hermes
once it can host outside agents — starts this process and talks JSON-RPC 2.0
to it, one JSON object per line. The model is not in here: like the desktop
window, this is a bridge to `symb daemon`, and it reuses the desktop's
DaemonBridge, which already knows the daemon's protocol and what to strip from
a reply (the speaker label, stop sentinels, spent tool calls, thinking spans).

    initialize                 capabilities; no authentication
    session/new                starts the daemon if it is not up, then opens
                               one conversation on it
    session/prompt             one turn: reply text as agent_message_chunk,
                               Symbio's activity lines as agent_thought_chunk,
                               its approval prompts as session/request_permission
    session/cancel             ends the turn for the host at once; the daemon
                               cannot be interrupted mid-generation, so the rest
                               of that turn is read and thrown away

stdout carries the protocol and nothing else; everything else goes to stderr.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from symbio_desktop.server import DaemonBridge, _config_summary, constants

PROTOCOL_VERSION = 1
# A cold daemon maps the headmaster's weights and then builds a session; on
# the 14B that is tens of seconds before the first prompt can be read.
DAEMON_START_S = 300.0
SESSION_READY_S = 180.0


def log(*parts: Any) -> None:
    print("[symbio-acp]", *parts, file=sys.stderr, flush=True)


def prompt_text(blocks: list[dict]) -> str:
    """An ACP prompt as one message for the model: text as it is, embedded
    resources inlined under their URI, links named."""
    parts: list[str] = []
    for block in blocks or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "resource":
            resource = block.get("resource") or {}
            if resource.get("text"):
                parts.append(f"[{resource.get('uri', 'attached')}]\n{resource['text']}")
        elif kind == "resource_link":
            parts.append(f"[{block.get('name') or block.get('uri', 'link')}: {block.get('uri', '')}]")
    return "\n\n".join(p for p in parts if p)


def ensure_daemon() -> tuple[bool, str]:
    """The resident model up and answering, started if it has to be."""
    if DaemonBridge.daemon_ready():
        return True, ""
    loading = False
    try:
        pid = int(constants.DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        loading = True          # up, still mapping weights
    except (OSError, ValueError):
        pass
    if not loading:
        log("no resident model; starting `symb daemon`")
        root = Path(__file__).resolve().parent.parent
        subprocess.run([sys.executable, "-m", "symbio.app.cli", "daemon", "start"],
                       cwd=str(root), stdin=subprocess.DEVNULL,
                       stdout=sys.stderr, stderr=sys.stderr, check=False)
    deadline = time.monotonic() + DAEMON_START_S
    while time.monotonic() < deadline:
        if DaemonBridge.daemon_ready():
            return True, ""
        time.sleep(0.5)
    return False, (f"The resident model did not come up in {DAEMON_START_S:.0f}s. "
                   f"See {constants.LOG_DIR / 'daemon.log'}.")


class Session:
    """One ACP session: one conversation on the daemon."""

    def __init__(self, session_id: str):
        self.id = session_id
        self.events: queue.Queue = queue.Queue()
        self.bridge = DaemonBridge(self.events.put, _config_summary()["assistant_name"])
        self.cancelled = threading.Event()
        self.lock = threading.Lock()          # one turn at a time
        self.draining = False                 # a cancelled turn still running

    def open(self) -> tuple[bool, str]:
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
        return True, ""

    def drain(self) -> None:
        """Read what is left of a cancelled turn, up to its end."""
        while self.draining:
            try:
                event = self.events.get(timeout=600)
            except queue.Empty:
                break
            if event.get("type") in ("done", "quit"):
                self.draining = False


class Agent:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.write_lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.waiting: dict[Any, queue.Queue] = {}
        self.ids = iter(range(1, 1 << 62))

    # ── JSON-RPC plumbing ────────────────────────────────────────────

    def send(self, message: dict) -> None:
        line = json.dumps(message, ensure_ascii=False)
        with self.write_lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def call(self, method: str, params: dict, timeout: float = 3600.0) -> dict:
        """A request to the host, answered on the reader thread."""
        request_id = f"symbio-{next(self.ids)}"
        box: queue.Queue = queue.Queue(maxsize=1)
        self.waiting[request_id] = box
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            reply = box.get(timeout=timeout)
        finally:
            self.waiting.pop(request_id, None)
        if "error" in reply:
            raise RuntimeError(reply["error"].get("message", "host error"))
        return reply.get("result") or {}

    def update(self, session: Session, update: dict) -> None:
        self.notify("session/update", {"sessionId": session.id, "update": update})

    def serve(self) -> int:
        for raw in self.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except ValueError:
                self.send({"jsonrpc": "2.0", "id": None,
                           "error": {"code": -32700, "message": "Parse error"}})
                continue
            if "method" not in message:
                box = self.waiting.get(message.get("id"))
                if box is not None:
                    box.put(message)
                continue
            if "id" in message:
                # Requests run off the reader: a prompt turn blocks until the
                # model is done, and meanwhile the host's answers to our own
                # permission requests have to be read.
                threading.Thread(target=self.handle, args=(message,), daemon=True).start()
            else:
                self.notice(message)
        for session in self.sessions.values():
            session.bridge.close()
        return 0

    def handle(self, message: dict) -> None:
        method, params = message["method"], message.get("params") or {}
        handler = {
            "initialize": self.initialize,
            "authenticate": lambda p: {},
            "session/new": self.new_session,
            "session/prompt": self.prompt,
        }.get(method)
        if handler is None:
            self.send({"jsonrpc": "2.0", "id": message["id"],
                       "error": {"code": -32601, "message": f"Method not found: {method}"}})
            return
        try:
            result = handler(params)
        except Exception as e:                          # the host sees why
            log(f"{method} failed: {type(e).__name__}: {e}")
            self.send({"jsonrpc": "2.0", "id": message["id"],
                       "error": {"code": -32603, "message": str(e)}})
            return
        self.send({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def notice(self, message: dict) -> None:
        if message.get("method") == "session/cancel":
            session = self.sessions.get((message.get("params") or {}).get("sessionId"))
            if session is not None:
                session.cancelled.set()

    # ── the methods ──────────────────────────────────────────────────

    def initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {"image": False, "audio": False,
                                       "embeddedContext": True},
                "mcpCapabilities": {"http": False, "sse": False},
            },
            "agentInfo": {"name": "symbio", "title": "Symbio", "version": "0.2.0"},
            "authMethods": [],
        }

    def new_session(self, params: dict) -> dict:
        ok, why = ensure_daemon()
        if not ok:
            raise RuntimeError(why)
        # The daemon serves one conversation at a time, so a new session
        # replaces the last one rather than queueing behind it.
        for old in list(self.sessions.values()):
            old.bridge.close()
        self.sessions.clear()
        session = Session(f"sess_{uuid.uuid4().hex[:16]}")
        ok, why = session.open()
        if not ok:
            session.bridge.close()
            raise RuntimeError(why)
        self.sessions[session.id] = session
        log(f"session {session.id} open on {constants.DAEMON_SOCKET}")
        return {"sessionId": session.id}

    def prompt(self, params: dict) -> dict:
        session = self.sessions.get(params.get("sessionId"))
        if session is None:
            raise RuntimeError("Unknown session; start one with session/new.")
        text = prompt_text(params.get("prompt") or [])
        if not text.strip():
            return {"stopReason": "end_turn"}
        with session.lock:
            session.drain()
            session.cancelled.clear()
            session.bridge.say(text)
            return self._turn(session)

    def _turn(self, session: Session) -> dict:
        while True:
            if session.cancelled.is_set():
                session.draining = True
                return {"stopReason": "cancelled"}
            try:
                event = session.events.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = event.get("type")
            if kind == "token":
                self.update(session, {"sessionUpdate": "agent_message_chunk",
                                      "content": {"type": "text", "text": event["text"]}})
            elif kind == "system":
                line = (event.get("text") or "").rstrip()
                if line:
                    self.update(session, {"sessionUpdate": "agent_thought_chunk",
                                          "content": {"type": "text", "text": line + "\n"}})
            elif kind == "confirm":
                session.bridge.confirm(self._permission(session, event.get("prompt", "")))
            elif kind == "done":
                return {"stopReason": "end_turn"}
            elif kind == "quit":
                self.sessions.pop(session.id, None)
                return {"stopReason": "end_turn"}

    def _permission(self, session: Session, prompt: str) -> bool:
        """Symbio's approval gate, asked through the host's own dialog."""
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        title = prompt.strip().splitlines()[0][:200] if prompt.strip() else "Symbio wants to act"
        tool_call = {"toolCallId": call_id, "title": title, "kind": "execute",
                     "status": "pending",
                     "content": [{"type": "content",
                                  "content": {"type": "text", "text": prompt.strip()}}]}
        self.update(session, {"sessionUpdate": "tool_call", **tool_call})
        try:
            result = self.call("session/request_permission", {
                "sessionId": session.id,
                "toolCall": tool_call,
                "options": [
                    {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "reject", "name": "Reject", "kind": "reject_once"},
                ],
            })
        except Exception as e:
            log(f"permission request failed: {e}")
            result = {}
        outcome = result.get("outcome") or {}
        approved = outcome.get("outcome") == "selected" and outcome.get("optionId") == "allow"
        self.update(session, {"sessionUpdate": "tool_call_update", "toolCallId": call_id,
                              "status": "completed" if approved else "failed"})
        return approved


def main() -> int:
    # Line-buffered text on both pipes, whatever the host set.
    sys.stdout.reconfigure(line_buffering=True)
    return Agent().serve()


if __name__ == "__main__":
    sys.exit(main())
