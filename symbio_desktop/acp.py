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
    session/set_mode           Learn or Private (below)

The fine-tune loop is what makes Symbio different from the other agents in an
ACP host's list, so the host is told about it in the protocol's own terms:

* **Commands.** The loop's steps are advertised as slash commands
  (available_commands_update) — /save, /learn, /train, /golden, /forget_last,
  /status — and pass straight through to Symbio's session.
* **Modes.** A session is Private (never trained on) or Learn (added to the
  fine-tune corpus when it closes). Without this every ACP conversation was
  thrown away: a host closing the pipe reads as EOF, and the session's
  "Save conversation for training? [y/N]" takes EOF for no.
* **Live training.** While a session is open, logs/training_live.json — the
  same account the desktop pet draws — becomes a tool call ("Fine-tuning the
  headmaster adapter", step and loss as it goes, the golden gate, then kept or
  rolled back) and a plan of the loop's four stages. The trainer's own
  per-step lines are then left out of the thinking, where they were noise.

stdout carries the protocol and nothing else; everything else goes to stderr.
"""

from __future__ import annotations

import json
import queue
import re
import signal
import sys
import threading
import time
import uuid
from typing import Any

from symbio_desktop.server import DaemonBridge, _config_summary, constants, wake_daemon

PROTOCOL_VERSION = 1
# A cold daemon maps the headmaster's weights and then builds a session; on
# the 14B that is tens of seconds before the first prompt can be read.
SESSION_READY_S = 180.0

MODES = [
    {"id": "private", "name": "Private",
     "description": "This conversation is not added to Symbio's fine-tune corpus."},
    {"id": "learn", "name": "Learn",
     "description": ("When the session closes, this conversation is added to "
                     "Symbio's fine-tune corpus, so the next /train learns from it.")},
]
COMMANDS = [
    {"name": "save", "description": "Add this conversation so far to the fine-tune corpus"},
    {"name": "learn", "description": "Learn from your last correction: it becomes training data"},
    {"name": "train", "description": ("Fine-tune now: golden baseline, a LoRA run, the golden "
                                      "gate, then keep the adapter or roll it back")},
    {"name": "golden", "description": "Run the golden checks against the current adapter"},
    {"name": "forget_last", "description": "Drop the last exchange so it is never trained on"},
    {"name": "status", "description": "Model, adapter and corpus"},
]
LOOP_STAGES = ("Collect: conversations and corrections in the corpus",
               "Train: a LoRA run on the corpus",
               "Golden gate: the golden checks against the new adapter",
               "Keep the new adapter, or roll back to the last one")
# The trainer's per-step lines, which the training tool call now carries.
_TRAINER_LINE = re.compile(r"^\s*(Iter \d+:|Calculating loss)")


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
    """The resident model up and answering, started if it has to be — by the
    same single-flight waker the desktop window uses, so an ACP host and a
    window that both find it down start one model between them, and a load
    that dies is reported when it dies rather than after five minutes."""
    return wake_daemon(lambda text: log(text))


def training_state() -> dict | None:
    name = getattr(constants, "TRAINING_LIVE_NAME", "training_live.json")
    try:
        state = json.loads((constants.LOG_DIR / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def plan(stage: int, done: bool = False) -> dict:
    """The loop's four stages, `stage` in progress (or everything done)."""
    entries = []
    for i, content in enumerate(LOOP_STAGES):
        status = ("completed" if done or i < stage else
                  "in_progress" if i == stage else "pending")
        entries.append({"content": content, "priority": "high" if i else "medium",
                        "status": status})
    return {"sessionUpdate": "plan", "entries": entries}


class Session:
    """One ACP session: one conversation on the daemon."""

    def __init__(self, session_id: str):
        self.id = session_id
        self.events: queue.Queue = queue.Queue()
        self.bridge = DaemonBridge(self.events.put, _config_summary()["assistant_name"])
        self.cancelled = threading.Event()
        self.lock = threading.Lock()          # one turn at a time
        self.draining = False                 # a cancelled turn still running
        self.mode = "private"
        self.turns = 0
        self.saved_at = 0                     # turns covered by the last /save
        self.opened_at = time.time()
        self.closed = threading.Event()
        self.training = False                 # a run is being shown as a tool call

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

    def finish(self) -> None:
        """Close the conversation, keeping it for training first in Learn mode."""
        self.closed.set()
        if self.mode == "learn" and self.turns > self.saved_at and self.bridge.alive():
            with self.lock:
                self.drain()
                self.bridge.say("/save")
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        event = self.events.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if event.get("type") == "system" and "Saved" in event.get("text", ""):
                        log(event["text"].strip())
                    if event.get("type") in ("done", "quit"):
                        break
        self.bridge.close()


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
        try:
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
                    # Requests run off the reader: a prompt turn blocks until
                    # the model is done, and meanwhile the host's answers to
                    # our own permission requests have to be read.
                    threading.Thread(target=self.handle, args=(message,), daemon=True).start()
                else:
                    self.notice(message)
        finally:
            # The host closed the pipe or stopped us: Learn sessions are kept
            # for training on the way out.
            for session in list(self.sessions.values()):
                session.finish()
        return 0

    def handle(self, message: dict) -> None:
        method, params = message["method"], message.get("params") or {}
        handler = {
            "initialize": self.initialize,
            "authenticate": lambda p: {},
            "session/new": self.new_session,
            "session/prompt": self.prompt,
            "session/set_mode": self.set_mode,
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
        if method == "session/new":
            self.opened(self.sessions[result["sessionId"]])

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
            old.finish()
        self.sessions.clear()
        session = Session(f"sess_{uuid.uuid4().hex[:16]}")
        ok, why = session.open()
        if not ok:
            session.bridge.close()
            raise RuntimeError(why)
        self.sessions[session.id] = session
        log(f"session {session.id} open on {constants.DAEMON_SOCKET}")
        return {"sessionId": session.id,
                "modes": {"currentModeId": session.mode, "availableModes": MODES}}

    def opened(self, session: Session) -> None:
        """After the host has the session id: the loop's commands, and a
        watch on training for as long as the session is open."""
        self.update(session, {"sessionUpdate": "available_commands_update",
                              "availableCommands": COMMANDS})
        threading.Thread(target=self.watch_training, args=(session,), daemon=True).start()

    def set_mode(self, params: dict) -> dict:
        session = self.sessions.get(params.get("sessionId"))
        if session is None:
            raise RuntimeError("Unknown session.")
        mode = params.get("modeId")
        if mode not in {m["id"] for m in MODES}:
            raise RuntimeError(f"Unknown mode: {mode}")
        session.mode = mode
        self.update(session, {"sessionUpdate": "current_mode_update", "currentModeId": mode})
        return {}

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
            if text.strip() == "/save":
                session.saved_at = session.turns
            elif not text.startswith("/"):
                session.turns += 1
            return self._turn(session, command=text.lstrip().startswith("/"))

    def _turn(self, session: Session, command: bool = False) -> dict:
        """One turn's events as updates. A command answers in output lines,
        not tokens, so for a command those lines are the reply; otherwise they
        are Symbio's activity, shown as its thinking."""
        line_kind = "agent_message_chunk" if command else "agent_thought_chunk"
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
                if line and not (session.training and _TRAINER_LINE.match(line)):
                    self.update(session, {"sessionUpdate": line_kind,
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

    # ── the fine-tune loop, as the host sees it ──────────────────────

    def watch_training(self, session: Session, poll: float = 0.5) -> None:
        """Runs that start while this session is open, as a tool call and a
        plan. Old runs already on disk are history, not news."""
        shown: dict[str, str] = {}          # run id → last thing said about it
        final: set[str] = set()             # runs whose call has its last status
        while not session.closed.wait(poll):
            state = training_state()
            if not state or not state.get("run"):
                continue
            if float(state.get("started_at") or 0) < session.opened_at - 1:
                continue
            run = str(state["run"])
            call_id = f"train_{run}"
            label = (state.get("role") or "headmaster").replace("_", " ")
            if run not in shown:
                session.training = True
                shown[run] = ""
                # The gate's follow-up run: the verdict lands on it, so the
                # run it follows is closed here instead of left spinning.
                follows = state.get("follows")
                if follows in shown and follows not in final:
                    final.add(follows)
                    self.update(session, {
                        "sessionUpdate": "tool_call_update", "toolCallId": f"train_{follows}",
                        "status": "completed",
                        "content": [{"type": "content", "content": {"type": "text", "text": (
                            "Trained. Some checks still failed, so the gate started "
                            "a follow-up run on them.")}}]})
                self.update(session, {
                    "sessionUpdate": "tool_call", "toolCallId": call_id,
                    "title": f"Fine-tuning the {label} adapter", "kind": "other",
                    "status": "in_progress",
                    "rawInput": {"role": state.get("role"), "steps": state.get("iters"),
                                 "resumed_from": state.get("prior_iters") or 0}})
                self.update(session, plan(1))
            said, status, stage = self._describe(state)
            if said == shown[run]:
                continue
            shown[run] = said
            update = {"sessionUpdate": "tool_call_update", "toolCallId": call_id,
                      "content": [{"type": "content",
                                   "content": {"type": "text", "text": said}}]}
            if status:
                update["status"] = status
                final.add(run)
                session.training = False
            self.update(session, update)
            if stage is not None:
                # A rollback completes the last stage as surely as a keep does.
                self.update(session, plan(stage, done=status is not None))

    @staticmethod
    def _describe(state: dict) -> tuple[str, str | None, int | None]:
        """What to say about a run now: text, final status, plan stage."""
        verdict, phase = state.get("verdict"), state.get("phase")
        steps = state.get("total_iters") or ((state.get("prior_iters") or 0) + (state.get("iter") or 0))
        if verdict == "kept":
            return f"Kept: the adapter now carries {steps} steps of training.", "completed", 3
        if verdict == "rolled_back":
            return ("Rolled back: the new adapter was not kept, and the previous one "
                    "is back in place."), "failed", 3
        if phase in ("failed", "stopped"):
            reason = state.get("reason") or phase
            return f"The run {phase}: {reason}.", "failed", None
        if phase == "trained":
            return ("Golden gate: re-running the golden checks against the new adapter…",
                    None, 2)
        train = state.get("train") or []
        budget = f"/{state['iters']}" if state.get("iters") else ""
        text = f"Step {state.get('iter') or 0}{budget}"
        if train:
            text += f" · loss {train[-1][1]:.3f} (started at {train[0][1]:.3f})"
        return text, None, None


def main() -> int:
    # Line-buffered text on both pipes, whatever the host set.
    sys.stdout.reconfigure(line_buffering=True)
    # A host that stops us with SIGTERM still gets Learn sessions saved: the
    # exit unwinds through serve()'s finally.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    return Agent().serve()


if __name__ == "__main__":
    sys.exit(main())
