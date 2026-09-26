"""Symbio Desktop — the window's backend.

Three decisions, all of them about memory.

**No web framework.** This was FastAPI on uvicorn, which is four packages
(fastapi, starlette, uvicorn, pydantic) that were not installed on the machine
it shipped from — `python3 -m symbio_desktop.cli` died on
`ModuleNotFoundError: No module named 'fastapi'`, so the app had never run at
all. The standard library has an HTTP server and enough socket to speak
WebSocket in about a hundred lines, and it costs nothing to import.

**The model lives in the daemon, not here.** A ChatSession constructed in this
process would pull the headmaster's weights into it — ~10 GB — and the desktop
would be the largest thing on the Mac. `symb daemon` already keeps one loaded
copy behind a Unix socket, so this process is a bridge: browser WebSocket on
one side, daemon socket on the other, nothing resident in between. With no
daemon running it starts one (`symb daemon start`, in its own process) and
holds the message until the model has loaded, rather than telling the reader
to go and run a command, or quietly loading a second copy of a 14B model here.

**Nothing is imported that is not needed to answer.** The API handlers read
JSON off disk; `symbio.app.dispatch` is imported inside the one handler that
needs the catalog. Importing the agent stack at module scope would drag mlx in
behind it.

The wire protocol is the FastAPI version's plus one frame: {type: connected|
token|system|progress|confirm|done|error|waking} out, {type: chat|
confirm_response|ping} in. `waking` is the model loading; a page that does not
know it ignores it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import socket
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).parent.resolve()
STATIC_DIR = APP_DIR / "static"


def _load_constants():
    """symbio/constants.py, loaded WITHOUT importing the symbio package.

    `from symbio import constants` runs symbio/__init__.py, which imports the
    agent, the chat loop, mlx, anthropic, fastmcp, telegram, pydantic,
    starlette and uvicorn behind it: measured at 130 MB resident for a process
    that wanted eight Path objects. constants.py imports nothing but pathlib
    and typing, so it can be loaded as a standalone module and the rest of the
    package left alone. Same 25 MB the standard library costs on its own.
    """
    import importlib.util

    path = APP_DIR.parent / "symbio" / "constants.py"
    spec = importlib.util.spec_from_file_location("_symbio_constants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


constants = _load_constants()

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ── data model builders ──────────────────────────────────────────────

def _adapter_info(role: str | None = None) -> dict[str, Any]:
    adapter_dir = constants.adapter_dir_for(role)
    info: dict[str, Any] = {
        "role": role or "headmaster",
        "path": str(adapter_dir),
        "exists": adapter_dir.exists(),
        "has_weights": (adapter_dir / "adapters.safetensors").exists(),
        "has_config": (adapter_dir / "adapter_config.json").exists(),
    }
    config_file = adapter_dir / "adapter_config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            lora = cfg.get("lora_parameters") or {}
            info["rank"] = lora.get("rank")
            info["num_layers"] = cfg.get("num_layers")
            info["base_model"] = cfg.get("model")
        except (OSError, json.JSONDecodeError):
            pass
    progress_file = adapter_dir / "training_progress.json"
    if progress_file.exists():
        try:
            info["training"] = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    safetensors = adapter_dir / "adapters.safetensors"
    if safetensors.exists():
        info["size_mb"] = round(safetensors.stat().st_size / (1024 * 1024), 1)
    return info


def _skill_notes() -> list[dict[str, Any]]:
    skills = []
    for note_path in sorted(constants.NOTES_DIR.glob("*Skill__*.md")):
        try:
            first_line = note_path.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        name = first_line.removeprefix("# Skill:").strip()
        slug = name.lower().replace(" ", "_").replace("'", "")
        slug = "".join(c for c in slug if c.isalnum() or c == "_").strip("_")

        health_entries = []
        health_path = note_path.with_suffix(note_path.suffix + ".health.jsonl")
        if health_path.exists():
            try:
                for line in health_path.read_text(encoding="utf-8").strip().splitlines():
                    if line.strip():
                        health_entries.append(json.loads(line))
            except (OSError, json.JSONDecodeError):
                pass

        skills.append({
            "name": name, "slug": slug, "role": slug,
            "note_path": str(note_path), "health": health_entries,
            "error_count": sum(1 for e in health_entries if e.get("type") == "error"),
            "correction_count": sum(1 for e in health_entries if e.get("type") == "correction"),
        })
    return skills


def _rag_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {
        "notes_count": len(list(constants.NOTES_DIR.glob("*.md"))),
        "training_samples": 0, "training_size_mb": 0, "sessions_count": 0,
    }
    train_file = constants.TRAIN_FILE
    if train_file.exists():
        try:
            stats["training_size_mb"] = round(train_file.stat().st_size / (1024 * 1024), 1)
            # Counted by scanning for newlines rather than by reading the file
            # into a list of lines: the corpus here is already 6.7 MB and this
            # endpoint is polled.
            with train_file.open("rb") as fh:
                stats["training_samples"] = sum(chunk.count(b"\n")
                                                for chunk in iter(lambda: fh.read(1 << 20), b""))
        except OSError:
            pass
    if constants.SESSIONS_DIR.exists():
        stats["sessions_count"] = len(list(constants.SESSIONS_DIR.glob("*.json")))
    return stats


def _config_summary() -> dict[str, Any]:
    try:
        cfg = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cfg = {}
    return {
        "model_name": cfg.get("model_name", ""),
        "assistant_name": cfg.get("assistant_name", "Symbio"),
        "user_name": cfg.get("user_name", "user"),
        "dispatch_enabled": cfg.get("dispatch", {}).get("enabled", False),
        "rag_enabled": cfg.get("rag", {}).get("enabled", True),
        "auto_train": cfg.get("learn", {}).get("auto_train", True),
        "lora_rank": cfg.get("lora", {}).get("rank", 8),
        "lora_iters": cfg.get("lora", {}).get("iters", 50),
    }


def _worker_catalog() -> dict[str, dict[str, Any]]:
    """The worker/skill catalog, read straight off disk.

    symbio.app.dispatch has load_catalog(), and importing it costs 40 MB of
    agent stack in a process whose entire job is to serve four JSON endpoints
    and forward a socket. The file is a JSON object; reading it is the whole
    function.
    """
    try:
        loaded = json.loads(constants.WORKER_MODELS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def get_ecosystem() -> dict[str, Any]:
    catalog = _worker_catalog()
    config = _config_summary()
    skills = _skill_notes()
    rag = _rag_stats()

    headmaster = _adapter_info(role=None)
    headmaster["model_name"] = config["model_name"]
    headmaster["type"] = "headmaster"

    workers = []
    for key, entry in catalog.items():
        role = entry.get("role", key)
        if entry.get("is_skill", False):
            continue
        info = _adapter_info(role=role)
        info["model_name"] = entry.get("model_name", "")
        info["description"] = entry.get("description", "")
        info["type"] = "worker"
        info["catalog_key"] = key
        workers.append(info)

    skill_nodes = []
    for skill in skills:
        role = skill["role"]
        info = _adapter_info(role=role)
        info["skill_name"] = skill["name"]
        info["type"] = "skill"
        info["error_count"] = skill["error_count"]
        info["correction_count"] = skill["correction_count"]
        info["health_entries"] = skill["health"]
        for key, entry in catalog.items():
            if entry.get("role") == role:
                info["system_prompt"] = entry.get("system_prompt", "")
                info["routing_rationale"] = entry.get("routing_rationale", "")
                info["catalog_key"] = key
                break
        skill_nodes.append(info)

    return {
        "headmaster": headmaster,
        "workers": workers,
        "skills": skill_nodes,
        "rag": {"type": "rag", **rag},
        "training": {"type": "training", "samples": rag["training_samples"],
                     "size_mb": rag["training_size_mb"],
                     "auto_train": config["auto_train"]},
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }


# What the window may switch, and what each switch means in the user's terms.
# Deliberately a SHORT list of user-facing capabilities: config.json holds
# ~90 booleans, most of them training internals nobody should flip from a
# chat window, and a settings page that offers all of them is a page nobody
# reads. Anything not named here is not reachable from the browser at all.
TOOL_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("terminal", "Shell commands", "Run commands on this Mac. The approval gate still applies."),
    ("code", "Python sandbox", "Run Python in the sandbox directory."),
    ("browser", "Browser control", "Drive a real Chrome window — open pages, click, type."),
    ("desktop", "Desktop control", "Read the screen's controls and click, type and drag on them."),
    ("web_search", "Web search", "Look things up online."),
    ("memory", "Memory", "Read and write saved memory and the user profile."),
    ("notes", "Notes", "Save, search and delete notes."),
    ("digest", "Digest notes", "Fold notes into training data."),
    ("train", "Training", "Run LoRA training and rebuild adapters."),
    ("cron", "Scheduled jobs", "Run work on a schedule."),
    ("delegate", "Worker models", "Hand tasks to smaller trained workers."),
    ("config", "Change settings", "Let the agent edit its own configuration."),
    ("system", "System checks", "Health checks and feature verification."),
)

# Single booleans, addressed by dotted path.
FEATURE_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("safety.enabled", "Safety gate", "Risk scoring and approval prompts before dangerous tools."),
    ("rag.enabled", "Retrieval", "Pull matching notes and past sessions into the prompt."),
    ("memory.enabled", "Durable memory", "Keep the agent's saved memory in every prompt."),
    ("vision.enabled", "Vision", "Look at the screen with the vision model when there is no control tree."),
    ("learn.auto_train", "Learn from corrections", "Train on mistakes once enough have collected."),
    ("agent.show_reasoning", "Show reasoning", "Stream the model's thinking, folded above each reply."),
    ("agent.stream_output", "Stream replies", "Show tokens as they arrive instead of all at once."),
    ("agent.show_tool_output", "Show tool results", "Print what each tool returned, in short, as it happens."),
    ("dispatch.enabled", "Delegation", "Let the headmaster route work to worker adapters."),
    ("telemetry.enabled", "Telemetry", "Send anonymous usage pings. Off by default."),
)


def _dotted(config: dict, path: str, default=None):
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def get_settings() -> dict[str, Any]:
    """Every switch the window offers, with its current value."""
    try:
        config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}
    groups = set(_dotted(config, "tools.enabled_groups", []) or [])
    return {
        "groups": [{"key": k, "label": label, "hint": hint, "on": k in groups}
                   for k, label, hint in TOOL_GROUPS],
        "features": [{"key": k, "label": label, "hint": hint,
                      "on": bool(_dotted(config, k, False))}
                     for k, label, hint in FEATURE_FLAGS],
        "restart_note": "Tool changes reach the model on its next turn; the "
                        "rest apply when the resident model restarts.",
    }


def set_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a settings change to config.json, and nothing else.

    Only keys named in TOOL_GROUPS and FEATURE_FLAGS are writable, so a
    request cannot reach model_name, the remote hosts table or the safety
    thresholds. Written through a temporary file and renamed: config.json is
    read by the daemon and by every CLI session, and a half-written one
    breaks all of them at once.
    """
    allowed_groups = {k for k, _l, _h in TOOL_GROUPS}
    allowed_flags = {k for k, _l, _h in FEATURE_FLAGS}
    try:
        config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"Could not read config.json: {e}"}

    changed: list[str] = []
    for key, value in (patch.get("groups") or {}).items():
        if key not in allowed_groups:
            continue
        groups = list(_dotted(config, "tools.enabled_groups", []) or [])
        if value and key not in groups:
            groups.append(key)
            changed.append(f"+{key}")
        elif not value and key in groups:
            groups.remove(key)
            changed.append(f"-{key}")
        config.setdefault("tools", {})["enabled_groups"] = groups

    for key, value in (patch.get("features") or {}).items():
        if key not in allowed_flags:
            continue
        head, _, leaf = key.rpartition(".")
        node = config
        for part in head.split("."):
            node = node.setdefault(part, {})
        if bool(node.get(leaf)) != bool(value):
            changed.append(f"{key}={'on' if value else 'off'}")
        node[leaf] = bool(value)

    if not changed:
        return {"ok": True, "changed": []}
    temporary = constants.CONFIG_FILE.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary, constants.CONFIG_FILE)
    except OSError as e:
        return {"ok": False, "error": f"Could not write config.json: {e}"}
    return {"ok": True, "changed": changed}


def get_sessions(limit: int = 60, session_id: str = "") -> dict[str, Any]:
    """Every conversation this agent has had, across every front end.

    The CLI, the Telegram gateway and this window all write to the same
    logs/sessions.db, and until now nothing listed them: 850 sessions and
    6,136 turns with no way to look at one without opening SQLite. Read
    immutably — the daemon has this file open and is writing to it.
    """
    db = constants.PROJECT_DIR / "logs" / "sessions.db"
    if not db.is_file():
        return {"sessions": [], "reason": "No session store yet."}
    try:
        connection = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    except sqlite3.Error as e:
        return {"sessions": [], "reason": f"Could not read the session store: {e}"}
    try:
        if session_id:
            rows = connection.execute(
                "SELECT timestamp, role, content FROM turns WHERE session_id = ? "
                "ORDER BY id LIMIT 400", (session_id,)).fetchall()
            return {"session": session_id,
                    "turns": [{"at": at, "role": role, "text": text}
                              for at, role, text in rows]}
        rows = connection.execute(
            "SELECT s.id, s.started, COUNT(t.id), "
            "       MIN(CASE WHEN t.role = 'user' THEN t.content END) "
            "FROM sessions s LEFT JOIN turns t ON t.session_id = s.id "
            "GROUP BY s.id ORDER BY s.started DESC LIMIT ?", (limit,)).fetchall()
        return {"sessions": [
            {"id": sid, "started": started, "turns": turns,
             "opening": (opening or "").strip()[:120]}
            for sid, started, turns, opening in rows]}
    except sqlite3.Error as e:
        return {"sessions": [], "reason": f"Could not read the session store: {e}"}
    finally:
        connection.close()


def _supervisor_state() -> dict[str, Any]:
    """The watcher's heartbeat, if one is running.

    Read straight off disk rather than through symbio.app.supervisor: the
    window does not import the agent package (see the memory note at the top
    of this file), and a heartbeat is four fields of JSON.
    """
    try:
        state = json.loads(
            (constants.PROJECT_DIR / "logs" / "supervisor.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"running": False}
    checked = str(state.get("checked_at", ""))
    fresh = False
    try:
        fresh = (datetime.now() - datetime.fromisoformat(checked)).total_seconds() < 300
    except ValueError:
        pass
    return {"running": fresh, "checked_at": checked,
            "model": state.get("model", "?"),
            "restarts": state.get("restarts", 0),
            "jobs_run": state.get("jobs_run", 0),
            "last_error": state.get("last_error", ""),
            "ran": (state.get("ran") or [])[-5:]}


def get_health() -> dict[str, Any]:
    skills = _skill_notes()
    return {
        "supervisor": _supervisor_state(),
        "total_skills": len(skills),
        "total_errors": sum(s["error_count"] for s in skills),
        "total_corrections": sum(s["correction_count"] for s in skills),
        "skills_with_errors": [s["name"] for s in skills if s["error_count"] > 0],
        "skills_with_corrections": [s["name"] for s in skills if s["correction_count"] > 0],
    }


# ── WebSocket, by hand ───────────────────────────────────────────────

class WSError(Exception):
    pass


def ws_accept_key(client_key: str) -> str:
    """RFC 6455's handshake: the client's key, the magic GUID, SHA-1, base64."""
    digest = hashlib.sha1((client_key + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def ws_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """One unmasked server frame. Servers never mask; clients always do."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < (1 << 16):
        header.append(126)
        header += struct.pack(">H", length)
    else:
        header.append(127)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


def _read_exactly(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise WSError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def ws_read_message(sock: socket.socket) -> tuple[int, bytes]:
    """The next complete message: (opcode, payload). Continuations joined."""
    payload = bytearray()
    first_opcode = None
    while True:
        b1, b2 = _read_exactly(sock, 2)
        fin = b1 & 0x80
        opcode = b1 & 0x0F
        masked = b2 & 0x80
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack(">H", _read_exactly(sock, 2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _read_exactly(sock, 8))[0]
        # A browser always masks. An unmasked frame is either a broken client
        # or something that is not a browser, and the RFC says to fail.
        if not masked:
            raise WSError("unmasked frame from client")
        mask = _read_exactly(sock, 4)
        data = bytearray(_read_exactly(sock, length))
        for i in range(length):
            data[i] ^= mask[i % 4]
        if first_opcode is None and opcode != 0x0:
            first_opcode = opcode
        payload += data
        if fin:
            return first_opcode or opcode, bytes(payload)


# ── waking the resident model ────────────────────────────────────────

# How long a message waits for the model to load. A 14B maps in about half a
# minute from a warm disk; a cold disk or a Mac under memory pressure is slower.
DAEMON_START_S = 300.0
# Between progress lines while it loads.
_WAKE_REPORT_S = 2.0
_WAKE_LOCK = threading.Lock()


def daemon_state() -> str:
    """"ready" (a live pid and its socket), "loading" (a live pid, no socket
    yet: the socket is bound only once the weights are in) or "down"."""
    try:
        pid = int(constants.DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return "down"
    return "ready" if constants.DAEMON_SOCKET.exists() else "loading"


def _is_our_daemon(pid: int) -> bool:
    """A pid file outlives an OOM kill, and the pid can come back as someone
    else's process — which would read as "loading" forever."""
    try:
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True                      # cannot tell: believe the pid file
    return "symbio" in command and "daemon" in command


def _daemon_log_tail(lines: int = 4) -> str:
    try:
        text = (constants.LOG_DIR / "daemon.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    noise = ("MallocStackLogging", "Fetching ", "it/s]")
    keep = [line.strip() for line in text.splitlines()
            if line.strip() and not any(n in line for n in noise)]
    return "\n".join(keep[-lines:])


def wake_daemon(report=None, timeout: float = DAEMON_START_S) -> tuple[bool, str]:
    """The resident model up and answering, started if it has to be.

    One starter at a time: a lock here for this process, and an flock inside
    `symb daemon start` for every process (see symbio/app/daemon.py). Two
    windows, or a window and an ACP host, that both saw "down" would otherwise
    both start one — two copies of a 14B, the out-of-memory kill this Mac
    keeps having.

    `report(text)` gets a progress line every couple of seconds while it loads.
    A load that dies (usually memory) is reported when it dies, with the end
    of daemon.log, not after the whole timeout.
    """
    report = report or (lambda _text: None)
    started = time.monotonic()
    # One starter per process here; across processes `symb daemon start`
    # itself holds an flock around check-and-start (symbio/app/daemon.py), so
    # a window, an ACP host and `symb watch` that all find it down start one.
    with _WAKE_LOCK:
        state = daemon_state()
        if state == "loading":
            try:
                pid = int(constants.DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = 0
            if pid and not _is_our_daemon(pid):
                for stale in (constants.DAEMON_PID_FILE, constants.DAEMON_SOCKET):
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                state = "down"
        if state == "down":
            report("Waking Symbio — starting the model…")
            try:
                done = subprocess.run(
                    [sys.executable, "-m", "symbio.app.cli", "daemon", "start"],
                    cwd=str(APP_DIR.parent), stdin=subprocess.DEVNULL,
                    capture_output=True, text=True, timeout=60, check=False)
                said = (done.stdout + done.stderr).strip()
            except (OSError, subprocess.SubprocessError) as e:
                said = str(e)
            if daemon_state() == "down":
                return False, ("Symbio could not start its model"
                               + (f": {said[-400:]}" if said else "."))
    last = 0.0
    while time.monotonic() - started < timeout:
        state = daemon_state()
        if state == "ready":
            return True, ""
        if state == "down":
            tail = _daemon_log_tail()
            return False, ("Symbio's model stopped while it was loading"
                           + (f":\n{tail}" if tail else ".")
                           + "\nThat is usually memory — another model or a training "
                             "run may be holding it. Send again to retry.")
        now = time.monotonic()
        if now - last >= _WAKE_REPORT_S:
            report(f"Waking Symbio — loading the model… {now - started:.0f}s")
            last = now
        time.sleep(0.5)
    return False, (f"Symbio's model did not finish loading in {timeout:.0f}s. "
                   f"See {constants.LOG_DIR / 'daemon.log'}.")


# ── the bridge to the resident model ─────────────────────────────────

# The one live conversation, kept across browser reloads.
#
# A refresh closes the WebSocket, which used to close the daemon connection
# under it -- so the ChatSession ended and the model forgot the conversation
# the user was still looking at. The daemon serves one session at a time
# anyway, so the bridge belongs to the server, not to a socket.
_BRIDGE: "DaemonBridge | None" = None
_BRIDGE_LOCK = threading.Lock()
# The first window of this server has asked for the model to be loaded.
_PREWARMED = False


class DaemonBridge:
    """One browser conversation, wired to one daemon connection.

    The daemon speaks the CLI's protocol -- it drives a real ChatSession whose
    input_fn blocks on the socket -- so the mapping is direct: its `stream` is
    a token, its `output` is a line of activity, its `confirm` is the same
    approval the terminal would print, and its `input_prompt` is the end of a
    turn.
    """

    # What the terminal adds and a window must not repeat. The daemon builds
    # its session with stream_prefix=True, so every turn arrives as
    # "Caine   : " -- a label that belongs to a transcript, not to a chat
    # bubble that already says who is speaking.
    #
    # Matched against the ASSISTANT'S NAME, not against "anything before a
    # colon". The loose pattern ate the opening of any reply that began with
    # a clause and a colon: "Here you go: " vanished and the answer started
    # mid-sentence. A label this window did not expect is better left in than
    # a sentence taken out.
    _FALLBACK_PREFIX = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_-]{0,23}\s{2,}:\s*")

    # Sentinels that are protocol, not text. They end a turn or a block, and
    # every one of them has been seen in a reply this window rendered.
    _SENTINELS = ("<end>", "<|im_end|>", "<|endoftext|>", "</s>")
    # Spans the reader must never see mid-reply: a tool call is an instruction
    # to the runtime that already ran, and a thinking block belongs in the
    # folded Thought section, not in the answer.
    _SPANS = (("<tool_call>", "</tool_call>"),
              ("<think>", "</think>"),
              ("<thinking>", "</thinking>"))

    def __init__(self, send_json, assistant_name: str = "") -> None:
        self.sink = send_json
        self.assistant_name = assistant_name
        self.prefix_re = (
            re.compile(r"^\s*" + re.escape(assistant_name) + r"\s*:\s*", re.IGNORECASE)
            if assistant_name else self._FALLBACK_PREFIX)
        self.turn_text = ""
        self.prefix_done = False
        self.carry = ""
        self.inside: tuple[str, str] | None = None
        self.sock: socket.socket | None = None
        self.rfile = None
        self.wfile = None
        # Turns sent and not yet answered. A count, not a flag: messages held
        # while the model loads go out together, and each needs its `done`.
        self.open_turns = 0
        self.waking = False
        self._wake_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        # say() and the first-prompt hand-off of `pending` both read `ready`
        # and touch `pending`; unlocked, a message sent in between could jump
        # ahead of the held ones or be stranded in the list.
        self._pending_lock = threading.Lock()
        self.woke_at = 0.0             # when a wake by this bridge loaded a model
        # The session prints its banner and THEN asks for input, so a message
        # sent the moment the socket opens arrives before the session is
        # listening: the banner's own input_prompt closed a turn that had not
        # started, the browser was told `done` with nothing in it, and the
        # reply the model then produced had no turn left to belong to.
        # Nothing is written to the daemon until it has asked once.
        self.ready = False
        self.pending: list[str] = []
        self.spoke = False

    @property
    def turn_open(self) -> bool:
        return self.open_turns > 0

    @turn_open.setter
    def turn_open(self, value: bool) -> None:
        self.open_turns = max(self.open_turns, 1) if value else 0

    @staticmethod
    def daemon_ready() -> bool:
        """Is there a loaded model behind the socket right now?

        The same two checks symbio.app.daemon makes, rather than an import of
        it: a Unix socket file outlives the process that bound it, so presence
        alone is a trap after an OOM kill, and the pid has to be alive too.
        Importing daemon.py here would pull the whole package back in.
        """
        try:
            if not constants.DAEMON_SOCKET.exists():
                return False
            pid = int(constants.DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False

    def connect(self, grace: float | None = None) -> tuple[bool, str]:
        # The socket handler and the waker can both get here for one message.
        with self._connect_lock:
            return self._connect(grace)

    def _connect(self, grace: float | None) -> tuple[bool, str]:
        if self.alive() or self.connecting():
            return True, ""
        if grace is None and time.monotonic() - self.woke_at < 120:
            # The first session after a load builds its prompt from cold:
            # give it longer before suggesting another window holds it.
            grace = 45.0
        if not self.daemon_ready():
            return False, (
                "No resident model is running, so there is nothing to talk to "
                "yet. Sending a message starts one (`symb daemon start` does "
                "the same) — it loads the headmaster once, in its own process, "
                "and this window stays a few megabytes."
            )
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(constants.DAEMON_SOCKET))
        except ConnectionRefusedError:
            # A live socket that refuses means its backlog is full: the daemon
            # serves one session at a time and several are already waiting.
            return False, (
                "The resident model is up but its queue is full — it serves "
                "one window at a time and something else is holding it. Close "
                "the other tab or `symb chat` session, then send again.")
        except OSError as e:
            return False, f"The resident model is running but refused a connection: {e}"
        self.sock = sock
        self.rfile = sock.makefile("rb")
        self.wfile = sock.makefile("wb")
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._report_if_queued,
                         args=(self._QUEUE_GRACE if grace is None else grace,),
                         daemon=True).start()
        return True, ""

    def connecting(self) -> bool:
        """Connected and waiting for the session's first prompt. Seen as not
        alive, this used to open a SECOND socket on the next message: it went
        into the daemon's backlog, the queued text was written to it, and the
        conversation hung behind itself."""
        return self.sock is not None and not self.ready

    def start_waking(self, then_connect: bool = True) -> None:
        """Load the model in the background, then (if asked) open the
        session; messages said meanwhile wait in `pending`. One at a time."""
        with self._wake_lock:
            if self.waking:
                return
            self.waking = True
        threading.Thread(target=self._wake, args=(then_connect,), daemon=True).start()

    def _wake(self, then_connect: bool) -> None:
        loaded_here = daemon_state() != "ready"
        try:
            ok, why = wake_daemon(lambda text: self.send_json({"type": "waking", "text": text}))
        except Exception as e:           # never leave `waking` stuck on
            ok, why = False, f"Could not start Symbio's model: {e}"
        finally:
            with self._wake_lock:
                self.waking = False
        if ok:
            if loaded_here:
                # Only a model this bridge watched load has a cold first
                # session; one already up that is slow to answer is being
                # held by another client, and the old warning is right.
                self.woke_at = time.monotonic()
            self.send_json({"type": "awake", "text": ""})
            # A pre-warm opens no session, but a message typed while it was
            # loading is waiting in `pending`. Seen live: it waited forever.
            # Checked after `waking` is cleared, so a message arriving from
            # here on connects by itself and none falls between the two.
            if then_connect or self.pending:
                ok, why = self.connect()
        if not ok:
            self.send_json({"type": "asleep", "text": why})
            self.send_json({"type": "system", "text": why})
            if self.pending:
                self.pending.clear()
                self.open_turns = 0
                self.send_json({"type": "done", "text": ""})

    # How long a silent connection is normal. The daemon prints its banner
    # and asks for input within a second of accepting; longer than this means
    # it never accepted, because it serves ONE client at a time and another
    # window is holding it.
    _QUEUE_GRACE = 8.0

    def _report_if_queued(self, grace: float = _QUEUE_GRACE) -> None:
        if time.monotonic() - self.woke_at < 120:
            # Just woken: nothing else can be holding a model this bridge
            # started. The first conversation reads the whole system prompt
            # before it asks for input — about a minute on the 14B when there
            # is no saved prompt cache — so say that, every few seconds,
            # instead of blaming another window after 45.
            started = time.monotonic()
            while self.sock is not None and not self.ready:
                self.send_json({"type": "waking", "text": (
                    "Symbio is reading its instructions (the first conversation "
                    f"after it wakes up)… {time.monotonic() - started:.0f}s")})
                time.sleep(_WAKE_REPORT_S)
            return
        self._report_if_queued_slowly(grace)

    def _report_if_queued_slowly(self, grace: float) -> None:
        """Say when the connection is sitting in the daemon's backlog.

        connect() succeeds either way -- the kernel completes the handshake
        into the listen backlog -- so a window queued behind another one looks
        exactly like a window whose model is thinking, forever. That silence
        is the thing to name.
        """
        time.sleep(grace)
        if not self.ready and self.sock is not None:
            self.send_json({"type": "system", "text": (
                "Connected, but the resident model has not answered in "
                f"{grace:.0f}s. It serves one window at a time — "
                "another tab or a `symb chat` session is probably holding it. "
                "Close that one and this window takes over.")})

    def send_json(self, payload: dict) -> None:
        """Out to whichever browser socket is attached right now.

        Between a reload and the new socket's arrival there is no sink. A turn
        in flight keeps running -- the daemon is mid-generation and stopping it
        would waste the work -- and its frames are dropped rather than queued:
        the reconnecting page restores its own transcript, and replaying half a
        turn into it would double the text.
        """
        sink = self.sink
        if sink is None:
            return
        try:
            sink(payload)
        except Exception:
            pass

    def attach(self, send_json) -> None:
        self.sink = send_json

    def detach(self) -> None:
        self.sink = None

    def alive(self) -> bool:
        return self.sock is not None and self.ready

    def _send(self, msg: dict) -> bool:
        if not self.wfile:
            return False
        try:
            self.wfile.write((json.dumps(msg) + "\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except OSError:
            # The daemon went away under us (stopped, OOM-killed). Forget the
            # connection so the next message wakes a model, instead of raising
            # out of the socket handler and ending the window's connection.
            self._forget_connection()
            return False

    def _forget_connection(self) -> None:
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        self.sock = None
        self.rfile = None
        self.wfile = None
        self.ready = False

    def _pump(self) -> None:
        """Daemon frames in, browser frames out, until either end hangs up."""
        try:
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except ValueError:
                    continue
                kind = msg.get("type")
                if kind == "stream":
                    text = self._clean_stream(msg.get("text", ""))
                    if text:
                        self.spoke = True
                        self.send_json({"type": "token", "text": text})
                elif kind == "output":
                    # Anything that is not a token means the reply has paused,
                    # so whatever is being held back while looking for the
                    # speaker label goes out first — or the activity line
                    # would appear above text that was generated before it.
                    self._flush_prefix()
                    self.send_json({"type": "system", "text": msg.get("text", "")})
                elif kind == "confirm":
                    self._flush_prefix()
                    self.send_json({"type": "confirm", "prompt": msg.get("prompt", "")})
                elif kind == "input_prompt":
                    # The session is asking for the next message. After a turn
                    # that means the turn is over -- the text has already gone
                    # out as tokens, so this carries no body. Before the first
                    # one it means the session has finished starting up.
                    if not self.ready:
                        with self._pending_lock:
                            self.ready = True
                            queued, self.pending = self.pending, []
                            for text in queued:
                                self._send({"type": "input", "text": text})
                        self.send_json({"type": "awake", "text": ""})
                    elif self.open_turns:
                        self._end_of_turn()
                        self.open_turns -= 1
                        self._new_turn()
                        self.send_json({"type": "done", "text": ""})
                elif kind == "done":
                    self.send_json({"type": "quit"})
                    break
        except (OSError, ValueError):
            pass
        finally:
            self.send_json({"type": "system", "text": "[The resident model closed the connection.]"})
            # A turn still open will never get its prompt back: close it, or the
            # page waits on it forever. Then drop the dead socket, so the next
            # message wakes the model again rather than writing into it.
            if self.open_turns:
                self.open_turns = 0
                with self._pending_lock:
                    self.pending.clear()
                self.send_json({"type": "done", "text": ""})
            if self.rfile is not None and self.sock is not None:
                self._forget_connection()

    # How much of a turn's opening is held back while looking for the speaker
    # label. The label is short ("Caine   : "); anything longer than this is
    # the reply itself and goes out unchanged.
    _PREFIX_WINDOW = 24

    def _scrub(self, chunk: str) -> str:
        """Drop what is protocol rather than reply, across chunk boundaries.

        A tag arrives a few tokens at a time, so nothing here can match on one
        chunk: the tail of the stream is carried between calls, spans are
        tracked open-to-close, and anything inside one is never emitted. What
        the model MEANT to say is unaffected; what leaks out otherwise is the
        stop sentinel, a tool call the runtime already executed, and a
        thinking block that belongs in the folded section.
        """
        text = self.carry + chunk
        self.carry = ""
        out: list[str] = []
        while text:
            if self.inside:
                _open, close = self.inside
                index = text.find(close)
                if index == -1:
                    # Hold only as much as a closing tag could span.
                    self.carry = text[-len(close):] if len(text) > len(close) else text
                    return "".join(out)
                text = text[index + len(close):]
                self.inside = None
                continue
            starts = [(text.find(o), (o, c)) for o, c in self._SPANS if text.find(o) != -1]
            if starts:
                at, span = min(starts)
                out.append(text[:at])
                text = text[at + len(span[0]):]
                self.inside = span
                continue
            for sentinel in self._SENTINELS:
                text = text.replace(sentinel, "")
            # A partial tag at the very end is carried rather than printed;
            # otherwise "<too" reaches the reader and is corrected a token
            # later by deleting text they already saw.
            longest = max(len(o) for o, _c in self._SPANS + tuple(
                (s, s) for s in self._SENTINELS))
            tail = text[-longest:]
            cut = 0
            for i in range(len(tail)):
                fragment = tail[i:]
                if any(o.startswith(fragment) for o, _c in self._SPANS) or \
                   any(s.startswith(fragment) for s in self._SENTINELS):
                    cut = len(tail) - i
                    break
            if cut:
                self.carry = text[len(text) - cut:]
                text = text[:len(text) - cut]
            out.append(text)
            break
        return "".join(out)

    def _clean_stream(self, chunk: str) -> str:
        """One streamed chunk as the window should show it.

        The label arrives a few tokens at a time -- "Cai", "ne", "   : " -- so
        no single chunk contains it and matching per chunk finds nothing. The
        opening of a turn is buffered until either the label is found and
        dropped or enough text has arrived to prove there is none.
        """
        chunk = self._scrub(chunk)
        if not chunk:
            return ""
        if self.prefix_done:
            return chunk
        self.turn_text += chunk
        stripped = self.prefix_re.sub("", self.turn_text, count=1)
        if stripped != self.turn_text:
            self.prefix_done = True
            return stripped
        if len(self.turn_text) >= self._PREFIX_WINDOW or "\n" in self.turn_text:
            self.prefix_done = True
            return self.turn_text
        return ""      # still inside the window: held, not dropped

    def _end_of_turn(self) -> None:
        """Release what is held, and say so when a turn ended mid-block.

        An unclosed <think> is the failure where a truncated thinking block is
        shown as the answer. Emitting the held text would reproduce it, and
        dropping it silently leaves an empty bubble, so the turn says what
        happened instead.
        """
        if self.inside is not None:
            self.inside = None
            self.carry = ""
            self.turn_text = ""
            self.prefix_done = True
            self.send_json({"type": "system", "text": (
                "[The reply ended inside a thinking block, so there is no "
                "answer to show. Ask again — the model usually closes it on a "
                "retry.]")})
            return
        if self.carry:
            held, self.carry = self.carry, ""
            if not self.prefix_done:
                self.turn_text += held
            else:
                self.send_json({"type": "token", "text": held})
        self._flush_prefix()

    def _flush_prefix(self) -> None:
        """Release a turn that ended inside the label window.

        A short reply -- "Hi Sam!" -- can finish before enough text arrives to
        decide whether it started with a speaker label. Without this it would
        be held forever, which is to say silently swallowed.
        """
        if not self.prefix_done and self.turn_text:
            self.prefix_done = True
            self.send_json({"type": "token", "text": self.turn_text})

    def _new_turn(self) -> None:
        self.turn_text = ""
        self.prefix_done = False
        self.carry = ""
        self.inside = None
        self.spoke = False

    def say(self, text: str) -> bool:
        """Send, or hold until the session is listening. False when the
        daemon connection turned out to be dead; the text is held for the
        next one."""
        if not self.open_turns:
            self._new_turn()     # not mid-stream: a queued turn resets at its start
        self.open_turns += 1
        with self._pending_lock:
            if not self.ready:
                self.pending.append(text)
                return True
            if self._send({"type": "input", "text": text}):
                return True
            self.pending.append(text)
            return False

    def confirm(self, approved: bool) -> None:
        self._send({"type": "confirm", "answer": bool(approved)})

    def close(self) -> None:
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass


# ── HTTP ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "Symbio/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - quiet by default
        if os.environ.get("SYMBIO_DESKTOP_VERBOSE"):
            super().log_message(fmt, *args)

    # -- routing --

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        path = self.path.split("?", 1)[0]
        if path == "/ws/chat":
            return self._websocket()
        if path == "/api/ecosystem":
            return self._json(get_ecosystem())
        if path == "/api/health":
            return self._json(get_health())
        if path == "/api/settings":
            return self._json(get_settings())
        if path == "/api/sessions":
            query = self.path.partition("?")[2]
            wanted = ""
            for pair in query.split("&"):
                key, _, value = pair.partition("=")
                if key == "id":
                    wanted = urllib.parse.unquote(value)
            return self._json(get_sessions(session_id=wanted))
        if path.startswith("/api/adapter/"):
            role = path.rsplit("/", 1)[-1]
            return self._json(_adapter_info(None if role == "headmaster" else role))
        if path in ("/", "/index.html"):
            return self._file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            return self._file(STATIC_DIR / path[len("/static/"):])
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        if self.path.split("?", 1)[0] != "/api/settings":
            return self.send_error(404, "Not found")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            return self._json({"ok": False, "error": "Malformed request."})
        if not isinstance(body, dict):
            return self._json({"ok": False, "error": "Malformed request."})
        return self._json(set_settings(body))

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        # Resolved and checked against the static root: this serves whatever
        # the URL names, and a path with .. in it would otherwise name
        # anything on the disk. The server binds to localhost, but "only I can
        # reach it" is not a reason to serve arbitrary files to it.
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
            body = resolved.read_bytes()
        except (OSError, ValueError):
            return self.send_error(404, "Not found")
        kind = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # -- websocket --

    def _websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or self.headers.get("Upgrade", "").lower() != "websocket":
            return self.send_error(400, "Expected a WebSocket upgrade")
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", ws_accept_key(key))
        self.end_headers()

        sock = self.connection
        write_lock = threading.Lock()

        def send_json(payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            try:
                with write_lock:
                    sock.sendall(ws_frame(data))
            except OSError:
                pass

        config = _config_summary()
        state = daemon_state()
        send_json({"type": "connected", "model_state": state,
                   **{k: config[k] for k in ("assistant_name", "user_name", "model_name")}})

        global _BRIDGE, _PREWARMED
        with _BRIDGE_LOCK:
            if _BRIDGE is not None and (_BRIDGE.alive() or _BRIDGE.connecting()
                                        or _BRIDGE.waking):
                bridge = _BRIDGE
                bridge.attach(send_json)
                send_json({"type": "system", "text":
                           "[Reattached to the conversation already running.]"})
            else:
                bridge = DaemonBridge(send_json, config.get("assistant_name", ""))
                _BRIDGE = bridge
            # Opening the window is the intent to chat, so the model starts
            # loading now, not when the first message is sent: half a minute
            # of the wait happens while the reader is still typing. Once per
            # window: a reconnect after a sleep must not reload a model that
            # was stopped on purpose. Loading only — no session is opened.
            prewarm = state == "down" and not _PREWARMED
            _PREWARMED = True
        if prewarm:
            bridge.start_waking(then_connect=False)

        def ensure_bridge() -> tuple[bool, str]:
            """Attach to the resident model on demand.

            Not at page load: a browser that reconnects its socket -- a
            refresh, a sleep/wake, the backoff loop after a restart -- would
            take one of the daemon's queue slots each time and hold it for a
            window nobody is typing in. With no model up yet, it is woken in
            the background and the message waits for it.
            """
            if bridge.alive() or bridge.connecting():
                return True, ""
            if DaemonBridge.daemon_ready():
                return bridge.connect()
            bridge.start_waking()
            return True, ""

        try:
            while True:
                opcode, payload = ws_read_message(sock)
                if opcode == 0x8:          # close
                    break
                if opcode == 0x9:          # ping
                    with write_lock:
                        sock.sendall(ws_frame(payload, opcode=0xA))
                    continue
                if opcode != 0x1:
                    continue
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except ValueError:
                    continue
                kind = msg.get("type")
                if kind == "chat":
                    text = (msg.get("message") or "").strip()
                    if not text:
                        continue
                    ready, why = ensure_bridge()
                    if not ready:
                        send_json({"type": "system", "text": why})
                        send_json({"type": "done", "text": ""})
                        continue
                    if not bridge.say(text):
                        # The daemon died since the last turn: the text is
                        # held, and a fresh model is woken for it.
                        bridge.start_waking()
                    elif bridge.pending and not (bridge.connecting() or bridge.waking):
                        bridge.start_waking()
                elif kind == "confirm_response":
                    bridge.confirm(msg.get("approved", False))
                elif kind == "ping":
                    send_json({"type": "pong"})
        except (WSError, OSError):
            pass
        finally:
            # Detach, do not close: the page may be reloading, and the next
            # socket picks this conversation up where it left off. The daemon
            # connection is released when the server exits, or when the daemon
            # itself hangs up.
            bridge.detach()


def serve(host: str = "127.0.0.1", port: int = 8742) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.serve_forever()
