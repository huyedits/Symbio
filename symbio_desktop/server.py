"""Symbio Desktop server — FastAPI backend that exposes the adapter/RAG ecosystem
as a REST API and a WebSocket chat endpoint consumed by the mind-map frontend."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from symbio import constants
from symbio.app.config import load_config
from symbio.app.dispatch import load_catalog

APP_DIR = Path(__file__).parent.resolve()
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="Symbio Desktop", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        except OSError:
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
            lines = train_file.read_text(encoding="utf-8").strip().splitlines()
            stats["training_samples"] = len(lines)
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


# ── REST API routes ──────────────────────────────────────────────────

@app.get("/api/ecosystem")
def get_ecosystem() -> dict[str, Any]:
    catalog = load_catalog()
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
                      "size_mb": rag["training_size_mb"], "auto_train": config["auto_train"]},
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/adapter/{role}")
def get_adapter(role: str) -> dict[str, Any]:
    if role == "headmaster":
        return _adapter_info(role=None)
    return _adapter_info(role=role)


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    skills = _skill_notes()
    total_errors = sum(s["error_count"] for s in skills)
    total_corrections = sum(s["correction_count"] for s in skills)
    return {
        "total_skills": len(skills),
        "total_errors": total_errors,
        "total_corrections": total_corrections,
        "skills_with_errors": [s["name"] for s in skills if s["error_count"] > 0],
        "skills_with_corrections": [s["name"] for s in skills if s["correction_count"] > 0],
    }


# ═══════════════════════════════════════════════════════════════════════
# WebSocket Chat
# ═══════════════════════════════════════════════════════════════════════

class DesktopChatSession:
    """Wraps a Symbio ChatSession for the desktop WebSocket frontend.

    The native ChatSession.run() is a blocking readline loop. This adapter
    calls _agent_turn() directly per inbound WebSocket message and routes
    all output (system messages, streaming tokens, confirmations) back over
    the wire as typed JSON frames.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.config = load_config()
        self._session: Any = None
        self._session_lock = threading.Lock()
        self._reply_buffer: list[str] = []
        self._confirm_response: bool | None = None
        self._confirm_event = threading.Event()
        self._progress_buf: list[str] = []
        self._progress_last_flush = 0.0

    def _output_fn(self, text: str):
        """System messages (tool observations, status lines)."""
        if text.strip():
            asyncio.run_coroutine_threadsafe(
                self._safe_send({"type": "system", "text": text}), _LOOP)

    def _stream_chunk_fn(self, chunk: str):
        """Streaming tokens from the model."""
        self._reply_buffer.append(chunk)
        asyncio.run_coroutine_threadsafe(
            self._safe_send({"type": "token", "text": chunk}), _LOOP)

    def _confirm_fn(self, prompt: str) -> bool:
        """Ask the user to confirm an action. Blocks until the client responds."""
        self._confirm_response = None
        self._confirm_event.clear()
        asyncio.run_coroutine_threadsafe(
            self._safe_send({"type": "confirm", "prompt": prompt}), _LOOP)
        self._confirm_event.wait(timeout=60)
        return self._confirm_response or False

    async def _safe_send(self, data: dict[str, Any]):
        try:
            await self.ws.send_json(data)
        except Exception:
            pass

    def _progress_write(self, text: str):
        """Called from the captured stdout/stderr during blocking ops.

        tqdm progress bars write bare \\r lines; the spinner writes
        animation frames. We batch them and flush every ~200ms so the
        UI gets smooth updates without flooding the WebSocket."""
        if not text:
            return
        # tqdm uses \\r to overwrite the same line — treat each \\r as a
        # new frame and keep only the last one per flush window.
        self._progress_buf.append(text)
        now = time.time()
        if now - self._progress_last_flush >= 0.2:
            self._flush_progress()

    def _flush_progress(self):
        if not self._progress_buf:
            return
        merged = "".join(self._progress_buf)
        self._progress_buf.clear()
        self._progress_last_flush = time.time()
        import re as _re
        # Strip ANSI escapes
        merged = _re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', merged)
        # tqdm writes frames separated by \r — take the last complete frame
        segments = [s.strip() for s in merged.split('\r') if s.strip()]
        if segments:
            cleaned = segments[-1]
        else:
            lines = [l.strip() for l in merged.split('\n') if l.strip()]
            cleaned = '\n'.join(lines[-5:])
        if cleaned:
            asyncio.run_coroutine_threadsafe(
                self._safe_send({"type": "progress", "text": cleaned}), _LOOP)

    class _ProgressCapture:
        """Context manager that redirects stdout/stderr to capture_fn.

        The original streams still receive output (so nothing is lost),
        but every write is also forwarded to the progress callback."""
        def __init__(self, capture_fn):
            self._fn = capture_fn
            self._stdout = None
            self._stderr = None

        def __enter__(self):
            import sys
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            fn = self._fn

            class _TeeOut:
                def write(self, text):
                    self._orig.write(text)
                    fn(text)
                def flush(self):
                    self._orig.flush()
                def __getattr__(self, name):
                    return getattr(self._orig, name)

            class _TeeErr:
                def write(self, text):
                    self._orig.write(text)
                    fn(text)
                def flush(self):
                    self._orig.flush()
                def __getattr__(self, name):
                    return getattr(self._orig, name)

            out = _TeeOut()
            out._orig = self._stdout
            err = _TeeErr()
            err._orig = self._stderr
            sys.stdout = out
            sys.stderr = err
            return self

        def __exit__(self, *args):
            import sys
            sys.stdout = self._stdout
            sys.stderr = self._stderr

    def _ensure_session(self):
        if self._session is not None:
            return
        with self._session_lock:
            if self._session is not None:
                return
            from symbio.app.chat import ChatSession
            from symbio.app.setup import ensure_identity_defaults

            ensure_identity_defaults(self.config)
            session = ChatSession(
                self.config,
                input_fn=lambda _="": "",
                output_fn=self._output_fn,
                confirm_fn=self._confirm_fn,
                stream_chunk_fn=self._stream_chunk_fn,
                stream_prefix=False,
                owner="desktop",
            )
            # Model loading writes tqdm progress bars to stderr — capture
            # them so the UI shows "Fetching 7 files…" instead of silence.
            with self._ProgressCapture(self._progress_write):
                session._ensure_model_loaded()
            self._flush_progress()
            self._session = session

    def process_message(self, text: str) -> str:
        """Run one agent turn. Returns the full reply text."""
        self._ensure_session()
        self._reply_buffer = []
        try:
            # The spinner and "thinking…" lines write to stdout — capture
            # them so the UI shows generation progress.
            with self._ProgressCapture(self._progress_write):
                self._session._agent_turn(text)
            self._flush_progress()
        except Exception as e:
            tb = traceback.format_exc()
            asyncio.run_coroutine_threadsafe(
                self._safe_send({"type": "error", "text": f"{e}\n{tb}"}), _LOOP)
        return "".join(self._reply_buffer)

    def handle_confirm(self, approved: bool):
        self._confirm_response = approved
        self._confirm_event.set()

    def handle_command(self, cmd: str) -> str | None:
        """Handle a slash command. Returns a quit signal string if the session ended."""
        self._ensure_session()
        from symbio.app.chat import _QUIT
        try:
            result = self._session._handle_command(cmd)
            if result == _QUIT:
                return "quit"
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                self._safe_send({"type": "error", "text": str(e)}), _LOOP)
        return None


# Global event loop reference for cross-thread scheduling
_LOOP: asyncio.AbstractEventLoop | None = None

# One chat session per WebSocket connection
_sessions: dict[str, DesktopChatSession] = {}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    global _LOOP
    _LOOP = asyncio.get_running_loop()

    await ws.accept()
    session = DesktopChatSession(ws)
    sid = str(id(session))
    _sessions[sid] = session

    config = _config_summary()
    await ws.send_json({
        "type": "connected",
        "assistant_name": config["assistant_name"],
        "user_name": config["user_name"],
        "model_name": config["model_name"],
    })

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "chat":
                text = data.get("message", "").strip()
                if not text:
                    continue
                if text.startswith("/"):
                    result = session.handle_command(text)
                    if result == "quit":
                        await ws.send_json({"type": "quit"})
                        break
                else:
                    # Run the turn in a thread so we don't block the event loop
                    loop = asyncio.get_running_loop()
                    reply = await loop.run_in_executor(None, session.process_message, text)
                    await ws.send_json({"type": "done", "text": reply})

            elif msg_type == "confirm_response":
                approved = data.get("approved", False)
                session.handle_confirm(approved)

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        _sessions.pop(sid, None)


# ── static files ─────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8742, log_level="info")


if __name__ == "__main__":
    main()
