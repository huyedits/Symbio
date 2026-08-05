"""Local, human-readable telemetry log.

Appends one timestamped line per event to ``telemetry/activity.txt`` (next
to config.json, gitignored, never committed). Unlike the anonymous/consent
gated remote telemetry in ``telemetry.py``, this is a simple local audit
log of what the assistant does each turn — handy for debugging behaviour
like tool-use regressions.

On by default. Disable with ``telemetry.local_log = false`` in config.json.
The enable flag is re-read when config.json's mtime changes, so /config set
toggles it live without a restart.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

from symbio import constants

_LOG_DIR = constants.PROJECT_DIR / "telemetry"
_LOG_FILE = _LOG_DIR / "activity.txt"
_LOCK = threading.Lock()
_MAX_FIELD = 120

# mtime-cached enable flag so we don't parse config.json on every event but
# still pick up live /config set changes.
_cfg_cache: dict[str, Any] = {"mtime": None, "enabled": True}


def _enabled() -> bool:
    try:
        p = constants.PROJECT_DIR / "config.json"
        m = p.stat().st_mtime
        if _cfg_cache["mtime"] != m:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            _cfg_cache["enabled"] = bool(cfg.get("telemetry", {}).get("local_log", True))
            _cfg_cache["mtime"] = m
    except Exception:
        _cfg_cache["enabled"] = True
    return _cfg_cache["enabled"]


def _truncate(v: Any) -> str:
    s = str(v).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= _MAX_FIELD else s[:_MAX_FIELD - 1] + "…"


def log_event(kind: str, **fields: Any) -> None:
    """Append one ``[timestamp] kind k=v ...`` line to the local telemetry .txt."""
    if not _enabled():
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = " ".join(f"{k}={_truncate(v)}" for k, v in fields.items())
    line = f"[{ts}] {kind}"
    if body:
        line += " " + body
    line += "\n"
    with _LOCK:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Telemetry must never break a turn.
            pass


def log_path() -> str:
    """Return the absolute path of the telemetry .txt (for /status etc.)."""
    return str(_LOG_FILE)