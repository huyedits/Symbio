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
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
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


# --------------------------------------------------------------------------
# Reading it back.
#
# Everything above has been writing since day one and nothing ever read a line
# of it — log_path() was defined and called from nowhere. activity.txt reached
# 985 KB and ~13,000 events on this machine, including a MEDIUM-risk security
# alert for a command that arrived through prompt injection: recorded at the
# moment it happened, and never seen, because there was no way to ask.
#
# A write-only log is not telemetry, it is a disk cost.

_LINE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\] (?P<kind>\w+)(?P<rest> .*)?$")
_FIELD_RE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")


def _parse_line(line: str) -> dict[str, Any] | None:
    m = _LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    out: dict[str, Any] = {"ts": m.group("ts"), "kind": m.group("kind")}
    for k, v in _FIELD_RE.findall((m.group("rest") or "").strip()):
        out[k] = v.strip()
    return out


def read_events(since: datetime | None = None,
                path: str | None = None) -> list[dict[str, Any]]:
    """Parse activity.txt into event dicts, oldest first."""
    target = Path(path) if path else _LOG_FILE
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(target, encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = _parse_line(line)
            if ev is None:
                continue
            if since is not None:
                try:
                    if datetime.strptime(ev["ts"], "%Y-%m-%d %H:%M:%S") < since:
                        continue
                except ValueError:
                    continue
            events.append(ev)
    return events


def summarise(days: int | None = None,
              path: str | None = None) -> dict[str, Any]:
    """Aggregate the activity log into the questions worth asking of it.

    The per-tool success rate is the one that matters. It is the same thing
    tool_eval.py measures in a lab, measured instead against what actually
    happened. A tool with a low rate is failing in real use; a tool that never
    appears at all is one the model is not reaching, which is the failure that
    leaves no other trace.
    """
    since = datetime.now() - timedelta(days=days) if days is not None else None
    events = read_events(since=since, path=path)

    tools: dict[str, dict[str, int]] = {}
    alerts: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for ev in events:
        counts[ev["kind"]] = counts.get(ev["kind"], 0) + 1
        if ev["kind"] != "tool":
            continue
        slot = tools.setdefault(ev.get("name", "?"), {"calls": 0, "ok": 0})
        slot["calls"] += 1
        if str(ev.get("ok", "")).lower() == "true":
            slot["ok"] += 1
        result = ev.get("result", "")
        if "Security alert" in result:
            alerts.append(ev)
        if result.startswith(("Blocked", "Refused")) or "denied" in result.lower():
            blocked.append(ev)

    return {
        "events": len(events),
        "counts": counts,
        "turns": counts.get("turn", 0),
        "tools": dict(sorted(tools.items(),
                             key=lambda kv: kv[1]["calls"], reverse=True)),
        "alerts": alerts,
        "blocked": blocked,
        # The file actually read, not the default — reporting _LOG_FILE while
        # summarising a different path names a file the numbers did not come
        # from, which is worse than naming none.
        "path": str(Path(path) if path else _LOG_FILE),
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "n/a"


def format_summary(report: dict[str, Any]) -> str:
    """Render summarise() for a terminal."""
    if not report["events"]:
        return f"  No activity recorded yet ({report['path']})."

    lines = [f"  {report['events']} events · {report['turns']} turns"]

    if report["tools"]:
        total = sum(t["calls"] for t in report["tools"].values())
        ok = sum(t["ok"] for t in report["tools"].values())
        lines.append(f"\n  Tool calls: {total} ({_pct(ok, total)} succeeded)")
        for name, t in report["tools"].items():
            lines.append(f"    {name:18s} {t['calls']:5d} calls  "
                         f"{_pct(t['ok'], t['calls']):>4s} ok")
    else:
        lines.append("\n  No tool calls recorded — the model is not reaching "
                     "its tools at all.")

    if report["alerts"]:
        lines.append(f"\n  Security alerts: {len(report['alerts'])}")
        for ev in report["alerts"][-3:]:
            lines.append(f"    [{ev['ts']}] {ev.get('name', '?')}: "
                         f"{ev.get('result', '')[:90]}")
    if report["blocked"]:
        lines.append(f"\n  Blocked or declined: {len(report['blocked'])}")

    lines.append(f"\n  {report['path']}")
    return "\n".join(lines)