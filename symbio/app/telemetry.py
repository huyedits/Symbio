"""Anonymous telemetry + /feedback client.

Off by default and never sends anything until the user has answered the Y/N
consent prompt (run_setup_wizard or /telemetry). With no `telemetry.endpoint`
configured, records are appended to local JSONL files under `telemetry/` and
nothing leaves the machine. With an endpoint set, records are POSTed to it
(a Cloudflare Worker that files /feedback as GitHub issues and appends
telemetry to a repo file), authenticated with a shared secret header.

Privacy contract (enforced in collect_env): the payload contains ONLY
anonymous environment/usage facts. It NEVER includes the user's name, the
assistant's name, conversation text, note contents, prompts, or file paths.
"""

from __future__ import annotations

import json
import platform
import sys
import urllib.error
import urllib.request
from datetime import date, datetime
from typing import Any

from symbio import constants

# Local store + state live next to config.json, never committed (.gitignore).
_TELEMETRY_DIR = constants.PROJECT_DIR / "telemetry"
_STATE_FILE = _TELEMETRY_DIR / "state.json"

_POST_TIMEOUT = 10.0


def _version() -> str:
    try:
        from importlib import metadata
        return metadata.version("symbio")
    except Exception:
        return "dev"


def collect_env(config: dict[str, Any]) -> dict[str, Any]:
    """Build the anonymous environment block shared by every record.

    Deliberately excludes any user-identifying or content-bearing fields: no
    user_name, assistant_name, message text, note contents, prompts, paths.
    """
    agent = config.get("agent", {})
    lora = config.get("lora", {})
    tools = config.get("tools", {})
    return {
        "symbio_version": _version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python": ".".join(map(str, sys.version_info[:3])),
        "model_name": config.get("model_name", ""),
        "lora_rank": lora.get("rank"),
        "lora_loaded": bool(config.get("model", {}).get("allow_lora"))
        and bool(lora.get("rank")),
        "tool_groups": list(tools.get("enabled_groups", [])),
        "speed_mode": agent.get("speed_mode", ""),
    }


def _new_state() -> dict[str, Any]:
    return {"session_count": 0, "error_count": 0, "last_ping_date": ""}


def load_state() -> dict[str, Any]:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _new_state()


def save_state(state: dict[str, Any]) -> None:
    try:
        _TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, indent=2) + "\n",
                               encoding="utf-8")
    except Exception:
        pass


def record_session(state: dict[str, Any]) -> None:
    state["session_count"] = int(state.get("session_count", 0)) + 1


def record_error(state: dict[str, Any]) -> None:
    state["error_count"] = int(state.get("error_count", 0)) + 1


def _today() -> str:
    return date.today().isoformat()


def _save_local(payload: dict[str, Any], kind: str) -> tuple[bool, str]:
    """Append a record to telemetry/<kind>.jsonl (no network)."""
    try:
        _TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        path = _TELEMETRY_DIR / f"{kind}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        return True, f"saved locally to {path}"
    except Exception as e:
        return False, f"could not save locally: {e}"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_feedback_block(payload: dict[str, Any]) -> str:
    """Human-readable feedback block appended to telemetry/feedback.txt —
    designed to be read as-is, shared via a PR, or pasted into a Discussion."""
    parts: list[str] = [_now_iso()]
    if payload.get("session_count") is not None:
        parts.append(f"session_count={payload['session_count']}")
    env = payload.get("env") or {}
    if isinstance(env, dict):
        for k, v in env.items():
            if isinstance(v, list):
                v = ",".join(str(x) for x in v)
            parts.append(f"{k}={v}")
    text = (payload.get("text") or "").strip()
    return f"=== {' | '.join(parts)} ===\n{text}\n---\n"


def _save_feedback_text(payload: dict[str, Any]) -> tuple[bool, str]:
    """Append a human-readable feedback block to telemetry/feedback.txt (no
    network). The file is the artifact the user submits — via a PR or by
    pasting into a GitHub Discussion."""
    try:
        _TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        path = _TELEMETRY_DIR / "feedback.txt"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(format_feedback_block(payload))
        return True, f"saved locally to {path} (share via PR or Discussion)"
    except Exception as e:
        return False, f"could not save locally: {e}"


def send(payload: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    """Send a record. With no endpoint, falls back to local JSONL. Returns
    (ok, human-readable message)."""
    tcfg = config.get("telemetry", {})
    endpoint = (tcfg.get("endpoint") or "").strip()
    if not endpoint:
        # No server: feedback -> a human-readable .txt the user submits by hand;
        # telemetry -> machine-readable ndjson. Nothing leaves the machine.
        if payload.get("type") == "feedback":
            return _save_feedback_text(payload)
        return _save_local(payload, "pings")
    kind = "pings" if payload.get("type") == "telemetry" else "feedback"

    body = json.dumps(payload, default=str).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Telemetry-Secret": tcfg.get("shared_secret", "") or "",
        "X-Telemetry-Kind": kind,
    }
    req = urllib.request.Request(endpoint, data=body, headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_POST_TIMEOUT) as resp:
            resp.read(4096)
        return True, "sent"
    except urllib.error.HTTPError as e:
        return False, f"server rejected (HTTP {e.code})"
    except Exception as e:
        return False, f"send failed: {e}"


def send_feedback(text: str, config: dict[str, Any],
                  state: dict[str, Any]) -> tuple[bool, str]:
    """Send a /feedback record (type=feedback), bundling anonymous env info."""
    text = (text or "").strip()
    if not text:
        return False, "feedback text is empty"
    payload = {
        "type": "feedback",
        "text": text,
        "env": collect_env(config),
        "session_count": int(state.get("session_count", 0)),
    }
    return send(payload, config)


def maybe_daily_ping(config: dict[str, Any],
                     state: dict[str, Any]) -> tuple[bool, str] | None:
    """Send one telemetry ping per day at most, if consented+enabled+ping_daily.

    Returns the send result, or None if no ping was attempted (disabled, or
    already pinged today). Always called from chat startup; cheap when idle.
    """
    tcfg = config.get("telemetry", {})
    if not (tcfg.get("enabled") and tcfg.get("consented")
            and tcfg.get("ping_daily", True)):
        return None
    today = _today()
    if state.get("last_ping_date") == today:
        return None
    payload = {
        "type": "telemetry",
        "env": collect_env(config),
        "session_count": int(state.get("session_count", 0)),
        "error_count": int(state.get("error_count", 0)),
    }
    result = send(payload, config)
    # Only stamp the date on a successful send (or a successful local save) so
    # a transient network failure retries the next session, not the next day.
    if result[0]:
        state["last_ping_date"] = today
        save_state(state)
    return result


def consent_summary(config: dict[str, Any]) -> str:
    """The exact data-disclosure block shown before the Y/N consent prompt,
    so what the user sees and what is actually sent can never drift apart."""
    return (
        "Telemetry — anonymous usage data helps improve Symbio.\n"
        "  /feedback writes a local file (telemetry/feedback.txt) you can share\n"
        "  via a PR or GitHub Discussion; with telemetry.endpoint set it is POSTed\n"
        "  to your server instead. Data collected (NO message/note/prompt text,\n"
        "  NO your name):\n"
        "    - Symbio version, OS, Python version, model_name, LoRA rank\n"
        "    - enabled tool groups, speed_mode, session count, tool-error count\n"
        "  You can say No and keep using Symbio. /telemetry re-asks anytime."
    )