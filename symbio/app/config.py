"""Configuration: defaults, loading, and model-driven self-configuration."""

import copy
import json
import os
from typing import Any

from symbio import constants

DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "assistant_name": "",
    "user_name": "",
    "lora": {
        "rank": 8,
        "dropout": 0.0,
        "scale": 20.0,
        "num_layers": 8,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "iters": 300,
        "max_seq_length": 512,
        "steps_per_eval": 100,
        "save_every": 100,
    },
    "agent": {
        "max_tool_rounds": 5,
        "history_limit": 40,
        "sandbox_timeout": 30,
        "code_timeout": 60,
        "max_output_len": 4000,
        "max_reply_tokens": 600,
        "temperature": 0.7,
        "top_p": 0.9,
        "cron_poll_seconds": 20,
        "stream_output": True,
        "prompt_cache_enabled": True,
        # How long a chat front-end should wait before showing a "thinking…"
        # placeholder if the model has not emitted a visible token yet.
        "first_chunk_timeout_ms": 1500,
        # Maximum character budget for the retained conversation window. One
        # giant observation (e.g. a full web page) can otherwise bloat every
        # later turn even with a turn-count history limit.
        "max_history_chars": 12000,
        # Lower temperature during tool-use rounds makes the model follow the
        # prompt's tag rules (browse vs cmd, press vs fake keydown) more
        # strictly instead of drifting into prose or hallucinated commands.
        "tool_use_temperature": 0.35,
    },
    "browser": {
        # Browser automation is off by default. When enabled, the agent launches
        # its own isolated Google Chrome / Chromium via Playwright, not the
        # user's personal browser profile. It must still ask for confirmation
        # the first time it visits a new domain.
        "enabled": False,
    },
    "web": {
        "search_results": 5,
        "http_timeout": 15,
        "auto_search_when_unsure": True,
        "auto_search_session_cap": 20,
    },
    "sandbox": {
        "blocked_commands": [
            "rm", "sudo", "su", "dd", "mkfs", "fdisk", "mount", "umount",
            "chmod", "chown", "curl", "wget", "ssh", "scp", "bash", "sh", "zsh",
            "fish", "python", "python3", "perl", "ruby", "php", "node", "npm",
        ],
        "blocked_imports": [
            "os", "sys", "subprocess", "pathlib", "shutil", "socket", "http",
            "urllib", "ftplib", "smtplib", "imaplib", "pickle", "ctypes",
            "multiprocessing", "threading", "tempfile", "asyncio", "importlib",
            "pkgutil", "site", "builtins",
        ],
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "allowed_chat_ids": [],
        "confirm_dangerous": True,
    },
    "rag": {
        "enabled": True,
        "top_k": 5,
        "max_context_tokens": 1500,
        "sources": ["notes", "sessions"],
    },
    "memory": {
        "enabled": True,
        "memory_char_limit": 2200,
        "profile_char_limit": 1375,
        "nudge_interval": 10,
        "flush_min_turns": 6,
    },
    "learn": {
        "enabled": True,
        "auto": True,
        "auto_train": True,
        "remember_research": True,
        "note_decay_days": 90,
        "mistake_threshold": 5,
        "batch_train_iters": 25,
        "iters_per_severity": 5,
        "max_batch_train_iters": 100,
        "boost_factor": 3,
        "severe_correction_phrases": [
            "wrong", "incorrect", "you misunderstood", "fix it", "not what",
        ],
        "golden_set_enabled": True,
        "golden_max_tokens": 150,
        "golden_regression_threshold": 0,
        "golden_rollback_on_regression": True,
        "adapter_idle_reminder_enabled": True,
        "adapter_idle_days": 30,
        "correction_phrases": [
            "no,", "not quite", "that's wrong", "incorrect", "wrong",
            "you misunderstood", "try again", "actually", "i meant",
            "i said", "i asked", "not what", "that's not", "you're wrong",
            "fix it", "correction", "rephrase",
        ],
    },
    "tools": {
        "enabled_groups": [
            "memory", "notes", "terminal", "code", "web_search",
            "digest", "train", "cron", "config", "delegate",
        ],
    },
    "dispatch": {
        # Off by default: MoA delegation loads and runs additional models
        # on your machine, which is a bigger behavior/resource change than
        # anything else here — opt in deliberately.
        "enabled": False,
        "max_resident_workers": 1,
        "worker_idle_unload_minutes": 10,
        "max_worker_rounds": 4,
        "worker_golden_set_enabled": True,
        "worker_golden_regression_threshold": 0,
        "worker_golden_rollback_on_regression": True,
    },
}

# Keys that must survive a restart to take effect.
_RESTART_KEYS = {"model_name"}


def _env_list(key: str) -> list[str] | None:
    """Parse a comma-separated env var into a list, or None if unset/empty."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Let environment variables override key identity/model/secret settings."""
    if os.environ.get("SYMBIO_MODEL_NAME", "").strip():
        config["model_name"] = os.environ["SYMBIO_MODEL_NAME"].strip()
    if os.environ.get("SYMBIO_ASSISTANT_NAME", "").strip():
        config["assistant_name"] = os.environ["SYMBIO_ASSISTANT_NAME"].strip()
    if os.environ.get("SYMBIO_USER_NAME", "").strip():
        config["user_name"] = os.environ["SYMBIO_USER_NAME"].strip()

    token = os.environ.get("SYMBIO_TELEGRAM_TOKEN", "").strip()
    if token:
        config.setdefault("telegram", {})
        config["telegram"]["bot_token"] = token
        config["telegram"]["enabled"] = True

    allowed = _env_list("SYMBIO_TELEGRAM_ALLOWED_CHAT_IDS")
    if allowed:
        config.setdefault("telegram", {})
        try:
            config["telegram"]["allowed_chat_ids"] = [int(x) for x in allowed]
        except ValueError:
            print("[Config warning] SYMBIO_TELEGRAM_ALLOWED_CHAT_IDS contains non-integer values; ignored.")


def load_config() -> dict[str, Any]:
    """Load config.json if present; merge with sensible defaults and env overrides."""
    # Deep copy: callers (e.g. set_config_value) mutate nested sections, and a
    # shallow copy would poison DEFAULT_CONFIG for every later load.
    config = copy.deepcopy(DEFAULT_CONFIG)
    if constants.CONFIG_FILE.exists():
        try:
            user_config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(user_config)
            for section in ("lora", "agent", "rag", "memory", "web", "sandbox", "learn", "telegram", "tools", "dispatch"):
                if section in user_config:
                    config[section] = {**DEFAULT_CONFIG[section], **user_config[section]}
        except Exception as e:
            print(f"[Config warning] Could not read {constants.CONFIG_FILE}: {e}")
    _apply_env_overrides(config)
    return config


def config_show(config: dict[str, Any]) -> str:
    """Return config as pretty JSON, with sensitive values redacted."""
    safe = copy.deepcopy(config)
    if safe.get("telegram", {}).get("bot_token"):
        safe["telegram"]["bot_token"] = "***REDACTED***"
    return json.dumps(safe, indent=2)


def _coerce_like(current: Any, raw: str) -> Any:
    """Parse raw into the same type as the current value."""
    if isinstance(current, bool):
        if raw.lower() in ("true", "yes", "on", "1"):
            return True
        if raw.lower() in ("false", "no", "off", "0"):
            return False
        raise ValueError(f"Expected true/false, got {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError("Expected a JSON list")
        return value
    return raw


def set_config_value(config: dict[str, Any], key: str, raw_value: str,
                     allow_sandbox: bool = False) -> str:
    """Set a dotted config key (e.g. agent.temperature), persist it to
    config.json, and apply it to the running config. Returns a status message."""
    key = key.strip()
    if key.startswith("sandbox.") and not allow_sandbox:
        return "sandbox.* settings can only be changed by the user via /config set."

    # Resolve the dotted path against the live config to validate it exists.
    parts = key.split(".")
    node: Any = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return f"Unknown config key: {key}"
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node or isinstance(node[leaf], dict):
        return f"Unknown config key: {key}"

    try:
        value = _coerce_like(node[leaf], raw_value.strip())
    except Exception as e:
        return f"Bad value for {key}: {e}"
    node[leaf] = value

    # Persist into config.json without disturbing unrelated user settings.
    user_config: dict[str, Any] = {}
    if constants.CONFIG_FILE.exists():
        try:
            user_config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            user_config = {}
    target = user_config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[leaf] = value
    constants.CONFIG_FILE.write_text(json.dumps(user_config, indent=2) + "\n", encoding="utf-8")

    note = " (takes effect after restart)" if parts[0] in _RESTART_KEYS else ""
    return f"Set {key} = {value!r}{note}."


_TELEGRAM_TOKEN_ENV = "SYMBIO_TELEGRAM_TOKEN"


def get_telegram_token(config: dict[str, Any], input_fn=input) -> str | None:
    """Return the Telegram bot token, in order of preference:
    1. SYMBIO_TELEGRAM_TOKEN environment variable
    2. config["telegram"]["bot_token"]
    3. Prompt the user and persist to config.json
    Returns None if the user declines to provide a token.
    """
    token = os.environ.get(_TELEGRAM_TOKEN_ENV, "").strip()
    if token:
        return token
    token = (config.get("telegram", {}) or {}).get("bot_token", "").strip()
    if token:
        return token
    try:
        token = input_fn(
            "Enter your Telegram bot token from @BotFather (or press Enter to skip): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not token:
        return None
    if not constants.CONFIG_FILE.exists():
        constants.CONFIG_FILE.write_text("{}", encoding="utf-8")
    try:
        user_config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        user_config = {}
    user_config.setdefault("telegram", {})["bot_token"] = token
    constants.CONFIG_FILE.write_text(json.dumps(user_config, indent=2) + "\n", encoding="utf-8")
    print("[Telegram] Token saved to config.json. Consider using SYMBIO_TELEGRAM_TOKEN instead.")
    return token
