"""Configuration: defaults, loading, and model-driven self-configuration."""

import copy
import json
from typing import Any

from symbio import constants

DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "Qwen/Qwen3-0.6B",
    "assistant_name": "Caine",
    "user_name": "Huy",
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
    },
    "web": {
        "search_results": 5,
        "http_timeout": 15,
        "auto_search_when_unsure": True,
    },
    "sandbox": {
        "blocked_commands": [
            "rm", "sudo", "su", "dd", "mkfs", "fdisk", "mount", "umount",
            "chmod", "chown", "wget", "ssh", "scp", "bash", "sh", "zsh",
            "fish", "python", "python3", "perl", "ruby", "php", "node", "npm",
        ],
        "blocked_imports": [
            "os", "sys", "subprocess", "pathlib", "shutil", "socket", "http",
            "urllib", "ftplib", "smtplib", "imaplib", "pickle", "ctypes",
            "multiprocessing", "threading", "tempfile", "asyncio", "importlib",
            "pkgutil", "site", "builtins",
        ],
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
        "mistake_threshold": 5,
        "batch_train_iters": 25,
        "boost_factor": 3,
        "correction_phrases": [
            "no,", "not quite", "that's wrong", "incorrect", "wrong",
            "you misunderstood", "try again", "actually", "i meant",
            "i said", "i asked", "not what", "that's not", "you're wrong",
            "fix it", "correction", "rephrase",
        ],
    },
}

# Keys that must survive a restart to take effect.
_RESTART_KEYS = {"model_name"}


def load_config() -> dict[str, Any]:
    """Load config.json if present; merge with sensible defaults."""
    # Deep copy: callers (e.g. set_config_value) mutate nested sections, and a
    # shallow copy would poison DEFAULT_CONFIG for every later load.
    config = copy.deepcopy(DEFAULT_CONFIG)
    if constants.CONFIG_FILE.exists():
        try:
            user_config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(user_config)
            for section in ("lora", "agent", "rag", "memory", "web", "sandbox", "learn"):
                if section in user_config:
                    config[section] = {**DEFAULT_CONFIG[section], **user_config[section]}
        except Exception as e:
            print(f"[Config warning] Could not read {constants.CONFIG_FILE}: {e}")
    return config


def config_show(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2)


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
