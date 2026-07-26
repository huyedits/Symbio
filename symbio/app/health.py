"""Self-diagnostic checks for the Symbio runtime environment."""

import json
import shutil
from pathlib import Path
from typing import Any

import httpx

from symbio import constants
from symbio.app import config as app_config
from symbio.mcp.config import settings


def _adapter_ok() -> dict[str, Any]:
    config_path = constants.ADAPTER_DIR / "adapter_config.json"
    weight_files = list(constants.ADAPTER_DIR.glob("adapters.*"))
    return {
        "present": config_path.exists() and bool(weight_files),
        "config": config_path.exists(),
        "weights": [f.name for f in weight_files],
        "dir": str(constants.ADAPTER_DIR),
    }


def _training_data_ok() -> dict[str, Any]:
    train = constants.TRAIN_FILE
    valid = train.parent / "valid.jsonl"
    return {
        "train_exists": train.exists(),
        "train_size": train.stat().st_size if train.exists() else 0,
        "valid_exists": valid.exists(),
        "valid_size": valid.stat().st_size if valid.exists() else 0,
        "samples": _count_jsonl(train),
    }


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _prompt_ok() -> dict[str, Any]:
    prompt = constants.PROMPT_FILE
    default = constants.PROMPT_DEFAULT_FILE
    return {
        "prompt_exists": prompt.exists(),
        "prompt_size": prompt.stat().st_size if prompt.exists() else 0,
        "default_exists": default.exists(),
        "default_size": default.stat().st_size if default.exists() else 0,
    }


def _browser_ok() -> dict[str, Any]:
    """Check whether a Playwright-controlled browser can be launched."""
    from symbio.computer import BrowserSession

    browser = BrowserSession(confirm_fn=lambda p: True)
    try:
        browser.open("about:blank")
        ok = True
        error = None
        browser.close()
    except Exception as exc:
        ok = False
        error = str(exc)
    return {"available": ok, "error": error}


def _ollama_ok() -> dict[str, Any]:
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        return {
            "reachable": True,
            "models": [m.get("name") for m in models],
        }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _frontier_ok() -> dict[str, Any]:
    return {
        "provider": settings.frontier_provider,
        "model": settings.frontier_model,
        "key_set": bool(settings.frontier_api_key),
    }


def _disk_ok() -> dict[str, Any]:
    usage = shutil.disk_usage(constants.PROJECT_DIR)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    return {
        "free_gb": round(free_gb, 2),
        "total_gb": round(total_gb, 2),
        "healthy": free_gb > 5.0,
    }


def _config_ok() -> dict[str, Any]:
    try:
        cfg = app_config.load_config()
        return {
            "loadable": True,
            "model": cfg.get("model_name"),
            "assistant": cfg.get("assistant_name"),
            "user": cfg.get("user_name"),
        }
    except Exception as exc:
        return {"loadable": False, "error": str(exc)}


def _recent_errors(lines: int = 20) -> list[str]:
    logs = sorted(constants.LOG_DIR.glob("chat_*.log"), reverse=True)
    if not logs:
        return []
    try:
        with open(logs[0], "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        return [line.strip() for line in all_lines[-lines:] if "error" in line.lower()]
    except Exception:
        return []


def system_check(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a full environmental self-check and return a structured report."""
    config_error = None
    if config is None:
        try:
            config = app_config.load_config()
        except Exception as exc:
            config = {}
            config_error = str(exc)

    return {
        "healthy": True,
        "config": _config_ok(),
        "config_error": config_error,
        "adapter": _adapter_ok(),
        "training_data": _training_data_ok(),
        "prompt": _prompt_ok(),
        "browser": _browser_ok(),
        "ollama": _ollama_ok(),
        "frontier": _frontier_ok(),
        "disk": _disk_ok(),
        "recent_errors": _recent_errors(),
    }
