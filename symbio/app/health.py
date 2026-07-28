"""Self-diagnostic checks for the Symbio runtime environment.

The `verify_enabled_features` function is the AI's runtime health guard: it
runs only the checks that correspond to features the user has enabled, tries
to repair safe/common failures automatically, and reports anything it cannot
fix so the agent can either retry, route around, or ask the human.
"""

import copy
import json
import os
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


def _browser_ok(config: dict[str, Any]) -> dict[str, Any]:
    """Check whether a Playwright-controlled browser can be launched.
    Only runs when browser automation is enabled; otherwise returns empty."""
    if not config.get("browser", {}).get("enabled", False):
        return {"checked": False}
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


def _ollama_ok(config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("dispatch", {}).get("enabled", False):
        return {"checked": False}
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


def _frontier_ok(config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("dispatch", {}).get("enabled", False):
        return {"checked": False}
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
        "browser": _browser_ok(config),
        "ollama": _ollama_ok(config),
        "frontier": _frontier_ok(config),
        "disk": _disk_ok(),
        "recent_errors": _recent_errors(),
    }


# ---------------------------------------------------------------------------
# Feature verification + auto-fix layer
# ---------------------------------------------------------------------------

class _CheckResult:
    __slots__ = ("name", "ok", "auto_fixed", "message", "severity")

    def __init__(
        self,
        name: str,
        ok: bool,
        auto_fixed: bool = False,
        message: str = "",
        severity: str = "info",
    ):
        self.name = name
        self.ok = ok
        self.auto_fixed = auto_fixed
        self.message = message
        self.severity = severity  # info | warning | error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "auto_fixed": self.auto_fixed,
            "message": self.message,
            "severity": self.severity,
        }


def _check_config_validity(config: dict[str, Any]) -> _CheckResult:
    """Ensure the config loads and has required identity keys."""
    try:
        cfg = app_config.load_config()
    except Exception as exc:
        return _CheckResult(
            "config_load", False, message=f"config.json is unreadable: {exc}", severity="error"
        )
    missing = []
    if not cfg.get("model_name"):
        missing.append("model_name")
    if not cfg.get("assistant_name"):
        missing.append("assistant_name")
    if not cfg.get("user_name"):
        missing.append("user_name")
    if missing:
        return _CheckResult(
            "config_identity",
            False,
            message=f"Missing identity fields: {', '.join(missing)}. Set them with /config set.",
            severity="warning",
        )
    return _CheckResult("config_load", True, message="Configuration valid.")


def _check_model_load(config: dict[str, Any]) -> _CheckResult:
    """Check the configured model exists and can be loaded. No auto-fix here:
    switching models is a human decision."""
    from mlx_lm import load

    model_name = config.get("model_name", "")
    if not model_name:
        return _CheckResult("model_load", False, message="No model_name configured.", severity="error")
    try:
        # Load in lazy mode to keep the check fast; only validate tokenizer access.
        _, tokenizer = load(model_name, lazy=True)
        tokenizer.encode("hello")
        return _CheckResult("model_load", True, message=f"Model '{model_name}' loadable.")
    except Exception as exc:
        return _CheckResult(
            "model_load",
            False,
            message=f"Could not load model '{model_name}': {exc}. Check the name or run `symb retrain` if you switched models.",
            severity="error",
        )


def _check_required_dirs(config: dict[str, Any]) -> _CheckResult:
    """Auto-create directories Symbio needs."""
    dirs = [
        constants.LOG_DIR,
        constants.DATA_DIR,
        constants.ADAPTER_DIR,
        constants.ADAPTER_ARCHIVE_DIR,
        constants.NOTES_DIR,
        constants.MISTAKES_DIR,
        constants.MISTAKES_ARCHIVE_DIR,
        constants.NOTES_ARCHIVE_DIR,
        constants.SANDBOX_DIR,
        constants.SCREENSHOTS_DIR,
        constants.SESSIONS_DIR,
    ]
    created = []
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d.name)
    msg = "Required directories present."
    if created:
        msg += f" Created missing: {', '.join(created)}."
    return _CheckResult("required_dirs", True, auto_fixed=bool(created), message=msg)


def _check_training_data(config: dict[str, Any]) -> _CheckResult:
    """If learning is enabled, ensure at least a minimal training file exists.
    Auto-fix: create an empty JSONL file so later seeding can append."""
    if not config.get("learn", {}).get("enabled", True):
        return _CheckResult("training_data", False, message="Learning disabled; skipped.", severity="info")

    auto_fixed = False
    if not constants.TRAIN_FILE.exists():
        constants.DATA_DIR.mkdir(parents=True, exist_ok=True)
        constants.TRAIN_FILE.write_text("", encoding="utf-8")
        auto_fixed = True
    if not constants.VALID_FILE.exists():
        constants.DATA_DIR.mkdir(parents=True, exist_ok=True)
        constants.VALID_FILE.write_text("", encoding="utf-8")
        auto_fixed = True

    samples = _count_jsonl(constants.TRAIN_FILE)
    msg = f"Training data file exists ({samples} samples)."
    if auto_fixed:
        msg += " Created missing train/valid files."
    return _CheckResult("training_data", True, auto_fixed=auto_fixed, message=msg)


def _check_rag(config: dict[str, Any]) -> _CheckResult:
    """If RAG is enabled, verify the retriever can initialize."""
    if not config.get("rag", {}).get("enabled", True):
        return _CheckResult("rag", False, message="RAG disabled; skipped.", severity="info")
    try:
        from rag import Retriever
        _ = Retriever(config, session_store=None, exclude_session_id=None)
        return _CheckResult("rag", True, message="RAG retriever initialized.")
    except Exception as exc:
        return _CheckResult(
            "rag",
            False,
            message=f"RAG retriever failed to initialize: {exc}. Notes/sessions won't be retrieved.",
            severity="warning",
        )


def _check_web_search(config: dict[str, Any]) -> _CheckResult:
    """If web_search tool group is enabled or auto-search is on, run a tiny
    connectivity probe."""
    enabled_groups = set(config.get("tools", {}).get("enabled_groups", []))
    if "web_search" not in enabled_groups and not config.get("web", {}).get("auto_search_when_unsure", False):
        return _CheckResult("web_search", False, message="Web search not enabled; skipped.", severity="info")
    try:
        import httpx
        response = httpx.get("https://api.duckduckgo.com/?q=test&format=json", timeout=10.0)
        response.raise_for_status()
        return _CheckResult("web_search", True, message="Web search backend reachable.")
    except Exception as exc:
        return _CheckResult(
            "web_search",
            False,
            message=f"Web search probe failed: {exc}. Searches may time out until connectivity returns.",
            severity="warning",
        )


def _check_browser(config: dict[str, Any]) -> _CheckResult:
    """If browser automation is enabled, verify Playwright can open a page.
    Auto-fix: disable browser.enabled if it is clearly broken and tell the user."""
    if not config.get("browser", {}).get("enabled", False):
        return _CheckResult("browser", False, message="Browser automation disabled; skipped.", severity="info")

    from symbio.computer import BrowserSession

    browser = BrowserSession(confirm_fn=lambda p: True)
    try:
        browser.open("about:blank")
        browser.close()
        return _CheckResult("browser", True, message="Browser automation available.")
    except Exception as exc:
        # Safe auto-fix: turn off the broken feature in memory so the rest of the
        # session can continue without repeated Playwright crashes. We do NOT
        # persist this to config.json — the human must decide to re-enable.
        config["browser"]["enabled"] = False
        return _CheckResult(
            "browser",
            False,
            auto_fixed=True,
            message=f"Browser automation failed (disabled for this session): {exc}. Re-enable with /config set browser.enabled true when Playwright is installed.",
            severity="warning",
        )
    finally:
        try:
            browser.close()
        except Exception:
            pass


def _check_telegram(config: dict[str, Any]) -> _CheckResult:
    """If Telegram is enabled, check token and allowed_chat_ids."""
    telegram_cfg = config.get("telegram", {})
    if not telegram_cfg.get("enabled", False):
        return _CheckResult("telegram", False, message="Telegram disabled; skipped.", severity="info")
    token = telegram_cfg.get("bot_token", "") or os.environ.get("SYMBIO_TELEGRAM_TOKEN", "")
    if not token:
        return _CheckResult(
            "telegram",
            False,
            message="Telegram enabled but no bot token. Set telegram.bot_token or SYMBIO_TELEGRAM_TOKEN.",
            severity="error",
        )
    allowed = telegram_cfg.get("allowed_chat_ids", [])
    if not allowed:
        return _CheckResult(
            "telegram",
            True,
            message="Telegram token set, but allowed_chat_ids is empty — the bot will reply with setup instructions only.",
            severity="info",
        )
    return _CheckResult("telegram", True, message=f"Telegram configured ({len(allowed)} allowed chat ID(s)).")


def _check_dispatch(config: dict[str, Any]) -> _CheckResult:
    """If MoA dispatch is enabled, verify Ollama/frontier is reachable."""
    if not config.get("dispatch", {}).get("enabled", False):
        return _CheckResult("dispatch", False, message="Dispatch disabled; skipped.", severity="info")

    frontier_key = bool(settings.frontier_api_key)
    try:
        url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        httpx.get(url, timeout=5.0).raise_for_status()
        ollama_ok = True
    except Exception:
        ollama_ok = False

    if ollama_ok:
        return _CheckResult("dispatch", True, message="Ollama reachable for dispatch.")
    if frontier_key:
        return _CheckResult("dispatch", True, message="Frontier API key set for dispatch fallback.")
    return _CheckResult(
        "dispatch",
        False,
        message="Dispatch enabled but neither Ollama is reachable nor a frontier API key is set. Delegation will fail.",
        severity="error",
    )


def _check_cron(config: dict[str, Any]) -> _CheckResult:
    """If cron tool group is enabled, ensure cron_jobs.json exists and is valid."""
    enabled_groups = set(config.get("tools", {}).get("enabled_groups", []))
    if "cron" not in enabled_groups:
        return _CheckResult("cron", False, message="Cron tool group not enabled; skipped.", severity="info")

    auto_fixed = False
    if not constants.CRON_FILE.exists():
        constants.CRON_FILE.write_text("[]", encoding="utf-8")
        auto_fixed = True
    else:
        try:
            json.loads(constants.CRON_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            # Back up the broken file and recreate it.
            backup = constants.CRON_FILE.with_suffix(".json.broken")
            constants.CRON_FILE.rename(backup)
            constants.CRON_FILE.write_text("[]", encoding="utf-8")
            auto_fixed = True
            return _CheckResult(
                "cron",
                True,
                auto_fixed=True,
                message=f"cron_jobs.json was invalid ({exc}); backed up to {backup.name} and reset.",
            )
    msg = "Cron jobs file is valid."
    if auto_fixed:
        msg += " Created empty cron_jobs.json."
    return _CheckResult("cron", True, auto_fixed=auto_fixed, message=msg)


def _check_memory(config: dict[str, Any]) -> _CheckResult:
    """If memory is enabled, seed identity notes if notes/ is empty.
    This mirrors the existing startup seeding so the self-check stays honest."""
    if not config.get("memory", {}).get("enabled", True):
        return _CheckResult("memory", False, message="Memory disabled; skipped.", severity="info")
    from symbio.app import memory

    if any(constants.NOTES_DIR.glob("*.md")):
        return _CheckResult("memory", True, message="Notes directory has entries.")
    memory.ensure_seed_notes(config)
    return _CheckResult(
        "memory", True, auto_fixed=True, message="Notes directory was empty; seeded identity notes."
    )


def _check_disk(config: dict[str, Any]) -> _CheckResult:
    """Warn if disk space is critically low. No auto-fix possible."""
    usage = shutil.disk_usage(constants.PROJECT_DIR)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 1.0:
        return _CheckResult(
            "disk", False, message=f"Only {free_gb:.1f} GB free — training and model downloads may fail.", severity="error"
        )
    if free_gb < 5.0:
        return _CheckResult(
            "disk", True, message=f"Disk space low ({free_gb:.1f} GB free).", severity="warning"
        )
    return _CheckResult("disk", True, message=f"Disk space healthy ({free_gb:.1f} GB free).")


def _check_python_env(config: dict[str, Any]) -> _CheckResult:
    """Verify required third-party packages are importable."""
    required = {
        "mlx_lm": "mlx-lm",
        "rag": "rag (local)",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} ({package})")
    if missing:
        return _CheckResult(
            "python_env",
            False,
            message=f"Missing required packages: {', '.join(missing)}. Install with `pip install -e .`.",
            severity="error",
        )
    return _CheckResult("python_env", True, message="Core Python dependencies present.")


def verify_enabled_features(
    config: dict[str, Any],
    *,
    verbose: bool = True,
    output_fn = print,
    skip_model_load: bool = False,
) -> dict[str, Any]:
    """Verify only the features the user has enabled.

    For each enabled feature, run a focused check. If the check can be repaired
    safely without changing user intent (missing directories, empty JSONL
    files, broken browser fallback), do it automatically. Anything that needs a
    human decision (wrong model, missing API key, no disk space) is reported.

    `skip_model_load` is used during the first-run setup wizard: the model will
    be loaded immediately afterward, so we avoid a duplicate (and possibly slow)
    load during onboarding.

    Returns a structured report the agent can consume, surface in `/selfcheck`,
    or relay to the user.
    """
    checks = [
        _check_config_validity(config),
        _check_python_env(config),
        _check_required_dirs(config),
    ]
    if not skip_model_load:
        checks.append(_check_model_load(config))
    checks.extend([
        _check_training_data(config),
        _check_memory(config),
        _check_rag(config),
        _check_web_search(config),
        _check_browser(config),
        _check_telegram(config),
        _check_dispatch(config),
        _check_cron(config),
        _check_disk(config),
    ])

    # Conditionally include skill/adapter checks only when at least one skill adapter exists.
    from symbio.app import skills as _skills
    adapters = _skills.list_skill_adapters()
    if adapters:
        missing_adapters = [a["role"] for a in adapters if not a.get("adapter_exists")]
        if missing_adapters:
            checks.append(_CheckResult(
                "skill_adapters",
                False,
                message=f"Skill catalog entries exist without trained adapters: {', '.join(missing_adapters)}. Run /new-skill or wait for background training to finish.",
                severity="warning",
            ))
        else:
            checks.append(_CheckResult(
                "skill_adapters",
                True,
                message=f"{len(adapters)} skill adapter(s) registered; all have trained weights.",
            ))

    fixed = [c for c in checks if c.auto_fixed]
    issues = [c for c in checks if not c.ok]
    errors = [c for c in issues if c.severity == "error"]
    warnings = [c for c in issues if c.severity == "warning"]

    report = {
        "healthy": not errors,
        "all_ok": not issues,
        "auto_fixed_count": len(fixed),
        "errors_count": len(errors),
        "warnings_count": len(warnings),
        "auto_fixed": [c.to_dict() for c in fixed],
        "errors": [c.to_dict() for c in errors],
        "warnings": [c.to_dict() for c in warnings],
        "checks": [c.to_dict() for c in checks],
    }

    if verbose:
        lines = ["[Self-check] Feature verification complete."]
        if fixed:
            lines.append(f"  Auto-fixed {len(fixed)} issue(s):")
            for c in fixed:
                lines.append(f"    • {c.name}: {c.message}")
        if errors:
            lines.append(f"  {len(errors)} error(s) need human attention:")
            for c in errors:
                lines.append(f"    ⚠ {c.name}: {c.message}")
        if warnings:
            lines.append(f"  {len(warnings)} warning(s):")
            for c in warnings:
                lines.append(f"    • {c.name}: {c.message}")
        if not issues and not fixed:
            lines.append("  All enabled features look healthy.")
        output_fn("\n".join(lines))

    return report


def summary_for_agent(report: dict[str, Any]) -> str:
    """One-sentence summary for injecting into the system prompt or telling the user."""
    if report["all_ok"]:
        return "Self-check passed: all enabled features are healthy."
    if report["errors"]:
        names = ", ".join(e["name"] for e in report["errors"])
        return (
            f"Self-check found {len(report['errors'])} issue(s) requiring human attention: {names}. "
            "Tell the user what needs fixing or ask them to run /selfcheck."
        )
    if report["warnings"]:
        names = ", ".join(w["name"] for w in report["warnings"])
        return f"Self-check passed with warnings: {names}."
    return "Self-check passed after auto-fixing minor issues."
