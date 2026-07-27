"""Interactive onboarding setup wizard for Symbio.

Guides new users through identity, model selection, feature toggles, and
Telegram configuration, then persists a merged config.json and runs a
self-check so the first launch is as smooth as possible.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable

from symbio import constants
from symbio.app import config as app_config
from symbio.app import health, memory


def _ask(prompt: str, input_fn: Callable[[str], str]) -> str:
    try:
        return input_fn(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _ask_yes_no(prompt: str, input_fn: Callable[[str], str], default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = _ask(prompt + suffix, input_fn).lower()
    if not answer:
        return default
    return answer in ("y", "yes", "true", "1", "on")


def _load_model_presets() -> dict[str, dict[str, Any]]:
    if not constants.MODELS_FILE.exists():
        return {}
    try:
        return json.loads(constants.MODELS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _list_presets() -> list[tuple[str, dict[str, Any]]]:
    presets = _load_model_presets()
    return [(key, info) for key, info in presets.items()]


def _select_model(config: dict[str, Any], input_fn: Callable[[str], str], output_fn: Callable[[str], Any]) -> str:
    default_model = config.get("model_name") or app_config.DEFAULT_CONFIG.get("model_name", "")
    presets = _list_presets()

    output_fn("\n  Model selection")
    output_fn(f"  Current default: {default_model}")
    if presets:
        output_fn("  Presets:")
        for i, (key, info) in enumerate(presets, start=1):
            line = f"    {i}. {info.get('model_name')} — {info.get('description', '')}"
            note = info.get("memory_note")
            if note:
                line += f" ({note})"
            output_fn(line)
        output_fn(f"    {len(presets) + 1}. Keep default ({default_model})")
        output_fn("    0. Enter a custom HuggingFace/MLX repo name")
    else:
        output_fn("  No presets found. Enter a model name manually.")
        output_fn(f"  Default: {default_model}")

    while True:
        choice = _ask("  Pick a model (number or repo name): ", input_fn).strip()
        if not choice:
            output_fn(f"  Keeping default: {default_model}")
            return default_model
        if choice == "0":
            custom = _ask("  Enter model repo/path: ", input_fn).strip()
            if custom:
                return custom
            output_fn("  Keeping default.")
            return default_model
        if choice.isdigit() and presets:
            idx = int(choice)
            if 1 <= idx <= len(presets):
                return presets[idx - 1][1]["model_name"]
            if idx == len(presets) + 1:
                return default_model
        # Allow typing the repo name directly too.
        if "/" in choice:
            return choice
        output_fn("  Invalid choice. Enter a number, a repo name, or press Enter for default.")


def _parse_chat_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def _setup_telegram(config: dict[str, Any], input_fn: Callable[[str], str], output_fn: Callable[[str], Any]) -> None:
    telegram_cfg = config.setdefault("telegram", {})

    token = os.environ.get("SYMBIO_TELEGRAM_TOKEN", "").strip()
    if token:
        output_fn("  SYMBIO_TELEGRAM_TOKEN is set; using it.")
    else:
        output_fn("\n  Telegram bot token")
        output_fn("  Get one from @BotFather on Telegram, or set SYMBIO_TELEGRAM_TOKEN env var.")
        token = _ask("  Enter bot token (press Enter to skip): ", input_fn).strip()
    if token:
        telegram_cfg["bot_token"] = token
    else:
        output_fn("  No token provided — Telegram bot will not start.")

    output_fn("\n  Allowed Telegram chat IDs")
    output_fn("  Who is allowed to message this bot? Enter one or more numeric chat IDs, comma-separated.")
    raw_ids = _ask("  Allowed chat IDs (comma-separated, or press Enter to configure later): ", input_fn)
    ids = _parse_chat_ids(raw_ids)
    if ids:
        telegram_cfg["allowed_chat_ids"] = ids
        output_fn(f"  Allowed {len(ids)} chat ID(s).")
    else:
        output_fn("  No chat IDs set — bot will reply with setup instructions only until IDs are added.")


def _write_config(config: dict[str, Any]) -> None:
    """Persist the merged config to config.json."""
    try:
        constants.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    constants.CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def is_first_run(config: dict[str, Any]) -> bool:
    """Return True if the wizard should run before the first chat session."""
    if not constants.CONFIG_FILE.exists():
        return True
    if config.get("first_run", True):
        return True
    if not config.get("assistant_name") or not config.get("user_name"):
        return True
    return False


def run_setup_wizard(
    config: dict[str, Any],
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """Run the interactive setup wizard and return the updated config.

    The wizard only mutates the supplied config dict and writes config.json
    at the end; it does not load or unload the model.
    """
    output_fn("\n" + "=" * 50)
    output_fn("  Symbio Setup Wizard")
    output_fn("=" * 50)
    output_fn("  Let's configure Symbio for your machine.")
    output_fn("  Press Enter to accept defaults, or 's' to skip the wizard entirely.")

    skip = _ask("  Skip setup? [s/N]: ", input_fn).lower()
    if skip == "s":
        output_fn("  Skipped. You can re-run anytime with: symb setup")
        config["first_run"] = False
        _write_config(config)
        return config

    # 1. Identity
    output_fn("\n  1. Identity")
    current_user = config.get("user_name") or ""
    current_assistant = config.get("assistant_name") or ""

    user_name = _ask(f"  What is your name? [{current_user or 'User'}]: ", input_fn).strip()
    if not user_name:
        user_name = current_user or "User"

    assistant_name = _ask(
        f"  What would you like to name me? [{current_assistant or 'Symbio'}]: ", input_fn
    ).strip()
    if not assistant_name:
        assistant_name = current_assistant or "Symbio"

    config["user_name"] = user_name
    config["assistant_name"] = assistant_name

    # 2. Model
    config["model_name"] = _select_model(config, input_fn, output_fn)

    # 3. Speed mode
    output_fn("\n  3. Speed preset")
    current_mode = config.get("agent", {}).get("speed_mode", "balanced")
    output_fn("  - balanced: default quality and context")
    output_fn("  - fast: shorter replies, smaller history, faster turns")
    mode = _ask(f"  Speed mode? [balanced/fast, default {current_mode}]: ", input_fn).strip().lower()
    if mode not in ("balanced", "fast"):
        mode = current_mode if current_mode in ("balanced", "fast") else "balanced"
    config.setdefault("agent", {})["speed_mode"] = mode

    # 4. Feature toggles
    output_fn("\n  4. Features")

    if _ask_yes_no("  Enable browser automation? (needs Playwright + Chrome)", input_fn, default=False):
        config.setdefault("browser", {})["enabled"] = True
    else:
        config.setdefault("browser", {})["enabled"] = False

    if _ask_yes_no("  Auto-search the web when the model sounds unsure?", input_fn, default=True):
        config.setdefault("web", {})["auto_search_when_unsure"] = True
    else:
        config.setdefault("web", {})["auto_search_when_unsure"] = False

    if _ask_yes_no("  Enable MOA delegation (loads extra worker models)?", input_fn, default=False):
        config.setdefault("dispatch", {})["enabled"] = True
    else:
        config.setdefault("dispatch", {})["enabled"] = False

    if _ask_yes_no("  Enable Telegram bot?", input_fn, default=False):
        config.setdefault("telegram", {})["enabled"] = True
        _setup_telegram(config, input_fn, output_fn)
    else:
        config.setdefault("telegram", {})["enabled"] = False

    if _ask_yes_no("  Create backups before editing existing files?", input_fn, default=True):
        config.setdefault("agent", {})["backup_before_edit"] = True
    else:
        config.setdefault("agent", {})["backup_before_edit"] = False

    # 5. Review
    output_fn("\n  5. Review")
    output_fn(f"    Assistant name: {config['assistant_name']}")
    output_fn(f"    User name:      {config['user_name']}")
    output_fn(f"    Model:            {config['model_name']}")
    output_fn(f"    Speed mode:       {config['agent']['speed_mode']}")
    output_fn(f"    Browser:          {'ON' if config['browser']['enabled'] else 'off'}")
    output_fn(f"    Auto-search:      {'ON' if config['web']['auto_search_when_unsure'] else 'off'}")
    output_fn(f"    Dispatch/MoA:     {'ON' if config['dispatch']['enabled'] else 'off'}")
    telegram_on = config["telegram"].get("enabled", False)
    output_fn(f"    Telegram:         {'ON' if telegram_on else 'off'}")
    output_fn(f"    File backups:     {'ON' if config['agent'].get('backup_before_edit', True) else 'off'}")
    if telegram_on:
        ids = config["telegram"].get("allowed_chat_ids", [])
        token_set = bool(config["telegram"].get("bot_token") or os.environ.get("SYMBIO_TELEGRAM_TOKEN"))
        output_fn(f"      Token set:      {'yes' if token_set else 'no'}")
        output_fn(f"      Allowed IDs:    {ids if ids else '(none yet)'}")

    ok = _ask_yes_no("  Save this configuration?", input_fn, default=True)
    if not ok:
        output_fn("  Discarded. Re-run with: symb setup")
        return config

    config["first_run"] = False
    _write_config(config)

    # Seed identity notes and run a lightweight self-check. We skip the model
    # load because the wizard runs just before the chat session loads it.
    memory.ensure_seed_notes(config)
    output_fn("\n  Running feature self-check...")
    try:
        health.verify_enabled_features(config, verbose=True, output_fn=output_fn, skip_model_load=True)
    except Exception as e:
        output_fn(f"  Self-check warning: {e}")

    output_fn("\n  Setup complete. Start chatting with: symbio")
    return config
