"""Symbio command-line interface.

Entry points `symbio` and `symb` both resolve here. Supports both a modern
subcommand style (`symb config`, `symb gateway start`) and the legacy flags
from `main.py` (`--telegram`, `--train`) so existing scripts keep working.
"""

import argparse
import atexit
import curses
import json
import os
import signal
import sys
from typing import Any

from symbio import constants
from symbio.app.chat import chat_loop
from symbio.app.config import config_show, get_telegram_token, load_config, set_config_value
from symbio.app.training import run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbio",
        description="Symbio — personal, autonomous, self-finetuning agent.",
    )
    # Legacy flags kept for backward compatibility with `python main.py --telegram`.
    parser.add_argument(
        "--telegram", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Base MLX model (overrides config.json)",
    )
    parser.add_argument(
        "--assistant-name", type=str, default=None,
        help="Assistant name (overrides config.json)",
    )
    parser.add_argument(
        "--user-name", type=str, default=None,
        help="User name (overrides config.json)",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="Start the interactive chat CLI (default)")

    config_parser = sub.add_parser("config", help="Show or edit configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("show", help="Show current config")
    config_sub.add_parser("edit", help="Open interactive config editor (default)")
    get_cmd = config_sub.add_parser("get", help="Print one config value")
    get_cmd.add_argument("key", help="Dotted config key, e.g. agent.temperature")
    set_cmd = config_sub.add_parser("set", help="Set a config value")
    set_cmd.add_argument("key", help="Dotted config key")
    set_cmd.add_argument("value", help="New value (coerced to the current type)")
    config_parser.set_defaults(config_command="edit")

    gateway_parser = sub.add_parser("gateway", help="Manage the Telegram gateway")
    gateway_sub = gateway_parser.add_subparsers(dest="gateway_command")
    gateway_sub.add_parser("start", help="Start the Telegram bot")
    gateway_sub.add_parser("status", help="Show gateway status")
    gateway_sub.add_parser("stop", help="Stop the running gateway")
    gateway_parser.set_defaults(gateway_command="status")

    sub.add_parser("train", help="Run LoRA training")

    retrain_parser = sub.add_parser("retrain", help="Rebuild the LoRA adapter from scratch after switching models")
    retrain_parser.add_argument(
        "--no-digest",
        action="store_true",
        help="Skip re-digesting notes and memory",
    )
    retrain_parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip seeding baseline training data",
    )

    mcp_parser = sub.add_parser("mcp", help="Start the local-brain MCP server")
    mcp_parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse"],
        help="MCP transport (stdio or sse)",
    )

    benchmark_parser = sub.add_parser("benchmark", help="Benchmark Ollama models as local brains")
    benchmark_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of Ollama models to test",
    )
    benchmark_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write a JSON benchmark report",
    )

    benchmark_mlx_parser = sub.add_parser("benchmark-mlx", help="Benchmark MLX/HuggingFace models as local brains")
    benchmark_mlx_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of MLX/HF models to test",
    )
    benchmark_mlx_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write a JSON benchmark report",
    )

    return parser


def _load_and_override_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    if args.model:
        config["model_name"] = args.model
    if args.assistant_name:
        config["assistant_name"] = args.assistant_name
    if args.user_name:
        config["user_name"] = args.user_name
    return config


def _config_get(config: dict[str, Any], key: str) -> str:
    parts = key.split(".")
    node: Any = config
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return f"Unknown config key: {key}"
        node = node[part]
    if isinstance(node, dict):
        return f"{key} is a section; use a leaf key."
    return json.dumps(node)


def _token_configured(config: dict[str, Any]) -> bool:
    return bool(
        os.environ.get("SYMBIO_TELEGRAM_TOKEN", "").strip()
        or (config.get("telegram", {}) or {}).get("bot_token", "").strip()
    )


def _gateway_running() -> tuple[bool, int | None]:
    if not constants.GATEWAY_PID_FILE.exists():
        return False, None
    try:
        pid = int(constants.GATEWAY_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True, pid
    except (ValueError, OSError, ProcessLookupError):
        try:
            constants.GATEWAY_PID_FILE.unlink()
        except OSError:
            pass
        return False, None


def _set_config_leaf(config: dict[str, Any], key: str, raw_value: str) -> str:
    """Set a dotted key directly on the mutable config dict and persist."""
    parts = key.split(".")
    node: Any = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    leaf = parts[-1]
    current = node.get(leaf)
    try:
        if isinstance(current, bool):
            value = raw_value.lower() in ("true", "yes", "on", "1")
        elif isinstance(current, int):
            value = int(raw_value)
        elif isinstance(current, float):
            value = float(raw_value)
        elif isinstance(current, list):
            value = json.loads(raw_value)
            if not isinstance(value, list):
                raise ValueError("expected a list")
        else:
            value = raw_value
    except Exception as e:
        return f"Bad value for {key}: {e}"
    node[leaf] = value

    # Persist into config.json.
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

    restart_note = " (takes effect after restart)" if parts[0] == "model_name" else ""
    return f"Set {key} = {value!r}{restart_note}."


def _run_config_tui(config: dict[str, Any]) -> int:
    """Open a two-level curses editor: pick a section, then edit its leaves.

    Returns 0 on clean save/quit, 1 if the terminal does not support TUI.
    """
    if not sys.stdin.isatty():
        print(config_show(config))
        return 0

    sections = _ConfigSections(config)

    def _loop(stdscr) -> int:
        return sections.run(stdscr)

    try:
        return curses.wrapper(_loop)
    except curses.error as e:
        print(f"Terminal does not support the config editor: {e}")
        print(config_show(config))
        return 0


class _ConfigSections:
    """Hold curses state for the section + leaf config editor."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.section_names = sorted(k for k in config.keys() if isinstance(config[k], dict))
        self.section_sel = 0
        self.section_top = 0
        self.message = "←/→ choose section | Enter enter section | q quit"
        self.dirty = False

    def _draw_section_menu(self, stdscr) -> int:
        curses.curs_set(0)
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Symbio Config Editor"[:width - 1], curses.A_BOLD)
        stdscr.addstr(1, 0, " Choose a section"[:width - 1], curses.A_DIM)
        footer = f" {self.message}"
        stdscr.addstr(height - 2, 0, footer[:width - 1], curses.A_DIM)
        stdscr.hline(height - 3, 0, curses.ACS_HLINE, width)

        visible = max(1, height - 5)
        for i in range(visible):
            idx = self.section_top + i
            if idx >= len(self.section_names):
                break
            name = self.section_names[idx]
            attr = curses.A_REVERSE if idx == self.section_sel else curses.A_NORMAL
            try:
                stdscr.addstr(3 + i, 2, f"  {name}  ", attr)
            except curses.error:
                pass

        # Draw a visible cursor next to the selected item.
        try:
            stdscr.addstr(3 + (self.section_sel - self.section_top), 0, "▶", curses.A_BOLD)
        except curses.error:
            pass

        stdscr.refresh()
        return height

    def run(self, stdscr) -> int:
        while True:
            self._draw_section_menu(stdscr)
            ch = stdscr.getch()
            if ch in (curses.KEY_LEFT, curses.KEY_UP, ord("k"), ord("h")):
                self.section_sel = max(0, self.section_sel - 1)
                if self.section_sel < self.section_top:
                    self.section_top = self.section_sel
            elif ch in (curses.KEY_RIGHT, curses.KEY_DOWN, ord("j"), ord("l")):
                self.section_sel = min(len(self.section_names) - 1, self.section_sel + 1)
                height, _ = stdscr.getmaxyx()
                visible = max(1, height - 5)
                if self.section_sel >= self.section_top + visible:
                    self.section_top = self.section_sel - visible + 1
            elif ch in (10, 13, curses.KEY_ENTER):
                section = self.section_names[self.section_sel]
                leaf_editor = _LeafEditor(self, section)
                leaf_editor.run(stdscr)
                self.message = "←/→ choose section | Enter enter section | q quit"
            elif ch in (17, ord("q"), 27):
                return 0


class _LeafEditor:
    """Edit the leaves of a single config section."""

    def __init__(self, sections: _ConfigSections, section: str) -> None:
        self.sections = sections
        self.section = section
        self.config = sections.config
        self.leaves = self._build_rows()
        self.sel = 0
        self.top = 0

    def _build_rows(self) -> list[tuple[str, Any]]:
        section = self.config[self.section]
        return [(k, section[k]) for k in sorted(section.keys())]

    def _draw(self, stdscr) -> int:
        curses.curs_set(0)
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f" Section: {self.section}"[:width - 1], curses.A_BOLD)
        stdscr.addstr(1, 0, " ↑/↓ move | Enter edit | Esc/b back | q quit", curses.A_DIM)
        footer = self.sections.message
        stdscr.addstr(height - 2, 0, f" {footer}"[:width - 1], curses.A_DIM)
        stdscr.hline(height - 3, 0, curses.ACS_HLINE, width)

        visible = max(1, height - 5)
        for i in range(visible):
            idx = self.top + i
            if idx >= len(self.leaves):
                break
            key, value = self.leaves[idx]
            line = f"  {key:30} = {json.dumps(value)[:width - 40]}"
            attr = curses.A_REVERSE if idx == self.sel else curses.A_NORMAL
            try:
                stdscr.addstr(3 + i, 0, line[:width - 1], attr)
            except curses.error:
                pass

        # Draw a visible cursor next to the selected leaf.
        try:
            stdscr.addstr(3 + (self.sel - self.top), 0, "▶", curses.A_BOLD)
        except curses.error:
            pass

        stdscr.refresh()
        return height

    def _edit_value(self, stdscr, key: str, value: Any) -> str | None:
        height, width = stdscr.getmaxyx()
        dotted = f"{self.section}.{key}"
        prompt = f"Edit {dotted} = "
        # Pre-populate with the raw value, not JSON-quoted strings.
        if isinstance(value, str):
            buf = value
        else:
            buf = json.dumps(value)
        cursor = len(buf)
        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, prompt[:width - 1])
            # Place input on its own line so it wraps cleanly.
            edit_line = 2
            stdscr.addstr(edit_line, 0, buf[:width - 1])
            # Cursor: clamp to visible width.
            cursor_y = edit_line
            cursor_x = min(cursor, width - 2)
            stdscr.move(cursor_y, cursor_x)
            curses.curs_set(1)
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (curses.KEY_ENTER, 10, 13):
                return buf
            if ch in (27,):  # Esc cancels
                return None
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                if cursor > 0:
                    buf = buf[:cursor - 1] + buf[cursor:]
                    cursor -= 1
            elif ch == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif ch == curses.KEY_RIGHT:
                cursor = min(len(buf), cursor + 1)
            elif ch == curses.KEY_HOME:
                cursor = 0
            elif ch == curses.KEY_END:
                cursor = len(buf)
            elif ch == curses.KEY_DC:
                if cursor < len(buf):
                    buf = buf[:cursor] + buf[cursor + 1:]
            elif 32 <= ch <= 126:
                buf = buf[:cursor] + chr(ch) + buf[cursor:]
                cursor += 1

    def run(self, stdscr) -> None:
        self.sections.message = "Editing section..."
        while True:
            self._draw(stdscr)
            ch = stdscr.getch()
            if ch in (curses.KEY_UP, ord("k")):
                self.sel = max(0, self.sel - 1)
                if self.sel < self.top:
                    self.top = self.sel
            elif ch in (curses.KEY_DOWN, ord("j")):
                self.sel = min(len(self.leaves) - 1, self.sel + 1)
                height, _ = stdscr.getmaxyx()
                visible = max(1, height - 5)
                if self.sel >= self.top + visible:
                    self.top = self.sel - visible + 1
            elif ch in (10, 13, curses.KEY_ENTER):
                key, value = self.leaves[self.sel]
                new_raw = self._edit_value(stdscr, key, value)
                if new_raw is not None:
                    dotted = f"{self.section}.{key}"
                    msg = _set_config_leaf(self.config, dotted, new_raw)
                    if msg.startswith("Set "):
                        self.leaves[self.sel] = (key, self.config[self.section][key])
                        self.sections.dirty = True
                        self.sections.message = f"Saved {dotted}"
                    else:
                        self.sections.message = msg
            elif ch in (ord("b"), ord("q"), 27):
                return


def _cmd_config(config: dict[str, Any], args: argparse.Namespace) -> int:
    sub = args.config_command
    if sub == "show":
        print(config_show(config))
    elif sub == "edit":
        return _run_config_tui(config)
    elif sub == "get":
        print(_config_get(config, args.key))
    elif sub == "set":
        print(set_config_value(config, args.key, args.value, allow_sandbox=True))
    else:
        print("Usage: symb config [show | edit | get <key> | set <key> <value>]")
        return 1
    return 0


def _cmd_train(config: dict[str, Any]) -> int:
    run_training(config)
    return 0


def _cmd_gateway_status(config: dict[str, Any]) -> int:
    running, pid = _gateway_running()
    token_set = _token_configured(config)
    allowed = (config.get("telegram", {}) or {}).get("allowed_chat_ids", [])
    adapter_exists = (constants.ADAPTER_DIR / "adapter_config.json").exists()

    print(f"Gateway running: {'yes' if running else 'no'}")
    if pid is not None:
        print(f"PID: {pid}")
    print(f"Bot token configured: {'yes' if token_set else 'no'}")
    print(f"Allowed chat IDs: {len(allowed)}")
    print(f"Model: {config['model_name']}")
    print(f"Adapter present: {'yes' if adapter_exists else 'no'}")
    return 0


def _cmd_gateway_stop() -> int:
    running, pid = _gateway_running()
    if not running or pid is None:
        print("Gateway is not running.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent stop signal to gateway (PID {pid}).")
    except PermissionError:
        print(f"Permission denied: cannot stop process {pid}.")
        return 1
    except ProcessLookupError:
        print(f"Process {pid} already exited.")
    finally:
        try:
            constants.GATEWAY_PID_FILE.unlink()
        except OSError:
            pass
    return 0


def _cmd_gateway_start(config: dict[str, Any]) -> int:
    running, pid = _gateway_running()
    if running and pid is not None:
        print(f"Gateway already running (PID {pid}).")
        return 0

    token = get_telegram_token(config)
    if not token:
        print(
            "No Telegram bot token configured. "
            "Set SYMBIO_TELEGRAM_TOKEN or run:\n"
            "  symb config set telegram.bot_token <token>"
        )
        return 1

    allowed = (config.get("telegram", {}) or {}).get("allowed_chat_ids", [])
    if not allowed:
        print(
            "Warning: telegram.allowed_chat_ids is empty. "
            "The bot will refuse all incoming chats until you add your chat ID."
        )

    from symbio.app.telegram import TelegramBot

    constants.GATEWAY_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    def _cleanup_pid() -> None:
        try:
            constants.GATEWAY_PID_FILE.unlink()
        except OSError:
            pass

    atexit.register(_cleanup_pid)

    print("Starting Telegram gateway. Model will load on the inference thread...")
    bot = TelegramBot(config)
    print("Gateway started. Press Ctrl-C to stop.")
    try:
        bot.run()
    finally:
        _cleanup_pid()
    return 0


def _resolve_command(args: argparse.Namespace) -> str:
    if args.telegram:
        return "gateway"
    if args.train:
        return "train"
    return args.command or "chat"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _load_and_override_config(args)

    command = _resolve_command(args)

    if command == "chat":
        chat_loop(config)
        return 0
    if command == "config":
        return _cmd_config(config, args)
    if command == "train":
        return _cmd_train(config)
    if command == "retrain":
        from symbio.app.retrain import retrain_model

        ok = retrain_model(config, digest=not args.no_digest, seed=not args.no_seed)
        return 0 if ok else 1
    if command == "mcp":
        from symbio.mcp.server import mcp

        mcp.run(transport=args.transport)
        return 0
    if command == "benchmark":
        import asyncio

        from symbio.mcp.benchmark import main as benchmark_main

        models = [m.strip() for m in args.models.split(",")] if args.models else None
        asyncio.run(benchmark_main(models=models, output_path=args.output))
        return 0
    if command == "benchmark-mlx":
        from symbio.mcp.benchmark_mlx import main as benchmark_mlx_main

        models = [m.strip() for m in args.models.split(",")] if args.models else None
        benchmark_mlx_main(models=models, output_path=args.output)
        return 0
    if command == "gateway":
        sub = getattr(args, "gateway_command", None) or "start"
        if sub == "start":
            return _cmd_gateway_start(config)
        if sub == "stop":
            return _cmd_gateway_stop()
        if sub == "status":
            return _cmd_gateway_status(config)
        print("Usage: symb gateway [start | status | stop]")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
