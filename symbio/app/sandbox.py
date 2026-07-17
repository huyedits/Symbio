"""Sandboxed shell commands and pure-computation Python scripts."""

import ast
import os
import shlex
import subprocess
import sys
import tempfile
from typing import Any

from symbio import constants


def _ask_command_permission(command: str, binary: str) -> bool:
    """Ask the user on the terminal to approve a normally-blocked command.
    Any failure to read an answer (EOF, no tty, interrupt) means no."""
    try:
        answer = input(
            f"\n  [Sandbox] '{binary}' is normally blocked. Allow once?\n"
            f"    $ {command}\n  [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        return False
    return answer in ("y", "yes")


def run_sandboxed(command: str, config: dict[str, Any], interactive: bool = True):
    command = command.strip()
    if not command:
        return False, "Empty command."
    try:
        args = shlex.split(command)
    except ValueError as e:
        return False, f"Parse error: {e}"
    if not args:
        return False, "Empty command."

    blocked = set(config["sandbox"]["blocked_commands"])
    if args[0] in blocked:
        # Blocked commands are not refused outright: the user can approve a
        # one-off run. Non-interactive callers (cron thread) never prompt.
        if not interactive or not _ask_command_permission(command, args[0]):
            return False, f"'{args[0]}' is blocked in sandbox (user did not approve it)."

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=config["agent"]["sandbox_timeout"],
            cwd=constants.SANDBOX_DIR,
            shell=False,
        )
        out = result.stdout
        if result.stderr:
            out += "\n" + result.stderr
        out = out.strip()
        max_len = config["agent"]["max_output_len"]
        if len(out) > max_len:
            out = out[:max_len] + "\n... (truncated)"
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {config['agent']['sandbox_timeout']}s."
    except FileNotFoundError:
        return False, f"Command not found: {args[0]}"
    except Exception as e:
        return False, str(e)


def _is_code_safe(code: str, blocked_imports: set[str]) -> tuple[bool, str]:
    """Reject imports that would let sandboxed code touch the filesystem,
    network, or host process; scripts are for pure computation."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in blocked_imports:
                    return False, f"Import '{alias.name}' is not allowed in the sandbox."
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return False, "Relative imports are not allowed in the sandbox."
            if (node.module or "").split(".")[0] in blocked_imports:
                return False, f"Import '{node.module}' is not allowed in the sandbox."
    return True, ""


def run_python_code(code: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Run a short Python script in the sandbox directory."""
    code = code.strip()
    if not code:
        return False, "Empty code."
    safe, msg = _is_code_safe(code, set(config["sandbox"]["blocked_imports"]))
    if not safe:
        return False, msg

    fd, path = tempfile.mkstemp(suffix=".py", dir=str(constants.SANDBOX_DIR), prefix="caine_code_")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=config["agent"]["code_timeout"],
            cwd=str(constants.SANDBOX_DIR),
            env={"PATH": os.environ.get("PATH", "")},
        )
        out = result.stdout
        if result.stderr:
            out += "\n" + result.stderr
        out = out.strip()
        max_len = config["agent"]["max_output_len"]
        if len(out) > max_len:
            out = out[:max_len] + "\n... (truncated)"
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {config['agent']['code_timeout']}s."
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
