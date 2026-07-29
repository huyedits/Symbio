"""Sandboxed shell commands and pure-computation Python scripts."""

import ast
import ast
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from symbio import constants


def _ask_command_permission(command: str, binary: str, ask_fn=None) -> bool:
    """Ask the user to approve a normally-blocked command.
    Any failure to read an answer (EOF, no tty, interrupt) means no.
    `ask_fn(prompt) -> bool` may be supplied by non-terminal front-ends."""
    prompt = (
        f"[Sandbox] '{binary}' is normally blocked. Allow once?\n"
        f"  $ {command}"
    )
    if ask_fn is not None:
        try:
            return ask_fn(prompt)
        except Exception:
            return False
    try:
        answer = input(f"\n  {prompt}\n  [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        return False
    return answer in ("y", "yes")


def _run_subprocess(args: list[str], config: dict[str, Any], cwd: str | None = None,
                    env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Run a subprocess and return a trimmed (ok, output) tuple."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=config["agent"]["sandbox_timeout"],
            cwd=cwd,
            env=env,
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


def run_sandboxed(command: str, config: dict[str, Any], interactive: bool = True,
                  confirm_fn=None):
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
        if not interactive or not _ask_command_permission(command, args[0], ask_fn=confirm_fn):
            return False, f"'{args[0]}' is blocked in sandbox (user did not approve it)."

    return _run_subprocess(args, config, cwd=constants.SANDBOX_DIR)


def run_shell(command: str, config: dict[str, Any], interactive: bool = True,
              confirm_fn=None) -> tuple[bool, str]:
    """Run a shell command locally through a configured shell.

    Useful for commands that need pipes, globbing, or env vars. Requires
    user approval if the chosen shell is in the blocked-shells list.
    """
    command = command.strip()
    if not command:
        return False, "Empty command."

    shell_cfg = config.get("sandbox", {})
    blocked_shells = set(shell_cfg.get("blocked_shells", ["bash", "sh", "zsh", "fish"]))
    shell_path = os.environ.get("SHELL", "/bin/sh")
    shell_name = Path(shell_path).name

    if not shell_cfg.get("shell_allow_localhost", True):
        return False, "Local shell execution is disabled in config."

    if shell_name in blocked_shells:
        if not interactive or not _ask_command_permission(command, shell_name, ask_fn=confirm_fn):
            return False, f"'{shell_name}' is blocked (user did not approve it)."

    return _run_subprocess([shell_path, "-c", command], config, cwd=constants.SANDBOX_DIR)


def run_remote(host: str, command: str, config: dict[str, Any], interactive: bool = True,
               confirm_fn=None) -> tuple[bool, str]:
    """Run a shell command on a configured remote host via SSH.

    localhost is handled by running the command through the local shell.
    Other hosts must be defined in config["remote"]["hosts"].
    """
    command = command.strip()
    if not command:
        return False, "Empty command."

    remote_cfg = config.get("remote", {})
    hosts: dict[str, Any] = remote_cfg.get("hosts", {})

    if host.lower() in ("localhost", "127.0.0.1", "::1"):
        if not config.get("sandbox", {}).get("shell_allow_localhost", True):
            return False, "Localhost shell execution is disabled in config."
        return run_shell(command, config, interactive=interactive, confirm_fn=confirm_fn)

    if not config.get("sandbox", {}).get("shell_allow_remote_hosts", True):
        return False, "Remote host execution is disabled in config."

    host_cfg = hosts.get(host)
    if host_cfg is None:
        return (
            False,
            f"Host '{host}' is not configured. Add it to remote.hosts (e.g. via /config set remote.hosts '<json>').",
        )

    hostname = host_cfg.get("hostname", host)
    user = host_cfg.get("user")
    port = host_cfg.get("port", 22)
    ssh_key = host_cfg.get("ssh_key")
    extra_opts = host_cfg.get("ssh_options", [])

    ssh_args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    ssh_args.extend(extra_opts)
    if ssh_key:
        ssh_args.extend(["-i", str(Path(ssh_key).expanduser())])
    if port != 22:
        ssh_args.extend(["-p", str(port)])
    target = f"{user}@{hostname}" if user else hostname
    ssh_args.extend([target, command])

    return _run_subprocess(ssh_args, config)


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
