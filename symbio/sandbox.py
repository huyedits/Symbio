"""Sandboxed command and code execution helpers for Symbio."""

import ast
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from symbio.constants import DEFAULT_CONFIG, PROJECT_DIR, SANDBOX_DIR
from symbio.utils import _scrub_env, _truncated


def _run_sandboxed(command: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Run a sandboxed shell command and return (ok, output)."""
    command = command.strip()
    if not command:
        return False, "Empty command."
    try:
        args = shlex.split(command)
    except ValueError as e:
        return False, f"Parse error: {e}"
    if not args:
        return False, "Empty command."

    blocked = {
        "rm", "sudo", "su", "dd", "mkfs", "fdisk", "mount", "umount",
        "chmod", "chown", "curl", "wget", "ssh", "scp", "bash", "sh", "zsh",
        "fish", "python", "python3", "perl", "ruby", "php", "node", "npm",
    }
    if args[0] in blocked:
        return False, f"'{args[0]}' is blocked in sandbox."

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=config["agent"]["sandbox_timeout"],
            cwd=str(SANDBOX_DIR),
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


def _write_symbio_tools_stub() -> Path:
    """Write the symbio_tools sandbox stub and a backward-compatible caine_tools alias."""
    stub_path = SANDBOX_DIR / "symbio_tools.py"
    shim_path = SANDBOX_DIR / "caine_tools.py"
    stub = '''\
"""Stub tool module for execute_code sandbox.\n\nWhitelisted tools: read_file, write_file, patch, search_files, terminal, web_search, web_extract.\n"""
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path("''' + str(PROJECT_DIR) + '''").resolve()
SANDBOX_DIR = PROJECT_DIR / "sandbox"
MAX_OUTPUT = 50000\n\ndef _truncated(text, limit=MAX_OUTPUT):\n    text = text.strip()\n    return text if len(text) <= limit else text[:limit] + "\\n... (truncated)"\n\ndef _project_path(path, must_exist=False):\n    target = (PROJECT_DIR / path).resolve()\n    if not str(target).startswith(str(PROJECT_DIR)):\n        raise ValueError("Path must be inside project directory.")\n    if must_exist and not target.exists():\n        raise FileNotFoundError(path)\n    return target\n\ndef read_file(path, offset=1, limit=100):\n    target = _project_path(path, must_exist=True)\n    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()\n    offset = max(0, offset - 1)\n    selected = lines[offset:offset + limit]\n    return "\\n".join(f"{offset + i + 1}: {line}" for i, line in enumerate(selected))\n\ndef write_file(path, content):\n    target = _project_path(path)\n    target.parent.mkdir(parents=True, exist_ok=True)\n    target.write_text(content, encoding="utf-8")\n    return f"Wrote {path}."\n\ndef patch(path, old_text, new_text):\n    target = _project_path(path, must_exist=True)\n    content = target.read_text(encoding="utf-8")\n    if old_text not in content:\n        raise ValueError("old_text not found")\n    target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")\n    return f"Patched {path}."\n\ndef search_files(query, glob=""):\n    matches = []\n    files = list(PROJECT_DIR.rglob(glob or "*")) if glob else list(PROJECT_DIR.rglob("*"))\n    for f in files:\n        if not f.is_file():\n            continue\n        try:\n            text = f.read_text(encoding="utf-8", errors="replace")\n            if query.lower() in text.lower():\n                matches.append(str(f.relative_to(PROJECT_DIR)))\n        except Exception:\n            pass\n    return "\\n".join(matches[:50]) or "No matches."\n\ndef terminal(cmd):\n    args = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)\n    if not args:\n        return "Empty command."\n    result = subprocess.run(args, capture_output=True, text=True, timeout=30, cwd=str(SANDBOX_DIR))\n    out = (result.stdout or "") + "\\n" + (result.stderr or "")\n    return _truncated(out.strip())\n\ndef web_search(query):\n    return f"Web search not configured. Query: {query}"\n\ndef web_extract(url):\n    return f"Web extract not configured. URL: {url}"\n'''
    stub_path.write_text(stub, encoding="utf-8")
    shim = '''\
"""Backward-compatible alias for symbio_tools."""
from symbio_tools import *
'''
    shim_path.write_text(shim, encoding="utf-8")
    return stub_path


_BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "pathlib", "shutil", "socket", "http", "urllib",
    "ftplib", "smtplib", "imaplib", "pickle", "ctypes", "multiprocessing", "threading",
    "tempfile", "asyncio", "importlib", "pkgutil", "site", "builtins",
})


def _is_code_safe(code: str) -> tuple[bool, str]:
    """Best-effort static check for execute_code sandbox scripts."""
    if "execute_code" in code:
        return False, "Recursive execute_code is blocked."
    if (
        "from symbio_tools import" not in code
        and "import symbio_tools" not in code
        and "from caine_tools import" not in code
        and "import caine_tools" not in code
    ):
        return False, "Scripts must import from symbio_tools (or caine_tools alias) to run in the sandbox."
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in _BLOCKED_IMPORTS:
                    return False, f"Import '{alias.name}' is not allowed in the sandbox."
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return False, "Relative imports are not allowed in the sandbox."
            base = (node.module or "").split(".")[0]
            if base in _BLOCKED_IMPORTS:
                return False, f"Import '{node.module}' is not allowed in the sandbox."
    return True, ""


def _run_execute_code(code: str, config: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[bool, str]:
    """Run a short Python script in the sandbox directory."""
    code = code.strip()
    if not code:
        return False, "Empty code."

    safe, msg = _is_code_safe(code)
    if not safe:
        return False, msg

    _write_symbio_tools_stub()
    fd, path = tempfile.mkstemp(suffix=".py", dir=str(SANDBOX_DIR), prefix="hermes_code_")
    with os.fdopen(fd, "w") as f:
        f.write(code)

    try:
        env = _scrub_env()
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=config["agent"]["code_timeout"],
            cwd=str(SANDBOX_DIR),
            env=env,
        )
        out = result.stdout
        if result.stderr:
            out += "\n" + result.stderr
        out = out.strip()
        max_stdout = 50_000
        max_stderr = 10_000
        if len(out) > max_stdout + max_stderr:
            out = out[:max_stdout] + "\n... (truncated)"
        return result.returncode == 0, _truncated(out, config["agent"]["max_output_len"])
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {config['agent']['code_timeout']}s."
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
