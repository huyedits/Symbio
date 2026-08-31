"""Sandboxed shell commands and pure-computation Python scripts."""

import ast
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.ansi_scanner import scan_text, strip_ansi


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
    """Run a subprocess and return a trimmed (ok, output) tuple.

    ANSI colour codes are preserved for scanning, then stripped from the final
    output so the LLM gets clean text. Red segments and error keywords are
    prepended when found.
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=False,
            timeout=config["agent"]["sandbox_timeout"],
            cwd=cwd,
            env=env,
        )
        raw = result.stdout + b"\n" + result.stderr
        out = raw.decode("utf-8", errors="replace").strip()
        max_len = config["agent"]["max_output_len"]
        if len(out) > max_len:
            out = out[:max_len] + "\n... (truncated)"

        scan = scan_text(out)
        clean = strip_ansi(out)
        if scan.looks_bad:
            report_lines = []
            if scan.has_red:
                report_lines.append("Red terminal text detected:")
                for seg in scan.red_segments:
                    report_lines.append(f"  - {seg}")
            if scan.error_keywords:
                report_lines.append("Error keywords: " + ", ".join(scan.error_keywords))
            report_lines.append("---")
            clean = "\n".join(report_lines) + "\n" + clean

        return result.returncode == 0, clean
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {config['agent']['sandbox_timeout']}s."
    except FileNotFoundError:
        return False, f"Command not found: {args[0]}"
    except Exception as e:
        return False, str(e)


# A blocked network binary and the in-sandbox tool that does the same job, so a
# refusal points somewhere instead of dead-ending. Keyed on the denylist name.
_NETWORK_TOOL_REDIRECT = {
    "curl": ("For an HTTP GET, use the read_page / fetch_html tools, or "
             "`from symbio_tools import fetch` inside <py> — fetch(url) returns "
             "the body and raises with the status code and body on a non-2xx."),
    "wget": ("To fetch a URL, use the read_page / fetch_html tools, or "
             "`from symbio_tools import fetch` inside <py>."),
}


# Programs that run another program named in their own arguments. Without
# stepping through these, `env rm x` and `xargs rm` present argv[0] as something
# harmless while executing something on the denylist.
_COMMAND_WRAPPERS = frozenset({
    "env", "nice", "nohup", "time", "timeout", "xargs", "stdbuf", "setsid",
    "doas", "sudo", "command", "builtin", "exec", "watch",
})


def _blocked_binary(args: list[str], blocked: set[str]) -> str | None:
    """The denylisted program this command would actually run, or None.

    `args[0] in blocked` was a literal string match, so the denylist was a list
    of spellings rather than of programs. Both of these ran to completion with
    no prompt on 2026-08-27, with "rm" on the list the whole time:

        /bin/rm /tmp/symbio_victim.txt     -> (True, '')   file deleted
        env rm  /tmp/nothing_here          -> rm executed

    A denylist any absolute path defeats is decoration.

    Two rules. Always match on the BASENAME, so /bin/rm, /usr/bin/rm and
    ../../bin/rm all resolve to "rm". And when the command starts with a
    wrapper that runs another program named in its arguments, scan every
    remaining token rather than trying to model each wrapper's option grammar —
    walking token by token stopped at the "5" in `nice -n 5 curl ...` and let
    curl through. Scanning everything can over-match (`env grep rm f` asks
    about rm), but a wrapper invocation is rare, the prompt is one keypress,
    and the failure it replaces silently deleted a file.
    """
    first = os.path.basename(args[0]) if args else ""
    if first in blocked:
        return first
    if first not in _COMMAND_WRAPPERS:
        return None
    for token in args[1:]:
        if not token or token.startswith("-"):
            continue
        name = os.path.basename(token)
        if name in blocked:
            return name
    return None


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
    hit = _blocked_binary(args, blocked)
    if hit is not None:
        # Blocked commands are not refused outright: the user can approve a
        # one-off run. Non-interactive callers (cron thread) never prompt.
        if not interactive or not _ask_command_permission(command, hit, ask_fn=confirm_fn):
            # A dead-end block is how the model gets stranded: it reaches for
            # the obvious tool (curl), is told only "no", finds no way to make
            # an HTTP call, and falls back to inventing the response. So when
            # the blocked binary HAS a working in-sandbox equivalent, name it.
            # curl/wget stay blocked for real reasons (any protocol, file://,
            # writing files, redirect-driven SSRF); fetch() is http/https-only,
            # read-only, size-capped, and surfaces the status code + body.
            redirect = _NETWORK_TOOL_REDIRECT.get(hit)
            hint = f" {redirect}" if redirect else ""
            return False, f"'{hit}' is blocked in sandbox (user did not approve it).{hint}"

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


# What symbio_tools offers in place of a library the script fumbled. The model
# follows what it was trained on, not what the tool catalog says: asked to parse
# a page it reached for selectolax three times running — CSSSelector, selectors,
# CSSSelector again — because its skill note says "parse with selectolax", and
# never once tried the select() sitting in the stub. Live 2026-08-25, three
# rounds burned on import errors. Another line of description does not fix that;
# the failure itself has to carry the answer.
_STUB_ALTERNATIVES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("selectolax", "bs4", "beautifulsoup", "lxml", "html.parser", "htmlparser"),
     "select(html, css) -> [{'text':..., 'attrs':{...}}]  (no parser library needed)"),
    (("requests", "urllib", "httpx", "http.client", "aiohttp"),
     "fetch(url) -> the page body as text"),
    (("pathlib", "shutil", "tempfile"),
     "read_file(path), write_file(path, text), patch(path, old, new), list_dir(path)"),
    (("os", "sys", "subprocess", "glob"),
     "list_dir(path) and search_files(query, glob) for the project; there is no shell"),
)


# An HTTP call whose response was parsed without the status ever being read.
_HTTP_CALL_RE = re.compile(
    r"\b(?:requests|httpx)\.(?:get|post|put|patch|delete|head|request)\s*\(")
_STATUS_CHECK_RE = re.compile(
    r"\.status_code\b|\.raise_for_status\s*\(|\.is_success\b|\.ok\b")


def _http_status_hint(code: str, out: str) -> str:
    """Guidance for a script that parsed a response body it never validated.

    Driving the CLI against an API that had moved (410) and was rate-limiting
    (429) on 2026-08-27, the model wrote `requests.get(...).json()['items']`
    three times running and never once looked at `.status_code`. Both failures
    were invisible *as failures*: requests returns a response object for 4xx
    just as it does for 200, so the only symptom was a KeyError on the error
    document. It read that as "wrong key" and went hunting for another key.

    Nothing else in the stack can catch this. _verify_api_usage checks
    attributes against imported classes, and `data['items']` is a dict
    subscript on a value that only exists at runtime.
    """
    if not _HTTP_CALL_RE.search(code) or _STATUS_CHECK_RE.search(code):
        return ""
    lowered = out.lower()
    if not any(m in lowered for m in
               ("keyerror", "typeerror", "indexerror", "jsondecodeerror",
                "expecting value", "not subscriptable")):
        return ""
    return (
        "\n\n[The script never checked the HTTP status. requests/httpx return a "
        "response object for 4xx and 5xx exactly as they do for 200, so .json() "
        "just parsed the *error* document — which is why the key you expected is "
        "missing. Print response.status_code and response.text before parsing.\n"
        "  404/410: the body usually names the path that replaced it.\n"
        "  401/403: the body usually names the header or scheme it wanted.\n"
        "  429: retry the same call after response.headers.get('Retry-After') "
        "seconds — loop inside this one script rather than answering without "
        "the data.]")


def _stub_hint_for_failure(code: str, out: str) -> str:
    """Guidance to append when a script failed over something the stub provides."""
    hints = _http_status_hint(code, out)
    lowered = out.lower()
    if not any(m in lowered for m in
               ("importerror", "modulenotfounderror", "no module named",
                "has no attribute", "cannot import name")):
        return hints
    offered = [repl for mods, repl in _STUB_ALTERNATIVES
               if any(m in lowered for m in mods)]
    if not offered:
        return hints
    already = "symbio_tools" in code
    lead = ("You already import symbio_tools — it also provides:"
            if already else
            "The sandbox provides these instead, via `from symbio_tools import ...`:")
    return hints + ("\n\n[" + lead + "\n  " + "\n  ".join(offered)
                    + "\nUse them rather than guessing another library's API.]")


# ---------------------------------------------------------------------------
# The script-side helper API.
# ---------------------------------------------------------------------------
# Ported from the Hermes agent's sandbox (symbio/sandbox.py), which has had one
# all along while this module — the loop the CLI actually runs — offered nothing
# and told the model "scripts are for pure computation".
#
# The point is that capability arrives as a curated API rather than as a hole in
# the import blocklist. A script that needs to write a file gets write_file,
# which resolves through _project_path and cannot leave the project. It does not
# need pathlib, and it does not need the bare open() builtin — which is
# unrestricted, can write anywhere on the disk, and was the sandbox's real
# exposure all along. Making the safe path the easy one is the only version of
# this that holds.
#
# fetch() is likewise real capability without a blocklist change: urllib stays
# refused to the *script*, while the stub — which is ours, not the model's —
# imports it on the script's behalf behind a http/https check.
#
# Deliberately NOT ported: the Hermes stub's terminal(), which runs
# subprocess.run on whatever it is handed. That is every entry in
# sandbox.blocked_commands bypassed by one line of Python. A script that needs a
# command should come back and ask for run_command, where the denylist and the
# risk gate live.
def _tools_stub_source() -> str:
    """Source of the symbio_tools module a sandboxed script may import."""
    # Plain substitution, not str.format: the stub is Python source and is full
    # of braces (dict literals, f-strings), every one of which would have to be
    # doubled to survive formatting. It did not survive — adding select() broke
    # the whole stub with KeyError: '"text"'.
    return (_STUB_TEMPLATE
            .replace("@@PROJECT@@", repr(str(constants.PROJECT_DIR)))
            .replace("@@SANDBOX@@", repr(str(constants.SANDBOX_DIR))))


_STUB_TEMPLATE = '''# Auto-generated for sandboxed scripts. Rewritten every run; do not edit.
import urllib.error as _ue
import urllib.request as _u
from pathlib import Path as _Path

PROJECT_DIR = _Path(@@PROJECT@@)
SANDBOX_DIR = _Path(@@SANDBOX@@)


def _project_path(path, must_exist=False):
    target = (PROJECT_DIR / str(path)).resolve()
    if not str(target).startswith(str(PROJECT_DIR.resolve())):
        raise ValueError("Path must be inside the project directory.")
    if must_exist and not target.exists():
        raise FileNotFoundError(path)
    return target


def read_file(path, offset=1, limit=200):
    """Text of a project file, as a string."""
    target = _project_path(path, must_exist=True)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\\n".join(lines[max(0, offset - 1):max(0, offset - 1) + limit])


def write_file(path, content):
    """Write a project file, creating parent directories."""
    target = _project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    return "Wrote " + str(path) + "."


def patch(path, old_text, new_text):
    """Replace the first occurrence of old_text in a project file."""
    target = _project_path(path, must_exist=True)
    content = target.read_text(encoding="utf-8")
    if old_text not in content:
        raise ValueError("old_text not found")
    target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return "Patched " + str(path) + "."


def list_dir(path="."):
    """Names in a project directory; directories get a trailing slash."""
    target = _project_path(path, must_exist=True)
    return sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())


def search_files(query, glob="*"):
    """Project files whose text contains query. Capped at 50."""
    matches = []
    for f in PROJECT_DIR.rglob(glob):
        if not f.is_file() or ".git" in f.parts or "venv" in f.parts:
            continue
        try:
            if query.lower() in f.read_text(encoding="utf-8", errors="replace").lower():
                matches.append(str(f.relative_to(PROJECT_DIR)))
        except OSError:
            pass
        if len(matches) >= 50:
            break
    return matches


def select(html, css):
    """Elements matching a CSS selector, as plain dicts.

    Each is {"text": ..., "attrs": {...}, "html": ...}. Deliberately not a
    selectolax object: the 8B could not retain that library's API across three
    corrections in a row — it tried tree.parse(), selectolax.fromstring() and
    HTMLParser().parse(), none of which exist — and a script should not have to
    know a third-party object model to read a page.
    """
    from selectolax.parser import HTMLParser as _H
    out = []
    for node in _H(str(html)).css(str(css)):
        out.append({
            "text": node.text(strip=True),
            "attrs": dict(node.attributes),
            "html": node.html,
        })
    return out


def select_one(html, css):
    """First element matching a CSS selector, or None."""
    found = select(html, css)
    return found[0] if found else None


def fetch(url, timeout=15):
    """Raw body of an http/https URL, as text.

    A non-2xx raises, and the message carries the status AND the body: urlopen
    alone raises "HTTP Error 410: Gone" and drops the response, which is exactly
    where an API puts "/items was removed in v2. Use /v2/items." Retry-After is
    surfaced for the same reason -- a 429 is a "same call, later", and the
    number is useless if it never reaches the caller.
    """
    if not str(url).startswith(("http://", "https://")):
        raise ValueError("Only http/https URLs can be fetched.")
    req = _u.Request(str(url), headers={"User-Agent": "symbio-sandbox/1.0"})
    try:
        with _u.urlopen(req, timeout=timeout) as resp:
            return resp.read(1_000_000).decode("utf-8", errors="replace")
    except _ue.HTTPError as e:
        body = e.read(20_000).decode("utf-8", errors="replace")
        retry = e.headers.get("Retry-After") if e.headers else None
        extra = f" Retry-After: {retry}s." if retry else ""
        raise RuntimeError(
            f"HTTP {e.code} from {url}.{extra} Response body: {body}") from None
'''


def _is_code_safe(code: str, blocked_imports: set[str]) -> tuple[bool, str]:
    """Reject imports that would let sandboxed code touch the filesystem,
    network, or host process; scripts are for pure computation."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Show the offending line. "expected an indented block after 'if'
        # statement on line 11" names a line the model cannot see, so it
        # regenerates from memory and reproduces the same mistake — observed
        # 2026-08-25, the identical error three rounds running, each announced
        # as a fix. Quoting the line turns a description of the fault into the
        # fault itself.
        detail = f"Syntax error: {e}"
        lines = code.splitlines()
        lineno = getattr(e, "lineno", None)
        if lineno and 1 <= lineno <= len(lines):
            start = max(0, lineno - 3)
            window = []
            for i in range(start, min(len(lines), lineno + 1)):
                marker = ">>" if i == lineno - 1 else "  "
                window.append(f"{marker} {i + 1:>3} | {lines[i]}")
            offset = getattr(e, "offset", None)
            if offset and 0 < offset <= len(lines[lineno - 1]) + 1:
                window.append(" " * (offset + 8) + "^")
            detail += "\n\nYour code around that line:\n" + "\n".join(window)
        return False, detail
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
        elif isinstance(node, ast.Call):
            # The blocklist above reads import *statements*, so every dynamic
            # route to the same module walked straight past it:
            #     __import__('shutil').rmtree('adapters')
            #     eval(compile("import shutil", 'x', 'exec'))
            # Both parse to a Call, not an Import, and both were allowed while
            # `import shutil` on the line above was refused. A blocklist that
            # only reads one spelling of the thing it blocks is decoration.
            func = node.func
            called = (func.id if isinstance(func, ast.Name)
                      else func.attr if isinstance(func, ast.Attribute) else "")
            if called in _DYNAMIC_EXEC:
                return False, (
                    f"'{called}' is not allowed in the sandbox: it can reach "
                    "modules the import blocklist refuses.")
        elif isinstance(node, (ast.Name, ast.Attribute)):
            # The interpreter's own back doors. Blocking __import__ by name is
            # not enough while `getattr(__builtins__, '__import__')` spells the
            # same thing, and __subclasses__ is the classic route from any
            # object back to a module the blocklist refuses.
            ident = (node.id if isinstance(node, ast.Name) else node.attr)
            if ident in _ESCAPE_ATTRS:
                return False, (
                    f"'{ident}' is not allowed in the sandbox: it reaches past "
                    "the import blocklist.")
    return True, ""


# Builtins that execute code or resolve a module by name at runtime. Blocking
# these is what makes the import blocklist mean anything — otherwise it only
# stops the honest spelling.
_DYNAMIC_EXEC = frozenset({
    "__import__", "eval", "exec", "compile", "globals", "vars",
})

# Names that hand back the interpreter regardless of what was imported.
# Deliberately not `getattr` itself, which honest computation uses; the escape
# needs one of these as its target, so blocking them costs nothing legitimate.
_ESCAPE_ATTRS = frozenset({
    "__builtins__", "__globals__", "__subclasses__", "__bases__", "__mro__",
    "__loader__", "__spec__", "__code__", "__closure__",
})


def run_python_code(code: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Run a short Python script in the sandbox directory."""
    code = code.strip()
    if not code:
        return False, "Empty code."
    safe, msg = _is_code_safe(code, set(config["sandbox"]["blocked_imports"]))
    if not safe:
        return False, msg

    # Refreshed every run so an edited stub on disk can never persist, and so
    # a script written against an older shape fails loudly rather than silently
    # importing something stale.
    try:
        constants.SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        (constants.SANDBOX_DIR / "symbio_tools.py").write_text(
            _tools_stub_source(), encoding="utf-8")
    except OSError:
        pass  # a script that does not import it is unaffected

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
        if result.returncode != 0:
            out += _stub_hint_for_failure(code, out)
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
