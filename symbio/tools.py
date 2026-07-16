"""AIAgent tool registry and standalone tool runners for Symbio."""

from __future__ import annotations

import concurrent.futures
import hashlib
import imaplib
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from email import message_from_bytes
from email.header import decode_header
from email.mime.text import MIMEText
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import smtplib

from symbio.computer import (
    BrowserSession,
    desktop_click,
    desktop_move,
    desktop_press,
    desktop_screenshot,
    desktop_type,
)
from symbio.config import can_run_lora, detect_model_type
from symbio.constants import (
    ADAPTER_DIR,
    DEFAULT_CONFIG,
    NOTES_DIR,
    PROJECT_DIR,
    _SHELL_COMMANDS,
)
from symbio.sandbox import _run_execute_code, _run_sandboxed
from symbio.store import SessionStore
from symbio.utils import _project_path, _safe_note_filename, _truncated, save_note

if TYPE_CHECKING:
    from symbio.agent import AIAgent


logger = logging.getLogger("chat")


def build_tool_registry(agent: AIAgent) -> list[dict[str, Any]]:
    """Return the full Hermes-style tool registry for an agent instance."""
    return [
        {
            "name": "note",
            "description": "Save, update, or remove a fact as a markdown note in notes/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                    "target": {"type": "string", "enum": ["note", "user"]},
                    "content": {"type": "string"},
                    "title": {"type": "string"},
                    "old_text": {"type": "string"},
                },
                "required": ["action", "target", "content"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_note(a, params),
        },
        {
            "name": "read_file",
            "description": "Read a text file inside the project directory, with optional offset/limit lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_read_file(a, params),
        },
        {
            "name": "write_file",
            "description": "Write or replace a file inside the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_write_file(a, params),
        },
        {
            "name": "patch",
            "description": "Apply a targeted find-and-replace edit to a file inside the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_patch(a, params),
        },
        {
            "name": "search_files",
            "description": "Search file contents or filenames inside the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["query"],
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_search_files(a, params),
        },
        {
            "name": "terminal",
            "description": "Run a sandboxed shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                },
                "required": ["cmd"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_terminal(a, params),
        },
        {
            "name": "execute_code",
            "description": "Run a short Python script in the sandbox directory that can call whitelisted tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
                "required": ["code"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_execute_code(a, params),
        },
        {
            "name": "web_search",
            "description": "Search the web for a query. Returns stub unless configured.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_web_search(a, params),
        },
        {
            "name": "web_extract",
            "description": "Extract page content as markdown. Returns stub unless configured.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_web_extract(a, params),
        },
        {
            "name": "list_threads",
            "description": "List unread email threads from the configured inbox.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_list_threads(a, params),
        },
        {
            "name": "get_thread",
            "description": "Read a specific email by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_get_thread(a, params),
        },
        {
            "name": "send_message",
            "description": "Send a new email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_send_message(a, params),
        },
        {
            "name": "reply_to_message",
            "description": "Reply to an email by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["id", "body"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_reply_to_message(a, params),
        },
        {
            "name": "digest_notes",
            "description": "Convert unsaved/changed notes into training samples.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": False,
            "run": lambda params, a=agent: _tool_digest_notes(a, params),
        },
        {
            "name": "train_adapter",
            "description": "Fine-tune the LoRA adapter on accumulated training data.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": False,
            "run": lambda params, a=agent: _tool_train_adapter(a, params),
        },
        {
            "name": "session_search",
            "description": "Search past conversation turns.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_session_search(a, params),
        },
        {
            "name": "browser_open",
            "description": "Open a web browser and navigate to a URL. Only http/https URLs are allowed; new domains require confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "channel": {"type": "string", "description": "Optional browser channel: chromium, chrome, safari. Defaults to bundled Chromium."},
                },
                "required": ["url"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_open(a, params),
        },
        {
            "name": "browser_navigate",
            "description": "Navigate the current browser tab to a new URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_navigate(a, params),
        },
        {
            "name": "browser_click",
            "description": "Click an element in the browser by CSS selector or visible text. Prefer text for buttons/links; selectors click the first visible match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_click(a, params),
        },
        {
            "name": "browser_type",
            "description": "Type text into the currently focused browser element or a selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "selector": {"type": "string"},
                    "press_enter": {"type": "boolean"},
                },
                "required": ["text"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_type(a, params),
        },
        {
            "name": "browser_press",
            "description": "Press a keyboard key in the browser (e.g. Enter, Tab, Escape).",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_press(a, params),
        },
        {
            "name": "browser_scroll",
            "description": "Scroll the current browser page up or down (e.g. to the next video in a shorts feed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["down", "up"]},
                    "amount": {"type": "integer", "description": "Pixels to scroll; default 800."},
                },
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_scroll(a, params),
        },
        {
            "name": "browser_get_text",
            "description": "Return the visible text of the current browser page.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": True,
            # Playwright's sync API binds to the thread that opened the
            # browser; never run these on the parallel executor.
            "serial": True,
            "run": lambda params, a=agent: _tool_browser_get_text(a, params),
        },
        {
            "name": "browser_get_html",
            "description": "Return the HTML of the current browser page.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": True,
            "serial": True,
            "run": lambda params, a=agent: _tool_browser_get_html(a, params),
        },
        {
            "name": "browser_evaluate",
            "description": "Evaluate JavaScript in the current browser page and return the result.",
            "parameters": {
                "type": "object",
                "properties": {"script": {"type": "string"}},
                "required": ["script"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_evaluate(a, params),
        },
        {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current browser page and save it to screenshots/. The user can view the file; the model receives the saved path.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_screenshot(a, params),
        },
        {
            "name": "browser_close",
            "description": "Close the browser session.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": False,
            "run": lambda params, a=agent: _tool_browser_close(a, params),
        },
        {
            "name": "desktop_screenshot",
            "description": "Take a full desktop screenshot and save it to screenshots/. The user can view the file; the model receives the saved path.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop_screenshot(a, params),
        },
        {
            "name": "desktop_click",
            "description": "Click the mouse at the given screen coordinates (x, y).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "clicks": {"type": "integer"},
                    "button": {"type": "string"},
                },
                "required": ["x", "y"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop_click(a, params),
        },
        {
            "name": "desktop_move",
            "description": "Move the mouse to the given screen coordinates (x, y).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop_move(a, params),
        },
        {
            "name": "desktop_type",
            "description": "Type text at the current desktop focus.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop_type(a, params),
        },
        {
            "name": "desktop_press",
            "description": "Press a keyboard key on the desktop (e.g. command, space, return).",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop_press(a, params),
        },
    ]


def openai_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool definitions in the OpenAI-style format expected by Qwen's chat template."""
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        schemas.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })
    return schemas


def tool_few_shots(config: dict[str, Any]) -> list[dict[str, str]]:
    """Return a short set of canonical tool-use examples for the model."""
    aname = config["assistant_name"]
    uname = config["user_name"]
    return [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": f"My name is {aname}."},
        {"role": "user", "content": f"Remember that {uname} likes coffee."},
        {
            "role": "assistant",
            "content": (
                f'<tool_call>{{"name": "note", "arguments": '
                f'{{"action": "add", "target": "note", "title": "User Preference", '
                f'"content": "{uname} likes coffee."}}}}</tool_call>Noted.'
            ),
        },
        {"role": "user", "content": "Show me config.json."},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>Reading config.json.',
        },
        {"role": "user", "content": "What is in the project directory?"},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls -la"}}</tool_call>Listing the project directory.',
        },
        {"role": "user", "content": "Check my unread emails."},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "list_threads", "arguments": {}}</tool_call>Checking unread emails.',
        },
    ]


def tool_metadata(name: str, tools: list[dict[str, Any]], agent: AIAgent) -> dict[str, Any]:
    """Return metadata for a named tool, including shell-command fallbacks."""
    for t in tools:
        if t["name"] == name:
            return t
    # Fallback: some small models emit shell commands as Hermes tool names.
    if name in _SHELL_COMMANDS:
        return {
            "readonly": False,
            "run": lambda params, n=name, a=agent: _tool_terminal(a, {"cmd": n}),
        }
    return {"readonly": False, "run": lambda _: f"Unknown tool: {name}"}


def _tool_note(agent: AIAgent, args: dict[str, Any]) -> str:
    """Save, replace, or remove a markdown note in notes/."""
    action = args.get("action", "add")
    target = args.get("target", "note")
    content = args.get("content", "")
    title = args.get("title", "")
    old_text = args.get("old_text", "")

    if action == "add":
        if not title:
            # Derive a title from the first line or target.
            lines = content.strip().splitlines()
            title = lines[0].strip() if lines else f"{target.capitalize()} Note"
        if len(title) > 60:
            title = title[:60] + "..."
        path = save_note(title, content.strip())
        agent.retriever.invalidate_cache()
        agent.planner.record_note_ref(path.name)
        return f"Saved note: {path.name}."

    if action in ("replace", "remove"):
        if not old_text:
            return "Error: old_text is required for replace/remove."
        # If a title is supplied, narrow the search to that note first.
        candidate_files = sorted(NOTES_DIR.glob("*.md"))
        if title:
            titled_path = NOTES_DIR / _safe_note_filename(title)
            if titled_path.exists():
                candidate_files = [titled_path]
            else:
                candidate_files = [
                    f for f in candidate_files
                    if title.lower() in f.read_text(encoding="utf-8", errors="replace").splitlines()[0].lower()
                ]
        matches = []
        for f in candidate_files:
            text = f.read_text(encoding="utf-8", errors="replace")
            if old_text in text:
                matches.append(f)
        if not matches:
            return "Error: old_text not found in any note."
        if len(matches) > 1:
            names = ", ".join(f.name for f in matches)
            return f"Error: old_text found in multiple notes ({names}); be more specific."
        path = matches[0]
        current = path.read_text(encoding="utf-8")
        if action == "replace":
            new = current.replace(old_text, content, 1)
            path.write_text(new, encoding="utf-8")
            agent.retriever.invalidate_cache()
            agent.planner.record_note_ref(path.name)
            return f"Updated note: {path.name}."
        else:
            new = current.replace(old_text, "", 1).strip()
            if new and not new.startswith("#"):
                new = f"# {path.stem.replace('_', ' ')}\n\n{new}"
            path.write_text(new, encoding="utf-8")
            agent.retriever.invalidate_cache()
            agent.planner.record_note_ref(path.name)
            return f"Removed from note: {path.name}."

    return f"Unknown note action: {action}"


def _tool_read_file(agent: AIAgent, args: dict[str, Any]) -> str:
    try:
        target = _project_path(args["path"], must_exist=True)
    except Exception as e:
        return str(e)
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        offset = max(0, args.get("offset", 1) - 1)
        limit = args.get("limit", 100)
        if limit <= 0:
            limit = 100
        selected = lines[offset:offset + limit]
        numbered = "\n".join(f"{offset + i + 1}: {line}" for i, line in enumerate(selected))
        header = f"File: {args['path']} (lines {offset + 1}-{offset + len(selected)} of {len(lines)})\n"
        return _truncated(header + numbered, agent.config["agent"]["max_output_len"])
    except Exception as e:
        return f"Failed to read file: {e}"


def _tool_write_file(agent: AIAgent, args: dict[str, Any]) -> str:
    try:
        target = _project_path(args["path"])
    except Exception as e:
        return str(e)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.get("content", ""), encoding="utf-8")
        return f"Wrote {args['path']}."
    except Exception as e:
        return f"Failed to write file: {e}"


def _tool_patch(agent: AIAgent, args: dict[str, Any]) -> str:
    try:
        target = _project_path(args["path"], must_exist=True)
    except Exception as e:
        return str(e)
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    content = target.read_text(encoding="utf-8")
    if old_text not in content:
        return "Error: old_text not found in file."
    content = content.replace(old_text, new_text, 1)
    target.write_text(content, encoding="utf-8")
    return f"Patched {args['path']}."


def _tool_search_files(agent: AIAgent, args: dict[str, Any]) -> str:
    query = args.get("query", "")
    glob = args.get("glob", "")
    if not query:
        return "No query provided."
    try:
        # Prefer ripgrep if available.
        cmd = ["rg", "-n", "-i", query, str(PROJECT_DIR)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, cwd=str(PROJECT_DIR)
        )
        if result.returncode in (0, 1):
            out = result.stdout.strip()
            if out:
                return _truncated(out, agent.config["agent"]["max_output_len"])
    except Exception:
        pass
    # Fallback: Python glob + simple search.
    matches = []
    files = list(PROJECT_DIR.rglob(glob or "*")) if glob else list(PROJECT_DIR.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if query.lower() in text.lower():
                matches.append(f"{f.relative_to(PROJECT_DIR)}")
        except Exception:
            pass
    if not matches:
        return f"No matches for '{query}'."
    return "\n".join(matches[:50])


def _tool_terminal(agent: AIAgent, args: dict[str, Any]) -> str:
    cmd = args.get("cmd", "")
    ok, out = _run_sandboxed(cmd, agent.config)
    return f"Command '{cmd}' exited {'ok' if ok else 'error'}.\n{out}"


def _tool_execute_code(agent: AIAgent, args: dict[str, Any]) -> str:
    code = args.get("code", "")
    agent._code_calls_this_turn += 1
    if agent._code_calls_this_turn > 1:
        return "Error: only one execute_code call per turn allowed."
    ok, out = _run_execute_code(code, agent.config, agent.tools)
    return f"Code execution {'ok' if ok else 'error'}:\n{out}"


def _tool_web_search(agent: AIAgent, args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return (
        f"Web search is not configured. To enable it, set a search API key or MCP. "
        f"You asked about: {query}"
    )


def _tool_web_extract(agent: AIAgent, args: dict[str, Any]) -> str:
    url = args.get("url", "")
    return f"Web extract is not configured. To enable it, set an extraction API or MCP. URL: {url}"


def _email_config_from_env() -> dict[str, str]:
    keys = ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"]
    cfg = {k: os.environ.get(k, "") for k in keys}
    cfg["EMAIL_ALLOWED_USERS"] = os.environ.get("EMAIL_ALLOWED_USERS", "")
    return cfg


def _email_not_configured() -> str:
    return (
        "Email is not configured. Set these environment variables:\n"
        "  EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, EMAIL_SMTP_HOST\n"
        "Optional: EMAIL_ALLOWED_USERS (comma-separated sender allowlist)."
    )


def _extract_email_text(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisp = str(part.get_content_disposition() or "")
            if ctype == "text/plain" and "attachment" not in cdisp:
                try:
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    break
                except Exception:
                    pass
        if not body:
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/html":
                    try:
                        html = part.get_payload(decode=True).decode("utf-8", errors="replace")
                        body = re.sub(r"<[^>]+>", " ", html)
                        body = re.sub(r"\s+", " ", body).strip()
                        break
                    except Exception:
                        pass
    else:
        try:
            body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
        except Exception:
            body = str(msg.get_payload())
    return body[:8000]


def _decode_header_value(value: str) -> str:
    parts = decode_header(value)
    out = []
    for text, charset in parts:
        if isinstance(text, bytes):
            try:
                out.append(text.decode(charset or "utf-8", errors="replace"))
            except Exception:
                out.append(text.decode("utf-8", errors="replace"))
        else:
            out.append(text)
    return "".join(out)


def _fetch_imap_inbox(limit: int = 20) -> list[dict[str, Any]]:
    cfg = _email_config_from_env()
    if not all(cfg[k] for k in ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST"]):
        raise RuntimeError(_email_not_configured())

    mail = imaplib.IMAP4_SSL(cfg["EMAIL_IMAP_HOST"])
    try:
        mail.login(cfg["EMAIL_ADDRESS"], cfg["EMAIL_PASSWORD"])
        mail.select("inbox")
        _, data = mail.uid("search", None, "(UNSEEN)")
        uids = data[0].split()
        allowed = [a.strip().lower() for a in cfg["EMAIL_ALLOWED_USERS"].split(",") if a.strip()]
        results: list[dict[str, Any]] = []
        for uid in uids[:limit]:
            _, fetched = mail.uid("fetch", uid, "(RFC822)")
            raw = fetched[0][1]
            msg = message_from_bytes(raw)
            sender = _decode_header_value(msg.get("From", ""))
            sender_email = re.search(r"<([^>]+)>", sender)
            sender_email = sender_email.group(1).lower() if sender_email else sender.lower()
            if allowed and sender_email not in allowed:
                continue
            results.append({
                "id": uid.decode(),
                "subject": _decode_header_value(msg.get("Subject", "(no subject)")),
                "from": sender,
                "date": msg.get("Date", ""),
                "body": _extract_email_text(msg),
            })
        return results
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _send_smtp(to: str, subject: str, body: str, in_reply_to: str = "", references: str = "") -> str:
    cfg = _email_config_from_env()
    if not all(cfg[k] for k in ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST"]):
        raise RuntimeError(_email_not_configured())

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = cfg["EMAIL_ADDRESS"]
    msg["To"] = to
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    with smtplib.SMTP_SSL(cfg["EMAIL_SMTP_HOST"]) as server:
        server.login(cfg["EMAIL_ADDRESS"], cfg["EMAIL_PASSWORD"])
        server.sendmail(cfg["EMAIL_ADDRESS"], [to], msg.as_string())
    return f"Sent email to {to} with subject '{subject}'."


def _tool_list_threads(agent: AIAgent, args: dict[str, Any]) -> str:
    try:
        limit = args.get("limit", 20)
        msgs = _fetch_imap_inbox(limit=limit)
        if not msgs:
            return "No unread emails found."
        lines = []
        for m in msgs:
            preview = m["body"][:100].replace("\n", " ")
            lines.append(f"- {m['id']}: {m['subject']} (from {m['from']}) — {preview}...")
        return "Unread emails:\n" + "\n".join(lines)
    except Exception as e:
        return f"Email error: {e}"


def _tool_get_thread(agent: AIAgent, args: dict[str, Any]) -> str:
    try:
        msg_id = args.get("id", "")
        msgs = _fetch_imap_inbox(limit=100)
        for m in msgs:
            if m["id"] == msg_id:
                return (
                    f"From: {m['from']}\n"
                    f"Subject: {m['subject']}\n"
                    f"Date: {m['date']}\n"
                    f"Body:\n{m['body']}"
                )
        return f"Email {msg_id} not found."
    except Exception as e:
        return f"Email error: {e}"


def _tool_send_message(agent: AIAgent, args: dict[str, Any]) -> str:
    try:
        return _send_smtp(args["to"], args["subject"], args["body"])
    except Exception as e:
        return f"Email error: {e}"


def _tool_reply_to_message(agent: AIAgent, args: dict[str, Any]) -> str:
    try:
        msg_id = args.get("id", "")
        msgs = _fetch_imap_inbox(limit=100)
        for m in msgs:
            if m["id"] == msg_id:
                return _send_smtp(m["from"], f"Re: {m['subject']}", args.get("body", ""), in_reply_to=msg_id)
        return f"Email {msg_id} not found."
    except Exception as e:
        return f"Email error: {e}"


def _tool_digest_notes(agent: AIAgent, _args: dict[str, Any]) -> str:
    from symbio.llm import digest_notes_to_training
    try:
        cnt = digest_notes_to_training(agent.tokenizer, agent.system_prompt)
        return f"Digested {cnt} new training samples from notes."
    except Exception as e:
        return f"Digest error: {e}"


def _tool_train_adapter(agent: AIAgent, _args: dict[str, Any]) -> str:
    from symbio.llm import run_training
    trained = run_training(agent.config)
    if trained:
        try:
            from mlx_lm import load
            agent.model, agent.tokenizer = load(
                agent.config["model_name"], adapter_path=str(ADAPTER_DIR)
            )
            agent.adapter_loaded = True
            return "Training complete. Adapter reloaded."
        except Exception as e:
            return f"Training done but adapter reload failed: {e}"
    return "Training skipped (no new data or failed)."


def _tool_session_search(agent: AIAgent, args: dict[str, Any]) -> str:
    query = args.get("query", "")
    rows = agent.store.search(query)
    if not rows:
        return f"No past sessions matched '{query}'."
    lines = [f"Past sessions matching '{query}':"]
    for r in rows:
        preview = r["content"][:120].replace("\n", " ")
        lines.append(f"  [{r['role']}] {preview}")
    return "\n".join(lines)


# ---------- Browser / desktop automation ----------

def _tool_browser_open(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available (playwright not installed)."
    return agent._browser_session.open(args.get("url", ""), channel=args.get("channel", ""))


def _tool_browser_navigate(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.navigate(args.get("url", ""))


def _tool_browser_click(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.click(
        selector=args.get("selector", ""), text=args.get("text", "")
    )


def _tool_browser_type(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.type_text(
        text=args.get("text", ""),
        selector=args.get("selector", ""),
        press_enter=bool(args.get("press_enter", False)),
    )


def _tool_browser_press(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.press(args.get("key", ""))


def _tool_browser_scroll(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.scroll(
        direction=args.get("direction", "down"),
        amount=int(args.get("amount", 0) or 0),
    )


def _tool_browser_get_text(agent: AIAgent, _args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.get_text()


def _tool_browser_get_html(agent: AIAgent, _args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.get_html()


def _tool_browser_evaluate(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.evaluate(args.get("script", ""))


def _tool_browser_screenshot(agent: AIAgent, _args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.screenshot()


def _tool_browser_close(agent: AIAgent, _args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.close()


def _tool_desktop_screenshot(agent: AIAgent, _args: dict[str, Any]) -> str:
    if desktop_screenshot is None:
        return "Desktop automation is not available (pyautogui not installed)."
    return desktop_screenshot()


def _tool_desktop_click(agent: AIAgent, args: dict[str, Any]) -> str:
    if desktop_click is None:
        return "Desktop automation is not available."
    return desktop_click(
        int(args.get("x", 0)),
        int(args.get("y", 0)),
        clicks=int(args.get("clicks", 1)),
        button=args.get("button", "left"),
    )


def _tool_desktop_move(agent: AIAgent, args: dict[str, Any]) -> str:
    if desktop_move is None:
        return "Desktop automation is not available."
    return desktop_move(int(args.get("x", 0)), int(args.get("y", 0)))


def _tool_desktop_type(agent: AIAgent, args: dict[str, Any]) -> str:
    if desktop_type is None:
        return "Desktop automation is not available."
    return desktop_type(args.get("text", ""))


def _tool_desktop_press(agent: AIAgent, args: dict[str, Any]) -> str:
    if desktop_press is None:
        return "Desktop automation is not available."
    return desktop_press(args.get("key", ""))


def _parallel_safe(meta: dict[str, Any]) -> bool:
    """Read-only tools run in parallel unless they are thread-bound (serial)."""
    return bool(meta.get("readonly")) and not meta.get("serial")


def execute_tools(agent: AIAgent, tools: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, str]]:
    """Execute tools. Consecutive parallel-safe read-only tools run concurrently."""
    results: list[tuple[str, str]] = []
    i = 0
    while i < len(tools):
        name, params = tools[i]
        meta = tool_metadata(name, agent.tools, agent)
        if _parallel_safe(meta):
            group: list[tuple[int, str, dict[str, Any]]] = []
            j = i
            while j < len(tools):
                n2, p2 = tools[j]
                if _parallel_safe(tool_metadata(n2, agent.tools, agent)):
                    group.append((j, n2, p2))
                    j += 1
                else:
                    break
            if len(group) == 1:
                _, n, p = group[0]
                results.append((n, run_single_tool(agent, n, p)))
                i = j
                continue
            outs: list[tuple[str, str]] = [("", "")] * len(group)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(group), 4)) as ex:
                futures = {
                    ex.submit(run_single_tool, agent, n, p): k
                    for k, (_, n, p) in enumerate(group)
                }
                for future in concurrent.futures.as_completed(futures):
                    k = futures[future]
                    _, n, _ = group[k]
                    try:
                        outs[k] = (n, future.result())
                    except Exception as e:
                        outs[k] = (n, f"Tool {n} crashed: {e}")
            results.extend(outs)
            i = j
        else:
            results.append((name, run_single_tool(agent, name, params)))
            i += 1
    return results


def run_single_tool(agent: AIAgent, name: str, params: dict[str, Any]) -> str:
    meta = tool_metadata(name, agent.tools, agent)
    runner: Callable[[dict[str, Any]], str] = meta.get("run", lambda _: f"Unknown tool: {name}")
    print(f"  [Tool: {name}]")
    try:
        return runner(params)
    except Exception as e:
        return f"Tool {name} error: {e}"
