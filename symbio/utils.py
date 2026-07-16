"""Small pure helpers for Symbio."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio.constants import (
    DEFAULT_CONFIG,
    MISTAKES_DIR,
    NOTES_DIR,
    PROJECT_DIR,
    _SHELL_COMMANDS,
)

# Common thinking/reasoning delimiters that must never reach the user or training data.
_THINKING_PATTERNS = [
    r"<thinking\b[^>]*>.*?</thinking>",
    r"</?thinking\b[^>]*>.*?</?thinking>",
    r"<analysis\b[^>]*>.*?</analysis>",
    r"<reasoning\b[^>]*>.*?</reasoning>",
    r"<think\b[^>]*>.*?\n?/?think\b[^>]*>",
    r" thinking\s+.*?/thinking",
    r"\bthinking\s*:?\s*\n.*?\n/?thinking",
    r"\breasoning\s*:?\s*\n.*?\n/?reasoning",
]


def clean_response(text: str) -> str:
    """Remove internal reasoning artifacts and clean up whitespace."""
    for pattern in _THINKING_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^user:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_generation_artifacts(text: str) -> str:
    """Remove chat-template artifacts that the model may hallucinate in its output."""
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text)
    text = re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?tool_response>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncated(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len] + "\n... (truncated)"
    return text


def _project_path(path: str, must_exist: bool = False) -> Path:
    target = (PROJECT_DIR / path).resolve()
    if not str(target).startswith(str(PROJECT_DIR)):
        raise ValueError("Path must be inside the project directory.")
    if must_exist and not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return target


def _scrub_env() -> dict[str, str]:
    """Strip env vars that look like secrets."""
    safe = {}
    for k, v in os.environ.items():
        if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD", "AUTH")):
            continue
        safe[k] = v
    return safe


def _safe_note_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)
    safe = safe.strip().replace(" ", "_")[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{safe}.md"


def save_note(title: str, body: str) -> Path:
    """Persist a markdown note in notes/ and return its path."""
    path = NOTES_DIR / _safe_note_filename(title)
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def ensure_seed_notes(config: dict[str, Any]):
    """Create identity notes if notes/ is empty."""
    if any(NOTES_DIR.glob("*.md")):
        return
    save_note("My Identity", f"I am {config['assistant_name']}, a helpful personal AI assistant.")
    save_note("User Identity", f"My user's name is {config['user_name']}.")


def _normalize_tool_call_tags(reply: str) -> str:
    """Fix malformed tool_call spans where the closing tag is wrong or missing."""
    # Turn <tool_call>{...}<tool_call> into <tool_call>{...}</tool_call>.
    reply = re.sub(
        r'(<tool_call\b[^>]*>\s*\{.*?\})\s*<tool_call\b[^>]*>',
        r'\1</tool_call>',
        reply,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Close a trailing <tool_call>{...} that hit end-of-generation unclosed.
    opens = len(re.findall(r'<tool_call\b', reply, re.IGNORECASE))
    closes = len(re.findall(r'</tool_call>', reply, re.IGNORECASE))
    if opens > closes:
        reply = re.sub(
            r'(<tool_call\b[^>]*>\s*\{.*\})\s*$',
            r'\1</tool_call>',
            reply,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return reply


def _escape_json_control_chars(payload: str) -> str:
    """Escape literal control characters inside JSON string values.

    Small models sometimes emit real newlines inside the "code" argument of an
    execute_code tool call. JSON strings cannot contain raw control chars, so
    we escape \n, \t, and \r that appear between unescaped double quotes.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in payload:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"' and not in_string:
            in_string = True
            out.append(ch)
            continue
        if ch == '"' and in_string:
            in_string = False
            out.append(ch)
            continue
        if in_string and ch in "\n\t\r":
            out.append({"\n": "\\n", "\t": "\\t", "\r": "\\r"}[ch])
            continue
        out.append(ch)
    return "".join(out)


def _repair_tool_json(payload: str) -> str:
    """Fix JSON malformations small models commonly emit in tool calls.

    Only used after strict parsing fails, so valid payloads are never touched.
    """
    # Key missing its closing quote: {"url: "https://x"} -> {"url": "https://x"}
    payload = re.sub(r'"(\w+): "', r'"\1": "', payload)
    # Unquoted key: {name: "x"} -> {"name": "x"}
    payload = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', payload)
    # Trailing comma before a closing brace/bracket.
    payload = re.sub(r',\s*([}\]])', r'\1', payload)
    return payload


def parse_tools(reply: str) -> list[tuple[str, dict[str, Any]]]:
    """Extract tool calls from model reply.

    Supports Hermes JSON-in-XML and legacy compact XML tags.
    """
    reply = _normalize_tool_call_tags(reply)
    tools: list[tuple[str, dict[str, Any]]] = []
    hermes_spans: list[tuple[int, int]] = []

    for m in re.finditer(
        r'<tool_call\b[^>]*>(.*?)</tool_call>', reply, re.DOTALL | re.IGNORECASE
    ):
        raw = _escape_json_control_chars(m.group(1).strip())
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                data = json.loads(_repair_tool_json(raw))
            except json.JSONDecodeError:
                continue
        name = data.get("name") or data.get("function")
        arguments = data.get("arguments") or data.get("args", {})
        if not isinstance(arguments, dict):
            arguments = {}
        # Some small models emit shell commands as Hermes tool names.
        if name in _SHELL_COMMANDS:
            name = "terminal"
            arguments = {"cmd": data.get("name", arguments.get("cmd", ""))}
        if name:
            tools.append((name, arguments))
            hermes_spans.append((m.start(), m.end()))

    def _inside_hermes(start: int) -> bool:
        return any(s <= start < e for s, e in hermes_spans)

    # Legacy <note> (double-quoted title)
    for m in re.finditer(
        r'<note\s+title="([^"]*?)"\s*>(.*?)</note>', reply, re.DOTALL
    ):
        if _inside_hermes(m.start()):
            continue
        tools.append(("note", {
            "action": "add",
            "target": "note",
            "title": m.group(1).strip(),
            "content": m.group(2).strip(),
        }))

    # Legacy <note> (single-quoted title, with apostrophe robustness)
    for m in re.finditer(
        r"<note\s+title='((?:[^']|'(?=>))*)'\s*>(.*?)</note>", reply, re.DOTALL
    ):
        if _inside_hermes(m.start()):
            continue
        tools.append(("note", {
            "action": "add",
            "target": "note",
            "title": m.group(1).strip(),
            "content": m.group(2).strip(),
        }))

    # Legacy <cmd>
    for m in re.finditer(r'<cmd>(.*?)</cmd>', reply, re.DOTALL):
        if _inside_hermes(m.start()):
            continue
        tools.append(("terminal", {"cmd": m.group(1).strip()}))

    if re.search(r'<digest\s*/>', reply) or re.search(r'<digest></digest>', reply):
        tools.append(("digest_notes", {}))

    if re.search(r'<train\s*/>', reply) or re.search(r'<train></train>', reply):
        tools.append(("train_adapter", {}))

    return tools


def has_dangling_tool_call(reply: str) -> bool:
    """True if the reply opens a <tool_call> that never parses into a tool."""
    return bool(re.search(r'<tool_call\b', reply, re.IGNORECASE)) and not parse_tools(reply)


def strip_dangling_tool_call(text: str) -> str:
    """Remove an unclosed trailing <tool_call> opener and any partial body."""
    complete = list(re.finditer(r'<tool_call\b[^>]*>.*?</tool_call>', text, re.DOTALL | re.IGNORECASE))
    end = complete[-1].end() if complete else 0
    m = re.search(r'<tool_call\b', text[end:], re.IGNORECASE)
    if m:
        return text[:end + m.start()].rstrip()
    return text


def strip_tool_tags(reply: str) -> str:
    """Return the reply with all tool markup removed."""
    display = reply
    display = re.sub(r'<tool_call\b[^>]*>.*?</tool_call>', '', display, flags=re.DOTALL | re.IGNORECASE)
    # A dangling opener means truncated tool markup: hide it and its partial body.
    display = re.sub(r'<tool_call\b.*$', '', display, flags=re.DOTALL | re.IGNORECASE)
    display = re.sub(r'</tool_call>', '', display, flags=re.IGNORECASE)
    display = re.sub(r'<note\s+title="[^"]*?"\s*>.*?</note>', '', display, flags=re.DOTALL)
    display = re.sub(r"<note\s+title='(?:[^']|'(?=>))*'\s*>.*?</note>", '', display, flags=re.DOTALL)
    display = re.sub(r'<cmd>.*?</cmd>', '', display, flags=re.DOTALL)
    display = re.sub(r'<digest\s*/>', '', display)
    display = re.sub(r'<digest></digest>', '', display)
    display = re.sub(r'<train\s*/>', '', display)
    display = re.sub(r'<train></train>', '', display)
    return strip_generation_artifacts(clean_response(display))


def _safe_mistake_filename(query: str) -> str:
    """Create a short safe filename for a mistake note."""
    slug = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in query)
    slug = slug.strip().replace(" ", "_")[:40].strip("_") or "correction"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slug}.md"
