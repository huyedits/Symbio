"""The tag language: parsing tool calls out of replies and cleaning text.

Supports two formats:
  1. Hermes-style JSON-in-XML: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
  2. Legacy short tags: <cmd>, <py>, <search>, <note>, <digest />, etc.
"""

import json
import re
from typing import Any

# Map each parsed tool name to the user-facing group used for enable/disable menus.
_TOOL_GROUPS: dict[str, str] = {
    "write_note": "notes",
    "save_skill": "notes",
    "run_command": "terminal",
    "execute_code": "code",
    "web_search": "web_search",
    "read_page": "browser",
    "browser_open": "browser",
    "browser_click": "browser",
    "browser_type": "browser",
    "browser_scroll": "browser",
    "save_memory": "memory",
    "config_show": "config",
    "config_set": "config",
    "digest_notes": "digest",
    "train_adapter": "train",
    "schedule_job": "cron",
}

# Hermes-style tool registry: JSON schemas for the system prompt <tools> block.
_TOOLS: list[dict[str, Any]] = [
    {
        "name": "terminal",
        "description": "Run a sandboxed shell command and return its output. Use when the user asks you to run a command.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string", "description": "The shell command to run."}},
            "required": ["cmd"],
        },
    },
    {
        "name": "execute_code",
        "description": "Run a short Python script in the sandbox directory (pure computation; no os/network imports).",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to execute."}},
            "required": ["code"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for a query and return result snippets.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": "Fetch a URL's text content.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to read."}},
            "required": ["url"],
        },
    },
    {
        "name": "browser_open",
        "description": "Open a URL in the live browser and return the page text.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to open."}},
            "required": ["url"],
        },
    },
    {
        "name": "write_note",
        "description": "Save a markdown note in notes/.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note filename title."},
                "body": {"type": "string", "description": "Markdown content."},
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "save_memory",
        "description": "Append a durable fact to always-in-context memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact to remember."},
                "replace": {"type": "boolean", "description": "If true, replace all existing memory."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "digest_notes",
        "description": "Convert unsaved/changed notes into training samples.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "train_adapter",
        "description": "Fine-tune the LoRA adapter on accumulated training data.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "config_show",
        "description": "Show the current configuration.",
        "parameters": {"type": "object", "properties": {}},
    },
]

# Hermes name -> internal name (most are already the same).
_HERMES_NAME_MAP: dict[str, str] = {
    "terminal": "run_command",
}

# Common thinking/reasoning delimiters that must never reach the user or training data.
_THINKING_PATTERNS = [
    r"<thinking\b[^>]*>.*?</thinking>",
    r"</?thinking\b[^>]*>.*?</?thinking>",
    r"<analysis\b[^>]*>.*?</analysis>",
    r"<reasoning\b[^>]*>.*?</reasoning>",
    r"<think\b[^>]*>.*?</think>",
    r" thinking\s+.*?/thinking",
    r"\bthinking\s*:?\s*\n.*?\n/?thinking",
    r"\breasoning\s*:?\s*\n.*?\n/?reasoning",
]


def clean_response(text: str) -> str:
    for pattern in _THINKING_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^user:\s*", "", text, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tool_group(name: str) -> str | None:
    """Return the user-facing group for a tool name, or None if unknown."""
    return _TOOL_GROUPS.get(name)


def build_tools_block() -> str:
    """Return the Hermes-style <tools> JSON block for the system prompt."""
    return "<tools>" + json.dumps(_TOOLS, indent=2, ensure_ascii=False) + "</tools>"


def tool_schemas() -> list[dict[str, Any]]:
    """Return the tool registry as a list of JSON schemas."""
    return list(_TOOLS)


def parse_tools(reply: str, enabled_groups: set[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Extract tool calls from the model reply.

    If `enabled_groups` is provided, drop tools whose group is disabled.
    """
    tools: list[tuple[str, dict[str, Any]]] = []

    for m in re.finditer(
        r'<note\s+title=[\'"]([^\'"]*?)[\'"]>(.*?)</note>', reply, re.DOTALL
    ):
        tools.append(("write_note", {
            "title": m.group(1).strip(),
            "body": m.group(2).strip(),
        }))

    for m in re.finditer(r'<cmd>(.*?)</cmd>', reply, re.DOTALL):
        tools.append(("run_command", {"cmd": m.group(1).strip()}))

    for m in re.finditer(r'<py>(.*?)</py>', reply, re.DOTALL):
        tools.append(("execute_code", {"code": m.group(1).strip()}))

    for m in re.finditer(r'<search>(.*?)</search>', reply, re.DOTALL):
        tools.append(("web_search", {"query": m.group(1).strip()}))

    for m in re.finditer(r'<read>(.*?)</read>', reply, re.DOTALL):
        tools.append(("read_page", {"url": m.group(1).strip()}))

    for m in re.finditer(r'<browse>(.*?)</browse>', reply, re.DOTALL):
        tools.append(("browser_open", {"url": m.group(1).strip()}))

    for m in re.finditer(r'<click>(.*?)</click>', reply, re.DOTALL):
        tools.append(("browser_click", {"target": m.group(1).strip()}))

    for m in re.finditer(
        r'<type(\s+enter=[\'"](?:true|yes|1)[\'"])?>(.*?)</type>', reply, re.DOTALL
    ):
        tools.append(("browser_type", {
            "text": m.group(2).strip(),
            "enter": bool(m.group(1)),
        }))

    for m in re.finditer(r'<scroll(?:\s+dir=[\'"](up|down)[\'"])?\s*/>', reply):
        tools.append(("browser_scroll", {"direction": m.group(1) or "down"}))

    for m in re.finditer(
        r'<skill\s+name=[\'"]([^\'"]*?)[\'"]>(.*?)</skill>', reply, re.DOTALL
    ):
        tools.append(("save_skill", {
            "name": m.group(1).strip(),
            "steps": m.group(2).strip(),
        }))

    if re.search(r'<config\s+show\s*/>', reply):
        tools.append(("config_show", {}))

    for m in re.finditer(
        r'<config\s+set=[\'"]([^\'"]+)[\'"]>(.*?)</config>', reply, re.DOTALL
    ):
        tools.append(("config_set", {
            "key": m.group(1).strip(),
            "value": m.group(2).strip(),
        }))

    for m in re.finditer(
        r'<memory(\s+replace=[\'"]all[\'"])?>(.*?)</memory>', reply, re.DOTALL
    ):
        tools.append(("save_memory", {
            "store": "memory",
            "content": m.group(2).strip(),
            "replace": bool(m.group(1)),
        }))

    for m in re.finditer(
        r'<profile(\s+replace=[\'"]all[\'"])?>(.*?)</profile>', reply, re.DOTALL
    ):
        tools.append(("save_memory", {
            "store": "profile",
            "content": m.group(2).strip(),
            "replace": bool(m.group(1)),
        }))

    if re.search(r'<digest\s*/>', reply) or re.search(r'<digest></digest>', reply):
        tools.append(("digest_notes", {}))

    if re.search(r'<train\s*/>', reply) or re.search(r'<train></train>', reply):
        tools.append(("train_adapter", {}))

    for m in re.finditer(r'<cron\s+expr=[\'"]([^\'"]*?)[\'"]>(.*?)</cron>', reply, re.DOTALL):
        tools.append(("schedule_job", {
            "schedule": m.group(1).strip(),
            "text": m.group(2).strip(),
        }))

    for m in re.finditer(r'<cron\s+at=[\'"]([^\'"]*?)[\'"]>(.*?)</cron>', reply, re.DOTALL):
        tools.append(("schedule_job", {
            "schedule": "at " + m.group(1).strip(),
            "text": m.group(2).strip(),
        }))

    # Hermes-style JSON-in-XML tool calls.
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', reply, re.DOTALL):
        try:
            call = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("function")
        params = call.get("arguments") or call.get("parameters") or call.get("args") or {}
        if not isinstance(params, dict):
            params = {}
        if isinstance(name, str):
            internal_name = _HERMES_NAME_MAP.get(name, name)
            tools.append((internal_name, params))

    if enabled_groups is not None:
        tools = [
            (name, params) for name, params in tools
            if _TOOL_GROUPS.get(name) in enabled_groups
        ]
    return tools


def strip_tool_tags(reply: str) -> str:
    display = reply
    display = re.sub(r'<note\s+title=[\'"][^\'"]*?[\'"]>(.*?)</note>', '', display, flags=re.DOTALL)
    display = re.sub(r'<cmd>(.*?)</cmd>', '', display, flags=re.DOTALL)
    display = re.sub(r'<py>(.*?)</py>', '', display, flags=re.DOTALL)
    display = re.sub(r'<search>(.*?)</search>', '', display, flags=re.DOTALL)
    display = re.sub(r'<read>(.*?)</read>', '', display, flags=re.DOTALL)
    display = re.sub(r'<browse>(.*?)</browse>', '', display, flags=re.DOTALL)
    display = re.sub(r'<click>(.*?)</click>', '', display, flags=re.DOTALL)
    display = re.sub(r'<type[^>]*>(.*?)</type>', '', display, flags=re.DOTALL)
    display = re.sub(r'<scroll[^>]*/>', '', display)
    display = re.sub(r'<skill\s+name=[\'"][^\'"]*?[\'"]>(.*?)</skill>', '', display, flags=re.DOTALL)
    display = re.sub(r'<memory[^>]*>(.*?)</memory>', '', display, flags=re.DOTALL)
    display = re.sub(r'<profile[^>]*>(.*?)</profile>', '', display, flags=re.DOTALL)
    display = re.sub(r'<config\s+show\s*/>', '', display)
    display = re.sub(r'<config\s+set=[\'"][^\'"]+[\'"]>(.*?)</config>', '', display, flags=re.DOTALL)
    display = re.sub(r'<digest\s*/>', '', display)
    display = re.sub(r'<digest></digest>', '', display)
    display = re.sub(r'<train\s*/>', '', display)
    display = re.sub(r'<train></train>', '', display)
    display = re.sub(r'<cron\s+[^>]*?>(.*?)</cron>', '', display, flags=re.DOTALL)
    display = re.sub(r'<tool_call>\s*.*?\s*</tool_call>', '', display, flags=re.DOTALL)
    # A reply cut off mid-tag leaves an unterminated tag; never show it.
    display = re.sub(
        r'<(?:cmd|py|search|read|browse|click|type|scroll|note|skill|cron|digest|train|memory|profile|config|tool_call)\b[^>]*>[^<]*$',
        '', display, flags=re.DOTALL,
    )
    return clean_response(display)
