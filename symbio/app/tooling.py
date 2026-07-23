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
    "browser_press": "browser",
    "save_memory": "memory",
    "config_show": "config",
    "config_set": "config",
    "digest_notes": "digest",
    "train_adapter": "train",
    "schedule_job": "cron",
    "list_cron_jobs": "cron",
    "delete_cron_job": "cron",
    "update_cron_job": "cron",
    "delegate_task": "delegate",
    "brain_solve": "frontier",
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
        "name": "browser_press",
        "description": "Press a keyboard key in the open browser (e.g. 'down', 'up', 'enter', 'esc', 'space'). Use for keyboard navigation; do not invent shell commands for key presses.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key name such as 'down', 'up', 'enter', 'esc', 'space', 'tab', 'home', 'end'."}},
            "required": ["key"],
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
    {
        "name": "schedule_job",
        "description": (
            "Create a new scheduled reminder or command. Always creates a new job; "
            "use delete_cron_job/update_cron_job to change existing jobs. "
            "Use a 5-field cron expression for recurring jobs or 'at YYYY-MM-DD HH:MM' for one-time jobs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "schedule": {
                    "type": "string",
                    "description": "5-field cron expression (minute hour day month weekday) or 'at YYYY-MM-DD HH:MM' for one-time.",
                },
                "text": {
                    "type": "string",
                    "description": "Reminder text, or 'cmd:<shell command>' to run a command when the job fires.",
                },
            },
            "required": ["schedule", "text"],
        },
    },
    {
        "name": "list_cron_jobs",
        "description": "Show all scheduled reminders and commands with their ids and schedules.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "delete_cron_job",
        "description": "Delete a scheduled job by its id (use list_cron_jobs to find the id).",
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer", "description": "The numeric id of the job to delete."}},
            "required": ["job_id"],
        },
    },
    {
        "name": "update_cron_job",
        "description": (
            "Edit an existing scheduled job by id. Use list_cron_jobs to find the id. "
            "Only the schedule and/or text you provide are changed; omitted fields are kept."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer", "description": "The numeric id of the job to edit."},
                "schedule": {
                    "type": "string",
                    "description": "New 5-field cron expression or 'at YYYY-MM-DD HH:MM'.",
                },
                "text": {
                    "type": "string",
                    "description": "New reminder text or 'cmd:<shell command>'.",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "brain_solve",
        "description": (
            "Delegate a difficult reasoning or coding problem to a stronger model "
            "(local Ollama brain first, then frontier fallback). Use when the answer "
            "requires deep reasoning, exact code, or facts beyond your weights."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The full task or question to hand to the stronger model.",
                },
                "use_frontier": {
                    "type": "boolean",
                    "description": "If true, skip the local Ollama brain and call the frontier model directly.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "delegate_task",
        "description": (
            "Hand a bounded sub-task off to a smaller, faster worker model "
            "instead of doing it yourself — use for narrow, repetitive "
            "decisions (e.g. summarizing a page, picking the next browser "
            "click) where a lightweight specialist is enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Which worker to use, e.g. 'summarize' or 'browser'."},
                "task": {"type": "string", "description": "The sub-task text to hand to the worker."},
            },
            "required": ["role", "task"],
        },
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

    for m in re.finditer(r'<press>(.*?)</press>', reply, re.DOTALL):
        tools.append(("browser_press", {"key": m.group(1).strip()}))

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

    for m in re.finditer(
        r'<delegate\s+role=[\'"]([^\'"]*?)[\'"]>(.*?)</delegate>', reply, re.DOTALL
    ):
        tools.append(("delegate_task", {
            "role": m.group(1).strip(),
            "task": m.group(2).strip(),
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


# Each pattern matches only a COMPLETE tag pair (open...close, or a
# self-closing tag) — never a truncated/unterminated one. Shared by
# strip_tool_tags (full replies) and StreamingStripper (incremental chunks),
# so both agree on what "safe to remove" means.
_COMPLETE_TAG_PATTERNS: list[str] = [
    r'<note\s+title=[\'"][^\'"]*?[\'"]>(.*?)</note>',
    r'<cmd>(.*?)</cmd>',
    r'<py>(.*?)</py>',
    r'<search>(.*?)</search>',
    r'<read>(.*?)</read>',
    r'<browse>(.*?)</browse>',
    r'<click>(.*?)</click>',
    r'<type[^>]*>(.*?)</type>',
    r'<scroll[^>]*/>',
    r'<press>(.*?)</press>',
    r'<skill\s+name=[\'"][^\'"]*?[\'"]>(.*?)</skill>',
    r'<memory[^>]*>(.*?)</memory>',
    r'<profile[^>]*>(.*?)</profile>',
    r'<config\s+show\s*/>',
    r'<config\s+set=[\'"][^\'"]+[\'"]>(.*?)</config>',
    r'<digest\s*/>',
    r'<digest></digest>',
    r'<train\s*/>',
    r'<train></train>',
    r'<cron\s+[^>]*?>(.*?)</cron>',
    r'<delegate\s+role=[\'"][^\'"]*?[\'"]>(.*?)</delegate>',
    r'<tool_call>\s*.*?\s*</tool_call>',
]

# Tag names recognized by the unterminated-tag cutoff below and by the
# streaming stripper's "might this become a tag" check.
_KNOWN_TAG_NAMES: tuple[str, ...] = (
    "cmd", "py", "search", "read", "browse", "click", "type", "scroll",
    "press",
    "note", "skill", "cron", "digest", "train", "memory", "profile",
    "config", "tool_call", "delegate",
)

# A reply cut off mid-tag leaves an unterminated tag; never show it.
_UNTERMINATED_TAG_RE = re.compile(
    r'<(?:' + '|'.join(_KNOWN_TAG_NAMES) + r')\b[^>]*>[^<]*$', re.DOTALL,
)


def _strip_complete_tag_pairs(text: str) -> str:
    for pattern in _COMPLETE_TAG_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text


def strip_tool_tags(reply: str) -> str:
    display = _strip_complete_tag_pairs(reply)
    display = _UNTERMINATED_TAG_RE.sub('', display)
    return clean_response(display)


def detect_malformed_tag(reply: str) -> str | None:
    """Did this reply contain something that looked like a tool call but
    never resolved into one — an unterminated tag (likely truncated by
    max_tokens, or just missing its close) or a <tool_call> whose content
    isn't valid JSON? Returns a short description for the model to see and
    self-correct on next round, or None if the reply was clean. Checked
    against the ORIGINAL reply, not stripped text — a syntactically
    complete but JSON-invalid <tool_call> is already removed by
    strip_tool_tags, so it must be caught here instead."""
    unterminated = _UNTERMINATED_TAG_RE.search(_strip_complete_tag_pairs(reply))
    if unterminated:
        return f"An unterminated tag was left open and unusable: {unterminated.group(0)[:120]!r}"
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', reply, re.DOTALL):
        try:
            json.loads(m.group(1).strip())
        except json.JSONDecodeError as e:
            return f"A <tool_call> contained invalid JSON and could not be used: {e}"
    return None


class StreamingStripper:
    """Incremental, best-effort view of a reply as it streams token-by-
    token: known tool tags are held back and dropped once confirmed closed
    (same rule as strip_tool_tags), so raw tag syntax never flashes on
    screen. This is a UX layer only — the authoritative parsed reply is
    still computed from the complete text with strip_tool_tags/parse_tools
    once generation finishes, so a quirk here can never change what the
    agent actually does, only how the in-progress text looks."""

    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Add newly generated text; return the text now safe to display."""
        self._buffer = _strip_complete_tag_pairs(self._buffer + chunk)
        cut = self._first_ambiguous_lt()
        if cut == -1:
            safe, self._buffer = self._buffer, ""
        else:
            safe, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return safe

    def finish(self) -> str:
        """Call once generation ends; returns any remaining safe text —
        plain prose with a stray '<' that never became a tag, or a
        genuinely truncated tag, either way handled by strip_tool_tags."""
        remaining = strip_tool_tags(self._buffer)
        self._buffer = ""
        return remaining

    def _first_ambiguous_lt(self) -> int:
        """Index of a '<' that might still be starting a known tag and
        can't be ruled out yet, or -1 if the buffer is unambiguously safe
        to show as-is (no '<', or every '<' has already diverged from
        every known tag name — e.g. "x < 5" is never held back)."""
        for m in re.finditer('<', self._buffer):
            tail = self._buffer[m.start() + 1:]
            if tail == "" or any(
                name.startswith(tail) or tail.startswith(name)
                for name in _KNOWN_TAG_NAMES
            ):
                return m.start()
        return -1
