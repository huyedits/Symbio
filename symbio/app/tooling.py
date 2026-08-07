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
    "browser_close": "browser",
    "save_memory": "memory",
    "compact_memory": "memory",
    "read_file": "terminal",
    "edit_file": "terminal",
    "write_file": "terminal",
    "config_show": "config",
    "config_set": "config",
    "digest_notes": "digest",
    "train_adapter": "train",
    "retrain_adapter": "train",
    "schedule_job": "cron",
    "list_cron_jobs": "cron",
    "delete_cron_job": "cron",
    "update_cron_job": "cron",
    "delegate_task": "delegate",
    "brain_solve": "frontier",
    "system_check": "system",
    "verify_features": "system",
    "run_remote": "terminal",
    "add_golden_case": "config",
}

# Hermes-style tool registry: JSON schemas for the system prompt <tools> block.
_TOOLS: list[dict[str, Any]] = [
    {
        "name": "terminal",
        "description": "Run a sandboxed shell command and return its output. Use when the user asks you to run a command. Some binaries (e.g. ssh, bash) may require user approval or use run_remote instead.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string", "description": "The shell command to run."}},
            "required": ["cmd"],
        },
    },
    {
        "name": "run_remote",
        "description": "Run a shell command on a configured remote host via SSH, or run a shell command locally if host is localhost. Use for SSHing into servers or for commands that need shell features (pipes, env vars, globs).",
        "parameters": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Configured host alias (from remote.hosts) or 'localhost'."},
                "command": {"type": "string", "description": "Shell command to run on the host."},
            },
            "required": ["host", "command"],
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
        "name": "browser_close",
        "description": "Close the controllable browser session.",
        "parameters": {
            "type": "object",
            "properties": {},
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
        "name": "compact_memory",
        "description": (
            "Compress the always-in-context memory or profile store when it exceeds "
            "its size limit. Archives the original and keeps a concise summary. Use when "
            "the memory or profile store is overfull or the user asks to clean it up."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "store": {
                    "type": "string",
                    "description": "Which store to compact: 'memory' (facts about the assistant) or 'profile' (facts about the user).",
                    "enum": ["memory", "profile"],
                },
            },
            "required": ["store"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file inside the project directory and return its text content. Use before editing a file to see its current contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute path inside the project directory."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Edit an existing file inside the project directory by replacing old_string "
            "with new_string. By default a numbered backup is created first; set backup=false "
            "to override for this single call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute path inside the project directory."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "backup": {"type": "boolean", "description": "Override config backup_before_edit for this call."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a file inside the project directory with the given content. "
            "If the file already exists and backups are enabled, a numbered backup is created first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute path inside the project directory."},
                "content": {"type": "string", "description": "Full file content."},
                "backup": {"type": "boolean", "description": "Override config backup_before_edit for this call."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "digest_notes",
        "description": "Convert unsaved/changed notes into training samples.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "train_adapter",
        "description": "Fine-tune the LoRA adapter on accumulated training data (incremental).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "retrain_adapter",
        "description": "Rebuild the LoRA adapter from scratch: remove old weights, re-seed baseline data, re-digest notes, and train fresh weights. Use when the user says 'retrain yourself' or after switching models.",
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
            "click) where a lightweight specialist is enough. "
            "Saved skills are also available as worker roles named skill_<slug>, "
            "e.g. skill_summarize_news."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Which worker to use, e.g. 'summarize', 'browser', or a saved skill like 'skill_summarize_news'."},
                "task": {"type": "string", "description": "The sub-task text to hand to the worker."},
            },
            "required": ["role", "task"],
        },
    },
    {
        "name": "system_check",
        "description": (
            "Run a self-diagnostic of the Symbio runtime environment. Use this "
            "when the user asks why something isn't working, before a retrain, "
            "or when you suspect a configuration or environment problem. "
            "Reports adapter status, training data, prompt files, browser "
            "availability, Ollama reachability, frontier API key, disk space, "
            "and recent log errors."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "verify_features",
        "description": (
            "Run a focused self-check on only the features the user has enabled. "
            "Auto-fixes safe failures (missing files/directories) and reports anything "
            "that needs human attention. Use this at startup hints, after config changes, "
            "or when the user asks whether everything is working."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "add_golden_case",
        "description": (
            "Add a new golden-set regression check by appending to golden_cases.json. "
            "Use when you notice a new feature or contract the model must not forget "
            "during future fine-tuning. The case will be included in pre/post-train "
            "golden checks, and the user can review it in golden_cases.json."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Short unique slug for the case, e.g. 'uses_run_remote'"},
                "description": {"type": "string", "description": "One-line summary of what this check verifies."},
                "prompt": {"type": "string", "description": "User prompt to feed the model. Use ASSISTANT_NAME and USER_NAME placeholders if needed."},
                "requirements": {
                    "type": "array",
                    "description": "List of requirements the model's reply must satisfy.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["sane_reply", "has_tool", "contains", "not_contains", "regex", "not_regex"],
                                "description": "Type of check to apply.",
                            },
                            "tool": {"type": "string", "description": "Tool name for kind='has_tool'."},
                            "text": {"type": "string", "description": "Text to search for kind='contains'/'not_contains'."},
                            "pattern": {"type": "string", "description": "Regex pattern for kind='regex'/'not_regex'."},
                        },
                        "required": ["kind"],
                    },
                },
                "ideal_reply": {"type": "string", "description": "Optional canonical reply used to generate remedy training samples if this case fails."},
            },
            "required": ["id", "description", "prompt", "requirements"],
        },
    },
]

# Hermes name -> internal name (most are already the same).
_HERMES_NAME_MAP: dict[str, str] = {
    "terminal": "run_command",
    "cmd": "run_command",
    "command": "run_command",
    "run": "run_command",
    "shell": "run_command",
    "exec": "run_command",
    "execute": "run_command",
    "search": "web_search",
    "web_search": "web_search",
    "google": "web_search",
    "read": "read_page",
    "read_page": "read_page",
    "fetch": "read_page",
    "browse": "browser_open",
    "open": "browser_open",
    "navigate": "browser_open",
    "goto": "browser_open",
    "go_to": "browser_open",
    "browser_open": "browser_open",
    "click": "browser_click",
    "browser_click": "browser_click",
    "type": "browser_type",
    "browser_type": "browser_type",
    "scroll": "browser_scroll",
    "browser_scroll": "browser_scroll",
    "press": "browser_press",
    "browser_press": "browser_press",
    "close": "browser_close",
    "browser_close": "browser_close",
    "note": "write_note",
    "save_note": "write_note",
    "write_note": "write_note",
    "remember": "write_note",
}

# Argument-name aliases: the model often emits natural argument names that
# differ from the canonical tool parameter (e.g. {"text": "Continue"} for
# browser_click which expects "target"). Map per-tool aliases so the call
# still executes without a correction round-trip.
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "run_command": {"cmd": "cmd", "command": "cmd", "shell": "cmd", "cmdline": "cmd"},
    "web_search": {"query": "query", "q": "query", "search": "query", "term": "query", "what": "query"},
    "read_page": {"url": "url", "link": "url", "page": "url", "site": "url", "address": "url"},
    "browser_open": {"url": "url", "link": "url", "page": "url", "site": "url", "address": "url", "to": "url"},
    "browser_click": {"target": "target", "text": "target", "selector": "target", "element": "target", "name": "target", "button": "target"},
    "browser_type": {"text": "text", "value": "text", "input": "text", "content": "text", "string": "text", "enter": "enter", "press_enter": "enter", "return": "enter"},
    "browser_scroll": {"direction": "direction", "dir": "direction", "way": "direction", "amount": "direction"},
    "browser_press": {"key": "key", "keys": "key", "press": "key", "button": "key"},
    "write_note": {"title": "title", "body": "body", "content": "body", "text": "body", "note": "body", "value": "body", "subject": "title"},
}


def _normalize_args(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Rename a tool call's arguments to the canonical parameter names."""
    aliases = _ARG_ALIASES.get(name)
    if not aliases:
        return params
    out: dict[str, Any] = {}
    for k, v in params.items():
        out[aliases.get(k, k)] = v
    return out

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


def _truncate_repetition(text: str) -> str:
    """Detect and truncate repetition loops in model output.

    A small quantized model at low temperature can get stuck repeating the same
    token sequence (e.g. a URL repeated dozens of times inside a tool call).
    This finds the longest suffix that is a pure repetition of a shorter prefix
    and cuts it off, keeping only the first occurrence.
    """
    # Only check reasonably long text. 60 chars catches 5+ repetitions of
    # a 12-char tag like </tool_call> while leaving short legitimate replies alone.
    if len(text) < 60:
        return text

    # The repetition may be followed by a short non-repeating suffix (e.g. a
    # closing quote+brace like '"}'). Try stripping up to 10 trailing chars
    # to reveal the repetition boundary, but only if the stripped suffix is
    # "structural" (quotes, braces, brackets, whitespace) — not prose.
    for strip_len in (0, 1, 2, 3, 4, 5, 6, 8, 10):
        if strip_len == 0:
            candidate = text
        else:
            suffix = text[-strip_len:]
            if not all(c in '\'"}])\n\r\t ' for c in suffix):
                continue
            candidate = text[:-strip_len]

        if len(candidate) < 60:
            continue

        # Strategy: look for a substring of length 8-80 chars that, when repeated
        # at least 3 times consecutively, covers the tail of the text.
        for win in range(8, min(81, len(candidate) // 3 + 1)):
            tail = candidate[-win:]
            # Count how many times this exact substring repeats at the end.
            count = 0
            pos = len(candidate)
            while pos >= win and candidate[pos - win:pos] == tail:
                count += 1
                pos -= win
            if count >= 3:
                # Found a repetition loop. Truncate to just before the repeats,
                # keeping one copy, then re-append the structural suffix.
                return candidate[:pos + win] + (text[-strip_len:] if strip_len else "")

    return text


def clean_response(text: str) -> str:
    for pattern in _THINKING_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    # The end-of-turn marker is a stop signal only; never keep it in the reply.
    text = END_TURN_RE.sub("", text)
    text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^user:\s*", "", text, flags=re.IGNORECASE)
    # Strip stray Qwen3 think delimiters that strip_reasoning_block missed —
    # it only unwraps blocks anchored at the start of the reply, so a lone
    # delimiter the adapter emits mid-answer survives it.
    text = text.replace(_QWEN_THINK_CLOSE, "")
    text = text.replace(_QWEN_THINK_OPEN, "")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Detect and truncate repetition loops: the small quantized model at low
    # temperature can get stuck repeating the same token sequence (e.g. a URL
    # repeated dozens of times inside a tool call). Find the longest suffix
    # that is a pure repetition of a shorter prefix and cut it off.
    text = _truncate_repetition(text)
    return text.strip()


# Qwen3 reasoning delimiters, built from codepoints so this file does not
# contain the literal tag characters (which confuse some tooling/parsers).
_QWEN_THINK_OPEN = "".join(chr(c) for c in [0x3c, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])
_QWEN_THINK_CLOSE = "".join(chr(c) for c in [0x3c, 0x2f, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])

# Explicit end-of-turn marker the model is taught to emit at the end of every
# reply (see tool_few_shots + the system prompt). It is used as an early stop
# sequence in _generate_reply: the moment it streams out, generation stops, so
# a weak/quantized model that forgets the real EOT token can't keep emitting
# repeated tool calls until max_reply_tokens. The \b after "end" means it only
# matches <end>, <end/>, <end > — not <endless> or <endocrine>.
END_TURN_TAG = "<end>"
END_TURN_RE = re.compile(r"<end\b[^>]*>")


def strip_reasoning_block(text: str) -> str:
    """Strip Qwen3 reasoning (think) blocks from a *generated* reply to
    recover the answer.

    A well-formed turn is one open...close block (the reasoning) followed by
    the answer. The small/quantized model with the LoRA adapter sometimes
    produces malformed output instead:

      * several leading think blocks, often empty:  O..C O..C answer
      * an empty think block, then a re-opened *unclosed* think block that
        contains the answer:  O..C O answer   (no second close)

    We strip every complete leading O..C block, then any trailing lone O
    (treating what follows it as the answer, since the model dropped the
    close). With no close delimiter at all and no leading O, the text is a
    plain answer (or pure reasoning with no answer) and is returned as-is so
    callers can decide; display/history/parse_tools then see only prose.
    """
    while True:
        s = text.lstrip("\n")
        if not s.startswith(_QWEN_THINK_OPEN):
            break
        cidx = s.find(_QWEN_THINK_CLOSE, len(_QWEN_THINK_OPEN))
        if cidx == -1:
            # Lone unclosed open delimiter: the model dropped the close, so
            # everything after this open is the answer.
            return s[len(_QWEN_THINK_OPEN):].lstrip("\n")
        # Drop this complete think block and look at what remains.
        text = s[cidx + len(_QWEN_THINK_CLOSE):]
    return text.lstrip("\n")


def tool_group(name: str) -> str | None:
    """Return the user-facing group for a tool name, or None if unknown."""
    group = _TOOL_GROUPS.get(name)
    if group is None and name.startswith("mcp_"):
        return "mcp"
    return group


def build_tools_block(groups: set[str] | None = None) -> str:
    """Return the Hermes-style <tools> JSON block for the system prompt.

    If `groups` is given, only tools whose group is in the set are included.
    The JSON is emitted compactly (no indentation) to keep prompt length down.
    """
    tools = _TOOLS
    if groups is not None:
        tools = [t for t in _TOOLS if _TOOL_GROUPS.get(t["name"]) in groups]
    return "<tools>" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")) + "</tools>"


def tool_schemas() -> list[dict[str, Any]]:
    """Return the tool registry as a list of JSON schemas."""
    return list(_TOOLS)


def refresh_mcp_tools(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Discover user-generated MCP tools on disk and register them.

    Returns the list of newly registered schemas. Safe to call repeatedly;
    existing tools are skipped. Each new schema is sanitized so a malicious
    description cannot smuggle instructions into the system prompt.
    """
    from symbio import safety as _safety
    from symbio.app import mcp_tools as _mcp_tools

    added: list[dict[str, Any]] = []
    for _mcp_schema in _mcp_tools.discover_mcp_tools():
        if any(t["name"] == _mcp_schema["name"] for t in _TOOLS):
            continue
        clean = _safety.sanitize_tool_schema(_mcp_schema, config)
        if clean is None:
            _safety.log_security_event("mcp_schema_rejected", {
                "name": _mcp_schema.get("name"),
                "reason": "injection markers in description/system_prompt",
            })
            continue
        _TOOLS.append(clean)
        _TOOL_GROUPS[clean["name"]] = "mcp"
        added.append(clean)
    return added


# A bare JSON tool call as the prompt examples teach it: a JSON object with a
# "name" (or "function") key, NOT wrapped in the XML tool-call tag. The parser
# and stripper used to only recognize the wrapped form, so a model that followed
# the prompt's bare-JSON examples had its calls silently dropped and the raw
# JSON leaked into the visible reply. These helpers recognize the bare form too.
_LT = chr(60)
_WRAPPED_TOOL_CALL_RE = re.compile(_LT + "tool_call" + r"\s*.*?\s*" + _LT + "/tool_call" + chr(62))
_BARE_TOOL_CALL_OPEN_RE = re.compile(r'\{\s*"(?:name|function)"\s*:')


def _balance_json_object(text: str, start: int) -> tuple[str | None, int]:
    """Given text[start] == '{', return (balanced_substring, end_index_exclusive)
    where end_index is just past the matching '}'. String-aware so braces inside
    JSON string values don't count. Returns (None, start) if unbalanced."""
    if start >= len(text) or text[start] != '{':
        return None, start
    depth = 0
    i = start
    in_str = False
    esc = False
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], i + 1
        i += 1
    return None, start


def _extract_bare_tool_calls(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    """Find bare (unwrapped) JSON tool-call objects in text and return
    (start, end_exclusive, parsed_dict) for each.

    A tool call is a JSON object with a 'name' or 'function' key whose value,
    after _HERMES_NAME_MAP, is a known tool name. Brace-balanced extraction
    handles nested arguments objects; string-aware so braces inside string
    values don't confuse the scan. Regions already wrapped in the XML tool-call
    tag are masked out so a wrapped call isn't double-counted. Calls inside a
    markdown code fence are skipped (the model is showing an example, not calling)."""
    masked = list(text)
    for m in _WRAPPED_TOOL_CALL_RE.finditer(text):
        for i in range(m.start(), m.end()):
            masked[i] = '\x00'
    scan = ''.join(masked)
    out: list[tuple[int, int, dict[str, Any]]] = []
    for m in _BARE_TOOL_CALL_OPEN_RE.finditer(scan):
        start = m.start()
        if text[:start].count("```") % 2 == 1:
            continue
        obj_text, end = _balance_json_object(scan, start)
        if obj_text is None:
            continue
        try:
            call = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("function")
        if not isinstance(name, str):
            continue
        resolved = _HERMES_NAME_MAP.get(name, name)
        if resolved not in _TOOL_GROUPS:
            # The small model sometimes stuffs a whole shell command into the
            # "name" field, e.g. {"name": "open -a 'Google Chrome' 'URL'"}. If
            # the name is unrecognized but looks like a command (multi-word),
            # treat it as a run_command call so the action still happens.
            stripped = name.strip()
            if " " in stripped and not stripped.startswith(("{", "[")):
                out.append((start, end, {"name": "run_command", "arguments": {"cmd": name}}))
            continue
        out.append((start, end, call))
    return out


def parse_tools(reply: str, enabled_groups: set[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Extract tool calls from the model reply.

    If `enabled_groups` is provided, drop tools whose group is disabled.
    """
    tools: list[tuple[str, dict[str, Any]]] = []

    for m in re.finditer(
        r'<note\s+title=(["\'])(.*?)\1>(.*?)</note>', reply, re.DOTALL
    ):
        tools.append(("write_note", {
            "title": m.group(2).strip(),
            "body": m.group(3).strip(),
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

    # A tag named after the tool itself: <browser_open>url</browser_open>.
    # Not the taught syntax, but the obvious guess once the model has seen
    # the tool's real name (in the <tools> catalog, a tool_response, or a
    # system observation), and it used to parse as nothing at all — the tag
    # was left in the visible reply and the turn quietly did nothing. The
    # intent is unambiguous, so honour it.
    for m in _ALIAS_TAG_RE.finditer(reply):
        tool = _ALIAS_TO_TOOL[m.group(1)]
        tools.append((tool, {_PRIMARY_ARG[tool]: m.group(2).strip()}))

    if re.search(r'<browser_close\s*/>', reply) or re.search(r'<browser_close></browser_close>', reply):
        tools.append(("browser_close", {}))

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

    # File tools (legacy tag form for quick edits).
    for m in re.finditer(r'<read_file>(.*?)</read_file>', reply, re.DOTALL):
        tools.append(("read_file", {"path": m.group(1).strip()}))

    for m in re.finditer(
        r'<edit_file\s+path=[\'"]([^\'"]*?)[\'"]\s+old_string=[\'"]([^\'"]*?)[\'"]\s+new_string=[\'"]([^\'"]*?)[\'"](?:\s+backup=[\'"]([^\'"]*?)[\'"])?\s*/?>',
        reply,
        re.DOTALL,
    ):
        backup_override = m.group(4)
        params: dict[str, Any] = {
            "path": m.group(1).strip(),
            "old_string": m.group(2),
            "new_string": m.group(3),
        }
        if backup_override is not None:
            params["backup"] = backup_override.lower() in ("true", "yes", "1", "on")
        tools.append(("edit_file", params))

    for m in re.finditer(
        r'<write_file\s+path=[\'"]([^\'"]*?)[\'"](?:\s+backup=[\'"]([^\'"]*?)[\'"])?\s*>(.*?)</write_file>',
        reply,
        re.DOTALL,
    ):
        backup_override = m.group(2)
        params = {"path": m.group(1).strip(), "content": m.group(3)}
        if backup_override is not None:
            params["backup"] = backup_override.lower() in ("true", "yes", "1", "on")
        tools.append(("write_file", params))

    if re.search(r'<digest\s*/>', reply) or re.search(r'<digest></digest>', reply):
        tools.append(("digest_notes", {}))

    if re.search(r'<train\s*/>', reply) or re.search(r'<train></train>', reply):
        tools.append(("train_adapter", {}))

    if re.search(r'<retrain\s*/>', reply) or re.search(r'<retrain></retrain>', reply):
        tools.append(("retrain_adapter", {}))

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
            tools.append((internal_name, _normalize_args(internal_name, params)))

    # Bare JSON tool calls (the form the prompt examples teach: a JSON object
    # with name/arguments, NOT wrapped in the XML tool-call tag).
    for _start, _end, call in _extract_bare_tool_calls(reply):
        name = call.get("name") or call.get("function")
        params = call.get("arguments") or call.get("parameters") or call.get("args") or {}
        if not isinstance(params, dict):
            params = {}
        if isinstance(name, str):
            internal_name = _HERMES_NAME_MAP.get(name, name)
            tools.append((internal_name, _normalize_args(internal_name, params)))

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
    r'<note\s+title=(["\'])(.*?)\1>(.*?)</note>',
    r'<cmd>(.*?)</cmd>',
    r'<py>(.*?)</py>',
    r'<search>(.*?)</search>',
    r'<read>(.*?)</read>',
    r'<browse>(.*?)</browse>',
    r'<click>(.*?)</click>',
    r'<type[^>]*>(.*?)</type>',
    r'<scroll[^>]*/>',
    r'<press>(.*?)</press>',
    r'<browser_close[^>]*/>',
    r'<browser_close>(.*?)</browser_close>',
    r'<end\s*/?>',
    r'</end>',
    r'<skill\s+name=[\'"][^\'"]*?[\'"]>(.*?)</skill>',
    r'<memory[^>]*>(.*?)</memory>',
    r'<profile[^>]*>(.*?)</profile>',
    r'<config\s+show\s*/>',
    r'<config\s+set=[\'"][^\'"]+[\'"]>(.*?)</config>',
    r'<digest\s*/>',
    r'<digest></digest>',
    r'<train\s*/>',
    r'<train></train>',
    r'<retrain\s*/>',
    r'<retrain></retrain>',
    r'<cron\s+[^>]*?>(.*?)</cron>',
    r'<delegate\s+role=[\'"][^\'"]*?[\'"]>(.*?)</delegate>',
    r'<mood>(.*?)</mood>',
    r'<tool_call>\s*.*?\s*</tool_call>',
    r'</tool_call>',
    r'<read_file[^>]*>.*?</read_file>',
    r'<edit_file[^>]*/?>',
    r'<write_file[^>]*>.*?</write_file>',
]

# Tag names recognized by the unterminated-tag cutoff below and by the
# streaming stripper's "might this become a tag" check.
_KNOWN_TAG_NAMES: tuple[str, ...] = (
    "cmd", "py", "search", "read", "browse", "click", "type", "scroll",
    "press", "browser_close", "end",
    "note", "skill", "cron", "digest", "train", "retrain", "memory", "profile",
    "config", "tool_call", "delegate",
    "read_file", "edit_file", "write_file", "mood",
)

# The single argument each tool takes when it is called as a bare
# <tool_name>value</tool_name> tag. Only tools whose one meaningful argument
# is unambiguous are listed — anything needing two (write_note, config_set)
# has no sensible bare form and is left to its own syntax.
_PRIMARY_ARG: dict[str, str] = {
    "run_command": "cmd",
    "execute_code": "code",
    "web_search": "query",
    "read_page": "url",
    "browser_open": "url",
    "browser_click": "target",
    "browser_type": "text",
    "browser_press": "key",
    "browser_scroll": "direction",
}

# Tool-name aliases usable as tags: every canonical name above plus the
# Hermes spellings that resolve to one (_HERMES_NAME_MAP does not list every
# canonical name as its own key, so seed the table with them). Short tags
# that already have their own, richer parsers are excluded — matching both
# would double-count the call and run the tool twice.
_ALIAS_TO_TOOL: dict[str, str] = {
    name: tool
    for name, tool in ({t: t for t in _PRIMARY_ARG} | {
        n: t for n, t in _HERMES_NAME_MAP.items() if t in _PRIMARY_ARG
    }).items()
    if name not in _KNOWN_TAG_NAMES
}
_ALIAS_TAG_NAMES: tuple[str, ...] = tuple(sorted(_ALIAS_TO_TOOL))
_ALIAS_TAG_RE = re.compile(
    r'<(' + '|'.join(_ALIAS_TAG_NAMES) + r')>(.*?)</\1>', re.DOTALL,
)
# Now that they execute, they must also be stripped from the visible reply.
_COMPLETE_TAG_PATTERNS.append(
    r'<(' + '|'.join(_ALIAS_TAG_NAMES) + r')>.*?</\1>'
)

# Every tag name the display/streaming layers must recognise, taught syntax
# and tool-name aliases alike.
_ALL_TAG_NAMES: tuple[str, ...] = _KNOWN_TAG_NAMES + _ALIAS_TAG_NAMES

# A reply cut off mid-tag leaves an unterminated tag; never show it.
_UNTERMINATED_TAG_RE = re.compile(
    r'<(?:' + '|'.join(_ALL_TAG_NAMES) + r')\b[^>]*>[^<]*$', re.DOTALL,
)

# Any known tag, open or close. Used to recognise text that still carries
# tool-call syntax — retrieved context in that shape teaches the model to
# emit tags where prose belongs, so callers filter on it.
_ANY_TAG_RE = re.compile(
    r'</?(?:' + '|'.join(_ALL_TAG_NAMES) + r')\b[^>]*/?>',
)


def contains_tool_tag(text: str) -> bool:
    """True if `text` still contains tool-call syntax of any known form."""
    return bool(_ANY_TAG_RE.search(text)) or bool(_extract_bare_tool_calls(text))


def _strip_complete_tag_pairs(text: str) -> str:
    for pattern in _COMPLETE_TAG_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text


def strip_tool_tags(reply: str) -> str:
    display = _strip_complete_tag_pairs(reply)
    # Drop bare JSON tool-call objects the model may emit unwrapped, so the
    # raw JSON never reaches the visible reply.
    for _start, _end, _c in sorted(_extract_bare_tool_calls(display), key=lambda s: s[0], reverse=True):
        display = display[:_start] + display[_end:]
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
    # A bare JSON tool call that opened but never balanced (truncated by
    # max_tokens) or contained invalid JSON is unusable; surface it so the
    # model can self-correct on the next round.
    for m in _BARE_TOOL_CALL_OPEN_RE.finditer(reply):
        if reply[:m.start()].count("```") % 2 == 1:
            continue
        obj, _end = _balance_json_object(reply, m.start())
        if obj is None:
            return f"A JSON tool call was left incomplete and could not be used: {reply[m.start():m.start()+120]!r}"
        try:
            json.loads(obj)
        except json.JSONDecodeError as e:
            return f"A JSON tool call contained invalid JSON and could not be used: {e}"
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
        # Qwen3 think block. With enable_thinking=True the generation prompt
        # ends at "<|im_start|>assistant\n" — the open delimiter is NOT part
        # of the prefix. So a reasoning reply *generates* its own
        # open...close block and a non-reasoning reply starts straight into
        # the answer (and with enable_thinking=False the template pre-closes
        # an empty block, so nothing is generated either).
        #
        # That means the block has to be detected, not assumed: hold text
        # back only once the stream actually opens with the delimiter (the
        # same rule strip_reasoning_block applies to the finished text). A
        # stream that never opens one — a direct answer, or a second-stage
        # stripper such as Telegram's, which is fed text the chat stripper
        # already de-thought — streams normally instead of being swallowed
        # whole while waiting for a close delimiter that never comes.
        self._think_state = "undecided"  # -> "inside" -> "done"
        # Leading whitespace (newlines left after the think block, or after a
        # re-opened delim the adapter drops) is discarded until the first real
        # answer character is shown, so the reply starts flush on its line.
        self._answer_started = False

    def feed(self, chunk: str) -> str:
        """Add newly generated text; return the text now safe to display."""
        self._buffer += chunk
        if self._think_state == "undecided":
            head = self._buffer.lstrip("\n")
            if head.startswith(_QWEN_THINK_OPEN):
                # A real reasoning block is starting — hide it from here.
                self._think_state = "inside"
                self._buffer = head[len(_QWEN_THINK_OPEN):]
            elif _QWEN_THINK_OPEN.startswith(head):
                # Still an unresolved prefix of the open delimiter ("<thi"),
                # which streams a character at a time — wait before showing
                # anything, so the delimiter never flashes on screen.
                return ""
            else:
                # This reply has no reasoning block; it is answer text.
                self._think_state = "done"
        if self._think_state == "inside":
            cidx = self._buffer.find(_QWEN_THINK_CLOSE)
            if cidx == -1:
                # Still inside the reasoning block — show nothing yet.
                # finish() drops the leftover once generation ends.
                return ""
            # Drop the reasoning block and everything up through the close
            # delimiter; what remains is the answer.
            self._buffer = self._buffer[cidx + len(_QWEN_THINK_CLOSE):]
            self._think_state = "done"
            # The answer may still carry think-delimiter artifacts (the
            # adapter sometimes emits an empty think block, then re-opens an
            # unclosed one around the answer). Both delimiters are fixed
            # 7-char markers that never appear in normal prose, so drop every
            # remaining one, then strip the leading newlines once at the
            # answer start.
            self._buffer = self._buffer.replace(_QWEN_THINK_OPEN, "").replace(_QWEN_THINK_CLOSE, "").lstrip("\n")
            if self._buffer == "":
                return ""
        else:
            # Drop stray think delimiters emitted mid-answer (re-opened or
            # empty blocks the adapter appends after the answer started).
            self._buffer = self._buffer.replace(_QWEN_THINK_OPEN, "").replace(_QWEN_THINK_CLOSE, "")
        self._buffer = _strip_complete_tag_pairs(self._buffer)
        # Drop complete bare-JSON tool calls so their raw JSON never flashes.
        for _start, _end, _c in sorted(_extract_bare_tool_calls(self._buffer), key=lambda s: s[0], reverse=True):
            self._buffer = self._buffer[:_start] + self._buffer[_end:]
        candidates = [c for c in (self._first_ambiguous_lt(), self._first_ambiguous_bare_json()) if c != -1]
        cut = min(candidates) if candidates else -1
        if cut == -1:
            safe, self._buffer = self._buffer, ""
        else:
            safe, self._buffer = self._buffer[:cut], self._buffer[cut:]
        if not self._answer_started:
            # Drop leading whitespace (newlines left after the think block or
            # a re-opened delim) until the first real answer character shows.
            stripped = safe.lstrip()
            if stripped:
                self._answer_started = True
            return stripped
        return safe

    def finish(self) -> str:
        """Call once generation ends; returns any remaining safe text —
        plain prose with a stray '<' that never became a tag, or a
        genuinely truncated tag, either way handled by strip_tool_tags.
        If the reply opened a think block whose close delimiter never
        arrived, the buffer is still reasoning and is discarded (the
        authoritative reply is computed separately via
        strip_reasoning_block, and the caller falls back to printing the
        consolidated reply, so dropping it here only affects the live
        display tail)."""
        if self._think_state == "inside":
            # Pure reasoning, no answer yet — never show it.
            self._buffer = ""
            return ""
        # "undecided" here means the whole reply was a short prefix of the
        # open delimiter ("<th") that never resolved — that is prose, not
        # reasoning, so fall through and flush it.
        self._think_state = "done"
        remaining = strip_tool_tags(
            self._buffer.replace(_QWEN_THINK_OPEN, "").replace(_QWEN_THINK_CLOSE, ""))
        self._buffer = ""
        if not self._answer_started:
            return remaining.lstrip()
        return remaining

    def _first_ambiguous_lt(self) -> int:
        """Index of a '<' that might still be starting a known tag and
        can't be ruled out yet, or -1 if the buffer is unambiguously safe
        to show as-is (no '<', or every '<' has already diverged from
        every known tag name — e.g. "x < 5" is never held back).

        Also holds a '<' that could be starting the Qwen3 think-open
        delimiter: after the reasoning block the adapter sometimes re-opens
        an (unclosed) think block, and that delimiter can arrive split
        across chunks ('<' then 'think>'). Holding the partial lets it
        complete, after which feed()'s delimiter-strip removes it so the
        raw marker never flashes on screen."""
        for m in re.finditer('<', self._buffer):
            tail = self._buffer[m.start() + 1:]
            if tail == "" or any(
                name.startswith(tail) or tail.startswith(name)
                for name in _KNOWN_TAG_NAMES
            ) or "think".startswith(tail) or tail.startswith("think"):
                return m.start()
        return -1

    def _first_ambiguous_bare_json(self) -> int:
        """Index of a '{' that might still be starting a bare JSON tool call
        and hasn't balanced yet, or -1 if none.

        Unlike an XML '<' (which is ambiguous the instant it appears), a bare
        JSON call only becomes recognizable once several chars of the opener
        have arrived. Holding back from the leading '{' (when it could be the
        start of a '"name"'/'"function"' key) lets the opener accumulate so the
        complete call can be dropped instead of flashing token-by-token."""
        buf = self._buffer
        i = 0
        n = len(buf)
        while i < n:
            if buf[i] != '{':
                i += 1
                continue
            obj, end = _balance_json_object(buf, i)
            if obj is not None:
                # Balanced object: not ambiguous. A real tool call was already
                # removed by _extract_bare_tool_calls in feed(); skip past it.
                i = end
                continue
            tail = buf[i + 1:].lstrip()
            # A tool-call opener is '{"name"'/'"function"' (optionally spaced),
            # so a '{' that may be starting one is followed by nothing or a '"'.
            if tail == "" or tail.startswith('"'):
                return i
            i += 1
        return -1
