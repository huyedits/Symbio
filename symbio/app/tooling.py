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
    "fetch_html": "browser",
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
        "description": "Run a Python script in the sandbox. Print what you want back — a value is only visible if the script prints it. Beyond the standard library you may `from symbio_tools import read_file, write_file, patch, list_dir, search_files, fetch, select` — read_file/write_file/patch/list_dir/search_files work on project paths, fetch(url) returns a URL's raw body, and select(html, css) returns the matching elements as dicts with 'text' and 'attrs' — use it rather than parsing HTML yourself. requests, selectolax and numpy are installed. os, sys, subprocess, pathlib and urllib are not importable; use symbio_tools instead.",
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
        "name": "fetch_html",
        "description": "Fetch a URL and return its raw HTML with tags and attributes intact. Use this when you need to parse a page by selector (data-testid, class, id) rather than just read its text; read_page strips all markup.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
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
            "Hand a bounded sub-task off to a specialist worker instead of "
            "doing it yourself — use for narrow, repetitive decisions (e.g. "
            "summarizing a page, picking the next browser click), and for any "
            "saved skill, whose procedure lives in that worker's weights. "
            "Call refresh_delegate_roles() to list the workers that exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                # Both the description and the enum are rewritten from the real
                # catalog by refresh_delegate_roles(); this is only the fallback
                # wording for a session where no catalog could be read.
                "role": {"type": "string", "description": "Which worker to use, e.g. 'summarize' or 'browser'."},
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
    "fetch_html": "fetch_html",
    "get_html": "fetch_html",
    "raw_html": "fetch_html",
    "html": "fetch_html",
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

# Prefix used to surface a Qwen3 thinking block to the user (StreamingStripper
# during streaming, chat.py's final print when not streaming). Plain ASCII so
# it survives any terminal/tooling HTML transformation.
REASONING_MARKER = "  [Reasoning] "

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
    stripped_a_block = False
    while True:
        s = text.lstrip("\n")
        if not s.startswith(_QWEN_THINK_OPEN):
            break
        stripped_a_block = True
        cidx = s.find(_QWEN_THINK_CLOSE, len(_QWEN_THINK_OPEN))
        if cidx == -1:
            # Lone unclosed open delimiter: the model dropped the close, so
            # everything after this open is the answer.
            return s[len(_QWEN_THINK_OPEN):].lstrip("\n")
        # Drop this complete think block and look at what remains.
        text = s[cidx + len(_QWEN_THINK_CLOSE):]

    # A close with no open before it. Not malformed output — it is what the
    # template produces whenever thinking is on: enable_thinking=True ends the
    # prompt with a bare open delimiter, so the block is already open when
    # generation starts and the model only ever emits the close. Everything up
    # to it is reasoning. Without this the whole monologue was printed to the
    # user ahead of the answer on every round after the first.
    #
    # Only when the reply carried no open of its own. Having already consumed
    # a block proves the model writes its own delimiters, so a later stray
    # close is junk inside the answer — treating it as a reasoning boundary
    # there discards real content, which a test caught it doing.
    s = text.lstrip("\n")
    if not stripped_a_block:
        cidx = s.find(_QWEN_THINK_CLOSE)
        if cidx != -1 and _QWEN_THINK_OPEN not in s[:cidx]:
            return s[cidx + len(_QWEN_THINK_CLOSE):].lstrip("\n")
    return s


def extract_reasoning(text: str) -> str:
    """Return the Qwen3 thinking block content from a generated reply, or "".

    Mirrors strip_reasoning_block's detection: only a block anchored at the
    start of the reply counts. The content is the text between the open and
    close delimiters, stripped. Used to surface the reasoning to the user
    (the reply itself still goes through strip_reasoning_block so tools and
    history never see it).
    """
    s = text.lstrip("\n")
    if not s.startswith(_QWEN_THINK_OPEN):
        # Mirror strip_reasoning_block's prompt-opened case: with thinking on,
        # the open delimiter is in the prompt and never in the reply, so the
        # reasoning is simply everything before the first close.
        cidx = s.find(_QWEN_THINK_CLOSE)
        if cidx != -1 and _QWEN_THINK_OPEN not in s[:cidx]:
            return s[:cidx].strip()
        return ""
    cidx = s.find(_QWEN_THINK_CLOSE, len(_QWEN_THINK_OPEN))
    if cidx == -1:
        return ""
    return s[len(_QWEN_THINK_OPEN):cidx].strip()


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


def refresh_delegate_roles() -> list[str]:
    """Point delegate_task at the workers that actually exist.

    The schema used to name a worked example — "a saved skill like
    skill_summarize_news" — and never list the real catalog, so the model had
    to guess a slug to route anything. Worse, the example was wrong: a skill is
    catalogued under the key `skill_<slug>` but its `role` is the bare slug, so
    a model that followed the description emitted a role the dispatcher could
    never resolve. Skills were effectively unreachable by delegation.

    Returns the role names now advertised.
    """
    from symbio.app import dispatch

    roles: list[str] = []
    described: list[str] = []
    try:
        catalog = dispatch.load_catalog()
    except Exception:
        return []
    for entry in catalog.values():
        role = entry.get("role")
        if not role or role in roles:
            continue
        roles.append(role)
        label = entry.get("skill_name") or entry.get("description") or role
        described.append(f"'{role}' ({label})")
    if not roles:
        return []

    for tool in _TOOLS:
        if tool["name"] != "delegate_task":
            continue
        role_schema = tool["parameters"]["properties"]["role"]
        # An enum is the strongest signal available in a JSON schema that the
        # value is drawn from a fixed set, and it is what stops the model
        # inventing a plausible-looking slug.
        role_schema["enum"] = list(roles)
        role_schema["description"] = (
            "Which worker to use. Must be one of: " + ", ".join(described) + "."
        )
    return roles


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


# Canonical tool names only, longest first so `list_cron_jobs` wins over a
# shorter name that prefixes it.
#
# Deliberately NOT including _HERMES_NAME_MAP's short aliases. Those contain
# bare words like "type", "read" and "open", and the dedicated XML handlers
# above already parse <type enter="true">...</type> — matching it here too
# would emit the same call twice. write_note is excluded for the same reason:
# the self-closing <note/>|<write_note/> handler already covers it.
_CALLABLE_NAMES = sorted(
    (n for n in _TOOL_GROUPS if n != "write_note"), key=len, reverse=True)
_NAME_ALT = "|".join(re.escape(n) for n in _CALLABLE_NAMES)

# Improvised function-call syntax, with arguments.
#
# 25 of the 27 tools declared in the <tools> block have no worked example
# anywhere in the assembled system prompt — only compact_memory and config_set
# do. A tool the model has seen declared but never seen *called* gets a call
# shape invented for it, and the invented shape is consistently a dotted or
# parenthesised function with keyword attributes:
#
#     .schedule_job schedule="0 9 * * *" text="stretch"
#     schedule_job(schedule="0 9 * * *", text="stretch")
#     <schedule_job schedule="0 9 * * *" text="stretch" />
#
# None of these matched any pattern, so the tool silently never ran and the
# raw text was printed to the user as the reply — a success-looking line for
# a job that was never created. That is the same failure the self-closing
# <note/> handler above was written for; this generalises it to every tool
# instead of fixing them one at a time as each is caught in the wild.
#
# Anchored on the exact declared names, so prose cannot trip it.
_FUNC_ATTR_RE = re.compile(
    r'[.<]?\b(' + _NAME_ALT + r')\b\s*\(?\s*'
    r'((?:\w+\s*=\s*(?:"[^"]*"|\'[^\']*\')\s*,?\s*)+)\)?\s*/?>?'
)

# The same, with no arguments: `.list_cron_jobs` or `list_cron_jobs()`. The
# dot or the parentheses are required — a bare tool name is far too common in
# ordinary prose ("I'll delegate_task to the worker") to treat as a call.
_FUNC_NOARG_RE = re.compile(
    r'(?:\.(' + _NAME_ALT + r')\b(?!\s*\()|\b(' + _NAME_ALT + r')\s*\(\s*\))'
)

_ATTR_PAIR_RE = re.compile(r'(\w+)\s*=\s*(["\'])(.*?)\2', re.DOTALL)

# The same improvisation with a colon instead of an equals sign, which is what
# a model writes when it is copying the tool *schema* rather than an example:
# `browser_scroll(direction: "down")`.
_FUNC_COLON_RE = re.compile(
    r'[.<]?\b(' + _NAME_ALT + r')\b\s*\(\s*'
    r'((?:\w+\s*:\s*(?:"[^"]*"|\'[^\']*\')\s*,?\s*)+)\)')
_COLON_PAIR_RE = re.compile(r'(\w+)\s*:\s*(["\'])(.*?)\2', re.DOTALL)

# A single positional argument: `web_search("weather in Tokyo")`. Only safe for
# the tools in _PRIMARY_ARG, whose one meaningful argument is unambiguous —
# anywhere else there is no way to know which parameter was meant.
_FUNC_POSITIONAL_RE = re.compile(
    r'[.<]?\b(' + _NAME_ALT + r')\s*\(\s*(["\'])(.*?)\2\s*\)', re.DOTALL)


def unparsed_tool_tags(reply: str) -> list[str]:
    """Tool names the reply used as a tag that this module does not parse.

    22 of the declared tools have no `<name>arg</name>` form — the alias table
    is derived from _PRIMARY_ARG, which only covers single-argument tools, and
    the rest are reached through their own richer syntax (<note title=...>,
    <config set=...>, <digest />). When the model reaches for the plain form
    anyway — which it does, having seen the tool's name in the <tools> catalog
    — the tag matches nothing, gets stripped from the display, and the turn
    quietly does nothing at all. No error, no retry, nothing for the model to
    learn from. Seen live 2026-08-24: <fetch_html>...</fetch_html> was printed
    as the entire visible reply and no tool ran.

    A malformed <tool_call> is NOT this case: it leaves a dangling tag, which
    the loop already detects and resamples on. This is the silent one.
    """
    used = {m.group(1) for m in re.finditer(r"<([a-z_][a-z0-9_]*)\s*>", reply, re.I)}
    return sorted(
        name for name in used
        if name in _TOOL_GROUPS and not parse_tools(f"<{name}>x</{name}>")
    )


def _unwrap_primary_arg(arg_name: str, raw: str) -> str:
    """Strip a parameter name the model wrote *inside* the tag body.

    An alias tag carries one value, so the name is redundant — but the model
    writes it anyway, having seen the tool's JSON schema. Live 2026-08-24:
    <read_file>path=\'symbio/app/chat.py\'</read_file> arrived as
    {"path": "path=\'symbio/app/chat.py\'"}, which resolved to a file of that
    literal name, and the user was told their file did not exist.

    Only the tool's OWN argument name is stripped, so a value that legitimately
    contains "=" (a URL query string, an env assignment in a command) is left
    exactly as written.
    """
    text = raw.strip()
    prefix = arg_name + "="
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _in_code_fence(text: str, index: int) -> bool:
    """True when `index` falls inside a markdown code fence — the model is
    showing an example there, not calling anything."""
    return text[:index].count("```") % 2 == 1


def _find_function_attr_calls(
        reply: str) -> list[tuple[int, int, str, dict[str, Any]]]:
    """Recognise the improvised function form for any declared tool.

    Returns (start, end, name, params) so callers can both execute the call and
    cut it out of the visible reply — a recovered call that still printed its
    raw tag to the user would be a worse bug than not recovering it.

    Regions already wrapped in the XML tool-call tag are masked out so a
    well-formed call is not also counted here.
    """
    masked = list(reply)
    for m in _WRAPPED_TOOL_CALL_RE.finditer(reply):
        for i in range(m.start(), m.end()):
            masked[i] = "\x00"
    scan = "".join(masked)

    out: list[tuple[int, int, str, dict[str, Any]]] = []
    seen: set[tuple[int, int]] = set()

    for m in _FUNC_ATTR_RE.finditer(scan):
        if _in_code_fence(reply, m.start()):
            continue
        name = _HERMES_NAME_MAP.get(m.group(1), m.group(1))
        if name not in _TOOL_GROUPS:
            continue
        params = {k.lower(): v for k, _q, v in _ATTR_PAIR_RE.findall(m.group(2))}
        if not params:
            continue
        seen.add((m.start(), m.end()))
        out.append((m.start(), m.end(), name, _normalize_args(name, params)))

    for m in _FUNC_COLON_RE.finditer(scan):
        if _in_code_fence(reply, m.start()):
            continue
        if any(s <= m.start() < e for s, e in seen):
            continue
        name = _HERMES_NAME_MAP.get(m.group(1), m.group(1))
        if name not in _TOOL_GROUPS:
            continue
        params = {k.lower(): v for k, _q, v in _COLON_PAIR_RE.findall(m.group(2))}
        if not params:
            continue
        seen.add((m.start(), m.end()))
        out.append((m.start(), m.end(), name, _normalize_args(name, params)))

    for m in _FUNC_POSITIONAL_RE.finditer(scan):
        if _in_code_fence(reply, m.start()):
            continue
        if any(s <= m.start() < e for s, e in seen):
            continue
        name = _HERMES_NAME_MAP.get(m.group(1), m.group(1))
        if name not in _TOOL_GROUPS or name not in _PRIMARY_ARG:
            continue
        seen.add((m.start(), m.end()))
        out.append((m.start(), m.end(), name,
                    _normalize_args(name, {_PRIMARY_ARG[name]: m.group(3)})))

    for m in _FUNC_NOARG_RE.finditer(scan):
        if _in_code_fence(reply, m.start()):
            continue
        if any(s <= m.start() < e for s, e in seen):
            continue
        raw = m.group(1) or m.group(2)
        name = _HERMES_NAME_MAP.get(raw, raw)
        if name not in _TOOL_GROUPS:
            continue
        out.append((m.start(), m.end(), name, {}))

    return out


def _extract_function_attr_calls(reply: str) -> list[tuple[str, dict[str, Any]]]:
    return [(n, p) for _s, _e, n, p in _find_function_attr_calls(reply)]


# One key, its value running to the delimiter that ends it: either a comma
# before the next "key": , or the closing brace of the arguments object.
_LOOSE_FIELD_RE = re.compile(
    r'"(?P<key>\w+)"\s*:\s*"(?P<val>.*?)"\s*(?=,\s*"\w+"\s*:|\s*\}\s*\}?\s*$)',
    re.DOTALL)


def _repair_tool_call_json(raw: str) -> dict[str, Any] | None:
    """Recover a tool call whose JSON the model failed to escape.

    Writing code through a JSON string is the one thing this model reliably
    gets wrong. Asked to insert a database row it produced:

        {"name": "execute_code", "arguments": {"code": "import sqlite3
        ...cur.execute("INSERT INTO users ...")"}}

    — a raw newline and an unescaped double quote inside a JSON string. The
    object does not parse, so the call vanished completely and the turn did
    nothing, silently. Any code containing a quote hits this, which is most
    code worth running.

    Rather than guess at arbitrary broken JSON, this pulls out the name and
    then each "key": "value" pair by scanning to the delimiter that ends it,
    taking the value verbatim. Returns None when the shape is not recognisable
    — a wrong repair would be worse than no call.
    """
    name_m = re.search(r'"(?:name|function)"\s*:\s*"([^"]+)"', raw)
    if not name_m:
        return None

    args_m = re.search(r'"(?:arguments|parameters|args)"\s*:\s*\{(.*)\}',
                       raw, re.DOTALL)
    params: dict[str, Any] = {}
    if args_m:
        body = args_m.group(1)
        for f in _LOOSE_FIELD_RE.finditer(body):
            value = f.group("val")
            # The model escaped some of it correctly; honour what it did.
            value = (value.replace("\\n", "\n").replace("\\t", "\t")
                          .replace('\\"', '"').replace("\\\\", "\\"))
            params[f.group("key")] = value
        # Numbers and booleans, which need no quote handling.
        for f in re.finditer(r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?|true|false)\b', body):
            key, lit = f.group(1), f.group(2)
            if key in params:
                continue
            params[key] = (True if lit == "true" else False if lit == "false"
                           else float(lit) if "." in lit else int(lit))

    return {"name": name_m.group(1), "arguments": params}


# Gemma 4's own tool-call dialect, which it falls back to whenever the prompt's
# examples do not fully take:
#
#     <|tool_call>call:read_file{path:<|"|>notes/todo.md<|"|>}<tool_call|>
#     <|tool_call>call:{"name": "list_cron_jobs", "arguments": {}}<tool_call|>
#
# That is not sloppiness — it is the syntax its own chat template teaches, down
# to the <|"|> string delimiters emitted by the template's format_parameters
# macro. Unrecognised, read_file failed 3/3 in the tool battery while choosing
# the right tool and the right path every time.
_GEMMA_TOOL_CALL_RE = re.compile(
    r'<\|tool_call>\s*(.*?)\s*<tool_call\|>', re.DOTALL)
# Channel markers leak into the call body ("call:<|channel>::web_search{...}"),
# as does the `call:` lead-in itself.
_GEMMA_NOISE_RE = re.compile(r'<\|channel>|<channel\|>|^\s*call\s*:|^[\s:/]+')
_GEMMA_QUOTE = '<|"|>'
# key:<|"|>value<|"|>  or  key:"value"  or  "key":"value"
_GEMMA_STR_ARG_RE = re.compile(
    r'"?(\w+)"?\s*:\s*(?:' + re.escape(_GEMMA_QUOTE) + r'(.*?)'
    + re.escape(_GEMMA_QUOTE) + r'|"(.*?)"|\'(.*?)\')', re.DOTALL)
_GEMMA_LIT_ARG_RE = re.compile(
    r'"?(\w+)"?\s*:\s*(-?\d+(?:\.\d+)?|true|false)\b')


def _extract_gemma_tool_calls(reply: str) -> list[tuple[str, dict[str, Any]]]:
    """Recognise Gemma 4's native <|tool_call>…<tool_call|> form."""
    out: list[tuple[str, dict[str, Any]]] = []
    for m in _GEMMA_TOOL_CALL_RE.finditer(reply):
        if _in_code_fence(reply, m.start()):
            continue
        body = m.group(1)
        # Strip the channel/lead-in noise repeatedly: it arrives stacked.
        prev = None
        while prev != body:
            prev = body
            body = _GEMMA_NOISE_RE.sub("", body).strip()

        if body.startswith("{"):
            # It reached for the taught JSON shape inside its own wrapper.
            try:
                call = json.loads(body)
            except json.JSONDecodeError:
                call = _repair_tool_call_json(body)
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("function")
            params = (call.get("arguments") or call.get("parameters")
                      or call.get("args") or {})
            if not isinstance(params, dict):
                params = {}
        else:
            head = re.match(r'([\w.]+)\s*\{(.*)\}\s*$', body, re.DOTALL)
            if not head:
                continue
            name = head.group(1).rsplit(".", 1)[-1]
            args_body = head.group(2)
            params = {}
            for a in _GEMMA_STR_ARG_RE.finditer(args_body):
                value = next(g for g in a.groups()[1:] if g is not None)
                params[a.group(1)] = value
            for a in _GEMMA_LIT_ARG_RE.finditer(args_body):
                key, lit = a.group(1), a.group(2)
                if key in params:
                    continue
                params[key] = (True if lit == "true" else False if lit == "false"
                               else float(lit) if "." in lit else int(lit))

        if not isinstance(name, str):
            continue
        internal = _HERMES_NAME_MAP.get(name, name)
        if internal not in _TOOL_GROUPS:
            continue
        out.append((internal, _normalize_args(internal, params)))
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

    # Self-closing attribute form: <write_note title="..." body="..." />.
    # The model writes this when it treats the tag as XML rather than as a
    # wrapper around the body. Nothing recognised it, so the note was never
    # written AND the raw tag was printed to the user as the reply — observed
    # verbatim in a session where a correction was being taught, which also
    # meant nothing was learned from that turn.
    for m in re.finditer(r'<(?:note|write_note)\b([^>]*?)/>', reply, re.DOTALL):
        attrs = {k.lower(): v for k, _q, v in
                 re.findall(r'(\w+)\s*=\s*(["\'])(.*?)\2', m.group(1), re.DOTALL)}
        body = next((attrs[k] for k in
                     ("body", "content", "text", "note", "value") if attrs.get(k)), "")
        if body.strip():
            tools.append(("write_note", {
                "title": attrs.get("title", "").strip(),
                "body": body.strip(),
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
        arg = _PRIMARY_ARG[tool]
        tools.append((tool, {arg: _unwrap_primary_arg(arg, m.group(2))}))

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
        tools.append(("read_file", {"path": _unwrap_primary_arg("path", m.group(1))}))

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
            call = _repair_tool_call_json(m.group(1).strip())
            if call is None:
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

    # Gemma 4's native wrapper. An explicit call, not an improvisation, so it
    # is read alongside the other supported syntaxes rather than as a fallback.
    # When it wraps a well-formed JSON call the bare-JSON scan above has
    # already found the same one, so drop the repeat — a duplicate here does
    # not read as noise, it runs the tool twice.
    for call in _extract_gemma_tool_calls(reply):
        if call not in tools:
            tools.append(call)

    # Improvised function form, for the tools the prompt declares but never
    # demonstrates. Last, so a well-formed call in any supported syntax is
    # already in `tools` and this only ever adds what nothing else caught.
    if not tools:
        tools.extend(_extract_function_attr_calls(reply))

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
    # Executed above, so it must be stripped here too or the raw tag is what
    # the user sees as the reply.
    r'<(?:note|write_note)\b[^>]*?/>',
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
    # Gemma 4's native call wrapper and its thought channel. Executed above, so
    # they must be stripped too — a recovered call that still prints its raw
    # tag to the user is the bug the recovery was meant to fix.
    r'<\|tool_call>(.*?)<tool_call\|>',
    r'<\|channel>thought(.*?)<channel\|>',
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
    # Registering the primary arg is what makes <fetch_html>URL</fetch_html>
    # parse, via _ALIAS_TO_TOOL below. Without it the model emitted exactly
    # that tag — the obvious guess once it has seen the tool's name in the
    # catalog — and it matched nothing, so the tag was printed as the visible
    # reply and the turn silently did nothing. Observed live 2026-08-24.
    "fetch_html": "url",
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
    # And the improvised function forms parse_tools now executes. Recovering
    # the call but still printing its raw tag is worse than not recovering it:
    # observed live as
    #     Caine: >tag
    #     <schedule_job schedule="0 9 * * *" text="stretch"/>
    # where the job WAS created and the user was shown the markup anyway.
    for _start, _end, _n, _p in sorted(
            _find_function_attr_calls(display), key=lambda s: s[0], reverse=True):
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

    def __init__(self, show_reasoning: bool = True):
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
        # When True, a completed thinking block is surfaced to the user as a
        # "[Reasoning] …" block (the answer still streams separately after
        # it). When False, the block is hidden exactly as before.
        self._show_reasoning = show_reasoning
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
            # The reasoning block is complete. If showing reasoning, emit it as
            # a distinct "[Reasoning] …" block and hold the answer buffer for
            # the next feed call (or finish()) so the answer streams
            # separately after it — the caller can then attach its reply
            # prefix to the answer, not the reasoning. Otherwise drop the
            # block as before.
            reasoning = self._buffer[:cidx].strip()
            self._buffer = self._buffer[cidx + len(_QWEN_THINK_CLOSE):]
            self._think_state = "done"
            # The answer may still carry think-delimiter artifacts (the
            # adapter sometimes emits an empty think block, then re-opens an
            # unclosed one around the answer). Both delimiters are fixed
            # 7-char markers that never appear in normal prose, so drop every
            # remaining one, then strip the leading newlines once at the
            # answer start.
            self._buffer = self._buffer.replace(_QWEN_THINK_OPEN, "").replace(_QWEN_THINK_CLOSE, "").lstrip("\n")
            if self._show_reasoning and reasoning:
                return f"{REASONING_MARKER}{reasoning}\n"
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
        self._buffer = self._buffer.replace(_QWEN_THINK_OPEN, "").replace(
            _QWEN_THINK_CLOSE, "")
        # Whatever _first_ambiguous_lt is still holding never completed:
        # generation ended mid-tag, or the reply was nothing but tool tags and
        # this is the leftover '<'. That is truncated syntax, not prose.
        # strip_tool_tags below only removes tags it can recognize, so a lone
        # '<' survives it and gets printed as the entire reply — seen live as
        # "Caine   : <" on a reply that was 25 <run_command/> tags. Cut the
        # fragment here instead. Prose is unaffected: a '<' is only held when
        # it is at the very end of the buffer or already matches a tag name,
        # so "x < 5" was never in the buffer to begin with.
        cut = self._first_ambiguous_lt()
        if cut != -1:
            self._buffer = self._buffer[:cut]
        remaining = strip_tool_tags(self._buffer)
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
