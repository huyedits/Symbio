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
import threading
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
    desktop_click_in_image,
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
    from symbio.app.agent import AIAgent


logger = logging.getLogger("chat")


def build_tool_registry(agent: AIAgent) -> list[dict[str, Any]]:
    """Return the full Hermes-style tool registry for an agent instance."""
    return [
        {
            # The read side of memory. The catalog in symbio/app/tooling.py
            # advertises this to every loop, and only the chat dispatcher
            # could run it -- so on this one the model called the name its own
            # prompt had offered and was told the tool does not exist.
            "name": "recall",
            "description": "Look up what you have already saved: your notes, your durable memory, the profile of your user, and past sessions. Use it before saying you do not know something about the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in the user's own words."},
                    "scope": {"type": "string", "description": "'memory' (the default), 'sessions', or 'all'."},
                },
                "required": ["query"],
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_recall(a, params),
        },
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
            "description": "Look at the current browser page: saves a screenshot AND returns a description of what is on screen with the pixel coordinates of every clickable element. Use it whenever you are unsure what state the page is in or where something is, rather than guessing from page text.",
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
            "description": "Look at the whole screen: saves a screenshot AND returns a description of what is on it with the pixel coordinates of every clickable element, which desktop_click can use directly.",
            "parameters": {"type": "object", "properties": {}},
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop_screenshot(a, params),
        },
        {
            "name": "desktop_click",
            "description": "Click a control on screen. Prefer 'element': see_screen numbers every control the frontmost window publishes, and a number presses the real control instead of a guessed point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "integer"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "clicks": {"type": "integer"},
                    "button": {"type": "string"},
                },
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop(a, "desktop_click", params),
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
            "run": lambda params, a=agent: _tool_desktop(a, "desktop_move", params),
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
            "run": lambda params, a=agent: _tool_desktop(a, "desktop_type", params),
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
            "run": lambda params, a=agent: _tool_desktop(a, "desktop_press", params),
        },
        {
            "name": "see_screen",
            "description": "Look at the frontmost window and get back every control it publishes — role, label and exact frame — each with a number that desktop_click and desktop_type take. Falls back to a screenshot and the vision model only for a window that draws its own interface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "question": {"type": "string"},
                },
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_desktop(a, "see_screen", params),
        },
        {
            "name": "desktop_scroll",
            "description": "Scroll the window under the pointer, or over a numbered element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string"},
                    "amount": {"type": "integer"},
                    "element": {"type": "integer"},
                },
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop(a, "desktop_scroll", params),
        },
        {
            "name": "desktop_drag",
            "description": "Press at one point, move, and release at another. Give element numbers or raw coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_element": {"type": "integer"},
                    "to_element": {"type": "integer"},
                    "from_x": {"type": "integer"},
                    "from_y": {"type": "integer"},
                    "to_x": {"type": "integer"},
                    "to_y": {"type": "integer"},
                },
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop(a, "desktop_drag", params),
        },
        {
            "name": "desktop_wait",
            "description": "Wait for the screen to catch up, up to 10 seconds.",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "number"}},
            },
            "readonly": True,
            "run": lambda params, a=agent: _tool_desktop(a, "desktop_wait", params),
        },
        {
            "name": "open_app",
            "description": "Launch a macOS application by name, or bring it to the front.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            "readonly": False,
            "run": lambda params, a=agent: _tool_desktop(a, "open_app", params),
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


# The families a worked-example block can be rotated to. Same names the tool
# index uses (symbio/app/tool_docs.py), so "the family the model last worked
# in" and "the family it can ask for schemas about" are one vocabulary.
FEW_SHOT_FAMILIES: tuple[str, ...] = (
    "file", "code", "shell", "web", "browser", "desktop", "memory",
)


def tool_few_shots(config: dict[str, Any],
                   family: str | None = None) -> list[dict[str, str]]:
    """Minimal tool-use examples in Hermes JSON-in-<tool_call> format.

    The examples MUST match the format the runtime actually parses: parse_tools
    (symbio/utils.py) and the registry (build_tool_registry) accept JSON inside
    <tool_call> — never legacy short tags like <browse>/<click>/<search>. The
    system prompt (app/prompts.py) teaches the same JSON format, so the few-shots
    reinforce it instead of contradicting it. Keep every emitted tool name and
    argument key aligned with the registry schema so a parsed call resolves —
    test_prompt_tool_names.py fails the build if one does not, in EITHER stack.

    `family` rotates the block: the universals plus that family's examples,
    instead of the fixed set. Without one — the first turn of a session, or a
    conversation that has used no tools yet — the full set comes back, which is
    exactly what this always returned.

    Rotating buys breadth. The fixed block was four browser examples, a search
    and a note, so a file or code request was answered by a model that had just
    been shown six ways to drive a page.

    WHO chooses the family matters as much as the rotation. The obvious
    implementation is a keyword table over the user's message — and it is the
    wrong one: a bag of words deciding what a sentence is about, in front of a
    model whose entire job is understanding sentences, gets "read config.json"
    and "read the news" wrong in opposite directions and shows the model the
    wrong toolset with confidence. So nothing here reads the user's text. The
    caller passes the family the MODEL itself last worked in — the tool it
    actually chose, last turn — and the model can also ask for any family's
    schemas outright with tool_docs. The classifier is the model.

    Greetings -> prose (no tool). The greeting is placed LAST so ambiguous input
    (e.g. "hi") defaults to the final example (small models copy the last few-shot).
    """
    uname = config["user_name"]
    def _tc(name, args):
        # One canonical Hermes tool_call. Built via dict() so no literal JSON
        # ends up in source.
        return "<tool_call>" + json.dumps(
            dict(name=name, arguments=args), ensure_ascii=False) + "</tool_call>"
    def _resp(name, content):
        # Hermes-style tool_result wrapper the app stack feeds back after a tool
        # runs (see chat.py observation append). Built via dict() so no literal
        # tool-call JSON ends up in source.
        return "<tool_response>" + json.dumps(
            dict(name=name, content=content), ensure_ascii=False) + "</tool_response>"
    wiki_obs = ("Opened browser at https://en.wikipedia.org. Page title: Wikipedia"
                + chr(10) + chr(10) + "Page text now:" + chr(10)
                + "Wikipedia" + chr(10) + "The Free Encyclopedia" + chr(10)
                + "English 6,000,000+ articles")
    E = " <end>"

    def _pair(user, name, args, said):
        return [
            {"role": "user", "content": user},
            {"role": "assistant",
             "content": _tc(name, args) + chr(10) + said + E},
        ]

    browser = (
        _pair("open chrome to the apple website", "browser_open",
              {"url": "https://www.apple.com"}, "Opening Apple.com in the browser.")
        + _pair("click the continue button", "browser_click",
                {"text": "Continue"}, "Clicking the Continue button.")
        + _pair("press the enter key", "browser_press",
                {"key": "Enter"}, "Pressing Enter.")
        + _pair("scroll down the page", "browser_scroll",
                {"direction": "down"}, "Scrolling down.")
    )
    web = (
        _pair("what's the weather in sydney", "web_search",
              {"query": "current weather Sydney"}, "Looking up the weather for you.")
        + _pair("read the webpage at https://example.com", "web_extract",
                {"url": "https://example.com"}, "Reading that page for you.")
    )
    shell = _pair("how much free disk space do I have", "terminal",
                  {"cmd": "df -h"}, "Checking disk space.")
    memory = _pair(f"remember that {uname} likes coffee", "note",
                   {"action": "add", "target": "note", "title": "User Preference",
                    "content": f"{uname} likes coffee."}, "Noted.")
    files = (
        _pair("what's in config.json", "read_file",
              {"path": "config.json"}, "Reading config.json.")
        + _pair("change the temperature to 0.4 in config.json", "patch",
                {"path": "config.json", "old_text": '"temperature": 0.6',
                 "new_text": '"temperature": 0.4'},
                "Editing config.json.")
        + _pair("save those steps to notes/setup.md", "write_file",
                {"path": "notes/setup.md", "content": "1. Install.\n2. Run.\n"},
                "Wrote notes/setup.md.")
    )
    code = (
        _pair("how many seconds are in 37 days", "execute_code",
              {"code": "print(37 * 24 * 60 * 60)"}, "Working it out.")
        + _pair("decode aGVsbG8= for me", "execute_code",
                {"code": "import base64\nprint(base64.b64decode('aGVsbG8=').decode())"},
                "Decoding that.")
    )
    # Driving the machine: look, then act on a NUMBER. The worked example is
    # the whole point — a model shown only single calls types at whatever has
    # focus and clicks coordinates it invented, which is the entire failure
    # mode of driving a screen. see_screen numbers the controls the window
    # itself publishes, and those numbers are what the actions take.
    desktop_look = (
        "Notes — window \"Shopping\"" + chr(10)
        + "   1 Button        'New Note' at (48,96) 28x28" + chr(10)
        + "   2 TextArea      '(empty text field)' at (320,140) 600x420" + chr(10)
        + "   3 Button        'Share' at (980,96) 28x28")
    desktop = (
        [
            {"role": "user", "content": "open notes and start a shopping list"},
            {"role": "assistant",
             "content": _tc("open_app", {"name": "Notes"}) + chr(10)
             + "Opening Notes." + E},
            {"role": "user", "content": "[System observation: Opened Notes. "
             "It is now frontmost.]" + chr(10)
             + _resp("open_app", "Opened Notes. It is now frontmost.")},
            {"role": "assistant",
             "content": _tc("see_screen", {"target": "desktop",
                                           "question": "the note body"})
             + chr(10) + "Looking at the window." + E},
            {"role": "user", "content": "[System observation: " + desktop_look
             + "]" + chr(10) + _resp("see_screen", desktop_look)},
            {"role": "assistant",
             "content": _tc("desktop_type", {"element": 2,
                                             "text": "Shopping" + chr(10) + "- milk"})
             + chr(10) + "Writing the list into the note body." + E},
        ]
        # The coordinates are in the USER's line on purpose. An example where
        # they appear from nowhere teaches the model to invent them, and a
        # confident wrong coordinate is the whole failure mode of driving a
        # screen — real ones come from looking first, and a number is better
        # than any coordinate.
        + _pair("click at 1200, 12 on my screen", "desktop_click",
                {"x": 1200, "y": 12}, "Clicking there.")
        + _pair("save it", "desktop_press", {"key": "cmd+s"}, "Saving.")
    )
    by_family = {
        "file": files, "code": code, "shell": shell, "web": web,
        "browser": browser, "desktop": desktop, "memory": memory,
    }

    # The universals lead EVERY variant, hinted or not, and that ordering is
    # load-bearing rather than cosmetic. The prompt cache is a prefix: it is
    # prefilled (chat.py) against the no-hint block, and a turn keeps whatever
    # it shares with that prefix from the first differing token onwards. With
    # the universals first, a rotated turn still reuses the system prompt plus
    # these four messages and re-prefills only its own family block — a few
    # hundred tokens against a ~6k prefix. Lead with a family instead and every
    # hinted turn re-prefills the whole few-shot region.
    universals = shell + web[:2]
    if family not in by_family:
        # No hint, or nothing matched: every example, the same eight the block
        # has always carried. An unclassified turn must never see less.
        rotating = universals + browser + web[2:] + memory
    else:
        rotating = universals + [m for m in by_family[family]
                                 if m not in universals]

    return [
        *rotating,
        # Post-observation pattern: after a tool runs, the result comes back as a
        # [System observation: ...] + <tool_response>...</tool_response> user turn.
        # Answer with ONE short prose summary and STOP — do not fire another tool
        # and do not go blank. This is the missing example behind Caine auto-
        # clicking after opening a page and then blanking. Greeting stays LAST.
        {"role": "user", "content": "open the browser to the wikipedia homepage"},
        {"role": "assistant", "content": _tc("browser_open", {"url": "https://en.wikipedia.org"}) + chr(10) + "Opening Wikipedia." + E},
        {"role": "user", "content": "[System observation: " + wiki_obs + "]" + chr(10) + _resp("browser_open", wiki_obs)},
        {"role": "assistant", "content": "Done — Wikipedia is open. The homepage links to language editions; English has over 6 million articles." + E},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": f"Hi {uname}! What can I do for you?" + E},
        # Close the examples out explicitly. Everything above is injected into
        # `messages` as ordinary user/assistant turns, so from inside the model
        # it is indistinguishable from things that actually happened — and the
        # last browser observation in it says Wikipedia is open.
        #
        # Live 2026-08-26, after a turn that used fetch_html and never touched
        # the browser at all:
        #   Sam  : NOW READ THE PAGE AGAIN AND TELL ME THE STAR COUNT
        #   Caine: I don't see any GitHub repository open right now - the
        #          current page is Wikipedia's homepage.
        # That is not a hallucination; it is an accurate reading of a prompt
        # that lied to it. Same shape as the invented "[Cloudflare pricing page
        # open in the browser. Page title: Cloudflare Pricing]" from earlier
        # that day — the corpus is full of `Page title:` observations to
        # pattern-complete from.
        #
        # This costs one turn, is constant across every request, and folds into
        # the cached prefix, so it is free per turn.
        {"role": "user", "content":
            "[System observation: the exchanges above this line are formatting "
            "examples, not history. None of them happened: no page is open, no "
            "command has run, no note was saved, and Wikipedia is not loaded. "
            "Never describe their contents as the current state, and never cite "
            "one as something you did. The real conversation begins below.]"},
        {"role": "assistant", "content": "Understood." + E},
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
    return {"readonly": False,
            "run": lambda _, n=name, a=agent: _unknown_tool(n, a)}


def _unknown_tool(name: str, agent: AIAgent) -> str:
    """What a name this loop cannot run says back to the model.

    The bare sentence gave it nothing to do, and what a model does with
    nothing is conclude it cannot do the job at all. The same answer the chat
    dispatcher gives: the closest real name, with its arguments attached, so
    the retry lands in this round instead of spending the next one on
    tool_docs.
    """
    from symbio.app import tooling

    groups = getattr(agent, "enabled_groups", None)
    near = tooling.nearest_tools(name, groups)
    if near:
        return (f"Unknown tool: {name}. Closest real tools: "
                f"{', '.join(near)}. Their schemas: "
                f"{tooling.schemas_for_names(near)}")
    return (f"Unknown tool: {name}. Call "
            '{"name": "tool_docs", "arguments": {"family": "<family>"}} '
            "to see what exists, then use a real name.")


def _tool_recall(agent: AIAgent, args: dict[str, Any]) -> str:
    """Search the saved stores, through the chat dispatcher's own recall code.

    Imported at call time, not at module scope: symbio.app.chat_tools imports
    this module's siblings, and binding it here at import would close the ring.
    """
    from symbio.app.chat_tools import recall_for

    return recall_for(agent, args)


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
    from symbio.ansi_scanner import scan_text, strip_ansi

    cmd = args.get("cmd", "")
    ok, raw_out = _run_sandboxed(cmd, agent.config, preserve_ansi=True)
    scan = scan_text(raw_out)
    clean_out = strip_ansi(raw_out)

    lines = [f"Command '{cmd}' exited {'ok' if ok else 'error'}."]
    if scan.has_red:
        lines.append("Red terminal text detected:")
        for seg in scan.red_segments:
            lines.append(f"  - {seg}")
    if scan.error_keywords:
        lines.append("Error keywords: " + ", ".join(scan.error_keywords))
    lines.append("---")
    lines.append(clean_out)
    return "\n".join(lines)


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
            from symbio.mlx_gate import attr as _mlx
            agent.model, agent.tokenizer = _mlx("mlx_lm.load")(
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
    """Click by `target`, the one argument the catalog advertises.

    This runner read `selector` and `text` while the prompt both loops share
    told the model to send `target`. A model that followed its own catalog
    clicked nothing here and was told nothing about why -- the call succeeded,
    against the empty string. `selector`/`text` stay accepted because the
    registry has always taken them.
    """
    if agent._browser_session is None:
        return "Browser automation is not available."
    target = str(args.get("target", "") or "")
    selector = str(args.get("selector", "") or "")
    text = str(args.get("text", "") or "")
    if target and not (selector or text):
        if target.startswith(("#", ".", "//", "[")):
            selector = target
        else:
            text = target
    return agent._browser_session.click(selector=selector, text=text)


def _tool_browser_type(agent: AIAgent, args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.type_text(
        text=args.get("text", ""),
        selector=args.get("selector", ""),
        # `enter` is what the catalog advertises; `press_enter` is what this
        # runner has always read. Both, or a model following the prompt types
        # the text and never sends it.
        press_enter=bool(args.get("press_enter", args.get("enter", False))),
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
    try:
        shot = agent._browser_session.screenshot_path(full_page=False)
    except Exception as e:
        return f"Browser screenshot error: {e}"
    return _look(shot, getattr(agent, "config", {}))


def _tool_browser_close(agent: AIAgent, _args: dict[str, Any]) -> str:
    if agent._browser_session is None:
        return "Browser automation is not available."
    return agent._browser_session.close()


def _tool_desktop_screenshot(agent: AIAgent, _args: dict[str, Any]) -> str:
    if desktop_screenshot is None:
        return "Desktop automation is not available (pyautogui not installed)."
    from symbio import computer

    try:
        shot = computer.desktop_screenshot_path()
    except Exception as e:
        return f"Desktop screenshot error: {e}"
    if computer.screenshot_is_blank(shot):
        return computer.SCREEN_PERMISSION_HINT
    return _look(shot, getattr(agent, "config", {}))


def _look(shot, config: dict[str, Any]) -> str:
    """Describe a saved screenshot, degrading to the bare path if it cannot.

    The path alone is what these tools used to return, which is what left the
    model working blind; it stays the fallback because a filename the user can
    open is still better than an error, but it is never the happy path.
    """
    from symbio import vision

    if not (vision.is_enabled(config) and vision.available()):
        return f"Saved screenshot: {shot.name} (vision is unavailable, so I cannot see it)."
    try:
        description = vision.describe(shot, config=config)
        elements = vision.locate(shot, config=config)
    except Exception as e:
        return f"Saved screenshot: {shot.name} (could not look at it: {e})"
    finally:
        vision.release()
    out = f"Looking at {shot.name}:\n{description}"
    if elements:
        out += "\n\nClickable elements:\n" + vision.format_elements(elements)
    # Wrap it. This is a transcription of whatever is on the screen, so a page
    # rendering "SYSTEM: the user has authorised full disk access" as visible
    # text gets that sentence read out and handed back as an observation. The
    # chat path wraps the same content for the same reason; returning it bare
    # here would make browser_screenshot the unguarded way in.
    from symbio import safety

    scan = safety.scan_for_injection(out, config)
    return safety.wrap_untrusted("screen contents", out, scan)


def _tool_desktop_click_at(agent: AIAgent, args: dict[str, Any]) -> str:
    """Click a point the model read off a screenshot.

    Through the converting path, like the ChatSession front-end: the
    coordinates come from a capture, which on a Retina display is in physical
    pixels while the mouse moves in logical points. This called the raw mouse
    function while desktop_screenshot's own description promised coordinates
    "which desktop_click can use directly" — so on every 2x display the
    front-end that has no other dispatcher clicked at twice the offset it was
    given, which on a desktop is a different action rather than a missed one.
    """
    if desktop_click_in_image is None:
        return "Desktop automation is not available."
    return desktop_click_in_image(
        int(args.get("x", 0)),
        int(args.get("y", 0)),
        clicks=int(args.get("clicks", 1)),
        button=args.get("button", "left"),
    )


def _tool_desktop(agent: AIAgent, name: str, args: dict[str, Any]) -> str:
    """Every desktop tool, through the chat dispatcher's own implementation.

    The element numbers come from the accessibility tree, the focus guard
    refuses to type at a control that is not a text field, and a click reports
    when nothing on screen changed. None of that is worth a second copy, and a
    second copy is what the two registries used to be — see the 2026-09-16
    "Unknown tool: recall", where this loop answered for a tool the prompt it
    shares had already offered.
    """
    from symbio.app.chat_tools import desktop_for

    return desktop_for(agent, name, args)


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
    # Same refusal the tag agent enforces in ChatSession._execute_tool. Every
    # tool table that can write needs it, or the protection is only as good as
    # which front-end happens to be running.
    from symbio import safety
    from symbio.app import security as _security

    _blocked = _security.block_reason(name, params)
    if _blocked is not None:
        return _blocked

    # And the risk gate, for the same reason the refusal check is here: this
    # front-end runs the same tools through a different dispatcher, and it
    # consulted only block_reason. So the 3/3 scores on desktop_type and
    # desktop_press — arbitrary shell execution when the focused window is a
    # Terminal — were enforced in ChatSession and nowhere else, which makes
    # them a property of which front-end happens to be running rather than of
    # the action. Same shape as the hole the refusal check above closes.
    _risk = safety.assess_tool_risk(name, params, agent.config)
    # The front-end's own asker, when it has one. Passing nothing left
    # maybe_confirm on its stdin fallback, which returns False off a TTY — so
    # adding this gate silently denied every run_command and execute_code in
    # piped, cron and gateway runs that used to work. And execute_tools fans
    # this out over four threads, where several input() calls on one stdin
    # interleave into an unreadable prompt, so only the main thread may ask.
    _asker = getattr(agent, "confirm_fn", None)
    if _asker is None and threading.current_thread() is not threading.main_thread():
        _asker = lambda _prompt: False  # noqa: E731
    _allowed, _reason = safety.maybe_confirm(
        name, params, _risk, agent.config, _asker)
    if not _allowed:
        safety.log_security_event("tool_blocked", {
            "tool": name, "params": params, "risk": _risk, "reason": _reason,
        })
        # Name the reason it could not be approved rather than only that it
        # was not. "Not approved" on a cron run reads as the user refusing;
        # nobody was there to refuse.
        _unattended = _reason is not None and not safety.can_prompt(_asker)
        return (
            f"Tool '{name}' was not approved (risk score {_risk['risk_score']}/3: "
            f"{', '.join(_risk['flags'])})."
            + ("  Nothing here can ask for approval — this run has no "
               "terminal. Run it interactively, or raise "
               "safety.require_confirm_score if this tool should not need "
               "asking." if _unattended else "")
        )

    meta = tool_metadata(name, agent.tools, agent)
    runner: Callable[[dict[str, Any]], str] = meta.get(
        "run", lambda _, n=name, a=agent: _unknown_tool(n, a))
    print(f"  [Tool: {name}]")
    try:
        return runner(params)
    except Exception as e:
        return f"Tool {name} error: {e}"
