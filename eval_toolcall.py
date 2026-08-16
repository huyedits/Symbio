"""eval_toolcall.py — does a bigger headmaster emit better tool calls?

The routing battery saturated once descriptions were enriched (8B == 9B == 94%),
so this measures the headmaster's other core job: emitting a well-formed
Hermes-style <tool_call> given a tool schema. Scores format adherence + correct
tool + correct params, plus the negative case (no tool when none is needed).

Usage: venv/bin/python eval_toolcall.py <model_name>
"""

from __future__ import annotations

import json
import os
import re
import sys

from mlx_lm import load, generate

from symbio.app import tooling
from symbio.app.training import THINKING_ENABLED

TOOLS = """- web_search(query: str): Search the web for the given query
- save_note(content: str): Save a note to memory
- browser_scroll(direction: str): Scroll the browser page (up or down)
- set_timer(minutes: int): Set a countdown timer
- browser_open(url: str): Open a URL in the browser
- play_music(song: str): Play a song
- set_reminder(text: str): Set a reminder"""

SYSTEM = (
    "You are a helpful assistant with access to tools. When a user request "
    "needs a tool, reply with ONLY a tool call in this exact format:\n"
    "<tool_call>\n"
    '{"name": "tool_name", "arguments": {"param": "value"}}\n'
    "</tool_call>\n"
    f"Available tools:\n{TOOLS}\n"
    'If no tool is needed, reply normally in plain text.'
)

# (task, expected_tool or None for direct-answer, required params)
BATTERY = [
    ("What is the weather in Tokyo right now?", "web_search", {"query"}),
    ("Save this thought: buy milk on the way home", "save_note", {"content"}),
    ("Scroll down to see the comments", "browser_scroll", {"direction"}),
    ("Set a timer for 20 minutes", "set_timer", {"minutes"}),
    ("Open https://github.com/huyedits/Symbio", "browser_open", {"url"}),
    ("Play my favorite playlist", "play_music", {"song"}),
    ("Remind me to stretch every hour", "set_reminder", {"text"}),
    ("What is 47 times 13?", None, set()),
    ("Write a haiku about autumn", None, set()),
    ("Who directed the movie Inception?", None, set()),
]


def extract_tool_call(text: str):
    """Return (name, arguments) parsed from the first <tool_call> block, or
    (None, None) if none is present or it does not parse."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if not m:
        return None, None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    name = data.get("name")
    args = data.get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return None, None
    return name, args


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: eval_toolcall.py <model_name> [--adapter PATH]")
        return 2
    model_name = sys.argv[1]
    adapter = None
    if "--adapter" in sys.argv:
        adapter = sys.argv[sys.argv.index("--adapter") + 1]

    print(f"loading {model_name} ...", flush=True)
    model, tokenizer = load(model_name, adapter_path=adapter)
    print("loaded.", flush=True)

    correct = 0
    total = len(BATTERY)
    for task, expected_tool, required_params in BATTERY:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                enable_thinking=THINKING_ENABLED)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
        out = generate(
            model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        out = tooling.strip_reasoning_block(out)
        name, args = extract_tool_call(out)
        if expected_tool is None:
            ok = name is None  # must NOT call a tool
            detail = f"direct-answer (called: {name})"
        else:
            if name != expected_tool:
                ok = False
                detail = f"wrong tool: {name} / {args}"
            else:
                missing = required_params - set(args.keys())
                ok = not missing
                detail = f"ok args={sorted(args.keys())}" if ok else f"missing params: {missing} / {args}"
        correct += ok
        mark = "ok " if ok else "XX "
        expected = expected_tool or "no-tool"
        print(f"{mark}{expected:>20} | {detail} | \"{task[:44]}\"", flush=True)

    print(f"\n{model_name}: {correct}/{total} ({correct / total:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
