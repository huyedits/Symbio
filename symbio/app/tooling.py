"""The tag language: parsing tool calls out of replies and cleaning text."""

import re
from typing import Any

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


def parse_tools(reply: str) -> list[tuple[str, dict[str, Any]]]:
    """Extract tool calls from the model reply."""
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
    # A reply cut off mid-tag leaves an unterminated tag; never show it.
    display = re.sub(
        r'<(?:cmd|py|search|read|browse|click|type|scroll|note|skill|cron|digest|train|memory|profile|config)\b[^>]*>[^<]*$',
        '', display, flags=re.DOTALL,
    )
    return clean_response(display)
