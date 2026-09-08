"""How the terminal renders what the session says.

Presentation only, and deliberately the last thing in the chain. The bracket
tags this restyles — "[Tool: browser_open]", "[Observation] ...", "[System
observation: ...]" — are a protocol before they are decoration: the same
vocabulary goes into `self.history`, where the model reads it, and the model
has been trained against that shape. So nothing here touches those strings.
`chat_loop` wires `style_line` into its own `_cli_output` and nowhere else,
which means the daemon protocol and the Telegram front-end keep the plain text
they already parse, and the history the model sees is byte-identical either
way. A prettier terminal must not become a second representation that can
drift from the real one — this codebase's bug history is largely two
representations disagreeing.

The glyphs follow the convention a reader of modern agent CLIs already knows:
a filled dot for "the agent did something", an elbow for "and this came back".
"""

from __future__ import annotations

import os
import re
import sys

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREY = "\033[38;5;245m"
_BLUE = "\033[38;5;39m"
_GREEN = "\033[38;5;71m"
_YELLOW = "\033[38;5;179m"
_RED = "\033[38;5;167m"

BULLET = "⏺"
ELBOW = "⎿"


def colors_enabled(stream=None) -> bool:
    """Colour only when a person is actually looking at a terminal.

    NO_COLOR is honoured because piping this into a file or a grep is a normal
    thing to do with a chat log, and escape codes make that output worse.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("SYMBIO_NO_COLOR"):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _paint(text: str, *codes: str, enabled: bool = True) -> str:
    if not enabled or not codes:
        return text
    return "".join(codes) + text + _RESET


# The status lines the session emits, mapped to how they should read. Order
# matters: the first match wins, so the specific patterns precede the generic
# "[Word] ..." fallback.
_TOOL_RE = re.compile(r"^\s*\[Tool:\s*([^\]]+)\]\s*$")
_OBSERVATION_RE = re.compile(r"^\s*\[Observation(?:\s+([^\]]+))?\]\s*(.*)$", re.DOTALL)
_MOOD_RE = re.compile(r"^\s*\[Mood:\s*([^\]]+)\]\s*$")
_TAGGED_RE = re.compile(r"^\s*\[([A-Za-z][\w /-]*)\]\s*(.*)$", re.DOTALL)
_SECURITY_RE = re.compile(r"^\s*\[(Security|Safety|Blocked)\b([^\]]*)\]\s*(.*)$", re.DOTALL)

# Tags that report something going wrong or needing attention, and so keep
# their colour rather than fading into the background.
_ALERT_TAGS = {"security", "safety", "blocked", "unverified", "mlx error", "error"}
_WORK_TAGS = {"vision", "dispatch", "learn", "train", "golden", "browser",
              "cache", "memory", "cron", "rag", "format", "reasoning", "blank"}


def _indent_continuations(body: str, pad: str) -> str:
    """Keep multi-line tool output lined up under its elbow."""
    lines = body.split("\n")
    return ("\n" + pad).join(lines) if len(lines) > 1 else body


def style_line(message: str, color: bool | None = None) -> str:
    """Render one status line the way the terminal should show it.

    Unrecognised text is returned unchanged: this is a skin over a protocol
    that keeps growing, and a formatter that swallows a line it does not know
    would hide exactly the new diagnostics someone just added.
    """
    if not isinstance(message, str) or not message.strip():
        return message
    on = colors_enabled() if color is None else color
    if not on:
        # Piped, redirected, or NO_COLOR: hand back the exact legacy line.
        # The glyphs are for a person looking at a terminal; a log file, a
        # grep, and the repo's own pty harnesses all match on "[Tool: ...]"
        # and "[Observation] ...", and reshaping those for an audience that
        # cannot see colour anyway would break them for no gain.
        return message

    # Style each line on its own. One output_fn call often carries several
    # status lines ("[Learn] captured ...\n[Learn] 3/5 collected ..."), and
    # the tag patterns end in DOTALL groups — so the first tag matched and
    # swallowed every line after it as its own body, leaving the rest raw.
    # Observations are the exception: their body is deliberately multi-line
    # and hangs under one elbow.
    if "\n" in message and not _OBSERVATION_RE.match(message):
        return "\n".join(style_line(part, color=on) if part.strip() else part
                          for part in message.split("\n"))

    m = _TOOL_RE.match(message)
    if m:
        return _paint(BULLET, _BLUE, enabled=on) + " " + _paint(
            m.group(1).strip(), _BOLD, enabled=on)

    m = _OBSERVATION_RE.match(message)
    if m:
        name, body = (m.group(1) or "").strip(), m.group(2).rstrip()
        head = f"{name}: " if name else ""
        pad = "     "
        return ("  " + _paint(ELBOW, _GREY, enabled=on) + "  "
                + _paint(head + _indent_continuations(body, pad), _GREY, enabled=on))

    m = _SECURITY_RE.match(message)
    if m:
        # group(2) keeps its own leading punctuation ("Security" + ": risk
        # score 3/3"), so joining with a space produced "Security : ...".
        label = (m.group(1) + (m.group(2) or "")).strip()
        body = m.group(3).strip()
        return ("  " + _paint(label, _YELLOW, enabled=on)
                + (" " + body if body else ""))

    m = _MOOD_RE.match(message)
    if m:
        return "  " + _paint(f"mood: {m.group(1).strip()}", _DIM, enabled=on)

    m = _TAGGED_RE.match(message)
    if m:
        tag, body = m.group(1).strip(), m.group(2).rstrip()
        key = tag.lower()
        if key in _ALERT_TAGS:
            return "  " + _paint(f"{tag} {body}".rstrip(), _RED, enabled=on)
        if key in _WORK_TAGS:
            return "  " + _paint(f"· {tag.lower()} {body}".rstrip(), _DIM, enabled=on)
        return "  " + _paint(f"· {tag.lower()} {body}".rstrip(), _DIM, enabled=on)

    return message


def assistant_prefix(color: bool | None = None) -> str:
    """What precedes the assistant's streamed reply."""
    on = colors_enabled() if color is None else color
    return _paint(BULLET, _GREEN, enabled=on) + " "


def user_prompt(color: bool | None = None) -> str:
    """The line the person types on."""
    on = colors_enabled() if color is None else color
    return _paint("›", _GREY, enabled=on) + " "
