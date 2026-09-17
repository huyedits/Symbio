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
import shutil
import sys
import unicodedata

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREY = "\033[38;5;245m"
_BLUE = "\033[38;5;39m"
_GREEN = "\033[38;5;71m"
_YELLOW = "\033[38;5;179m"
_RED = "\033[38;5;167m"
_ACCENT = "\033[38;5;173m"

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
    if os.environ.get("TERM") == "dumb":
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
        # Work tags and unknown tags render the same, deliberately: a tag
        # nobody has classified yet is still work, and giving it the work
        # style is the forgiving reading. The two branches that used to say
        # this returned byte-identical strings, so the _WORK_TAGS set that
        # gated one of them decided nothing — configuration in appearance
        # only, and now deleted.
        return "  " + _paint(f"· {tag.lower()} {body}".rstrip(), _DIM, enabled=on)

    return message


def assistant_prefix(color: bool | None = None) -> str:
    """What precedes the assistant's streamed reply."""
    on = colors_enabled() if color is None else color
    return _paint(BULLET, _GREEN, enabled=on) + " "


def user_prompt(color: bool | None = None) -> str:
    """The line the person types on."""
    on = colors_enabled() if color is None else color
    return _paint("❯", _ACCENT, _BOLD, enabled=on) + " "


def _fit(text: str, width: int) -> str:
    """Fit a single line in terminal cells, without splitting wide characters.

    Only chrome is clipped. Replies, tool results and copyable paths in the
    command reference still pass through whole. Config values cannot inject
    terminal escapes, newlines or bidi controls into the frame.
    """
    if width <= 0:
        return ""
    clean = _plain(text)
    if _cells(clean) <= width:
        return clean
    result = []
    used = 0
    for ch in clean:
        size = _cell_width(ch)
        if used + size > width - 1:
            break
        result.append(ch)
        used += size
    return "".join(result) + "…"


# Strip complete ANSI sequences before filtering controls, so an injected
# colour/OSC title doesn't leave its payload visible inside the welcome panel.
_ANSI_RE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-_])")


def _plain(text: str) -> str:
    return "".join(ch for ch in _ANSI_RE.sub("", str(text))
                   if not unicodedata.category(ch).startswith("C")
                   and ch not in "\u2028\u2029")


def _cell_width(ch: str) -> int:
    if unicodedata.category(ch) in ("Mn", "Me"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _cells(text: str) -> int:
    return sum(_cell_width(ch) for ch in _plain(text))


def terminal_width(width: int | None = None) -> int:
    """Read on every draw, including resizes; never exceed the real window."""
    if width is None:
        try:
            width = shutil.get_terminal_size((80, 24)).columns
        except OSError:
            width = 80
    return max(1, min(int(width), 96))


def model_label(model: str) -> str:
    """A model ID, not a snapshot hash or an entire Hugging Face cache path."""
    model = _plain(model).rstrip("/")
    match = re.search(r"(?:^|/)models--([^/]+)/snapshots(?:/|$)", model)
    if match:
        return match.group(1).replace("--", "/")
    if model.startswith(("/", "~/")):
        return model.rsplit("/", 1)[-1]
    return model


def workspace_label(workspace: str | None = None) -> str:
    if workspace is None:
        try:
            workspace = os.getcwd()
        except OSError:
            workspace = "(workspace unavailable)"
    home = os.path.expanduser("~")
    if workspace == home:
        workspace = "~"
    elif workspace.startswith(home + os.sep):
        workspace = "~" + workspace[len(home):]
    return _plain(workspace)


def welcome_panel(config: dict, *, width: int | None = None,
                  color: bool | None = None, workspace: str | None = None,
                  detail: str = "") -> list[str]:
    """Compact, scrollback-friendly welcome; no model or filesystem work."""
    on = colors_enabled() if color is None else color
    width = terminal_width(width)
    name = config.get("assistant_name") or "Symbio"
    user = config.get("user_name")
    greeting = f"Welcome, {user}." if user and user != "you" else "Welcome."
    model = model_label(config.get("model_name") or "not configured")
    workspace = workspace_label(workspace)
    if width < 24:
        # A split pane can be narrower than any useful box. Drop the frame
        # rather than enforcing a minimum width that would wrap every row.
        return [_paint(_fit(text, width), _ACCENT if i == 0 else _DIM, enabled=on)
                for i, text in enumerate(("Symbio", str(name), model, workspace,
                                           "/ for commands"))]

    room = width - 6
    menu = ("/ commands   /status session   /tools capabilities" if room >= 50
            else "/ commands   /status   /tools" if room >= 29
            else "/ for commands")
    rows = [
        ("", ()),
        (f"✦ {name} · personal agent", (_BOLD,)),
        (f"{greeting} What are we working on?", (_DIM,)),
        ("", ()),
        (f"Model  {model}", (_GREY,)),
        (f"cwd    {workspace}", (_GREY,)),
    ]
    if detail:
        rows.append((detail, (_DIM,)))
    rows += [("", ()), (menu, (_DIM,))]
    # The fixed header is ten cells before the closing corner. Reserve that
    # corner explicitly so the frame never autowraps at the right edge.
    lines = ["", _paint("╭─ Symbio " + "─" * max(0, width - 11) + "╮",
                         _ACCENT, enabled=on)]
    edge = _paint("│", _ACCENT, enabled=on)
    for text, codes in rows:
        body = _fit(text, room)
        lines.append(edge + "  " + _paint(body, *codes, enabled=on)
                     + " " * (room - _cells(body) + 2) + edge)
    lines.append(_paint("╰" + "─" * (width - 2) + "╯", _ACCENT, enabled=on))
    return lines


def prompt_context(config: dict, *, width: int | None = None,
                   color: bool | None = None) -> str:
    """A divider and real key hints, kept OUTSIDE readline's editable line."""
    on = colors_enabled() if color is None else color
    width = terminal_width(width)
    hints = "Enter send · Tab complete · /quit exit" if width >= 64 else "/ commands · Tab complete"
    label = _fit(f" {config.get('assistant_name') or 'Symbio'} · {hints} ", width - 2)
    return _paint("─" + label + "─" * (width - 1 - _cells(label)), _GREY, enabled=on)


def readline_prompt(prompt: str) -> str:
    """Mark ANSI as zero-width for both GNU readline and macOS libedit.

    These markers belong ONLY in input(), never in printed prompt redraws.
    Without them, editing a long coloured prompt wraps at the wrong column.
    """
    if "\033" not in prompt or not sys.stdin.isatty():
        return prompt
    try:
        import readline  # noqa: F401 — input() only uses markers with readline
    except ImportError:
        return prompt
    return _ANSI_RE.sub(lambda match: "\001" + match.group() + "\002", prompt)


def status_frame(text: str, *, width: int | None = None,
                 color: bool | None = None) -> str:
    """One spinner row; leave the last cell free to avoid terminal autowrap."""
    on = colors_enabled() if color is None else color
    text = _fit("  " + text, terminal_width(width) - 1)
    return _paint(text, _ACCENT, enabled=on)


def install_command_completion(command_names) -> bool:
    """Share slash completion between local chat and the daemon terminal.

    The callable is read on every Tab, so commands saved during a session
    appear immediately. Completion never loads a model or runs a command.
    """
    if not sys.stdin.isatty():
        return False
    try:
        import readline

        def complete(text: str, state: int):
            try:
                buffer = readline.get_line_buffer().lstrip()
                if not buffer.startswith("/") or any(ch.isspace() for ch in buffer):
                    return None
                typed = text.lstrip("/").lower()
                matches = [f"/{name}" for name in command_names()
                           if name.startswith(typed)]
                return matches[state] if state < len(matches) else None
            except Exception:
                return None

        readline.set_completer(complete)
        readline.set_completer_delims(" \t\n")
        if "libedit" in (getattr(readline, "__doc__", "") or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind("set show-all-if-ambiguous on")
    except Exception:
        return False
    return True


def _demo() -> None:
    """A deterministic visual preview, with no config, credentials or model."""
    import argparse

    parser = argparse.ArgumentParser(description="Preview Symbio's terminal UI without a model.")
    parser.add_argument("--demo", action="store_true", help="show an example conversation")
    parser.add_argument("--width", type=int, default=None, help="preview a terminal width")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return
    on = colors_enabled() and not args.no_color
    config = {"assistant_name": "Caine", "user_name": "you",
              "model_name": "Qwen/Qwen3-8B"}
    for line in welcome_panel(config, width=args.width, color=on,
                              workspace="~/projects/symbio", detail="Demo · no model or tools are running"):
        print(line)
    print(prompt_context(config, width=args.width, color=on))
    print(user_prompt(color=on) + "Help me improve this CLI.")
    print()
    print(assistant_prefix(color=on) + "I'll check the terminal rendering first.")
    print(style_line("  [Tool: read_file]", color=on))
    print(style_line("  [Observation] symbio/app/chat_style.py\nPresentation stays separate from the model.", color=on))
    print(assistant_prefix(color=on) + "Ready to build. Type / to explore commands.")
    print()
    print(prompt_context(config, width=args.width, color=on))
    print(user_prompt(color=on))


if __name__ == "__main__":
    _demo()
