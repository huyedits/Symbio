"""Terminal presentation for the chat loop: the spinner, the rainbow banner,
adapter/learning status lines, the per-session log handler, and the health
report writer.

Kept apart from the session logic so that changing how something is displayed
never means opening the file that decides what happens.
"""

import json
import logging
import os
import shutil
import sys
import textwrap
import threading
import time
from datetime import datetime
from typing import Any

from symbio import constants
from symbio.app import learn


def _persist_health_report(session_id: str, report: dict[str, Any]):
    """Write the session health report to both a per-session file and a
    rolling 'latest' file inside sessions/."""
    constants.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path = constants.SESSIONS_DIR / f"{session_id}_health.json"
    report["_persisted"] = True
    session_path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                            encoding="utf-8")
    latest_path = constants.SESSIONS_DIR / "latest_health.json"
    latest_path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                           encoding="utf-8")


def _make_chat_logger() -> logging.Logger:
    logger = logging.getLogger("chat")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # One handler per session; drop stale ones so lines don't fan out to
    # every log file ever opened in this process.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    path = constants.LOG_DIR / f"chat_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    constants.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, delay=True)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger

# xterm-256 stops around the hue circle. Skipping pure blue (21) keeps every
# glyph legible on a dark terminal.
_RAINBOW_COLORS: tuple[int, ...] = (196, 202, 220, 46, 51, 33, 129)


def term_width(default: int = 80) -> int:
    """How wide the terminal is right now, clamped to something readable.

    Read per call, not cached: a window resized mid-session is the common case
    and a cached width is wrong for the rest of the session. The floor keeps a
    very narrow window from producing one-word-per-line columns; the ceiling
    keeps a maximized window from producing lines too long to scan. A pipe or
    a front-end with no terminal reports nothing, and gets the default.
    """
    try:
        columns = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        columns = default
    return max(40, min(int(columns or default), 110))


def _wrap(text: str, width: int) -> list[str]:
    """textwrap with the two defaults that are wrong for this content.

    break_long_words splits a path or a URL across lines, which makes it
    uncopyable; break_on_hyphens turns "/skill-adapters" into "/skill-" and
    "adapters", which reads as two commands, neither of which exists. A token
    longer than the window overhangs instead, and the terminal soft-wraps it.
    """
    return textwrap.wrap(text, width, break_long_words=False,
                         break_on_hyphens=False)


def two_column(label: str, text: str, indent: int = 4, gap: int = 2,
               width: int | None = None) -> list[str]:
    """A label and its description, laid out for the width there actually is.

    Wide enough, and it is the familiar two columns with the description
    wrapped under itself. Too narrow for both — the label alone would leave
    fewer than ~24 columns for the text — and the description drops to its own
    indented line instead of being squeezed into a gutter. Returns lines so the
    caller keeps control of its own output_fn.
    """
    width = width or term_width()
    pad = " " * indent
    label_width = len(label)
    text = " ".join((text or "").split())
    if not text:
        return [f"{pad}{label}"]
    if label_width + gap + 24 <= width - indent:
        column = label_width + gap
        body = _wrap(text, max(20, width - indent - column)) or [""]
        first = f"{pad}{label}{' ' * gap}{body[0]}"
        rest = [f"{pad}{' ' * column}{line}" for line in body[1:]]
        return [first] + rest
    lines = [f"{pad}{label}"]
    lines += [f"{pad}  {line}"
              for line in _wrap(text, max(20, width - indent - 2))]
    return lines


def wrapped_list(items, indent: int = 4, width: int | None = None) -> list[str]:
    """A comma-separated list broken to fit, e.g. a family's tool names."""
    width = width or term_width()
    pad = " " * indent
    joined = ", ".join(items)
    return [f"{pad}{line}"
            for line in _wrap(joined, max(20, width - indent))] or [pad]


def rainbow(text: str) -> str:
    """Colour each visible character a step further around the hue circle.

    Falls back to the bare text when colour would be wrong or unwanted: a
    redirected stdout (logs, the piped harnesses), or NO_COLOR set. Spaces are
    left uncoloured so the cycle tracks glyphs rather than gaps.
    """
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    out: list[str] = []
    step = 0
    for ch in text:
        if ch.isspace():
            out.append(ch)
            continue
        out.append(f"\033[38;5;{_RAINBOW_COLORS[step % len(_RAINBOW_COLORS)]}m{ch}")
        step += 1
    out.append("\033[0m")
    return "".join(out)


class _Spinner:
    """Terminal spinner shown while waiting for visible model output.

    Runs on a daemon thread and anchors itself with carriage returns; stop()
    erases the line so streamed text can take its place. No-op when stdout
    is not a TTY (tests, pipes, or non-terminal front-ends).
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏67"

    def __init__(self, label: str = "thinking…"):
        self.label = label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.active = sys.stdout.isatty()
        self._start_time: float | None = None
        self._gen_tokens = 0
        self._lock = threading.Lock()

    def set_gen_tokens(self, n: int):
        with self._lock:
            self._gen_tokens = n

    def set_label(self, label: str):
        with self._lock:
            self.label = label

    def start(self):
        if self._thread is not None:
            return
        if not self.active:
            # No animation without a TTY, but silence is not the alternative:
            # a turn that prints nothing between the prompt and the reply looks
            # like a hang, and the wait here is tens of seconds on an 8B. One
            # static line costs nothing and cannot be mistaken for frozen.
            sys.stdout.write(f"  {self.label}\n")
            sys.stdout.flush()
            return
        self._stop_event.clear()
        self._start_time = time.perf_counter()

        def _spin():
            i = 0
            while not self._stop_event.wait(0.08):
                elapsed = time.perf_counter() - self._start_time
                frame = self._FRAMES[i % len(self._FRAMES)]
                with self._lock:
                    gen_tokens = self._gen_tokens
                tok_info = f" | generated {gen_tokens} tokens" if gen_tokens else ""
                if elapsed >= 5:
                    label = f"{self.label} ({int(elapsed)}s){tok_info}"
                else:
                    label = f"{self.label}{tok_info}"
                sys.stdout.write(f"\r{frame} {label}")
                sys.stdout.flush()
                i += 1

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()



def _adapter_trained_at() -> datetime | None:
    """mtime of the adapter weights — the last time a LoRA run wrote them."""
    weights = constants.ADAPTER_DIR / "adapters.safetensors"
    if weights.exists():
        try:
            return datetime.fromtimestamp(weights.stat().st_mtime)
        except OSError:
            return None
    return None


def _adapter_iters() -> int | None:
    """iters recorded in the last training run's adapter_config.json."""
    cfg = constants.ADAPTER_DIR / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        return int(json.loads(cfg.read_text(encoding="utf-8")).get("iters", 0)) or None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _fmt_ago(then: datetime, now: datetime | None = None) -> str:
    """Compact '2h ago'-style relative time."""
    now = now or datetime.now()
    secs = int((now - then).total_seconds())
    if secs < 60:
        return "just now"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hrs = mins // 60
    if hrs < 48:
        return f"{hrs}h ago"
    return f"{hrs // 24}d ago"


def learn_progress_line(config: dict[str, Any]) -> str:
    """One-line summary of the self-finetune loop: mistake counter state.

    e.g. '3/5 mistakes to next tune', '5/5 mistakes — tuning due', or
    'learn: off' when the loop is disabled."""
    learn_cfg = config.get("learn", {}) or {}
    if not learn_cfg.get("enabled", True):
        return "learn: off"
    threshold = max(1, int(learn_cfg.get("mistake_threshold", 5)))
    count = learn.mistake_note_count()
    suffix = "" if learn_cfg.get("auto_train", True) else " (auto-train off)"
    # Name which kinds of mistake are stacking up, so the counter says what the
    # next tune will actually be trained to fix, not just how many. e.g.
    # '3/5 mistakes to next tune (2 tool_error, 1 wrong_tool)'.
    counts = learn.mistake_category_counts()
    breakdown = ""
    # A lone "general" bucket is every note saying "unclassified" — the state
    # before any classification has run, and the state on every note written
    # before the field existed. It names nothing, so it earns no parentheses.
    if counts and set(counts) != {"general"}:
        breakdown = " (" + ", ".join(f"{n} {cat}" for cat, n in counts.items()) + ")"
    if count >= threshold:
        return f"{count}/{threshold} mistakes — tuning due{suffix}{breakdown}"
    return f"{count}/{threshold} mistakes to next tune{suffix}{breakdown}"


def adapter_status_value(config: dict[str, Any], adapter_loaded: bool) -> str:
    """Legible adapter + learn state, e.g.

    'loaded (trained 2h ago, 50 iters) · 3/5 mistakes to next tune'
    'none (base) · 3/5 mistakes to next tune'
    """
    progress = learn_progress_line(config)
    if not adapter_loaded:
        return f"none (base) · {progress}"
    bits: list[str] = []
    trained = _adapter_trained_at()
    if trained is not None:
        bits.append(f"trained {_fmt_ago(trained)}")
    iters = _adapter_iters()
    if iters is not None:
        bits.append(f"{iters} iters")
    detail = f" ({', '.join(bits)})" if bits else ""
    return f"loaded{detail} · {progress}"


def print_banner(config: dict[str, Any], adapter_loaded: bool, dataset_size: int,
                 output_fn=print):
    note_count = len(list(constants.NOTES_DIR.glob("*.md")))
    # Everything below is sized to the window. The command list used to be
    # four hand-maintained strings well over 200 characters each, which on
    # anything but a maximized terminal wrapped into an unreadable block — and
    # drifted out of date every time a command was added. It comes from the one
    # table now (chat_constants.BUILTIN_COMMANDS) and is wrapped to fit.
    from symbio.app.chat_constants import BUILTIN_COMMAND_NAMES

    width = term_width()
    output_fn("\n" + "=" * width)
    output_fn(f"  {config['assistant_name'].upper()} — PERSONAL CHAT-FINETUNE CLI")
    for label, value in (
            ("Model ", config["model_name"]),
            ("User  ", config["user_name"]),
            ("LoRA  ", adapter_status_value(config, adapter_loaded)),
            ("Data  ", f"{dataset_size:,} bytes"),
            ("Notes ", str(note_count)),
    ):
        # The model name is a full snapshot path and is the one line here that
        # reliably overflows a narrow window.
        for line in two_column(f"{label} :", str(value), indent=3, gap=1,
                               width=width):
            output_fn(line)
    output_fn("-" * width)
    for i, line in enumerate(wrapped_list(
            [f"/{n}" for n in BUILTIN_COMMAND_NAMES], indent=10, width=width)):
        output_fn(("Commands:" + line[9:]) if i == 0 else line)
    for line in two_column(
            "", "Type / on its own for the whole menu, with what each one "
                "does, or / and Tab to complete.", indent=2, gap=0, width=width):
        output_fn(line)
    for line in two_column(
            "", f"{config['assistant_name']} can also use <note>, <cmd>, <py>, "
                f"<digest />, <train />, <cron> by itself.",
            indent=2, gap=0, width=width):
        output_fn(line)
    output_fn("-" * width)
