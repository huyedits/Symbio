"""Terminal presentation for the chat loop: the spinner, the rainbow banner,
adapter/learning status lines, the per-session log handler, and the health
report writer.

Kept apart from the session logic so that changing how something is displayed
never means opening the file that decides what happens.
"""

import json
import logging
import os
import sys
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
    if count >= threshold:
        return f"{count}/{threshold} mistakes — tuning due{suffix}"
    return f"{count}/{threshold} mistakes to next tune{suffix}"


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
    output_fn("\n" + "=" * 50)
    output_fn(f"  {config['assistant_name'].upper()} — PERSONAL CHAT-FINETUNE CLI")
    output_fn(f"   Model  : {config['model_name']}")
    output_fn(f"   User   : {config['user_name']}")
    output_fn(f"   LoRA   : {adapter_status_value(config, adapter_loaded)}")
    output_fn(f"   Data   : {dataset_size:,} bytes")
    output_fn(f"   Notes  : {note_count}")
    output_fn("-" * 50)
    output_fn("Commands: /quit  /save  /train  /retrain  /train_worker  /resume  /golden [audit|prune]  /security  /learn  /forget_last  /status  /think  /backup  /restore-adapter  /prune  /selfcheck  /setup  /compact  /standing  /help")
    output_fn("         /run <cmd>  /note [title]  /notes  /index-notes [--force]  /auto-index on|off  /new-skill <name> | <steps>  /skills  /skill-adapters  /digest  /cron  /config  /archive  /restore")
    output_fn("         /build-mcp <name> | <description>  /mcp-tools  /hosts  /telemetry on|off  /feedback <text>")
    output_fn("  (Caine can also use <note>, <cmd>, <py>, <digest />, <train />, <cron> by itself)")
    output_fn("-" * 50)
