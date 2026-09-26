"""What the cat is looking at: Symbio's state, read off disk and out of `ps`.

Three sources, none of them an import of the agent:

* The fine-tune in progress: LOG_DIR/training_live.json, written by
  symbio/app/training_live.py from the trainer's own output and the golden
  gate's verdict.
* Whether a model is loaded, and whether it is working: the daemon's pid file
  and socket (the same two checks symbio_desktop makes), or failing that a
  local `symb chat` holding its own copy. "Working" is that process's CPU time
  moving, sampled with `ps` — psutil is not a dependency.
* How much training the headmaster's adapter holds: adapters/
  training_progress.json, which training.record_adapter_iters keeps.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).parent.resolve()

# A process using more than this share of one core is working, not waiting on
# a socket. An idle daemon sits at 0.0; generating, it keeps most of a core.
BUSY_CPU = 0.2
# A run whose process died without saying so is shown fainted for this long,
# then forgotten. Not forever: the file stays behind until the next run.
FAINT_S = 600.0
# A gated run waits this long for its verdict before the pet stops waiting.
# The golden set on a 14B takes minutes, not an hour.
JUDGE_S = 1800.0
# How long a failed run keeps the slime looking hurt.
HURT_S = 60.0
# A verdict older than this when first seen is history, not an event.
VERDICT_FRESH_S = 15.0
# The file training.record_adapter_iters writes into each adapter directory.
PROGRESS_FILE = "training_progress.json"


def load_constants():
    """symbio/constants.py, loaded WITHOUT importing the symbio package.

    `from symbio import constants` runs symbio/__init__.py, which pulls in the
    agent, the chat loop and mlx behind it (see symbio_desktop/server.py).
    constants.py imports nothing but os, pathlib and typing.
    """
    import importlib.util

    path = APP_DIR.parent / "symbio" / "constants.py"
    spec = importlib.util.spec_from_file_location("_symbio_constants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Snapshot:
    """Everything the slime needs to know about one moment."""

    presence: str = "asleep"        # asleep | waking | awake
    busy: bool = False              # the model process is working
    activity: str = "idle"          # idle | training | judging | failed | fainted
    run: str | None = None          # which run the training fields belong to
    skill: str | None = None        # worker role of the run; None = headmaster
    iter: int = 0                   # steps done in this run
    iters: int | None = None        # steps budgeted
    prior_iters: int = 0            # steps the adapter had before this run
    total_iters: int | None = None  # the adapter's steps once the run ended
    train: list = field(default_factory=list)   # [[step, loss], ...]
    val: list = field(default_factory=list)
    verdict: str | None = None      # "kept" / "rolled_back", on the poll it happens
    verdict_skill: str | None = None
    adapter_iters: int = 0          # the headmaster adapter's steps, on disk
    has_adapter: bool = False
    reason: str = ""
    status: str = ""                # one line for the tooltip


def run_label(role: str | None) -> str:
    return role.replace("_", " ") if role else "headmaster"


def run_steps(snap: Snapshot) -> int:
    """How much training the core stands for while a run is on screen."""
    if snap.total_iters is not None:
        return snap.total_iters
    return snap.prior_iters + snap.iter


def status_line(snap: Snapshot) -> str:
    label = run_label(snap.skill)
    if snap.activity == "training":
        budget = f"/{snap.iters}" if snap.iters else ""
        text = f"Training the {label} adapter: step {snap.iter}{budget}"
        if snap.train:
            text += f" · loss {snap.train[-1][1]:.3f} (from {snap.train[0][1]:.3f})"
        return text
    if snap.activity == "judging":
        return f"Golden checks on the new {label} adapter…"
    if snap.activity == "failed":
        return f"The {label} run failed" + (f": {snap.reason}" if snap.reason else ".")
    if snap.activity == "fainted":
        return f"The {label} run's process died mid-run (killed or crashed)."
    if snap.presence == "asleep":
        return "Asleep — no model loaded. `symb daemon start` wakes me."
    if snap.presence == "waking":
        return "Waking up — the model is loading."
    if snap.busy:
        return "Thinking…"
    core = (f"adapter: {snap.adapter_iters} steps" if snap.has_adapter
            else "no adapter yet, base model only")
    return f"Awake ({core}). Double-click to chat."


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def boot_time() -> float:
    """When the machine booted: a pid recorded before it names nothing.

    `sysctl kern.boottime` prints "{ sec = 1790000000, usec = 123 } ...".
    """
    out = _run(["sysctl", "-n", "kern.boottime"])
    try:
        return float(out.split("sec =", 1)[1].split(",", 1)[0])
    except (IndexError, ValueError):
        return 0.0


def parse_cputime(text: str) -> float | None:
    """`ps -o cputime` as seconds: "0:01.52", "12:34.56", "1:02:03.45",
    "2-01:02:03.45"."""
    text = text.strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        try:
            days = int(head)
        except ValueError:
            return None
    total = 0.0
    try:
        for part in text.split(":"):
            total = total * 60 + float(part)
    except ValueError:
        return None
    return days * 86400 + total


def cpu_seconds(pid: int) -> float | None:
    return parse_cputime(_run(["ps", "-o", "cputime=", "-p", str(pid)]))


def local_chat_pid(ps_output: str) -> int | None:
    """A `symb chat` holding its own model, when no daemon is up.

    Matched on the argv the entry points actually produce: the ./symb wrapper
    runs `python3 -m symbio.app.cli ...`, and the installed console scripts
    run `python .../bin/symb ...` or `.../bin/symbio ...`. With no daemon,
    `chat` and a bare `symb` load the model in that process.
    """
    for row in ps_output.splitlines():
        row = row.strip()
        pid_text, _, command = row.partition(" ")
        argv = command.split()
        if len(argv) < 2 or "python" not in os.path.basename(argv[0]).lower():
            continue
        if "symbio.app.cli" in argv:
            rest = argv[argv.index("symbio.app.cli") + 1:]
        elif os.path.basename(argv[1]) in ("symb", "symbio"):
            rest = argv[2:]
        else:
            continue
        words = [a for a in rest if not a.startswith("-")]
        if (words[0] if words else "chat") == "chat":
            try:
                return int(pid_text)
            except ValueError:
                continue
    return None


class Feed:
    """Polls Symbio's state and hands the slime a Snapshot."""

    def __init__(self, constants=None, clock=time.time):
        self.c = constants if constants is not None else load_constants()
        self.clock = clock
        self.boot = boot_time()
        self._verdict_seen: float | None = None
        self._cpu: tuple[int, float, float] | None = None
        self._busy = False
        self._local_pid: int | None = None
        self._local_checked = -1e9
        self._adapter_checked = -1e9
        self._adapter = (False, 0)

    @property
    def live_path(self) -> Path:
        return self.c.LOG_DIR / getattr(self.c, "TRAINING_LIVE_NAME", "training_live.json")

    def _presence(self) -> tuple[str, int | None]:
        pid = _read_pid(self.c.DAEMON_PID_FILE)
        if pid is not None and pid_alive(pid):
            # The socket appears only once the model is loaded.
            return ("awake" if self.c.DAEMON_SOCKET.exists() else "waking"), pid
        now = time.monotonic()
        if now - self._local_checked > 6.0:
            self._local_checked = now
            self._local_pid = local_chat_pid(_run(["ps", "-Ao", "pid=,command="]))
        if self._local_pid is not None and pid_alive(self._local_pid):
            return "awake", self._local_pid
        return "asleep", None

    def _is_busy(self, pid: int | None) -> bool:
        if pid is None:
            self._cpu = None
            return False
        seconds = cpu_seconds(pid)
        now = time.monotonic()
        previous, self._cpu = self._cpu, ((pid, seconds, now) if seconds is not None else None)
        if seconds is not None and previous and previous[0] == pid and now > previous[2]:
            self._busy = (seconds - previous[1]) / (now - previous[2]) > BUSY_CPU
        return self._busy

    def _adapter_state(self, force: bool = False) -> tuple[bool, int]:
        now = time.monotonic()
        if force or now - self._adapter_checked > 5.0:
            self._adapter_checked = now
            directory = self.c.ADAPTER_DIR
            has = any((directory / name).exists()
                      for name in ("adapters.safetensors", "adapter_model.safetensors"))
            progress = _read_json(directory / PROGRESS_FILE) or {}
            try:
                steps = int(progress.get("total_iters", 0))
            except (TypeError, ValueError):
                steps = 0
            self._adapter = (has, steps if has else 0)
        return self._adapter

    def poll(self) -> Snapshot:
        now = self.clock()
        presence, pid = self._presence()
        snap = Snapshot(presence=presence,
                        busy=presence != "asleep" and self._is_busy(pid))

        live = _read_json(self.live_path)
        if live:
            self._read_run(live, snap, now)
        snap.has_adapter, snap.adapter_iters = self._adapter_state(force=bool(snap.verdict))
        snap.status = status_line(snap)
        return snap

    def _read_run(self, live: dict[str, Any], snap: Snapshot, now: float) -> None:
        phase = live.get("phase")
        started = float(live.get("started_at") or 0)
        ended = float(live.get("ended_at") or 0)
        verdict = live.get("verdict")
        # The pid alone lies after a reboot: the kernel hands low pids out
        # again, so a dead trainer's pid is soon somebody else's.
        owner_alive = pid_alive(live.get("pid")) and started >= self.boot - 1

        activity = "idle"
        if not verdict:
            if phase == "training":
                if owner_alive:
                    activity = "training"
                elif now - float(live.get("updated_at") or started) < FAINT_S:
                    activity = "fainted"
            elif phase == "trained" and live.get("gated"):
                if owner_alive and now - ended < JUDGE_S:
                    activity = "judging"
            elif phase == "failed" and now - ended < HURT_S:
                activity = "failed"

        verdict_at = live.get("verdict_at")
        if verdict and verdict_at and verdict_at != self._verdict_seen:
            self._verdict_seen = verdict_at
            if now - float(verdict_at) < VERDICT_FRESH_S:
                snap.verdict = verdict
                snap.verdict_skill = live.get("verdict_role")

        snap.activity = activity
        snap.run = live.get("run")
        snap.skill = live.get("role")
        snap.iter = int(live.get("iter") or 0)
        snap.iters = live.get("iters")
        snap.prior_iters = int(live.get("prior_iters") or 0)
        snap.total_iters = live.get("total_iters")
        snap.train = list(live.get("train") or [])
        snap.val = list(live.get("val") or [])
        snap.reason = str(live.get("reason") or "")
