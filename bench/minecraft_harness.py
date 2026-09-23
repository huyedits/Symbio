"""minecraft_harness.py — can the desktop harness drive Minecraft with nobody there?

Written after the first unattended attempt, 2026-09-23. The launch click
landed, and then three things stopped the run, none of them about playing:

  1. the game's JVM crashed about 4s into loading (a JIT compiler crash; the
     only trace is replay_pid<N>.log — no crash-reports/ entry, and
     latest.log just stops mid-startup);
  2. nothing noticed, so the run waited on a game that no longer existed;
  3. by the time anyone looked the session was locked, every screenshot was
     solid black, and OCR on a black frame answers "nothing there".

So before any camera or movement rung, the harness has to answer: can I run
right now without a person, and if not, which thing needs the person? That is
`preflight()`. Each check says who can fix it: "harness" means the harness
fixes it itself (keep the display awake, launch the game); "human" means it
cannot (unlock the screen, grant a permission).

What the game shows is read from the F3 debug overlay with Apple's OCR —
position, facing, target block — so every rung is graded off the game's own
numbers, not a model's description of a screenshot.

Usage:
    venv/bin/python bench/minecraft_harness.py preflight
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MC_DIR = Path.home() / "Library" / "Application Support" / "minecraft"
LAUNCHER = Path("/Applications/Feather Launcher.app")

# Games that take the same mouse and keyboard. Injected input goes to
# whatever is frontmost; one of these coming forward mid-run turns a camera
# test into clicks in someone else's game. Not a blocker on its own:
# post_to_game() refuses to send anything unless the game is frontmost.
INPUT_RIVALS = ("RobloxPlayer", "GeometryDash", "HD-Player")

# macOS virtual keycodes for the keys the rungs need.
KEY = {"w": 13, "a": 0, "s": 1, "d": 2, "space": 49, "shift": 56,
       "e": 14, "esc": 53, "f3": 99, "t": 17, "enter": 36}


# ------------------------------------------------------------------ checks

@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    who: str = "human"      # who can make it ok: "harness" or "human"


def autonomous(checks: list[Check]) -> bool:
    """True when nothing failing needs a person."""
    return all(c.ok or c.who == "harness" for c in checks)


def screen_locked() -> bool:
    import Quartz

    session = Quartz.CGSessionCopyCurrentDictionary() or {}
    return bool(session.get("CGSSessionScreenIsLocked", False))


def display_sleep_minutes(pmset_output: str | None = None) -> int:
    """Minutes of idle before the display sleeps; 0 = never."""
    if pmset_output is None:
        pmset_output = subprocess.run(["pmset", "-g"], capture_output=True,
                                      text=True).stdout
    m = re.search(r"^\s*displaysleep\s+(\d+)", pmset_output, re.MULTILINE)
    return int(m.group(1)) if m else 0


def running_commands() -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                         text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def game_pid(commands: list[str] | None = None) -> int | None:
    """The running game's JVM, told apart from the launcher's own helpers.

    Feather's helpers are Electron processes that also mention "minecraft"
    in their arguments; the game is the java process running a Minecraft
    main class.
    """
    for line in commands if commands is not None else running_commands():
        pid, _, cmd = line.partition(" ")
        # Not cmd.split()[0]: the bundled JRE lives under "Application
        # Support", so the executable path itself contains a space.
        if not re.search(r"(^|/)java(\s|$)", cmd):
            continue
        if "minecraft" in cmd.lower() or "net.fabricmc" in cmd:
            return int(pid)
    return None


def rivals(commands: list[str] | None = None) -> list[str]:
    found = []
    for line in commands if commands is not None else running_commands():
        for name in INPUT_RIVALS:
            if name in line and name not in found:
                found.append(name)
    return found


def preflight(probe: dict[str, Callable[[], Any]] | None = None) -> list[Check]:
    """Everything that decides whether a run can start and finish unattended.

    `probe` overrides any of the live readings, for tests:
    locked, screen_recording, capture_blank, accessibility, launcher,
    display_sleep, caffeinated, game_pid, rivals.
    """
    live: dict[str, Callable[[], Any]] = {
        "locked": screen_locked,
        "screen_recording": _screen_recording,
        "capture_blank": _capture_blank,
        "accessibility": _accessibility,
        "launcher": LAUNCHER.exists,
        "display_sleep": display_sleep_minutes,
        "caffeinated": _display_held_awake,
        "game_pid": game_pid,
        "rivals": rivals,
    }
    live.update(probe or {})
    read = {k: (lambda f=f: f()) for k, f in live.items()}

    checks = []
    locked = read["locked"]()
    checks.append(Check(
        "session unlocked", not locked,
        "screen is locked: every capture is black and input goes nowhere"
        if locked else "unlocked"))

    recording = read["screen_recording"]()
    checks.append(Check(
        "screen recording permission", recording,
        "granted" if recording else
        "not granted to this terminal: System Settings > Privacy > Screen Recording"))

    # Permission granted and unlocked, and the frame is still black: then
    # something else is wrong, and OCR would read "nothing" off it.
    if recording and not locked:
        blank = read["capture_blank"]()
        checks.append(Check(
            "capture shows the screen", not blank,
            "a full-screen capture came back blank" if blank else "capture has content"))

    trusted = read["accessibility"]()
    checks.append(Check(
        "accessibility permission", trusted,
        "granted" if trusted else
        "not granted: injected keys and mouse moves are dropped silently"))

    checks.append(Check(
        "launcher installed", bool(read["launcher"]()), str(LAUNCHER)))

    minutes = read["display_sleep"]()
    held = read["caffeinated"]()
    safe = minutes == 0 or held
    checks.append(Check(
        "display stays awake", safe,
        "held awake" if held else
        ("never sleeps" if minutes == 0 else
         f"display sleeps after {minutes} min idle; a run outlasting that "
         f"can end at a lock screen — hold it with caffeinate -d"),
        who="harness"))

    pid = read["game_pid"]()
    checks.append(Check(
        "game running", pid is not None,
        f"pid {pid}" if pid else "not running — the harness launches it",
        who="harness"))

    others = read["rivals"]()
    checks.append(Check(
        "no rival for the input", not others,
        "none" if not others else
        f"{', '.join(others)} running: input is only sent while the game is "
        f"frontmost, so a rival coming forward pauses the run, not hijacks it",
        who="harness"))
    return checks


def _screen_recording() -> bool:
    import Quartz

    return bool(Quartz.CGPreflightScreenCaptureAccess())


def _capture_blank() -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from symbio import computer

    return computer.screenshot_is_blank(capture())


def _accessibility() -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from symbio import ax

    return ax.trusted()


def _display_held_awake() -> bool:
    out = subprocess.run(["pmset", "-g", "assertions"], capture_output=True,
                         text=True).stdout
    m = re.search(r"^\s*PreventUserIdleDisplaySleep\s+(\d+)", out, re.MULTILINE)
    return bool(m and int(m.group(1)))


def hold_display_awake() -> subprocess.Popen:
    """Keep the display (and so the lock screen) away for this run."""
    return subprocess.Popen(["caffeinate", "-d", "-i"])


# ------------------------------------------------------------------ crashes

CRASH_PATTERNS = ("crash-reports/*.txt", "hs_err_pid*.log", "replay_pid*.log")


def detect_crash(since: float, mc_dir: Path = MC_DIR) -> str | None:
    """The newest sign the game died after `since` (epoch seconds), or None.

    replay_pid is on the list because it was the ONLY trace of the real
    crash: a HotSpot JIT failure writes it next to the working directory and
    leaves crash-reports/ empty, since the game never got far enough to
    write its own report.
    """
    hits = []
    for pattern in CRASH_PATTERNS:
        for path in mc_dir.glob(pattern):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= since:
                hits.append((mtime, path))
    if not hits:
        return None
    mtime, path = max(hits)
    kind = ("JVM JIT compiler crash" if path.name.startswith("replay_pid")
            else "JVM crash" if path.name.startswith("hs_err_pid")
            else "game crash report")
    return f"{kind}: {path.name} at {time.strftime('%H:%M:%S', time.localtime(mtime))}"


def watch_launch(since: float, timeout: float = 120.0, poll: float = 2.0,
                 pid_fn: Callable[[], int | None] = game_pid,
                 crash_fn: Callable[[float], str | None] = detect_crash,
                 window_fn: Callable[[int], bool] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> tuple[str, str]:
    """After a launch click: ("ready", pid) | ("crashed", why) | ("timeout", why).

    A crash is checked on every poll, not only when the process vanishes: the
    JVM can write its crash files and linger for seconds while it unwinds.
    """
    window_fn = window_fn or game_window_up
    seen_pid = None
    waited = 0.0
    while waited <= timeout:
        why = crash_fn(since)
        if why:
            return "crashed", why
        pid = pid_fn()
        if pid:
            seen_pid = pid
            if window_fn(pid):
                return "ready", str(pid)
        elif seen_pid:
            return "crashed", f"pid {seen_pid} exited with no crash file"
        sleep(poll)
        waited += poll
    return "timeout", ("game process never started — the launch click missed, "
                       "or the launcher is asking for something"
                       if not seen_pid else f"pid {seen_pid} never opened a window")


def game_window_up(pid: int) -> bool:
    import Quartz

    for w in Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID) or []:
        if w.get("kCGWindowOwnerPID") == pid and w.get("kCGWindowLayer") == 0:
            bounds = w.get("kCGWindowBounds") or {}
            if bounds.get("Width", 0) > 200 and bounds.get("Height", 0) > 200:
                return True
    return False


# ------------------------------------------------------------------ reading F3

_NUM = r"(-?\d+(?:[.,]\d+)?)"
_SEP = r"\s*[/|Il1]\s*"     # OCR reads the slash as / | I l, occasionally 1
_XYZ = re.compile(rf"XYZ[:;]?\s*{_NUM}{_SEP}{_NUM}{_SEP}{_NUM}", re.IGNORECASE)
_FACING = re.compile(
    rf"Facing[:;]?\s*(north|south|east|west)?.*?\(\s*{_NUM}{_SEP}{_NUM}\s*\)\s*$",
    re.IGNORECASE)
_BLOCK = re.compile(rf"Block[:;]?\s*{_NUM}\s+{_NUM}\s+{_NUM}", re.IGNORECASE)


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def parse_f3(lines: list[str]) -> dict[str, Any]:
    """Position, facing and block from F3 overlay lines, as OCR reads them.

    Keys are present only when read: a missing key means "not on screen",
    which a rung must treat as "cannot grade", never as zero.
    """
    out: dict[str, Any] = {}
    for line in lines:
        text = line.strip()
        if "xyz" not in out and (m := _XYZ.search(text)):
            out["xyz"] = (_f(m.group(1)), _f(m.group(2)), _f(m.group(3)))
        elif "facing" not in out and (m := _FACING.search(text)):
            out["facing"] = (m.group(1) or "").lower() or None
            out["yaw"], out["pitch"] = _f(m.group(2)), _f(m.group(3))
        elif "block" not in out and (m := _BLOCK.search(text)):
            out["block"] = (int(_f(m.group(1))), int(_f(m.group(2))), int(_f(m.group(3))))
    return out


def read_lines(image_path: str | Path) -> list[str]:
    """Whole OCR lines. Not symbio.ocr.read_text: it splits at "/", the
    separator inside every F3 number triple."""
    import Vision
    from Foundation import NSURL

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(str(image_path)), None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(0)
    request.setUsesLanguageCorrection_(False)
    handler.performRequests_error_([request], None)
    return [str(o.topCandidates_(1)[0].string())
            for o in (request.results() or []) if o.topCandidates_(1)]


def capture(path: str | Path = "/tmp/mc_harness.png") -> Path:
    subprocess.run(["screencapture", "-x", str(path)], check=True)
    return Path(path)


def turned(before: float, after: float) -> float:
    """Signed yaw change in degrees, across the ±180 seam."""
    return ((after - before + 180.0) % 360.0) - 180.0


# ------------------------------------------------------------------ input

def mouse_delta_event(dx: int, dy: int, at: tuple[float, float],
                      move_position: bool = False):
    """A mouse-moved event carrying a raw delta.

    With the cursor captured (in-world), the game ignores the cursor
    position and reads the delta — so a pyautogui move, which only sets the
    position, may do nothing. Whether the delta fields alone are enough is
    exactly what rung 1 measures on the live game.
    """
    import Quartz

    x, y = at
    if move_position:
        x, y = x + dx, y + dy
    event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved,
                                           (x, y), 0)
    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventDeltaX, int(dx))
    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventDeltaY, int(dy))
    return event


def key_event(key: str, down: bool):
    import Quartz

    return Quartz.CGEventCreateKeyboardEvent(None, KEY[key], down)


def post(event) -> None:
    import Quartz

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def frontmost_pid() -> int | None:
    from AppKit import NSWorkspace

    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return int(app.processIdentifier()) if app else None


class NotFrontmost(RuntimeError):
    pass


def post_to_game(event, pid: int,
                 front: Callable[[], int | None] = frontmost_pid) -> None:
    """Post only while the game is the frontmost app.

    Checked on every event, not once per rung: a notification, a launcher
    popping back up, or Roblox coming forward mid-hold would otherwise take
    the rest of the keystrokes.
    """
    if front() != pid:
        raise NotFrontmost(f"frontmost app is pid {front()}, not the game ({pid})")
    post(event)


def hold_key(key: str, seconds: float, pid: int,
             front: Callable[[], int | None] = frontmost_pid,
             send: Callable[[Any, int], None] | None = None) -> None:
    """Movement needs a held key, not a press: a tap moves a fraction of a block.

    The key-up is sent even if focus was lost mid-hold — to whatever has
    focus, since a key left down is worse than a stray key-up.
    """
    send = send or (lambda e, p: post_to_game(e, p, front))
    send(key_event(key, True), pid)
    try:
        time.sleep(seconds)
    finally:
        post(key_event(key, False))


# ------------------------------------------------------------------ cli

def main(argv: list[str]) -> int:
    if not argv or argv[0] != "preflight":
        print(__doc__)
        return 2
    checks = preflight()
    for c in checks:
        mark = "ok  " if c.ok else ("FIX " if c.who == "harness" else "NEED")
        print(f"  {mark} {c.name:30} {c.detail}")
    if autonomous(checks):
        print("\ncan run unattended: YES")
        return 0
    needs = [c.name for c in checks if not c.ok and c.who == "human"]
    print(f"\ncan run unattended: NO — needs a person for: {', '.join(needs)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
