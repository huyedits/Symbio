"""Symbio's own screen: a display the harness works on, so the user's stays theirs.

Every desktop tool used to act on the one screen there was -- the user's. A
look captured what they were looking at, open_app pulled an app in front of
their work, and a click or a keystroke went through their pointer and their
keyboard focus. Two people cannot share one mouse, and the Mac has one.

So Symbio gets a screen of its own. macOS can add a display with no panel
behind it: a private CoreGraphics class, CGVirtualDisplay, held by the small
native helper in symbio/native/desk_helper.m. The window server lays it out,
draws on it and captures it like any monitor, and nobody is looking at it.
With `desk.enabled` on:

  * open_app launches in the background and moves the app's windows onto the
    desk, so nothing takes the user's screen or focus
  * see_screen reads the desk's front window from the accessibility tree, and
    a capture is of the desk alone -- the user's screen is never in it
  * clicks and typing go through the accessibility API first, which needs no
    pointer and no focus. Real input events are the fallback: posted to the
    app itself, which leaves the pointer alone, and only then borrowed from
    the user -- and only once they have left the Mac alone for
    desk.borrow_input_after_idle_s, with the pointer and the front app put
    back afterwards
  * the automated browser opens its window on the desk as well

Measured on macOS 26.5 (2026-09-30); worth knowing before debugging any of it:

  * while the Mac's own panel is asleep the window server defers every
    display unplug, so a desk that was stopped stays online until the screen
    next wakes, and a new desk cannot reuse its serial in the meantime (the
    helper picks a free one). Such a leftover is never used: it vanishes on
    wake and would take any window on it back to the user's screen.
  * while the Mac is LOCKED the desk is still drawn and can be captured, but
    every app's windows drop out of the accessibility tree (the app answers
    as its own window) and out of captures, and displays cannot be
    rearranged (CGCompleteDisplayConfiguration returns 1014). The browser
    keeps working -- Playwright drives Chrome through its own protocol -- and
    the rest says the Mac is locked rather than failing strangely.

ctypes against the system frameworks, as in ax.py: pyobjc is not a
dependency, and this ships to other people's Macs.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from symbio import constants

SOURCE = Path(__file__).resolve().parent / "native" / "desk_helper.m"
# Read back through CGDisplayVendorNumber/ModelNumber: how a desk is told
# apart from a real monitor. Must match the helper.
VENDOR = 0x5359
PRODUCT = 0x4442
DEFAULT_NAME = "Symbio Desk"
DEFAULT_SIZE = (1440, 900)
# How long start() waits for the helper to name its display. Creating one
# takes well under a second; a first build of the helper happens before this.
_START_WAIT_S = 8.0
# macOS draws a menu bar on every display and keeps windows below it. 30 is
# what it measured on the desk (a window asked for y=0 landed at y=30).
MENU_BAR = 30
# Where a moved window's top-left goes, inside the menu bar and the edges.
_MARGIN = 24

_LOAD_ERROR = ""
try:
    _cg = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreGraphics.framework/Versions/A/CoreGraphics")
    _cf = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation")
except Exception as e:  # pragma: no cover - not macOS
    _cg = _cf = None
    _LOAD_ERROR = str(e)


class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class CGRect(ctypes.Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]


# Every signature spelled out: without them ctypes passes pointers as 32-bit
# ints and structs in the wrong registers, and the process dies instead of
# getting an error back. See ax.py for the same rule.
if _cg is not None:
    _u32, _i32, _vp = ctypes.c_uint32, ctypes.c_int32, ctypes.c_void_p
    for _name in ("CGGetOnlineDisplayList", "CGGetActiveDisplayList"):
        _fn = getattr(_cg, _name)
        _fn.restype = _i32
        _fn.argtypes = [_u32, ctypes.POINTER(_u32), ctypes.POINTER(_u32)]
    _cg.CGDisplayBounds.restype = CGRect
    _cg.CGDisplayBounds.argtypes = [_u32]
    for _name in ("CGDisplayVendorNumber", "CGDisplayModelNumber",
                  "CGDisplaySerialNumber", "CGDisplayIsAsleep"):
        _fn = getattr(_cg, _name)
        _fn.restype = _u32
        _fn.argtypes = [_u32]
    _cg.CGMainDisplayID.restype = _u32
    _cg.CGMainDisplayID.argtypes = []
    for _name in ("CGDisplayPixelsWide", "CGDisplayPixelsHigh"):
        _fn = getattr(_cg, _name)
        _fn.restype = ctypes.c_size_t
        _fn.argtypes = [_u32]
    _cg.CGBeginDisplayConfiguration.restype = _i32
    _cg.CGBeginDisplayConfiguration.argtypes = [ctypes.POINTER(_vp)]
    _cg.CGConfigureDisplayOrigin.restype = _i32
    _cg.CGConfigureDisplayOrigin.argtypes = [_vp, _u32, _i32, _i32]
    _cg.CGCompleteDisplayConfiguration.restype = _i32
    _cg.CGCompleteDisplayConfiguration.argtypes = [_vp, _u32]
    _cg.CGCancelDisplayConfiguration.restype = _i32
    _cg.CGCancelDisplayConfiguration.argtypes = [_vp]
    _cg.CGWindowListCopyWindowInfo.restype = _vp
    _cg.CGWindowListCopyWindowInfo.argtypes = [_u32, _u32]
    _cg.CGRectMakeWithDictionaryRepresentation.restype = ctypes.c_bool
    _cg.CGRectMakeWithDictionaryRepresentation.argtypes = [_vp, ctypes.POINTER(CGRect)]
    _cg.CGSessionCopyCurrentDictionary.restype = _vp
    _cg.CGSessionCopyCurrentDictionary.argtypes = []
    _cg.CGEventSourceSecondsSinceLastEventType.restype = ctypes.c_double
    _cg.CGEventSourceSecondsSinceLastEventType.argtypes = [_i32, _u32]
    _cg.CGEventCreate.restype = _vp
    _cg.CGEventCreate.argtypes = [_vp]
    _cg.CGEventGetLocation.restype = CGPoint
    _cg.CGEventGetLocation.argtypes = [_vp]
    _cg.CGEventCreateMouseEvent.restype = _vp
    _cg.CGEventCreateMouseEvent.argtypes = [_vp, _u32, CGPoint, _u32]
    _cg.CGEventCreateKeyboardEvent.restype = _vp
    _cg.CGEventCreateKeyboardEvent.argtypes = [_vp, ctypes.c_uint16, ctypes.c_bool]
    _cg.CGEventKeyboardSetUnicodeString.restype = None
    _cg.CGEventKeyboardSetUnicodeString.argtypes = [_vp, ctypes.c_ulong,
                                                    ctypes.POINTER(ctypes.c_uint16)]
    _cg.CGEventSetFlags.restype = None
    _cg.CGEventSetFlags.argtypes = [_vp, ctypes.c_uint64]
    _cg.CGEventSetIntegerValueField.restype = None
    _cg.CGEventSetIntegerValueField.argtypes = [_vp, _u32, ctypes.c_int64]
    # The non-variadic form: ctypes cannot pass variadic arguments the way
    # the arm64 ABI wants them, so CGEventCreateScrollWheelEvent is out.
    _cg.CGEventCreateScrollWheelEvent2.restype = _vp
    _cg.CGEventCreateScrollWheelEvent2.argtypes = [_vp, _u32, _u32, _i32, _i32, _i32]
    _cg.CGEventSetLocation.restype = None
    _cg.CGEventSetLocation.argtypes = [_vp, CGPoint]
    _cg.CGEventPostToPid.restype = None
    _cg.CGEventPostToPid.argtypes = [_i32, _vp]
    _cg.CGEventPost.restype = None
    _cg.CGEventPost.argtypes = [_u32, _vp]
    _cg.CGWarpMouseCursorPosition.restype = _i32
    _cg.CGWarpMouseCursorPosition.argtypes = [CGPoint]
    _cg.CGAssociateMouseAndMouseCursorPosition.restype = _i32
    _cg.CGAssociateMouseAndMouseCursorPosition.argtypes = [_u32]

    _cf.CFRelease.argtypes = [_vp]
    _cf.CFArrayGetCount.restype = ctypes.c_long
    _cf.CFArrayGetCount.argtypes = [_vp]
    _cf.CFArrayGetValueAtIndex.restype = _vp
    _cf.CFArrayGetValueAtIndex.argtypes = [_vp, ctypes.c_long]
    _cf.CFDictionaryGetValue.restype = _vp
    _cf.CFDictionaryGetValue.argtypes = [_vp, _vp]
    _cf.CFStringCreateWithCString.restype = _vp
    _cf.CFStringCreateWithCString.argtypes = [_vp, ctypes.c_char_p, _u32]
    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFStringGetCString.argtypes = [_vp, ctypes.c_char_p, ctypes.c_long, _u32]
    _cf.CFNumberGetValue.restype = ctypes.c_bool
    _cf.CFNumberGetValue.argtypes = [_vp, ctypes.c_long, _vp]
    _cf.CFBooleanGetValue.restype = ctypes.c_bool
    _cf.CFBooleanGetValue.argtypes = [_vp]
    _cf.CFGetTypeID.restype = ctypes.c_ulong
    _cf.CFGetTypeID.argtypes = [_vp]
    for _name in ("CFStringGetTypeID", "CFBooleanGetTypeID", "CFNumberGetTypeID"):
        getattr(_cf, _name).restype = ctypes.c_ulong

_UTF8 = 0x08000100
_CFNUMBER_SINT64 = 4
_CFNUMBER_FLOAT64 = 6
# CGWindowList option bits: on-screen windows only, not the desktop picture.
_ON_SCREEN_ONLY = 1
_EXCLUDE_DESKTOP = 16
_CONFIGURE_FOR_SESSION = 1
_HID_SYSTEM_STATE = 1
_ANY_INPUT_EVENT = 0xFFFFFFFF
_HID_EVENT_TAP = 0
_MOUSE_MOVED = 5
_CLICK_STATE_FIELD = 1
# Which window a mouse event is for. The window server fills these in for a
# real event; one posted straight to an app has to carry them itself.
_WINDOW_UNDER_POINTER_FIELD = 91
_WINDOW_THAT_CAN_HANDLE_FIELD = 92
# (down, up, dragged) event types and the button number, per button.
_MOUSE = {"left": (1, 2, 6, 0), "right": (3, 4, 7, 1), "middle": (25, 26, 27, 2)}


class DeskError(RuntimeError):
    """The desk cannot do this, with the reason in words a person can act on."""


# ---------- CoreFoundation plumbing ----------

_KEYS: dict[str, int] = {}


def _key(name: str) -> int:
    """A CFString for a dictionary key, made once and kept."""
    ref = _KEYS.get(name)
    if ref is None:
        ref = _KEYS[name] = _cf.CFStringCreateWithCString(None, name.encode("utf-8"), _UTF8)
    return ref


def _number(dictionary: int, key: str, floating: bool = False) -> float | int | None:
    ref = _cf.CFDictionaryGetValue(dictionary, _key(key))
    if not ref:
        return None
    if floating:
        out = ctypes.c_double()
        ok = _cf.CFNumberGetValue(ref, _CFNUMBER_FLOAT64, ctypes.byref(out))
    else:
        out = ctypes.c_int64()
        ok = _cf.CFNumberGetValue(ref, _CFNUMBER_SINT64, ctypes.byref(out))
    return out.value if ok else None


def _string(dictionary: int, key: str) -> str:
    ref = _cf.CFDictionaryGetValue(dictionary, _key(key))
    if not ref or _cf.CFGetTypeID(ref) != _cf.CFStringGetTypeID():
        return ""
    buf = ctypes.create_string_buffer(1024)
    if _cf.CFStringGetCString(ref, buf, 1024, _UTF8):
        return buf.value.decode("utf-8", "replace")
    return ""


def _rect(dictionary: int, key: str) -> tuple[int, int, int, int] | None:
    ref = _cf.CFDictionaryGetValue(dictionary, _key(key))
    rect = CGRect()
    if not ref or not _cg.CGRectMakeWithDictionaryRepresentation(ref, ctypes.byref(rect)):
        return None
    return (int(rect.origin.x), int(rect.origin.y),
            int(rect.size.width), int(rect.size.height))


# ---------- displays ----------

@dataclass(frozen=True)
class Display:
    id: int
    x: int
    y: int
    width: int
    height: int
    desk: bool
    main: bool
    asleep: bool

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def available() -> bool:
    return _cg is not None and sys.platform == "darwin"


def _display_ids(active: bool = False) -> list[int]:
    if _cg is None:
        return []
    ids = (ctypes.c_uint32 * 64)()
    count = ctypes.c_uint32()
    fn = _cg.CGGetActiveDisplayList if active else _cg.CGGetOnlineDisplayList
    if fn(64, ids, ctypes.byref(count)) != 0:
        return []
    return [int(ids[i]) for i in range(count.value)]


def _bounds(display: int) -> tuple[int, int, int, int]:
    r = _cg.CGDisplayBounds(display)
    return (int(r.origin.x), int(r.origin.y), int(r.size.width), int(r.size.height))


def _is_desk_display(display: int) -> bool:
    return (_cg.CGDisplayVendorNumber(display) == VENDOR
            and _cg.CGDisplayModelNumber(display) == PRODUCT)


def displays(active: bool = False) -> list[Display]:
    """Every display the window server knows, desks included."""
    if _cg is None:
        return []
    main = int(_cg.CGMainDisplayID())
    out = []
    for display in _display_ids(active):
        x, y, w, h = _bounds(display)
        out.append(Display(display, x, y, w, h, _is_desk_display(display),
                           display == main, bool(_cg.CGDisplayIsAsleep(display))))
    return out


# ---------- the desk ----------

@dataclass
class Desk:
    """The running desk: which display it is and where it sits, in points."""

    display: int
    pid: int
    serial: int
    x: int
    y: int
    width: int
    height: int
    name: str = DEFAULT_NAME

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def to_global(self, x: float, y: float, scale: float = 1.0) -> tuple[int, int]:
        """A point read off a capture of the desk, in the space windows live in."""
        scale = scale or 1.0
        return (int(round(self.x + x / scale)), int(round(self.y + y / scale)))

    def to_local(self, x: float, y: float) -> tuple[int, int]:
        """A point in the window server's space, as it sits on a desk capture."""
        return (int(round(x - self.x)), int(round(y - self.y)))


def enabled(config: dict[str, Any] | None) -> bool:
    return bool(((config or {}).get("desk") or {}).get("enabled"))


def _read_state() -> dict[str, Any] | None:
    try:
        state = json.loads(constants.DESK_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


# Pids already checked to be a desk helper, and when. A pid number can be
# reused by another process only after the helper has exited, which the
# kill(0) below still sees at once; the `ps` that tells the two apart is what
# is cached, since every desktop action asks for the desk.
_VERIFIED: dict[int, float] = {}
_VERIFY_TTL_S = 10.0


def _helper_alive(pid: int) -> bool:
    """Is `pid` running, and is it a desk helper rather than a reused number?"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _VERIFIED.pop(pid, None)
        return False
    except PermissionError:
        pass
    now = time.monotonic()
    if now - _VERIFIED.get(pid, -_VERIFY_TTL_S) < _VERIFY_TTL_S:
        return True
    try:
        command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    if "symbio-desk" not in command:
        return False
    _VERIFIED[pid] = now
    return True


def current() -> Desk | None:
    """The desk this install's helper is holding, or None.

    Only the display the helper named counts. A desk-looking display that no
    running helper owns is a stopped desk whose unplug macOS is deferring
    while its panel sleeps; it goes the moment the screen wakes, taking any
    window put on it back to the user's screen.
    """
    if _cg is None:
        return None
    state = _read_state()
    if not state or not state.get("ok"):
        return None
    try:
        pid, display = int(state.get("pid") or 0), int(state.get("display") or 0)
    except (TypeError, ValueError):
        return None
    if not display or not _helper_alive(pid):
        return None
    if display not in _display_ids() or not _is_desk_display(display):
        return None
    x, y, w, h = _bounds(display)
    return Desk(display=display, pid=pid, serial=int(state.get("serial") or 0),
                x=x, y=y, width=w, height=h,
                name=str(state.get("name") or DEFAULT_NAME))


def leftovers() -> list[Display]:
    """Desk displays no running helper holds: stopped while the screen slept."""
    desk = current()
    return [d for d in displays() if d.desk and (desk is None or d.id != desk.display)]


# ---------- the helper process ----------

_build_error = ""


def helper_binary() -> Path | None:
    """The compiled helper, built from source when the source changed."""
    global _build_error
    if sys.platform != "darwin":
        _build_error = "Symbio's desk needs macOS."
        return None
    if not SOURCE.exists():
        _build_error = f"The desk helper's source is missing: {SOURCE}"
        return None
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
    target = constants.CACHE_DIR / f"symbio-desk-{digest}"
    if target.exists():
        return target
    clang = shutil.which("clang")
    if not clang:
        _build_error = ("clang was not found. Install the Xcode command-line "
                        "tools (xcode-select --install) and try again.")
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".building")
    try:
        done = subprocess.run(
            [clang, "-fobjc-arc", "-O2", "-framework", "Foundation",
             "-framework", "CoreGraphics", str(SOURCE), "-o", str(partial)],
            capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        _build_error = f"Building the desk helper failed: {e}"
        return None
    if done.returncode != 0 or not partial.exists():
        _build_error = ("Building the desk helper failed: "
                        + (done.stderr or done.stdout).strip()[-400:])
        return None
    os.replace(partial, target)
    for old in target.parent.glob("symbio-desk-*"):
        # Older builds only. A running helper keeps its inode; only the name goes.
        if old != target and re.fullmatch(r"symbio-desk-[0-9a-f]{12}", old.name):
            try:
                old.unlink()
            except OSError:
                pass
    return target


@contextmanager
def _start_lock() -> Iterator[None]:
    """One start at a time across processes: the daemon, the CLI, a test."""
    import fcntl

    path = constants.CACHE_DIR / "desk.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _helper_error(log_path: Path) -> str:
    """The reason the helper gave for leaving, from the last line it wrote."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-20:]):
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if isinstance(message, dict) and message.get("error"):
            return f"The desk could not start: {message['error']}."
    return ""


def start(width: int = DEFAULT_SIZE[0], height: int = DEFAULT_SIZE[1],
          name: str = DEFAULT_NAME) -> Desk:
    """Bring the desk up, or return the one already running."""
    desk = current()
    if desk is not None:
        return desk
    if not available():
        raise DeskError("Symbio's desk needs macOS." + (f" ({_LOAD_ERROR})" if _LOAD_ERROR else ""))
    binary = helper_binary()
    if binary is None:
        raise DeskError(_build_error or "The desk helper could not be built.")
    with _start_lock():
        desk = current()
        if desk is not None:
            return desk
        state_file = constants.DESK_STATE_FILE
        state_file.unlink(missing_ok=True)
        log_path = constants.LOG_DIR / "desk.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(binary), "--state", str(state_file), "--width", str(int(width)),
                 "--height", str(int(height)), "--name", name],
                start_new_session=True, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + _START_WAIT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise DeskError(_helper_error(log_path)
                                or f"The desk helper exited with status {process.returncode}.")
            state = _read_state()
            if state and state.get("pid") == process.pid:
                desk = current()
                if desk is not None:
                    return desk
            time.sleep(0.05)
        process.terminate()
        raise DeskError(f"The desk helper did not name a display within "
                        f"{_START_WAIT_S:.0f} s. See {log_path}.")


def ensure(config: dict[str, Any] | None) -> Desk | None:
    """The desk when desk mode is on (started if needed), None when it is off.

    Raises DeskError when it is on and cannot run. Callers must not fall back
    to the user's screen then: the setting is the user saying that is not
    where Symbio works.
    """
    if not enabled(config):
        return None
    section = (config or {}).get("desk") or {}
    desk = start(int(section.get("width") or DEFAULT_SIZE[0]),
                 int(section.get("height") or DEFAULT_SIZE[1]))
    keep_awake()
    _placed(desk, str(section.get("placement") or "corner"))
    return desk


# The display-sleep hold for the work in progress. Replaced on every desk
# action, so displays sleep again a couple of minutes after the last one.
_AWAKE: subprocess.Popen | None = None
_AWAKE_S = 120


def keep_awake(seconds: int = _AWAKE_S) -> bool:
    """Wake the desk if the displays slept, and hold them awake a while.

    Display sleep is system-wide: when the Mac's panel sleeps after its idle
    time, the desk sleeps with it, drops out of the active displays and
    cannot be captured -- exactly when the user has left Symbio to work.
    Declaring user activity wakes it (measured: it does not reset the idle
    time that borrowing input is judged by). Nothing is woken at the lock
    screen, where no app window can be worked anyway.
    """
    global _AWAKE
    if _cg is None or session_locked():
        return False
    if any(d.asleep for d in displays() if d.desk):
        try:
            subprocess.run(["caffeinate", "-u", "-t", "1"], capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return False
        _wait(lambda: not any(d.asleep for d in displays() if d.desk), 3.0, 0.1)
    if _AWAKE is not None and _AWAKE.poll() is None:
        _AWAKE.terminate()
    try:
        _AWAKE = subprocess.Popen(["caffeinate", "-d", "-t", str(int(seconds))],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
    except OSError:
        _AWAKE = None
    return not any(d.asleep for d in displays() if d.desk)


def stop(timeout: float = 5.0) -> str:
    """Hand the display back. Windows on it move to the user's screens."""
    state = _read_state() or {}
    try:
        pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if not _helper_alive(pid):
        constants.DESK_STATE_FILE.unlink(missing_ok=True)
        return "No desk is running." + _leftover_note()
    display = state.get("display")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _helper_alive(pid):
        time.sleep(0.05)
    if _helper_alive(pid):
        return f"Asked the desk (PID {pid}) to stop; it is still running."
    note = ""
    if display and int(display) in _display_ids():
        note = (" macOS is keeping the display until your screen next wakes; it "
                "is not used meanwhile.")
    return f"Stopped the desk (display {display})." + note


def _leftover_note() -> str:
    extra = leftovers()
    if not extra:
        return ""
    return (f" {len(extra)} stopped desk display(s) are still listed; macOS "
            "removes them when your screen next wakes.")


# ---------- where it sits ----------

def shared_edge(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """How long an edge two rectangles share. 0 when they meet at a corner or not at all.

    The pointer crosses from one display to another only along a shared edge,
    so this is exactly how much of a desk a mouse can wander onto.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ax + aw == bx or bx + bw == ax:
        return max(0, min(ay + ah, by + bh) - max(ay, by))
    if ay + ah == by or by + bh == ay:
        return max(0, min(ax + aw, bx + bw) - max(ax, bx))
    return 0


def plan_corner(size: tuple[int, int], others: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    """Where the desk's top-left goes to touch the user's screens at one corner only.

    Diagonally off the bottom-left of the whole arrangement: the desk's
    top-right corner meets the bottom-left corner of the screens, so no edge
    is shared, and every edge the user's pointer can reach still stops at the
    end of their own screen.
    """
    width, _height = size
    left = min(x for x, _, _, _ in others)
    bottom = max(y + h for _, y, _, h in others)
    return (left - width, bottom)


def _configure_origin(display: int, x: int, y: int) -> int:
    config = ctypes.c_void_p()
    err = _cg.CGBeginDisplayConfiguration(ctypes.byref(config))
    if err:
        return int(err)
    err = _cg.CGConfigureDisplayOrigin(config, display, int(x), int(y))
    if err:
        _cg.CGCancelDisplayConfiguration(config)
        return int(err)
    return int(_cg.CGCompleteDisplayConfiguration(config, _CONFIGURE_FOR_SESSION))


_PLACED: set[tuple[int, int]] = set()
# place() could not act yet -- the user's screens are asleep, and macOS
# refuses to rearrange displays then -- so it is tried again next time.
_NOT_YET = "not yet: "


def _placed(desk: Desk, how: str) -> str:
    """Place the desk once per desk, retrying for as long as macOS says not yet."""
    key = (desk.display, desk.pid)
    if key in _PLACED:
        return ""
    note = place(desk, how)
    if note.startswith(_NOT_YET):
        return note[len(_NOT_YET):]
    _PLACED.add(key)
    return note


def place(desk: Desk, how: str = "corner") -> str:
    """Move the desk to a corner of the arrangement. "" when it is where it should be.

    Measured on macOS 26.5: a corner-only arrangement is accepted and kept,
    locked or not, but no arrangement change at all is accepted while the
    Mac's own panel sleeps (CGCompleteDisplayConfiguration returns 1014).
    """
    if how in ("", "none", "off", "leave") or _cg is None:
        return ""
    others = [d.rect for d in displays(active=True) if not d.desk]
    if not others:
        return _NOT_YET + "Your screens are asleep; the desk is arranged when they wake."
    if max(shared_edge(desk.rect, o) for o in others) == 0 and _touches(desk.rect, others):
        return ""
    target = plan_corner((desk.width, desk.height), others)
    err = _configure_origin(desk.display, *target)
    if err:
        return _NOT_YET + f"macOS would not move the desk yet (error {err}); it is tried again."
    desk.x, desk.y, desk.width, desk.height = _bounds(desk.display)
    edge = max(shared_edge(desk.rect, o) for o in others)
    if edge:
        # macOS insisted on an edge. The least of one: overlap a single point.
        left, bottom = target
        _configure_origin(desk.display, left + 1, bottom)
        desk.x, desk.y, desk.width, desk.height = _bounds(desk.display)
        edge = max(shared_edge(desk.rect, o) for o in others)
    if edge > 1:
        return (f"macOS put the desk beside your screen, sharing {edge} points of "
                "edge: a pointer pushed off that edge goes onto the desk, where "
                "you cannot see it.")
    return ""


def arrangement_note(desk: Desk) -> str:
    """Where the desk sits against the user's screens, in one sentence."""
    others = [d.rect for d in displays(active=True) if not d.desk]
    if not others:
        return "Your own screen is asleep, so the desk is the only display drawing."
    edge = max(shared_edge(desk.rect, o) for o in others)
    if edge > 1:
        return (f"It shares {edge} points of edge with your screen: a pointer pushed "
                "off that edge goes onto the desk, where you cannot see it.")
    if _touches(desk.rect, others) or edge == 1:
        return ("It touches your screen at one corner only, so your pointer "
                "cannot wander onto it.")
    return "It sits apart from your screens."


def _touches(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    x, y, w, h = rect
    corners = {(x, y), (x + w, y), (x, y + h), (x + w, y + h)}
    for ox, oy, ow, oh in others:
        if corners & {(ox, oy), (ox + ow, oy), (ox, oy + oh), (ox + ow, oy + oh)}:
            return True
    return False


# ---------- the session ----------

def session_locked() -> bool:
    """Is the Mac at its lock screen? Windows are hidden from the desk's tools then."""
    if _cg is None:
        return False
    info = _cg.CGSessionCopyCurrentDictionary()
    if not info:
        return False
    try:
        ref = _cf.CFDictionaryGetValue(info, _key("CGSSessionScreenIsLocked"))
        return bool(ref) and bool(_cf.CFBooleanGetValue(ref))
    finally:
        _cf.CFRelease(info)


LOCKED_NOTE = (
    "The Mac is locked. While it is, macOS hides every app's windows from the "
    "accessibility API and from screen captures -- on the desk too -- so desktop "
    "apps cannot be opened, read or worked until it is unlocked. The browser "
    "still works: it is driven through Chrome itself, not the screen.")


def user_idle_seconds() -> float:
    """Seconds since the person last touched the mouse, trackpad or keyboard."""
    if _cg is None:
        return 0.0
    return float(_cg.CGEventSourceSecondsSinceLastEventType(_HID_SYSTEM_STATE, _ANY_INPUT_EVENT))


def front_app_pid() -> int:
    """The app the user's keyboard goes to, from Launch Services.

    Not the first window in the window list: with a desk that is often a desk
    window, which is in front in stacking order without being in front of
    anything the user can see.
    """
    try:
        asn = subprocess.run(["lsappinfo", "front"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        if not asn:
            return 0
        out = subprocess.run(["lsappinfo", "info", "-only", "pid", asn],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    match = re.search(r'"pid"\s*=\s*(\d+)', out)
    return int(match.group(1)) if match else 0


# ---------- windows ----------

@dataclass
class Window:
    number: int
    pid: int
    owner: str
    title: str
    x: int
    y: int
    width: int
    height: int

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


def windows(desk: Desk | None = None) -> list[Window]:
    """Ordinary on-screen windows, front to back; only those on `desk` if given."""
    if _cg is None:
        return []
    info = _cg.CGWindowListCopyWindowInfo(_ON_SCREEN_ONLY | _EXCLUDE_DESKTOP, 0)
    if not info:
        return []
    out: list[Window] = []
    try:
        for i in range(_cf.CFArrayGetCount(info)):
            entry = _cf.CFArrayGetValueAtIndex(info, i)
            # Layer 0 is an application window; menus, the Dock and the menu
            # bar sit above it and are never what a task is about.
            if _number(entry, "kCGWindowLayer") != 0:
                continue
            alpha = _number(entry, "kCGWindowAlpha", floating=True)
            rect = _rect(entry, "kCGWindowBounds")
            if rect is None or rect[2] < 2 or rect[3] < 2 or (alpha is not None and alpha <= 0):
                continue
            window = Window(int(_number(entry, "kCGWindowNumber") or 0),
                            int(_number(entry, "kCGWindowOwnerPID") or 0),
                            _string(entry, "kCGWindowOwnerName"),
                            _string(entry, "kCGWindowName"), *rect)
            if desk is not None and not desk.contains(*window.centre):
                continue
            out.append(window)
    finally:
        _cf.CFRelease(info)
    return out


def front_window(desk: Desk) -> Window | None:
    """The desk's front window: what a look and an unaddressed action are about."""
    found = windows(desk)
    return found[0] if found else None


def window_at(desk: Desk, x: float, y: float) -> Window | None:
    """The desk window under a point, in the window server's space."""
    for window in windows(desk):
        if window.contains(x, y):
            return window
    return None


# ---------- looking ----------

def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return (0, 0)


def capture(desk: Desk, path: Path | None = None) -> Path:
    """The desk as a PNG, and nothing of the user's screen.

    screencapture numbers displays in the window server's active order; the
    result is checked against the desk's own pixel size, so a wrong number
    cannot quietly hand over a picture of the user's screen instead.
    """
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = constants.SCREENSHOTS_DIR / f"desk_{stamp}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    active = _display_ids(active=True)
    if desk.display not in active and keep_awake():
        active = _display_ids(active=True)
    if desk.display not in active:
        if session_locked():
            raise DeskError(LOCKED_NOTE)
        raise DeskError("The desk is asleep with the Mac's displays and would not "
                        "wake, so it cannot be captured.")
    done = subprocess.run(["screencapture", "-x", "-D", str(active.index(desk.display) + 1),
                           str(path)], capture_output=True, text=True, timeout=20)
    if done.returncode != 0 or not path.exists():
        raise DeskError("Capturing the desk failed: "
                        + ((done.stderr or done.stdout).strip() or f"status {done.returncode}"))
    expected = (int(_cg.CGDisplayPixelsWide(desk.display)),
                int(_cg.CGDisplayPixelsHigh(desk.display)))
    size = _image_size(path)
    if size != (0, 0) and size != expected:
        path.unlink(missing_ok=True)
        raise DeskError(f"The capture came back {size[0]}x{size[1]}, not the desk's "
                        f"{expected[0]}x{expected[1]}; it was thrown away rather than "
                        "risk it being your screen.")
    return path


def fingerprint(desk: Desk) -> str:
    """A cheap hash of what the desk shows, to tell a landed action from a lost one."""
    try:
        path = capture(desk, constants.CACHE_DIR / "desk_fingerprint.png")
    except (DeskError, OSError, subprocess.SubprocessError):
        return ""
    try:
        from PIL import Image

        with Image.open(path) as image:
            small = image.convert("L").resize((96, 60))
            return hashlib.sha1(small.tobytes()).hexdigest()
    except Exception:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    finally:
        path.unlink(missing_ok=True)


# ---------- apps ----------

def app_pid(name: str) -> int:
    """The running process for an application name, or 0."""
    try:
        out = subprocess.run(["lsappinfo", "info", "-only", "pid", name],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    match = re.search(r'"pid"\s*=\s*(\d+)', out)
    return int(match.group(1)) if match else 0


def bundle_id(name: str) -> str:
    """The bundle identifier for an app name, without launching it; "" if unknown.

    For the names Launch Services does not list as they were typed: the
    model says "VS Code", the process calls itself "Code".
    """
    try:
        return subprocess.run(["osascript", "-e", f'id of app "{_applescript(name)}"'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _wait(predicate, timeout: float, step: float = 0.1):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value or time.monotonic() >= deadline:
            return value
        time.sleep(step)


def open_app(name: str, desk: Desk) -> str:
    """Launch or reach an app WITHOUT bringing it in front of the user, on the desk.

    `open -g` launches in the background. The app's windows then move onto the
    desk -- all of them when this launched it, and when it was already open
    and in use on the user's screen, a NEW window it is asked for through its
    own File menu, never the one they have open.
    """
    from symbio import ax

    name = (name or "").strip()
    if not name:
        return "Open failed: no application named."
    if session_locked():
        return LOCKED_NOTE
    if not ax.trusted():
        return ax.PERMISSION_HINT
    bundle = bundle_id(name)
    before_pid = app_pid(name) or (app_pid(bundle) if bundle else 0)
    # The user's windows, if the app was already up: those are never taken.
    theirs = ax.window_elements(before_pid) if before_pid else []
    done = subprocess.run(["open", "-g", "-a", name], capture_output=True, text=True, timeout=30)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        return f"Could not open {name!r}: {detail[0] if detail else 'no such application'}."
    pid = before_pid or _wait(lambda: app_pid(name) or (app_pid(bundle) if bundle else 0),
                              12.0, 0.25)
    if not pid:
        return (f"Asked macOS to open {name!r}, but no running app by that name "
                "appeared. Check the name, or wait and look again.")
    ax_windows = _wait(lambda: ax.window_elements(pid), 10.0, 0.2) or []
    if not ax_windows:
        return (f"{name} is running but has no window yet. Wait a moment with "
                "desktop_wait, then open it again to bring its window to the desk.")
    on_desk = [w for w in ax_windows if w.get("frame") and desk.contains(*_frame_centre(w))]
    if on_desk:
        if before_pid:
            return f"{name} is already open on your desk."
        return f"Opened {name} on your desk."
    if theirs:
        # In use on the user's screen: ask the app for a fresh window and take
        # only that. Elements are new objects on every read, so windows are
        # told apart by the window server's number, or by frame without one.
        known = {_window_identity(w) for w in ax_windows}
        if not ax.press_menu_item(pid, _NEW_WINDOW):
            return (f"{name} is already open on the user's screen, and it has no "
                    "New Window item to open a separate one for the desk. Their "
                    "window was left where it is; ask them to close it or to hand "
                    "it over.")
        fresh = _wait(lambda: [w for w in ax.window_elements(pid)
                               if _window_identity(w) not in known], 5.0, 0.15) or []
        if not fresh:
            return (f"Asked {name} for a new window for the desk, but none appeared. "
                    "Look with see_screen before trying again.")
        moved = move_windows(fresh, desk)
        return f"Opened a new {name} window on your desk ({moved} moved); the user's own window was left alone."
    moved = move_windows(ax_windows, desk)
    if not moved:
        return (f"Opened {name}, but macOS would not move its window onto the desk. "
                "It may be on the user's screen; say so rather than working in it.")
    return f"Opened {name} on your desk."


_NEW_WINDOW = re.compile(r"^New( [A-Za-z]+)? (Window|Document)$|^New Window$|^New$")


def _window_identity(window: dict[str, Any]) -> Any:
    return window.get("number") or ("frame", window.get("frame"), window.get("title"))


def _frame_centre(window: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = window["frame"]
    return (x + w / 2, y + h / 2)


def move_windows(ax_windows: list[dict[str, Any]], desk: Desk) -> int:
    """Put these windows on the desk, cascaded, shrunk to fit. How many moved."""
    from symbio import ax

    moved = 0
    usable_w = desk.width - 2 * _MARGIN
    usable_h = desk.height - MENU_BAR - 2 * _MARGIN
    for i, window in enumerate(ax_windows):
        offset = (i % 8) * 24
        x = desk.x + _MARGIN + offset
        y = desk.y + MENU_BAR + _MARGIN + offset
        frame = window.get("frame")
        if frame and (frame[2] > usable_w - offset or frame[3] > usable_h - offset):
            ax.set_size(window["_ref"], min(frame[2], usable_w - offset),
                        min(frame[3], usable_h - offset))
        if ax.set_position(window["_ref"], x, y):
            frame = ax.frame_of(window["_ref"])
            if frame and desk.contains(frame[0] + frame[2] / 2, frame[1] + frame[3] / 2):
                moved += 1
    return moved


# ---------- input posted to one app ----------

_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
    "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38,
    "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46, ".": 47,
    "`": 50, "enter": 36, "return": 36, "tab": 48, "space": 49, "backspace": 51,
    "delete": 51, "escape": 53, "esc": 53, "forwarddelete": 117, "home": 115,
    "end": 119, "pageup": 116, "pagedown": 121, "left": 123, "right": 124,
    "down": 125, "up": 126, "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96,
    "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}
_KEY_ALIASES = {"arrowdown": "down", "arrowup": "up", "arrowleft": "left",
                "arrowright": "right", "del": "delete", "pgup": "pageup",
                "pgdn": "pagedown", "spacebar": "space"}
_FLAGS = {"cmd": 1 << 20, "command": 1 << 20, "meta": 1 << 20, "super": 1 << 20,
          "shift": 1 << 17, "ctrl": 1 << 18, "control": 1 << 18,
          "option": 1 << 19, "opt": 1 << 19, "alt": 1 << 19, "fn": 1 << 23}


def parse_chord(combo: str) -> tuple[int, int, str, list[str]] | None:
    """'cmd+shift+s' -> (keycode, flags, key, modifiers), or None for an unknown key."""
    parts = [p for p in re.split(r"[+\s]+|(?<=\w)-(?=\w)", (combo or "").strip().lower()) if p]
    if not parts:
        return None
    *mods, key = parts
    key = _KEY_ALIASES.get(key, key)
    flags = 0
    names = []
    for mod in mods:
        if mod not in _FLAGS:
            return None
        flags |= _FLAGS[mod]
        names.append({"command": "cmd", "meta": "cmd", "super": "cmd", "control": "ctrl",
                      "opt": "option", "alt": "option"}.get(mod, mod))
    if key not in _KEYCODES:
        return None
    return _KEYCODES[key], flags, key, names


def _post(event: int, pid: int | None) -> None:
    if pid:
        _cg.CGEventPostToPid(int(pid), event)
    else:
        _cg.CGEventPost(_HID_EVENT_TAP, event)
    _cf.CFRelease(event)


def click(pid: int | None, x: float, y: float, button: str = "left",
          clicks: int = 1, window: int = 0) -> None:
    """Mouse down and up at (x, y). To the app `pid` alone, or, with pid None, for real.

    Posted to one app, the event carries its window itself -- the window
    server would fill that in for a real one -- and the pointer does not move.
    """
    down, up, _drag, number = _MOUSE.get(button, _MOUSE["left"])
    point = CGPoint(float(x), float(y))
    if not pid:
        _post(_cg.CGEventCreateMouseEvent(None, _MOUSE_MOVED, point, number), None)
    for n in range(1, max(1, int(clicks)) + 1):
        for kind in (down, up):
            event = _cg.CGEventCreateMouseEvent(None, kind, point, number)
            _cg.CGEventSetIntegerValueField(event, _CLICK_STATE_FIELD, n)
            if window:
                _cg.CGEventSetIntegerValueField(event, _WINDOW_UNDER_POINTER_FIELD, window)
                _cg.CGEventSetIntegerValueField(event, _WINDOW_THAT_CAN_HANDLE_FIELD, window)
            _post(event, pid)
            time.sleep(0.02)


def move(pid: int | None, x: float, y: float, window: int = 0) -> None:
    event = _cg.CGEventCreateMouseEvent(None, _MOUSE_MOVED, CGPoint(float(x), float(y)), 0)
    if window:
        _cg.CGEventSetIntegerValueField(event, _WINDOW_UNDER_POINTER_FIELD, window)
    _post(event, pid)


def drag(pid: int | None, x1: float, y1: float, x2: float, y2: float,
         button: str = "left", steps: int = 12, window: int = 0) -> None:
    down, up, dragged, number = _MOUSE.get(button, _MOUSE["left"])
    start_point = CGPoint(float(x1), float(y1))
    if not pid:
        _post(_cg.CGEventCreateMouseEvent(None, _MOUSE_MOVED, start_point, number), None)
    _post(_cg.CGEventCreateMouseEvent(None, down, start_point, number), pid)
    for i in range(1, steps + 1):
        point = CGPoint(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps)
        event = _cg.CGEventCreateMouseEvent(None, dragged, point, number)
        if window:
            _cg.CGEventSetIntegerValueField(event, _WINDOW_UNDER_POINTER_FIELD, window)
        _post(event, pid)
        time.sleep(0.02)
    _post(_cg.CGEventCreateMouseEvent(None, up, CGPoint(float(x2), float(y2)), number), pid)


def scroll(pid: int | None, x: float, y: float, dy: int = 0, dx: int = 0) -> None:
    """Scroll by lines at (x, y): positive dy is up, as the wheel counts it."""
    if not pid:
        # A real wheel scrolls whatever is under the pointer, wherever the
        # event says it happened: the pointer has to be there first.
        _post(_cg.CGEventCreateMouseEvent(None, _MOUSE_MOVED, CGPoint(float(x), float(y)), 0),
              None)
    event = _cg.CGEventCreateScrollWheelEvent2(None, 1, 2, int(dy), int(dx), 0)
    _cg.CGEventSetLocation(event, CGPoint(float(x), float(y)))
    _post(event, pid)


def key(pid: int | None, keycode: int, flags: int = 0) -> None:
    for down in (True, False):
        event = _cg.CGEventCreateKeyboardEvent(None, keycode, down)
        if flags:
            _cg.CGEventSetFlags(event, flags)
        _post(event, pid)
        time.sleep(0.01)


def type_text(pid: int | None, text: str) -> None:
    """Type `text` as characters, not keycodes, so any layout and any script works."""
    units = text.encode("utf-16-le")
    codes = [int.from_bytes(units[i:i + 2], "little") for i in range(0, len(units), 2)]
    # The event carries at most 20 UTF-16 units; longer strings go in chunks.
    for start_at in range(0, len(codes), 20):
        chunk = codes[start_at:start_at + 20]
        buffer = (ctypes.c_uint16 * len(chunk))(*chunk)
        for down in (True, False):
            event = _cg.CGEventCreateKeyboardEvent(None, 0, down)
            _cg.CGEventKeyboardSetUnicodeString(event, len(chunk), buffer)
            _post(event, pid)
        time.sleep(0.01)


def pointer() -> tuple[float, float]:
    event = _cg.CGEventCreate(None)
    try:
        point = _cg.CGEventGetLocation(event)
        return (float(point.x), float(point.y))
    finally:
        _cf.CFRelease(event)


class NotNow(DeskError):
    """Real input is off-limits right now; the reason says why and when."""


def focus_on_desk(desk: Desk, pid: int) -> bool:
    """Is the window app `pid` would type into one on the desk?

    Keys go to an app's key window. An app with a window on the desk and
    another on the user's screen may hold its focus in theirs, and keys sent
    to it then are typed into the user's document.
    """
    from symbio import ax

    app = ax.app_element(pid)
    window = ax._attr(app, "AXFocusedWindow") if app else None
    frame = ax.frame_of(window) if window else None
    return bool(frame) and desk.contains(frame[0] + frame[2] / 2, frame[1] + frame[3] / 2)


@contextmanager
def borrowed(pid: int, idle_after: float, window: int = 0) -> Iterator[None]:
    """The user's pointer and keyboard, for one action, and only if they are away.

    Some controls answer nothing but a real event: a canvas, a game, a field
    that ignores the accessibility API. Those need the pointer ON the desk and
    the app in front, with the desk's window as its key window. That is taken
    only after `idle_after` seconds without the user touching the Mac, and
    put back -- pointer position and front app -- the moment it is done.
    """
    from symbio import ax

    if idle_after <= 0:
        raise NotNow("Borrowing the pointer and keyboard is turned off "
                     "(desk.borrow_input_after_idle_s is 0).")
    if session_locked():
        raise NotNow(LOCKED_NOTE)
    idle = user_idle_seconds()
    if idle < idle_after:
        raise NotNow(f"The user touched the Mac {idle:.0f}s ago. Real mouse and "
                     f"keyboard input is borrowed only after {idle_after:.0f}s "
                     "without them, so this has to wait or go another way.")
    saved_pointer = pointer()
    saved_front = front_app_pid()
    ax.set_frontmost(pid)
    if window:
        ax.raise_window(pid, window)
    time.sleep(0.2)
    try:
        yield
    finally:
        time.sleep(0.1)
        _cg.CGWarpMouseCursorPosition(CGPoint(*saved_pointer))
        _cg.CGAssociateMouseAndMouseCursorPosition(1)
        if saved_front and saved_front != pid:
            ax.set_frontmost(saved_front)


@contextmanager
def keep_user_front(settle: float = 2.0) -> Iterator[None]:
    """Give the user's app its focus back if something launched over it.

    Chrome activates itself when Playwright starts it, even with its window on
    the desk -- the user's keystrokes would then land in the agent's browser.
    """
    from symbio import ax

    before = front_app_pid() if available() else 0
    try:
        yield
    finally:
        if before:
            deadline = time.monotonic() + settle
            while time.monotonic() < deadline:
                now = front_app_pid()
                if now and now != before:
                    ax.set_frontmost(before)
                    break
                time.sleep(0.1)


def browser_args(config: dict[str, Any] | None) -> list[str]:
    """Chrome flags that open the automated browser's window on the desk; [] when off."""
    if not enabled(config) or not ((config or {}).get("desk") or {}).get("browser", True):
        return []
    desk = ensure(config)
    if desk is None:
        return []
    return [f"--window-position={desk.x},{desk.y + MENU_BAR}",
            f"--window-size={desk.width},{desk.height - MENU_BAR}"]


def self_test(config: dict[str, Any] | None = None) -> list[tuple[str, bool, str]]:
    """Drive the whole path once, for real, on a scratch document: `symb desk check`.

    Opens a TextEdit document of its own in the background, moves it onto
    the desk, reads its controls, types into it through the accessibility API
    and by posting keys to TextEdit alone, and checks at every step that the
    user's front app and pointer never moved. It needs the Mac unlocked --
    nothing about app windows can be checked at the lock screen -- and cleans
    up after itself: the document window is closed, and TextEdit quit if this
    started it.
    """
    from symbio import ax

    steps: list[tuple[str, bool, str]] = []

    def step(name: str, ok: Any, detail: str = "") -> bool:
        steps.append((name, bool(ok), detail))
        return bool(ok)

    if not step("macOS", available(), _LOAD_ERROR):
        return steps
    if not step("unlocked", not session_locked(), "" if not session_locked() else LOCKED_NOTE):
        return steps
    if not step("accessibility granted", ax.trusted(), "" if ax.trusted() else ax.PERMISSION_HINT):
        return steps
    front_before, pointer_before = front_app_pid(), pointer()
    section = {**DEFAULT_SECTION, **(((config or {}).get("desk")) or {}), "enabled": True}
    was_up = current() is not None
    try:
        running = ensure({"desk": section})
    except DeskError as e:
        step("desk running", False, str(e))
        return steps
    try:
        _check_on(running, steps, step, front_before, pointer_before)
    finally:
        # A desk the check brought up for itself goes again with desk mode off.
        if not was_up and not enabled(config):
            stop()
    return steps


def _check_on(running: Desk, steps: list, step, front_before: int,
              pointer_before: tuple[float, float]) -> None:
    from symbio import ax

    began = time.monotonic()
    step("desk running", running is not None,
         f"display {running.display}, {running.width}x{running.height} at ({running.x}, {running.y})")
    step("arrangement", True, arrangement_note(running))
    try:
        shot = capture(running)
        step("capture of the desk alone", _image_size(shot)[0] == running.width, str(shot))
    except DeskError as e:
        step("capture of the desk alone", False, str(e))

    document = constants.CACHE_DIR / "symbio-desk-check.txt"
    document.write_text("", encoding="utf-8")
    was_running = bool(app_pid("TextEdit"))
    mine: list[dict[str, Any]] = []
    try:
        subprocess.run(["open", "-g", "-a", "TextEdit", str(document)],
                       capture_output=True, timeout=30)
        pid = _wait(lambda: app_pid("TextEdit"), 12.0, 0.25)
        mine = _wait(lambda: [w for w in ax.window_elements(pid)
                              if document.stem in str(w.get("title") or "")], 10.0, 0.2) or []
        if not step("scratch document opened in the background", mine,
                    "" if mine else "TextEdit never showed the document's window"):
            return
        step("moved onto the desk", move_windows(mine[:1], running) == 1)
        number = mine[0].get("number") or 0
        step("window server has it on the desk",
             [w for w in windows(running) if not number or w.number == number])
        snap = ax.snapshot(limit=40, pid=pid, window=mine[0]["_ref"],
                           origin=(running.x, running.y))
        areas = [e for e in snap.get("elements", []) if e["role"] == "AXTextArea"]
        step("controls read from the tree", areas, f"{len(snap.get('elements', []))} controls")
        if areas:
            wrote = ax.insert_text(areas[0], "Symbio desk check")
            step("typed through the accessibility API",
                 wrote and "Symbio desk check" in ax.value_of(areas[0]),
                 ax.value_of(areas[0])[:60])
            type_text(pid, " + posted")
            time.sleep(0.4)
            posted = " + posted" in ax.value_of(areas[0])
            steps.append(("typed by posting keys to TextEdit alone (informational)", posted,
                          "works for this app" if posted else
                          "a background app ignored posted keys: the harness borrows "
                          "the real keyboard only while you are away"))
        # Only meaningful if nobody touched the Mac meanwhile: a person moving
        # their own mouse during the check is not the check moving it.
        if user_idle_seconds() < time.monotonic() - began:
            note = " (informational: you used the Mac during the check)"
            steps.append(("your front app never changed" + note,
                          front_app_pid() == front_before, ""))
            steps.append(("your pointer never moved" + note, pointer() == pointer_before, ""))
        else:
            step("your front app never changed", front_app_pid() == front_before)
            step("your pointer never moved", pointer() == pointer_before)
    finally:
        # The scratch window, then TextEdit if this started it.
        if mine:
            close = ax._attr(mine[0]["_ref"], "AXCloseButton")
            if close:
                ax.press({"_ref": close})
        if not was_running:
            subprocess.run(["osascript", "-e", 'tell application "TextEdit" to quit saving no'],
                           capture_output=True, timeout=15)
        document.unlink(missing_ok=True)


DEFAULT_SECTION = {"width": DEFAULT_SIZE[0], "height": DEFAULT_SIZE[1],
                   "placement": "corner", "borrow_input_after_idle_s": 30, "browser": True}


def attach(browser: Any, config: Any) -> None:
    """Open a BrowserSession's window on the desk whenever desk mode is on.

    Read at launch, not now: `config` is the live dict, or a callable that
    returns it freshly read, so `symb desk on` applies to the next window
    without a restart.
    """
    if browser is None:
        return
    try:
        browser.desk_window = lambda: browser_args(config() if callable(config) else config)
    except Exception:
        # A stand-in without attributes (a test double) keeps its old launch.
        pass


def describe(config: dict[str, Any] | None = None) -> str:
    """One paragraph for `symb desk status` and /selfcheck."""
    if not available():
        return "Desk: not available (needs macOS)."
    desk = current()
    state = "on" if enabled(config) else "off"
    lines = []
    if desk is None:
        lines.append(f"Desk mode: {state}. No desk is running.")
    else:
        found = windows(desk)
        lines.append(f"Desk mode: {state}. Desk running: display {desk.display}, "
                     f"{desk.width}x{desk.height} at ({desk.x}, {desk.y}), PID {desk.pid}.")
        lines.append("On it: " + (", ".join(sorted({w.owner or '?' for w in found}))
                                  if found else "nothing yet."))
    extra = leftovers()
    if extra:
        lines.append(f"{len(extra)} stopped desk display(s) still listed (removed "
                     "when your screen next wakes).")
    if session_locked():
        lines.append("The Mac is locked: desktop apps cannot be read or worked until it is unlocked.")
    return "\n".join(lines)
