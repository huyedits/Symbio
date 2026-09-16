"""macOS accessibility tree: the desktop's own list of its controls.

Pixel vision answers "where is the Post button" with a coordinate guessed from
an image, and it cannot ground anything smaller than a patch -- X's composer
is 28px and the vision worker could never point at it. The operating system
already knows the answer exactly: every native control publishes its role, its
title and its frame through the accessibility API, and that is what
VoiceOver reads. This module asks the same question the screen reader asks.

What it buys, against the vision path:

  * no vision model, so no deep-sleep swap and no second set of weights
    resident next to the headmaster -- the whole listing costs no RAM
  * exact text, so "Post" is "Post" and not "Post" misread from 11px type
  * exact frames, with no patch floor and no Retina scale to undo
  * absence is real: an element that is not in the tree is not on the screen,
    where a vision miss and a missing control look identical

ctypes against the system framework, deliberately, and not pyobjc: this is
shipped to other people's Macs and `pyobjc-framework-ApplicationServices` is
not in requirements.txt. The same choice, for the same reason, as
set_keyboard_layout.py's Carbon calls.

Everything here needs the Accessibility grant (System Settings > Privacy &
Security > Accessibility) for whatever runs Symbio. Without it the API does
not error -- it returns empty trees -- so `trusted()` is checked first and the
grant is named in the reply.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import time
from typing import Any

_APPSERV_PATH = ("/System/Library/Frameworks/ApplicationServices.framework"
                 "/Versions/A/ApplicationServices")
_CF_PATH = ("/System/Library/Frameworks/CoreFoundation.framework"
            "/Versions/A/CoreFoundation")
_CG_PATH = ("/System/Library/Frameworks/CoreGraphics.framework"
            "/Versions/A/CoreGraphics")

_LOAD_ERROR = ""
try:
    _ax = ctypes.cdll.LoadLibrary(_APPSERV_PATH)
    _cf = ctypes.cdll.LoadLibrary(_CF_PATH)
    _cg = ctypes.cdll.LoadLibrary(_CG_PATH)
except Exception as e:  # pragma: no cover - not macOS
    _ax = _cf = _cg = None
    _LOAD_ERROR = str(e)


PERMISSION_HINT = (
    "The accessibility tree is empty because macOS has not granted this "
    "process Accessibility permission — it returns nothing rather than an "
    "error. Grant it in System Settings > Privacy & Security > Accessibility "
    "for the terminal (or app) Symbio runs in, restart it, and look again."
)

# Every argtype spelled out. Without them ctypes truncates pointers to 32 bits
# and hands CoreFoundation a garbage address, which takes the interpreter down
# with it rather than returning an error.
if _ax is not None:
    _cf.CFArrayGetCount.restype = ctypes.c_long
    _cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    _cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    _cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                       ctypes.c_long, ctypes.c_uint32]
    _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    _cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                              ctypes.c_uint32]
    _cf.CFGetTypeID.restype = ctypes.c_ulong
    _cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
    _cf.CFStringGetTypeID.restype = ctypes.c_ulong
    _cf.CFArrayGetTypeID.restype = ctypes.c_ulong
    _cf.CFBooleanGetTypeID.restype = ctypes.c_ulong
    _cf.CFNumberGetTypeID.restype = ctypes.c_ulong
    _cf.CFBooleanGetValue.restype = ctypes.c_bool
    _cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
    _cf.CFNumberGetValue.restype = ctypes.c_bool
    _cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_long,
                                     ctypes.c_void_p]
    _cf.CFRelease.argtypes = [ctypes.c_void_p]

    _ax.AXIsProcessTrusted.restype = ctypes.c_bool
    _ax.AXUIElementCreateSystemWide.restype = ctypes.c_void_p
    _ax.AXUIElementCreateApplication.restype = ctypes.c_void_p
    _ax.AXUIElementCreateApplication.argtypes = [ctypes.c_int]
    _ax.AXUIElementCopyAttributeValue.restype = ctypes.c_int
    _ax.AXUIElementCopyAttributeValue.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    _ax.AXUIElementSetAttributeValue.restype = ctypes.c_int
    _ax.AXUIElementSetAttributeValue.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _ax.AXUIElementCopyActionNames.restype = ctypes.c_int
    _ax.AXUIElementCopyActionNames.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    _ax.AXUIElementPerformAction.restype = ctypes.c_int
    _ax.AXUIElementPerformAction.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ax.AXUIElementGetPid.restype = ctypes.c_int
    _ax.AXUIElementGetPid.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_int)]
    _ax.AXValueGetValue.restype = ctypes.c_bool
    _ax.AXValueGetValue.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                    ctypes.c_void_p]

    _cf.CFDictionaryGetValue.restype = ctypes.c_void_p
    _cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _cg.CGWindowListCopyWindowInfo.restype = ctypes.c_void_p
    _cg.CGWindowListCopyWindowInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32]

_UTF8 = 0x08000100
_AX_ERROR_SUCCESS = 0
_AX_VALUE_CGPOINT = 1
_AX_VALUE_CGSIZE = 2
_CFNUMBER_LONG = 10


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


def available() -> bool:
    """Is the accessibility API callable at all (i.e. are we on a Mac)?"""
    return _ax is not None


def trusted() -> bool:
    """Has this process been granted Accessibility permission?"""
    if _ax is None:
        return False
    try:
        return bool(_ax.AXIsProcessTrusted())
    except Exception:
        return False


# ---------- CoreFoundation plumbing ----------

def _cfstr(text: str):
    return _cf.CFStringCreateWithCString(None, text.encode("utf-8"), _UTF8)


def _from_cfstring(ref) -> str:
    buf = ctypes.create_string_buffer(2048)
    if _cf.CFStringGetCString(ref, buf, 2048, _UTF8):
        return buf.value.decode("utf-8", "replace")
    return ""


def _unwrap(ref) -> Any:
    """A CFTypeRef as the closest Python value, or the raw ref for elements.

    AXUIElementRef has no public type id worth matching, so anything that is
    not a string, number, boolean, array or AXValue comes back as the pointer
    itself — which is exactly what a child element should be.
    """
    if not ref:
        return None
    type_id = _cf.CFGetTypeID(ref)
    if type_id == _cf.CFStringGetTypeID():
        return _from_cfstring(ref)
    if type_id == _cf.CFBooleanGetTypeID():
        return bool(_cf.CFBooleanGetValue(ref))
    if type_id == _cf.CFNumberGetTypeID():
        out = ctypes.c_long()
        if _cf.CFNumberGetValue(ref, _CFNUMBER_LONG, ctypes.byref(out)):
            return int(out.value)
        return None
    if type_id == _cf.CFArrayGetTypeID():
        return [_cf.CFArrayGetValueAtIndex(ref, i)
                for i in range(_cf.CFArrayGetCount(ref))]
    point = _CGPoint()
    if _ax.AXValueGetValue(ref, _AX_VALUE_CGPOINT, ctypes.byref(point)):
        return (float(point.x), float(point.y))
    size = _CGSize()
    if _ax.AXValueGetValue(ref, _AX_VALUE_CGSIZE, ctypes.byref(size)):
        return (float(size.width), float(size.height))
    return ref


def _attr(element, name: str) -> Any:
    """One attribute of one element, or None if it does not carry it."""
    if not element:
        return None
    out = ctypes.c_void_p()
    key = _cfstr(name)
    try:
        err = _ax.AXUIElementCopyAttributeValue(element, key, ctypes.byref(out))
    finally:
        _cf.CFRelease(key)
    if err != _AX_ERROR_SUCCESS or not out:
        return None
    return _unwrap(out)


def _actions(element) -> list[str]:
    out = ctypes.c_void_p()
    if _ax.AXUIElementCopyActionNames(element, ctypes.byref(out)) != _AX_ERROR_SUCCESS:
        return []
    if not out:
        return []
    return [_from_cfstring(ref) for ref in (_unwrap(out) or [])]


# ---------- the tree ----------

# Roles a person can act on. AXStaticText and AXGroup are structure, not
# controls, and listing them buries the six things the turn can actually click
# under three hundred labels.
ACTIONABLE_ROLES = frozenset({
    "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton",
    "AXMenuItem", "AXMenuBarItem", "AXTextField", "AXTextArea", "AXSearchField",
    "AXComboBox", "AXLink", "AXSlider", "AXIncrementor", "AXStepper",
    "AXDisclosureTriangle", "AXTabGroup", "AXCell", "AXImage", "AXToolbarButton",
})

# A tree walk is IPC per node, so a deep app (Safari with a big page) can spend
# seconds if it is allowed to. These caps are what keeps a look interactive;
# the listing says when it stopped early rather than pretending it is whole.
_MAX_NODES = 1500
_MAX_DEPTH = 14

_TEXT_ROLES = frozenset({"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"})


def _label(element, role: str) -> str:
    """The words a person would use for this control."""
    for attribute in ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXHelp"):
        value = _attr(element, attribute)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = _attr(element, "AXValue")
    if isinstance(value, str) and value.strip():
        return value.strip()[:120]
    if role in _TEXT_ROLES:
        return "(empty text field)"
    return ""


def _frame(element) -> tuple[int, int, int, int] | None:
    position = _attr(element, "AXPosition")
    size = _attr(element, "AXSize")
    if not (isinstance(position, tuple) and isinstance(size, tuple)):
        return None
    x, y = position
    w, h = size
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


# CGWindowList option bits: on-screen windows only, and not the desktop
# picture or its icons.
_ON_SCREEN_ONLY = 1
_EXCLUDE_DESKTOP = 16


def frontmost_window() -> tuple[int, str, str]:
    """(pid, application name, window title) of the window in front, or (0,..).

    Read from the window server rather than from the accessibility API. The
    system-wide element is supposed to answer this through
    AXFocusedApplication, and on this machine -- granted, trusted, with every
    per-application tree readable -- that call returns kAXErrorCannotComplete
    (-25204) every time. Trusting it alone made a fully working accessibility
    path report itself as a missing permission.

    The window list needs no permission at all for the owner and the frame;
    only the window TITLE requires Screen Recording, and nothing here depends
    on the title.
    """
    if _cg is None:
        return 0, "", ""
    try:
        windows = _cg.CGWindowListCopyWindowInfo(
            _ON_SCREEN_ONLY | _EXCLUDE_DESKTOP, 0)
        if not windows:
            return 0, "", ""
        for i in range(_cf.CFArrayGetCount(windows)):
            entry = _cf.CFArrayGetValueAtIndex(windows, i)
            # Layer 0 is an ordinary application window. Menu bar extras, the
            # Dock and Control Centre sit above it and are never the thing a
            # request is about; the list is already in front-to-back order.
            if _dict_value(entry, "kCGWindowLayer") != 0:
                continue
            pid = _dict_value(entry, "kCGWindowOwnerPID")
            if not isinstance(pid, int):
                continue
            return (pid, str(_dict_value(entry, "kCGWindowOwnerName") or ""),
                    str(_dict_value(entry, "kCGWindowName") or ""))
    except Exception:
        return 0, "", ""
    return 0, "", ""


def _dict_value(dictionary, key: str) -> Any:
    ref = _cf.CFDictionaryGetValue(dictionary, _cfstr(key))
    return _unwrap(ref) if ref else None


def _focused_app():
    """The application element for whatever is in front, or None."""
    system = _ax.AXUIElementCreateSystemWide()
    app = _attr(system, "AXFocusedApplication")
    if app:
        return app
    pid, _name, _title = frontmost_window()
    if pid:
        return _ax.AXUIElementCreateApplication(pid)
    # Neither route answered: no windows, or the grant really is missing.
    return None


def _app_name(app) -> str:
    for attribute in ("AXTitle", "AXRoleDescription"):
        value = _attr(app, attribute)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "?"


def snapshot(limit: int = 40, include_text: bool = False) -> dict[str, Any]:
    """The frontmost window's controls, as a list the model can act on.

    Returns a dict rather than a string so the caller can both render it and
    keep the element handles: the numbers in the rendering are indexes into
    `elements`, and `click`/`set_text` below take the same number. The handles
    stay valid while the app's tree does, which is why the snapshot is taken
    again after anything that changes the screen.
    """
    if _ax is None:
        return {"ok": False, "reason": f"Not available here: {_LOAD_ERROR}",
                "elements": []}
    if not trusted():
        return {"ok": False, "reason": PERMISSION_HINT, "elements": []}
    app = _focused_app()
    if not app:
        return {"ok": False, "reason": PERMISSION_HINT, "elements": []}

    window = _attr(app, "AXFocusedWindow") or _attr(app, "AXMainWindow")
    roots = [window] if window else (_attr(app, "AXWindows") or [])
    title = _attr(window, "AXTitle") if window else None

    wanted = set(ACTIONABLE_ROLES)
    if include_text:
        wanted = wanted | {"AXStaticText"}

    elements: list[dict[str, Any]] = []
    visited = 0
    truncated = False
    stack: list[tuple[Any, int]] = [(root, 0) for root in roots if root]
    while stack:
        element, depth = stack.pop(0)
        visited += 1
        if visited > _MAX_NODES or len(elements) >= limit:
            truncated = True
            break
        role = _attr(element, "AXRole")
        if not isinstance(role, str):
            role = ""
        frame = _frame(element)
        if role in wanted and frame:
            label = _label(element, role)
            if label:
                elements.append({
                    "index": len(elements) + 1,
                    "role": role,
                    "label": label,
                    "x": frame[0], "y": frame[1], "w": frame[2], "h": frame[3],
                    "enabled": _attr(element, "AXEnabled") is not False,
                    "focused": _attr(element, "AXFocused") is True,
                    "_ref": element,
                })
        if depth < _MAX_DEPTH:
            children = _attr(element, "AXChildren")
            if isinstance(children, list):
                stack.extend((child, depth + 1) for child in children)

    return {"ok": True, "app": _app_name(app), "window": title or "",
            "elements": elements, "truncated": truncated,
            "taken_at": time.time()}


def render(snap: dict[str, Any], limit: int = 40) -> str:
    """The listing as the model reads it: one control per line, numbered."""
    if not snap.get("ok"):
        return str(snap.get("reason") or "The accessibility tree is unavailable.")
    elements = snap.get("elements", [])[:limit]
    head = f"{snap.get('app', '?')}"
    if snap.get("window"):
        head += f" — window \"{snap['window']}\""
    if not elements:
        return (f"{head}\nNo controls are exposed by this window. It may draw "
                "its own interface (a canvas, a game, a screen share), in "
                "which case looking with vision is the only way to see it.")
    lines = [head]
    for element in elements:
        flags = ""
        if element.get("focused"):
            flags += " [focused]"
        if not element.get("enabled", True):
            flags += " [disabled]"
        lines.append(
            f"  {element['index']:>2} {element['role'][2:]:<14} "
            f"{element['label'][:60]!r} at ({element['x']},{element['y']}) "
            f"{element['w']}x{element['h']}{flags}")
    if snap.get("truncated"):
        lines.append("  … more controls exist than were listed; ask for a "
                     "higher limit or act on what is here.")
    lines.append("Act on one by its number: "
                 '{"name": "desktop_click", "arguments": {"element": 1}}')
    return "\n".join(lines)


def centre(element: dict[str, Any]) -> tuple[int, int]:
    return (int(element["x"] + element["w"] / 2),
            int(element["y"] + element["h"] / 2))


def press(element: dict[str, Any]) -> bool:
    """Ask the control to do its own default action. False if it has none.

    An AXPress is not a mouse click: it reaches the control whether or not it
    is on top, and it cannot land on whatever happens to be over it. Where it
    is refused, the caller falls back to clicking the frame's centre.
    """
    ref = element.get("_ref")
    if not ref:
        return False
    if "AXPress" not in _actions(ref):
        return False
    key = _cfstr("AXPress")
    try:
        return _ax.AXUIElementPerformAction(ref, key) == _AX_ERROR_SUCCESS
    finally:
        _cf.CFRelease(key)


def set_text(element: dict[str, Any], text: str) -> bool:
    """Put `text` into a field directly, rather than typing it at the screen.

    Typing goes to whatever holds focus, which is how a message meant for a
    composer ends up firing keyboard shortcuts at a page. Setting AXValue
    names the field it is going into.
    """
    ref = element.get("_ref")
    if not ref:
        return False
    value = _cfstr(text)
    key = _cfstr("AXValue")
    try:
        return _ax.AXUIElementSetAttributeValue(ref, key, value) == _AX_ERROR_SUCCESS
    finally:
        _cf.CFRelease(key)
        _cf.CFRelease(value)


def focus(element: dict[str, Any]) -> bool:
    """Give the control keyboard focus, so a following type lands in it."""
    ref = element.get("_ref")
    if not ref:
        return False
    true_ref = ctypes.c_void_p.in_dll(_cf, "kCFBooleanTrue")
    key = _cfstr("AXFocused")
    try:
        return _ax.AXUIElementSetAttributeValue(ref, key, true_ref) == _AX_ERROR_SUCCESS
    finally:
        _cf.CFRelease(key)


def value_of(element: dict[str, Any]) -> str:
    """What the control holds right now, read fresh from the tree."""
    ref = element.get("_ref")
    if not ref:
        return ""
    value = _attr(ref, "AXValue")
    return value if isinstance(value, str) else ""


def focused_element() -> dict[str, Any] | None:
    """The control that currently has keyboard focus, or None.

    This is the question to ask BEFORE typing: keys sent at a window with no
    text field focused are not discarded, they are shortcuts.
    """
    if _ax is None or not trusted():
        return None
    app = _focused_app()
    if not app:
        return None
    ref = _attr(app, "AXFocusedUIElement")
    if not ref:
        return None
    role = _attr(ref, "AXRole")
    role = role if isinstance(role, str) else ""
    frame = _frame(ref) or (0, 0, 0, 0)
    return {"role": role, "label": _label(ref, role), "x": frame[0],
            "y": frame[1], "w": frame[2], "h": frame[3],
            "takes_text": role in _TEXT_ROLES, "_ref": ref}


def request_trust() -> bool:
    """Ask macOS for the Accessibility grant, showing its dialog once.

    The dialog is the only way a user can be given the grant from here: the
    permission cannot be set programmatically, and without the prompt the
    setting is four levels deep in System Settings under a name that does not
    mention this program.
    """
    if _ax is None:
        return False
    try:
        _ax.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
        _ax.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
        _cf.CFDictionaryCreate.restype = ctypes.c_void_p
        _cf.CFDictionaryCreate.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_long,
            ctypes.c_void_p, ctypes.c_void_p]
        key = ctypes.c_void_p.in_dll(_ax, "kAXTrustedCheckOptionPrompt")
        true_ref = ctypes.c_void_p.in_dll(_cf, "kCFBooleanTrue")
        keys = (ctypes.c_void_p * 1)(key)
        values = (ctypes.c_void_p * 1)(true_ref)
        options = _cf.CFDictionaryCreate(None, keys, values, 1, None, None)
        return bool(_ax.AXIsProcessTrustedWithOptions(options))
    except Exception:
        return False


if __name__ == "__main__":  # pragma: no cover - a hand check, not a test
    import sys

    if not available():
        print(f"Accessibility API unavailable: {_LOAD_ERROR}")
        sys.exit(1)
    if not trusted():
        print("Not trusted yet — asking macOS to show the grant dialog.")
        request_trust()
        print(PERMISSION_HINT)
        sys.exit(1)
    print(render(snapshot(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 40)))
    focused = focused_element()
    if focused:
        print(f"\nfocused: {focused['role']} {focused['label']!r} "
              f"takes_text={focused['takes_text']}")
