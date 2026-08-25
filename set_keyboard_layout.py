#!/usr/bin/env python3
"""Read, and optionally switch, the active macOS keyboard layout.

There is no `defaults write` that does this. HIToolbox's preference keys are
owned by cfprefsd and writing them does not take effect; the only supported
route is Carbon's Text Input Sources API — TISEnableInputSource to make a
layout available, TISSelectInputSource to make it current. Both take effect
immediately and need no logout, no admin, and no Accessibility grant.

Defaults to a dry run: it resolves the layout and reports what it WOULD do.
Switching requires --apply, because this changes what every keystroke on the
machine means until it is switched back.

    set_layout.py                 # what is active now
    set_layout.py --list          # every installed layout
    set_layout.py Colemak         # dry run: resolve, report, change nothing
    set_layout.py Colemak --apply # actually switch
"""
import argparse
import ctypes
import sys

_carbon = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/Carbon.framework/Versions/A/Carbon")
_cf = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation")

# Every argtype spelled out. Without them ctypes truncates pointers to 32 bits
# and CFArrayGetCount is handed a garbage address — which crashes the
# interpreter with an ObjC selector error that names neither cause nor fix.
_cf.CFArrayGetCount.restype = ctypes.c_long
_cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
_cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
_cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
_cf.CFStringGetCString.restype = ctypes.c_bool
_cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                   ctypes.c_long, ctypes.c_uint32]
_carbon.TISCreateInputSourceList.restype = ctypes.c_void_p
_carbon.TISCreateInputSourceList.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_carbon.TISCopyCurrentKeyboardInputSource.restype = ctypes.c_void_p
_carbon.TISGetInputSourceProperty.restype = ctypes.c_void_p
_carbon.TISGetInputSourceProperty.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_carbon.TISEnableInputSource.restype = ctypes.c_int32
_carbon.TISEnableInputSource.argtypes = [ctypes.c_void_p]
_carbon.TISSelectInputSource.restype = ctypes.c_int32
_carbon.TISSelectInputSource.argtypes = [ctypes.c_void_p]

_UTF8 = 0x08000100


def _prop(src, key_name):
    key = ctypes.c_void_p.in_dll(_carbon, key_name)
    val = _carbon.TISGetInputSourceProperty(src, key)
    if not val:
        return None
    buf = ctypes.create_string_buffer(512)
    if _cf.CFStringGetCString(val, buf, 512, _UTF8):
        return buf.value.decode()
    return None


def layouts():
    """(name, source_id, ref) for every installed keyboard layout."""
    lst = _carbon.TISCreateInputSourceList(None, True)
    out = []
    for i in range(_cf.CFArrayGetCount(lst)):
        src = _cf.CFArrayGetValueAtIndex(lst, i)
        sid = _prop(src, "kTISPropertyInputSourceID") or ""
        if "keylayout" not in sid:
            continue
        out.append((_prop(src, "kTISPropertyLocalizedName") or "?", sid, src))
    return out


def current():
    src = _carbon.TISCopyCurrentKeyboardInputSource()
    return _prop(src, "kTISPropertyLocalizedName"), _prop(src, "kTISPropertyInputSourceID")


def resolve(want):
    want = want.lower()
    exact = [x for x in layouts() if x[0].lower() == want or x[1].lower() == want]
    if exact:
        return exact[0]
    near = [x for x in layouts() if want in x[0].lower() or want in x[1].lower()]
    if len(near) == 1:
        return near[0]
    if not near:
        raise SystemExit(f"No installed layout matches {want!r}. Try --list.")
    raise SystemExit(f"{want!r} is ambiguous: " + ", ".join(n for n, _, _ in near))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("layout", nargs="?", help="layout name or source id")
    ap.add_argument("--list", action="store_true", help="list installed layouts")
    ap.add_argument("--apply", action="store_true",
                    help="actually switch (without this it is a dry run)")
    args = ap.parse_args()

    if args.list:
        for name, sid, _ in sorted(layouts()):
            print(f"{name:24} {sid}")
        return 0

    now_name, now_id = current()
    if not args.layout:
        print(f"{now_name} ({now_id})")
        return 0

    name, sid, src = resolve(args.layout)
    if sid == now_id:
        print(f"Already on {name} ({sid}); nothing to do.")
        return 0

    if not args.apply:
        print(f"DRY RUN — would switch: {now_name} ({now_id}) -> {name} ({sid})")
        print(f"Re-run with --apply to switch. To undo: {sys.argv[0]} '{now_name}' --apply")
        return 0

    if _carbon.TISEnableInputSource(src) != 0:
        raise SystemExit(f"Could not enable {name}.")
    if _carbon.TISSelectInputSource(src) != 0:
        raise SystemExit(f"Could not select {name}.")
    after_name, after_id = current()
    ok = after_id == sid
    print(f"{'Switched' if ok else 'FAILED'}: {now_name} -> {after_name} ({after_id})")
    print(f"To undo: {sys.argv[0]} '{now_name}' --apply")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
