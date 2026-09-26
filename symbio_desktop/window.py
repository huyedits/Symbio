"""A native macOS window around the desktop page — WKWebView, no Electron.

An Electron shell is a second copy of Chromium: 200-400 MB resident before it
has drawn anything, for a page that is a sidebar, a thread and a text box.
WebKit is already on the machine and already running; asking for a view of it
costs a window.

Built on pyobjc-core alone, through `objc.loadBundle`, because the framework
wrapper packages (pyobjc-framework-WebKit and friends) are not in
requirements.txt and this ships to other people's Macs. loadBundle pulls the
classes out of the system framework at runtime, which needs no wrapper.

If any of that is missing, `open_window` says why and falls back to the
browser rather than failing the launch.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

ICON = Path(__file__).parent / "static" / "icon.png"


def _main_menu(NSMenu, NSMenuItem):
    """The menus every Mac app has. Without an Edit menu, ⌘C, ⌘V and ⌘A do
    nothing in the page's text box: AppKit routes those keys through it."""
    bar = NSMenu.alloc().init()
    for title, entries in (
            ("Symbio", (("Hide Symbio", "hide:", "h"), None,
                        ("Quit Symbio", "terminate:", "q"))),
            ("Edit", (("Undo", "undo:", "z"), ("Redo", "redo:", "Z"), None,
                      ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                      ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a"))),
            ("Window", (("Minimize", "performMiniaturize:", "m"),
                        ("Close", "performClose:", "w")))):
        top = NSMenuItem.alloc().init()
        menu = NSMenu.alloc().initWithTitle_(title)
        for entry in entries:
            if entry is None:
                menu.addItem_(NSMenuItem.separatorItem())
            else:
                menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(*entry))
        top.setSubmenu_(menu)
        bar.addItem_(top)
    return bar


def available() -> tuple[bool, str]:
    """Can a native window be opened here, and if not, why not."""
    try:
        import objc  # noqa: F401
        import Foundation  # noqa: F401
    except ImportError as e:
        return False, (f"No native window here ({e.name} is missing). "
                       "`pip install pyobjc-core pyobjc-framework-Cocoa` adds "
                       "it, or run without --window to use your browser — "
                       "which is the cheaper option anyway: a browser tab "
                       "costs this project nothing, and hosting WebKit in "
                       "this process costs about 400 MB.")
    return True, ""


def open_window(url: str, width: int = 1180, height: int = 800) -> int:
    """Run a Cocoa app whose whole content is a web view of `url`. Blocks."""
    ok, why = available()
    if not ok:
        print(f"  {why}")
        webbrowser.open(url)
        return 0

    import objc
    from Foundation import NSBundle, NSObject, NSProcessInfo, NSURL, NSURLRequest, NSMakeRect

    # AppKit and WebKit classes by name: pyobjc-core resolves them out of the
    # loaded frameworks without their wrapper packages.
    namespace: dict = {}
    objc.loadBundle("AppKit", namespace,
                    bundle_path="/System/Library/Frameworks/AppKit.framework")
    objc.loadBundle("WebKit", namespace,
                    bundle_path="/System/Library/Frameworks/WebKit.framework")
    NSApplication = namespace["NSApplication"]
    NSWindow = namespace["NSWindow"]
    WKWebView = namespace.get("WKWebView")
    if WKWebView is None:
        print("  WebKit did not provide WKWebView; opening a browser instead.")
        webbrowser.open(url)
        return 0

    # An app called Symbio, not "Python": the menu bar reads the bundle name,
    # which has to be set before the application object exists.
    info = NSBundle.mainBundle().localizedInfoDictionary() or NSBundle.mainBundle().infoDictionary()
    if info is not None:
        info["CFBundleName"] = "Symbio"
    NSProcessInfo.processInfo().setProcessName_("Symbio")

    app = NSApplication.sharedApplication()
    # Regular, so it gets a Dock icon, a menu bar and keyboard focus. An
    # accessory app would open a window nobody can type into.
    app.setActivationPolicy_(0)
    icon = namespace["NSImage"].alloc().initWithContentsOfFile_(str(ICON))
    if icon is not None:
        app.setApplicationIconImage_(icon)
    app.setMainMenu_(_main_menu(namespace["NSMenu"], namespace["NSMenuItem"]))

    style = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)  # titled, closable, mini, resizable
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, width, height), style, 2, False)
    window.setTitle_("Symbio")
    window.center()
    # Where the user last left it, and how big.
    window.setFrameAutosaveName_("SymbioChat")

    view = WKWebView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
    view.setAutoresizingMask_((1 << 1) | (1 << 4))  # width | height
    view.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(url)))
    window.setContentView_(view)

    class _Delegate(NSObject):
        def applicationShouldTerminateAfterLastWindowClosed_(self, _sender):
            return True

    delegate = _Delegate.alloc().init()
    app.setDelegate_(delegate)
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)
    app.run()
    return 0
