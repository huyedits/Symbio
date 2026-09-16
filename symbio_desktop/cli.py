"""Symbio Desktop — launch the window.

Two shapes, one server. `--window` opens a native macOS window around the page
(WKWebView, no Electron, no bundled browser); without it the page opens in the
browser that is already running, which costs this project nothing at all.

The budget is the point. The whole desktop — this process and, with --window,
the WebKit views it hosts — is meant to sit in tens of megabytes, not the
hundreds an Electron shell starts at. The model is not in here: it lives in
`symb daemon`, and this talks to it over a socket.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="symbio-desktop", description="Symbio Desktop — chat, adapters, health")
    parser.add_argument("--port", type=int, default=8742, help="Port (default 8742)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--window", action="store_true",
                        help="Open a native macOS window instead of a browser tab")
    parser.add_argument("--no-open", action="store_true", help="Serve only")
    args = parser.parse_args()

    from symbio_desktop.server import serve

    url = f"http://{args.host}:{args.port}"
    threading.Thread(target=serve, args=(args.host, args.port), daemon=True).start()
    # The socket is listening before the page is asked for, so nothing races
    # the first request.
    time.sleep(0.3)

    print(f"\n  Symbio Desktop  →  {url}")
    # Through the server's own check, not symbio.app.daemon: importing that
    # runs symbio/__init__.py, which is 105 MB of agent stack, in the process
    # whose whole selling point is that it does not carry one.
    from symbio_desktop.server import DaemonBridge

    print("  Resident model: ready" if DaemonBridge.daemon_ready()
          else "  Resident model: not running — start it with `symb daemon start`")
    print("  Ctrl+C to stop\n")

    if args.window:
        from symbio_desktop.window import open_window

        return open_window(url)
    if not args.no_open:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
