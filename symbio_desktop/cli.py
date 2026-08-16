"""Symbio Desktop CLI — launch the premium adapter ecosystem visualizer."""

from __future__ import annotations

import argparse
import webbrowser
import time
import threading


def main():
    parser = argparse.ArgumentParser(
        prog="symbio-desktop",
        description="Symbio Desktop — Adapter Ecosystem Mind Map",
    )
    parser.add_argument(
        "--port", type=int, default=8742,
        help="Port to listen on (default: 8742)",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't open the browser automatically",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    import uvicorn

    if not args.no_open:
        def _open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║          🧠  Symbio Desktop  v1.0            ║")
    print("  ║    Adapter Ecosystem • Mind Map • RAG        ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    print(f"  →  http://{args.host}:{args.port}")
    print()
    print("  Press Ctrl+C to stop")
    print()

    uvicorn.run(
        "symbio_desktop.server:app",
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
