"""Talking to the resident daemon, off the UI thread.

The daemon already speaks a JSON-lines protocol over a Unix socket — the same
one `symb chat` and the desktop window use — so this is a client, not a second
brain. Message types: output, stream, status, input_prompt, confirm, done.

The socket is read on a worker thread and every frame is handed to the app
through a callback. Nothing here touches a widget: Textual owns the UI thread,
and a socket read that blocks it would freeze the input box mid-turn, which is
the one thing a persistent input box must never do.
"""
from __future__ import annotations

import socket
import threading
from typing import Any, Callable

from symbio import constants
from symbio.app.daemon import _decode_msg, _encode_msg


class DaemonLink:
    """A connection to the resident model, with its reader on its own thread."""

    def __init__(self, on_frame: Callable[[dict], None],
                 on_close: Callable[[str], None]):
        self._on_frame = on_frame
        self._on_close = on_close
        self._sock: socket.socket | None = None
        self._wfile: Any = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> str:
        """Returns "" on success, or why it could not connect."""
        path = constants.DAEMON_SOCKET
        if not path.exists():
            return (f"No daemon socket at {path}. Start one with "
                    f"`symb daemon start`.")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(path))
        except OSError as e:
            return f"Could not reach the daemon ({e})."
        self._sock = sock
        self._wfile = sock.makefile("wb")
        rfile = sock.makefile("rb")

        def read_forever():
            try:
                while not self._stop.is_set():
                    line = rfile.readline()
                    if not line:
                        break
                    self._on_frame(_decode_msg(line))
            except Exception as e:
                self._on_close(str(e))
                return
            self._on_close("the daemon closed the connection")

        self._reader = threading.Thread(target=read_forever, daemon=True)
        self._reader.start()
        return ""

    def send(self, message: dict) -> None:
        if self._wfile is None:
            return
        try:
            self._wfile.write(_encode_msg(message))
            self._wfile.flush()
        except OSError as e:
            self._on_close(f"write failed ({e})")

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
