"""Drive OBS Studio's recording over its WebSocket server (protocol v5).

Why this exists: a task the user is filming should be able to end itself. The
first use is a recorded demo in which Symbio drives the browser to a finished
result and then stops the recording, so the clip ends on the result and not on
a person reaching for the mouse.

Why the WebSocket server and not a hotkey or a click: OBS has shipped
obs-websocket built in since version 28, and it answers with a real result —
StopRecord returns the path of the file it just wrote, and GetRecordStatus
says whether a recording is running. A hotkey sends a key and hopes; a click
on the OBS window pulls it in front of the thing being recorded.

Standard library only, like the desktop window's server: the client side of
RFC 6455 is a handshake and a frame format, and adding a dependency for it
would put one more package on every install for a feature most will not use.

Connection settings, first match wins:
  config["obs"]["host"/"port"/"password"]
  SYMBIO_OBS_PASSWORD
  OBS's own obs-websocket config on this machine (same user, same Mac): the
  port and password OBS itself was given. Nothing is written to that file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
DEFAULT_PORT = 4455
TIMEOUT = 5.0

ENABLE_HINT = ("In OBS: Tools -> WebSocket Server Settings -> tick 'Enable "
               "WebSocket server' (it is off by default), then try again.")


class ObsError(RuntimeError):
    """Anything that stops a request; the message is written for the user."""


# ------------------------------------------------------------------ settings

def obs_config_path() -> Path:
    """Where OBS keeps obs-websocket's settings for this user."""
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support" / "obs-studio"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", home)) / "obs-studio"
    else:
        base = home / ".config" / "obs-studio"
    return base / "plugin_config" / "obs-websocket" / "config.json"


def settings(config: dict[str, Any] | None = None,
             obs_file: Path | None = None) -> dict[str, Any]:
    cfg = (config or {}).get("obs", {}) or {}
    local: dict[str, Any] = {}
    path = obs_file or obs_config_path()
    try:
        local = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    password = (cfg.get("password") or os.environ.get("SYMBIO_OBS_PASSWORD")
                or (local.get("server_password") if local.get("auth_required", True) else "")
                or "")
    return {
        "host": cfg.get("host") or "127.0.0.1",
        "port": int(cfg.get("port") or local.get("server_port") or DEFAULT_PORT),
        "password": password,
        "server_enabled": local.get("server_enabled"),
    }


# ------------------------------------------------------------------ websocket

class _Socket:
    """The client half of RFC 6455: one handshake, masked frames out."""

    def __init__(self, host: str, port: int, timeout: float = TIMEOUT):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: obswebsocket.json\r\n\r\n"
        ).encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ObsError("OBS closed the connection during the handshake.")
            head += chunk
        head, self._buf = head.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        if " 101 " not in lines[0] + " ":
            raise ObsError(f"OBS refused the WebSocket upgrade: {lines[0]}")
        expect = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        headers = {l.split(":", 1)[0].strip().lower(): l.split(":", 1)[1].strip()
                   for l in lines[1:] if ":" in l}
        if headers.get("sec-websocket-accept") != expect:
            raise ObsError("OBS answered the handshake with the wrong accept key.")

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ObsError("OBS closed the connection.")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        head = bytearray([0x81])
        n = len(data)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        self.sock.sendall(bytes(head) + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self) -> dict[str, Any]:
        message = b""
        while True:
            b0, b1 = self._read(2)
            opcode, n = b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else b""
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 0x8:
                code = struct.unpack(">H", data[:2])[0] if len(data) >= 2 else 0
                raise _Closed(code, data[2:].decode("utf-8", "replace"))
            if opcode == 0x9:           # ping -> pong
                self.sock.sendall(bytes([0x8A, 0x80 | len(data)]) + b"\0\0\0\0" + data)
                continue
            if opcode in (0x1, 0x0):
                message += data
                if b0 & 0x80:
                    return json.loads(message)

    def close(self) -> None:
        try:
            self.sock.sendall(bytes([0x88, 0x80]) + b"\0\0\0\0")
        except OSError:
            pass
        self.sock.close()


class _Closed(ObsError):
    def __init__(self, code: int, reason: str):
        super().__init__(f"OBS closed the connection ({code}: {reason})")
        self.code = code


def _auth(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


# ------------------------------------------------------------------ requests

class Session:
    """One identified connection; several requests can share it."""

    def __init__(self, config: dict[str, Any] | None = None, obs_file: Path | None = None):
        s = settings(config, obs_file)
        where = f"{s['host']}:{s['port']}"
        try:
            self.ws = _Socket(s["host"], s["port"])
        except OSError as e:
            off =" OBS's settings say the server is switched off." if s["server_enabled"] is False else ""
            raise ObsError(f"Could not reach OBS at {where} ({e.__class__.__name__}). Is OBS "
                           f"open?{off} {ENABLE_HINT}") from None
        try:
            hello = self.ws.recv()
            identify: dict[str, Any] = {"rpcVersion": 1, "eventSubscriptions": 0}
            auth = (hello.get("d") or {}).get("authentication")
            if auth:
                if not s["password"]:
                    raise ObsError("OBS wants a password and none is configured. Set "
                                   "obs.password in config.json or SYMBIO_OBS_PASSWORD.")
                identify["authentication"] = _auth(s["password"], auth["salt"], auth["challenge"])
            self.ws.send({"op": 1, "d": identify})
            reply = self.ws.recv()
            if reply.get("op") != 2:
                raise ObsError(f"OBS did not accept the connection: {reply}")
        except _Closed as e:
            self.ws.close()
            if e.code == 4009:
                raise ObsError("OBS rejected the password. Check obs.password against OBS -> "
                               "Tools -> WebSocket Server Settings -> Show Connect Info.") from None
            raise
        except ObsError:
            self.ws.close()
            raise

    def request(self, request_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        rid = uuid.uuid4().hex
        body: dict[str, Any] = {"requestType": request_type, "requestId": rid}
        if data:
            body["requestData"] = data
        self.ws.send({"op": 6, "d": body})
        while True:
            msg = self.ws.recv()
            d = msg.get("d") or {}
            if msg.get("op") == 7 and d.get("requestId") == rid:
                status = d.get("requestStatus") or {}
                if not status.get("result"):
                    raise ObsError(f"OBS refused {request_type}: "
                                   f"{status.get('comment') or status.get('code')}")
                return d.get("responseData") or {}

    def close(self) -> None:
        self.ws.close()


def _status(session: Session) -> dict[str, Any]:
    return session.request("GetRecordStatus")


def _wait_for(session: Session, active: bool, seconds: float = 3.0) -> dict[str, Any]:
    deadline = time.time() + seconds
    while True:
        st = _status(session)
        if bool(st.get("outputActive")) == active or time.time() > deadline:
            return st
        time.sleep(0.2)


def record(action: str, config: dict[str, Any] | None = None,
           obs_file: Path | None = None) -> str:
    """start | stop | status. Every answer is read back from OBS, not assumed.

    "stop" reports the file OBS wrote, and only says STOPPED once OBS itself
    reports no recording running: the same rule as submit_form, where a click
    is not a submission until the result is seen.
    """
    action = (action or "status").strip().lower()
    if action not in ("start", "stop", "status"):
        return f"Unknown action '{action}'. Use start, stop or status."
    try:
        session = Session(config, obs_file)
    except ObsError as e:
        return f"[OBS unavailable] {e}"
    try:
        st = _status(session)
        running = bool(st.get("outputActive"))
        if action == "status":
            if not running:
                return "[OBS] Not recording."
            return (f"[OBS] Recording{' (paused)' if st.get('outputPaused') else ''}, "
                    f"{st.get('outputTimecode', '?')} so far.")
        if action == "start":
            if running:
                return f"[OBS] Already recording ({st.get('outputTimecode', '?')} so far)."
            session.request("StartRecord")
            st = _wait_for(session, True)
            if not st.get("outputActive"):
                return "[Recording NOT started] OBS accepted StartRecord but reports no recording."
            return "[Recording STARTED] OBS is recording."
        # stop
        if not running:
            return "[OBS] Nothing to stop: OBS is not recording."
        out = session.request("StopRecord")
        st = _wait_for(session, False)
        if st.get("outputActive"):
            return "[Recording NOT stopped] OBS still reports a recording running."
        path = out.get("outputPath") or "the OBS recording folder"
        return f"[Recording STOPPED] Saved to {path}"
    except ObsError as e:
        return f"[OBS error] {e}"
    finally:
        session.close()
