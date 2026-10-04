"""symbio/obs.py against a fake OBS that speaks the real protocol.

The fake does what obs-websocket v5 does on the wire: the RFC 6455 handshake,
a Hello carrying an auth challenge, a close with code 4009 on a wrong
password, and Record requests that change state. So these tests prove the
client's framing and authentication, not just its string formatting.
"""

import base64
import hashlib
import json
import socket
import struct
import threading

import pytest

from symbio import obs

PASSWORD = "hunter2-hunter2!"


class FakeObs:
    def __init__(self, password=PASSWORD):
        self.password = password
        self.recording = False
        self.requests = []
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    # -- wire helpers (server side: unmasked out, masked in) --
    @staticmethod
    def _send(conn, obj=None, opcode=0x1, raw=b""):
        data = json.dumps(obj).encode() if obj is not None else raw
        head = bytearray([0x80 | opcode])
        head += bytes([len(data)]) if len(data) < 126 else b"\x7e" + struct.pack(">H", len(data))
        conn.sendall(bytes(head) + data)

    @staticmethod
    def _recv(conn):
        def read(n):
            buf = b""
            while len(buf) < n:
                chunk = conn.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError
                buf += chunk
            return buf
        b0, b1 = read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", read(2))[0]
        assert b1 & 0x80, "client frames must be masked"
        mask = read(4)
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(read(n)))
        if b0 & 0x0F == 0x8:
            raise ConnectionError
        return json.loads(data)

    def _serve(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        try:
            req = b""
            while b"\r\n\r\n" not in req:
                req += conn.recv(4096)
            key = [l.split(":", 1)[1].strip() for l in req.decode().split("\r\n")
                   if l.lower().startswith("sec-websocket-key")][0]
            accept = base64.b64encode(hashlib.sha1((key + obs._WS_GUID).encode()).digest()).decode()
            conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                          f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n").encode())
            salt, challenge = "c2FsdA==", "Y2hhbGxlbmdl"
            self._send(conn, {"op": 0, "d": {"rpcVersion": 1,
                                              "authentication": {"salt": salt, "challenge": challenge}}})
            ident = self._recv(conn)
            if ident["d"].get("authentication") != obs._auth(self.password, salt, challenge):
                self._send(conn, opcode=0x8, raw=struct.pack(">H", 4009) + b"Authentication failed.")
                return
            self._send(conn, {"op": 2, "d": {"negotiatedRpcVersion": 1}})
            while True:
                msg = self._recv(conn)["d"]
                kind = msg["requestType"]
                self.requests.append(kind)
                data = {}
                if kind == "GetRecordStatus":
                    data = {"outputActive": self.recording, "outputPaused": False,
                            "outputTimecode": "00:00:42.000"}
                elif kind == "StartRecord":
                    self.recording = True
                elif kind == "StopRecord":
                    self.recording = False
                    data = {"outputPath": "/Users/me/Movies/2026-09-25 12-00-00.mov"}
                self._send(conn, {"op": 7, "d": {"requestType": kind, "requestId": msg["requestId"],
                                                  "requestStatus": {"result": True, "code": 100},
                                                  "responseData": data}})
        except (ConnectionError, OSError, IndexError):
            pass
        finally:
            conn.close()

    def config(self, password=PASSWORD):
        return {"obs": {"host": "127.0.0.1", "port": self.port, "password": password}}


@pytest.fixture
def fake():
    server = FakeObs()
    yield server
    server.srv.close()


def test_start_then_stop_reports_the_saved_file(fake):
    assert obs.record("status", fake.config()) == "[OBS] Not recording."
    assert obs.record("start", fake.config()).startswith("[Recording STARTED]")
    assert fake.recording
    out = obs.record("stop", fake.config())
    assert out == "[Recording STOPPED] Saved to /Users/me/Movies/2026-09-25 12-00-00.mov"
    assert not fake.recording
    # STOPPED is only said after OBS itself reported no recording running.
    assert fake.requests[-2:] == ["StopRecord", "GetRecordStatus"]


def test_stop_when_not_recording_says_so(fake):
    assert obs.record("stop", fake.config()) == "[OBS] Nothing to stop: OBS is not recording."
    assert "StopRecord" not in fake.requests


def test_a_wrong_password_is_named(fake):
    out = obs.record("status", fake.config(password="wrong"))
    assert out.startswith("[OBS unavailable]") and "rejected the password" in out


def test_nothing_listening_points_at_the_setting():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    out = obs.record("stop", {"obs": {"port": port, "password": "x"}},
                     obs_file=__import__("pathlib").Path("/nonexistent"))
    assert out.startswith("[OBS unavailable]") and "Enable WebSocket server" in out


def test_the_password_comes_from_obs_own_settings(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("SYMBIO_OBS_PASSWORD", raising=False)
    local = tmp_path / "config.json"
    local.write_text(json.dumps({"auth_required": True, "server_enabled": True,
                                 "server_password": PASSWORD, "server_port": fake.port}))
    assert obs.record("status", {}, obs_file=local) == "[OBS] Not recording."


def test_an_unknown_action_is_refused_without_connecting():
    assert obs.record("pause", {}).startswith("Unknown action")


def test_obs_record_is_advertised_and_runnable_in_both_stacks(monkeypatch):
    """The catalog and both dispatchers — see the two-registries note."""
    from symbio import tools as agent_tools
    from symbio.app import chat_tools, tooling

    assert "obs_record" in {t["name"] for t in tooling._BUILTIN_TOOLS}
    calls = []
    monkeypatch.setattr(obs, "record", lambda action, config=None: calls.append(action) or "ok")

    class S(chat_tools.ToolsMixin):
        config = {}
        enabled_groups = None

    assert S()._desktop_action("obs_record", {"action": "stop"}) == "ok"
    assert calls == ["stop"]
    registry = {t["name"]: t for t in agent_tools.build_tool_registry(object())}
    assert "obs_record" in registry, "advertised to the AIAgent loop but not runnable there"
