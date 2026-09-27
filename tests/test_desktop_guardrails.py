"""The window's side of the guardrails: who may talk to its server, and how an
approval card travels between the daemon and the page.

The server listens on localhost, and every web page in every browser on the
Mac can reach localhost. A WebSocket is exempt from the same-origin policy, so
until these checks any site could open /ws/chat, type to an agent that runs
shell commands, and answer its approval cards; and a form posted as text/plain
reached /api/settings.
"""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from symbio_desktop import server


@pytest.mark.parametrize("headers,websocket,writes,refused", [
    ({"Host": "127.0.0.1:8742", "Origin": "http://127.0.0.1:8742"}, True, False, False),
    ({"Host": "localhost:8742", "Origin": "http://localhost:8742"}, True, False, False),
    ({"Host": "127.0.0.1:8742"}, True, False, False),                     # not a browser
    ({"Host": "127.0.0.1:8742", "Origin": "https://evil.example"}, True, False, True),
    ({"Host": "127.0.0.1:8742", "Origin": "null"}, True, False, True),    # sandboxed frame
    ({"Host": "evil.example:8742"}, False, False, True),                  # DNS rebinding
    ({"Host": "127.0.0.1:8742", "Origin": "http://127.0.0.1:8742",
      "Content-Type": "text/plain"}, False, True, True),                  # a cross-site form
    ({"Host": "127.0.0.1:8742", "Origin": "http://127.0.0.1:8742",
      "Content-Type": "application/json"}, False, True, False),
])
def test_only_this_window_is_let_in(headers, websocket, writes, refused):
    why = server.request_refusal(headers, 8742, websocket=websocket, writes=writes)
    assert bool(why) is refused, why


@pytest.fixture
def served(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(server.constants, "CONFIG_FILE", config)
    monkeypatch.setattr(server.constants, "LOG_DIR", tmp_path / "logs")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], config
    httpd.shutdown()


def _request(port, method, path, headers, body=b""):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_a_web_page_cannot_open_the_chat_socket(served):
    port, _config = served

    status, _ = _request(port, "GET", "/ws/chat", {
        "Host": f"127.0.0.1:{port}", "Origin": "https://evil.example",
        "Upgrade": "websocket", "Connection": "Upgrade",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==", "Sec-WebSocket-Version": "13"})

    assert status == 403


def test_a_forged_form_cannot_loosen_the_guardrails(served):
    port, config = served
    forged = b'{"kind": "publish", "mode": "allow", "x": "="}'

    status, _ = _request(port, "POST", "/api/guardrails", {
        "Host": f"127.0.0.1:{port}", "Origin": "https://evil.example",
        "Content-Type": "text/plain", "Content-Length": str(len(forged))}, forged)

    assert status == 403
    assert json.loads(config.read_text()) == {}


def test_the_panel_reads_and_writes_the_guardrails(served):
    port, config = served
    body = b'{"kind": "publish", "mode": "block"}'

    status, data = _request(port, "POST", "/api/guardrails", {
        "Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}",
        "Content-Type": "application/json", "Content-Length": str(len(body))}, body)
    assert status == 200 and json.loads(data)["ok"] is True
    assert json.loads(config.read_text())["guardrails"]["modes"]["publish"] == "block"

    status, data = _request(port, "GET", "/api/guardrails", {"Host": f"127.0.0.1:{port}"})
    panel = json.loads(data)
    assert status == 200
    assert {k["id"]: k["mode"] for k in panel["kinds"]}["publish"] == "block"
    assert panel["floors"] and panel["modes"]


def test_the_panel_writes_nothing_but_a_known_kind_and_mode(served):
    port, config = served
    body = b'{"kind": "model_name", "mode": "allow"}'

    _status, data = _request(port, "POST", "/api/guardrails", {
        "Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}",
        "Content-Type": "application/json", "Content-Length": str(len(body))}, body)

    assert json.loads(data)["ok"] is False
    assert json.loads(config.read_text()) == {}


class _Pipe:
    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


_CARD = {"headline": "Run a shell command on this Mac.", "details": "$ ls",
         "kind": "commands", "kind_label": "Run commands and code", "reason": "",
         "said_by": "harness", "warning": ""}


def test_a_pending_card_is_sent_again_to_a_window_that_reloads():
    """The daemon's confirm_fn is a real thread waiting on this answer. A card
    that vanished with the old page left the turn hanging, nothing to click."""
    first, second = [], []
    bridge = server.DaemonBridge(first.append)
    bridge.ready = True
    bridge.rfile = _Pipe([json.dumps({"type": "confirm", "prompt": "x",
                                      "card": _CARD}).encode() + b"\n"])
    bridge._pump()
    bridge.detach()
    bridge.attach(second.append)

    assert first[0]["card"] == _CARD
    assert second and second[0]["type"] == "confirm" and second[0]["card"] == _CARD


def test_an_answered_card_is_not_sent_again():
    later = []
    bridge = server.DaemonBridge(lambda _m: None)
    bridge.ready = True
    bridge.rfile = _Pipe([json.dumps({"type": "confirm", "prompt": "x",
                                      "card": _CARD}).encode() + b"\n"])
    bridge._pump()
    bridge.confirm(False)
    bridge.attach(later.append)

    assert later == []


def test_always_allow_is_passed_through_only_with_a_yes():
    written = []

    class _Out:
        def write(self, data):
            written.append(data)

        def flush(self):
            pass

    bridge = server.DaemonBridge(lambda _msg: None)
    bridge.wfile = _Out()
    bridge.confirm(True, always=True)
    bridge.confirm(False, always=True)

    messages = [json.loads(line) for line in b"".join(written).splitlines()]
    assert messages == [{"type": "confirm", "answer": True, "always": True},
                        {"type": "confirm", "answer": False}]


def test_the_daemon_sends_the_cards_parts_and_writes_always_allow(tmp_path, monkeypatch):
    """confirm_fn in the daemon: a Card goes out with its parts, and "always"
    on a yes switches that kind to allow in config.json."""
    import inspect

    from symbio.app import daemon

    source = inspect.getsource(daemon._serve_connection)
    confirm = source.split("def confirm_fn")[1].split("\n    def ")[0]
    assert 'frame["card"] = prompt.as_dict()' in confirm
    assert 'guardrails.set_mode(constants.CONFIG_FILE, prompt.kind, "allow")' in confirm
    assert "session._guardrails_file = constants.CONFIG_FILE" in source
