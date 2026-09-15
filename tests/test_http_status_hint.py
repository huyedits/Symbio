"""HTTP failures have to be visible as failures.

Driving the CLI on 2026-08-27 against an API that had moved (410) and was rate
limiting (429), the model wrote `requests.get(...).json()['items']` three times
and never read `.status_code`. requests hands back a response object for 4xx
just as it does for 200, so both obstacles surfaced only as a KeyError on the
error document — and the model read that as "wrong key" and hunted for another
one until the tool-round budget ran out.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from symbio.app import sandbox

LIVE = ("import requests\n"
        "response = requests.get('http://127.0.0.1:8787/items')\n"
        "data = response.json()\n"
        "print(data['items'])")
KEYERROR = "Traceback (most recent call last):\nKeyError: 'items'"


def test_hint_fires_on_the_live_failure():
    hint = sandbox._stub_hint_for_failure(LIVE, KEYERROR)
    assert "never checked the HTTP status" in hint
    assert "Retry-After" in hint


@pytest.mark.parametrize("check", [
    "if response.status_code != 200: raise SystemExit(response.text)",
    "response.raise_for_status()",
    "if not response.ok: print(response.text)",
])
def test_silent_when_the_script_already_checks(check):
    code = LIVE.replace("data = response.json()", check + "\ndata = response.json()")
    assert sandbox._stub_hint_for_failure(code, KEYERROR) == ""


def test_silent_without_an_http_call():
    code = "data = {'a': 1}\nprint(data['items'])"
    assert sandbox._stub_hint_for_failure(code, KEYERROR) == ""


def test_silent_on_an_unrelated_error():
    assert sandbox._stub_hint_for_failure(LIVE, "ZeroDivisionError: division by zero") == ""


def test_import_hint_still_works_alongside():
    """The pre-existing stub-alternative hint must not be displaced."""
    code = "import selectolax\nrequests.get('http://x')"
    out = sandbox._stub_hint_for_failure(code, "ModuleNotFoundError: No module named 'selectolax'")
    assert "from symbio_tools import" in out


# ---- the fetch() stub ----

class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"detail": "/items was removed in v2. Use /v2/items."}).encode()
        self.send_response(410)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", "7")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def gone_server():
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/items"
    srv.shutdown()


def test_fetch_surfaces_status_body_and_retry_after(gone_server, tmp_path, monkeypatch):
    """urlopen alone raises 'HTTP Error 410: Gone' and discards the body —
    which is the one place the API says where the endpoint went."""
    monkeypatch.setattr(sandbox.constants, "SANDBOX_DIR", tmp_path)
    config = {"sandbox": {"blocked_imports": ["os", "urllib"]},
              "agent": {"code_timeout": 30, "max_output_len": 4000}}
    ok, out = sandbox.run_python_code(
        f"from symbio_tools import fetch\nprint(fetch({gone_server!r}))", config)
    assert not ok
    assert "HTTP 410" in out
    assert "Use /v2/items" in out
    assert "Retry-After: 7s" in out
