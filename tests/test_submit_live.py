#!/usr/bin/env python3
"""The submit chain, driven against a real browser and a real HTTP server.

Every other test of this path uses a fake page object, and a fake page answers
whatever it was written to answer. The bug this file exists for survived all of
them: `fill_form` reported "Filled 1/1 ... every field was read back and holds
its value" against a logged-out Hacker News — a page whose entire body is "You
have to be logged in to submit." and which contains zero <input> elements.

The rig below is Hacker News' submit form reduced to its shape: GET /submit
returns title and url inputs, POST redirects to /item?id=N, and flipping
`logged_in` serves the login wall instead. Nothing here touches the network.
"""
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import pytest

pytest.importorskip("playwright", reason="the live browser is an optional extra")

from symbio.computer import BrowserSession  # noqa: E402

FORM = b"""<html><body>
<form method="post" action="/r">
title:<input type="text" name="title" size="50">
url:<input type="text" name="url" size="50">
<input type="submit" value="submit">
</form></body></html>"""

WALL = b"<html><body>You have to be logged in to submit.</body></html>"


class _Rig:
    """An HN-shaped submit form on localhost."""

    def __init__(self):
        self.logged_in = True
        self.posted = None
        rig = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, location=None):
                self.send_response(code)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/item"):
                    title = (rig.posted or {}).get("title", [""])[0]
                    self._send(200, f"<html><body><a>{title}</a> | 1 point</body></html>".encode())
                elif not rig.logged_in:
                    self._send(200, WALL)
                else:
                    self._send(200, FORM)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                rig.posted = parse_qs(self.rfile.read(n).decode())
                self._send(302, b"", location="/item?id=1001")

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self._srv = HTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self._srv.shutdown()


@pytest.fixture
def rig():
    r = _Rig()
    yield r
    r.stop()


@pytest.fixture
def browser():
    session = BrowserSession(confirm_fn=lambda *a, **k: True)
    try:
        yield session
    finally:
        try:
            session.close()
        except Exception:
            pass


def test_the_whole_submit_chain_against_a_real_form(rig, browser):
    """Open, fill by selector, submit, and be told CONFIRMED only because the
    code watched the URL land on the story page."""
    browser.open(f"{rig.url}/submit")

    filled = browser.fill_form({
        "input[name='title']": "Symbio: a self-finetuning local agent",
        "input[name='url']": "https://github.com/huyedits/Symbio",
    })
    verdict = browser.submit_form(target="submit",
                                  expected_url=f"{rig.url}/item?id=")

    assert "Filled 2/2" in filled
    assert verdict.startswith("[Submit CONFIRMED")
    # The server is the witness: the form really arrived, with both values.
    assert rig.posted["title"] == ["Symbio: a self-finetuning local agent"]
    assert rig.posted["url"] == ["https://github.com/huyedits/Symbio"]


def test_a_login_wall_is_not_a_filled_form(rig, browser):
    """The bug. Zero inputs on the page, and fill_form said every field was
    read back and holds its value."""
    rig.logged_in = False
    browser.open(f"{rig.url}/submit")

    filled = browser.fill_form({"input[name='title']": "anything"})

    assert "Filled" not in filled
    assert "no element on this page matches it" in filled
    assert "Do not submit this form yet" in filled
    assert rig.posted is None


def test_nothing_is_submitted_from_a_page_without_the_form(rig, browser):
    rig.logged_in = False
    browser.open(f"{rig.url}/submit")

    verdict = browser.submit_form(target="submit",
                                  expected_url=f"{rig.url}/item?id=")

    assert not verdict.startswith("[Submit CONFIRMED")
    assert rig.posted is None


def test_a_second_session_in_the_same_process_still_gets_a_browser(rig):
    """close() keeps the Playwright driver alive so reopening is fast — and it
    kept it on the INSTANCE, so a second BrowserSession started a second sync
    driver beside the first and Playwright refused:

        Browser open error: It looks like you are using Playwright Sync API
        inside the asyncio loop.

    After that every browser tool answers "Browser is not open. Load the target
    URL first" to a model that just loaded it. Any task needing more than one
    visit — open Hacker News, hit the login wall, close, come back — died
    there.
    """
    first = BrowserSession(confirm_fn=lambda *a, **k: True)
    first.open(f"{rig.url}/submit")
    first.close()

    second = BrowserSession(confirm_fn=lambda *a, **k: True)
    try:
        opened = second.open(f"{rig.url}/submit")

        assert "error" not in opened.lower(), opened
        assert second.is_open
        assert "Filled 1/1" in second.fill_form({"input[name='title']": "back again"})
    finally:
        second.close()
