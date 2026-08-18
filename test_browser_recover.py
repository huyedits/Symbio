"""Reopening the page when the model clicks at a browser that is not open.

This is the largest tool failure in the system by volume. In the local
activity log: 567 browser_click calls, 453 failures, and 450 of those were
"Browser is not open". _last_browsed_url existed for exactly this recovery
since the beginning and was never read.
"""
import pytest

from symbio.app import chat


class FakeBrowser:
    """Open/closed browser whose actions fail while closed, like the real one."""

    def __init__(self, is_open=False, reopen_result="Opened."):
        self.is_open = is_open
        self.reopen_result = reopen_result
        self.opened: list[str] = []
        self.clicks = 0

    def open(self, url):
        self.opened.append(url)
        if "blocked" in self.reopen_result or "error" in self.reopen_result.lower():
            return self.reopen_result
        self.is_open = True
        return self.reopen_result

    def click(self, selector="", text=""):
        self.clicks += 1
        if not self.is_open:
            return "Browser click error: Browser is not open."
        return f"Clicked {text or selector}."

    def type_text(self, text, press_enter=False):
        return "Typed." if self.is_open else "Browser is not open."

    def scroll(self, direction="down"):
        return "Scrolled." if self.is_open else "Browser is not open."

    def press(self, key=""):
        return "Pressed." if self.is_open else "Browser is not open."

    def close(self):
        self.is_open = False
        return "Browser is not open."

    def status(self):
        return "page open"


class Session:
    """Only what _execute_tool's browser branch touches."""

    _execute_tool = chat.ChatSession._execute_tool
    _dispatch_tool = chat.ChatSession._dispatch_tool
    _status = chat.ChatSession._status

    def __init__(self, browser, last_url=""):
        self.browser = browser
        self._last_browsed_url = last_url
        # safety off so these tests exercise the browser branch, not the risk
        # gate — which has its own tests and would ask for approval here.
        self.config = {
            "browser": {"enabled": True},
            "agent": {},
            "safety": {"enabled": False},
        }
        self.confirm_fn = None
        self.enabled_groups = None
        self.output_fn = lambda *_a, **_k: None
        self.statuses: list[str] = []


@pytest.fixture(autouse=True)
def no_peek(monkeypatch):
    """_browser_peek reads real page content; irrelevant here."""
    monkeypatch.setattr(chat, "_browser_peek", lambda _b: "")


@pytest.fixture(autouse=True)
def quiet_telemetry(monkeypatch):
    events = []
    monkeypatch.setattr(chat.local_telemetry, "log_event",
                        lambda kind, **f: events.append((kind, f)))
    return events


def click(session):
    return session._execute_tool("browser_click", {"target": "Sign in"})


def test_a_closed_browser_is_reopened_and_the_click_retried():
    b = FakeBrowser(is_open=False)
    out = click(Session(b, last_url="https://example.com"))
    assert b.opened == ["https://example.com"]
    assert b.clicks == 2, "should click, fail, reopen, then click again"
    assert "Clicked Sign in" in out


def test_recovery_is_reported_to_telemetry(quiet_telemetry):
    click(Session(FakeBrowser(is_open=False), last_url="https://example.com"))
    kinds = [k for k, _ in quiet_telemetry]
    assert "browser_recover" in kinds


def test_no_recovery_without_a_url_this_session():
    """Nothing was ever opened, so there is nothing to reopen. The clear
    failure is better than guessing a URL."""
    b = FakeBrowser(is_open=False)
    out = click(Session(b, last_url=""))
    assert b.opened == []
    assert b.clicks == 1
    assert "Browser is not open" in out
    assert "browse>" in out, "should still tell the model how to fix it"


def test_a_working_browser_is_not_reopened():
    b = FakeBrowser(is_open=True)
    out = click(Session(b, last_url="https://example.com"))
    assert b.opened == []
    assert b.clicks == 1
    assert "Clicked Sign in" in out


def test_it_retries_only_once():
    """A retry loop against a page that will not load is worse than a clear
    failure — the reopen 'succeeds' but the page is still unusable."""
    class NeverUsable(FakeBrowser):
        def open(self, url):
            self.opened.append(url)
            return "Opened."          # claims success, stays closed

    b = NeverUsable(is_open=False)
    out = click(Session(b, last_url="https://example.com"))
    assert len(b.opened) == 1
    assert b.clicks == 2
    assert "Browser is not open" in out


def test_a_blocked_reopen_does_not_retry():
    """If the domain gate refuses the reopen, do not click into the void."""
    b = FakeBrowser(is_open=False, reopen_result="Browser open blocked: user denied.")
    out = click(Session(b, last_url="https://example.com"))
    assert b.opened == ["https://example.com"]
    assert b.clicks == 1
    assert "Browser is not open" in out


def test_closing_a_closed_browser_does_not_reopen_it():
    """Reopening a page in order to close it is absurd."""
    b = FakeBrowser(is_open=False)
    session = Session(b, last_url="https://example.com")
    session._execute_tool("browser_close", {})
    assert b.opened == []


@pytest.mark.parametrize("tool,params", [
    ("browser_type", {"text": "hello"}),
    ("browser_scroll", {"direction": "down"}),
    ("browser_press", {"key": "down"}),
])
def test_the_other_page_actions_recover_too(tool, params):
    b = FakeBrowser(is_open=False)
    session = Session(b, last_url="https://example.com")
    out = session._execute_tool(tool, params)
    assert b.opened == ["https://example.com"]
    assert "Browser is not open" not in out


class RaisingBrowser(FakeBrowser):
    """Reports a closed session by raising instead of returning."""

    def click(self, selector="", text=""):
        self.clicks += 1
        if not self.is_open:
            raise RuntimeError("Browser is not open. Call browser_open first.")
        return f"Clicked {text or selector}."


def test_a_raised_not_open_recovers_the_same_way():
    """The browser reports a closed session two ways, and the activity log
    shows both: 314 returned it, 122 raised it and surfaced as "failed
    unexpectedly". Recovering only the returned form would leave more than a
    quarter of the failures untouched."""
    b = RaisingBrowser(is_open=False)
    out = click(Session(b, last_url="https://example.com"))
    assert b.opened == ["https://example.com"]
    assert b.clicks == 2
    assert "Clicked Sign in" in out


def test_an_unrelated_exception_is_not_swallowed():
    """Only 'not open' is translated. Anything else must keep propagating to
    the handler that knows how to report it."""
    class Broken(FakeBrowser):
        def click(self, selector="", text=""):
            raise RuntimeError("the page crashed")

    out = click(Session(Broken(is_open=True), last_url="https://example.com"))
    assert "failed unexpectedly" in out
    assert "the page crashed" in out
