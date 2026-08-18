"""Launching a GUI app that the model tried to start as a CLI command.

481 of the 542 run_command failures in the activity log are this one mistake:

    241  chrome        120  chromebrowser        120  chrome-app

env_note() tells the model on every turn that "GUI apps have no CLI names like
'chrome'" and to use `open -a 'Google Chrome'`. It did it anyway, 481 times,
so the correction belongs in code rather than in another sentence of prompt.
"""
import sys

import pytest

from symbio.app import chat
from symbio.app.chat import _gui_app_for

NOT_FOUND = "Command not found: chrome"

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


# ---- which commands are recognised ----

@darwin_only
@pytest.mark.parametrize("word,app", [
    ("chrome", "Google Chrome"),
    ("chromebrowser", "Google Chrome"),
    ("chrome-app", "Google Chrome"),
    ("Chrome", "Google Chrome"),
    ("safari", "Safari"),
    ("spotify", "Spotify"),
])
def test_a_bare_gui_app_name_is_recognised(word, app):
    assert _gui_app_for(word, f"Command not found: {word}") == app


@darwin_only
def test_surrounding_whitespace_and_quotes_are_tolerated():
    assert _gui_app_for("  'chrome' ", NOT_FOUND) == "Google Chrome"


# ---- and which are deliberately not ----

@darwin_only
def test_an_unknown_command_is_left_alone():
    """A wide net would turn a genuine typo into a surprise app launch."""
    assert _gui_app_for("chrom", "Command not found: chrom") is None
    assert _gui_app_for("ls", "Command not found: ls") is None


@darwin_only
def test_a_command_with_arguments_is_left_alone():
    """`chrome --headless x.html` is not "please open Chrome"; rewriting it to
    `open -a` would silently drop what was asked for."""
    assert _gui_app_for("chrome --headless page.html", NOT_FOUND) is None


@darwin_only
def test_only_a_not_found_failure_recovers():
    """A command that failed for any other reason is a real failure."""
    assert _gui_app_for("chrome", "Permission denied") is None
    assert _gui_app_for("chrome", "exited error. Output: crashed") is None


def test_nothing_happens_off_macos(monkeypatch):
    monkeypatch.setattr(chat.sys, "platform", "linux")
    assert _gui_app_for("chrome", NOT_FOUND) is None


# ---- the dispatch path ----

class Session:
    _dispatch_tool = chat.ChatSession._dispatch_tool
    _status = chat.ChatSession._status

    def __init__(self):
        self.config = {"agent": {}, "browser": {}, "safety": {"enabled": False}}
        self.confirm_fn = None
        self.enabled_groups = None
        self.output_fn = lambda *_a, **_k: None


@pytest.fixture
def sandbox_calls(monkeypatch):
    calls = []

    def fake_run(cmd, config, confirm_fn=None, **kw):
        calls.append(cmd)
        if cmd.strip() == "chrome":
            return False, "Command not found: chrome"
        return True, "launched"

    monkeypatch.setattr(chat.sandbox, "run_sandboxed", fake_run)
    monkeypatch.setattr(chat.local_telemetry, "log_event", lambda *a, **k: None)
    return calls


@darwin_only
def test_a_failed_chrome_is_retried_as_open_a(sandbox_calls):
    out = Session()._dispatch_tool("run_command", {"cmd": "chrome"})
    assert sandbox_calls == ["chrome", "open -a 'Google Chrome'"]
    assert "exited ok" in out
    assert "open -a 'Google Chrome'" in out


@darwin_only
def test_the_app_name_is_quoted_so_the_space_survives(sandbox_calls):
    Session()._dispatch_tool("run_command", {"cmd": "chrome"})
    assert sandbox_calls[1] == "open -a 'Google Chrome'"


@darwin_only
def test_a_command_that_works_is_not_touched(sandbox_calls):
    out = Session()._dispatch_tool("run_command", {"cmd": "uname"})
    assert sandbox_calls == ["uname"]
    assert "exited ok" in out


@darwin_only
def test_an_unrelated_failure_is_reported_not_retried(monkeypatch):
    calls = []

    def fake_run(cmd, config, confirm_fn=None, **kw):
        calls.append(cmd)
        return False, "Permission denied"

    monkeypatch.setattr(chat.sandbox, "run_sandboxed", fake_run)
    out = Session()._dispatch_tool("run_command", {"cmd": "chrome"})
    assert calls == ["chrome"], "must not retry a non-'not found' failure"
    assert "exited error" in out
