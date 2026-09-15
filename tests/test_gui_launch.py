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


# ---- the names nobody has invented yet ----
#
# The alias table above is an exact-match list of misspellings, which cannot
# win: once it covered chrome/chromebrowser/chrome-app, the model moved on to
# launch-chrome, chrome-x and start-chrome. Those failed 15 times on 2026-08-24
# alone, in five identical three-step cycles — each returning "Command not
# found: launch-chrome", an error that teaches nothing, so the model tried the
# next variant and then fabricated an answer. Matching the stem covers 539 of
# the 541 historical failures instead of the 485 the table could name.

@darwin_only
@pytest.mark.parametrize("word", [
    "launch-chrome", "chrome-x", "start-chrome", "chrome_launcher",
    "run.chrome", "open-chrome-now", "chromebrowser",
])
def test_an_invented_wrapper_around_a_known_app_still_resolves(word):
    assert _gui_app_for(word, f"Command not found: {word}") == "Google Chrome"


@darwin_only
@pytest.mark.parametrize("word", ["launch-safari", "spotify-app", "vscode-x"])
def test_stem_matching_is_not_chrome_specific(word):
    assert _gui_app_for(word, f"Command not found: {word}") is not None


@darwin_only
@pytest.mark.parametrize("word", ["ls", "git", "node", "python3", "open", "uptime"])
def test_a_real_binary_is_never_captured_by_a_stem(word):
    # Only ever consulted after the shell has said the binary does not exist,
    # but the names must not collide even so.
    assert _gui_app_for(word, f"Command not found: {word}") is None


@darwin_only
def test_a_stem_inside_an_unrelated_word_is_not_a_launch():
    # "code" is deliberately not a stem: decode/encode would swallow it.
    assert _gui_app_for("decode", "Command not found: decode") is None
    assert _gui_app_for("encoder", "Command not found: encoder") is None


@darwin_only
def test_a_command_with_arguments_is_still_not_a_bare_launch():
    # A real invocation that happens to mention an app is not a bare app name.
    assert _gui_app_for("chrome --headless x.com", NOT_FOUND) is None


# ---- /run gets the same help the model gets ----
#
# _cmd_run called sandbox.run_sandboxed directly, so `/run chrome` typed by
# hand died on "command not found" while the identical command emitted by the
# model was quietly corrected. The person driving got less help than the thing
# being driven.

class _RunSession:
    _cmd_run = chat.ChatSession._cmd_run

    def __init__(self):
        self.config = {"agent": {}, "browser": {}, "safety": {"enabled": False}}
        self.confirm_fn = None
        self.tokenizer = None
        self.system_prompt = ""
        self.lines = []
        self.output_fn = lambda t: self.lines.append(t)

    @property
    def text(self):
        return "\n".join(self.lines)


@pytest.fixture
def run_session(monkeypatch):
    session = _RunSession()
    calls = []
    logged = []

    def fake_sandbox(cmd, config, confirm_fn=None):
        calls.append(cmd)
        if cmd.startswith("open -a"):
            return True, ""                      # a GUI launch prints nothing
        if cmd == "uname":
            return True, "Darwin"
        return False, f"Command not found: {cmd}"

    monkeypatch.setattr(chat.sandbox, "run_sandboxed", fake_sandbox)
    monkeypatch.setattr(chat.training, "append_chat_pair",
                        lambda **kw: logged.append(kw))
    return session, calls, logged


@darwin_only
def test_run_retries_an_invented_app_name(run_session):
    session, calls, _ = run_session
    session._cmd_run("chrome-x")
    assert calls == ["chrome-x", "open -a 'Google Chrome'"]
    assert "[ok]" in session.text


@darwin_only
def test_run_does_not_invent_a_retry_for_a_real_failure(run_session):
    session, calls, _ = run_session
    session._cmd_run("frobnicate")
    assert calls == ["frobnicate"]
    assert "[err]" in session.text


@darwin_only
def test_a_silent_command_is_not_written_to_the_training_corpus(run_session):
    # `open -a`, mkdir and touch all print nothing. Logging those pairs teaches
    # the model that the right answer to a command is no answer at all.
    session, _, logged = run_session
    session._cmd_run("chrome-x")
    assert logged == []
    assert "not logged" in session.text


def test_a_command_with_output_is_still_logged(run_session):
    session, _, logged = run_session
    session._cmd_run("uname")
    assert len(logged) == 1
    assert logged[0]["assistant_msg"] == "Darwin"
