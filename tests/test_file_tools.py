"""Tests for read_file/edit_file/write_file chat tools."""

from pathlib import Path
from unittest.mock import patch

import pytest

from symbio import constants
from symbio.app.chat import ChatSession
from symbio.app.config import DEFAULT_CONFIG


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A ChatSession whose project dir is an isolated tmp_path."""
    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    cfg = DEFAULT_CONFIG.copy()
    cfg["agent"] = DEFAULT_CONFIG["agent"].copy()
    cfg["agent"]["backup_before_edit"] = True
    session = ChatSession(config=cfg)
    session._history = []
    return session


def test_read_file_returns_contents(session, tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")
    result = session._handle_file_tool("read_file", {"path": "notes.txt"})
    assert "hello world" in result


def test_read_file_rejects_outside_project(session, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    result = session._handle_file_tool("read_file", {"path": str(outside)})
    assert "outside" in result.lower() or "not allowed" in result.lower()


def test_write_file_creates_file(session, tmp_path):
    result = session._handle_file_tool("write_file", {"path": "new.md", "content": "# New file"})
    assert (tmp_path / "new.md").exists()
    assert "Wrote new.md" in result


def test_write_file_with_backup(session, tmp_path):
    existing = tmp_path / "config.txt"
    existing.write_text("old", encoding="utf-8")
    result = session._handle_file_tool("write_file", {"path": "config.txt", "content": "new"})
    backups = list(tmp_path.glob("config.txt.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old"
    assert existing.read_text(encoding="utf-8") == "new"


def test_write_file_no_backup(session, tmp_path):
    session.config["agent"]["backup_before_edit"] = False
    existing = tmp_path / "config.txt"
    existing.write_text("old", encoding="utf-8")
    result = session._handle_file_tool("write_file", {"path": "config.txt", "content": "new"})
    assert not list(tmp_path.glob("config.txt.*.bak"))
    assert existing.read_text(encoding="utf-8") == "new"


def test_edit_file_replaces_exact_text(session, tmp_path):
    target = tmp_path / "config.json"
    target.write_text('{"temperature": 0.7}', encoding="utf-8")
    result = session._handle_file_tool(
        "edit_file",
        {
            "path": "config.json",
            "old_string": '"temperature": 0.7',
            "new_string": '"temperature": 0.9',
        },
    )
    assert target.read_text(encoding="utf-8") == '{"temperature": 0.9}'
    assert "Edited config.json" in result


def test_edit_file_creates_backup(session, tmp_path):
    target = tmp_path / "config.json"
    target.write_text('{"temperature": 0.7}', encoding="utf-8")
    session._handle_file_tool(
        "edit_file",
        {
            "path": "config.json",
            "old_string": '"temperature": 0.7',
            "new_string": '"temperature": 0.9',
        },
    )
    backups = list(tmp_path.glob("config.json.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == '{"temperature": 0.7}'


def test_edit_file_missing_old_string(session, tmp_path):
    target = tmp_path / "config.json"
    target.write_text('{"temperature": 0.7}', encoding="utf-8")
    result = session._handle_file_tool(
        "edit_file",
        {
            "path": "config.json",
            "old_string": '"temperature": 0.99',
            "new_string": '"temperature": 0.9',
        },
    )
    assert "could not find" in result.lower()
    assert target.read_text(encoding="utf-8") == '{"temperature": 0.7}'


def test_edit_file_backup_disabled_per_call(session, tmp_path):
    target = tmp_path / "config.json"
    target.write_text('{"temperature": 0.7}', encoding="utf-8")
    session._handle_file_tool(
        "edit_file",
        {
            "path": "config.json",
            "old_string": '"temperature": 0.7',
            "new_string": '"temperature": 0.9',
            "backup": False,
        },
    )
    assert not list(tmp_path.glob("config.json.*.bak"))


def test_make_backup_overflow(tmp_path):
    target = tmp_path / "x.txt"
    target.write_text("x", encoding="utf-8")
    for i in range(1, 10000):
        (tmp_path / f"x.txt.{i}.bak").write_text("x", encoding="utf-8")
    session = ChatSession(config=DEFAULT_CONFIG)
    with pytest.raises(RuntimeError, match="free backup slot"):
        session._make_backup(target)


# ---- a project path written with a leading slash ----
#
# Live 2026-08-24: asked for the size of symbio/app/chat.py, the model sent
# "/symbio/app/chat.py". That resolved outside the project, tripped the
# path_escape risk flag, put a HIGH-risk approval prompt in front of the user,
# and then failed anyway with "Must be inside the project directory" — three
# bad outcomes for a file sitting in the project the whole time.

class _Resolver:
    _resolve_project_path = ChatSession._resolve_project_path


def test_a_rooted_project_path_is_relocated_into_the_project():
    from symbio import constants
    target = _Resolver()._resolve_project_path("/symbio/app/chat.py")
    assert target is not None
    assert target.resolve() == (constants.PROJECT_DIR / "symbio/app/chat.py").resolve()


def test_a_relative_path_is_unaffected():
    from symbio import constants
    target = _Resolver()._resolve_project_path("symbio/app/chat.py")
    assert target.resolve() == (constants.PROJECT_DIR / "symbio/app/chat.py").resolve()


@pytest.mark.parametrize("path", [
    "/etc/passwd",              # exists at root: left alone, then refused
    "/nonexistent/nothing.py",  # names nothing anywhere
    "../../../etc/passwd",      # ordinary traversal
    "/../etc/passwd",
])
def test_escapes_are_still_refused(path):
    # The relocation may only ever move a path INTO the project, never out.
    assert _Resolver()._resolve_project_path(path) is None


# ---- a project path the model got only partly right ----
#
# Shell commands run from the sandbox directory while read_file resolves
# against the project root, so a project-relative path fails with a bare "No
# such file or directory" and the model concludes the file is missing. Live
# 2026-08-24/25 it produced three variants of the same mistake — "./symbio/app/
# chat.py", "/symbio/app/chat.py", and "/usr/local/home/user/symbio/app/web.py"
# (the real tree is under .../Downloads/agi) — and each time told the user
# their file did not exist.

def test_a_relative_path_is_recognised():
    from symbio.app.chat import _project_paths_in
    assert _project_paths_in("ls -l symbio/app/chat.py") == ["symbio/app/chat.py"]
    assert _project_paths_in("ls -l ./symbio/app/chat.py") == ["symbio/app/chat.py"]


def test_an_invented_absolute_prefix_is_seen_through():
    from symbio.app.chat import _project_paths_in
    # Wrong prefix, right tail — the tail names the file exactly.
    assert _project_paths_in("wc -c /usr/local/home/user/symbio/app/web.py") == \
        ["symbio/app/web.py"]


@pytest.mark.parametrize("cmd", [
    "cat /etc/passwd",
    "cat /var/log/system.log",
    "ls nope/nothing.py",
    "ls -la",
    "echo hello",
])
def test_paths_outside_the_project_are_not_claimed(cmd):
    from symbio.app.chat import _project_paths_in
    assert _project_paths_in(cmd) == []


def test_a_bare_filename_is_never_resolved():
    from symbio.app.chat import _project_paths_in
    # At least two components must survive, so a lone "chat.py" — which several
    # directories could satisfy — is not guessed at.
    assert _project_paths_in("cat chat.py") == []


def test_the_annotation_tells_the_model_not_to_deny_the_file():
    from symbio.app.chat import _annotate_sandbox_cwd
    out = _annotate_sandbox_cwd(
        "wc -c symbio/app/web.py", "wc: symbio/app/web.py: No such file or directory")
    assert "symbio/app/web.py" in out
    assert "sandbox" in out
    assert "does not exist" in out          # the "Do NOT tell the user" clause


def test_an_unrelated_failure_is_left_alone():
    from symbio.app.chat import _annotate_sandbox_cwd
    msg = "cat: nope.txt: No such file or directory"
    assert _annotate_sandbox_cwd("cat nope.txt", msg) == msg


# ---- the sandboxed-script helper API ----
#
# execute_code offered nothing but "pure computation", so a script could not do
# real work — while the bare open() builtin was unrestricted the whole time and
# could write anywhere on disk. The stub inverts that: the safe, contained path
# is now also the easy one.

def _run(code):
    import json
    from symbio.app import sandbox
    return sandbox.run_python_code(code, json.loads(
        (constants.PROJECT_DIR / "config.json").read_text(encoding="utf-8")))


def test_pure_computation_still_needs_no_import():
    ok, out = _run("print(2 + 2)")
    assert ok and out.strip() == "4"


def test_a_script_can_write_and_read_a_project_file(tmp_path):
    ok, out = _run(
        "from symbio_tools import write_file, read_file\n"
        "write_file('sandbox/_t_probe.txt', 'abc')\n"
        "print(read_file('sandbox/_t_probe.txt'))")
    assert ok, out
    assert "abc" in out
    (constants.PROJECT_DIR / "sandbox" / "_t_probe.txt").unlink(missing_ok=True)


def test_the_helper_cannot_leave_the_project():
    ok, out = _run(
        "from symbio_tools import write_file\n"
        "try:\n"
        "    write_file('../../../../tmp/escaped.txt', 'nope')\n"
        "except ValueError as e:\n"
        "    print('refused:', e)")
    assert ok and "refused" in out
    assert not Path("/tmp/escaped.txt").exists()


def test_fetch_rejects_non_http_schemes():
    ok, out = _run(
        "from symbio_tools import fetch\n"
        "try:\n"
        "    fetch('file:///etc/passwd')\n"
        "except ValueError as e:\n"
        "    print('refused:', e)")
    assert ok and "refused" in out


@pytest.mark.parametrize("mod", ["subprocess", "os", "ctypes", "urllib.request", "pathlib"])
def test_the_escape_modules_are_still_refused(mod):
    # The stub adds capability; it must not add a way around the blocklist.
    ok, out = _run(f"import {mod}\nprint(1)")
    assert not ok
    assert "not allowed" in out


def test_the_stub_offers_no_shell():
    # The Hermes stub's terminal() would bypass every entry in
    # sandbox.blocked_commands; it is deliberately not ported.
    from symbio.app import sandbox
    src = sandbox._tools_stub_source()
    assert "subprocess" not in src
    assert "def terminal" not in src


# ---- browser_get_text ----
#
# chat.py's own browser-retry nudge has been telling the model to "use
# browser_get_text if needed" since it was written, and no such tool existed in
# this runtime. The model reached for it under the name `browser_read` on
# 2026-08-26 and the call was dropped by the group filter, leaving an invented
# page summary as the reply. The tool exists now; this pins its behaviour.

class _FakeBrowser:
    """Mirrors BrowserSession's real surface.

    `is_open` is a @property on the real class. This fake first declared it as
    a method, so `self.browser.is_open()` passed here and raised
    "'bool' object is not callable" against the live browser — the tool was
    broken in every real session while its tests were green. A fake that does
    not match the shape of what it stands in for tests nothing.
    """

    def __init__(self, open_=True, text="Star 28\nFork 2\n201 Commits"):
        self._open, self._text = open_, text

    @property
    def is_open(self):
        return self._open

    def get_text(self):
        return self._text


def _session_with_browser(browser):
    from symbio.app import chat as chat_mod
    session = chat_mod.ChatSession.__new__(chat_mod.ChatSession)
    session.browser = browser
    session.config = {"browser": {"enabled": True}, "safety": {"enabled": False},
                      "agent": {"max_page_chars": 12000}}
    session._untrusted_this_turn = False
    return session


def test_browser_get_text_returns_the_page_wrapped_as_untrusted():
    session = _session_with_browser(_FakeBrowser())
    out = session._dispatch_tool("browser_get_text", {})
    assert "201 Commits" in out
    # Page text is attacker-controllable; it must arrive marked as data.
    assert "untrusted" in out.lower()
    assert session._untrusted_this_turn is True


def test_browser_get_text_says_so_when_nothing_is_open():
    session = _session_with_browser(_FakeBrowser(open_=False))
    out = session._dispatch_tool("browser_get_text", {})
    assert "not open" in out.lower()
    assert "browser_open" in out


def test_browser_get_text_respects_the_browser_switch():
    session = _session_with_browser(_FakeBrowser())
    session.config["browser"]["enabled"] = False
    assert "disabled" in session._dispatch_tool("browser_get_text", {}).lower()


def test_the_fake_browser_matches_the_real_one():
    """Pin the fake to BrowserSession's actual surface.

    `is_open` being a property rather than a method is exactly the difference
    that let browser_get_text ship broken with passing tests. Checking it here
    means the next drift fails in CI instead of in a live session.
    """
    from symbio.computer import BrowserSession

    real_is_open = type(BrowserSession).__mro__ and getattr(
        BrowserSession, "is_open", None)
    assert isinstance(real_is_open, property), (
        "BrowserSession.is_open stopped being a property — update _FakeBrowser "
        "and every call site that reads it as an attribute")
    assert isinstance(getattr(_FakeBrowser, "is_open", None), property), (
        "the fake declares is_open as a method; the real class uses a property")

    for name in ("get_text",):
        assert callable(getattr(BrowserSession, name, None)), name
        assert callable(getattr(_FakeBrowser, name, None)), name


# ---- page text limits are configurable, not magic numbers ----

def _wrapped_payload(out):
    """The text between the untrusted header and footer.

    Parsed structurally, not by matching header wording — the header text
    changed on 2026-08-27 to close a permission loophole and took two tests
    with it. The wrapper's SHAPE (header line, payload, footer line) is the
    contract; its prose is not.
    """
    body = out.split("[End untrusted")[0]
    return body.split("]\n", 1)[1] if "]\n" in body else body

def test_browser_get_text_honours_max_page_chars():
    long_page = "x" * 20000
    session = _session_with_browser(_FakeBrowser(text=long_page))
    session.config["agent"] = {"max_page_chars": 500}
    out = session._dispatch_tool("browser_get_text", {})
    payload = _wrapped_payload(out).split("\n... (truncated")[0]
    assert len(payload) == 500
    # A silent cut teaches the model the page ended there.
    assert "truncated at 500" in out
    assert "max_page_chars" in out


def test_browser_get_text_falls_back_to_max_output_len():
    session = _session_with_browser(_FakeBrowser(text="y" * 9000))
    session.config["agent"] = {"max_output_len": 4000}
    out = session._dispatch_tool("browser_get_text", {})
    payload = _wrapped_payload(out).split("\n... (truncated")[0]
    assert len(payload) == 4000


def test_a_short_page_is_not_marked_truncated():
    session = _session_with_browser(_FakeBrowser(text="Example Domain"))
    session.config["agent"] = {"max_page_chars": 12000}
    out = session._dispatch_tool("browser_get_text", {})
    assert "Example Domain" in out and "truncated" not in out


def test_the_post_action_peek_has_its_own_smaller_budget():
    """It lands after EVERY browser action, so it must not inherit the big one."""
    from symbio.app import chat as chat_mod
    peek = chat_mod._browser_peek(_FakeBrowser(text="z" * 5000),
                                  {"agent": {"browser_peek_chars": 200,
                                             "max_page_chars": 12000}})
    assert peek.count("z") == 200
