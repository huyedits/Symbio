"""Tests for the /history command's session listing and reading.

The command itself is a thin shell over two SessionStore helpers; these pin the
helpers so the command's three branches (list, view-by-id, search) have a
correct foundation.
"""

import json

import pytest

from symbio import constants
from symbio.app import sessions


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(constants, "SESSIONS_DIR", d)
    return d


def write_session(d, session_id, turns):
    path = d / f"{session_id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for role, content, ts in turns:
            f.write(json.dumps(
                {"role": role, "content": content, "timestamp": ts}) + "\n")
    return path


def test_list_sessions_returns_metadata_newest_first(sessions_dir):
    write_session(sessions_dir, "2026-01-01_00-00-00-000001", [
        ("user", "hello there", "2026-01-01T00:00:01"),
        ("assistant", "hi", "2026-01-01T00:00:02"),
    ])
    write_session(sessions_dir, "2026-01-02_00-00-00-000002", [
        ("user", "what is the weather", "2026-01-02T00:00:01"),
    ])

    listed = sessions.SessionStore.list_sessions()

    assert [s["session_id"] for s in listed] == [
        "2026-01-02_00-00-00-000002",
        "2026-01-01_00-00-00-000001",
    ]
    assert listed[0]["turns"] == 1
    assert listed[0]["first_user"] == "what is the weather"
    assert listed[1]["turns"] == 2
    assert listed[1]["first_user"] == "hello there"


def test_list_sessions_ignores_non_jsonl_files(sessions_dir):
    write_session(sessions_dir, "2026-01-01_00-00-00-000001", [
        ("user", "hello", "2026-01-01T00:00:01"),
    ])
    (sessions_dir / "2026-01-01_00-00-00-000001_health.json").write_text("{}")

    listed = sessions.SessionStore.list_sessions()

    assert [s["session_id"] for s in listed] == ["2026-01-01_00-00-00-000001"]


def test_read_session_returns_turns_in_order(sessions_dir):
    write_session(sessions_dir, "2026-01-01_00-00-00-000001", [
        ("user", "first", "2026-01-01T00:00:01"),
        ("assistant", "second", "2026-01-01T00:00:02"),
    ])

    rows = sessions.SessionStore.read_session("2026-01-01_00-00-00-000001")

    assert [r["content"] for r in rows] == ["first", "second"]


def test_read_session_missing_returns_empty(sessions_dir):
    assert sessions.SessionStore.read_session("nope") == []


def _fake_commands(session_id="current"):
    """A minimal CommandsMixin shell: the command handler only needs output_fn
    and a session_id to exclude from search."""
    from types import SimpleNamespace

    from symbio.app.chat_commands import CommandsMixin

    out: list[str] = []

    class Fake(CommandsMixin):
        def __init__(self):
            self.output_fn = out.append
            self.session_id = session_id
            self.retriever = SimpleNamespace(invalidate_cache=lambda: None)

    return Fake(), out


def test_history_command_lists_and_views(sessions_dir):
    write_session(sessions_dir, "2026-01-01_00-00-00-000001", [
        ("user", "hello there", "2026-01-01T00:00:01"),
        ("assistant", "hi", "2026-01-01T00:00:02"),
    ])

    fake, out = _fake_commands()
    fake._cmd_history("")
    fake._cmd_history("2026-01-01_00-00-00-000001")

    assert any("hello there" in line for line in out)
    assert any("[user] hello there" in line for line in out)


def test_history_command_search_excludes_the_current_session(sessions_dir):
    write_session(sessions_dir, "2026-01-01_00-00-00-000001", [
        ("user", "what is the weather", "2026-01-01T00:00:01"),
    ])

    fake, out = _fake_commands(session_id="2026-01-01_00-00-00-000001")
    fake._cmd_history("weather")

    assert any("No past sessions match" in line for line in out)
