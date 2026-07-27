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
