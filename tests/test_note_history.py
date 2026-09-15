"""Tests for note version history (symbio/app/note_history.py).

Notes are created and deleted; before a delete (or a same-second overwrite) the
current content is snapshotted into notes/.history/<stem>/ so it can be viewed
and restored. These tests pin the snapshot/versions/restore cycle and the
name-matching the /note-history command relies on.
"""

import pytest

from symbio import constants
from symbio.app import memory, note_history


@pytest.fixture
def notes(tmp_path, monkeypatch):
    d = tmp_path / "notes"
    d.mkdir()
    monkeypatch.setattr(constants, "NOTES_DIR", d)
    return d


def write(notes, name, title, body="Some body text here."):
    p = notes / name
    p.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return p


def test_delete_snapshots_the_note(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info")

    deleted, _ = memory.delete_note("proxy info")

    assert deleted
    assert not p.exists()
    vers = note_history.versions(p)
    assert len(vers) == 1
    assert "Proxy Info" in vers[0].read_text(encoding="utf-8")


def test_restore_brings_a_deleted_note_back(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info", "old body")

    memory.delete_note("proxy info")
    vers = note_history.versions(p)

    note_history.restore(p, vers[0])

    assert p.exists()
    assert "old body" in p.read_text(encoding="utf-8")


def test_restore_snapshots_the_current_content_first(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info", "v1")
    note_history.snapshot(p)
    p.write_text("# Proxy Info\n\nv2\n", encoding="utf-8")

    assert len(note_history.versions(p)) == 1  # only v1 was snapshotted

    note_history.restore(p, note_history.versions(p)[0])

    assert "v1" in p.read_text(encoding="utf-8")
    # The v2 that was on disk before the restore is itself preserved.
    assert len(note_history.versions(p)) == 2


def test_matching_history_dirs_finds_by_title_fragment(notes):
    p = write(notes, "20260101_000000_Learned__who_won_the_world_cup.md",
              "Learned: who won the world cup")
    note_history.snapshot(p)

    dirs = note_history.matching_history_dirs("who won the world cup")

    assert len(dirs) == 1
    assert dirs[0].name == p.stem


def test_matching_history_dirs_returns_empty_for_no_match(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info")
    note_history.snapshot(p)

    assert note_history.matching_history_dirs("vpn settings") == []


def test_snapshot_missing_note_returns_none(notes):
    p = notes / "20260101_000000_ghost.md"
    assert note_history.snapshot(p) is None


def test_all_history_dirs_lists_only_snapshotted_notes(notes):
    a = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info")
    write(notes, "20260102_000000_Other.md", "Other")  # never snapshotted
    note_history.snapshot(a)

    dirs = note_history.all_history_dirs()

    assert [d.name for d in dirs] == [a.stem]


def _fake_commands():
    """A minimal CommandsMixin shell: the command handler only needs output_fn
    and a retriever whose cache it can invalidate."""
    from types import SimpleNamespace

    from symbio.app.chat_commands import CommandsMixin

    out: list[str] = []

    class Fake(CommandsMixin):
        def __init__(self):
            self.output_fn = out.append
            self.retriever = SimpleNamespace(invalidate_cache=lambda: None)

    return Fake(), out


def test_note_history_command_accepts_a_multi_word_name(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info", "v1")
    note_history.snapshot(p)
    p.write_text("# Proxy Info\n\nv2\n", encoding="utf-8")

    fake, out = _fake_commands()
    fake._cmd_note_history("proxy info")

    assert any("version(s)" in line for line in out)
    assert not any("Invalid version index" in line for line in out)


def test_note_history_command_restores_by_trailing_index(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info", "v1")
    note_history.snapshot(p)
    p.write_text("# Proxy Info\n\nv2\n", encoding="utf-8")

    fake, out = _fake_commands()
    fake._cmd_note_history("proxy info 0")

    assert "v1" in p.read_text(encoding="utf-8")
    assert any("Restored" in line for line in out)
