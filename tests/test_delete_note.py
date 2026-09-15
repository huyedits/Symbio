"""The model-issued note delete (symbio/app/memory.py:delete_note).

write_note only ever creates, so this is the one path by which a tool call
removes a file from notes/ for good — there is no archive step and no undo.
These tests pin the three ways it must refuse: nothing matched, more than one
thing matched, and the note being one the pruner is already forbidden to touch.
"""
import pytest

from symbio import constants
from symbio.app import memory


@pytest.fixture
def notes(tmp_path, monkeypatch):
    d = tmp_path / "notes"
    d.mkdir()
    monkeypatch.setattr(constants, "NOTES_DIR", d)
    return d


def write(notes, name: str, title: str, body: str = "Some body text here."):
    p = notes / name
    p.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return p


def test_a_single_match_is_deleted(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info")

    deleted, message = memory.delete_note("proxy info")

    assert deleted
    assert not p.exists()
    assert "Proxy Info" in message


def test_nothing_matched_deletes_nothing(notes):
    p = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info")

    deleted, message = memory.delete_note("vpn settings")

    assert not deleted
    assert p.exists()
    assert "No note matches" in message


def test_an_ambiguous_query_deletes_nothing_and_names_the_candidates(notes):
    a = write(notes, "20260101_000000_Proxy_Info.md", "Proxy Info")
    b = write(notes, "20260102_000000_Proxy_Setup.md", "Proxy Setup")

    deleted, message = memory.delete_note("proxy")

    assert not deleted
    assert a.exists() and b.exists()
    assert "Proxy Info" in message and "Proxy Setup" in message


@pytest.mark.parametrize("title", [
    "Skill: Scrape A Listing Page",
    "My Identity",
    "User Identity",
])
def test_a_protected_note_survives_a_tool_call(notes, title):
    """A skill note is the readable half of a trained worker adapter, and the
    identity notes are what the assistant knows about itself and its user.
    prune.py refuses to archive either; a delete_note call has no more right,
    and the adapter it would orphan still loads and still routes."""
    p = write(notes, "20260101_000000_note.md", title)

    deleted, message = memory.delete_note("note")

    assert not deleted
    assert p.exists()
    assert "protected" in message.lower()
    assert title in message


def test_the_refusal_is_not_routable_around_by_matching_on_the_body(notes):
    """The guard reads the file that would actually be unlinked, so it holds
    however the note was found — find_notes falls back to a body substring
    when no title matches."""
    p = write(notes, "20260101_000000_skill.md", "Skill: Fix The Wifi",
              "Toggle the adapter, then renew the lease.")

    deleted, _ = memory.delete_note("renew the lease")

    assert not deleted
    assert p.exists()


def test_a_hash_in_the_title_is_not_eaten_by_the_prefix_strip(notes):
    """`lstrip("# ")` strips a run of '#'/' ' characters, not the markdown
    prefix: it turned '# # 1 Proxy' into '1 Proxy'. prune.note_title owns the
    heading shape, so the title reported back is the one on disk."""
    write(notes, "20260101_000000_ranked.md", "# 1 Ranked Proxy")

    deleted, message = memory.delete_note("ranked")

    assert deleted
    assert "# 1 Ranked Proxy" in message
