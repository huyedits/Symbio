"""Self-pruning of the stores RAG reads back (symbio/app/prune.py).

The pruner deletes nothing outright — notes move to notes/archive/ and
session logs are rewritten without their junk entries — but it still edits
the user's real data, so these tests pin both halves of the contract: the
junk shapes it must catch, and the load-bearing content it must never touch.
"""
import json

import pytest

from symbio import constants
from symbio.app import prune, tooling


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect notes/ and sessions/ at a scratch tree."""
    notes = tmp_path / "notes"
    archive = notes / "archive"
    sessions = tmp_path / "sessions"
    for d in (notes, archive, sessions):
        d.mkdir(parents=True)
    monkeypatch.setattr(constants, "NOTES_DIR", notes)
    monkeypatch.setattr(constants, "NOTES_ARCHIVE_DIR", archive)
    monkeypatch.setattr(constants, "SESSIONS_DIR", sessions)
    return {"notes": notes, "archive": archive, "sessions": sessions}


def write_note(store, name: str, title: str, body: str):
    p = store["notes"] / name
    p.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return p


def write_session(store, name: str, entries: list[dict]):
    p = store["sessions"] / name
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return p


def read_session(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---- note classification ----

def test_degenerate_note_is_junk():
    # What a misparsed <note> tag leaves behind: one-letter title, fragment.
    assert prune.classify_note("# T\n\nwell note that\n") == "degenerate title"


def test_empty_body_note_is_junk():
    assert prune.classify_note("# Some Real Title\n\n\n") == "empty body"


def test_subjectless_learned_note_is_junk():
    text = (
        "# Learned: CHECK ONLINE\n\n"
        "**Question:** CHECK ONLINE\n\n"
        "**Answer (from web research):** The 2026 World Cup was won by Spain.\n"
    )
    assert prune.classify_note(text) == "subjectless question"


def test_real_research_note_is_kept():
    text = (
        "# Learned: who won the 2026 world cup\n\n"
        "**Question:** who won the 2026 world cup\n\n"
        "**Answer (from web research):** Spain.\n"
    )
    assert prune.classify_note(text) is None


def test_skill_notes_are_never_junk():
    # A skill note backs a worker LoRA adapter; pruning it would orphan one.
    assert prune.classify_note("# Skill: Fix wifi\n\n1. Toggle off. 2. On.\n") is None
    # Even a skill with a near-empty body survives.
    assert prune.classify_note("# Skill: X\n\n-\n") is None


def test_identity_notes_are_never_junk():
    for title in ("My Identity", "User Identity"):
        assert prune.classify_note(f"# {title}\n\nI am Caine.\n") is None


def test_short_but_real_note_is_kept():
    assert prune.classify_note("# Coffee\n\nHuy takes it black.\n") is None


def test_capability_notes_are_kept():
    """The seeded capability notes are re-created every boot; if the pruner
    archived them each run would churn them back and forth."""
    from symbio.app import memory

    for title, body in memory._CAPABILITY_NOTES.items():
        assert prune.classify_note(f"# {title}\n\n{body}\n") is None, title


# ---- note pruning on disk ----

def test_prune_notes_archives_junk_and_keeps_the_rest(store):
    write_note(store, "20260730_1_T.md", "T", "well note that")
    write_note(store, "20260801_2_Real.md", "Coffee", "Huy takes it black.")
    write_note(store, "20260802_3_Skill.md", "Skill: Fix wifi", "1. Toggle.")

    pruned = prune.prune_notes()

    assert [p["name"] for p in pruned] == ["20260730_1_T.md"]
    remaining = sorted(p.name for p in store["notes"].glob("*.md"))
    assert remaining == ["20260801_2_Real.md", "20260802_3_Skill.md"]
    # Archived, not destroyed — and into a subdirectory the RAG glob skips.
    assert (store["archive"] / "20260730_1_T.md").exists()


def test_prune_notes_keeps_the_oldest_of_a_duplicate_pair(store):
    write_note(store, "20260730_a_Fact.md", "Fact", "The sky is blue.")
    write_note(store, "20260802_b_Fact.md", "Fact", "The sky is blue.")

    pruned = prune.prune_notes()

    assert len(pruned) == 1
    assert pruned[0]["name"] == "20260802_b_Fact.md"
    assert "duplicate" in pruned[0]["reason"]
    assert (store["notes"] / "20260730_a_Fact.md").exists()


def test_prune_notes_dry_run_changes_nothing(store):
    write_note(store, "20260730_1_T.md", "T", "well note that")

    pruned = prune.prune_notes(dry_run=True)

    assert len(pruned) == 1
    assert (store["notes"] / "20260730_1_T.md").exists()
    assert not any(store["archive"].glob("*.md"))


def test_prune_notes_is_idempotent(store):
    write_note(store, "20260730_1_T.md", "T", "well note that")
    write_note(store, "20260801_2_Real.md", "Coffee", "Huy takes it black.")

    assert len(prune.prune_notes()) == 1
    assert prune.prune_notes() == []


def test_archiving_twice_does_not_clobber(store):
    # Same filename pruned in two different runs must not overwrite the first.
    write_note(store, "dupe.md", "T", "first junk")
    prune.prune_notes()
    write_note(store, "dupe.md", "T", "second junk")
    prune.prune_notes()

    archived = sorted(p.name for p in store["archive"].glob("*.md"))
    assert len(archived) == 2, archived
    bodies = {(store["archive"] / n).read_text() for n in archived}
    assert any("first junk" in b for b in bodies)
    assert any("second junk" in b for b in bodies)


# ---- session entry pruning ----

def test_tagged_twin_is_dropped_and_visible_text_kept():
    # The duplicate-logging bug wrote each tool reply twice in the same second.
    entries = [
        {"role": "assistant", "content": "Clicking the button.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Clicking the button.", "timestamp": "T1"},
    ]
    kept, dropped = prune.prune_entries(entries)

    assert dropped == ["tagged twin"]
    assert [e["content"] for e in kept] == ["Clicking the button."]


def test_tagged_reply_without_a_twin_is_kept():
    # A tool call that was never double-logged is real history, not an artifact.
    entries = [
        {"role": "assistant", "content": "<click>Go</click> Clicking.", "timestamp": "T1"},
    ]
    kept, dropped = prune.prune_entries(entries)

    assert dropped == []
    assert len(kept) == 1


def test_tagged_twin_at_a_different_timestamp_is_kept():
    # Same words a minute later is the model repeating itself, not one reply
    # logged twice — dropping it would rewrite history rather than de-dupe it.
    entries = [
        {"role": "assistant", "content": "Clicking the button.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Clicking the button.", "timestamp": "T2"},
    ]
    kept, dropped = prune.prune_entries(entries)

    assert dropped == []
    assert len(kept) == 2


def test_runaway_repeats_are_capped():
    entries = [{"role": "assistant", "content": "Nothing more to say.", "timestamp": f"T{i}"}
               for i in range(93)]
    kept, dropped = prune.prune_entries(entries, max_copies=2)

    assert len(kept) == 2
    assert len(dropped) == 91


def test_distinct_entries_are_never_dropped():
    entries = [
        {"role": "user", "content": "what is 2+2", "timestamp": "T1"},
        {"role": "assistant", "content": "4", "timestamp": "T2"},
        {"role": "user", "content": "and 3+3", "timestamp": "T3"},
        {"role": "assistant", "content": "6", "timestamp": "T4"},
    ]
    kept, dropped = prune.prune_entries(entries)

    assert dropped == []
    assert len(kept) == 4


def test_a_user_repeating_themselves_twice_is_kept():
    # People do say "click continue" twice; only a stuck loop gets trimmed.
    entries = [{"role": "user", "content": "click continue", "timestamp": f"T{i}"}
               for i in range(2)]
    kept, dropped = prune.prune_entries(entries, max_copies=2)

    assert dropped == []
    assert len(kept) == 2


# ---- session pruning on disk ----

def test_prune_sessions_rewrites_only_dirty_files(store):
    dirty = write_session(store, "2026-01-01_a.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])
    clean = write_session(store, "2026-01-02_b.jsonl", [
        {"role": "user", "content": "hello", "timestamp": "T1"},
    ])
    clean_before = clean.read_text()

    report = prune.prune_sessions()

    assert [r["name"] for r in report] == ["2026-01-01_a.jsonl"]
    assert len(read_session(dirty)) == 1
    assert clean.read_text() == clean_before


def test_prune_sessions_never_touches_the_live_session(store):
    live = write_session(store, "live.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])
    before = live.read_text()

    report = prune.prune_sessions(exclude="live")

    assert report == []
    assert live.read_text() == before


def test_prune_sessions_dry_run_changes_nothing(store):
    path = write_session(store, "a.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])
    before = path.read_text()

    report = prune.prune_sessions(dry_run=True)

    assert report and report[0]["dropped"] == 1
    assert path.read_text() == before


def test_prune_sessions_is_idempotent(store):
    write_session(store, "a.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])
    assert prune.prune_sessions()
    assert prune.prune_sessions() == []


def test_prune_sessions_drops_malformed_lines(store):
    path = store["sessions"] / "a.jsonl"
    path.write_text(
        json.dumps({"role": "user", "content": "hi", "timestamp": "T1"}) + "\n"
        + "{not json at all\n",
        encoding="utf-8",
    )

    report = prune.prune_sessions()

    assert report[0]["dropped"] == 1
    assert len(read_session(path)) == 1


def test_prune_sessions_leaves_no_temp_files(store):
    write_session(store, "a.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])
    prune.prune_sessions()

    assert not list(store["sessions"].glob("*.tmp"))


# ---- combined entry point ----

def test_prune_all_reports_both_stores(store):
    write_note(store, "20260730_1_T.md", "T", "well note that")
    write_session(store, "a.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])

    report = prune.prune_all({"prune": {}})

    assert len(report["notes"]) == 1
    assert report["total"] == 2


def test_prune_all_respects_disabled_sources(store):
    write_note(store, "20260730_1_T.md", "T", "well note that")
    write_session(store, "a.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])

    report = prune.prune_all({"prune": {"notes": False, "sessions": False}})

    assert report["total"] == 0
    assert (store["notes"] / "20260730_1_T.md").exists()


# ---- retrieval-side hardening ----

def test_contains_tool_tag_catches_legacy_short_tags():
    assert tooling.contains_tool_tag("<click>Go</click> Clicking.")
    assert tooling.contains_tool_tag("<browse>https://x.com</browse>")
    assert tooling.contains_tool_tag('<tool_call>{"name": "x"}</tool_call>')
    assert tooling.contains_tool_tag('{"name": "terminal", "arguments": {"cmd": "ls"}}')


def test_contains_tool_tag_does_not_flag_prose():
    # Prose that merely mentions tool words, or uses a bare '<', is not a tag.
    assert not tooling.contains_tool_tag("I use a browser daily and read books.")
    assert not tooling.contains_tool_tag("if x < 5 then y")
    assert not tooling.contains_tool_tag("Clicking the button.")
    assert not tooling.contains_tool_tag('the record {"name": "Alice", "age": 30} in db')


def test_tagged_twin_is_not_retrieved_as_context(store, monkeypatch):
    """The junk already on disk must stop reaching the model even before a
    prune runs — retrieval filters it, so old logs can't teach tag syntax."""
    import rag

    class FakeStore:
        def search(self, query, limit=5, exclude_session=None):
            return [
                {"role": "assistant", "timestamp": "T1",
                 "content": "<click>Go</click> Clicking the button."},
                {"role": "assistant", "timestamp": "T1",
                 "content": "Clicking the button."},
            ]

    r = rag.Retriever({"rag": {"enabled": True, "sources": ["sessions"], "top_k": 5}},
                      session_store=FakeStore())
    texts = [h["text"] for h in r.search_sessions("clicking")]

    assert texts == ["Clicking the button."], texts


# ---- boot integration ----

def test_self_prune_runs_at_boot_and_reports(store, monkeypatch):
    """The pruner is wired into ChatSession's boot maintenance, not just
    available as a command."""
    from symbio.app import chat

    write_note(store, "20260730_1_T.md", "T", "well note that")
    write_session(store, "old.jsonl", [
        {"role": "assistant", "content": "Hi.", "timestamp": "T1"},
        {"role": "assistant", "content": "<click>Go</click> Hi.", "timestamp": "T1"},
    ])

    seen: list[str] = []

    class FakeSession:
        config = {"prune": {"enabled": True, "session_max_copies": 2}}
        session_id = "live"
        output_fn = staticmethod(lambda t: seen.append(t))
        retriever = type("R", (), {"invalidate_cache": lambda self: None})()
        logger = type("L", (), {"warning": lambda self, m: None})()

    report = chat.ChatSession._self_prune(FakeSession())

    assert len(report["notes"]) == 1
    assert report["total"] == 2
    assert any("Tidy" in line for line in seen), seen
    # Applied for real, not just reported.
    assert (store["archive"] / "20260730_1_T.md").exists()


def test_self_prune_never_raises_into_boot(store, monkeypatch):
    """A broken prune must not stop the session from coming up."""
    from symbio.app import chat

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(prune, "prune_all", boom)

    warned: list[str] = []

    class FakeSession:
        config = {"prune": {"enabled": True}}
        session_id = "live"
        output_fn = staticmethod(lambda t: None)
        retriever = type("R", (), {"invalidate_cache": lambda self: None})()
        logger = type("L", (), {"warning": lambda self, m: warned.append(m)})()

    report = chat.ChatSession._self_prune(FakeSession())

    assert report["total"] == 0
    assert warned and "disk gone" in warned[0]


def test_self_prune_honours_the_disabled_flag(store):
    from symbio.app import chat

    write_note(store, "20260730_1_T.md", "T", "well note that")

    class FakeSession:
        config = {"prune": {"enabled": False}}
        session_id = "live"
        output_fn = staticmethod(lambda t: None)
        retriever = type("R", (), {"invalidate_cache": lambda self: None})()
        logger = type("L", (), {"warning": lambda self, m: None})()

    report = chat.ChatSession._self_prune(FakeSession())

    assert report["total"] == 0
    assert (store["notes"] / "20260730_1_T.md").exists()
