"""Tests for the lightweight RAG retriever."""

import json
import time

import pytest

from rag import Retriever


@pytest.fixture
def base_config(tmp_path, monkeypatch):
    """Isolated RAG paths."""
    from rag import DATA_DIR, NOTES_DIR, PROJECT_DIR, TRAIN_FILE
    monkeypatch.setattr("rag.PROJECT_DIR", tmp_path)
    monkeypatch.setattr("rag.NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr("rag.DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr("rag.TRAIN_FILE", tmp_path / "training_data" / "train.jsonl")
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_data").mkdir(parents=True, exist_ok=True)
    return {
        "rag": {
            "enabled": True,
            "top_k": 5,
            "max_context_tokens": 500,
            "sources": ["notes"],
            "context_cache_ttl_seconds": 0.5,
            "context_cache_max_entries": 4,
        }
    }


def test_build_context_caches_result(base_config, tmp_path):
    note = tmp_path / "notes" / "hobbies.md"
    note.write_text("# Hobbies\nThe user likes hiking and coffee.")

    retriever = Retriever(base_config)
    ctx1 = retriever.build_context("hobbies")
    assert "hiking" in ctx1

    # Delete the note file; a cached result should still return the same text.
    note.unlink()
    ctx2 = retriever.build_context("hobbies")
    assert ctx2 == ctx1


def test_invalidate_cache_clears_context_cache(base_config, tmp_path):
    note = tmp_path / "notes" / "hobbies.md"
    note.write_text("# Hobbies\nThe user likes hiking.")

    retriever = Retriever(base_config)
    ctx1 = retriever.build_context("hobbies")
    retriever.invalidate_cache()
    note.write_text("# Hobbies\nThe user likes swimming.")
    ctx2 = retriever.build_context("hobbies")
    assert ctx1 != ctx2
    assert "swimming" in ctx2


def test_context_cache_expires_after_ttl(base_config, tmp_path):
    note = tmp_path / "notes" / "hobbies.md"
    note.write_text("# Hobbies\nThe user likes hiking.")

    retriever = Retriever(base_config)
    ctx1 = retriever.build_context("hobbies")
    note.write_text("# Hobbies\nThe user likes swimming.")
    ctx2 = retriever.build_context("hobbies")
    # Within TTL, the cached value is returned.
    assert ctx2 == ctx1

    # After TTL expires, a fresh retrieval (with a fresh note cache) sees the
    # updated note. We manually drop the note cache here because in normal
    # operation note writes call invalidate_cache(), which clears both caches.
    time.sleep(0.6)
    retriever._note_cache = None
    ctx3 = retriever.build_context("hobbies")
    assert "swimming" in ctx3


def test_search_training_data_scans_recent_bytes_only(base_config, tmp_path):
    base_config["rag"]["sources"] = ["training_data"]
    train_file = tmp_path / "training_data" / "train.jsonl"

    # Fill with many lines so scanning the whole file would be slow.
    filler = "\n".join(
        json.dumps({"text": f"old sample number {i} with keyword ancient"})
        for i in range(2000)
    )
    train_file.write_text(filler + "\n", encoding="utf-8")

    # Append a recent line containing the query term.
    recent = json.dumps({"text": "recent sample mentions hiking trail"})
    with open(train_file, "a", encoding="utf-8") as f:
        f.write(recent + "\n")

    retriever = Retriever(base_config)
    results = retriever.search_training_data("hiking", top_k=5)
    texts = [r["text"] for r in results]
    assert any("recent sample" in t for t in texts)
    assert not any("ancient" in t for t in texts)


# ---- a wrong note is worse than no note ----
#
# Retrieved notes are pasted into the model's context, and skill notes are
# procedures, so the model performs them. Measured live: a request to save a
# bicycle skill retrieved the Browser Driver note and the agent produced a
# repetition loop of "Clicking the Steps. Scrolling down." and a real
# browser_open on google.com. IDF ranked correctly the whole time — it just
# has no way to say "none of these", so when the topical words matched no note
# the ranking fell to leftovers like "skill", "for" and "how".

@pytest.fixture
def skill_notes(base_config, tmp_path):
    """A corpus shaped like the real one: procedures sharing boilerplate."""
    notes = tmp_path / "notes"
    bodies = {
        "Skill__Browser_Driver.md": "# Skill: Browser Driver\n1. Click the element. 2. Scroll down. 3. Read the page.",
        "Skill__Fix_wifi.md": "# Skill: Fix wifi\n1. Toggle the wifi adapter. 2. Forget the network. 3. Rejoin.",
        "Skill__Coffee_Making.md": "# Skill: Coffee Making\n1. Grind beans. 2. Add filter. 3. Brew.",
        "Skill__Device_Awareness.md": "# Skill: Device Awareness\n1. Save a fact when the user gives one. 2. Do arithmetic.",
        "Skill__Researcher.md": "# Skill: Researcher\n1. Search online. 2. Summarise findings. 3. Cite sources.",
        "Skill__Repotting.md": "# Skill: Repotting\n1. Check roots. 2. Choose a pot. 3. Water it.",
    }
    for name, text in bodies.items():
        (notes / name).write_text(text, encoding="utf-8")
    return Retriever(base_config)


def test_a_query_matching_no_note_returns_nothing(skill_notes):
    """Nothing here is about bicycles, so three confident wrong answers is the
    one result that must not come back."""
    assert skill_notes.search_notes("tuning a bicycle") == []
    # The query that actually derailed the trial.
    assert skill_notes.search_notes(
        "Emit one <skill> tag for bicycle tuning. Write the steps yourself.") == []


def test_a_real_shared_word_is_still_a_match(skill_notes):
    """The floor drops notes with nothing distinguishing in common — not
    notes that genuinely share a content word. "save" appears in the Device
    Awareness note, so matching it is correct behaviour, not a regression."""
    hits = skill_notes.search_notes("save a fact for me")
    assert hits and "Device_Awareness" in hits[0]["title"]


def test_the_word_skill_alone_retrieves_no_skill_note(skill_notes):
    """It appears in every one of them, so it distinguishes nothing."""
    assert skill_notes.search_notes("skill") == []
    assert skill_notes.search_notes("what are the steps") == []


def test_an_on_topic_query_still_finds_its_note(skill_notes):
    hits = skill_notes.search_notes("how do I fix my wifi")
    assert hits, "the floor must not suppress a real match"
    assert "Fix_wifi" in hits[0]["title"]


def test_every_skill_finds_its_own_note(skill_notes):
    for query, expected in [("how do I make coffee", "Coffee_Making"),
                            ("repotting a plant", "Repotting"),
                            ("click the button and scroll", "Browser_Driver")]:
        hits = skill_notes.search_notes(query)
        assert hits and expected in hits[0]["title"], (query, hits)


def test_stopwords_never_qualify_a_note(skill_notes):
    """On a small technical corpus "how" and "do" are genuinely rare — most
    notes are numbered steps and never use them — so a frequency test alone
    called them discriminative and let them admit an unrelated note."""
    terms = skill_notes._discriminative_terms(
        ["how", "do", "the", "steps", "coffee"],
        [["coffee"], ["a"], ["b"], ["c"], ["d"], ["e"]])
    assert terms == {"coffee"}


def test_a_small_corpus_keeps_the_old_behaviour(base_config, tmp_path):
    """With four notes one is 25% and two is 50%, so the share cutoff would be
    deciding on noise."""
    notes = tmp_path / "notes"
    for i in range(3):
        (notes / f"n{i}.md").write_text(f"note {i} about widgets", encoding="utf-8")
    r = Retriever(base_config)
    assert r._discriminative_terms(["widgets"], [["a"], ["b"], ["c"]]) is None
    assert r.search_notes("widgets"), "still finds things in a tiny corpus"
