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
