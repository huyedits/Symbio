"""Tests for the hierarchical tag-based RAG index."""

import json
from pathlib import Path

import pytest

from tag_rag import TagIndex


@pytest.fixture
def base_tag_config(tmp_path, monkeypatch):
    """Isolated tag RAG paths."""
    from tag_rag import LLMFn

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    db_path = tmp_path / "tags.db"
    broad_tags = ["identity", "projects", "reference"]

    def fake_llm(prompt: str) -> str:
        lowered = prompt.lower()
        if "indexing a markdown note" in lowered:
            # Parse the note content from the prompt to make meta tags relevant.
            content = prompt.split("---", 2)[1] if "---" in prompt else ""
            meta = ["api", "auth", "users"] if "jwt" in content.lower() else ["general"]
            chunks = []
            if "# API" in content:
                chunks = [
                    {"heading_path": "# API", "summary": "API overview"},
                    {"heading_path": "# API / ## Auth", "summary": "Auth with JWT"},
                    {"heading_path": "# API / ## Users", "summary": "User endpoints"},
                ]
            else:
                chunks = [{"heading_path": "# (top)", "summary": "Top level note"}]
            return json.dumps({
                "broad_tag": "reference",
                "meta_tags": meta,
                "description": "Test note description",
                "chunks": chunks,
            })
        if "routing a user query" in lowered:
            return json.dumps({
                "broad_tag": "reference",
                "meta_tags": ["auth"],
                "keywords": ["jwt"],
            })
        if "selecting which note sections" in lowered:
            # Pick the Auth chunk (index 0 in the score-sorted candidate list) if JWT appears.
            query = prompt.split("Query:", 1)[1].split("\n", 1)[0] if "Query:" in prompt else ""
            if "jwt" in query.lower():
                return json.dumps({"selected_indices": [0]})
            return json.dumps({"selected_indices": []})
        return ""

    return {
        "notes_dir": notes_dir,
        "db_path": db_path,
        "broad_tags": broad_tags,
        "fake_llm": fake_llm,
    }


def test_chunk_markdown_basic(base_tag_config):
    content = "# API\n\n## Auth\nUse JWT.\n\n## Users\nCreate users."
    chunks = TagIndex._chunk_markdown(content)
    assert len(chunks) == 3
    assert chunks[0]["heading_path"] == "# API"
    assert chunks[1]["heading_path"] == "# API / ## Auth"
    assert chunks[1]["start_line"] == 3
    assert chunks[1]["end_line"] == 5  # 0-based exclusive boundary before next heading
    assert chunks[2]["heading_path"] == "# API / ## Users"


def test_chunk_markdown_no_headings(base_tag_config):
    content = "Just some text.\nMore text."
    chunks = TagIndex._chunk_markdown(content)
    assert len(chunks) == 1
    assert chunks[0]["heading_path"] == "# (top)"


def test_broad_tag_guardrail(base_tag_config):
    cfg = base_tag_config
    note = cfg["notes_dir"] / "x.md"
    note.write_text("# X\n")

    def bad_llm(prompt: str) -> str:
        return json.dumps({
            "broad_tag": "forbidden",
            "meta_tags": ["x"],
            "description": "x",
            "chunks": [{"heading_path": "# X", "summary": "x"}],
        })

    index = TagIndex(cfg["notes_dir"], cfg["db_path"], ["reference"], bad_llm)
    assert index.index_file(note)
    rows = list(index._conn().execute("SELECT broad_tag FROM docs"))
    assert len(rows) == 1
    assert rows[0]["broad_tag"] == "reference"  # coerced to first allowed tag


def test_index_and_search(base_tag_config):
    cfg = base_tag_config
    note = cfg["notes_dir"] / "api.md"
    note.write_text("# API\n\n## Auth\nUse JWT.\n\n## Users\nCreate users here.")

    index = TagIndex(cfg["notes_dir"], cfg["db_path"], cfg["broad_tags"], cfg["fake_llm"])
    stats = index.index_all()
    assert stats == {"indexed": 1, "failed": 0, "removed": 0}

    results = index.search("how do I authenticate with JWT?", top_k=3)
    assert len(results) == 1
    assert "JWT" in results[0].text
    assert results[0].heading_path == "# API / ## Auth"
    assert results[0].broad_tag == "reference"
    assert "auth" in results[0].meta_tags


def test_index_all_removes_stale_files(base_tag_config):
    cfg = base_tag_config
    note = cfg["notes_dir"] / "api.md"
    note.write_text("# API\nUseful info.")

    index = TagIndex(cfg["notes_dir"], cfg["db_path"], cfg["broad_tags"], cfg["fake_llm"])
    index.index_all()
    rows = list(index._conn().execute("SELECT path FROM docs"))
    assert len(rows) == 1

    note.unlink()
    stats = index.index_all()
    assert stats["removed"] == 1
    rows = list(index._conn().execute("SELECT path FROM docs"))
    assert len(rows) == 0


def test_search_falls_back_to_candidates(base_tag_config):
    """If the LLM selects nothing, return the top keyword-matched candidates."""
    cfg = base_tag_config
    note = cfg["notes_dir"] / "api.md"
    note.write_text("# API\n\n## Auth\nUse JWT.")

    def picky_llm(prompt: str) -> str:
        lowered = prompt.lower()
        if "indexing" in lowered:
            return cfg["fake_llm"](prompt)
        if "routing" in lowered:
            return json.dumps({"broad_tag": "reference", "meta_tags": [], "keywords": ["jwt"]})
        if "selecting" in lowered:
            return json.dumps({"selected_indices": []})
        return ""

    index = TagIndex(cfg["notes_dir"], cfg["db_path"], cfg["broad_tags"], picky_llm)
    index.index_all()
    results = index.search("jwt", top_k=2)
    assert len(results) >= 1
    assert "JWT" in results[0].text


def test_rag_uses_tag_index_when_enabled(base_tag_config, monkeypatch):
    """Integration: when tag_index_enabled is true, rag.Retriever returns tag chunks."""
    cfg = base_tag_config
    note = cfg["notes_dir"] / "api.md"
    note.write_text("# API\n\n## Auth\nUse JWT.")

    # Build the tag index first.
    index = TagIndex(cfg["notes_dir"], cfg["db_path"], cfg["broad_tags"], cfg["fake_llm"])
    index.index_all()

    from rag import Retriever
    monkeypatch.setattr("rag.NOTES_DIR", cfg["notes_dir"])
    monkeypatch.setattr("rag.DATA_DIR", cfg["notes_dir"].parent / "training_data")
    monkeypatch.setattr("rag.TRAIN_FILE", cfg["notes_dir"].parent / "training_data" / "train.jsonl")

    config = {
        "rag": {
            "enabled": True,
            "top_k": 3,
            "max_context_tokens": 500,
            "sources": ["notes"],
            "tag_index_enabled": True,
            "broad_tags": cfg["broad_tags"],
            "tag_index_db": str(cfg["db_path"]),
        }
    }
    retriever = Retriever(config)
    results = retriever.search_notes("how do I authenticate with JWT?")
    assert len(results) >= 1
    assert results[0]["source"] == "note"
    assert "JWT" in results[0]["text"]
    assert results[0]["broad_tag"] == "reference"
    assert results[0]["heading_path"] == "# API / ## Auth"


def test_rag_falls_back_to_keyword_when_tag_index_empty(base_tag_config, monkeypatch):
    """If the tag DB is empty, keyword search still works."""
    cfg = base_tag_config
    note = cfg["notes_dir"] / "hobbies.md"
    note.write_text("# Hobbies\nThe user likes hiking.")

    from rag import Retriever
    monkeypatch.setattr("rag.NOTES_DIR", cfg["notes_dir"])
    monkeypatch.setattr("rag.DATA_DIR", cfg["notes_dir"].parent / "training_data")
    monkeypatch.setattr("rag.TRAIN_FILE", cfg["notes_dir"].parent / "training_data" / "train.jsonl")

    config = {
        "rag": {
            "enabled": True,
            "top_k": 3,
            "max_context_tokens": 500,
            "sources": ["notes"],
            "tag_index_enabled": True,
            "broad_tags": cfg["broad_tags"],
            "tag_index_db": str(cfg["db_path"]),
        }
    }
    retriever = Retriever(config)
    results = retriever.search_notes("hiking")
    assert len(results) == 1
    assert "hiking" in results[0]["text"]
