"""Smoke test for the memory store."""

import tempfile
from pathlib import Path

from symbio.mcp.memory import MemoryStore
from symbio.mcp.models import MemoryEntry


def test_save_and_count(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    store = MemoryStore(db)
    store.save(
        MemoryEntry(
            skill_tag="math",
            prompt="2+2=?",
            local_output="5",
            frontier_output="4",
            failure_reason="wrong",
        )
    )
    assert store.count_misses("math") == 1
    assert store.count_misses("general") == 0
    assert store.count_misses() == 1


def test_export_jsonl(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    store = MemoryStore(db)
    store.save(
        MemoryEntry(
            skill_tag="json",
            prompt='Extract: {"name": "Ada"}',
            local_output=None,
            frontier_output='{"name": "Ada"}',
            failure_reason="not json",
        )
    )
    out = tmp_path / "out.jsonl"
    n = store.export_jsonl("json", out)
    assert n == 1
    assert out.exists()
