"""Persistent memory / learning dataset."""

import json
import sqlite3
from pathlib import Path

from symbio.mcp.config import settings
from symbio.mcp.models import MemoryEntry


class MemoryStore:
    """SQLite-backed store for frontier-labeled examples."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.memory_db
        self._ensure_db()

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    skill_tag TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    local_output TEXT,
                    frontier_output TEXT NOT NULL,
                    failure_reason TEXT,
                    validator TEXT,
                    expected_schema TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_skill_tag ON memory(skill_tag)"
            )
            # Backwards-compatible migration for older tables missing new columns.
            for col in ("validator", "expected_schema"):
                try:
                    conn.execute(f"ALTER TABLE memory ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass

    def save(self, entry: MemoryEntry) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO memory (skill_tag, prompt, local_output, frontier_output, failure_reason, validator, expected_schema)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.skill_tag,
                    entry.prompt,
                    entry.local_output,
                    entry.frontier_output,
                    entry.failure_reason,
                    entry.validator,
                    json.dumps(entry.expected_schema) if entry.expected_schema else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def count_misses(self, skill_tag: str | None = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            if skill_tag:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memory WHERE skill_tag = ?",
                    (skill_tag,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memory").fetchone()
            return row[0] if row else 0

    def get_examples(self, skill_tag: str, limit: int = 100) -> list[MemoryEntry]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, created_at, skill_tag, prompt, local_output, frontier_output, failure_reason, validator, expected_schema
                FROM memory WHERE skill_tag = ? ORDER BY created_at DESC LIMIT ?
                """,
                (skill_tag, limit),
            ).fetchall()
        return [
            MemoryEntry(
                id=r["id"],
                created_at=r["created_at"],
                skill_tag=r["skill_tag"],
                prompt=r["prompt"],
                local_output=r["local_output"],
                frontier_output=r["frontier_output"],
                failure_reason=r["failure_reason"],
                validator=r["validator"],
                expected_schema=json.loads(r["expected_schema"]) if r["expected_schema"] else None,
            )
            for r in rows
        ]

    def export_jsonl(self, skill_tag: str, output_path: Path) -> int:
        examples = self.get_examples(skill_tag, limit=10_000)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": ex.prompt},
                                {"role": "assistant", "content": ex.frontier_output},
                            ],
                            "metadata": {
                                "local_output": ex.local_output,
                                "failure_reason": ex.failure_reason,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(examples)
