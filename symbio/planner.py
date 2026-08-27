"""Deferred training planner for Symbio.

Accumulates conversation history, detects recurring facts, cross-references them,
and runs a lightweight neutrality review before any fine-tuning happens.

The planner is conservative: training only proceeds after enough turns,
enough repetitions, and (if enabled) a successful neutrality review.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).parent.resolve()
PLANNER_DB = PROJECT_DIR / "logs" / "planner.db"


_SUBJECTIVE_WORDS = frozenset({
    "always", "never", "best", "worst", "hate", "love", "terrible", "amazing",
    "perfect", "awful", "must", "should", "obviously", "clearly", "definitely",
    "undoubtedly", "everyone knows", "nobody", "all", "none",
})

_UNCERTAIN_FACT_PATTERNS = [
    re.compile(r"\b(is|are|was|were|will be|has|have)\s+[^.]{3,60}\b", re.IGNORECASE),
]


class TrainingPlanner:
    """SQLite-backed planner that gates fine-tuning behind review thresholds."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.cfg = config.get("training_planner", {})
        PLANNER_DB.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(PLANNER_DB))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS turns ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "timestamp TEXT,"
                "user_input TEXT,"
                "assistant_reply TEXT,"
                "tools TEXT"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS note_refs ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "timestamp TEXT,"
                "note_title TEXT,"
                "count INTEGER DEFAULT 0"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS samples ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "timestamp TEXT,"
                "source TEXT,"
                "text TEXT,"
                "status TEXT DEFAULT 'pending',"
                "reason TEXT"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_note_refs_title ON note_refs(note_title)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status)"
            )
            conn.commit()
        finally:
            conn.close()

    # ---------- Recording ----------

    def record_turn(self, user_input: str, assistant_reply: str, tools: list[str] | None = None):
        if not self.cfg.get("enabled", True):
            return
        tools_json = json.dumps(tools or [])
        conn = sqlite3.connect(str(PLANNER_DB))
        try:
            conn.execute(
                "INSERT INTO turns (timestamp, user_input, assistant_reply, tools) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), user_input, assistant_reply, tools_json),
            )
            conn.commit()
        finally:
            conn.close()

    def record_note_ref(self, note_title: str):
        if not self.cfg.get("enabled", True):
            return
        ts = datetime.now().isoformat()
        conn = sqlite3.connect(str(PLANNER_DB))
        try:
            # Aggregate per title.
            row = conn.execute(
                "SELECT id, count FROM note_refs WHERE note_title = ? ORDER BY id DESC LIMIT 1",
                (note_title,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE note_refs SET count = count + 1, timestamp = ? WHERE id = ?",
                    (ts, row[0]),
                )
            else:
                conn.execute(
                    "INSERT INTO note_refs (timestamp, note_title, count) VALUES (?, ?, ?)",
                    (ts, note_title, 1),
                )
            conn.commit()
        finally:
            conn.close()

    def record_correction(self, user_input: str):
        """Record an explicit correction from the user."""
        if not self.cfg.get("enabled", True):
            return
        self.record_turn(user_input, "[user correction]", tools=["correction"])

    def add_sample(self, text: str, source: str = "manual"):
        """Add a candidate training sample awaiting review."""
        if not self.cfg.get("enabled", True):
            return None
        status, reason = self.review_sample(text)
        conn = sqlite3.connect(str(PLANNER_DB))
        try:
            cur = conn.execute(
                "INSERT INTO samples (timestamp, source, text, status, reason) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), source, text, status, reason),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    # ---------- Analysis ----------

    def turn_count(self) -> int:
        conn = sqlite3.connect(str(PLANNER_DB))
        try:
            row = conn.execute("SELECT COUNT(*) FROM turns").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def top_note_refs(self, limit: int = 10) -> list[tuple[str, int]]:
        conn = sqlite3.connect(str(PLANNER_DB))
        try:
            rows = conn.execute(
                "SELECT note_title, SUM(count) FROM note_refs GROUP BY note_title ORDER BY SUM(count) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        finally:
            conn.close()

    def pending_samples(self, status: str | None = None) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(PLANNER_DB))
        try:
            if status:
                rows = conn.execute(
                    "SELECT id, timestamp, source, text, status, reason FROM samples WHERE status = ? ORDER BY id",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, timestamp, source, text, status, reason FROM samples ORDER BY id"
                ).fetchall()
            return [
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "source": r[2],
                    "text": r[3],
                    "status": r[4],
                    "reason": r[5],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ---------- Neutrality review ----------

    def review_sample(self, text: str) -> tuple[str, str]:
        """Return (status, reason) for a candidate training sample.

        Status values:
          approved  - passes heuristic checks.
          pending   - needs model-based review or manual review.
          rejected  - clearly biased or unsafe.
        """
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))
        subjective_hits = words & _SUBJECTIVE_WORDS
        if subjective_hits:
            return (
                "pending",
                f"Subjective/bias words detected: {', '.join(sorted(subjective_hits))}. "
                "Awaiting cross-reference or manual review.",
            )

        # Flag unsupported factual claims if the sample makes strong assertions.
        if len(text) > 80:
            fact_like = sum(1 for p in _UNCERTAIN_FACT_PATTERNS if p.search(text))
            if fact_like > 2 and not self.cfg.get("neutrality_review", True):
                # If model review is disabled, be conservative and mark pending.
                return (
                    "pending",
                    "Multiple unsupported factual claims detected; neutrality review is disabled.",
                )

        return "approved", "Heuristic review passed."

    # ---------- Training decision ----------

    def should_train(self, model_type: str = "dense") -> tuple[bool, str]:
        """Return (ok, reason) for whether fine-tuning should proceed now."""
        if not self.cfg.get("enabled", True):
            return False, "Training planner is disabled."

        min_turns = int(self.cfg.get("min_turns", 200))
        min_reps = int(self.cfg.get("min_repetitions", 3))
        auto_train = bool(self.cfg.get("auto_train", False))

        turns = self.turn_count()
        if turns < min_turns:
            return False, f"Only {turns}/{min_turns} recorded turns; accumulate more history first."

        top_notes = self.top_note_refs(limit=1)
        if not top_notes or top_notes[0][1] < min_reps:
            return False, f"No fact/note has been referenced at least {min_reps} times."

        approved = self.pending_samples(status="approved")
        if not approved:
            return False, "No approved training samples pending. Run /digest or wait for approved history pairs."

        if model_type == "moe":
            mode = self.config.get("model", {}).get("moe_fine_tuning_mode", "rag_only")
            if not self.config.get("model", {}).get("allow_moe_lora", False):
                return False, f"MoE model: fine-tuning is in '{mode}' mode. Use /train --force to override."

        if not auto_train:
            return False, (
                f"Thresholds met ({turns} turns, top note referenced {top_notes[0][1]} times, "
                f"{len(approved)} approved samples). Auto-train is disabled; use /train --force."
            )

        return True, (
            f"Thresholds met ({turns} turns, top note referenced {top_notes[0][1]} times, "
            f"{len(approved)} approved samples). Auto-train enabled."
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.cfg.get("enabled", True),
            "turn_count": self.turn_count(),
            "top_note_refs": self.top_note_refs(limit=5),
            "pending_samples": len(self.pending_samples(status="pending")),
            "approved_samples": len(self.pending_samples(status="approved")),
            "rejected_samples": len(self.pending_samples(status="rejected")),
        }
