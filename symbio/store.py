"""SQLite session store for Symbio."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionStore:
    """SQLite-backed store for conversation turns with full-text search."""

    def __init__(self, path: Path):
        self.path = path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.path))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS turns ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "timestamp TEXT,"
                "role TEXT,"
                "content TEXT,"
                "session_id TEXT"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "id TEXT PRIMARY KEY,"
                "started TEXT"
                ")"
            )
            try:
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(content, content_rowid=rowid)")
            except sqlite3.Error:
                pass
            conn.commit()
        finally:
            conn.close()

    def new_session(self, session_id: str):
        conn = sqlite3.connect(str(self.path))
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, started) VALUES (?, ?)",
                (session_id, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def append(self, session_id: str, role: str, content: str):
        ts = datetime.now().isoformat()
        conn = sqlite3.connect(str(self.path))
        try:
            cur = conn.execute(
                "INSERT INTO turns (timestamp, role, content, session_id) VALUES (?, ?, ?, ?)",
                (ts, role, content, session_id),
            )
            try:
                conn.execute(
                    "INSERT INTO turns_fts (rowid, content) VALUES (?, ?)",
                    (cur.lastrowid, content),
                )
            except sqlite3.Error:
                pass
            conn.commit()
        finally:
            conn.close()

    def search(
        self, query: str, limit: int = 10, exclude_session: str | None = None
    ) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(self.path))
        try:
            exclude_sql = " AND t.session_id != ?" if exclude_session else ""
            exclude_params = (exclude_session,) if exclude_session else ()
            try:
                rows = conn.execute(
                    "SELECT t.timestamp, t.role, t.content FROM turns_fts f "
                    f"JOIN turns t ON t.id = f.rowid WHERE turns_fts MATCH ?{exclude_sql} "
                    "ORDER BY rank LIMIT ?",
                    (query, *exclude_params, limit),
                ).fetchall()
            except sqlite3.Error:
                rows = conn.execute(
                    "SELECT timestamp, role, content FROM turns t WHERE content LIKE ?"
                    f"{exclude_sql} ORDER BY id DESC LIMIT ?",
                    (f"%{query}%", *exclude_params, limit),
                ).fetchall()
            return [{"timestamp": r[0], "role": r[1], "content": r[2]} for r in rows]
        finally:
            conn.close()
