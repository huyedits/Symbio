"""Per-session JSONL conversation history, searchable across past sessions."""

import json
import re
from datetime import datetime
from typing import Any

from symbio import constants


class SessionStore:
    """One JSONL file per chat session; keyword search across past sessions
    feeds the RAG retriever so old conversations stay findable."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.path = constants.SESSIONS_DIR / f"{session_id}.jsonl"

    def log(self, role: str, content: str):
        content = content.strip()
        if not content:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            json.dump({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }, f)
            f.write("\n")

    @staticmethod
    def search(query: str, limit: int = 5, exclude_session: str | None = None) -> list[dict[str, Any]]:
        terms = {w for w in re.sub(r"[^\w\s]", " ", query.lower()).split() if len(w) > 1}
        if not terms:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        # Newest sessions first; cap the scan so search stays fast forever.
        files = sorted(constants.SESSIONS_DIR.glob("*.jsonl"), reverse=True)[:100]
        for path in files:
            if exclude_session and path.stem == exclude_session:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                words = set(re.sub(r"[^\w\s]", " ", row.get("content", "").lower()).split())
                overlap = len(terms & words)
                if overlap:
                    scored.append((overlap / len(terms), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:limit]]
