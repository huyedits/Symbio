"""Lightweight retrieval-augmented generation (RAG) for Symbio.

No external embedding model or vector DB is required. Retrieval uses:
- keyword overlap + simple term-frequency scoring over notes and training data,
- SQLite FTS5 over past conversation sessions.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).parent.resolve()
NOTES_DIR = PROJECT_DIR / "notes"
DATA_DIR = PROJECT_DIR / "training_data"
TRAIN_FILE = DATA_DIR / "train.jsonl"


def _token_count_approx(text: str) -> int:
    """Rough token count: ~1 token per 4 characters for typical English text."""
    return max(1, len(text) // 4)


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    return [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(w) > 1]


class Retriever:
    """Keyword-based retriever over notes, sessions, and training data."""

    def __init__(
        self,
        config: dict[str, Any],
        session_store: Any | None = None,
        exclude_session_id: str | None = None,
    ):
        self.config = config
        self.rag_cfg = config.get("rag", {})
        self.session_store = session_store
        # The live session is already in the agent's history; retrieving it
        # again just echoes the current question back into the prompt.
        self.exclude_session_id = exclude_session_id
        self._note_cache: dict[str, str] | None = None

    def _enabled_sources(self) -> list[str]:
        return list(self.rag_cfg.get("sources", ["notes", "sessions"]))

    def _top_k(self) -> int:
        return int(self.rag_cfg.get("top_k", 5))

    def _max_context_tokens(self) -> int:
        return int(self.rag_cfg.get("max_context_tokens", 1500))

    def _load_notes(self) -> dict[str, str]:
        if self._note_cache is not None:
            return self._note_cache
        notes: dict[str, str] = {}
        if NOTES_DIR.exists():
            for path in sorted(NOTES_DIR.glob("*.md")):
                try:
                    notes[path.name] = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
        self._note_cache = notes
        return notes

    def invalidate_cache(self):
        """Call after notes are written or removed."""
        self._note_cache = None

    def _score(self, query_terms: list[str], text: str) -> float:
        terms = _normalize(text)
        if not terms:
            return 0.0
        counts = Counter(terms)
        score = sum(counts[t] for t in query_terms)
        # Normalize by document length so long docs do not always win.
        return score / (len(terms) ** 0.5 + 1)

    def search_notes(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if "notes" not in self._enabled_sources():
            return []
        query_terms = _normalize(query)
        if not query_terms:
            return []
        notes = self._load_notes()
        scored = []
        for name, text in notes.items():
            s = self._score(query_terms, text)
            if s > 0:
                scored.append({
                    "source": "note",
                    "title": name,
                    "text": text,
                    "score": s,
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: (top_k or self._top_k())]

    def search_sessions(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if "sessions" not in self._enabled_sources() or self.session_store is None:
            return []
        rows = self.session_store.search(
            query,
            limit=top_k or self._top_k(),
            exclude_session=self.exclude_session_id,
        )
        query_terms = _normalize(query)
        results = []
        for r in rows:
            # Tool transcripts and system observations are noise as retrieved
            # context and can derail the model's own tool-call formatting.
            if r["role"] == "tool":
                continue
            content = r["content"]
            if "<tool_call" in content or content.startswith("[System observation"):
                continue
            # A past turn that just repeats the current question adds nothing
            # and echoing it back destabilizes the model.
            if _normalize(content) == query_terms:
                continue
            preview = content[:500].replace("\n", " ")
            results.append({
                "source": "session",
                "timestamp": r["timestamp"],
                "role": r["role"],
                "text": preview,
                "score": 1.0,
            })
        return results

    def search_training_data(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if "training_data" not in self._enabled_sources():
            return []
        query_terms = _normalize(query)
        if not query_terms or not TRAIN_FILE.exists():
            return []
        scored = []
        try:
            with open(TRAIN_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                    except Exception:
                        text = line
                    s = self._score(query_terms, text)
                    if s > 0:
                        scored.append({
                            "source": "training_data",
                            "text": text[:500].replace("\n", " "),
                            "score": s,
                        })
        except Exception:
            pass
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: (top_k or self._top_k())]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return ranked results across all enabled sources."""
        all_results: list[dict[str, Any]] = []
        all_results.extend(self.search_notes(query, top_k=top_k))
        all_results.extend(self.search_sessions(query, top_k=top_k))
        all_results.extend(self.search_training_data(query, top_k=top_k))
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[: (top_k or self._top_k())]

    def build_context(self, query: str) -> str:
        """Build a compact, citation-rich context string for the prompt."""
        if not self.rag_cfg.get("enabled", True):
            return ""

        results = self.retrieve(query)
        if not results:
            return ""

        max_tokens = self._max_context_tokens()
        lines = ["Retrieved context (use this first when answering):"]
        used_tokens = _token_count_approx(lines[0])

        for i, r in enumerate(results, 1):
            source = r["source"]
            if source == "note":
                header = f"[Note: {r['title']}]"
            elif source == "session":
                ts = r.get("timestamp", "?")
                role = r.get("role", "?")
                header = f"[Past session {ts} / {role}]"
            else:
                header = f"[Training sample]"
            body = r["text"].strip().replace("\n", " ")
            snippet = f"{header}\n{body}"
            tokens = _token_count_approx(snippet)
            if used_tokens + tokens > max_tokens:
                break
            lines.append(snippet)
            used_tokens += tokens

        return "\n\n".join(lines)
