"""Hierarchical tag-based RAG index for markdown notes.

The index is intentionally bare:
- SQLite stores only metadata (path, tags, descriptions, chunk summaries, line ranges).
- No embeddings, no full file content in the DB.
- At query time the AI routes broad tag → meta tags → description → chunk,
  then reads only the selected line ranges from disk.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

LLMFn = Callable[[str], str]

_BROAD_TAG_PROMPT = """You are indexing a markdown note for a retrieval system.
Read the note below and produce a JSON object with these exact keys:
- "broad_tag": one of {broad_tags}
- "meta_tags": an array of 3 to 7 specific topic keywords (lowercase, no spaces)
- "description": one sentence summarizing the note (max 160 characters)
- "chunks": an array of objects, one per major heading section, each with:
    - "heading_path": the heading breadcrumb, e.g. "# API / ## Auth" (use "# (top)" if no heading)
    - "summary": one sentence describing what this section covers (max 120 characters)

Available broad tags: {broad_tags}

Note content:
---
{content}
---

Respond with valid JSON only. Do not include markdown code fences."""

_QUERY_ROUTE_PROMPT = """You are routing a user query to the right note category.
Given the query, pick the best broad tag and extract narrow concepts.

Query: {query}
Available broad tags: {broad_tags}

Respond with valid JSON only:
{{
  "broad_tag": "one of the available broad tags",
  "meta_tags": ["specific", "topic", "keywords"],
  "keywords": ["important", "query", "terms"]
}}

Do not include markdown code fences."""

_CHUNK_SELECT_PROMPT = """You are selecting which note sections to read to answer a query.
Pick only sections that are directly relevant.

Query: {query}

Candidate sections:
{candidates}

Respond with valid JSON only:
{{
  "selected_indices": [0, 2, ...]
}}
Use an empty array if none are relevant. Do not include markdown code fences."""


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    return [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(w) > 2]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _json_extract(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find the first JSON object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


@dataclass(frozen=True)
class ChunkResult:
    path: Path
    heading_path: str
    start_line: int
    end_line: int
    text: str
    score: float
    broad_tag: str
    meta_tags: list[str]
    description: str
    summary: str


class TagIndex:
    """Bare hierarchical tag index over a notes directory."""

    def __init__(
        self,
        notes_dir: Path,
        db_path: Path,
        broad_tags: list[str],
        llm_fn: LLMFn | None = None,
    ):
        self.notes_dir = Path(notes_dir).resolve()
        self.db_path = Path(db_path)
        self.broad_tags = [t.lower().strip() for t in broad_tags]
        self.llm_fn = llm_fn
        self._local_conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        if self._local_conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local_conn = sqlite3.connect(str(self.db_path))
            self._local_conn.row_factory = sqlite3.Row
        return self._local_conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                broad_tag TEXT NOT NULL,
                meta_tags TEXT NOT NULL,
                description TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                heading_path TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                summary TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES docs(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_docs_broad ON docs(broad_tag);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        conn.commit()

    def invalidate(self) -> None:
        """Close any open DB connection."""
        if self._local_conn is not None:
            try:
                self._local_conn.close()
            except Exception:
                pass
            self._local_conn = None

    def _rel_path(self, path: Path) -> str:
        """Return a stable relative path string for a note file."""
        abs_path = Path(path).resolve()
        try:
            return str(abs_path.relative_to(self.notes_dir))
        except ValueError:
            return str(abs_path)

    def _validate_broad_tag(self, tag: str) -> str:
        """Guardrail: broad tags must come from the configured whitelist.

        An empty/whitespace tag means "unsure" and is preserved so the search
        can fall back to scanning all broad tags. Non-empty but invalid tags
        are coerced to the first allowed tag.
        """
        cleaned = tag.lower().strip()
        if not cleaned:
            return ""
        if cleaned not in self.broad_tags:
            if self.broad_tags:
                cleaned = self.broad_tags[0]
            else:
                raise ValueError(f"No allowed broad tags configured; cannot accept '{tag}'")
        return cleaned

    @staticmethod
    def _chunk_markdown(content: str) -> list[dict[str, Any]]:
        """Split markdown into heading-section chunks.

        A section starts at its heading and continues until the next heading
        of equal or higher precedence (lower or equal '#' count).
        """
        lines = content.splitlines()
        if not lines:
            return []

        # First pass: find heading boundaries.
        headings: list[tuple[int, int, str]] = []  # (level, line_index, title)
        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append((level, i, title))

        if not headings:
            # No headings: treat the whole file as one chunk.
            return [{"heading_path": "# (top)", "start_line": 1, "end_line": len(lines)}]

        chunks: list[dict[str, Any]] = []
        stack: list[tuple[int, str]] = []

        for idx, (level, line_index, title) in enumerate(headings):
            # Pop stack until the top is strictly higher level (smaller number).
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, f"{'#' * level} {title}"))
            heading_path = " / ".join(t for _, t in stack)

            # Section ends at the next heading of equal or higher precedence,
            # or at the end of the file.
            end_line = len(lines)
            for next_level, next_line, _ in headings[idx + 1 :]:
                if next_level <= level:
                    end_line = next_line
                    break

            chunks.append(
                {
                    "heading_path": heading_path,
                    "start_line": line_index + 1,
                    "end_line": end_line,
                }
            )

        return chunks

    def _read_lines(self, path: Path, start: int, end: int) -> str:
        """Read a line range from a file (1-based, inclusive start)."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            start_idx = max(0, start - 1)
            end_idx = max(start_idx, min(len(lines), end))
            return "\n".join(lines[start_idx:end_idx])
        except Exception:
            return ""

    def _generate_file_metadata(
        self, path: Path, content: str
    ) -> dict[str, Any] | None:
        """Ask the LLM to produce broad tag, meta tags, description, and chunk summaries."""
        if self.llm_fn is None:
            return None

        raw_chunks = self._chunk_markdown(content)
        chunk_descriptions = "\n".join(
            f"{i}. {c['heading_path']}: lines {c['start_line']}-{c['end_line']}"
            for i, c in enumerate(raw_chunks)
        )

        prompt = _BROAD_TAG_PROMPT.format(
            broad_tags=json.dumps(self.broad_tags),
            content=content[:8000],
            chunk_list=chunk_descriptions,
        )

        try:
            response = self.llm_fn(prompt)
            parsed = _json_extract(response)
        except Exception:
            return None

        broad_tag = self._validate_broad_tag(str(parsed.get("broad_tag", "")))
        meta_tags = [str(t).lower().strip() for t in parsed.get("meta_tags", []) if t]
        description = str(parsed.get("description", "")).strip()[:200]

        chunk_summaries = parsed.get("chunks", [])
        if not isinstance(chunk_summaries, list):
            chunk_summaries = []

        # Merge LLM summaries with our line ranges.
        chunks: list[dict[str, Any]] = []
        for i, raw in enumerate(raw_chunks):
            summary = ""
            if i < len(chunk_summaries):
                summary = str(chunk_summaries[i].get("summary", "")).strip()[:160]
            if not summary:
                summary = f"Section {raw['heading_path']}"
            chunks.append(
                {
                    "heading_path": raw["heading_path"],
                    "start_line": raw["start_line"],
                    "end_line": raw["end_line"],
                    "summary": summary,
                }
            )

        return {
            "broad_tag": broad_tag,
            "meta_tags": meta_tags,
            "description": description,
            "chunks": chunks,
        }

    def index_file(self, path: Path) -> bool:
        """Index or reindex a single markdown file. Returns True on success."""
        path = Path(path)
        if not path.exists() or path.suffix.lower() != ".md":
            return False

        rel = self._rel_path(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        metadata = self._generate_file_metadata(path, content)
        if metadata is None:
            return False

        conn = self._conn()
        conn.execute("DELETE FROM docs WHERE path = ?", (rel,))
        cur = conn.execute(
            """
            INSERT INTO docs (path, broad_tag, meta_tags, description, indexed_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (
                rel,
                metadata["broad_tag"],
                json.dumps(metadata["meta_tags"]),
                metadata["description"],
            ),
        )
        doc_id = cur.lastrowid
        for chunk in metadata["chunks"]:
            conn.execute(
                """
                INSERT INTO chunks (doc_id, heading_path, start_line, end_line, summary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    chunk["heading_path"],
                    chunk["start_line"],
                    chunk["end_line"],
                    chunk["summary"],
                ),
            )
        conn.commit()
        return True


    def _is_fresh(self, path: Path, rel: str) -> bool:
        """Return True if the file has already been indexed and not changed since.

        Files are reindexed when:
        - they are not in the DB yet,
        - their mtime is newer than the stored indexed_at timestamp, or
        - their size changed.
        """
        row = self._conn().execute(
            "SELECT indexed_at FROM docs WHERE path = ?", (rel,)
        ).fetchone()
        if row is None:
            return False
        try:
            stat = path.stat()
            indexed_at = datetime.fromisoformat(row["indexed_at"])
            # Reindex if mtime is newer than the index timestamp.
            if datetime.fromtimestamp(stat.st_mtime) > indexed_at:
                return False
        except Exception:
            return False
        return True

    def index_all(self, force: bool = False) -> dict[str, Any]:
        """Index all markdown files in notes_dir. Return stats + errors."""
        if not self.notes_dir.exists():
            return {"indexed": 0, "failed": 0, "removed": 0, "errors": []}

        conn = self._conn()

        # Find existing indexed paths.
        existing_rows = conn.execute("SELECT path FROM docs").fetchall()
        existing = {row["path"] for row in existing_rows}

        current: set[str] = set()
        indexed = 0
        failed = 0
        skipped = 0
        errors: list[str] = []

        for path in sorted(self.notes_dir.rglob("*.md")):
            rel = self._rel_path(path)
            current.add(rel)
            # Skip files that have not changed since the last index.
            if not force and rel in existing and self._is_fresh(path, rel):
                skipped += 1
                continue
            try:
                if self.index_file(path):
                    indexed += 1
                else:
                    failed += 1
                    errors.append(f"{rel}: metadata generation failed (LLM returned nothing or bad JSON)")
            except Exception as exc:
                failed += 1
                errors.append(f"{rel}: {exc}")

        # Remove stale entries for files that no longer exist.
        removed = 0
        for stale in existing - current:
            conn.execute("DELETE FROM docs WHERE path = ?", (stale,))
            removed += 1
        conn.commit()

        return {"indexed": indexed, "failed": failed, "removed": removed, "skipped": skipped, "errors": errors}


    def _route_query(self, query: str) -> dict[str, Any]:
        """Use the LLM to pick a broad tag and extract query concepts."""
        if self.llm_fn is None:
            return {"broad_tag": "", "meta_tags": [], "keywords": _normalize(query)}

        prompt = _QUERY_ROUTE_PROMPT.format(
            query=query,
            broad_tags=json.dumps(self.broad_tags),
        )
        try:
            parsed = _json_extract(self.llm_fn(prompt))
        except Exception:
            parsed = {}

        broad_tag = self._validate_broad_tag(str(parsed.get("broad_tag", "")))
        meta_tags = [str(t).lower().strip() for t in parsed.get("meta_tags", []) if t]
        keywords = [str(t).lower().strip() for t in parsed.get("keywords", []) if t]
        if not keywords:
            keywords = _normalize(query)

        return {"broad_tag": broad_tag, "meta_tags": meta_tags, "keywords": keywords}

    def _select_chunks(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Ask the LLM which candidate chunks are relevant."""
        if self.llm_fn is None or not candidates:
            return candidates

        numbered = "\n".join(
            f"{i}. [{c['path']}] {c['chunk']['heading_path']}: {c['chunk']['summary']}"
            for i, c in enumerate(candidates)
        )
        prompt = _CHUNK_SELECT_PROMPT.format(query=query, candidates=numbered)

        try:
            parsed = _json_extract(self.llm_fn(prompt))
            indices = [int(i) for i in parsed.get("selected_indices", [])]
        except Exception:
            return candidates

        return [candidates[i] for i in indices if 0 <= i < len(candidates)]

    def search(self, query: str, top_k: int = 5) -> list[ChunkResult]:
        """Hierarchical tag search: broad tag → meta tags → chunks.

        Fast path: if the index is empty, return nothing immediately so the
        caller can fall back to keyword search without burning an LLM call.
        """
        conn = self._conn()
        doc_count = conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"]
        if doc_count == 0:
            return []

        route = self._route_query(query)
        broad_tag = route["broad_tag"]
        meta_tags = set(route["meta_tags"])
        keywords = set(route["keywords"])

        # Broad-tag filter: if the LLM picked a valid tag, restrict to it;
        # otherwise scan all docs.
        if broad_tag and broad_tag in self.broad_tags:
            rows = conn.execute(
                "SELECT * FROM docs WHERE broad_tag = ?", (broad_tag,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM docs").fetchall()

        if not rows:
            return []

        # Score docs by meta tag + description keyword overlap.
        doc_scores: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            row_meta = set(json.loads(row["meta_tags"]))
            meta_score = _jaccard(meta_tags, row_meta) * 2.0
            desc_score = _jaccard(keywords, set(_normalize(row["description"]))) * 1.0
            doc_scores.append((meta_score + desc_score, row))

        doc_scores.sort(key=lambda x: x[0], reverse=True)
        top_docs = doc_scores[: max(3, top_k)]

        # Gather candidate chunks from top docs.
        candidates: list[dict[str, Any]] = []
        for _, doc in top_docs:
            chunk_rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ?", (doc["id"],)
            ).fetchall()
            for chunk in chunk_rows:
                chunk_keywords = set(_normalize(chunk["summary"]))
                score = _jaccard(keywords, chunk_keywords)
                candidates.append(
                    {
                        "doc": doc,
                        "chunk": chunk,
                        "score": score,
                        "path": doc["path"],
                    }
                )

        # Rank by chunk score and ask the LLM to pick.
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[: max(top_k * 3, 10)]
        selected = self._select_chunks(query, candidates)

        # Fall back to top candidates if the LLM selected nothing.
        if not selected:
            selected = candidates[:top_k]

        # Keep the highest-scored selected chunks and return them in score order.
        selected.sort(key=lambda x: x["score"], reverse=True)

        results: list[ChunkResult] = []
        for item in selected[:top_k]:
            doc = item["doc"]
            chunk = item["chunk"]
            abs_path = self.notes_dir / doc["path"]
            text = self._read_lines(abs_path, chunk["start_line"], chunk["end_line"])
            results.append(
                ChunkResult(
                    path=abs_path,
                    heading_path=chunk["heading_path"],
                    start_line=chunk["start_line"],
                    end_line=chunk["end_line"],
                    text=text,
                    score=item["score"],
                    broad_tag=doc["broad_tag"],
                    meta_tags=json.loads(doc["meta_tags"]),
                    description=doc["description"],
                    summary=chunk["summary"],
                )
            )

        return results
