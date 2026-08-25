"""Lightweight retrieval-augmented generation (RAG) for Symbio.

No external embedding model or vector DB is required. Retrieval uses:
- keyword overlap + simple term-frequency scoring over notes and training data,
- SQLite FTS5 over past conversation sessions.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from symbio import safety

try:
    from tag_rag import TagIndex
except Exception:
    TagIndex = None  # type: ignore


PROJECT_DIR = Path(__file__).parent.resolve()
NOTES_DIR = PROJECT_DIR / "notes"
DATA_DIR = PROJECT_DIR / "training_data"
TRAIN_FILE = DATA_DIR / "train.jsonl"

# Cap how much of train.jsonl is scanned per query. Most relevant samples are
# near the recent end of the file; scanning the whole file is a common latency
# cliff for long-running bots.
_TRAIN_SCAN_MAX_BYTES = 2_000_000


def _token_count_approx(text: str) -> int:
    """Rough token count: ~1 token per 4 characters for typical English text."""
    return max(1, len(text) // 4)


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    return [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(w) > 1]


def _default_tag_llm_fn(prompt: str) -> str:
    """Best-effort synchronous LLM for tag indexing and query routing.

    Tries the local Ollama brain first (cheap, offline-friendly), then falls
    back to the frontier provider if no local brain is reachable or no API key.
    Returns an empty string on total failure so the tag index can fall back to
    keyword-based retrieval.
    """
    try:
        from symbio.mcp.config import settings
    except Exception:
        return ""

    # Try local Ollama brain first.
    try:
        import httpx
        headers = {}
        if settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            headers=headers,
            json={
                "model": settings.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.local_temperature,
                    "num_predict": min(settings.local_max_tokens, 2048),
                },
            },
            timeout=min(settings.local_timeout, 5.0),
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        # Strip thinking tags that local models sometimes emit.
        text = re.sub(r"\bthinking\b.*?/\bthinking\b", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()
    except Exception:
        pass

    # Fall back to frontier provider.
    if not settings.frontier_api_key:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=settings.frontier_api_key,
            timeout=settings.frontier_timeout,
        )
        response = client.messages.create(
            model=settings.frontier_model,
            max_tokens=min(settings.frontier_max_tokens, 2048),
            temperature=settings.frontier_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(block.text for block in response.content if block.type == "text")
    except Exception:
        return ""


class Retriever:
    """Keyword-based retriever over notes, sessions, and training data."""

    def __init__(
        self,
        config: dict[str, Any],
        session_store: Any | None = None,
        exclude_session_id: str | None = None,
        llm_fn: Any = None,
    ):
        self.config = config
        self.rag_cfg = config.get("rag", {})
        self.session_store = session_store
        # Tag-LLM used by the hierarchical tag index. Defaults to the Ollama/
        # frontier fallback; the live agent passes its already-loaded MLX model
        # here so the index doesn't spin up a separate Ollama process (which
        # cost ~20s/turn). The MLX fn lazily loads the model on first use, so
        # passing it at construction (before the model is loaded) is safe.
        self._llm_fn = llm_fn
        # The live session is already in the agent's history; retrieving it
        # again just echoes the current question back into the prompt.
        self.exclude_session_id = exclude_session_id
        self._note_cache: dict[str, str] | None = None
        # LRU-ish cache for build_context so repeated or rephrased questions
        # do not re-scan notes/sessions/training data every turn.
        self._context_cache: dict[str, tuple[str, float]] = {}
        self._context_cache_ttl = float(self.rag_cfg.get("context_cache_ttl_seconds", 30))
        self._context_cache_max = int(self.rag_cfg.get("context_cache_max_entries", 32))
        # Hierarchical tag index. Built lazily; disabled if no broad_tags or no LLM.
        self._tag_index: TagIndex | None = None
        self._tag_index_built = False

    def _enabled_sources(self) -> list[str]:
        return list(self.rag_cfg.get("sources", ["notes", "sessions"]))

    def _top_k(self) -> int:
        return int(self.rag_cfg.get("top_k", 5))

    def _max_context_tokens(self) -> int:
        return int(self.rag_cfg.get("max_context_tokens", 1500))

    def _tag_index_enabled(self) -> bool:
        # Default to False when the key is missing so existing unit tests are
        # unaffected; real configs merge DEFAULT_CONFIG which sets it to True.
        return bool(self.rag_cfg.get("tag_index_enabled", False))

    def _build_tag_index(self) -> TagIndex | None:
        if self._tag_index_built:
            return self._tag_index
        self._tag_index_built = True
        if TagIndex is None or not self._tag_index_enabled():
            return None
        broad_tags = self.rag_cfg.get("broad_tags", [])
        if not broad_tags:
            return None
        db_path = self.rag_cfg.get("tag_index_db", "notes/tags.db")
        db_path = Path(db_path)
        if not db_path.is_absolute():
            db_path = PROJECT_DIR / db_path
        self._tag_index = TagIndex(
            notes_dir=NOTES_DIR,
            db_path=db_path,
            broad_tags=broad_tags,
            llm_fn=self._llm_fn or _default_tag_llm_fn,
        )
        return self._tag_index

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
        self._context_cache.clear()
        if self._tag_index is not None:
            self._tag_index.invalidate()
            self._tag_index = None
        self._tag_index_built = False

    @staticmethod
    def _context_cache_key(query: str) -> str:
        """Stable key for repeated/rephrased queries."""
        return " ".join(sorted(_normalize(query)))

    # A term carried by more than this share of the corpus cannot be what
    # makes one note the right answer. Measured on this project's notes:
    # topical words sit at or below 11% ("wifi" 5%, "coffee" 11%, "bicycle"
    # 5%) while the words that were actually deciding the ranking sit far
    # above it ("skill" 53%, "the" 68%, "for" 47%). A third is the gap.
    _MAX_DISCRIMINATIVE_SHARE = 1 / 3
    # Corpus statistics alone cannot catch these. On a small, technical set of
    # notes "how" and "do" are genuinely rare — most notes are written as
    # numbered steps and simply never use them — so the share test called them
    # discriminative and admitted the Device Awareness note for "how do I make
    # coffee". A stopword is never the reason a note is the right answer, at
    # any frequency, so it is excluded on principle rather than by counting.
    _NEVER_DISCRIMINATIVE = frozenset("""
        how do does did can could should would will shall may might must
        what when where which who whom whose why is are was were be been being
        have has had am the and or but if then than that this these those
        for to of in on at by with from as into about over under
        my me you your yours it its they them their we us our
        one two some any all no not so just now new get got make made made
        need want like use used using please tell show give take put
        making getting doing having save saved saving
        thing things stuff way ways time times step steps skill skills
        up down out off back away here there
        ever never always again still yet already
        very really quite much more most less least too also even
        anything something nothing everything anyone someone everyone
        """.split())
    # Particles and degree adverbs are in that list for a specific reason. The
    # preposition half was there from the start; the particles were not, and
    # "up" is the whole story of one live failure: an abstract question —
    # "weigh UP ... whether a small model can understand anything" — retrieved
    # the Browser Driver note, whose triggers say "pull up / bring up / throw
    # up / chuck up". It was the only query term the note contained, so it both
    # made the note eligible and supplied 11.8 of its 22.9 score. The headmaster
    # delegated to the browser worker and tried to open google.com.
    #
    # IDF cannot catch this and in fact makes it worse: "up" is rare ACROSS
    # notes, so it earns a high IDF, while being frequent WITHIN one note.
    # Rarity is not aboutness.

    @classmethod
    def _is_discriminative(cls, term: str, appears: int, total: int) -> bool:
        if term in cls._NEVER_DISCRIMINATIVE:
            return False
        return 0 < appears <= total * cls._MAX_DISCRIMINATIVE_SHARE
    # Below this many notes the share of a term is too noisy to act on — with
    # four notes, one is 25% and two is 50%, so the cutoff would be deciding
    # on nothing. Small corpora keep the old behaviour.
    _MIN_NOTES_FOR_FLOOR = 5

    @classmethod
    def _discriminative_terms(cls, query_terms: list[str],
                              normalized_docs) -> set[str] | None:
        """Query terms rare enough to justify returning a note at all.

        IDF ranks well and cannot say "none of these". When a query's topical
        words appear in no note — asking about bread with no bread note — the
        whole ranking falls to whatever common words are left, and the top
        three come back as confidently as a real match. Measured: "save a
        skill for making bread" returned Device Awareness, Coffee Making and
        Quick Task Helper, decided by "skill", "for" and "save".

        That is not a relevance problem. A retrieved note is pasted into the
        model's context, skill notes are procedures, and the model performs
        them: a bicycle question pulled the Browser Driver note and produced
        real clicks on google.com. An empty result is strictly better than a
        confident wrong one.

        Returns None when the corpus is too small to judge.
        """
        docs = [set(d) for d in normalized_docs]
        total = len(docs)
        if total < cls._MIN_NOTES_FOR_FLOOR:
            return None
        return {t for t in set(query_terms)
                if cls._is_discriminative(
                    t, sum(1 for d in docs if t in d), total)}

    @staticmethod
    def _idf(query_terms: list[str], documents) -> dict[str, float]:
        """How much each query term distinguishes one document from another.

        Without this every term counts the same, so "the" is worth as much as
        "power". Measured: routing "what is 3 to the power of 27, exactly"
        scored the Researcher note 27 — of which "the" contributed 12 and no
        content word contributed anything — over the Device note's 19, which
        was the only one to match "power" and "exactly" at all. The router was
        deciding on stopwords.

        A term in every document scores 0; a term in one of many scores high.
        """
        docs = [_normalize(d) for d in documents]
        total = len(docs) or 1
        idf = {}
        for term in set(query_terms):
            appears = sum(1 for d in docs if term in d)
            # +1 inside the log keeps a term present everywhere at exactly 0
            # rather than negative, so it is ignored rather than penalised.
            idf[term] = math.log(1 + (total - appears + 0.5) / (appears + 0.5))
        return idf

    def _score(self, query_terms: list[str], text: str,
               idf: dict[str, float] | None = None,
               restrict: set[str] | None = None) -> float:
        """Rank a note against the query, on topical terms only.

        The eligibility gate and the ranking used to disagree with each other:
        _discriminative_terms decided WHETHER to answer using a filtered set of
        terms, and then this scored every query term, stopwords included. So a
        note admitted on a real content word could still be ranked above its
        rivals by "from" and "can" — and one admitted on a junk term was ranked
        by that same junk term. Scoring the terms the gate already rejected is
        the bug the gate exists to prevent, arriving one step later.

        `restrict` is that gate's own set, so the two now agree by construction.
        """
        terms = _normalize(text)
        if not terms:
            return 0.0
        counts = Counter(terms)
        scoring = [t for t in query_terms if t not in self._NEVER_DISCRIMINATIVE]
        if restrict is not None:
            scoring = [t for t in scoring if t in restrict]
        if not scoring:
            return 0.0
        if idf is None:
            score = sum(counts[t] for t in scoring)
        else:
            score = sum(counts[t] * idf.get(t, 0.0) for t in scoring)
        # Normalize by document length so long docs do not always win.
        return score / (len(terms) ** 0.5 + 1)

    def search_notes(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if "notes" not in self._enabled_sources():
            return []
        query_terms = _normalize(query)
        if not query_terms:
            return []

        # Try hierarchical tag index first. If it returns nothing or the index
        # is unavailable, fall back to the existing keyword overlap search.
        tag_index = self._build_tag_index()
        if tag_index is not None:
            try:
                tag_results = tag_index.search(query, top_k=top_k or self._top_k())
                if tag_results:
                    return [
                        {
                            "source": "note",
                            "title": r.path.name,
                            "text": f"{r.heading_path}\n{r.text}",
                            "score": r.score,
                            "path": str(r.path),
                            "heading_path": r.heading_path,
                            "broad_tag": r.broad_tag,
                            "meta_tags": r.meta_tags,
                            "description": r.description,
                        }
                        for r in tag_results
                    ]
            except Exception:
                pass

        notes = self._load_notes()
        normalized = {name: _normalize(text) for name, text in notes.items()}
        idf = self._idf(query_terms, notes.values())
        # Nothing rare enough to justify an answer means the honest answer is
        # no notes, not the three least-bad ones.
        discriminative = self._discriminative_terms(query_terms, normalized.values())
        if discriminative is not None and not discriminative:
            return []
        # One rare word in common is coincidence; two is aboutness. Requiring a
        # single overlap admitted a note on the strength of one incidental term
        # and then pasted its procedure into the model's context: "weigh up ...
        # whether a small model can understand anything" matched Quick Task
        # Helper on "small" alone, and before "up" was stoplisted it matched
        # Browser Driver on "up" alone and opened a browser.
        #
        # Measured over the routing cases: every correct match shares 2-6
        # discriminative terms with its note, every wrong one shares exactly 1.
        # The floor is a count, not a score threshold, so it does not need
        # retuning as the corpus grows.
        #
        # min() so a genuinely one-word query ("colemak") can still match on
        # its single term — the rule is "use what the query gives you", not
        # "demand two words the user never typed".
        required = min(2, len(discriminative)) if discriminative is not None else 0
        scored = []
        for name, text in notes.items():
            if discriminative is not None and len(
                    discriminative & set(normalized[name])) < required:
                continue
            s = self._score(query_terms, text, idf, restrict=discriminative)
            if s > 0:
                scored.append({
                    "source": "note",
                    "title": name,
                    "text": text,
                    "score": s,
                    "path": str(NOTES_DIR / name),
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: (top_k or self._top_k())]

    def search_sessions(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if "sessions" not in self._enabled_sources() or self.session_store is None:
            return []
        # Imported here, not at module scope: symbio.app's package __init__
        # pulls in chat, which imports this module back.
        from symbio.app import learn, tooling

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
            # Catches both the real user-role scaffold and an assistant reply
            # that impersonated it — the latter is already logged as a normal
            # turn, so without this it feeds straight back in as context and
            # teaches the model the habit all over again.
            if learn.looks_like_observation_echo(content):
                continue
            if learn.looks_degenerate(content):
                continue
            # Any tool-call syntax, not just <tool_call>: a reply logged in
            # its raw form still carries legacy short tags (<click>, <browse>,
            # <search>), and feeding those back as retrieved prose is exactly
            # what teaches the model to emit stray tags mid-answer.
            if tooling.contains_tool_tag(content):
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
                # Recent samples are usually more relevant; read from the end
                # up to a byte cap to avoid scanning multi-megabyte files.
                f.seek(0, 2)
                size = f.tell()
                start = max(0, size - _TRAIN_SCAN_MAX_BYTES)
                f.seek(start)
                if start > 0:
                    f.readline()  # discard the possibly partial line
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

        key = self._context_cache_key(query)
        now = time.monotonic()
        cached = self._context_cache.get(key)
        if cached is not None:
            text, expires = cached
            if now < expires:
                return text
            self._context_cache.pop(key, None)

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

        text = "\n\n".join(lines)
        # Treat everything retrieved from notes/sessions/training data as
        # untrusted data. Scan it for hidden commands and wrap it with explicit
        # markers so the model knows it must not follow instructions found here.
        scan = safety.scan_for_injection(text)
        text = safety.wrap_untrusted("retrieved context", text, scan)
        # Keep the cache small and fresh; stale entries expire via TTL checks.
        if len(self._context_cache) >= self._context_cache_max:
            oldest = min(self._context_cache, key=lambda k: self._context_cache[k][1])
            self._context_cache.pop(oldest, None)
        self._context_cache[key] = (text, now + self._context_cache_ttl)
        return text
