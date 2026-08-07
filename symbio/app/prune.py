"""Self-pruning for the stores that feed retrieval.

Everything the agent writes down is also something it later reads back:
notes and past sessions are RAG sources, so a bad write is not a one-off
mistake — it becomes context on every future turn that happens to match it.
A few junk shapes recur often enough to be worth removing automatically:

  * Degenerate notes. A misparsed `<note>` tag saves the leftover fragment
    of the user's sentence under a one-letter title ("# T" / "well note
    that"). It carries no information but still competes for retrieval slots.
  * Mis-keyed research notes. A subjectless command ("CHECK ONLINE") becomes
    the note's Question, so the note is filed under words that have nothing
    to do with its answer and surfaces for unrelated queries.
  * Duplicate notes. The same body saved repeatedly outranks everything else
    simply by having more copies.
  * Tagged twins in the session log. The reply is logged twice — once as the
    visible text, once raw with its tool tags still in it — so retrieval can
    feed the model its own tag syntax back as if it were prose.

Pruning is conservative by construction: notes are moved to notes/archive/
(a subdirectory, so the non-recursive globs that feed RAG skip it) rather
than deleted, and the classifiers below refuse to touch anything load-bearing
— skills, identity notes, or the session that is currently being written.
"""

import json
import re
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.app import tooling

# Notes that other machinery depends on and that must never be pruned,
# however sparse they look: skills back a worker LoRA adapter, and the two
# identity notes are re-seeded (and re-read) as the agent's sense of self.
_PROTECTED_TITLES = {"my identity", "user identity"}
_SKILL_PREFIX = "skill:"

# A note title this short carries no retrievable subject — it is what a
# misparsed tag leaves behind, never something a deliberate save produces.
_MIN_TITLE_CHARS = 3
_MIN_BODY_CHARS = 3

# Bare commands that a "Learned:" note sometimes records as its Question.
# They describe an action rather than a subject, so the note ends up filed
# under words unrelated to the answer it actually holds.
_SUBJECTLESS_QUESTIONS = {
    "check online", "check the web", "check web", "search", "search online",
    "search the web", "look it up", "look up", "lookup", "google it",
    "find out", "find online", "check", "check it", "check that",
}

_TITLE_RE = re.compile(r"^#\s*(.+?)\s*$", re.MULTILINE)
_QUESTION_RE = re.compile(r"^\*\*Question:\*\*\s*(.+?)\s*$", re.MULTILINE)


def note_title(text: str) -> str:
    """The note's `# ` heading, or "" when it has none."""
    m = _TITLE_RE.search(text)
    return m.group(1).strip() if m else ""


def note_body(text: str) -> str:
    """Everything after the heading line, stripped."""
    m = _TITLE_RE.search(text)
    if not m:
        return text.strip()
    return text[m.end():].strip()


def is_protected_note(text: str) -> bool:
    """True for notes that must survive pruning regardless of their shape."""
    title = note_title(text).strip().lower()
    return title in _PROTECTED_TITLES or title.startswith(_SKILL_PREFIX)


def classify_note(text: str) -> str | None:
    """Return a junk reason for this note's content, or None to keep it.

    Duplicate detection needs to compare notes against each other, so it is
    handled by prune_notes() rather than here.
    """
    if is_protected_note(text):
        return None
    title = note_title(text)
    body = note_body(text)
    if len(body) < _MIN_BODY_CHARS:
        return "empty body"
    if len(title) < _MIN_TITLE_CHARS:
        # "# T" with a sentence fragment under it — a misparsed <note> tag.
        return "degenerate title"
    m = _QUESTION_RE.search(text)
    if m and m.group(1).strip().lower().rstrip("?.!") in _SUBJECTLESS_QUESTIONS:
        return "subjectless question"
    return None


def _archive_note(path: Path) -> None:
    constants.NOTES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = constants.NOTES_ARCHIVE_DIR / path.name
    # Never clobber an existing archived note of the same name.
    n = 1
    while target.exists():
        target = constants.NOTES_ARCHIVE_DIR / f"{path.stem}.{n}{path.suffix}"
        n += 1
    path.rename(target)


def prune_notes(dry_run: bool = False) -> list[dict[str, str]]:
    """Archive junk notes. Returns one record per pruned note.

    Only the top level of notes/ is considered: mistakes/ and archive/ are
    subdirectories, so the same non-recursive glob RAG uses skips them here.
    """
    if not constants.NOTES_DIR.exists():
        return []
    pruned: list[dict[str, str]] = []
    seen_bodies: dict[str, str] = {}
    # Sorted by filename, which starts with a timestamp — so when the same
    # body was saved twice, the first one seen is the original that stays.
    for path in sorted(constants.NOTES_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        reason = classify_note(text)
        if reason is None and not is_protected_note(text):
            key = " ".join(note_body(text).lower().split())
            if key in seen_bodies:
                reason = f"duplicate of {seen_bodies[key]}"
            else:
                seen_bodies[key] = path.name
        if reason is None:
            continue
        pruned.append({"name": path.name, "reason": reason})
        if not dry_run:
            _archive_note(path)
    return pruned


# ---- session logs ----

def _stripped(content: str) -> str:
    """The visible text of a logged reply, with any tool tags removed."""
    try:
        return " ".join(tooling.strip_tool_tags(content).split())
    except Exception:
        return " ".join(content.split())


def prune_entries(entries: list[dict[str, Any]],
                  max_copies: int = 2) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop junk entries from one session's log.

    Two shapes are removed:

      * A tagged twin — an assistant entry whose tool tags strip down to the
        exact text of another assistant entry logged in the same second. That
        is the same reply written twice (visible form and raw form); the raw
        one is dropped so retrieval can't feed tag syntax back as prose.
      * Repeats beyond `max_copies` of an identical (role, content) pair, so
        a stuck retry loop can't bury everything else in the store.

    Returns (kept_entries, reasons_for_dropped).
    """
    # Text of every assistant entry that carries no tags, keyed by timestamp:
    # a tagged entry is a twin only if its stripped form matches one of these.
    plain_by_ts: dict[str, set[str]] = {}
    for e in entries:
        if e.get("role") != "assistant":
            continue
        content = e.get("content", "")
        if content == tooling.strip_tool_tags(content):
            plain_by_ts.setdefault(e.get("timestamp", ""), set()).add(
                " ".join(content.split()))

    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    counts: dict[tuple[str, str], int] = {}
    for e in entries:
        role = e.get("role", "")
        content = e.get("content", "")
        if role == "assistant" and content != tooling.strip_tool_tags(content):
            if _stripped(content) in plain_by_ts.get(e.get("timestamp", ""), set()):
                dropped.append("tagged twin")
                continue
        key = (role, content)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > max_copies:
            dropped.append("repeat")
            continue
        kept.append(e)
    return kept, dropped


def prune_sessions(dry_run: bool = False, max_copies: int = 2,
                   exclude: str | None = None) -> list[dict[str, Any]]:
    """Prune every session log except the one currently being written.

    `exclude` is the live session id — rewriting the file the running session
    still has open would race its appends and lose the current turn.
    """
    if not constants.SESSIONS_DIR.exists():
        return []
    report: list[dict[str, Any]] = []
    for path in sorted(constants.SESSIONS_DIR.glob("*.jsonl")):
        if exclude and path.stem == exclude:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        entries: list[dict[str, Any]] = []
        malformed = 0
        for line in raw:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
        kept, dropped = prune_entries(entries, max_copies=max_copies)
        if not dropped and not malformed:
            continue
        report.append({
            "name": path.name,
            "dropped": len(dropped) + malformed,
            "kept": len(kept),
        })
        if dry_run:
            continue
        # Write via a temp file in the same directory, then replace, so an
        # interrupted prune can never leave a half-written session log.
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in kept:
                json.dump(e, f)
                f.write("\n")
        tmp.replace(path)
    return report


def prune_all(config: dict[str, Any] | None = None, dry_run: bool = False,
              exclude_session: str | None = None) -> dict[str, Any]:
    """Prune notes and session logs. Returns a report for display/logging."""
    cfg = (config or {}).get("prune", {})
    max_copies = int(cfg.get("session_max_copies", 2))
    notes = prune_notes(dry_run=dry_run) if cfg.get("notes", True) else []
    sessions = (
        prune_sessions(dry_run=dry_run, max_copies=max_copies,
                       exclude=exclude_session)
        if cfg.get("sessions", True) else []
    )
    return {
        "notes": notes,
        "sessions": sessions,
        "total": len(notes) + sum(s["dropped"] for s in sessions),
    }
