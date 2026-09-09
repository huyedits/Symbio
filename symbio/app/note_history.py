"""Version history for markdown notes.

Notes are created and deleted (there is no in-place edit in the live path);
before a note is deleted or overwritten, its current content is snapshotted into
notes/.history/<stem>/ so it can be viewed and restored later. The directory is
a subdirectory of notes/, so the non-recursive ``glob("*.md")`` scans in rag.py
and memory.py never pick the snapshots up as live notes.

Paths are resolved through ``constants.NOTES_DIR`` at call time (never cached at
import time) so the test suite's path redirection in conftest.py is honoured.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from symbio import constants


def _history_root() -> Path:
    return constants.NOTES_DIR / ".history"


def _history_dir_for(path: Path) -> Path:
    return _history_root() / path.stem


def snapshot(path: Path) -> Path | None:
    """Copy a note's current content into its history dir.

    Returns the snapshot path, or None if the note does not exist.
    """
    if not path.exists():
        return None
    dest_dir = _history_dir_for(path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = dest_dir / f"{ts}.md"
    dest.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return dest


def versions(path: Path) -> list[Path]:
    """List snapshots for a note, oldest first."""
    dest_dir = _history_dir_for(path)
    if not dest_dir.exists():
        return []
    return sorted(dest_dir.glob("*.md"))


def restore(path: Path, version: Path) -> Path:
    """Write a snapshot back to the note, snapshotting the current content first."""
    snapshot(path)
    path.write_text(version.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return path


def all_history_dirs() -> list[Path]:
    """List every note that has at least one snapshot, oldest first."""
    root = _history_root()
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())


def _normalize(s: str) -> list[str]:
    return [w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 1]


def matching_history_dirs(name: str) -> list[Path]:
    """History dirs whose stem matches `name` (a title fragment or timestamp)."""
    q = name.strip().lower()
    if not q:
        return []
    q_terms = _normalize(q)
    dirs = all_history_dirs()
    if not q_terms:
        return [d for d in dirs if q in d.name.lower()]
    return [d for d in dirs if all(t in _normalize(d.name) for t in q_terms)]
