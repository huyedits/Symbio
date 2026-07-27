"""Markdown notes plus the two small always-in-context curated stores:
agent_memory.md (durable facts/conventions the agent learned) and
user_profile.md (who the user is). Char caps force consolidation instead
of hoarding."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from symbio import constants


def save_note(title: str, body: str) -> Path:
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)
    safe = safe.strip().replace(" ", "_")[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = constants.NOTES_DIR / f"{ts}_{safe}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def ensure_seed_notes(config: dict[str, Any]):
    """If notes/ is empty, seed the two identity facts as markdown notes."""
    if any(constants.NOTES_DIR.glob("*.md")):
        return
    save_note("My Identity", f"I am {config['assistant_name']}, a helpful personal AI assistant.")
    save_note("User Identity", f"My user's name is {config['user_name']}.")


def save_skill(name: str, steps: str) -> Path:
    """Persist a reusable multi-step skill as a 'Skill:' note; RAG retrieves
    it when a similar task appears, and digest bakes it into the weights."""
    return save_note(f"Skill: {name}", steps)


def list_skills() -> list[tuple[str, Path]]:
    """All saved skills as (title, path), detected by their '# Skill:' heading."""
    skills = []
    for f in sorted(constants.NOTES_DIR.glob("*.md")):
        try:
            first_line = f.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            continue
        if first_line.lower().startswith("# skill:"):
            skills.append((first_line[2:].strip(), f))
    return skills


def _store_path(store: str) -> Path:
    return constants.PROFILE_FILE if store == "profile" else constants.MEMORY_FILE


def _store_limit(store: str, config: dict[str, Any]) -> int:
    key = "profile_char_limit" if store == "profile" else "memory_char_limit"
    return int(config["memory"][key])


def save_memory(store: str, content: str, config: dict[str, Any], replace: bool = False) -> str:
    """Append (or replace) an entry in a curated memory store; nag when full."""
    content = content.strip()
    if not content:
        return "Empty memory content."
    path = _store_path(store)
    if replace or not path.exists():
        path.write_text(content + "\n", encoding="utf-8")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
    size = len(path.read_text(encoding="utf-8"))
    limit = _store_limit(store, config)
    name = path.name
    if size > limit:
        return (
            f"Saved to {name}, but it is now {size}/{limit} chars — over the limit. "
            f"Rewrite it smaller with <{'profile' if store == 'profile' else 'memory'} "
            f"replace='all'>...</...> keeping only what matters."
        )
    return f"Saved to {name} ({size}/{limit} chars)."


def compact_store(
    store: str,
    config: dict[str, Any],
    summarize_fn: Callable[[str], str] | None = None,
) -> tuple[str, Path | None]:
    """Compress a curated memory store when it exceeds its char limit.

    If `summarize_fn` is provided, the full store text is passed to it and the
    returned summary replaces the live store. The original full text is archived
    to notes/archive/ so nothing is lost. If no summarizer is supplied, the
    store is truncated to its most recent entries that fit under the limit.

    Returns (status_message, archive_path_or_None).
    """
    path = _store_path(store)
    if not path.exists():
        return f"No {store} store to compact.", None

    full_text = path.read_text(encoding="utf-8")
    size = len(full_text)
    limit = _store_limit(store, config)
    if size <= limit:
        return f"{store} store is within limit ({size}/{limit} chars).", None

    name = path.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{timestamp}_{name}.archive.md"
    archive_path = constants.NOTES_ARCHIVE_DIR / archive_name
    archive_path.write_text(f"# Archived {name} (before compaction)\n\n{full_text}\n", encoding="utf-8")

    if summarize_fn is not None:
        try:
            prompt = (
                f"Compress the following personal memory store into a shorter version "
                f"under {limit} characters. Keep facts, preferences, and recurring tasks; "
                f"drop redundant wording. Do not add commentary.\n\n{full_text}"
            )
            compacted = summarize_fn(prompt).strip()
        except Exception as exc:
            compacted = ""
            archive_path.unlink(missing_ok=True)
            return f"Could not summarize {store} store: {exc}", None
    else:
        # No summarizer: keep the tail of the file under the limit.
        compacted = full_text
        while len(compacted) > limit and "\n" in compacted:
            compacted = compacted.split("\n", 1)[1]
        compacted = compacted.strip()

    if not compacted:
        compacted = f"[Compacted on {datetime.now():%Y-%m-%d}. Full version archived in {archive_name}.]"

    path.write_text(compacted + "\n", encoding="utf-8")
    new_size = len(path.read_text(encoding="utf-8"))
    return (
        f"Compacted {store} store from {size} to {new_size} chars. "
        f"Full version archived to {archive_path.name}.",
        archive_path,
    )


def curated_memory_block(config: dict[str, Any]) -> str:
    """The always-on memory injected into the system prompt each turn."""
    if not config["memory"]["enabled"]:
        return ""
    parts = []
    if constants.MEMORY_FILE.exists():
        text = constants.MEMORY_FILE.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"[Your saved memory]\n{text}")
    if constants.PROFILE_FILE.exists():
        text = constants.PROFILE_FILE.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"[About {config['user_name']}]\n{text}")
    return ("\n\n" + "\n\n".join(parts)) if parts else ""
