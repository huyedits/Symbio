"""Markdown notes plus the two small always-in-context curated stores:
agent_memory.md (durable facts/conventions the agent learned) and
user_profile.md (who the user is). Char caps force consolidation instead
of hoarding."""

from datetime import datetime
from pathlib import Path
from typing import Any

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
