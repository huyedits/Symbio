"""Markdown notes plus the two small always-in-context curated stores:
agent_memory.md (durable facts/conventions the agent learned) and
user_profile.md (who the user is). Char caps force consolidation instead
of hoarding."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from symbio import constants, safety
from symbio.app import skills


def save_note(title: str, body: str) -> Path:
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)
    safe = safe.strip().replace(" ", "_")[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = constants.NOTES_DIR / f"{ts}_{safe}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    skills.record_note_usage(path)
    return path


def ensure_seed_notes(config: dict[str, Any]):
    """If notes/ is empty, seed the two identity facts as markdown notes."""
    if any(constants.NOTES_DIR.glob("*.md")):
        return
    save_note("My Identity", f"I am {config['assistant_name']}, a helpful personal AI assistant.")
    save_note("User Identity", f"My user's name is {config['user_name']}.")


# Facts about the agent's own tools that it cannot work out from the
# conversation and keeps getting wrong. These live as notes rather than in the
# system prompt so RAG surfaces them exactly when the topic comes up, instead
# of spending prefix tokens on every unrelated turn.
_CAPABILITY_NOTES: dict[str, str] = {
    "Browser Control": (
        "I have my own controllable Chrome window, separate from the user's browser.\n\n"
        "- `<browse>https://...</browse>` opens a page in it. This is the ONLY way to\n"
        "  open a page I can then act on.\n"
        "- Once a page is open it stays open across turns. I do not need to reopen it\n"
        "  to keep working on it.\n"
        "- On the open page I can use `<click>visible text</click>`,\n"
        "  `<type>words</type>`, `<press>Enter</press>`, `<scroll />`, and\n"
        "  `<browser_close />`.\n\n"
        "Running a shell command like `open -a 'Google Chrome' <url>` opens the page in\n"
        "the user's own browser, which I cannot see or click. If I have done that and\n"
        "the user asks me to click something, I must open the page with `<browse>`\n"
        "first, then click.\n\n"
        "If a browser action fails with \"Browser is not open\", the fix is to\n"
        "`<browse>` the URL and then retry the same action — not to explain the error."
    ),
}


def ensure_capability_notes():
    """Create any missing capability note, matched by title.

    Unlike ensure_seed_notes this runs even when notes/ already has content:
    an existing user would otherwise never receive a note added in a later
    version. Idempotent, so it is safe on every boot.
    """
    existing: set[str] = set()
    for path in constants.NOTES_DIR.glob("*.md"):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").lstrip().splitlines()
        except OSError:
            continue
        if lines and lines[0].startswith("# "):
            existing.add(lines[0][2:].strip())
    created = []
    for title, body in _CAPABILITY_NOTES.items():
        if title not in existing:
            save_note(title, body)
            created.append(title)
    return created


def save_skill(
    name: str,
    steps: str,
    config: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
    auto_train_adapter: bool = True,
) -> Path | dict[str, Any]:
    """Persist a reusable multi-step skill as a 'Skill:' note.

    If `config` and `tokenizer` are provided, also create a dedicated worker
    LoRA adapter for the skill and return a result dict. Otherwise just save
    the note and return its path (legacy behavior).
    """
    path = save_note(f"Skill: {name}", steps)
    if config is not None and tokenizer is not None:
        from symbio.app import skills

        result = skills.save_skill_adapter(
            name, steps, config, tokenizer, auto_train=auto_train_adapter
        )
        result["note_path"] = str(path)
        return result
    return path


def list_skills() -> list[tuple[str, Path]]:
    """All saved skills as (title, path), detected by their '# Skill:' heading."""
    # Not `skills`: this module imports symbio.app.skills at the top, and a
    # local of that name makes it unresolvable for the rest of the function.
    # Harmless here only because nothing below touches the module — which is
    # the same accident that made /skill-adapters unreachable in chat.py.
    found = []
    for f in sorted(constants.NOTES_DIR.glob("*.md")):
        try:
            first_line = f.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            continue
        if first_line.lower().startswith("# skill:"):
            found.append((first_line[2:].strip(), f))
    return found


def _store_path(store: str) -> Path:
    return constants.PROFILE_FILE if store == "profile" else constants.MEMORY_FILE


def _store_limit(store: str, config: dict[str, Any]) -> int:
    key = "profile_char_limit" if store == "profile" else "memory_char_limit"
    return int(config["memory"][key])


def save_memory(store: str, content: str, config: dict[str, Any], replace: bool = False) -> str:
    """Append (or replace) an entry in a curated memory store; nag when full.

    Refuses to save content that looks like a prompt injection so the
    always-in-context store cannot be used to override the system prompt.
    """
    content = content.strip()
    if not content:
        return "Empty memory content."
    scan = safety.scan_for_injection(content, config)
    if scan["risk_score"] >= 2:
        safety.log_security_event("memory_injection_refused", {
            "store": store,
            "flags": scan["flags"],
            "snippet": scan["snippet"],
        })
        return (
            f"Refused to save to {store}: content looks like a prompt injection "
            f"({', '.join(scan['flags'])}). Only save facts that were actually said or observed."
        )
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
    """The always-on memory block, marked as untrusted data.

    It is prepended to a user-role message, never the system prompt, so it
    cannot override the default instructions.
    """
    if not config["memory"]["enabled"]:
        return ""
    parts = []
    if constants.MEMORY_FILE.exists():
        text = constants.MEMORY_FILE.read_text(encoding="utf-8").strip()
        if text:
            parts.append(safety.wrap_untrusted("saved memory", text, safety.scan_for_injection(text, config)))
    if constants.PROFILE_FILE.exists():
        text = constants.PROFILE_FILE.read_text(encoding="utf-8").strip()
        if text:
            parts.append(safety.wrap_untrusted(f"about {config['user_name']}", text, safety.scan_for_injection(text, config)))
    return ("\n\n" + "\n\n".join(parts)) if parts else ""
