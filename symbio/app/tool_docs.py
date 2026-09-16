"""One markdown file per tool, and the compact index that replaces the catalog.

Two problems, one file format.

The first is that the tool catalog was a 2,600-token JSON blob in every system
prompt — every schema of every tool, on every turn, whether the request was
about files or the weather. On a 14B with a KV budget that is real money: it is
prompt the model re-reads instead of reasoning, and prefill it pays for before
the first token.

The second is that adding a tool meant editing a Python list, a groups dict, a
families dict and a dispatch branch. Three of those four are description, not
behaviour.

So a tool's description lives in `tools/<name>.md`:

    ---
    name: browser_click
    family: browser
    group: browser
    ---

    Click an element in the open browser, identified by its visible text.

    ```json
    {"type": "object",
     "properties": {"target": {"type": "string"}},
     "required": ["target"]}
    ```

and the prompt carries only the index — the families, the names in each, one
line about what the family is for. When the model needs a tool's exact
arguments it calls `tool_docs`, which hands back these files. That is the
"range of tools for x, y and z" shape: mention the range, spend tokens on the
one being used.

The files are SEEDED from the in-code catalog on first run, the way prompt.md
is, so nothing is shipped as data that the code cannot regenerate — and an
install that deletes the directory gets the built-in catalog back rather than
an agent with no tools.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from symbio import constants

# Frontmatter is three keys and nothing clever. A tool file is written by this
# module or by a person; neither needs YAML.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)

# What each family is FOR, in the one line the index spends on it. The model
# reads this to decide which family to ask about, so it says what kind of job
# the family does rather than listing its tools again.
FAMILY_BLURBS: dict[str, str] = {
    "file": "read, edit and create files inside the project",
    "code": "run Python for exact computation, parsing and conversion",
    "shell": "run shell commands here or on a configured remote host",
    "web": "search the web and pull a page's text or raw HTML",
    "browser": "drive a real Chrome window: open, look, click, type, submit",
    "desktop": "drive the whole machine: click, type and press keys on screen",
    "memory": "notes, durable memory, saved skills and saved commands",
    "admin": "your own config, schedule, training, health and workers",
    "other": "tools added by MCP or by you",
    # Not listed in the prompt's index (the header names it directly), but
    # /tools shows every family and an unlabelled row reads like a bug.
    "core": "the tool that hands you the other tools' exact arguments",
}


_README = """Every *.md in this directory defines one tool.

    ---
    name: check_tide          # must match what the model calls
    family: web               # file code shell web browser desktop memory admin
    group: browser            # the on/off group in config.json tools.enabled_groups
    ---

    One paragraph. This is the description the model reads, so say when to
    use the tool, not just what it is.

    ```json
    {"type": "object",
     "properties": {"harbour": {"type": "string", "description": "..."}},
     "required": ["harbour"]}
    ```

Editing a file here changes what the model is told about that tool, from the
next turn. Deleting one puts the built-in description back (the files are
seeded from code, never the other way round).

Adding a NEW file makes the tool describable and advertised — it still needs a
branch in symbio/app/chat_tools.py `_dispatch_tool` to actually run. A file
with no `group:` lands in "core", which cannot be switched off.

The system prompt does not carry all of these. It carries the index — families,
names, one line each — and the model fetches the rest with `tool_docs`. Run
`/tools` to see the index, `/tools <family>` for the schemas.
"""


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split `text` into its frontmatter mapping and the body after it."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip()
    return meta, text[m.end():]


def parse_tool_file(path: Path) -> dict[str, Any] | None:
    """One tool's schema, read from its markdown file, or None if unreadable.

    Unreadable is deliberately not fatal: a half-finished file someone is
    editing should cost that one tool, not the session's whole toolset.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    meta, body = _parse_frontmatter(text)
    name = meta.get("name") or path.stem
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        return None
    block = _JSON_BLOCK_RE.search(body)
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    if block:
        try:
            loaded = json.loads(block.group(1))
            if isinstance(loaded, dict):
                parameters = loaded
        except json.JSONDecodeError:
            return None
        body = body[:block.start()] + body[block.end():]
    description = " ".join(body.split())
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "_family": meta.get("family", "other"),
        "_group": meta.get("group", ""),
        "_seeded": meta.get("seeded", ""),
    }


def schema_fingerprint(description: str, parameters: dict[str, Any]) -> str:
    """A short hash of what a tool file SAYS, ignoring how it is laid out.

    Written into the file when it is seeded, so a later sync can tell a file
    nobody has touched from one the user has edited. Without that, a
    description improved in code is invisible on every install that already
    seeded the old one -- the shipped guard that the running config never
    sees.
    """
    payload = json.dumps(
        {"d": " ".join((description or "").split()), "p": parameters or {}},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_tool_file(schema: dict[str, Any], family: str, group: str) -> str:
    """The markdown for one tool — the inverse of parse_tool_file."""
    return (
        "---\n"
        f"name: {schema['name']}\n"
        f"family: {family}\n"
        f"group: {group}\n"
        f"seeded: {schema_fingerprint(schema.get('description', ''), schema.get('parameters', {}))}\n"
        "---\n\n"
        f"{schema.get('description', '').strip()}\n\n"
        "```json\n"
        + json.dumps(schema.get("parameters", {"type": "object", "properties": {}}),
                     indent=2, ensure_ascii=False)
        + "\n```\n"
    )


def ensure_seeded(schemas: list[dict[str, Any]],
                  families: dict[str, str],
                  groups: dict[str, Any],
                  hermes_names: dict[str, str] | None = None) -> int:
    """Write a .md for every built-in tool that does not have one yet.

    Returns how many files were written. Existing files are never touched: the
    directory is the user's to edit, and re-seeding over an edit would be the
    same mistake as regenerating a customized prompt.md.
    """
    hermes_names = hermes_names or {}
    written = 0
    try:
        constants.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return 0
    # Deliberately not a .md: every *.md in here is read as a tool definition,
    # and a README that parsed as one would be a tool called "README".
    readme = constants.TOOLS_DIR / "README"
    if not readme.exists():
        try:
            readme.write_text(_README, encoding="utf-8")
        except Exception:
            pass
    for schema in schemas:
        path = constants.TOOLS_DIR / f"{schema['name']}.md"
        if path.exists():
            continue
        internal = hermes_names.get(schema["name"], schema["name"])
        group = groups.get(internal, "")
        if isinstance(group, tuple):
            group = group[0]
        try:
            path.write_text(
                render_tool_file(schema, families.get(internal, "other"), str(group)),
                encoding="utf-8")
            written += 1
        except Exception:
            continue
    return written


def load_tool_files() -> list[dict[str, Any]]:
    """Every tool defined on disk, in name order."""
    if not constants.TOOLS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(constants.TOOLS_DIR.glob("*.md")):
        parsed = parse_tool_file(path)
        if parsed is not None:
            out.append(parsed)
    return out


def index_block(schemas: list[dict[str, Any]],
                family_of,
                core: set[str] | None = None) -> str:
    """The compact catalog: families, their tool names, and the core schemas.

    `core` names the tools whose full schema stays inline. Those are the ones a
    turn reaches for without thinking — the shell, a search, a file read — and
    making them cost a lookup round would trade prompt tokens for latency on
    every ordinary request. Everything else is a name in a list until asked
    for, which is what makes the block small.
    """
    core = core or set()
    by_family: dict[str, list[str]] = {}
    for schema in schemas:
        by_family.setdefault(family_of(schema["name"]), []).append(schema["name"])

    lines = [
        "<tools>",
        "Your tools, by family. The names are exact. You do NOT have the "
        "arguments for most of them in front of you: before calling anything "
        "whose schema is not printed below, call "
        '{"name": "tool_docs", "arguments": {"family": "<family>"}} '
        "(or pass \"names\" instead, for specific tools) and use what it "
        "returns. Asking costs one round; guessing an argument name costs the "
        "whole attempt.",
    ]
    for family in ("file", "code", "shell", "web", "browser", "desktop",
                   "memory", "admin", "other"):
        names = sorted(by_family.get(family, []))
        if not names:
            continue
        blurb = FAMILY_BLURBS.get(family, "")
        # Names on their own line, never sharing one with prose. A blurb can
        # contain a comma or a colon; a parser — the model's or a test's —
        # should never have to tell those apart from the list.
        lines.append(f"  {family} — {blurb}")
        lines.append(f"    {', '.join(names)}")
    inline = [s for s in schemas if s["name"] in core]
    if inline:
        lines.append("Full schemas for the tools used most often:")
        lines.append(json.dumps(
            [{"name": s["name"], "description": s["description"],
              "parameters": s["parameters"]} for s in inline],
            ensure_ascii=False, separators=(",", ":")))
    lines.append("</tools>")
    return "\n".join(lines)


def docs_for(schemas: list[dict[str, Any]], family_of,
             family: str = "", names: str = "") -> str:
    """The full schemas a `tool_docs` call asked for, as the model sees them.

    Answers generously: an unknown family comes back as the list of real ones
    rather than an error, because a model that asked for "files" instead of
    "file" has told you exactly what it wants and should not lose a round to
    a spelling.
    """
    wanted: list[dict[str, Any]] = []
    asked_names = [n.strip() for n in re.split(r"[,\s]+", names or "") if n.strip()]
    if asked_names:
        by_name = {s["name"]: s for s in schemas}
        for n in asked_names:
            hit = by_name.get(n) or by_name.get(n.rstrip("s"))
            if hit is not None and hit not in wanted:
                wanted.append(hit)
    if family:
        key = family.strip().lower().rstrip("s")
        aliases = {"files": "file", "filesystem": "file", "python": "code",
                   "terminal": "shell", "command": "shell", "search": "web",
                   "internet": "web", "note": "memory", "notes": "memory",
                   "config": "admin", "system": "admin", "screen": "desktop"}
        key = aliases.get(family.strip().lower(), key)
        for s in schemas:
            if family_of(s["name"]) == key and s not in wanted:
                wanted.append(s)
    if not wanted:
        known = sorted({family_of(s["name"]) for s in schemas})
        return (
            f"No tool matched family={family!r} names={names!r}. "
            f"The families are: {', '.join(known)}. "
            "Call tool_docs again with one of those."
        )
    return json.dumps(
        [{"name": s["name"], "description": s["description"],
          "parameters": s["parameters"]} for s in wanted],
        ensure_ascii=False, separators=(",", ":"))


def refresh(schemas: list[dict[str, Any]],
            families: dict[str, str],
            groups: dict[str, Any],
            hermes_names: dict[str, str] | None = None,
            names: list[str] | None = None,
            force: bool = False) -> tuple[list[str], list[str]]:
    """Rewrite tool files from the built-ins. Returns (rewritten, kept).

    A file is rewritten when its contents still fingerprint as the version
    that was seeded -- nobody has edited it -- and the built-in has since
    changed. Anything the user has touched is KEPT and named in the second
    list, because this directory is theirs; `force` (or naming the file
    explicitly with `names`) is the way to overwrite one deliberately.

    Files seeded before fingerprints existed carry no marker. They cannot be
    told apart from an edit, so they are kept and reported rather than
    silently replaced.
    """
    hermes_names = hermes_names or {}
    wanted = set(names or [])
    rewritten: list[str] = []
    kept: list[str] = []
    for schema in schemas:
        name = schema["name"]
        if wanted and name not in wanted:
            continue
        path = constants.TOOLS_DIR / f"{name}.md"
        if not path.exists():
            continue
        parsed = parse_tool_file(path)
        if parsed is None:
            kept.append(name)
            continue
        current = schema_fingerprint(schema.get("description", ""),
                                     schema.get("parameters", {}))
        on_disk = schema_fingerprint(parsed["description"], parsed["parameters"])
        if on_disk == current:
            continue  # already says what the code says
        unedited = bool(parsed.get("_seeded")) and parsed["_seeded"] == on_disk
        if not (unedited or force or name in wanted):
            kept.append(name)
            continue
        internal = hermes_names.get(name, name)
        group = groups.get(internal, "")
        if isinstance(group, tuple):
            group = group[0]
        try:
            path.write_text(
                render_tool_file(schema, families.get(internal, "other"),
                                 str(group)),
                encoding="utf-8")
            rewritten.append(name)
        except Exception:
            kept.append(name)
    return rewritten, kept
