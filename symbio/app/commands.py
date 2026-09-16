"""User-defined slash commands: one markdown file per command.

A command is a prompt the user has decided is worth a name. `commands/standup.md`
makes `/standup` a thing they can type, and typing it sends that file's body as
their next message — with `$ARGUMENTS` replaced by whatever they typed after the
name, and `$1`, `$2`, … by the individual words.

Same shape as the tool files next door (`tools/*.md`) and prompt.md: the thing
the model reads is a file the user can open, edit and delete, not a string
buried in Python.

    ---
    description: Summarize what I did yesterday and what is blocked
    argument-hint: [project]
    author: user
    ---

    Read my notes from the last two days$ARGUMENTS, then give me three lines:
    what moved, what is blocked, what I should start with today.

The assistant can write one too, through the `save_command` tool, which is why
`author:` is recorded. A command the model wrote is still only ever run when
the user types its name — the file is a shortcut, not a trigger — but the
listing says who wrote it so nobody is surprised by a command they never
created.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from symbio import constants

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# A command name is what the user types after the slash. Kept to the shape a
# shell-ish reader expects, so a name can never collide with a path, a flag or
# another command's arguments.
_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    body: str
    argument_hint: str = ""
    author: str = "user"
    path: Path | None = None

    @property
    def usage(self) -> str:
        return f"/{self.name}" + (f" {self.argument_hint}" if self.argument_hint else "")


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.fullmatch((name or "").strip().lower()))


_README = """Every *.md in this directory is a slash command you can type.

    ---
    name: standup                             # optional; the filename wins
    description: What moved and what is blocked
    argument-hint: [project]
    ---

    Read my notes from the last two days$ARGUMENTS, then give me three lines:
    what moved, what is blocked, what I should start with today.

Typing `/standup acme` sends that body as your message, with $ARGUMENTS
replaced by "acme" ($1, $2, ... take the individual words). A body that uses
neither gets your words appended rather than losing them.

`/` on its own lists every command; `/` then Tab completes. `/commands new
<name> | [description] | <prompt>` writes one of these files for you, and
`/commands rm <name>` deletes it.

The assistant can write one too, with its save_command tool — those carry
`author: assistant` and are marked in the listing. Saving a command never runs
it; only typing its name does.
"""


def _parse(path: Path) -> Command | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    meta: dict[str, str] = {}
    m = _FRONTMATTER_RE.match(text)
    body = text
    if m:
        for line in m.group(1).splitlines():
            key, _, value = line.partition(":")
            if value:
                meta[key.strip().lower()] = value.strip()
        body = text[m.end():]
    name = (meta.get("name") or path.stem).strip().lower()
    if not valid_name(name):
        return None
    return Command(
        name=name,
        description=meta.get("description", "").strip(),
        body=body.strip(),
        argument_hint=meta.get("argument-hint", meta.get("argument_hint", "")).strip(),
        author=meta.get("author", "user").strip() or "user",
        path=path,
    )


def load_commands() -> list[Command]:
    """Every command on disk, by name. A malformed file costs itself, nothing else."""
    if not constants.COMMANDS_DIR.is_dir():
        return []
    out = []
    for path in sorted(constants.COMMANDS_DIR.glob("*.md")):
        cmd = _parse(path)
        if cmd is not None:
            out.append(cmd)
    return out


def get_command(name: str) -> Command | None:
    name = (name or "").strip().lower().lstrip("/")
    if not valid_name(name):
        return None
    path = constants.COMMANDS_DIR / f"{name}.md"
    return _parse(path) if path.exists() else None


def save_command(name: str, body: str, description: str = "",
                 argument_hint: str = "", author: str = "user") -> Path:
    """Write (or overwrite) a command file. Returns its path."""
    name = (name or "").strip().lower().lstrip("/")
    if not valid_name(name):
        raise ValueError(
            f"Invalid command name {name!r}: lowercase letters, digits, - and _ only.")
    if not (body or "").strip():
        raise ValueError("A command needs a body — the prompt it stands for.")
    constants.COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    # Not a .md, so it is never read back as a command of its own.
    readme = constants.COMMANDS_DIR / "README"
    if not readme.exists():
        try:
            readme.write_text(_README, encoding="utf-8")
        except Exception:
            pass
    path = constants.COMMANDS_DIR / f"{name}.md"
    front = ["---", f"name: {name}"]
    if description.strip():
        front.append(f"description: {description.strip()}")
    if argument_hint.strip():
        front.append(f"argument-hint: {argument_hint.strip()}")
    front.append(f"author: {author.strip() or 'user'}")
    front.append("---")
    path.write_text("\n".join(front) + "\n\n" + body.strip() + "\n",
                    encoding="utf-8")
    return path


def delete_command(name: str) -> bool:
    cmd = get_command(name)
    if cmd is None or cmd.path is None:
        return False
    try:
        cmd.path.unlink()
        return True
    except Exception:
        return False


def display_path(path: Path) -> str:
    """A path to show a person: project-relative when it is inside the project,
    absolute when it is not. relative_to() RAISES on a path outside the base,
    and a status line is the last place that should be able to take a session
    down."""
    try:
        return str(path.relative_to(constants.PROJECT_DIR))
    except ValueError:
        return str(path)


def render(cmd: Command, args: str) -> str:
    """The message text `/name args` stands for.

    $ARGUMENTS takes the whole argument string; $1..$9 take the words. A body
    that uses neither gets the arguments appended, because a command invoked
    with arguments that silently drops them is worse than one that is slightly
    clumsy about where they land.
    """
    args = (args or "").strip()
    words = args.split()
    text = cmd.body
    used = "$ARGUMENTS" in text or any(f"${i}" in text for i in range(1, 10))
    text = text.replace("$ARGUMENTS", args)
    for i in range(9, 0, -1):
        text = text.replace(f"${i}", words[i - 1] if len(words) >= i else "")
    if args and not used:
        text = f"{text.rstrip()}\n\n{args}"
    return text.strip()


def suggest(typed: str, known: list[str], limit: int = 3) -> list[str]:
    """Close matches for a command the user mistyped.

    A prefix match first — half-typed is the common case and difflib is bad at
    it — then difflib for genuine typos.
    """
    typed = (typed or "").strip().lower().lstrip("/")
    if not typed:
        return []
    prefix = [n for n in known if n.startswith(typed)]
    close = [n for n in difflib.get_close_matches(typed, known, n=limit, cutoff=0.6)
             if n not in prefix]
    return (prefix + close)[:limit]
