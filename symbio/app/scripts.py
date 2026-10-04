"""Scripts Symbio writes for itself, keeps, and runs again -- now, or on a schedule.

execute_code runs a piece of Python once and forgets it. A script is the same
Python saved under a name, with one line saying what it is for: run_script
runs it again with arguments, list_saved_scripts shows what there is, and a
scheduled job whose text is `script:<name> [args]` runs it when it comes due,
with no model turn at all. It runs in the same sandbox as execute_code -- the
same blocked imports, the same timeout, symbio_tools for files and fetch -- so
a saved script can do nothing a one-off could not.

They live in sandbox/scripts/, per install, and are never shipped.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from symbio import constants

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
# The first line of every saved script: what it is for, read back by
# list_saved_scripts so the listing says more than a file name.
_HEADER = "# symbio-script: "


def scripts_dir() -> Path:
    # Resolved at call time: the suite moves SANDBOX_DIR after import.
    return constants.SANDBOX_DIR / "scripts"


def normalize_name(name: Any) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(name or "").strip().lower()).strip("_-")
    if not _NAME.match(slug):
        raise ValueError(f"{name!r} is not a usable script name: lowercase letters, "
                         "digits, _ or -, starting with a letter.")
    return slug


def _split(text: str) -> tuple[str, str]:
    first, _, rest = text.partition("\n")
    if first.startswith(_HEADER):
        return first[len(_HEADER):].strip(), rest
    return "", text


def save_script(name: Any, code: Any, description: Any,
                config: dict[str, Any]) -> dict[str, Any]:
    """Write the script, refusing what the sandbox would refuse to run."""
    from symbio.app import sandbox

    slug = normalize_name(name)
    code = str(code or "").strip("\n")
    if not code.strip():
        raise ValueError("The script is empty.")
    # Checked now, not first at run time: a script that can never run is
    # better refused while the model that wrote it is still looking.
    safe, why = sandbox._is_code_safe(code, set(config["sandbox"]["blocked_imports"]))
    if not safe:
        raise ValueError(why)
    path = scripts_dir() / f"{slug}.py"
    replaced = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = " ".join(str(description or "").split())[:200]
    path.write_text(f"{_HEADER}{summary}\n{code}\n", encoding="utf-8")
    return {"name": slug, "path": str(path), "replaced": replaced, "description": summary}


def load_script(name: Any) -> tuple[str, str]:
    """(description, code) of a saved script; ValueError naming what exists."""
    slug = normalize_name(name)
    path = scripts_dir() / f"{slug}.py"
    try:
        return _split(path.read_text(encoding="utf-8"))
    except OSError:
        known = ", ".join(s["name"] for s in list_scripts()) or "none yet"
        raise ValueError(f"No script called {slug!r}. Saved scripts: {known}.") from None


def list_scripts() -> list[dict[str, Any]]:
    out = []
    for path in sorted(scripts_dir().glob("*.py")):
        try:
            description, code = _split(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        out.append({"name": path.stem, "description": description,
                    "lines": len(code.splitlines())})
    return out


def delete_script(name: Any) -> dict[str, Any]:
    slug = normalize_name(name)
    path = scripts_dir() / f"{slug}.py"
    if not path.exists():
        load_script(slug)  # raises with the list of what does exist
    path.unlink()
    return {"name": slug}


def parse_args(args: Any) -> list[str]:
    if args is None:
        return []
    if isinstance(args, str):
        try:
            return shlex.split(args)
        except ValueError:
            return args.split()
    if isinstance(args, (list, tuple)):
        return [str(a) for a in args]
    return [str(args)]


def run_script(name: Any, args: Any, config: dict[str, Any]) -> tuple[bool, str]:
    """Run a saved script in the sandbox, its arguments in the list ARGS."""
    from symbio.app import sandbox

    _description, code = load_script(name)
    # One line ahead of the script: its line numbers in a traceback are one
    # higher than in the file.
    prelude = f"ARGS = {json.dumps(parse_args(args), ensure_ascii=False)}\n"
    return sandbox.run_python_code(prelude + code, config)


def split_job_text(text: str) -> tuple[str, list[str]]:
    """'script:report --days 7' -> ('report', ['--days', '7'])."""
    parts = parse_args(text[len("script:"):].strip())
    if not parts:
        raise ValueError("A script: job needs a script name, e.g. 'script:daily_report'.")
    return parts[0], parts[1:]
