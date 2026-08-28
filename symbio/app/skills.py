"""Skill adapters: every saved skill gets its own worker LoRA adapter.

A skill is a markdown note under notes/ with a '# Skill: <name>' heading.
When a skill is saved we also create a worker role for it, store the role
in worker_models.json, and train a dedicated LoRA adapter under
adapters/workers/<slug>/ so the headmaster can later delegate to it.

Unused skills and adapters are archived after a configurable idle threshold.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.app import dispatch, memory, pending, training


NOTES_USAGE_FILE = constants.NOTES_DIR / ".last_used.json"
ADAPTER_ARCHIVE_DIR = constants.ADAPTER_ARCHIVE_DIR
_SKILL_FLAG = {"is_skill": True}


def _skill_health_path(note_path: Path) -> Path:
    """Sidecar file that holds health errors and corrections for a skill note."""
    return note_path.with_suffix(note_path.suffix + ".health.jsonl")


def _is_skill_note(path: Path) -> bool:
    """True if the markdown file begins with '# Skill:'."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].strip().lower()
    except (OSError, IndexError):
        return False
    return first.startswith("# skill:")


def _append_health_entry(note_path: Path, entry_type: str, text: str):
    """Append a dated error/correction entry to a skill's sidecar file."""
    if not _is_skill_note(note_path):
        return
    sidecar = _skill_health_path(note_path)
    entry = {
        "t": datetime.now().isoformat(),
        "type": entry_type,
        "text": text,
    }
    with open(sidecar, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def record_skill_error(note_path: Path, error: str):
    """Record a health error against the skill note at note_path."""
    _append_health_entry(note_path, "error", error)


def record_skill_correction(note_path: Path, correction: str):
    """Record a user correction against the skill note at note_path."""
    _append_health_entry(note_path, "correction", correction)


def read_skill_health(note_path: Path) -> list[dict[str, Any]]:
    """Return all recorded health/correction entries for a skill note."""
    sidecar = _skill_health_path(note_path)
    if not sidecar.exists():
        return []
    entries = []
    with open(sidecar, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _skill_slug(name: str) -> str:
    """Stable, filesystem-safe identifier from a skill title."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "skill"


def _skill_prompt_opener(name: str) -> str:
    """The one sentence shared by both skill prompt forms.

    Training and evaluation must agree on this framing, so it lives in one
    place rather than being written out twice.
    """
    return f"You are the specialist worker for the skill '{name}'."


def build_worker_system_prompt(name: str) -> str:
    """The prompt a skill worker is TRAINED under: names the skill, withholds
    the procedure.

    The steps are deliberately absent. If they were present here, the adapter
    would only ever learn to copy a procedure out of its own context, and no
    later evaluation could tell weight-learning apart from prompt-following.
    Keeping them out is what makes 'the skill lives in the weights' a claim
    that can be tested — see symbio.app.skill_eval.
    """
    return (
        f"{_skill_prompt_opener(name)} "
        "Follow the steps for that skill exactly, produce only the requested "
        "output, and do not add extra commentary."
    )


def _build_skill_system_prompt(name: str, steps: str) -> str:
    """The prompt a skill worker is SERVED under: includes the steps.

    Production keeps its safety net — a weak adapter still behaves because
    the procedure is right there. Serving with more context than training is
    harmless; the reverse would not be.
    """
    return (
        f"{_skill_prompt_opener(name)} "
        "Follow the steps below exactly, produce only the requested output, "
        "and do not add extra commentary.\n\n"
        f"Steps:\n{steps}\n\n"
        "Reply with the result of applying these steps to the user's request."
    )


def skill_note_body(name: str, steps: str) -> str:
    """The steps plus a Triggers block, which is what makes the note findable.

    Retrieval is term-frequency over the note body, so a tight four-line
    procedure loses to any long note that happens to repeat a common word.
    Measured 2026-08-24: for the query "scrape a listing page", the
    Browser Control note (152 words, "page" many times over) outscored the
    Scrape A Listing Page skill itself (57 words, "page" once) — 1.153 to
    1.088 — so the agent opened a browser instead of running the runbook it
    had just been trained on. A skill that cannot be retrieved is a skill the
    agent does not have.

    The triggers are derived from the procedure rather than written by hand:
    its own distinctive vocabulary is exactly what should route to it, and
    deriving them means every skill gets them instead of only the ones
    somebody remembered to annotate.
    """
    from symbio.app import skill_eval

    keywords = skill_eval._keywords(skill_eval._step_body(steps))
    name_terms = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", name)]
    # Name terms first: they are what a user actually types.
    seen, terms = set(), []
    for t in name_terms + keywords:
        if t not in seen and len(t) > 2:
            seen.add(t)
            terms.append(t)

    examples = [task.prompt for task in skill_eval.default_tasks(name)]
    return (
        f"{steps}\n\n"
        f"## Triggers\n\n"
        f"Keywords: {', '.join(terms)}\n\n"
        f"Examples:\n\n"
        + "\n".join(f"- {e}" for e in examples)
        + "\n"
    )


def _load_worker_catalog() -> dict[str, Any]:
    if not constants.WORKER_MODELS_FILE.exists():
        return {}
    try:
        return json.loads(constants.WORKER_MODELS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_worker_catalog(catalog: dict[str, Any]):
    constants.WORKER_MODELS_FILE.write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )


def _ensure_skill_catalog_entry(
    name: str, config: dict[str, Any], system_prompt: str
) -> str:
    """Add or update a worker catalog entry for this skill. Returns the role slug."""
    role = _skill_slug(name)
    catalog = _load_worker_catalog()

    # Remove any existing entry with the same role to keep catalog clean.
    for key, entry in list(catalog.items()):
        if entry.get("role") == role:
            del catalog[key]

    catalog[f"skill_{role}"] = {
        "model_name": worker_model_name(config),
        "role": role,
        "description": f"Skill: {name}",
        "adapter_compatible": True,
        "memory_note": "worker-size RAM at runtime, alongside the headmaster",
        "system_prompt": system_prompt,
        "is_skill": True,
        "skill_name": name,
    }
    _save_worker_catalog(catalog)
    return role


def _seed_user_turns(name: str) -> list[str]:
    """Ways a user might ask for this skill, for the seed samples.

    Kept deliberately distinct from skill_eval.default_tasks: if the seeds
    and the eval prompts were the same strings, a passing score would only
    show memorisation of the training set.
    """
    lower = name.lower()
    return [
        f"Apply the skill '{name}'.",
        f"How do I perform '{name}'?",
        f"What are the steps for {lower}?",
        f"Can you take care of {lower}?",
        f"Use your {lower} skill on this.",
        f"Time for {lower}.",
    ]


# Two seed kinds were tried here and reverted, because they were measured and
# they lost. Adding per-step questions ("what comes after X?") plus contrast
# samples pairing off-topic asks with a decline did stop the adapter reciting
# the procedure for "how do I reset my bluetooth headphones?" — and cost far
# more than it bought. On the same held-out battery, recall coverage fell from
# ~98% to ~38%: legitimate requests ("keys expired, what do I run first?") were
# declined, single steps came back in place of the procedure, and the decline
# string itself became an attractor strong enough to degrade unrelated
# generation into "That that that".
#
# The cause is structural, not a bad word list. A specialist worker only ever
# sees "fix my X" shaped prompts, so "my printer is offline" and "my keys
# expired" are the same shape to it, and 20-odd synthetic samples at several
# epochs cannot draw that boundary — whichever behaviour is repeated most just
# wins. Over-triggering inside a worker is also the cheaper failure: the
# headmaster decides what reaches it, so a false decline loses the skill
# outright while a false recite is merely noise.
#
# Worth revisiting with real usage samples rather than synthetic contrast, or
# with fewer iterations over a tiny corpus. Not worth shipping as it stood.


_LAST_TEACHER_ERROR: list[str] = []

_WORKED_PROMPT = """A specialist worker must carry out this procedure exactly:

{steps}

Write ONE training example showing the worker DOING it -- not describing it, not \
reporting what it would have done.

Rules:
- The output must be a complete, runnable Python script in a ```python fence.
- The script must literally perform every numbered step above, in order.
- Use the exact library names, ports, paths and attribute names the steps give.
- No status summaries, no JSON result objects, no prose explanation. Code only.
{fixture}{sandbox}{api}{avoid}{feedback}
Reply in exactly this format:
REQUEST: <one line: what the user asked for. Name the concrete file or URL AND \
say what to produce from it. "Process a.json and generate b.json" is NOT \
acceptable -- it names two files and states no task.>
OUTPUT:
```python
<the complete script>
```"""


_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.+?)```", re.S)


def _input_fixture_note(steps: str, config: dict[str, Any] | None) -> str:
    """Show the model the input files the procedure names, if they are present.

    A traceback localises a failure without diagnosing it. Told
    "'NoneType' object has no attribute 'text'" four times, the model kept
    patching the dereference -- .text, then .iter -- because nothing in that
    message says its selector matched nothing. The file is sitting in the
    sandbox and the real attribute values are right there, so stop making it
    guess: the same fix as the API reference and the blocked-import list.
    """
    if not config:
        return ""
    from symbio import constants

    names = set(re.findall(r"\b[\w-]+\.(?:json|jsonl|html|xml|csv|txt|ya?ml|ini|toml)\b",
                           steps, re.I))
    blocks = []
    for name in sorted(names):
        f = constants.SANDBOX_DIR / name
        try:
            body = f.read_text(encoding="utf-8")[:400]
        except OSError:
            continue
        blocks.append(f"  {name} starts with:\n    {body.strip()}")
    if not blocks:
        return ""
    return ("\nThe input files already exist here. Match the real attribute names "
            "and structure below exactly -- do not invent selectors:\n"
            + "\n".join(blocks) + "\n")


def _api_reference(steps: str) -> str:
    """Real signatures for any library the steps name, read off this machine.

    Correcting the teacher after the fact does not work: told eight times, with
    the right method named in the message, an 8B at temperature 0 rewrites the
    same `parser.find(...)` byte for byte. It cannot derive an API it never
    learned, and greedy decoding means a retry is not another sample.

    So ground the INPUT instead. The procedure says "parse with selectolax";
    selectolax is installed; its real classes and methods are one import away.
    Handing those over before generation costs nothing and removes the guess.
    """
    import importlib
    import importlib.util

    words = {w.strip(".,:;()[]'\"").lower()
             for w in re.split(r"[\s/]+", steps) if len(w.strip(".,:;()[]'\"")) > 2}
    # Ordinary English words that are also importable modules. "select the
    # container by data-testid" is not a request for the stdlib select module,
    # and dumping its kqueue API into the prompt is pure noise.
    skip = {"the", "and", "for", "with", "from", "never", "step", "then", "each",
            "port", "site", "twice", "rows", "last", "before", "move", "write",
            "select", "code", "string", "time", "calendar", "platform", "this",
            "keyword", "operator", "token", "types", "copy", "array", "queue",
            "signal", "stat", "glob", "parser", "pipes", "sched", "grp", "cmd"}
    blocks: list[str] = []
    for word in sorted(words - skip):
        if not word.isidentifier():
            continue
        try:
            if importlib.util.find_spec(word) is None:
                continue
            mod = importlib.import_module(word)
        except Exception:
            continue
        lines = [f"{word}:"]
        # Prefer the submodule that actually holds the classes -- selectolax's
        # HTMLParser lives in selectolax.parser, not the package root.
        targets = [(word, mod)]
        for sub in ("parser", "core", "api", "client"):
            try:
                targets.append((f"{word}.{sub}", importlib.import_module(f"{word}.{sub}")))
            except Exception:
                pass
        for modname, m in targets:
            for attr in sorted(a for a in dir(m) if not a.startswith("_")):
                obj = getattr(m, attr, None)
                if isinstance(obj, type):
                    methods = sorted(a for a in dir(obj) if not a.startswith("_"))[:14]
                    lines.append(f"  from {modname} import {attr}"
                                 f"   # methods: {', '.join(methods)}")
        if len(lines) > 1:
            blocks.append("\n".join(lines[:16]))
    if not blocks:
        return ""
    return ("\nThese libraries are installed here. Use these exact names and "
            "methods -- do not invent others:\n" + "\n".join(blocks) + "\n")


def _repair_imports(script: str) -> tuple[str, list[str]]:
    """Fix imports the machine can prove wrong, using the installed packages.

    The teacher writes `from selectolax import HTMLParser` and, told precisely
    that selectolax.HTMLParser does not exist, writes it again -- eight times,
    identically, because generation is greedy and an 8B that does not know the
    real module path cannot derive it from a one-line error. Asking a model to
    guess harder is the wrong move when the answer is sitting in site-packages.

    So the system looks it up: walk the package's submodules for one that really
    exports the name and rewrite the import to match. Ground truth from the
    environment, not another sample from a model that has already been wrong.

    Returns (script, list of repairs made).
    """
    import importlib
    import pkgutil

    repairs: list[str] = []
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return script, repairs

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        try:
            mod = importlib.import_module(node.module)
        except Exception:
            continue
        missing = [a.name for a in node.names
                   if a.name != "*" and not hasattr(mod, a.name)]
        if not missing:
            continue
        # Walk the package for a submodule that actually exports every name.
        found = None
        for info in pkgutil.iter_modules(getattr(mod, "__path__", [])):
            candidate = f"{node.module}.{info.name}"
            try:
                sub = importlib.import_module(candidate)
            except Exception:
                continue
            if all(hasattr(sub, n) for n in missing):
                found = candidate
                break
        if found:
            names = ", ".join(a.name for a in node.names)
            script = script.replace(f"from {node.module} import {names}",
                                    f"from {found} import {names}")
            repairs.append(f"{node.module} -> {found} for {', '.join(missing)}")
    return script, repairs


def _verify_example_code(script: str) -> str | None:
    """Reason this script cannot possibly run, or None if it might.

    The headmaster is being asked for a procedure that, by construction, nothing
    pre-trained contains -- so it guesses APIs. Its first attempts here included
    `from selectolax import HTMLParser` (the class lives in selectolax.parser)
    and calls to methods that do not exist. Seeding those teaches the worker to
    reproduce broken code forever, and no amount of training iterations recovers
    from a corpus that is wrong.

    So the system refuses to train on code it cannot compile and whose imports
    it cannot resolve. This does not make a script CORRECT -- only runnable --
    which is why execution against the real target is the layer above this one.
    """
    import importlib
    import importlib.util

    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return f"syntax: {e.msg} (line {e.lineno})"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if importlib.util.find_spec(root) is None:
                    return f"no module {alias.name!r}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            root = node.module.split(".")[0]
            try:
                if importlib.util.find_spec(node.module) is None:
                    return f"no module {node.module!r}"
            except (ImportError, ValueError, ModuleNotFoundError):
                return f"no module {node.module!r}"
            # `from selectolax import HTMLParser` resolves the module and still
            # fails, so the imported names have to be checked too.
            try:
                mod = importlib.import_module(node.module)
            except Exception as e:
                return f"import {node.module!r} failed: {type(e).__name__}"
            for alias in node.names:
                if alias.name != "*" and not hasattr(mod, alias.name):
                    if importlib.util.find_spec(f"{node.module}.{alias.name}") is None:
                        return f"{node.module}.{alias.name} does not exist"
    return None


def _verify_api_usage(script: str) -> str | None:
    """Reject method calls the installed classes do not actually have.

    Import repair fixed `from selectolax import HTMLParser`, and the very next
    line was still `parser.find('div[data-testid=...]')` -- HTMLParser has no
    `find`; the real API is `css_first`/`css`. That is an AttributeError at
    runtime, so import checking cannot see it, and executing the script cannot
    reach it either without a live site to fetch first.

    But the class is importable right here, so the machine can simply ask it.
    Track variables assigned from a constructor of an imported class, then check
    every attribute taken off them. The message names close matches, because
    "no .find" tells an 8B nothing while "no .find; did you mean css_first, css"
    is a correction it can act on.
    """
    import difflib
    import importlib

    try:
        tree = ast.parse(script)
    except SyntaxError:
        return None  # the syntax check already covers this

    # imported name -> live object
    imported: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            try:
                mod = importlib.import_module(node.module)
            except Exception:
                continue
            for alias in node.names:
                obj = getattr(mod, alias.name, None)
                if obj is not None:
                    imported[alias.asname or alias.name] = obj

    # var -> class, for `x = SomeImportedClass(...)`
    instances: dict[str, Any] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in imported):
            instances[node.targets[0].id] = imported[node.value.func.id]

    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in instances):
            cls = instances[node.value.id]
            if hasattr(cls, node.attr):
                continue
            options = [a for a in dir(cls) if not a.startswith("_")]
            near = difflib.get_close_matches(node.attr, options, n=3, cutoff=0.3)
            hint = f"; did you mean {', '.join(near)}" if near else ""
            name = getattr(cls, "__name__", node.value.id)
            return f"{name} has no .{node.attr}{hint}"
    return None


def _steps_want_code(steps: str) -> bool:
    """Does this procedure produce a program, or something a human does?

    'Bump the cursor in state.json' and 'parse with selectolax' name software;
    'let the tea steep for three minutes' does not. Only the first kind should
    have its worked examples rejected for lacking a script.
    """
    lowered = steps.lower()
    # A named data file settles it on its own. The counting rule below scores
    # each hint once, so "read rows.json ... write top.json ... state.json"
    # matched only ".json" and fell under the threshold -- which silently
    # disabled code verification for an obviously-code procedure, and the
    # examples went into the corpus unexecuted and unchecked.
    if re.search(r"\b[\w-]+\.(json|csv|txt|py|ya?ml|db|sqlite|xml|html|jsonl|log|ini|toml)\b",
                 lowered):
        return True
    hints = ("import", "port", "http", "url", "file", "script", "parse",
             "directory", "run ", "command", "api", "database", "query",
             "endpoint", "request", "header", "socket", "regex", "stdout")
    return sum(h in lowered for h in hints) >= 2


def _parse_worked_example(text: str) -> tuple[str, str] | None:
    """Pull (request, output) out of a generated example, or None."""
    if "REQUEST:" not in text or "OUTPUT:" not in text:
        return None
    _, rest = text.split("REQUEST:", 1)
    request, output = rest.split("OUTPUT:", 1)
    request, output = request.strip(), output.strip()
    if not request or not output or len(request) > 400:
        return None
    return request.splitlines()[0].strip(), output


# Errors that are the model's fault and it can act on, versus the ones that only
# say the environment was not there. A NameError means the code is wrong; a
# refused connection means the target is not running, and discarding the example
# for that would empty the corpus for every skill whose service happens to be
# down -- which is how attribute-checking alone took worked examples from 6 to 0.
_MODEL_FAULT = (
    # Both spellings on purpose. A traceback says "SyntaxError"; the sandbox
    # pre-compiles and reports its own "Syntax error: ..." summary line, which
    # matched none of these -- so a syntax error fell through to the generic
    # branch below and came back as the caret from the source excerpt. The
    # teacher was then handed "your attempt FAILED when it was actually run:
    # ^", which is not a defect anyone can act on.
    "SyntaxError", "Syntax error", "IndentationError", "NameError",
    "AttributeError", "TypeError", "ImportError", "ModuleNotFoundError",
    "UnboundLocalError", "KeyError", "ZeroDivisionError",
)
_ENV_FAULT = (
    "ConnectionRefusedError", "URLError", "HTTPError", "timeout", "TimeoutError",
    "ConnectionError", "socket.gaierror", "Errno 61", "Errno 8",
    # The sandbox blocks urllib/requests, so a procedure that fetches anything
    # cannot be executed there at all. That is a policy boundary, not a defect
    # in the generated code -- treating it as the model's fault would reject
    # every networked skill's examples and leave the corpus empty, which is
    # exactly how attribute-checking alone took worked examples from 6 to 0.
    # Such an example is accepted UNVERIFIED: see _seed_worked_examples.
    "is not allowed in the sandbox",
)


_DID_YOU_MEAN = re.compile(
    r"has no attribute '(?P<wrong>\w+)'\.\s*Did you mean:\s*'(?P<right>\w+)'", re.I)


def _repair_from_traceback(code: str, fault: str) -> tuple[str, str | None]:
    """Apply the fix CPython already named, instead of asking the model again.

    Python's AttributeError carries its own suggestion -- "'Node' object has no
    attribute 'attr'. Did you mean: 'attrs'?" -- and the model still rewrote
    `.attr` on the next attempt, exactly as it rewrote `parser.find` after being
    told the method name eight times. Correction-after-the-fact does not
    converge on a greedy generation; the interpreter's suggestion is ground
    truth, so use it directly.

    Only ever applies the name CPython itself proposed, never a guess of ours.
    """
    m = _DID_YOU_MEAN.search(fault or "")
    if not m:
        return code, None
    wrong, right = m.group("wrong"), m.group("right")
    patched = re.sub(rf"\.{re.escape(wrong)}\b", f".{right}", code)
    if patched == code:
        return code, None
    return patched, f".{wrong} -> .{right}"


def _informative_line(text: str) -> str:
    """The line of an error report that actually names the defect.

    A Python traceback puts it last, which is why the callers read from the
    end. The sandbox's own pre-compile check puts it FIRST and follows it with
    a source excerpt whose final line is a lone caret -- so reading from the
    end returned "^". Skip lines that carry no words.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if any(ch.isalpha() for ch in line.replace("|", "")):
            return line[:200]
    return (lines[0][:200] if lines else "exited non-zero")


def _execution_feedback(code: str, config: dict[str, Any]) -> tuple[str, str | None]:
    """Run the candidate. Return (usable, the model's own mistake to fix).

    This is the signal every static check was standing in for. `dir()` can say
    HTMLParser has no `.find`; only running the thing says
    "'list' object has no attribute 'first'" with the line that did it -- which
    is what a model needs to actually repair its own code, and what no amount of
    re-sampling a greedy generation will produce on its own.

    Returns one of three states, never a boolean -- a boolean is what let a
    script that the sandbox REFUSED to run be counted as "verified by
    execution", producing a 1/1 for code whose very next line would have raised:

        "ran"      executed cleanly; this is the only thing that earns "verified"
        "fault"    a traceback the model can act on; comes back as feedback
        "unrun"    could not be executed here at all (blocked import, dead
                   target). Keep the example -- rejecting it would empty the
                   corpus for every networked skill -- but never call it verified.
    """
    from symbio.app import sandbox

    try:
        ok, out = sandbox.run_python_code(code, config)
    except Exception as e:
        return "unrun", f"{type(e).__name__}: {e}"
    if ok:
        return "ran", None
    tail = (out or "").strip()
    if any(marker in tail for marker in _ENV_FAULT):
        return "unrun", tail.splitlines()[-1].strip()[:200] if tail else None
    for marker in _MODEL_FAULT:
        if marker in tail:
            line = next((ln for ln in reversed(tail.splitlines())
                         if marker in ln), marker)
            return "fault", line.strip()[:200]
    return "fault", _informative_line(tail) if tail else "exited non-zero"


def _seed_worked_examples(
    name: str, steps: str, generate_fn: Any, count: int = 6,
    config: dict[str, Any] | None = None,
) -> list[tuple[str, str, bool]]:
    """Ask the headmaster for worked examples of the procedure.

    The seeds above teach a worker to RECITE a procedure -- the assistant turn
    is the steps verbatim -- so a worker trained only on them can state what it
    would do and cannot do it. That is fine for a procedure a human then
    follows, and useless for one whose whole point is an artifact.

    This is the big-model-teaches-small-model half of the loop: the headmaster
    is already resident in chat, so producing examples costs no extra load, and
    generation finishes before the worker's trainer starts (no double residency).

    Diversity is enforced rather than hoped for. A batch of near-identical
    targets is what made an earlier hand-built scrape corpus score 7/8 on the
    page it had memorised and 1/8 on any other -- mean pairwise similarity 0.987
    -- so each request is generated knowing the previous ones, and a duplicate
    is retried once and then dropped.

    Returns (request, output, verified) per example, where `verified` means
    the script ran clean in the sandbox rather than merely passing the static
    checks. The flag used to be a local tally printed once and discarded,
    which was enough to report on seeding and not enough for anything to act
    on -- skill_perform mints its battery only from examples that ran, since
    a perturbed twin of an unrunnable example can never be executed either.
    """
    from difflib import SequenceMatcher

    wants_code = _steps_want_code(steps)
    api = _api_reference(steps) if wants_code else ""
    # Same move as the API reference: say what is unavailable BEFORE generating.
    # The model reaches for os/pathlib for file work by habit, both blocked, and
    # a blocked import means the candidate cannot be executed -- so it can never
    # be verified, however correct it happens to be. Telling it up front costs a
    # line; discovering it afterwards costs the whole verification.
    fixture_note = _input_fixture_note(steps, config) if wants_code else ""
    sandbox_note = ""
    if wants_code and config is not None:
        blocked = config.get("sandbox", {}).get("blocked_imports") or []
        if blocked:
            sandbox_note = (
                "\nThis code is executed in a sandbox. These imports are NOT "
                f"available: {', '.join(sorted(blocked))}. Use builtins instead --"
                " open() for files, json for parsing. Do not import them.\n")
    verified = 0  # examples that actually ran clean, not merely passed static checks
    # Why a requested example never arrived. Both of the paths counted here
    # used to drop silently, and the summary below reported "1/1 verified" for
    # a run where five of six requested slots produced nothing -- which reads
    # as success and is how a worker ends up with a corpus that is six parts
    # recitation to one part demonstration. That ratio is not a detail: this
    # file already states that on a corpus this small whichever behaviour is
    # repeated most just wins, and a worker seeded that way recites perfectly
    # and cannot perform. Measured on the first real trial of the perform
    # battery, 2026-08-28.
    dropped: Counter = Counter()
    examples: list[tuple[str, str, bool]] = []
    repeated = 0       # accepted despite matching one we already have
    # Set once the teacher has proved it writes the same script whatever the
    # avoid hint says. Diversity is worth three extra generations per slot to
    # find out; it is not worth 18 to keep re-confirming, and on a 14B teacher
    # those are minutes.
    deterministic = False
    for _ in range(count):
        # Carried across attempts. Generation is greedy so that seed data is not
        # a dice roll, which means an identical prompt reproduces an identical
        # rejection -- eight retries of "selectolax.HTMLParser does not exist"
        # taught nothing. The verifier already names the defect precisely, so
        # the retry hands it back and asks for that one thing to be fixed. A
        # teacher does not have to be right first time, only correctable.
        feedback = ""
        attempts = 1 if deterministic else 4
        for attempt in range(attempts):
            last_attempt = attempt == attempts - 1
            avoid = ""
            if examples:
                seen = "; ".join(r for r, _, _ok in examples[-4:])
                avoid = ("\nMake it clearly different from these, which are already "
                         f"covered: {seen}\n")
            try:
                raw = generate_fn(
                    _WORKED_PROMPT.format(steps=steps, fixture=fixture_note,
                                          sandbox=sandbox_note, api=api,
                                          avoid=avoid, feedback=feedback), 700)
            except Exception as e:
                # Never silent. A teacher that dies leaves the worker with
                # recall-only seeds that look exactly like success on the
                # sample count, which hid two failed runs before this line
                # existed.
                print(f"[skills] worked-example teacher failed: {type(e).__name__}: {e}",
                      file=__import__("sys").stderr, flush=True)
                _LAST_TEACHER_ERROR.append(f"{type(e).__name__}: {e}")
                return examples
            parsed = _parse_worked_example(raw or "")
            if parsed is None:
                dropped["unparseable"] += 1
                continue
            request, output = parsed
            ran_clean = False
            fence = _CODE_FENCE.search(output)
            if wants_code and fence is not None:
                code, repairs = _repair_imports(fence.group(1))
                if repairs:
                    print(f"[skills] repaired import(s): {'; '.join(repairs)}",
                          file=__import__("sys").stderr, flush=True)
                    output = output.replace(fence.group(1), code)
                broken = _verify_example_code(code) or _verify_api_usage(code)
                if not broken and config is not None:
                    state, fault = _execution_feedback(code, config)
                    if state == "fault":
                        patched, fix = _repair_from_traceback(code, fault or "")
                        if fix:
                            print(f"[skills] repaired from traceback: {fix}",
                                  file=__import__("sys").stderr, flush=True)
                            state, refault = _execution_feedback(patched, config)
                            if state == "ran":
                                output = output.replace(code, patched)
                                code, fault = patched, None
                            else:
                                fault = refault or fault
                        if state != "ran":
                            broken = fault or "did not run"
                    elif state == "ran":
                        ran_clean = True
                if broken:
                    print(f"[skills] rejected worked example -- {broken}",
                          file=__import__("sys").stderr, flush=True)
                    feedback = (f"\nYour previous attempt FAILED when it was actually run:\n"
                                f"  {broken}\n"
                                "Fix exactly that and keep the rest. This is the real "
                                "error from executing your code, not a guess.\n")
                    continue
            if wants_code and fence is None:
                feedback = ("\nYour previous attempt had no ```python fence. "
                            "Reply with runnable code, not a description.\n")
                # The 8B's first instinct on "produce the artifact" is a JSON
                # status report ({"status": "success", "moved": {...}}), which
                # would teach the worker to narrate outcomes it never produced.
                continue
            duplicate = any(SequenceMatcher(None, output, prev).ratio() > 0.95
                            for _, prev, _ok in examples)
            if duplicate and not last_attempt:
                dropped["duplicate"] += 1
                continue  # try for a genuinely different one first
            if duplicate:
                # Take the repeat rather than the empty slot. A procedure
                # specified tightly enough to be worth a worker has close to one
                # correct script, and a deterministic teacher will keep writing
                # it however the avoid hint is worded -- measured here, six
                # requested examples produced one, the other five all
                # near-identical. The empty slots are not the harmless outcome:
                # they left a corpus of 6 recall samples to 1 demonstration, and
                # this file already states that on a corpus that small whichever
                # behaviour is repeated most just wins. That worker recited the
                # four steps perfectly and answered every real request, memorised
                # or not, with fabricated JSON. A repeated demonstration is a
                # much smaller problem than a corpus that teaches reciting.
                repeated += 1
                deterministic = True
            examples.append((request, output, ran_clean))
            if ran_clean:
                verified += 1
            break
    recall_count = len(_seed_user_turns(name))
    distinct = len(examples) - repeated
    if len(examples) < count or repeated:
        detail = ", ".join(f"{n} {why}" for why, n in sorted(dropped.items()))
        print(f"[skills] {len(examples)}/{count} worked example(s): {distinct} "
              f"distinct, {repeated} repeated"
              + (f" ({detail} retried)" if detail else "")
              + f". Corpus is {recall_count} recall to {len(examples)} "
              f"demonstration sample(s).",
              file=__import__("sys").stderr, flush=True)
    if wants_code and config is not None:
        print(f"[skills] {verified}/{len(examples)} worked example(s) VERIFIED by "
              f"execution; {len(examples) - verified} could not be run here "
              f"(blocked import or dead target) and are static-checked only",
              file=__import__("sys").stderr, flush=True)
    return examples


_PY_TAG_RE = re.compile(r"<py>(.*?)</py>", re.S)
_PY_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.+?)```", re.S)
_OBSERVATION_PREFIX = "[System observation:"
# What the sandbox says when a script actually ran. Anything else -- a
# traceback, a refusal, a blocked import -- is not a demonstration.
_RAN_MARKERS = ("Python script exited ok", "exited ok")


def _harvested_request(history: list[dict[str, Any]], upto: int) -> str | None:
    """The last thing the USER actually asked, before assistant turn `upto`.

    Observations are appended to history as user turns, so the most recent
    user message is usually the tool result rather than the request.
    """
    for i in range(upto - 1, -1, -1):
        msg = history[i]
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content or content.startswith(_OBSERVATION_PREFIX):
            continue
        if "<tool_response>" in content:
            continue
        return content
    return None


def harvest_worked_examples(
    history: list[dict[str, Any]] | None, max_examples: int = 4,
) -> list[tuple[str, str, bool]]:
    """Worked examples taken from what the session ACTUALLY just did.

    A skill is usually saved right after the work it describes has been done,
    and until now that work was thrown away: seeding asked the headmaster to
    *invent* examples of a procedure it had this moment carried out for real.
    The invented ones are strictly worse evidence. They are guesses at what the
    code should look like, they have to be executed afterwards to find out
    whether they run, and a greedy teacher writes the same one six times over
    (measured: 6 requested, 1 distinct). A harvested example is a real request
    paired with code that already ran in this sandbox against this machine's
    real files -- verified by having happened, not by a check bolted on after.

    Returned newest-first as (request, output, verified) so it drops straight
    into the same list _seed_worked_examples fills. `verified` is True only
    where the observation says the script ran; a script that raised is not a
    demonstration of anything.
    """
    if not history:
        return []
    out: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for i in range(len(history) - 1, -1, -1):
        if len(out) >= max_examples:
            break
        msg = history[i]
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        match = _PY_TAG_RE.search(content) or _PY_FENCE_RE.search(content)
        if match is None:
            continue
        code = match.group(1).strip()
        if not code or code in seen:
            continue
        # The observation for this call is the next user turn.
        ran = False
        for j in range(i + 1, min(i + 3, len(history))):
            nxt = history[j]
            if nxt.get("role") != "user":
                continue
            body = nxt.get("content") or ""
            if any(m in body for m in _RAN_MARKERS):
                ran = True
            break
        if not ran:
            continue
        request = _harvested_request(history, i)
        if not request:
            continue
        seen.add(code)
        out.append((request, f"```python\n{code}\n```", True))
    return out


def _seed_skill_training_data(
    role: str, name: str, steps: str, tokenizer: Any,
    example_generator: Any = None, config: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> int:
    """Write synthetic training samples for a brand-new skill worker.

    The system turn names the skill but withholds the procedure; the steps
    appear only in the assistant turn. That is what pushes the procedure into
    the weights instead of teaching the adapter to copy it out of context.

    Known limits, measured rather than assumed: with these seeds the procedure
    is recalled reliably for held-out phrasings, but its steps transpose on
    unusually worded requests (~80% step order), and an unrelated
    troubleshooting question can draw the whole procedure. See the note above
    for what was tried against that and why it was reverted.

    Returns the number of samples written. Real usage samples accumulate
    automatically in dispatch.WorkerPool.run_delegated_task.
    """
    data_dir = constants.data_dir_for(role)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"

    # Clear any stale auto-seeded samples so re-saving a skill refreshes the seed.
    if train_file.exists():
        lines = [
            ln for ln in train_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not json.loads(ln).get("metadata", {}).get("skill_seed")
        ]
    else:
        lines = []

    system_prompt = build_worker_system_prompt(name)
    # Chat templates are model-specific: ChatML turn markers mean nothing to a
    # Mistral tokenizer and vice versa. Record which model's template rendered
    # these samples so training can refuse a mismatched pairing instead of
    # burning an hour learning noise. See seed_model_mismatch().
    tokenized_for = getattr(tokenizer, "name_or_path", None) or ""

    samples = [(turn, steps, "recall") for turn in _seed_user_turns(name)]

    # Worked examples sit ALONGSIDE the recall seeds, never in place of them.
    # The note above records two seed kinds that were tried and reverted after
    # recall coverage fell ~98% -> ~38%; whichever behaviour is repeated most
    # just wins on a corpus this small, so the recall half is kept whole and
    # the additions are capped at roughly the same count.
    # What the session actually did comes first, and the teacher is only asked
    # to make up the shortfall. A skill is usually saved straight after the
    # work it describes, so this is the one moment real demonstrations are
    # free -- already run, already correct, already varied by having happened.
    worked = harvest_worked_examples(history)
    if worked:
        print(f"[skills] harvested {len(worked)} worked example(s) from work "
              f"this session actually ran; asking the teacher for the rest.",
              file=__import__("sys").stderr, flush=True)

    if example_generator is not None or worked:
        if example_generator is not None:
            wanted = max(0, len(_seed_user_turns(name)) - len(worked))
            if wanted:
                worked += _seed_worked_examples(
                    name, steps, example_generator, count=wanted, config=config)
        for request, output, _verified in worked:
            samples.append((request, output, "worked"))

        # Mint the perform battery from the same examples, before they are
        # rendered into training text. These are the only two things that know
        # both what the worker was taught and that it demonstrably ran, and a
        # held-out check has to be built from that pair or it is guessing.
        if worked and config is not None:
            try:
                from symbio.app import skill_perform

                minted = skill_perform.mint_and_save(
                    role, worked, config, wants_code=_steps_want_code(steps))
                print(f"[skills] minted {minted} held-out performance check(s) "
                      f"for '{role}'"
                      + ("" if minted else " -- no example had a value shared "
                                          "between its request and its script"),
                      file=__import__("sys").stderr, flush=True)
            except Exception as e:
                # A battery that fails to mint costs a guard rail, not the
                # skill. Loud, and never fatal to the seeding it hangs off.
                print(f"[skills] performance battery not minted for '{role}': "
                      f"{type(e).__name__}: {e}",
                      file=__import__("sys").stderr, flush=True)

    demonstrations = sum(1 for _u, _a, kind in samples if kind == "worked")
    recitations = sum(1 for _u, _a, kind in samples if kind == "recall")
    if demonstrations < recitations:
        # The one ratio that decides what the worker becomes, said plainly, and
        # said here rather than inside the teacher: examples can now arrive by
        # being harvested from real work instead of generated, and a harvest-only
        # seeding skipped the warning entirely.
        print(f"[skills] WARNING: recitation outnumbers demonstration "
              f"{recitations}:{demonstrations} for '{name}'. Expect a worker "
              f"that states the steps rather than carrying them out.",
              file=__import__("sys").stderr, flush=True)

    written = 0
    for user_turn, answer, kind in samples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        lines.append(json.dumps({
            "text": text,
            # Carried alongside the rendered text so mlx_lm selects ChatDataset
            # and can mask the prompt out of the loss. Without it these samples
            # depend on training.upgrade_corpus_to_messages recovering the
            # structure by re-parsing, which only works for templates it knows.
            "messages": messages,
            "metadata": {
                "skill_seed": True,
                "skill": name,
                "seed_kind": kind,
                "tokenized_for": tokenized_for,
            },
        }))
        written += 1

    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def _model_stem(model_name: str) -> str:
    """Bare model id, ignoring the publishing org.

    'Qwen/Qwen3-8B-MLX-4bit' and 'mlx-community/Qwen3-8B-MLX-4bit' are the
    same weights republished; comparing full paths would flag them as a
    mismatch and block a training run that is actually fine.
    """
    return model_name.rsplit("/", 1)[-1].strip().lower()


def delegatable_role_for_note(note_path: Path, config: dict[str, Any]) -> str | None:
    """The worker role a retrieved skill note can be handed to, if any.

    Retrieval already does the matching work: it scores the user's message
    against every note, and skills *are* notes. When "my wifi is broken" pulls
    up the 'Skill: Fix wifi' note, that hit is a routing signal — it was simply
    never consulted for dispatch, so the model had to rediscover from the tool
    schema alone that a matching specialist existed.

    Returns None unless the skill has a trained adapter that belongs to the
    model that would load it: suggesting a worker whose weights are absent or
    were built for another model would send the turn somewhere worse than
    answering directly.
    """
    if not config.get("dispatch", {}).get("enabled", False):
        return None
    if not _is_skill_note(note_path):
        return None
    try:
        heading = note_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    name = heading.split(":", 1)[1].strip() if ":" in heading else ""
    if not name:
        return None
    role = _skill_slug(name)

    from symbio.app import dispatch

    entry = dispatch.catalog_entry_for_role(role)
    if entry is None:
        return None
    adapter_dir = constants.adapter_dir_for(role)
    if not (adapter_dir / "adapters.safetensors").exists():
        return None
    if not dispatch.adapter_matches_model(adapter_dir, entry["model_name"]):
        return None
    return role


def worker_model_name(config: dict[str, Any]) -> str:
    """Which model a skill worker should run.

    Defaults to `dispatch.worker_model_name` so a worker is not a second copy
    of the headmaster's weights — a skill answers one narrow question under a
    short prompt, and does not need the size the general agent does. Falls back
    to the headmaster's own model when unset.
    """
    configured = config.get("dispatch", {}).get("worker_model_name")
    return configured or config["model_name"]


def worker_tokenizer(model_name: str, fallback: Any) -> Any:
    """The tokenizer of the model this worker will actually be trained on.

    Seeds are stamped with the tokenizer that rendered them and
    seed_model_mismatch refuses to train when that disagrees with the worker's
    model. Once workers stopped sharing the headmaster's model, seeding with
    the headmaster's tokenizer would stamp every new skill with the wrong name
    and block its first training run. Falls back to the caller's tokenizer if
    the worker's cannot be loaded — a mismatch is then reported honestly rather
    than being silently papered over.
    """
    current = getattr(fallback, "name_or_path", "") or ""
    if current and _model_stem(current) == _model_stem(model_name):
        return fallback
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return fallback


def seed_model_mismatch(role: str, model_name: str) -> str | None:
    """Explain why a role's seed data cannot train `model_name`, or None.

    The seeds are rendered by the headmaster's tokenizer at skill-creation
    time, but training uses the model recorded in the worker catalog. When a
    skill outlives a model switch those two drift apart, and the mismatch is
    invisible: training runs happily to completion on samples whose turn
    markers the model has never seen.
    """
    train_file = constants.data_dir_for(role) / "train.jsonl"
    if not train_file.exists():
        return None
    for raw in train_file.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            meta = json.loads(raw).get("metadata", {})
        except json.JSONDecodeError:
            continue
        stamped = meta.get("tokenized_for")
        if not stamped:
            continue
        if _model_stem(stamped) != _model_stem(model_name):
            return (
                f"Training data for role '{role}' was tokenized for "
                f"'{stamped}' but the worker is configured to train "
                f"'{model_name}'. Their chat templates differ, so this run "
                f"would learn nothing useful. Re-save the skill to re-seed "
                f"it for the current model."
            )
    return None


def save_skill_adapter(
    name: str,
    steps: str,
    config: dict[str, Any],
    tokenizer: Any,
    auto_train: bool = True,
    example_generator: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save a skill note and create a dedicated worker adapter for it.

    Returns a dict with note_path, role, adapter_dir, and training status.
    """
    note_path = memory.save_skill(name, steps)
    system_prompt = _build_skill_system_prompt(name, steps)
    role = _ensure_skill_catalog_entry(name, config, system_prompt)

    # Refresh dispatch's in-memory view of the catalog.
    # (load_catalog is lazy, so stale disk state is harmless on next call.)

    # Make the new role visible to delegate_task in this session rather than
    # at the next restart — a skill the model is not told about is one it
    # cannot route to.
    try:
        from symbio.app import tooling

        tooling.refresh_delegate_roles()
    except Exception:
        pass

    # Seed with the tokenizer of the model the worker will be trained on, not
    # the headmaster's: the samples are stamped with it, and training refuses
    # to run when that stamp disagrees with the worker's model.
    worker_model = worker_model_name(config)
    seed_tokenizer = tokenizer
    if worker_model != config.get("model_name"):
        seed_tokenizer = worker_tokenizer(worker_model, tokenizer)

    # Recall seeds are written synchronously -- they are six string formats and
    # cost nothing. Worked examples need the headmaster to generate them, which
    # takes a minute or two, so they are added on the training thread below
    # rather than blocking the chat turn that saved the skill.
    seeded = _seed_skill_training_data(role, name, steps, seed_tokenizer)
    adapter_dir = constants.adapter_dir_for(role)

    result = {
        "note_path": str(note_path),
        "role": role,
        "adapter_dir": str(adapter_dir),
        "seeded_samples": seeded,
        "trained": False,
        "message": f"Skill '{name}' saved as worker role '{role}' with {seeded} seed samples.",
    }

    # The seeds are on disk and the adapter is not: from this line until a
    # training run finishes, the skill is real but cannot answer from its own
    # weights. Written down before the thread starts and before the crash
    # window opens — guarded_train_worker supersedes this entry when it picks
    # the work up, and clears it when it finishes, so a skill can never end up
    # permanently seeded-but-untrained without something saying so.
    pending.defer("train_worker", f"first adapter for skill '{name}'",
                  role=role, reason="skill saved; adapter not trained yet")

    if auto_train:
        # Training the headmaster-sized model blocks for minutes; run in the
        # background so the chat front-end stays responsive.
        def _train():
            # Generate worked examples first: the headmaster is still resident
            # here, and the worker's trainer has not loaded anything yet, so
            # this window is the one place both do not overlap.
            if example_generator is not None:
                try:
                    added = _seed_skill_training_data(
                        role, name, steps, seed_tokenizer,
                        example_generator=example_generator, config=config,
                        history=history)
                    result["seeded_samples"] = added
                except Exception as e:  # a failed teacher must not lose the skill
                    result["seed_error"] = str(e)
            trained, msg = dispatch.guarded_train_worker(role, config, iters=None)
            result["trained"] = trained
            result["training_message"] = msg

        threading.Thread(target=_train, daemon=True, name=f"train-skill-{role}").start()
        result["message"] += " Adapter training started in the background."
    else:
        result["message"] += " Run /train_worker {} when ready to train.".format(role)

    return result


def list_skill_adapters() -> list[dict[str, Any]]:
    """Return metadata for every active skill adapter."""
    out = []
    catalog = _load_worker_catalog()
    for entry in catalog.values():
        if not entry.get("is_skill"):
            continue
        role = entry["role"]
        adapter_dir = constants.adapter_dir_for(role)
        exists = (adapter_dir / "adapter_config.json").exists()
        last_used = training.adapter_last_used(role=role)
        out.append({
            "role": role,
            "name": entry.get("skill_name", role),
            "description": entry.get("description", ""),
            "adapter_exists": exists,
            "adapter_dir": str(adapter_dir),
            "last_used": last_used.isoformat() if last_used else None,
        })
    return out


def delete_skill_adapter(role: str) -> dict[str, Any]:
    """Remove a skill's worker catalog entry, adapter weights, training data,
    and any health sidecar tied to its note."""
    catalog = _load_worker_catalog()
    removed_keys = [k for k, e in catalog.items() if e.get("role") == role and e.get("is_skill")]
    for k in removed_keys:
        del catalog[k]
    _save_worker_catalog(catalog)

    # Remove the skill note and its health sidecar if they exist.
    note_path = None
    for title, p in memory.list_skills():
        if _skill_slug(title[7:].strip()) == role:
            note_path = p
            break
    if note_path and note_path.exists():
        note_path.unlink()
        sidecar = _skill_health_path(note_path)
        if sidecar.exists():
            sidecar.unlink()

    adapter_dir = constants.adapter_dir_for(role)
    data_dir = constants.data_dir_for(role)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    return {"role": role, "removed_entries": removed_keys}


# ---- Usage tracking and archival ----


def record_note_usage(path: Path):
    """Update the last-accessed timestamp for a markdown note."""
    manifest = _load_note_usage_manifest()
    manifest[str(path.resolve())] = datetime.now().isoformat()
    _save_note_usage_manifest(manifest)


def _load_note_usage_manifest() -> dict[str, str]:
    if not NOTES_USAGE_FILE.exists():
        return {}
    try:
        return json.loads(NOTES_USAGE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_note_usage_manifest(manifest: dict[str, str]):
    NOTES_USAGE_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _note_mtime(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.min


def _is_protected_note(path: Path) -> bool:
    """Identity and preference notes should never be auto-archived."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].lower()
    except (OSError, IndexError):
        return False
    protected = {
        "# my identity",
        "# user identity",
        "# user preference",
        "# assistant identity",
    }
    return any(first.startswith(p) for p in protected)


def archive_idle_notes(config: dict[str, Any], dry_run: bool = False) -> list[str]:
    """Move markdown notes that haven't been used recently to notes/archive/.

    Returns the list of archived filenames. In dry-run mode the candidates are
    returned but nothing is moved.
    """
    days = int(config.get("archive", {}).get("note_idle_days", 90))
    if days <= 0:
        return []
    cutoff = datetime.now() - timedelta(days=days)
    manifest = _load_note_usage_manifest()
    archived: list[str] = []

    for f in sorted(constants.NOTES_DIR.glob("*.md")):
        if not f.is_file() or _is_protected_note(f):
            continue
        # Use explicit last-used if available, else file mtime.
        last_used_str = manifest.get(str(f.resolve()))
        if last_used_str:
            try:
                last_used = datetime.fromisoformat(last_used_str)
            except ValueError:
                last_used = _note_mtime(f)
        else:
            last_used = _note_mtime(f)
        if last_used > cutoff:
            continue
        archived.append(f.name)
        if dry_run:
            continue
        dest = constants.NOTES_ARCHIVE_DIR / f.name
        counter = 1
        while dest.exists():
            dest = constants.NOTES_ARCHIVE_DIR / f"{f.stem}_{counter}{f.suffix}"
            counter += 1
        f.rename(dest)
        # Move any health sidecar with the note.
        sidecar = _skill_health_path(f)
        if sidecar.exists():
            sidecar_dest = _skill_health_path(dest)
            sidecar.rename(sidecar_dest)
        # Drop from manifest so a restored note starts fresh.
        manifest.pop(str(f.resolve()), None)

    if archived and not dry_run:
        _save_note_usage_manifest(manifest)
    return archived


def archive_idle_adapters(config: dict[str, Any], dry_run: bool = False) -> list[str]:
    """Move worker/skill adapters that haven't been loaded recently to an archive dir.

    The headmaster's own adapter (role=None) is never archived. Returns the
    list of archived role names. In dry-run mode the candidates are returned
    but nothing is moved.
    """
    days = int(config.get("archive", {}).get("adapter_idle_days", 90))
    if days <= 0:
        return []
    cutoff = datetime.now() - timedelta(days=days)
    archived: list[str] = []

    catalog = _load_worker_catalog()
    active_roles = {e.get("role") for e in catalog.values() if e.get("role")}

    if not dry_run:
        ADAPTER_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for role in active_roles:
        if not role:
            continue
        adapter_dir = constants.adapter_dir_for(role)
        if not adapter_dir.exists():
            continue
        last_used = training.adapter_last_used(role=role)
        if last_used is None:
            # Never loaded; use directory mtime as a proxy.
            last_used = datetime.fromtimestamp(adapter_dir.stat().st_mtime)
        if last_used > cutoff:
            continue
        archived.append(role)
        if dry_run:
            continue
        dest = constants.adapter_archive_dir_for(role).with_suffix(
            f".bak.{datetime.now():%Y%m%d_%H%M%S}"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(adapter_dir), str(dest))

    return archived


def archive_idle_items(config: dict[str, Any], dry_run: bool = False) -> dict[str, list[str]]:
    """Run both archival passes and return what would be archived."""
    return {
        "notes": archive_idle_notes(config, dry_run=dry_run),
        "adapters": archive_idle_adapters(config, dry_run=dry_run),
    }


# ---- Restore ----


def list_archived_notes() -> list[str]:
    """Return filenames of notes currently in notes/archive/."""
    if not constants.NOTES_ARCHIVE_DIR.exists():
        return []
    return sorted(f.name for f in constants.NOTES_ARCHIVE_DIR.glob("*.md"))


def list_archived_adapters() -> list[str]:
    """Return basenames of archived adapter directories."""
    if not ADAPTER_ARCHIVE_DIR.exists():
        return []
    return sorted(
        f.name for f in ADAPTER_ARCHIVE_DIR.rglob("*")
        if f.is_dir() and (f / "adapter_config.json").exists()
    )


def restore_archived_note(filename: str) -> Path | None:
    """Move a note from notes/archive/ back to notes/."""
    src = constants.NOTES_ARCHIVE_DIR / filename
    if not src.exists():
        return None
    dest = constants.NOTES_DIR / filename
    counter = 1
    while dest.exists():
        dest = constants.NOTES_DIR / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), str(dest))
    # Restore any health sidecar alongside the note.
    sidecar_src = _skill_health_path(src)
    if sidecar_src.exists():
        sidecar_src.rename(_skill_health_path(dest))
    record_note_usage(dest)
    return dest


def restore_archived_adapter(role: str) -> Path | None:
    """Restore the most recently archived adapter for a role to its live path.

    Returns the restored adapter directory path, or None if no archive exists.
    """
    adapter_dir = constants.adapter_dir_for(role)
    candidates = sorted(
        ADAPTER_ARCHIVE_DIR.rglob(f"{role}.bak.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    src = candidates[0]
    if adapter_dir.exists():
        # Back up the live adapter before overwriting, just in case.
        backup = ADAPTER_ARCHIVE_DIR / f"{role}.live.bak.{datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(str(adapter_dir), str(backup))
    shutil.move(str(src), str(adapter_dir))
    training.mark_adapter_used(role=role)
    return adapter_dir
