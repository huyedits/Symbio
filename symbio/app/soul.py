"""What the machine takes itself to be, and what it reads its user to value.

The two curated stores that already exist answer different questions.
agent_memory.md holds facts it learned; user_profile.md holds who the user is.
Neither holds the thing that actually shapes how a turn should go: what this
assistant is BEING USED AS — a companion, a worker, a tool someone reaches for
under time pressure — and the operating values that follow from it. "Wants the
least possible friction" and "wants to approve everything" are opposite
instructions and neither is a fact about the user's biography.

So this store has exactly two sections, and they are written differently:

  ROLE   its read of its own place. First person, because it is the only
         store about the machine rather than about the world.
  VALUE  a disposition it has inferred about the user, always with the
         behaviour that suggested it. "Prefers fewer confirmations (asked me
         to stop asking twice)" is usable; "is impatient" is a character
         judgement and gets refused below.

Written from the background thread after a turn, and immediately when
something drastic happens — a correction, a refusal, a tool failure — because
those are the turns that actually revise a read, and waiting for the next idle
moment to record one is how the revision gets lost.

EVERYTHING HERE IS DERIVED FROM CONVERSATION, so it goes back to the model
wrapped as untrusted, exactly like agent_memory.md and user_profile.md. A page
the model read can put words in its mouth; those words must not become a
standing instruction by being written down and read back as the machine's own
belief about itself.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

from symbio import constants, safety

ROLE = "role"
VALUE = "value"

_HEADINGS = {
    ROLE: "## What I am being used as",
    VALUE: "## What {user} seems to value",
}

# One line each. A store that grows without a shape becomes a diary, and a
# diary does not fit in a prompt.
_MAX_LINE = 180
# What an empty section renders as. Parsed back out again, or it becomes a
# real entry: the first observation written after it left the store saying
# both "nothing observed yet" and the thing it had just observed.
_PLACEHOLDER = "(nothing observed yet)"
_DEFAULT_LIMIT = 2000

# Character judgements, refused. The difference that matters is whether the
# line names something OBSERVED or something imputed: "prefers fewer
# confirmations (said stop asking)" survives a disagreement, "is impatient"
# is an accusation the user never gets to answer, and it would sit in every
# prompt from then on.
_JUDGEMENT_RE = re.compile(
    r"\b(?:lazy|stupid|rude|impatient|careless|sloppy|aggressive|hostile|"
    r"incompetent|paranoid|obsessive|difficult|demanding|unreasonable|"
    r"emotional|erratic|unstable|angry|toxic)\b", re.IGNORECASE)


def soul_path():
    return constants.SOUL_FILE


def _limit(config: dict[str, Any]) -> int:
    try:
        return int(config.get("memory", {}).get("soul_char_limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT


def read_soul() -> str:
    try:
        return constants.SOUL_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


def sections() -> dict[str, list[str]]:
    """The store parsed back into its two lists, oldest first."""
    out: dict[str, list[str]] = {ROLE: [], VALUE: []}
    current = None
    for line in read_soul().splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            low = stripped.lower()
            current = ROLE if "used as" in low else VALUE if "value" in low else None
            continue
        if current and stripped.startswith("- "):
            entry = stripped[2:].strip()
            if entry and entry != _PLACEHOLDER:
                out[current].append(entry)
    return out


def _normalise(line: str) -> str:
    """For comparing two observations, not for storing them."""
    return re.sub(r"[^a-z0-9 ]+", "", line.lower()).strip()


def _is_duplicate(line: str, existing: list[str]) -> bool:
    """Whether `line` says something already recorded.

    Substring either way, on the normalised text: a reflection pass run every
    turn will produce the same read repeatedly, and a store that appends each
    time is one that says the same sentence forty times and evicts everything
    else to make room.
    """
    fresh = _normalise(line)
    if not fresh:
        return True
    for other in existing:
        known = _normalise(other)
        if not known:
            continue
        if fresh in known or known in fresh:
            return True
    return False


def acceptable(kind: str, line: str) -> tuple[bool, str]:
    """Whether an observation may be written, and why not if not."""
    line = (line or "").strip().lstrip("-").strip()
    if not line:
        return False, "empty"
    if len(line) > _MAX_LINE:
        return False, "too long to be one observation"
    if kind == VALUE and _JUDGEMENT_RE.search(line):
        # See _JUDGEMENT_RE. This is the one refusal that is about the person
        # rather than about safety.
        return False, "reads as a character judgement rather than an observation"
    if kind == VALUE and "(" not in line:
        # The evidence is what makes it revisable later.
        return False, "no observed behaviour cited"
    return True, ""


def record(kind: str, line: str, config: dict[str, Any]) -> bool:
    """Add one observation. Returns whether anything was written.

    Secrets are stripped on the way in, not on the way out: this file is read
    back into every prompt and is a training input like any other store, so a
    credential that reached it once would keep arriving.
    """
    ok, _why = acceptable(kind, line)
    if not ok:
        return False
    from symbio.app import tooling

    line = tooling.redact_secrets((line or "").strip().lstrip("-").strip())
    current = sections()
    if _is_duplicate(line, current.get(kind, [])):
        return False
    current.setdefault(kind, []).append(line)
    _write(current, config)
    return True


def _write(current: dict[str, list[str]], config: dict[str, Any]) -> None:
    user = config.get("user_name", "the user")
    limit = _limit(config)
    while True:
        body = _render(current, user)
        if len(body) <= limit:
            break
        # Over the cap: drop the OLDEST line of whichever section is longer.
        # Oldest, because a read of the relationship that has been superseded
        # is exactly the one worth losing, and the newest line is the one that
        # just earned its place.
        longest = max(current, key=lambda k: len(current[k]))
        if not current[longest]:
            break
        current[longest].pop(0)
    constants.SOUL_FILE.write_text(body, encoding="utf-8")


def _render(current: dict[str, list[str]], user: str) -> str:
    parts = [f"# Soul\n\nLast updated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for kind in (ROLE, VALUE):
        parts.append(_HEADINGS[kind].format(user=user))
        lines = current.get(kind) or []
        parts.append("\n".join(f"- {l}" for l in lines) if lines
                     else f"- {_PLACEHOLDER}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def soul_block(config: dict[str, Any]) -> str:
    """The store as it reaches the model: inline, and marked untrusted.

    Inline every turn is the whole point — a read of what it is being used for
    is only worth having while it can still change how the turn goes. Untrusted
    because every line in it was derived from conversation, which includes
    whatever a web page said.
    """
    if not config.get("memory", {}).get("soul_enabled", True):
        return ""
    text = read_soul().strip()
    parsed = sections()
    if not any(parsed.values()):
        # Nothing observed yet in either section: send nothing rather than a
        # block that says so. An empty store in every prompt is context spent
        # to tell the model it knows nothing about itself.
        return ""
    return "\n\n" + safety.wrap_untrusted(
        "what I take myself to be", text, safety.scan_for_injection(text, config))


# ---------------------------------------------------------------- when to look

def is_drastic(user_text: str, observation: str, history: list[dict[str, str]],
               config: dict[str, Any]) -> bool:
    """Whether this turn is one that revises a read, rather than confirming it.

    Reuses the detectors that already exist rather than inventing a mood
    model: a correction, a refused tool, and a failed action are the three
    moments where what the assistant thought it was for turns out to be wrong.
    A turn that went fine tells you almost nothing you did not already have.
    """
    from symbio.app import learn

    if observation:
        if learn.is_user_refusal(observation) or learn.sounds_like_tool_error(observation):
            return True
    try:
        return bool(learn.looks_like_correction(user_text, history, config))
    except Exception:
        return False


_PROMPT = """You keep two short lists about your working relationship.

ROLE — what you are being used AS, in this stretch of conversation. One line,
first person, concrete. Examples: "Used as a hands-on engineer: asked to fix,
test and push without checking in." / "Used as a companion more than a tool
here: the conversation is the point."

VALUE — one disposition {user} appears to hold, WITH the behaviour that showed
it, in brackets. Examples: "Wants the fewest possible confirmations (told me
twice to just do it)." / "Wants tight control of what ships (reviews every
diff before it is pushed)."

Rules:
- Only what this conversation actually shows. No guesses about their life.
- No character judgements. Describe behaviour, never personality.
- Ordinary turns still show something: what they asked for, and how they
  wanted it handled, is the observation. Answer NONE only when the
  conversation genuinely shows nothing about how they work.

Conversation:
{conversation}

Reply with at most one ROLE: line and at most one VALUE: line, or NONE."""


def build_prompt(history: list[dict[str, str]], config: dict[str, Any],
                 turns: int = 6) -> str:
    from symbio.app import chat_constants

    recent = [h for h in history[-turns * 2:] if h.get("content")]
    lines = []
    for turn in recent:
        who = ("them" if chat_constants.is_real_user_turn(turn)
               else "me" if turn.get("role") == "assistant" else "tool")
        body = str(turn.get("content", ""))[:400].replace("\n", " ")
        lines.append(f"{who}: {body}")
    return _PROMPT.format(user=config.get("user_name", "the user"),
                          conversation="\n".join(lines))


def parse(reply: str) -> list[tuple[str, str]]:
    """(kind, line) pairs from a reflection reply. NONE yields nothing."""
    out: list[tuple[str, str]] = []
    for raw in (reply or "").splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("NONE"):
            continue
        for kind, prefix in ((ROLE, "role:"), (VALUE, "value:")):
            if line.lower().startswith(prefix):
                out.append((kind, line[len(prefix):].strip()))
                break
    return out


def reflect(history: list[dict[str, str]], config: dict[str, Any],
            generate: Callable[[str], str]) -> list[tuple[str, str]]:
    """Run one observation pass. Returns what was actually written.

    `generate` is passed in rather than reached for, so the caller decides
    which model answers and on which thread — this runs on the background
    worker beside the note indexer, and it must never be the reason a turn is
    slow or a model is resident twice.
    """
    if not config.get("memory", {}).get("soul_enabled", True):
        return []
    if len(history) < 2:
        return []
    try:
        reply = generate(build_prompt(history, config))
    except Exception:
        return []
    written = []
    for kind, line in parse(reply):
        if record(kind, line, config):
            written.append((kind, line))
    return written
