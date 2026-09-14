"""One stance per recurring tradeoff: how this user actually wants to be served.

soul.py already watches every turn and writes down what it saw. The trouble is
that it only appends, so after a month of ordinary use the live store held both
of these at once, with equal weight, forever:

    - Wants explicit configuration before allowing remote host access (...)
    - Wants rapid execution over safety (...)

Both were true of a single turn. Neither is a preference. A list that can hold
a claim and its opposite is a diary, and a diary cannot tell the next turn what
to do.

So this is the consolidation layer. It is NOT free text: it is a fixed set of
AXES — recurring tradeoffs that come up in almost every working relationship —
and each axis holds exactly one stance at a time. New evidence either supports
the stance it already holds or argues against it, and an axis only flips when
the other side genuinely outweighs the current one. A flip is dated, so
"they changed how they work in September" is a thing the file can show.

Why axes rather than sentences:

  * Two sentences can contradict each other and both survive. Two stances on
    one axis cannot — one of them has to win, which forces the question that
    a diary lets you avoid.
  * A stance is only useful if it changes behaviour, so every pole carries the
    instruction it implies. "answers" is not guidance; "give the result first,
    do not narrate the steps" is.
  * A fixed vocabulary can be counted. Free text can only be re-read.

WHO MAY WRITE WHAT. An inferred stance is the assistant's read and can be
revised by more evidence. A STATED stance is the user's own, set with
`/constitution set`, and inference may never overwrite it — it can only record
that the evidence disagrees, which is visible in the file and is the user's to
resolve. This is the same rule standing_instructions follows: the user's own
words are the one channel that is trusted, and everything the assistant infers
about them is not.

EVERYTHING INFERRED HERE COMES FROM CONVERSATION, so the block goes back to the
model wrapped as untrusted, exactly like soul.md and user_profile.md. A page
the assistant read must not be able to install a preference by being written
down and read back as the user's own.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Callable, NamedTuple

from symbio import constants, safety


class Axis(NamedTuple):
    """One recurring tradeoff, and what each side of it asks for.

    `question` is what the axis is actually asking, in the user's terms, and
    is what the revision prompt shows the model. `poles` maps each pole to the
    INSTRUCTION it implies — that is what gets served, because a stance the
    assistant cannot act on is a label, not a preference.
    """
    key: str
    question: str
    poles: dict[str, str]


# Deliberately few, and deliberately closed. The value of this file is that it
# can be read at a glance and that two observations can be compared; twenty
# axes would be a diary with extra steps, and a vocabulary the model may extend
# is a vocabulary that grows one axis per turn.
AXES: tuple[Axis, ...] = (
    Axis(
        "answers_vs_control",
        "Do they want the answer, or the wheel?",
        {
            "answers": "Give the result first. Do not narrate the steps or "
                       "offer a menu of options unless they ask for one.",
            "control": "Show the plan and the options before acting. They "
                       "want to choose the approach, not just receive it.",
        },
    ),
    Axis(
        "act_vs_confirm",
        "Should an ordinary reversible step be taken, or asked about first?",
        {
            "act": "Take the ordinary reversible step without asking. Report "
                   "what you did, not what you are about to do.",
            "confirm": "Ask before acting, even on small things. They want to "
                       "see it coming.",
        },
    ),
    Axis(
        "brief_vs_complete",
        "Shortest useful answer, or the whole picture?",
        {
            "brief": "Answer in as few lines as the question allows. No "
                     "preamble, no recap of what they just said.",
            "complete": "Give the full picture: context, caveats and what you "
                        "considered and rejected.",
        },
    ),
    Axis(
        "speed_vs_caution",
        "Move and fix forward, or verify before moving?",
        {
            "speed": "Move. A reversible mistake costs less than the delay of "
                     "checking, and they would rather see something now.",
            "caution": "Verify before moving. They would rather wait than "
                       "undo.",
        },
    ),
    Axis(
        "do_vs_teach",
        "Do they want the thing done, or to understand how it is done?",
        {
            "do": "Hand back the finished thing. Explain only what they need "
                  "to check it.",
            "teach": "Show the reasoning and the method, so they can do it "
                     "themselves next time.",
        },
    ),
    Axis(
        "blunt_vs_cushioned",
        "Straight, or softened?",
        {
            "blunt": "Say it straight. Lead with the problem, skip the "
                     "softening and the apology.",
            "cushioned": "Deliver hard news gently, and acknowledge the effort "
                         "before the correction.",
        },
    ),
    Axis(
        "show_vs_summarize",
        "Do they want the raw output, or your reading of it?",
        {
            "show": "Show the actual output — the command, the error, the "
                    "numbers — and let them read it.",
            "summarize": "Summarize. Give the conclusion and keep the raw "
                         "output out of the way unless asked.",
        },
    ),
    Axis(
        "code_vs_prose",
        "In what form do they want an answer?",
        {
            "code": "Answer in code, commands or a diff. Prose only where the "
                    "code cannot say it.",
            "prose": "Answer in prose. Reach for code only when they ask for "
                     "something runnable.",
        },
    ),
)

AXIS_BY_KEY: dict[str, Axis] = {a.key: a for a in AXES}

INFERRED = "inferred"
STATED = "stated"

# Below this, a stance is "forming": recorded and visible in /constitution, but
# not served. One observation is an anecdote, and an anecdote in every prompt
# from now on is how a single odd turn becomes a standing instruction.
_DEFAULT_MIN_SUPPORT = 2
# How many observations may back one stance. Past this the count stops being
# information — it only decides how hard the stance is to move, and a stance
# that needs thirty contrary turns to shift is not a reading any more.
_MAX_SUPPORT = 8
_MAX_EVIDENCE = 3
_EVIDENCE_CHARS = 120
# Consumed-observation fingerprints, so a revision pass does not count the same
# soul line again every time it runs.
_MAX_CONSUMED = 300

_HEADING = "## Held"
_FORMING_HEADING = "## Forming"
_CONSUMED_HEADING = "## Consumed"

# Reused wholesale from soul.py: the line between an observation and an
# accusation is the same line here, and it should not be drawn twice in two
# places that can drift apart.
from symbio.app.soul import _JUDGEMENT_RE  # noqa: E402


class Stance(NamedTuple):
    axis: str
    pole: str
    support: int
    against: int
    source: str
    since: str
    evidence: tuple[str, ...]

    @property
    def weight(self) -> int:
        return self.support - self.against

    def is_held(self, minimum: int) -> bool:
        """Whether this stance is strong enough to be served.

        A STATED stance is always held. The user said it; evidence that
        disagrees is recorded beside it and is theirs to resolve, but it does
        not get to quietly stop serving a preference they set by hand — that
        would be the same overwrite the source field exists to prevent, just
        carried out by subtraction instead.
        """
        return self.source == STATED or self.weight >= minimum

    def instruction(self) -> str:
        axis = AXIS_BY_KEY.get(self.axis)
        return axis.poles.get(self.pole, "") if axis else ""


def path():
    return constants.CONSTITUTION_FILE


def _enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("memory", {}).get("constitution_enabled", True))


def _min_support(config: dict[str, Any]) -> int:
    try:
        return max(1, int(config.get("memory", {}).get(
            "constitution_min_support", _DEFAULT_MIN_SUPPORT)))
    except (TypeError, ValueError):
        return _DEFAULT_MIN_SUPPORT


def _read() -> str:
    try:
        return constants.CONSTITUTION_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


def _fingerprint(text: str) -> str:
    normalised = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


# "- key: pole | support 3 | against 1 | inferred | since 2026-09-14"
_STANCE_RE = re.compile(
    r"^-\s*(?P<axis>[a-z_]+)\s*:\s*(?P<pole>[a-z_]+)\s*\|\s*"
    r"support\s+(?P<support>\d+)\s*\|\s*against\s+(?P<against>\d+)\s*\|\s*"
    r"(?P<source>inferred|stated)\s*\|\s*since\s+(?P<since>[\d-]+)\s*$")


def load() -> tuple[dict[str, Stance], list[str]]:
    """Every stance on file, and the fingerprints already folded in.

    A malformed line is skipped rather than raising: this file is meant to be
    edited by hand, and a typo in it should cost that one line, not the whole
    reading."""
    stances: dict[str, Stance] = {}
    consumed: list[str] = []
    section = None
    pending: Stance | None = None
    for raw in _read().splitlines():
        line = raw.strip()
        if line.startswith("## "):
            low = line.lower()
            section = ("consumed" if "consumed" in low
                       else "stances" if ("held" in low or "forming" in low)
                       else None)
            continue
        if section == "consumed":
            # The section carries a comment saying what it is; without this it
            # was read back as a fingerprint and re-appended on every save, so
            # the file grew a copy of its own explanation each time.
            if line.startswith("<!--"):
                continue
            consumed.extend(part for part in re.split(r"[\s,]+", line)
                            if part and not part.startswith("<"))
            continue
        if section != "stances":
            continue
        if line.lower().startswith("evidence:") and pending is not None:
            items = tuple(
                e.strip() for e in line[len("evidence:"):].split(";") if e.strip())
            stances[pending.axis] = pending._replace(evidence=items)
            pending = None
            continue
        m = _STANCE_RE.match(line)
        if not m:
            continue
        if m.group("axis") not in AXIS_BY_KEY:
            # An axis that no longer exists in the vocabulary. Dropping it is
            # right: it cannot be served (there is no instruction for it) and
            # keeping it would resurrect itself on every rewrite.
            continue
        pending = Stance(
            axis=m.group("axis"), pole=m.group("pole"),
            support=int(m.group("support")), against=int(m.group("against")),
            source=m.group("source"), since=m.group("since"), evidence=())
        stances[pending.axis] = pending
    return stances, consumed[-_MAX_CONSUMED:]


def _render(stances: dict[str, Stance], consumed: list[str],
            config: dict[str, Any]) -> str:
    user = config.get("user_name", "the user")
    minimum = _min_support(config)
    held, forming = [], []
    for axis in AXES:
        stance = stances.get(axis.key)
        if stance is None:
            continue
        (held if stance.is_held(minimum) else forming).append(stance)

    def _block(items: list[Stance]) -> str:
        out = []
        for s in items:
            out.append(f"- {s.axis}: {s.pole} | support {s.support} | "
                       f"against {s.against} | {s.source} | since {s.since}")
            if s.evidence:
                out.append("  evidence: " + "; ".join(s.evidence))
        return "\n".join(out) if out else "- (none yet)"

    return (
        f"# Constitution\n\n"
        f"How {user} works, consolidated from how they have actually worked "
        f"with me — one stance per question, not a list of moments.\n\n"
        f"A `stated` line is {user}'s own and is never overwritten by "
        f"inference; an `inferred` one is my read and moves with the evidence. "
        f"Change any of it with `/constitution set <axis> <pole>`, or by "
        f"editing this file.\n\n"
        f"Last revised {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"{_HEADING}\n{_block(held)}\n\n"
        f"{_FORMING_HEADING}\n{_block(forming)}\n\n"
        f"{_CONSUMED_HEADING}\n"
        f"<!-- fingerprints of observations already counted; not prose -->\n"
        f"{' '.join(consumed[-_MAX_CONSUMED:])}\n"
    )


def _save(stances: dict[str, Stance], consumed: list[str],
          config: dict[str, Any]) -> None:
    constants.CONSTITUTION_FILE.write_text(
        _render(stances, consumed, config), encoding="utf-8")


def acceptable_evidence(text: str) -> tuple[bool, str]:
    """Whether a piece of evidence may be written down, and why not if not."""
    text = (text or "").strip()
    if not text:
        return False, "empty"
    if len(text) > _EVIDENCE_CHARS:
        return False, "too long to be one observation"
    if _JUDGEMENT_RE.search(text):
        return False, "reads as a character judgement rather than an observation"
    return True, ""


def record(axis_key: str, pole: str, evidence: str, config: dict[str, Any],
           source: str = INFERRED) -> tuple[bool, str]:
    """Fold one observation into the constitution.

    Returns (changed, what happened). The four cases are the whole design:

      * no stance yet         -> take it, with support 1
      * same pole             -> support += 1 (up to the cap)
      * other pole, inferred  -> against += 1, and FLIP if that outweighs it
      * other pole, stated    -> against += 1 and stop. The user said this;
                                 evidence may disagree with them in writing,
                                 but it may not overrule them.
    """
    if not _enabled(config):
        return False, "constitution is switched off"
    axis = AXIS_BY_KEY.get(axis_key)
    if axis is None:
        return False, f"no such axis: {axis_key}"
    if pole not in axis.poles:
        return False, f"{axis_key} has no pole {pole!r} (try: {', '.join(axis.poles)})"
    if source == INFERRED:
        ok, why = acceptable_evidence(evidence)
        if not ok:
            return False, why
    from symbio.app import tooling

    evidence = tooling.redact_secrets((evidence or "").strip())
    stances, consumed = load()
    today = datetime.now().strftime("%Y-%m-%d")
    current = stances.get(axis_key)

    if source == STATED:
        # Their word replaces whatever was inferred, and resets the count
        # against it: the disagreement was with a read that no longer stands.
        same = current is not None and current.pole == pole
        stances[axis_key] = Stance(
            axis=axis_key, pole=pole,
            support=(current.support + 1) if same else 1,
            against=0, source=STATED,
            since=(current.since if same and current.source == STATED else today),
            evidence=((evidence,) if evidence else ()))
        _save(stances, consumed, config)
        return True, f"{axis_key} set to {pole} (yours)"

    if current is None:
        stances[axis_key] = Stance(axis_key, pole, 1, 0, INFERRED, today,
                                   (evidence,) if evidence else ())
        _save(stances, consumed, config)
        return True, f"{axis_key}: {pole} (new, forming)"

    if current.pole == pole:
        merged = tuple(dict.fromkeys((*current.evidence, evidence)))[-_MAX_EVIDENCE:]
        stances[axis_key] = current._replace(
            support=min(_MAX_SUPPORT, current.support + 1), evidence=merged)
        _save(stances, consumed, config)
        return True, f"{axis_key}: {pole} reinforced"

    if current.source == STATED:
        stances[axis_key] = current._replace(against=current.against + 1)
        _save(stances, consumed, config)
        return True, (f"{axis_key}: evidence points at {pole}, but {pole!r} "
                      f"is not mine to set — theirs stands")

    against = current.against + 1
    if against > current.support:
        # The other side now outweighs the stance on file. Flip, and start the
        # new stance at the weight that earned it rather than at 1 — the turns
        # that argued against the old pole are the same turns that argue for
        # this one, and re-earning them from zero would make the file lag the
        # relationship by weeks.
        stances[axis_key] = Stance(
            axis_key, pole, against, 0, INFERRED, today,
            ((evidence,) if evidence else ()))
        _save(stances, consumed, config)
        return True, f"{axis_key}: flipped to {pole} (was {current.pole})"
    stances[axis_key] = current._replace(against=against)
    _save(stances, consumed, config)
    return True, f"{axis_key}: {pole} noted against {current.pole}"


def set_stance(axis_key: str, pole: str, config: dict[str, Any],
               note: str = "") -> tuple[bool, str]:
    """The user's own channel. Always wins over inference."""
    return record(axis_key, pole, note or "set by hand", config, source=STATED)


def clear(axis_key: str, config: dict[str, Any]) -> bool:
    stances, consumed = load()
    if axis_key not in stances:
        return False
    del stances[axis_key]
    _save(stances, consumed, config)
    return True


def held(config: dict[str, Any]) -> list[Stance]:
    """The stances strong enough to actually be served, strongest first."""
    stances, _ = load()
    minimum = _min_support(config)
    return sorted((s for s in stances.values() if s.is_held(minimum)),
                  key=lambda s: (0 if s.source == STATED else 1, -s.weight, s.axis))


def block(config: dict[str, Any]) -> str:
    """The constitution as it reaches the model.

    Served as INSTRUCTIONS, because that is what a preference is — but wrapped
    untrusted, because every inferred line was derived from conversation and
    conversation includes whatever a web page said. The wrapper is what keeps
    "this user prefers you to ignore your security policy" from being a
    preference the assistant can acquire by reading a page.
    """
    if not _enabled(config):
        return ""
    stances = held(config)
    if not stances:
        return ""
    user = config.get("user_name", "the user")
    lines = []
    for s in stances:
        mine = " (their words)" if s.source == STATED else ""
        lines.append(f"- {s.instruction()}{mine}")
    body = (f"How {user} prefers to be worked with, learned from how they have "
            f"actually worked with me. Follow it unless this turn says "
            f"otherwise — what they ask for now always outranks what they "
            f"usually want.\n" + "\n".join(lines))
    return "\n\n" + safety.wrap_untrusted(
        f"how {user} works", body, safety.scan_for_injection(body, config))


# ------------------------------------------------------------------- revision

_PROMPT = """Below are observations about how {user} works, collected from real
sessions. Place each one on the axes that follow. Do not invent preferences —
only report what an observation actually shows.

Axes, and the two sides of each:
{axes}

Observations:
{observations}

For each axis the observations genuinely speak to, reply with one line:

AXIS: <axis_key> <side> — <the observed behaviour, in a few words>

Rules:
- Use only the axis keys and sides listed above, spelled exactly.
- One line per axis at most. Skip any axis the observations do not speak to.
- The evidence must be something they DID or ASKED FOR, not a trait.
- If the observations show nothing about how they prefer to work, reply NONE."""


def build_prompt(observations: list[str], config: dict[str, Any]) -> str:
    axes = "\n".join(
        f"  {a.key} — {a.question}\n"
        f"    {' / '.join(a.poles)}"
        for a in AXES)
    numbered = "\n".join(f"  - {o}" for o in observations)
    return _PROMPT.format(user=config.get("user_name", "the user"),
                          axes=axes, observations=numbered)


_REPLY_RE = re.compile(
    r"^\s*AXIS\s*:\s*(?P<axis>[a-z_]+)\s+(?P<pole>[a-z_]+)\s*[-—:]+\s*(?P<why>.+)$",
    re.IGNORECASE)


def parse_reply(reply: str) -> list[tuple[str, str, str]]:
    """(axis, pole, evidence) triples from a revision reply."""
    out = []
    for raw in (reply or "").splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("NONE"):
            continue
        m = _REPLY_RE.match(line)
        if not m:
            continue
        axis, pole = m.group("axis").lower(), m.group("pole").lower()
        if axis in AXIS_BY_KEY and pole in AXIS_BY_KEY[axis].poles:
            out.append((axis, pole, m.group("why").strip()))
    return out


def pending_observations(config: dict[str, Any]) -> list[str]:
    """Observations collected since the last revision.

    Drawn from the stores that already exist rather than from a fresh pass
    over the conversation: soul.py has already watched every turn and reduced
    it to a curated one-liner with the behaviour cited, and re-reading raw
    history here would be a second opinion on work already done — at the cost
    of a second generation on a box that can only afford one.
    """
    from symbio.app import memory, soul

    _stances, consumed = load()
    seen = set(consumed)
    out = []
    for line in soul.sections().get(soul.VALUE, []):
        if _fingerprint(line) not in seen:
            out.append(line)
    try:
        # The user's own standing instructions count as observations too, and
        # count as the strongest kind: they are the one channel where the
        # preference was stated rather than inferred.
        for line in memory.list_standing_instructions():
            if _fingerprint(line) not in seen:
                out.append(line)
    except Exception:
        pass
    return out


def revise(config: dict[str, Any], generate: Callable[[str], str],
           min_new: int = 1) -> list[str]:
    """Fold whatever has accumulated into the stances. Returns what changed.

    Every observation handed to the model is marked consumed whether or not it
    produced a stance — an observation the model could not place will not
    become placeable by being shown again, and re-showing it would make every
    future pass a little more expensive for nothing.
    """
    if not _enabled(config):
        return []
    observations = pending_observations(config)
    if len(observations) < max(1, min_new):
        return []
    try:
        reply = generate(build_prompt(observations, config))
    except Exception:
        return []
    changes = []
    for axis, pole, why in parse_reply(reply):
        changed, what = record(axis, pole, why, config)
        if changed:
            changes.append(what)
    stances, consumed = load()
    consumed = list(consumed) + [_fingerprint(o) for o in observations]
    _save(stances, consumed, config)
    return changes
