"""Open goals: what the user wants worked toward across turns.

A small model hands the turn back after one step and has forgotten the larger
aim a few turns later — the context is short and the aim was said once. The
goals file is the aim, re-read every turn: listed at the end of the system
prompt, and handed to the "am I finished?" check (chat_turn._judges_unfinished)
so "done" means done against the goal, not against the last sentence.

Only the user writes it, through /goal. The model has no tool that adds or
edits a goal, which is what lets the block be served like the standing
instructions — as the user's own words — rather than wrapped as untrusted
data: a goal the model could write would be an instruction channel it could
write to itself. It can only say a goal looks done; ticking it off is /goal
done N.
"""

from __future__ import annotations

from datetime import datetime

from symbio import constants

LIMIT = 8                 # goals kept; more than this is a backlog, not a focus
CHAR_LIMIT = 300          # per goal


def _path():
    return constants.GOALS_FILE


def list_goals() -> list[str]:
    p = _path()
    if not p.exists():
        return []
    return [ln[2:].strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.startswith("- ") and ln[2:].strip()]


def _write(goals: list[str]) -> None:
    p = _path()
    if goals:
        p.write_text("".join(f"- {g}\n" for g in goals), encoding="utf-8")
    elif p.exists():
        p.unlink()


def add_goal(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "Say what the goal is: /goal <what you want done>."
    if len(text) > CHAR_LIMIT:
        return f"Keep a goal under {CHAR_LIMIT} characters; this one is {len(text)}."
    goals = list_goals()
    if any(g.split(" (set ")[0].lower() == text.lower() for g in goals):
        return f"Already a goal: {text}."
    if len(goals) >= LIMIT:
        return f"{LIMIT} goals already open. Finish or drop one first (/goal done N)."
    goals.append(f"{text} (set {datetime.now():%Y-%m-%d})")
    _write(goals)
    return f"Goal {len(goals)} set: {text}."


def done_goal(n: int) -> str:
    goals = list_goals()
    if not 1 <= n <= len(goals):
        return f"No goal {n}; there are {len(goals)}."
    gone = goals.pop(n - 1)
    _write(goals)
    return f"Done: {gone.split(' (set ')[0]}."


def clear_goals() -> str:
    count = len(list_goals())
    _write([])
    return f"Cleared {count} goal(s)." if count else "No goals to clear."


def goals_block(user_name: str = "the user") -> str:
    """The open goals for the system prompt, or "" when there are none."""
    goals = list_goals()
    if not goals:
        return ""
    lines = "\n".join(f"{i}. {g.split(' (set ')[0]}" for i, g in enumerate(goals, 1))
    return (f"\n\n## Open goals {user_name} set\n"
            f"Work toward these across turns, one concrete step at a time. When "
            f"one looks done, say so and which number — {user_name} ticks it off.\n"
            f"{lines}")
