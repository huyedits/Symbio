"""What to say to a model that has just failed again, and again after that.

The harness already refuses a repeated call and names the tools that have not
been tried. That is the right answer to the SECOND failure and the wrong answer
to the fifth: by then the problem is rarely which tool, it is something the
model believes that is not true — the file it is sure it read, the value it
"decoded" by eye, the page it thinks is open. Telling it to pick a different
tool at that point just moves the same wrong belief onto a new tool.

So the pressure escalates, and what it attacks moves inward:

    1  the call        use the error you just got
    2  the approach    that is the same attempt; here is what you have not tried
    3  the assumptions name what this rests on and test the weakest one
    4  the evidence    you may be wrong about what you SAW; go and re-read it
    5  the method      solve it as if the tool you keep reaching for didn't exist
    6  the problem     you may be solving the wrong problem

Each rung is a different question, not a louder version of the last one. A
model that is asked "try something else" six times answers it six times the
same way; asked "which of your assumptions is doing the most work here, and how
would you check it?", it has to produce something it has not produced yet.

The ladder does not end. Once the last rung is reached it keeps being served,
because the alternative — running out of things to say and falling through to a
stop — is exactly the giving-up this exists to prevent. What bounds the turn is
the round budget and the per-family budgets in chat_turn, which are counted in
work actually attempted; nothing here decides to stop, and that separation is
deliberate. Giving up is a budget decision, never a rhetorical one.

The one thing every rung says is the same: an honest "here is what blocked me"
is always available and is never a failure. Pressure to keep trying, with no
way out that isn't success, is pressure to fabricate one — and a fabricated
completion is the single most expensive thing this model does.
"""

from __future__ import annotations

from typing import NamedTuple


class Challenge(NamedTuple):
    """One rung: what it attacks, and the words that attack it."""
    name: str
    # What the rung is aimed at, for the status line the user sees.
    target: str
    body: str


# The escalation. `{untried}` is filled with a sentence naming tools not yet
# reached for, or dropped when there are none left to name.
LADDER: tuple[Challenge, ...] = (
    Challenge(
        "retry", "the call",
        "That failed. Read the error itself rather than your expectation of "
        "it, change the one thing it actually complains about, and try again."
    ),
    Challenge(
        "approach", "the approach",
        "That is the same attempt again, and it will fail the same way. "
        "Change something real — a different tool, a different target, a value "
        "you actually read rather than assumed.{untried}"
    ),
    Challenge(
        "assumptions", "your assumptions",
        "Stop trying to make this approach work and attack your own reasoning "
        "instead. This attempt rests on things you have assumed: that the path "
        "exists, that the format is what you think, that the last step did what "
        "it reported. State the assumption doing the most work here, say how "
        "you would check it in ONE call, and make that call — not another "
        "attempt at the same thing.{untried}"
    ),
    Challenge(
        "evidence", "what you think you saw",
        "Consider that you are wrong about what you observed. You may be "
        "working from a remembered version of an output rather than the output: "
        "a value you decoded by eye, a file you are sure you read, a page you "
        "believe is open. Go and get the real thing again — read the file, "
        "print the value, re-read the page — and compare it to what you believe "
        "it says. If they differ, that difference is the bug."
    ),
    Challenge(
        "method", "the method",
        "The tool you keep reaching for may simply be the wrong instrument. "
        "Solve this as if it did not exist: what is the completely different "
        "route to the same end? Take it.{untried}"
    ),
    Challenge(
        "problem", "the problem itself",
        "You may be solving the wrong problem. Re-read what was actually asked "
        "and check it against what you have been doing — the goal may be "
        "reachable without the step you are stuck on, or the thing you are "
        "fixing may not be the thing that is broken. Say in one line what you "
        "now think the real task is, then do that."
    ),
)

# Appended to every rung. Persistence without a truthful exit is pressure to
# invent a success, which is worse than stopping.
_HONEST_EXIT = (
    " If you genuinely cannot get further, say so plainly in one line — what "
    "you were after, what you actually got, and the exact error that blocked "
    "you. That is an acceptable answer. Claiming it worked is not."
)


def failure_signature(tool: str, observation: str) -> str:
    """What KIND of failure this was, with the specifics stripped out.

    The persistence gate counts attempts, and it used to count them by their
    arguments — so seven `read_file` calls on seven invented filenames scored
    as seven distinct attempts and the turn was judged well-explored enough to
    stop. Live 2026-09-14, that is exactly what happened: the model lost its
    footing, guessed `page.html`, `top.json`, `clean.json`, got "file not
    found" every time, and the harness read the variety as progress.

    Distinct-by-arguments is not distinct-by-idea. Seven calls that fail the
    same way are one idea tried seven times, and the signature is what says so:
    the tool plus the first few words of the error, with anything carrying a
    digit, a dot or a slash dropped — those are the parts that differ between
    two attempts at the same mistake.
    """
    words: list[str] = []
    for raw in (observation or "").lower().split():
        token = raw.strip(".,:;'\"()[]{}")
        if not token:
            continue
        if any(ch.isdigit() for ch in token) or "/" in token or "." in token:
            continue
        words.append(token)
        if len(words) == 5:
            break
    return f"{tool}:{' '.join(words)}"


def challenge_for(failures: int, untried: list[str] | None = None,
                  attempts: int = 0) -> Challenge:
    """The rung for a turn that has now failed `failures` times.

    Clamps to the last rung rather than running out: see the module docstring.
    `failures` is 1-based — the first failure gets the first rung.
    """
    index = max(0, min(int(failures) - 1, len(LADDER) - 1))
    rung = LADDER[index]
    untried_text = ""
    if untried:
        untried_text = (" Not tried yet this turn: "
                        + ", ".join(untried[:8]) + ".")
    body = rung.body.replace("{untried}", untried_text) + _HONEST_EXIT
    if attempts:
        body = (f"{attempts} distinct attempt(s) so far this turn; repeating "
                f"one does not make another. " + body)
    return rung._replace(body=body)


def observation(failures: int, untried: list[str] | None = None,
                attempts: int = 0, preamble: str = "") -> str:
    """The rung, as the `[System observation: ...]` the model actually reads."""
    rung = challenge_for(failures, untried, attempts)
    lead = f"{preamble.strip()} " if preamble else ""
    return f"[System observation: {lead}{rung.body}]"
