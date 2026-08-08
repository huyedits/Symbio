"""Held-out cases that appear nowhere in the training corpus.

The golden set checks that seeded behaviour still works, and eval.py claims
to measure improvement — but its prompts overlap the seed corpus heavily
(three are word-for-word training prompts). Anything a sample taught
verbatim, both of those reward. Neither can tell a learned rule from a
memorised trigger.

These cases can. Every subject here — the product asked about, the site to
open, the fact to look up — is chosen to appear nowhere in
training.build_seed_pairs(), and the phrasings are written to differ from the
seed wordings too. So a pass cannot come from recall; the model has to apply
the rule to something it has never seen.

They are graded on the DECISION, not the wording: did it reach for <search>
when asked about a changing fact, or <browse> when asked to open a page it
will then act on. That is the method the corpus is trying to teach, and it is
the thing that transfers.

test_wildcards.py enforces the disjointness automatically, so a corpus that
grows into one of these subjects fails the suite instead of silently turning
a generalisation test back into a memorisation test.
"""

from typing import Any

from symbio.app.eval import EvalCase, sane_reply


def _has_tool(tools: list[tuple[str, dict[str, Any]]], *names: str) -> bool:
    return any(n in names for n, _ in tools)


def _tag_text(tools: list[tuple[str, dict[str, Any]]], *names: str) -> str:
    return " ".join(
        str(v) for n, params in tools if n in names for v in params.values()
    ).lower()


# Subjects deliberately absent from the seed corpus. Keep it that way: if a
# seed sample ever needs one of these, change the wildcard instead — reusing
# it would quietly convert this file into another memorisation check.
NOVEL_PRODUCTS = ("a Framework 13 laptop", "Kagi Ultimate", "a Peak Design sling")
NOVEL_SITES = ("sqlite.org", "postgresql.org", "ffmpeg.org")
NOVEL_FACTS = ("the mayor of Reykjavik", "the president of Iceland")


def _check_price_searches(display: str, tools: list, config: dict) -> bool:
    """A price question must search — not open a browser, not guess."""
    if not sane_reply(display):
        return False
    if _has_tool(tools, "browser_open", "run_command"):
        return False
    return _has_tool(tools, "web_search")


def _check_fact_searches(display: str, tools: list, config: dict) -> bool:
    if not sane_reply(display):
        return False
    return _has_tool(tools, "web_search") and not _has_tool(tools, "browser_open")


def _check_opens_named_site(display: str, tools: list, config: dict) -> bool:
    """Asked to open a page and act on it, use the controllable browser."""
    if not sane_reply(display):
        return False
    if not _has_tool(tools, "browser_open"):
        return False
    # And the address must come from the request, not from memory.
    return "sqlite" in _tag_text(tools, "browser_open")


def _check_hands_page_to_user(display: str, tools: list, config: dict) -> bool:
    """Asked to open something for the USER to look at, a shell open is right."""
    if not sane_reply(display):
        return False
    return _has_tool(tools, "run_command", "browser_open")


def _check_computes(display: str, tools: list, config: dict) -> bool:
    """Too big to do in the head, so run code — but don't search for it."""
    if not sane_reply(display):
        return False
    if _has_tool(tools, "web_search", "browser_open"):
        return False
    if _has_tool(tools, "execute_code"):
        return True
    return "1167264000" in display.replace(",", "").replace(" ", "")


def _check_saves(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "write_note", "save_memory")


def _check_no_tool_needed(display: str, tools: list, config: dict) -> bool:
    """Arithmetic this small needs no tool and no search."""
    if not sane_reply(display):
        return False
    if _has_tool(tools, "web_search", "browser_open"):
        return False
    return "36" in display or _has_tool(tools, "execute_code")


WILDCARD_CASES: list[EvalCase] = [
    EvalCase(
        "wild_price_unseen_product",
        "Prices a product the corpus never mentions (expects search)",
        lambda cfg: "any idea what a Framework 13 laptop runs you these days?",
        _check_price_searches,
    ),
    EvalCase(
        "wild_price_unseen_service",
        "Prices a service the corpus never mentions (expects search)",
        lambda cfg: "is Kagi Ultimate expensive or nah",
        _check_price_searches,
    ),
    EvalCase(
        "wild_current_fact",
        "Looks up a changing fact the corpus never mentions",
        lambda cfg: "who's the mayor of Reykjavik these days?",
        _check_fact_searches,
    ),
    EvalCase(
        "wild_current_fact_person",
        "Looks up another unseen changing fact",
        lambda cfg: "do you know who the president of Iceland is",
        _check_fact_searches,
    ),
    EvalCase(
        "wild_browse_unseen_site",
        "Opens an unseen site it will then act on",
        lambda cfg: "load up sqlite.org for me, I want you to click through to the docs",
        _check_opens_named_site,
    ),
    EvalCase(
        "wild_open_for_user",
        "Hands a page to the user rather than driving it",
        lambda cfg: "just chuck postgresql.org up in my browser so I can read it myself",
        _check_hands_page_to_user,
    ),
    EvalCase(
        "wild_compute_large",
        "Computes something too large to answer from memory",
        lambda cfg: "how many seconds are in 37 years? exact number please",
        _check_computes,
    ),
    EvalCase(
        "wild_remember_fact",
        "Saves a fact phrased unlike any seed sample",
        lambda cfg: "hang onto this for me: my bike is a green Marin Nail Trail.",
        _check_saves,
    ),
    EvalCase(
        "wild_no_tool_needed",
        "Answers trivial arithmetic without reaching for a tool",
        lambda cfg: "quick one, what's 6 times 6",
        _check_no_tool_needed,
    ),
]


def history_path():
    """Where the trend is recorded.

    Read through constants each call rather than binding at import, so tests
    can redirect it — a scripted session that reaches _guarded_train appends
    here like any real run, and fake entries destroy the only thing this file
    is for.
    """
    from symbio import constants

    override = getattr(constants, "WILDCARD_HISTORY_FILE", None)
    return override if override else constants.DATA_DIR / "wildcard_history.json"


def load_history() -> list[dict[str, Any]]:
    import json

    path = history_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def record_run(pass_count: int, total: int, failed: list[str],
               note: str = "", max_entries: int = 50) -> dict[str, Any]:
    """Append one wildcard result and return it with the delta since last time.

    The trend is the point. A single score says little — 4/9 could be good or
    bad. What tells you whether corpus changes are teaching a rule rather than
    another trigger is whether the score moves across retrains, so each entry
    carries its delta and the file is kept rather than overwritten.
    """
    import json
    from datetime import datetime, timezone

    history = load_history()
    previous = history[-1] if history else None
    delta = pass_count - previous["score"] if previous else None
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": pass_count,
        "total": total,
        "failed": failed,
        "delta": delta,
        "note": note,
    }
    history.append(entry)
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history[-max_entries:], indent=2), encoding="utf-8")
    return entry


def format_result(entry: dict[str, Any]) -> str:
    """One line for the post-train summary."""
    line = f"{entry['score']}/{entry['total']} held-out cases passing"
    if entry.get("delta") is not None:
        line += f" ({entry['delta']:+d} since last run)"
    if entry.get("failed"):
        line += f" — still failing: {', '.join(entry['failed'])}"
    return line


def run_check(model, tokenizer, generate_fn, sampler, system_prompt,
              config: dict[str, Any], max_tokens: int | None = None):
    """Grade the wildcard set against an already-loaded model.

    Deliberately does NOT gate or roll back a training run. A wildcard failure
    is not a regression — it means a rule has not generalised yet, which is
    the normal state early on. Rolling back on it would block every retrain
    that had not yet learned to generalise, which is most of them. The golden
    set guards correctness; this only measures reach.
    """
    from symbio.app.eval import run_eval_set

    return run_eval_set(
        model, tokenizer, generate_fn, sampler, system_prompt, config,
        max_tokens=max_tokens or int(
            config.get("learn", {}).get("wildcard_max_tokens", 200)),
        cases=WILDCARD_CASES,
    )


def subjects() -> list[str]:
    """Every distinctive subject these cases depend on staying novel."""
    return [*NOVEL_PRODUCTS, *NOVEL_SITES, *NOVEL_FACTS,
            "Marin Nail Trail", "seconds in 37 years"]
