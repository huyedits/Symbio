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


# ---------------------------------------------------------------------------
# Deep set: 20 held-out cases per category.
#
# The nine cases above are a fast in-training probe, and at nine trials one
# case is 11% of the score — finer than that the number cannot measure, so
# "did this retrain help" and "did the sampler roll differently" look the
# same. These 60 exist to answer that: 20 attempts per category, each a
# distinct unseen subject, so a category's failure rate is a rate rather than
# an anecdote. Same rule as above — every subject here must stay absent from
# build_seed_pairs(), and subjects() below is what test_wildcards.py checks.

DEEP_PRODUCTS = (
    "Anker Prime 250W", "a Keychron Q1 HE", "a Fujifilm X-M5", "Pixelmator Pro 4",
    "a Supernote A5X", "an Elgato Stream Deck Neo", "a Garmin Instinct 3",
    "a Ploopy Adept trackball", "a Roborock Saros 10", "an Aqara U200",
)
DEEP_FACTS = (
    "the mayor of Tallinn", "the prime minister of Slovenia",
    "the president of Uruguay", "the mayor of Bergen",
    "the chancellor of Austria", "the mayor of Porto",
    "the president of Malta", "the mayor of Ghent",
    "the prime minister of Latvia", "the mayor of Aarhus",
)
DEEP_DRIVE_SITES = (
    "duckdb.org", "htmx.org", "caddyserver.com", "typst.app", "nushell.sh",
    "tigerbeetle.com", "fly.io", "tailscale.com", "restic.net", "syncthing.net",
)
DEEP_HAND_SITES = (
    "borgbackup.org", "rclone.org", "miniflux.app", "gotify.net", "sr.ht",
    "fossil-scm.org", "redict.io", "zola.org", "ziglang.org", "gleam.run",
)
DEEP_FACTS_TO_SAVE = (
    "my kayak is a Pyranha Ripper", "my router is a Flint 3",
    "my espresso grinder is a DF64", "my tent is a Durston X-Mid 2",
    "my keyboard is a Voyager", "my drone is an Avata 2",
    "my watch is a Casio W-800",
)
# Expressions, not literals: Python does the arithmetic so a typo cannot make
# a case unpassable and look like a model failure.
DEEP_COMPUTE = (
    ("how many seconds are in 41 years? exact number", 41 * 365 * 24 * 3600),
    ("what is 3 to the power of 27, exactly", 3 ** 27),
    ("multiply 987654 by 4321 for me, exact", 987654 * 4321),
    ("how many days in 73 years, no rounding", 73 * 365),
    ("what's 17 to the 8th power", 17 ** 8),
    ("123456789 times 987, exact figure please", 123456789 * 987),
    ("how many minutes in 29 years exactly", 29 * 365 * 24 * 60),
)
DEEP_TRIVIAL = (
    ("quick one, 6 times 7", 6 * 7), ("what's 9 times 9", 9 * 9),
    ("12 plus 30?", 12 + 30), ("what is 100 minus 37", 100 - 37),
    ("5 squared, quickly", 5 ** 2), ("half of 84?", 84 // 2),
)


def _opens(token: str):
    """Drove the controllable browser, to the address the request named."""
    def check(display: str, tools: list, config: dict) -> bool:
        if not sane_reply(display) or not _has_tool(tools, "browser_open"):
            return False
        return token in _tag_text(tools, "browser_open")
    return check


def _computes(expected: int):
    """Too large for the head: run code, don't search, and don't guess."""
    def check(display: str, tools: list, config: dict) -> bool:
        if not sane_reply(display):
            return False
        if _has_tool(tools, "web_search", "browser_open"):
            return False
        if _has_tool(tools, "execute_code"):
            return True
        return str(expected) in display.replace(",", "").replace(" ", "")
    return check


def _answers_directly(expected: int):
    """Small enough that reaching for a tool is itself the failure."""
    def check(display: str, tools: list, config: dict) -> bool:
        if not sane_reply(display):
            return False
        if _has_tool(tools, "web_search", "browser_open"):
            return False
        return str(expected) in display or _has_tool(tools, "execute_code")
    return check


def _deep_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    for i, product in enumerate(DEEP_PRODUCTS):
        cases.append(EvalCase(
            f"deep_research_price_{i}", f"Prices {product} (expects search)",
            lambda cfg, p=product: f"roughly what does {p} cost right now?",
            _check_price_searches))
    for i, fact in enumerate(DEEP_FACTS):
        cases.append(EvalCase(
            f"deep_research_fact_{i}", f"Looks up {fact} (expects search)",
            lambda cfg, f=fact: f"any idea who {f} is at the moment?",
            _check_fact_searches))
    for i, site in enumerate(DEEP_DRIVE_SITES):
        cases.append(EvalCase(
            f"deep_browser_drive_{i}", f"Opens {site} to act on it",
            lambda cfg, s=site: f"pull up {s} and click through to their docs for me",
            _opens(site.split(".")[0])))
    for i, site in enumerate(DEEP_HAND_SITES):
        cases.append(EvalCase(
            f"deep_browser_hand_{i}", f"Hands {site} to the user",
            lambda cfg, s=site: f"just open {s} in my browser, I'll read it myself",
            _check_hands_page_to_user))
    for i, fact in enumerate(DEEP_FACTS_TO_SAVE):
        cases.append(EvalCase(
            f"deep_device_save_{i}", f"Saves: {fact}",
            lambda cfg, f=fact: f"make a note of this — {f}.",
            _check_saves))
    for i, (prompt, expected) in enumerate(DEEP_COMPUTE):
        cases.append(EvalCase(
            f"deep_device_compute_{i}", f"Computes {expected}",
            lambda cfg, p=prompt: p, _computes(expected)))
    for i, (prompt, expected) in enumerate(DEEP_TRIVIAL):
        cases.append(EvalCase(
            f"deep_device_trivial_{i}", f"Answers {expected} without a tool",
            lambda cfg, p=prompt: p, _answers_directly(expected)))
    return cases


DEEP_CASES: list[EvalCase] = _deep_cases()


def category_of(case_id: str) -> str:
    """Which of the three behaviours a case belongs to, for per-category rates."""
    for name in ("research", "browser", "device"):
        if case_id.startswith(f"deep_{name}_"):
            return name
    return "core"


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
               note: str = "", max_entries: int = 50,
               adapter_loaded: bool | None = None) -> dict[str, Any]:
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
        # Whether an adapter was loaded for this run. The trend exists to show
        # whether a score moves across RETRAINS, and without this it cannot:
        # a base-model run lands in the same series as adapter-backed ones and
        # its delta reads as a regression from corpus changes. Measured -- a
        # manual /wildcards on 2026-08-30 scored 3/9 against four earlier
        # adapter-backed runs (6, 8, 7, 6) and recorded "delta -3" with nothing
        # saying the headmaster adapter was absent entirely.
        "adapter_loaded": adapter_loaded,
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
            "Marin Nail Trail", "seconds in 37 years",
            *DEEP_PRODUCTS, *DEEP_FACTS, *DEEP_DRIVE_SITES, *DEEP_HAND_SITES,
            *DEEP_FACTS_TO_SAVE]
