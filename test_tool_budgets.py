#!/usr/bin/env python3
"""Persistence, and the shape of it: family budgets and distinct attempts.

The failure these exist for is a turn that dies early and quietly. One call,
one error, a paragraph explaining the error, done — with fourteen rounds of
budget unspent and a whole toolset never touched. The two halves of the fix:

  * a repeat is refused IN WRITING, with the untried tools named, instead of
    being dropped on the floor while the model's prose claims it worked;
  * one toolset cannot spend the whole turn's budget, so there are rounds left
    for a different approach when the first one is exhausted.
"""

import builtins
import json

from symbio.app import chat, persistence, tooling
from symbio.app import config as app_config
from test_main_loop import FakeBrowser, ScriptedSession, scratch_notes_dir


# ---------------------------------------------------------------- families

def test_every_catalog_tool_has_a_family():
    """A tool with no family gets the "other" bucket's shared budget and is
    listed under "other" in the index — survivable, but never on purpose."""
    unmapped = [
        t["name"] for t in tooling._TOOLS
        if tooling.tool_family(t["name"]) == "other"
        and not t["name"].startswith("mcp_")
    ]
    assert not unmapped, f"tools with no family: {unmapped}"


def test_a_hermes_alias_resolves_to_its_family():
    """The budget counts what the model called, and the model calls
    "terminal", not "run_command"."""
    assert tooling.tool_family("terminal") == "shell"
    assert tooling.tool_family("cmd") == "shell"
    assert tooling.tool_family("patch") == "file"


def test_untried_tools_leads_with_a_family_not_yet_touched():
    """A model told only "no" has one move left, which is to say the same
    thing again. The list is what makes the refusal actionable — and the first
    suggestion must not be more of what it was already doing."""
    options = tooling.untried_tools({"browser_click", "browser_open"})
    assert options, "no alternatives offered at all"
    assert tooling.tool_family(options[0]) != "browser"
    assert "browser_click" not in options


def test_untried_tools_respects_the_enabled_groups():
    """Suggesting a tool the user switched off sends the model at a wall."""
    options = tooling.untried_tools(set(), groups={"terminal"})
    assert options
    assert all(tooling.tool_group_enabled(n, {"terminal"}) for n in options)


# ------------------------------------------------------------ in the loop

def _session(replies, config=None, **agent_overrides):
    config = config or app_config.load_config()
    config["agent"].update(agent_overrides)
    return ScriptedSession(
        user_inputs=["do the thing", "/quit", "n"],
        model_replies=replies,
        config=config,
    )


def test_a_repeated_failing_call_is_refused_in_writing():
    """Before this, the third identical call was filtered out silently and the
    turn ended on whatever prose sat beside it."""
    with scratch_notes_dir():
        session = _session(
            ["<cmd>definitely-not-a-real-binary</cmd> Trying."] * 5,
            max_tool_rounds=8,
        )
        session.run()
    everything = " ".join(session.prompts_seen)
    assert "already FAILED this turn" in everything, everything[-800:]


def test_the_refusal_names_something_that_has_not_been_tried():
    with scratch_notes_dir():
        session = _session(
            ["<cmd>definitely-not-a-real-binary</cmd> Trying."] * 5,
            max_tool_rounds=8,
        )
        session.run()
    everything = " ".join(session.prompts_seen)
    assert "Not tried yet this turn" in everything


def test_the_challenge_escalates_as_the_failures_pile_up():
    """The whole point of the ladder: the fifth failure must not be answered
    with the same sentence as the second. A model asked "try something else"
    six times answers it the same way six times."""
    with scratch_notes_dir():
        session = _session(
            ["<cmd>definitely-not-a-real-binary</cmd> Trying."] * 10,
            max_tool_rounds=12,
        )
        session.run()
    # The last prompt carries the whole turn's history, so it is where every
    # challenge that was issued can be counted at once.
    transcript = session.prompts_seen[-1]
    rungs = [c.name for c in persistence.LADDER
             if c.body.split("{")[0][:40].strip() in transcript]
    assert len(rungs) >= 2, (rungs, transcript[-1500:])
    # And it moved inward: something past the first rung was reached.
    assert rungs != ["retry"], rungs


def test_persistence_never_runs_out_of_things_to_say():
    """The ladder clamps at its last rung rather than ending. Running out of
    challenges would make the harness stop pushing for a reason that has
    nothing to do with the budget — which is giving up, dressed as a limit."""
    last = persistence.challenge_for(len(persistence.LADDER) + 50)
    assert last.name == persistence.LADDER[-1].name


def test_every_challenge_leaves_an_honest_way_out():
    """Pressure to keep trying, with no acceptable answer but success, is
    pressure to fabricate one — and a fabricated completion is the most
    expensive thing this model does."""
    for n in range(1, len(persistence.LADDER) + 2):
        body = persistence.challenge_for(n).body.lower()
        assert "cannot get further" in body or "blocked you" in body, n
        assert "claiming it worked is not" in body, n


def test_the_turn_still_ends():
    """Persistence is not a loop. What ends the turn is the round budget —
    work attempted — not the harness running out of words."""
    with scratch_notes_dir():
        session = _session(
            ["<cmd>definitely-not-a-real-binary</cmd> Trying."] * 30,
            max_tool_rounds=8,
        )
        session.run()
    assert len(session.prompts_seen) <= 9, len(session.prompts_seen)


def test_the_challenge_budget_can_be_switched_off():
    """0 leaves only the round budget, which is what an install that finds the
    escalation too chatty should get — not a broken loop."""
    session = _session(["done"], max_persistence_challenges=0)
    bound = chat.ChatSession.__new__(chat.ChatSession)
    bound.config = session.config
    assert bound._challenge_budget() == 0
    session.config["agent"]["max_persistence_challenges"] = "nonsense"
    assert bound._challenge_budget() == len(persistence.LADDER)


def test_one_toolset_cannot_spend_the_whole_turn():
    """With a browser budget of 3, the fourth browser call does not run — and
    says so — while rounds remain for a different approach."""
    real_browser = chat.BrowserSession
    chat.BrowserSession = FakeBrowser
    try:
        with scratch_notes_dir():
            config = app_config.load_config()
            config["browser"]["enabled"] = True
            groups = config.setdefault("tools", {}).setdefault("enabled_groups", [])
            if "browser" not in groups:
                groups.append("browser")
            # A different URL every round, so every call is a genuinely
            # DIFFERENT one: the repeat filter never fires and only the family
            # budget can stop this, which is what the test is for.
            replies = [f"<browse>https://example{i}.com</browse> Opening."
                       for i in range(10)]
            session = _session(replies, config=config, max_tool_rounds=12,
                               tool_family_rounds={"default": 6, "browser": 3})
            session.run()
    finally:
        chat.BrowserSession = real_browser
    everything = " ".join(session.prompts_seen)
    assert "Budget for the 'browser' tools is spent" in everything, everything[-600:]


def test_a_missing_budget_setting_behaves_like_no_budget():
    """A malformed or absent agent.tool_family_rounds must not lock a family
    out — it must put the harness back how it was before budgets existed."""
    session = _session(["done"], max_tool_rounds=11)
    session.config["agent"]["tool_family_rounds"] = "not a dict"
    bound = chat.ChatSession.__new__(chat.ChatSession)
    bound.config = session.config
    assert bound._family_budget("browser") == 11
    session.config["agent"]["tool_family_rounds"] = {"default": 0}
    assert bound._family_budget("browser") == 11


# ------------------------------------------------- distinct BY IDEA, not args

def test_failures_of_the_same_kind_are_one_idea():
    sig = persistence.failure_signature
    assert (sig("read_file", "File not found: step2_numbers.txt")
            == sig("read_file", "File not found: page.html")
            == sig("read_file", "File not found: top.json"))
    assert sig("read_file", "File not found: x.txt") != \
        sig("read_file", "Permission denied reading x.txt")
    assert sig("read_file", "File not found: x") != sig("run_command", "File not found: x")


def test_guessing_filenames_does_not_buy_the_right_to_stop():
    """Live 2026-09-14: the model lost its footing and guessed page.html,
    top.json, clean.json — seven calls, seven different arguments, every one
    "file not found". The gate counted seven distinct attempts, decided the
    turn was well explored, and let it stop. Variety is not progress."""
    with scratch_notes_dir():
        session = _session(
            [f"<tool_call>{json.dumps({'name': 'read_file', 'arguments': {'path': f'invented_{i}.json'}})}</tool_call> Trying."
             for i in range(6)] + ["I could not find it."],
            max_tool_rounds=10,
            min_distinct_attempts=3,
        )
        session.run()
    everything = " ".join(session.prompts_seen)
    assert "genuinely different approaches" in everything, everything[-900:]


def test_real_progress_still_satisfies_the_gate():
    """The counter must not punish a turn that actually tried different
    things: three approaches that fail three DIFFERENT ways is the exploration
    the gate exists to require, and it should be allowed to stop."""
    with scratch_notes_dir():
        session = _session(
            ["<cmd>definitely-not-a-real-binary</cmd> One.",
             "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"nope_missing.json\"}}</tool_call> Two.",
             "<tool_call>{\"name\": \"execute_code\", \"arguments\": {\"code\": \"print(1/0)\"}}</tool_call> Three.",
             "Three different things failed; here is what blocked me."],
            max_tool_rounds=10,
            min_distinct_attempts=3,
        )
        session.run()
    everything = " ".join(session.prompts_seen)
    assert "genuinely different approaches" not in everything


# ------------------------------------------------- values produced by eye

def test_a_value_no_tool_produced_is_challenged():
    """Live 2026-09-14: a file held C#O#R#M#O#R#A#N#T#-#7#7#4#1 and said to
    remove every '#'. The model did it mentally, re-checked, listed the letters
    correctly, and answered "COROMORANT-7741". Seven tool calls, all
    successful, zero failures counted — nothing in the persistence machinery
    could see it, because nothing failed."""
    from symbio.app.chat_constants import unverified_tokens

    obs = ["Contents of vault.txt:\nC#O#R#M#O#R#A#N#T#-#7#7#4#1"]
    assert unverified_tokens("The token is **COROMORANT-7741**.", obs) == \
        ["COROMORANT-7741"]


def test_a_value_a_tool_did_produce_is_left_alone():
    from symbio.app.chat_constants import unverified_tokens

    obs = ["Python script exited ok. Output: CORMORANT-7741"]
    assert unverified_tokens("The token is **CORMORANT-7741**.", obs) == []


def test_ordinary_answers_are_not_challenged():
    """This predicate reads every answer, so its false-positive rate is the
    whole question."""
    from symbio.app.chat_constants import unverified_tokens

    for reply, obs, said in (
        ("Done — I wrote notes/setup.md.", ["Wrote notes/setup.md."], ""),
        ("Disk is 58% full, 412G free.", ["58% 412Gi"], ""),
        ("You're on Python 3.12.1.", ["Python 3.12.1"], ""),
        ("I opened example.com for you.", ["Opened browser at https://example.com"], ""),
        ("Today is 2026-09-14 and all 12 tests passed.", ["12 passed"], ""),
        # The user supplied it themselves; repeating it is not invention.
        ("Your key AKIA1234567890XY is in the file.", ["nothing"],
         "check AKIA1234567890XY please"),
    ):
        assert unverified_tokens(reply, obs, said) == [], reply


def test_the_challenge_fires_in_a_real_turn():
    with scratch_notes_dir():
        session = _session(
            ["<cmd>echo C#O#R#M#O#R#A#N#T#-#7#7#4#1</cmd> Reading it.",
             "The final token is COROMORANT-7741.",
             "Checked with a tool: CORMORANT-7741."],
            max_tool_rounds=8,
        )
        session.run()
    everything = " ".join(session.prompts_seen)
    assert "no tool output this turn contains that value" in everything, \
        everything[-800:]
