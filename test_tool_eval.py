"""The simulated-instance harness, driven by a scripted model.

Every test here fakes generation, so the harness is verified without loading a
model. What is being checked is that the funnel reports the RIGHT stage — a
harness that says "failed" without saying where is no better than the silence
it was built to replace.
"""
import pytest

from symbio.app import tool_eval
from symbio.app.tool_eval import ToolCase, run_tool_cases


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        # Keep the messages visible so a scripted model can react to them.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def scripted(*replies):
    """A generate_fn that returns each reply in turn."""
    seq = list(replies)

    def gen(model, tokenizer, prompt=None, sampler=None, max_tokens=0,
            verbose=False):
        return seq.pop(0) if seq else ""
    return gen


CASE = ToolCase(
    id="cron",
    description="daily reminder reaches schedule_job",
    prompt="Remind me to stretch every day at 9am.",
    expect_tool="schedule_job",
    observation="Scheduled job 12.",
    check_args=lambda a: bool(a.get("schedule")),
    check_final=lambda d: "stretch" in d.lower(),
)


def run(*replies, case=CASE):
    return run_tool_cases(
        model=object(), tokenizer=FakeTokenizer(), system_prompt="SYS",
        config={}, generate_fn=scripted(*replies), cases=(case,),
        output_fn=lambda *_a, **_k: None)


def test_a_clean_round_trip_reaches_completed():
    report = run('<schedule_job schedule="0 9 * * *" text="stretch"/>',
                 "Done — I'll remind you to stretch at 9am.")
    assert report["passed"] == 1
    assert report["funnel"]["completed"] == 1


def test_a_reply_that_invokes_nothing_stops_at_the_start():
    """The exact live failure: prose that reads like success."""
    report = run("schedule_jobs: add a daily reminder to stretch at 9am.")
    assert report["passed"] == 0
    assert report["funnel"]["invoked"] == 0
    assert report["results"][0]["note"] == "no resolvable tool call"


def test_calling_the_wrong_tool_stops_at_invoked():
    report = run("<cmd>crontab -e</cmd>")
    assert report["funnel"]["invoked"] == 1
    assert report["funnel"]["right_tool"] == 0
    assert report["results"][0]["called"] == "run_command"


def test_missing_arguments_stop_at_right_tool():
    report = run('<schedule_job text="stretch"/>')
    assert report["funnel"]["right_tool"] == 1
    assert report["funnel"]["right_args"] == 0
    assert report["results"][0]["note"] == "arguments unusable"


def test_ignoring_the_observation_stops_at_right_args():
    """It called correctly, then failed to use what came back — a different
    problem from never calling, and the funnel must distinguish them."""
    report = run('<schedule_job schedule="0 9 * * *" text="stretch"/>',
                 "Okay.")
    assert report["funnel"]["right_args"] == 1
    assert report["funnel"]["completed"] == 0
    assert "did not use the result" in report["results"][0]["note"]


def test_the_observation_is_fed_back_the_way_chat_does_it():
    """A pass here has to mean the same exchange would work live, so the
    simulated result must arrive as chat.py delivers a real one."""
    seen = []

    def gen(model, tokenizer, prompt=None, sampler=None, max_tokens=0,
            verbose=False):
        seen.append(prompt)
        return ('<schedule_job schedule="0 9 * * *" text="stretch"/>'
                if len(seen) == 1 else "Done, stretch at 9am.")

    run_tool_cases(model=object(), tokenizer=FakeTokenizer(),
                   system_prompt="SYS", config={}, generate_fn=gen,
                   cases=(CASE,), output_fn=lambda *_a, **_k: None)

    assert tool_eval.OBSERVATION_PREFIX in seen[1]
    assert "Scheduled job 12." in seen[1]
    # And the model's own first reply must be in the history it sees.
    assert "schedule_job" in seen[1]


def test_a_generation_error_is_recorded_not_raised():
    def boom(*a, **k):
        raise RuntimeError("model exploded")

    report = run_tool_cases(model=object(), tokenizer=FakeTokenizer(),
                            system_prompt="SYS", config={}, generate_fn=boom,
                            cases=(CASE,), output_fn=lambda *_a, **_k: None)
    assert report["passed"] == 0
    assert report["results"][0]["note"] == "generation error"


def test_nothing_is_executed():
    """The safety property inherited from golden.py: this harness simulates a
    result, it never runs the tool. A case asking for a shell command must not
    reach the sandbox."""
    import symbio.app.tooling as tooling_mod
    assert not hasattr(tool_eval, "run_sandboxed")
    assert not hasattr(tool_eval, "execute_tool")
    # The only tooling functions it uses are the two pure parsers.
    src = open(tool_eval.__file__).read()
    assert "tooling.parse_tools" in src
    assert "tooling.strip_tool_tags" in src
    for forbidden in ("sandbox.", "run_shell", "subprocess"):
        assert forbidden not in src, f"tool_eval must not reference {forbidden}"


def test_the_default_battery_leads_with_the_tools_that_broke():
    ids = [c.id for c in tool_eval.DEFAULT_CASES]
    assert ids[0].startswith("cron")
    assert any(c.expect_tool == "schedule_job" for c in tool_eval.DEFAULT_CASES)
    assert any(c.expect_tool == "list_cron_jobs" for c in tool_eval.DEFAULT_CASES)


@pytest.mark.parametrize("reply,expected_stage", [
    ('<schedule_job schedule="0 9 * * *" text="s"/>', "right_args"),
    ('.schedule_job schedule="0 9 * * *" text="s"', "right_args"),
    ('schedule_job(schedule="0 9 * * *", text="s")', "right_args"),
])
def test_every_improvised_shape_counts_as_invoked(reply, expected_stage):
    """The harness must credit the shapes the parser now recovers, or it will
    under-report exactly the fix that made them work."""
    report = run(reply, "Okay.")
    assert report["results"][0]["reached"] == expected_stage


def test_a_wrong_tool_failure_names_what_it_actually_called():
    """Reporting only "expected list_cron_jobs" says a case failed without
    saying how to fix it. For a wrong-tool failure the tool it chose instead
    is the entire diagnosis."""
    report = run("<cmd>crontab -l</cmd>")
    assert report["results"][0]["note"] == (
        "called run_command instead of schedule_job")


def test_an_acceptable_alternative_counts_as_the_right_tool():
    case = CASE._replace(expect_tool="web_search", also_accept=("read_page",),
                         check_args=None)
    report = run("<read>https://example.com</read>", "stretch", case=case)
    assert report["results"][0]["called"] == "read_page"
    assert report["passed"] == 1


def test_the_right_tool_still_counts_when_it_is_not_the_first_call():
    """A reply that calls the right tool second was being graded as a
    wrong-tool failure, because only tools[0] was ever examined."""
    reply = ('<cmd>date</cmd> <tool_call>{"name": "schedule_job", "arguments": '
             '{"schedule": "0 9 * * *", "text": "stretch"}}</tool_call>')
    report = run(reply, "Done, stretch at 9am.")
    assert report["results"][0]["called"] == "schedule_job"
    assert report["passed"] == 1


def test_an_improvised_call_is_lost_when_another_syntax_also_matched():
    """A known cost of the improvised-form recovery, pinned so it is a
    decision rather than a surprise.

    parse_tools only reaches for the improvised shapes when nothing else
    parsed, which is what stops <type enter=".."> and <note title=".."> from
    double-firing. The price is that a reply mixing a well-formed call with an
    improvised one drops the improvised one. Rare in practice — the model
    picks one syntax per reply — and the alternative is span-tracking every
    XML handler in parse_tools to dedupe safely."""
    reply = '<cmd>date</cmd> <schedule_job schedule="0 9 * * *" text="stretch"/>'
    report = run(reply, "Done.")
    assert report["results"][0]["called"] == "run_command"
    assert "instead of schedule_job" in report["results"][0]["note"]


def test_the_failing_reply_is_shown_in_the_summary():
    lines = []
    run_tool_cases(model=object(), tokenizer=FakeTokenizer(),
                   system_prompt="SYS", config={},
                   generate_fn=scripted("I will schedule that for you."),
                   cases=(CASE,), output_fn=lambda m="": lines.append(str(m)))
    assert any("said: I will schedule that for you." in ln for ln in lines)


def test_repeats_report_a_rate_not_a_verdict():
    """A case that passes sometimes must read as flaky, because that is a
    different problem from one that never passes."""
    replies = ['<schedule_job schedule="0 9 * * *" text="stretch"/>', "stretch scheduled",
               "schedule_jobs: I'll remind you to stretch.",
               '<schedule_job schedule="0 9 * * *" text="stretch"/>', "stretch scheduled"]
    report = run_tool_cases(
        model=object(), tokenizer=FakeTokenizer(), system_prompt="SYS",
        config={}, generate_fn=scripted(*replies), cases=(CASE,), repeats=3,
        output_fn=lambda *_a, **_k: None)
    assert report["per_case"]["cron"]["runs"] == 3
    assert report["per_case"]["cron"]["passed"] == 2


# ---- the extended battery ----

def test_the_extended_battery_expects_only_resolvable_tools():
    """A case expecting a name parse_tools cannot resolve can never pass, and
    would read as a permanent model failure that no prompt change could fix."""
    from symbio.app.tooling import _TOOL_GROUPS
    bad = [c.id for c in tool_eval.EXTENDED_CASES
           if c.expect_tool not in _TOOL_GROUPS]
    assert bad == []


def test_every_extended_case_has_a_distinct_id():
    """Ids key the per-case rates; a duplicate silently merges two cases."""
    ids = [c.id for c in tool_eval.EXTENDED_CASES]
    assert len(ids) == len(set(ids))


def test_the_worker_delegation_cases_name_real_registered_roles():
    """A delegate_task case is worthless if it names a worker that does not
    exist — the model would be right to refuse, and the case would read as a
    dispatch failure forever."""
    import json

    from symbio import constants
    catalog = json.loads(
        constants.WORKER_MODELS_FILE.read_text(encoding="utf-8"))
    roles = {e.get("role") for e in catalog.values()}
    checked = 0
    for case in tool_eval.EXTENDED_CASES:
        if case.expect_tool != "delegate_task":
            continue
        checked += 1
        assert [r for r in roles if r and r in case.prompt], \
            f"{case.id} names no registered worker role"
    assert checked, "no delegate_task cases to check"


def test_a_correct_answer_in_different_words_is_not_a_failure():
    """The grader must not manufacture failures. Live: handed "All subsystems
    healthy. 0 errors." the model answered "System check passed." — correct,
    and scored as a miss because the check demanded the word "health"."""
    case = CASE._replace(check_final=tool_eval._any_of("health", "passed"),
                         check_args=None)
    report = run('<schedule_job schedule="0 9 * * *"/>', "System check passed.",
                 case=case)
    assert report["passed"] == 1
