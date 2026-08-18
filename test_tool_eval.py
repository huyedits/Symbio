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
