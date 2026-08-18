"""Does the model actually reach its tools? A simulated-instance harness.

golden.py answers "did the reply look right" in a single tool-free turn. It
deliberately never executes a tool, which is what makes it safe to run around
every LoRA update — but it also means nothing here has ever measured the thing
that keeps breaking: whether an invocation the model writes is one the parser
can actually resolve, and whether the model can finish the job once a result
comes back.

That gap is not theoretical. The whole cron group was unreachable in practice
while being correctly declared, correctly grouped and correctly implemented,
because the model wrote a call shape nothing recognised and a success-looking
sentence went to the user. Nothing failed. Nothing was logged. It just quietly
did not happen.

This runs each case as a two-turn conversation and never executes anything:

    system + user(prompt)
      -> model reply            (turn 1: did a resolvable call come out?)
      -> [System observation: canned result]   <- simulated, not executed
      -> model reply            (turn 2: did it finish the job?)

The observation is fed back exactly as chat.py feeds a real one — a user turn
whose content starts with "[System observation: " — so a pass here means the
same shape of exchange would have worked live.

Results are a four-stage funnel rather than a pass/fail, because where it
breaks is the whole diagnostic value:

    invoked      the reply contained a call the parser could resolve at all
    right_tool   ...and it was the tool the case asked for
    right_args   ...and the arguments were usable
    completed    ...and after the result came back, the final answer was right

A model that scores 0% on `invoked` needs prompt examples or constrained
decoding. One that invokes fine but fails `completed` needs different training
entirely. A single number could not tell those apart.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

from . import prompts, tooling

OBSERVATION_PREFIX = "[System observation: "

STAGES = ("invoked", "right_tool", "right_args", "completed")


class ToolCase(NamedTuple):
    id: str
    description: str
    prompt: str
    expect_tool: str
    # What the simulated tool "returns". Kept literal and boring on purpose:
    # a case should fail because the model could not use a normal result, not
    # because the fixture was exotic.
    observation: str
    check_args: Callable[[dict[str, Any]], bool] | None = None
    check_final: Callable[[str], bool] | None = None


class CaseResult(NamedTuple):
    id: str
    reached: str            # furthest stage reached, or "" if nothing worked
    called: str | None      # the tool actually invoked, if any
    args: dict[str, Any]
    turn1: str
    turn2: str
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.reached == STAGES[-1]


def _contains(*needles: str) -> Callable[[str], bool]:
    """Final-answer check: the reply mentions each needle, case-insensitively.

    Deliberately loose. This harness measures whether the tool round-trip
    works, not whether the prose is elegant — a strict match would fail good
    answers and make the funnel lie about where the problem is.
    """
    def check(display: str) -> bool:
        low = display.lower()
        return bool(display.strip()) and all(n.lower() in low for n in needles)
    return check


def _args_have(*keys: str) -> Callable[[dict[str, Any]], bool]:
    def check(args: dict[str, Any]) -> bool:
        return all(str(args.get(k, "")).strip() for k in keys)
    return check


# The default battery leads with the tools that actually failed in the wild.
DEFAULT_CASES: tuple[ToolCase, ...] = (
    ToolCase(
        id="cron_schedule",
        description="A daily reminder reaches schedule_job",
        prompt="Remind me to stretch every day at 9am.",
        expect_tool="schedule_job",
        observation="Scheduled job 12: 'stretch' at 0 9 * * *.",
        check_args=_args_have("schedule"),
        check_final=_contains("stretch"),
    ),
    ToolCase(
        id="cron_list",
        description="Asking what is scheduled reaches list_cron_jobs",
        prompt="What reminders do I have set up?",
        expect_tool="list_cron_jobs",
        observation="1 job: id 12, 'stretch', 0 9 * * *.",
        check_final=_contains("stretch"),
    ),
    ToolCase(
        id="shell_command",
        description="A shell request reaches run_command",
        prompt="Run `uname -s` and tell me what it prints.",
        expect_tool="run_command",
        observation="Darwin",
        check_args=_args_have("cmd"),
        check_final=_contains("darwin"),
    ),
    ToolCase(
        id="web_search",
        description="A lookup reaches web_search",
        prompt="Search the web for the current Mars rover mission name.",
        expect_tool="web_search",
        observation="Top result: NASA's Perseverance rover is active on Mars.",
        check_args=_args_have("query"),
        check_final=_contains("perseverance"),
    ),
    ToolCase(
        id="read_file",
        description="A file read reaches read_file",
        prompt="Read the file notes/todo.md and summarise it.",
        expect_tool="read_file",
        observation="- buy milk\n- call the dentist",
        check_final=_contains("dentist"),
    ),
    ToolCase(
        id="system_check",
        description="A health question reaches system_check",
        prompt="Run a health check on yourself and report the result.",
        expect_tool="system_check",
        observation="All subsystems healthy. 0 errors.",
        check_final=_contains("health"),
    ),
)


def run_tool_cases(
    model,
    tokenizer,
    system_prompt: str,
    config: dict[str, Any],
    generate_fn,
    sampler=None,
    cases: tuple[ToolCase, ...] | None = None,
    max_tokens: int = 200,
    enabled_groups: set[str] | None = None,
    enable_thinking: bool = False,
    output_fn=print,
) -> dict[str, Any]:
    """Run every case as a simulated two-turn exchange. Executes nothing."""
    cases = cases if cases is not None else DEFAULT_CASES
    context = system_prompt + prompts.env_note() + prompts.time_note()
    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        output_fn(f"  [ToolEval] {i}/{len(cases)} {case.id}...")
        messages = [{"role": "system", "content": context},
                    {"role": "user", "content": case.prompt}]
        turn1 = _generate(model, tokenizer, messages, generate_fn, sampler,
                          max_tokens, enable_thinking)
        if turn1 is None:
            results.append(CaseResult(case.id, "", None, {}, "", "",
                                      "generation error"))
            continue

        tools = tooling.parse_tools(turn1, enabled_groups)
        if not tools:
            # The failure that started all this: a reply that reads like
            # success and invokes nothing.
            results.append(CaseResult(case.id, "", None, {}, turn1, "",
                                      "no resolvable tool call"))
            continue

        called, args = tools[0]
        reached = "invoked"
        if called != case.expect_tool:
            results.append(CaseResult(case.id, reached, called, args, turn1, "",
                                      f"expected {case.expect_tool}"))
            continue

        reached = "right_tool"
        if case.check_args is not None and not case.check_args(args):
            results.append(CaseResult(case.id, reached, called, args, turn1, "",
                                      "arguments unusable"))
            continue

        reached = "right_args"
        # Feed the result back the way chat.py does — as a user turn.
        messages += [
            {"role": "assistant", "content": turn1},
            {"role": "user",
             "content": f"{OBSERVATION_PREFIX}{case.observation}]"},
        ]
        turn2 = _generate(model, tokenizer, messages, generate_fn, sampler,
                          max_tokens, enable_thinking)
        if turn2 is None:
            results.append(CaseResult(case.id, reached, called, args, turn1, "",
                                      "generation error on turn 2"))
            continue

        display = tooling.strip_tool_tags(turn2)
        ok = case.check_final is None or case.check_final(display)
        results.append(CaseResult(
            case.id, "completed" if ok else reached, called, args, turn1,
            display, "" if ok else "final answer did not use the result"))

    return _summarise(results, output_fn)


def _generate(model, tokenizer, messages, generate_fn, sampler, max_tokens,
              enable_thinking) -> str | None:
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)
    try:
        return generate_fn(model, tokenizer, prompt=chat_prompt,
                           sampler=sampler, max_tokens=max_tokens,
                           verbose=False).strip()
    except Exception:
        return None


def _summarise(results: list[CaseResult], output_fn) -> dict[str, Any]:
    total = len(results) or 1
    funnel = {}
    for idx, stage in enumerate(STAGES):
        reached = sum(1 for r in results
                      if r.reached and STAGES.index(r.reached) >= idx)
        funnel[stage] = reached

    output_fn("\n  [ToolEval] funnel over "
              f"{len(results)} case(s):")
    for stage in STAGES:
        n = funnel[stage]
        output_fn(f"    {stage:11s} {n}/{len(results)}  ({100 * n / total:.0f}%)")

    failures = [r for r in results if not r.passed]
    if failures:
        output_fn("\n  [ToolEval] where it stopped:")
        for r in failures:
            output_fn(f"    {r.id:16s} {r.reached or 'nothing':11s} {r.note}")

    return {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "funnel": funnel,
        "results": [r._asdict() for r in results],
    }
