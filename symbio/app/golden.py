"""Golden set: a small, fixed battery of prompts that exercise behaviors
directly seeded into every install's training corpus (symbio.app.training.
seed_training_data) — identity, tool-tag formatting, and the never-guess
contract. Fine-tuning should never make these worse.

Each case is single-turn and side-effect-free: the model's raw reply is
parsed for tool tags but no tool is actually executed, so running the set
never touches the shell, network, or notes. That makes it safe to run
automatically before and after every LoRA update, so a regression (a tag
format that stopped parsing, an identity the model forgot, a runaway
repetition loop from an overfit adapter) is caught immediately instead of
surfacing later as a silently worse assistant.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

from symbio.app import prompts, tooling

_LEAKED_TAG_MARKERS = ("<tool_call", "</tool_call>", "<tool_response")


def _looks_degenerate(text: str) -> bool:
    """A short phrase repeating many times is the classic signature of an
    overfit or corrupted LoRA adapter looping instead of answering."""
    words = text.split()
    if len(words) < 12:
        return False
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return max(Counter(trigrams).values(), default=0) >= 4


def sane_reply(display: str) -> bool:
    """Baseline every case requires: no runaway repetition, and no tool tag
    that leaked through stripping (a sign the parser and the model's output
    format have drifted apart)."""
    if _looks_degenerate(display):
        return False
    return not any(marker in display for marker in _LEAKED_TAG_MARKERS)


def _has_tool(tools: list[tuple[str, dict[str, Any]]], name: str) -> bool:
    return any(n == name for n, _ in tools)


class GoldenCase(NamedTuple):
    id: str
    description: str
    prompt_fn: Callable[[dict[str, Any]], str]
    check: Callable[[str, list[tuple[str, dict[str, Any]]], dict[str, Any]], bool]


def _check_greeting(display: str, tools: list, config: dict) -> bool:
    return bool(display.strip()) and sane_reply(display)


def _check_identity_self(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and config["assistant_name"].lower() in display.lower()


def _check_identity_not_user(display: str, tools: list, config: dict) -> bool:
    lowered = display.strip().lower()
    return (
        sane_reply(display)
        and config["assistant_name"].lower() in lowered
        and not lowered.startswith(("yes", "yeah", "yep", "correct"))
    )


def _check_save_note(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and (_has_tool(tools, "write_note") or _has_tool(tools, "save_memory"))


def _check_schedule(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "schedule_job")


def _check_run_code(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "execute_code")


def _check_web_search(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "web_search")


def _check_open_app(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "run_command")


# Prompts and expected behavior mirror pairs baked into seed_training_data,
# so every install can learn them fresh even before any real conversation —
# a case failing means fine-tuning eroded a contract that was demonstrably
# present in the training data, not that the base model never knew it.
GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        "greeting", "Replies to a plain greeting without degenerating",
        lambda cfg: "Hey there!",
        _check_greeting,
    ),
    GoldenCase(
        "identity_self", "States its own name when asked",
        lambda cfg: "What is your name?",
        _check_identity_self,
    ),
    GoldenCase(
        "identity_not_user", "Does not confuse itself with the user",
        lambda cfg: f"Are you {cfg['user_name']}?",
        _check_identity_not_user,
    ),
    GoldenCase(
        "save_note", "Saves a fact the user asks it to remember",
        lambda cfg: "Please remember that I prefer concise replies.",
        _check_save_note,
    ),
    GoldenCase(
        "schedule_reminder", "Schedules a cron reminder on request",
        lambda cfg: "Remind me every day at 9am to stretch.",
        _check_schedule,
    ),
    GoldenCase(
        "run_code_for_math", "Runs code for an exact computation",
        lambda cfg: "Run code to calculate 7 factorial.",
        _check_run_code,
    ),
    GoldenCase(
        "web_search_unknown", "Searches instead of guessing at current info",
        lambda cfg: "What is the latest news?",
        _check_web_search,
    ),
    GoldenCase(
        "open_app_command", "Emits a shell command to open an application",
        lambda cfg: "Open Chrome.",
        _check_open_app,
    ),
]


@dataclass
class GoldenResult:
    results: dict[str, bool]
    replies: dict[str, str]

    @property
    def passing(self) -> set[str]:
        return {case_id for case_id, ok in self.results.items() if ok}

    @property
    def pass_count(self) -> int:
        return sum(self.results.values())

    @property
    def total(self) -> int:
        return len(self.results)


def run_golden_set(
    model, tokenizer, generate_fn, sampler, system_prompt: str,
    config: dict[str, Any], enabled_groups: set[str] | None = None,
    max_tokens: int | None = None, cases: list[GoldenCase] | None = None,
) -> GoldenResult:
    """Run every golden case as a single-turn, tool-free generation and
    grade it. Never executes a tool — only parses the reply — so it is safe
    to run automatically around every LoRA update. `cases` defaults to the
    headmaster's identity/tool-tag battery (GOLDEN_CASES); a worker role
    passes its own smaller, task-scoped list (see dispatch.WORKER_GOLDEN_CASES)."""
    cases = cases if cases is not None else GOLDEN_CASES
    max_tokens = max_tokens or int(config.get("learn", {}).get("golden_max_tokens", 150))
    context = system_prompt + prompts.env_note() + prompts.time_note()

    results: dict[str, bool] = {}
    replies: dict[str, str] = {}
    for case in cases:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": case.prompt_fn(config)},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        try:
            raw_reply = generate_fn(
                model, tokenizer, prompt=chat_prompt, sampler=sampler,
                max_tokens=max_tokens, verbose=False,
            ).strip()
        except Exception as e:
            results[case.id] = False
            replies[case.id] = f"[generation error: {e}]"
            continue

        tools = tooling.parse_tools(raw_reply, enabled_groups)
        display = tooling.strip_tool_tags(raw_reply)
        replies[case.id] = raw_reply
        try:
            results[case.id] = bool(case.check(display, tools, config))
        except Exception:
            results[case.id] = False

    return GoldenResult(results, replies)
