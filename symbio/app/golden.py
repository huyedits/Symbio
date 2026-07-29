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

Users and the AI itself can extend the golden set by editing
`golden_cases.json` in the project root. Built-in cases are always present;
extensions add new behavioral contracts the model should keep.
"""

from __future__ import annotations

import json
import platform as platform_module
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

from symbio import constants
from symbio.app import prompts, tooling, training

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
    """Opening an app by name must use a native GUI opener command via the
    terminal tool: macOS `open -a 'App Name'`, Windows `start app`, or
    Linux `xdg-open app`."""
    if not sane_reply(display):
        return False
    for _, params in tools:
        cmd = params.get("cmd", "")
        if cmd.startswith("open -a") or cmd.startswith("start ") or cmd.startswith("xdg-open "):
            return True
    return False


def _check_browse_for_interaction(display: str, tools: list, config: dict) -> bool:
    """Opening a named site 'in Chrome' with a follow-up click must use the
    controllable browser, not the user's default browser via <cmd>open."""
    return sane_reply(display) and _has_tool(tools, "browser_open")


def _check_browser_press(display: str, tools: list, config: dict) -> bool:
    """Pressing a key in the browser must use <press>, not a fake shell command."""
    return sane_reply(display) and _has_tool(tools, "browser_press")


def _check_system_check(display: str, tools: list, config: dict) -> bool:
    """A request to check health should call the system_check tool."""
    return sane_reply(display) and _has_tool(tools, "system_check")


# Maps string check names from golden_cases.json to real checker functions.
# New built-in checkers can be added here; users can also request custom ones.
_CHECK_REGISTRY: dict[str, Callable[[str, list, dict], bool]] = {
    "sane_reply": lambda display, tools, cfg: sane_reply(display),
    "has_tool": lambda display, tools, cfg, name="": _has_tool(tools, name),
    "identity_self": _check_identity_self,
    "identity_not_user": _check_identity_not_user,
    "save_note": _check_save_note,
    "schedule": _check_schedule,
    "execute_code": _check_run_code,
    "web_search": _check_web_search,
    "open_app": _check_open_app,
    "browse": _check_browse_for_interaction,
    "browser_press": _check_browser_press,
    "system_check": _check_system_check,
    "non_empty": lambda display, tools, cfg: bool(display.strip()),
}


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
    GoldenCase(
        "browse_to_interact", "Uses the controllable browser when asked to open a site and click",
        lambda cfg: "Open cloudflare.com in Chrome and click the first button.",
        _check_browse_for_interaction,
    ),
    GoldenCase(
        "browse_apple", "Uses the controllable browser to read apple.com",
        lambda cfg: "Read what apple.com says.",
        _check_browse_for_interaction,
    ),
    GoldenCase(
        "browser_press_key", "Presses a browser key instead of inventing a shell command",
        lambda cfg: "Press the down arrow key.",
        _check_browser_press,
    ),
    GoldenCase(
        "run_health_check", "Calls the self-diagnostic tool when asked to check health",
        lambda cfg: "Run a health check.",
        _check_system_check,
    ),
]


def _golden_ideal_replies(config: dict[str, Any]) -> dict[str, str]:
    """The canonical assistant reply for each built-in golden case, used to build
    remedy training samples when a case consistently fails after a LoRA
    update. Values mirror the seed corpus so the remedy reinforces the
    same contract, but are name/platform-aware.

    User-defined golden cases can optionally supply an `ideal_reply` field in
    golden_cases.json to be used for remedy samples."""
    assistant = config.get("assistant_name", "Assistant")
    user = config.get("user_name", "User")
    platform = platform_module.system()
    if platform == "Darwin":
        open_chrome = "open -a 'Google Chrome'"
    elif platform == "Windows":
        open_chrome = "start chrome"
    else:
        open_chrome = "xdg-open https://www.google.com"
    return {
        "greeting": "Hey! What can I help with?",
        "identity_self": f"I am {assistant}, your personal AI assistant.",
        "identity_not_user": f"No — I'm {assistant}, your assistant. You're {user}.",
        "save_note": "Got it. <note title='Pref'>Prefers concise replies.</note>",
        "schedule_reminder": f"Will do, {user}. <cron expr='0 9 * * *'>stretch</cron>",
        "run_code_for_math": "<py>import math\nprint(math.factorial(7))</py> Running that now.",
        "web_search_unknown": "<search>latest news</search> Searching now.",
        "open_app_command": f"<cmd>{open_chrome}</cmd> Opening Chrome.",
        "browse_to_interact": "<browse>https://www.cloudflare.com</browse> Opening cloudflare.com so I can click the first button.",
        "browse_apple": "<browse>https://www.apple.com</browse> Opening apple.com to read it.",
        "browser_press_key": "<press>down</press> Pressing the down arrow key.",
        "run_health_check": '<tool_call>{"name": "system_check", "arguments": {}}</tool_call> Running a self-diagnostic now.',
    }


def _user_ideal_reply(case_id: str, config: dict[str, Any]) -> str | None:
    """Return an ideal reply for a user-defined golden case, if provided."""
    if not constants.GOLDEN_CASES_FILE.exists():
        return None
    try:
        data = json.loads(constants.GOLDEN_CASES_FILE.read_text(encoding="utf-8"))
        spec = data.get(case_id, {})
        reply = spec.get("ideal_reply")
        if not reply:
            return None
        return reply.replace("ASSISTANT_NAME", config.get("assistant_name", "Assistant")).replace(
            "USER_NAME", config.get("user_name", "User")
        )
    except Exception:
        return None


def append_golden_remedy_samples(
    failing_case_ids: list[str],
    tokenizer,
    system_prompt: str,
    config: dict[str, Any],
    role: str | None = None,
    copies: int = 3,
) -> int:
    """Write boosted (prompt, ideal-reply) training samples for golden cases
    that consistently fail. Returns the number of samples appended.

    Built-in cases use the shipped ideal replies; user-defined cases use the
    `ideal_reply` field from golden_cases.json when available."""
    ideal = _golden_ideal_replies(config)
    case_by_id = {case.id: case for case in all_golden_cases()}
    added = 0
    for case_id in failing_case_ids:
        case = case_by_id.get(case_id)
        if case is None:
            continue
        target = ideal.get(case_id) or _user_ideal_reply(case_id, config)
        if target is None:
            continue
        user_msg = case.prompt_fn(config)
        for _ in range(max(1, copies)):
            training.append_chat_pair(user_msg, target, tokenizer, system_prompt, role=role)
            added += 1
    return added


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


def _make_dynamic_check(requirements: list[dict[str, Any]]) -> Callable[[str, list, dict], bool]:
    """Build a check function from JSON requirements.

    Each requirement is {"kind": "sane_reply" | "has_tool" | "contains" | "not_contains", ...}.
    All requirements must pass.
    """
    def check(display: str, tools: list, config: dict) -> bool:
        if not sane_reply(display):
            return False
        for req in requirements:
            kind = req.get("kind")
            if kind == "sane_reply":
                if not sane_reply(display):
                    return False
            elif kind == "has_tool":
                if not _has_tool(tools, req.get("tool", "")):
                    return False
            elif kind == "contains":
                text = req.get("text", "")
                if text.lower() not in display.lower():
                    return False
            elif kind == "not_contains":
                text = req.get("text", "")
                if text.lower() in display.lower():
                    return False
            elif kind == "regex":
                pattern = req.get("pattern", "")
                if not re.search(pattern, display):
                    return False
            elif kind == "not_regex":
                pattern = req.get("pattern", "")
                if re.search(pattern, display):
                    return False
            else:
                # Unknown requirement kind fails closed.
                return False
        return True
    return check


def load_user_golden_cases() -> list[GoldenCase]:
    """Load extra golden cases from golden_cases.json if present.

    The file format is a JSON object mapping case id to:
      {"description": ..., "prompt": "...", "requirements": [...]}
    Prompts may contain {assistant_name} and {user_name} placeholders.
    """
    cases: list[GoldenCase] = []
    if not constants.GOLDEN_CASES_FILE.exists():
        return cases
    try:
        data = json.loads(constants.GOLDEN_CASES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return cases
    if not isinstance(data, dict):
        return cases
    for case_id, spec in data.items():
        if not isinstance(spec, dict):
            continue
        description = spec.get("description", "")
        prompt_template = spec.get("prompt", "")
        requirements = spec.get("requirements", [])
        if not isinstance(requirements, list):
            continue

        def prompt_fn(cfg, template=prompt_template):
            try:
                return template.replace("ASSISTANT_NAME", cfg.get("assistant_name", "Assistant")).replace(
                    "USER_NAME", cfg.get("user_name", "User")
                )
            except Exception:
                return template

        cases.append(GoldenCase(
            id=case_id,
            description=description,
            prompt_fn=prompt_fn,
            check=_make_dynamic_check(requirements),
        ))
    return cases


def all_golden_cases() -> list[GoldenCase]:
    """Return built-in golden cases plus any user-defined extensions."""
    user_cases = load_user_golden_cases()
    if not user_cases:
        return list(GOLDEN_CASES)
    # Avoid id collisions: built-ins take precedence.
    seen = {case.id for case in GOLDEN_CASES}
    extras = [c for c in user_cases if c.id not in seen]
    return list(GOLDEN_CASES) + extras


def run_golden_set(
    model, tokenizer, generate_fn, sampler, system_prompt: str,
    config: dict[str, Any], enabled_groups: set[str] | None = None,
    max_tokens: int | None = None, cases: list[GoldenCase] | None = None,
) -> GoldenResult:
    """Run every golden case as a single-turn, tool-free generation and
    grade it. Never executes a tool — only parses the reply — so it is safe
    to run automatically around every LoRA update. `cases` defaults to the
    headmaster's identity/tool-tag battery plus any user-defined extensions
    from golden_cases.json; a worker role passes its own smaller, task-scoped
    list (see dispatch.WORKER_GOLDEN_CASES)."""
    cases = cases if cases is not None else all_golden_cases()
    max_tokens = max_tokens or int(config.get("learn", {}).get("golden_max_tokens", 150))
    context = system_prompt + prompts.env_note() + prompts.time_note()

    results: dict[str, bool] = {}
    replies: dict[str, str] = {}
    print(f"  [Golden] Running {len(cases)} pre/post-train checks...")
    for i, case in enumerate(cases, 1):
        print(f"  [Golden] {i}/{len(cases)} {case.id}...", end=" ", flush=True)
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
            print("ERROR")
            continue

        tools = tooling.parse_tools(raw_reply, enabled_groups)
        display = tooling.strip_tool_tags(raw_reply)
        replies[case.id] = raw_reply
        try:
            ok = bool(case.check(display, tools, config))
            results[case.id] = ok
            print("PASS" if ok else "FAIL")
        except Exception:
            results[case.id] = False
            print("FAIL")

    passing = sum(results.values())
    print(f"  [Golden] {passing}/{len(cases)} checks passed.")
    return GoldenResult(results, replies)


def run_golden_set_retry(
    model, tokenizer, generate_fn, sampler, system_prompt: str,
    config: dict[str, Any], enabled_groups: set[str] | None = None,
    max_tokens: int | None = None, cases: list[GoldenCase] | None = None,
) -> tuple[GoldenResult, set[str]]:
    """Run the golden set and, if any cases fail, run it a second time.
    Returns the second (or only) result plus the set of case ids that failed
    on both runs. Callers use the consistent-failure set to ignore flaky
    generation noise."""
    first = run_golden_set(
        model, tokenizer, generate_fn, sampler, system_prompt,
        config, enabled_groups, max_tokens, cases,
    )
    failing_first = {case_id for case_id, ok in first.results.items() if not ok}
    if not failing_first:
        return first, set()

    print(f"  [Golden] Re-checking {len(failing_first)} failing case(s)...")
    second = run_golden_set(
        model, tokenizer, generate_fn, sampler, system_prompt,
        config, enabled_groups, max_tokens, cases,
    )
    consistent = {case_id for case_id in failing_first if not second.results.get(case_id, True)}
    return second, consistent
