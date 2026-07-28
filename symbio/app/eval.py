"""Held-out evaluation set and before/after LoRA benchmark.

The golden set in symbio.app.golden is for regression prevention: it checks
that behaviours seeded into every install still work after a LoRA update.
This module is for measuring improvement: it runs a separate, held-out set
of tasks both on the base model and on the current adapter, then writes a
JSON report showing whether the adapter is better, worse, or unchanged.

All tasks are single-turn and side-effect-free: replies are parsed for tool
tags but no tool is executed, so the benchmark is safe to run unattended.
"""

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from symbio import constants
from symbio.app import prompts, tooling


@dataclass
class EvalCase:
    id: str
    description: str
    prompt_fn: Callable[[dict[str, Any]], str]
    check: Callable[[str, list[tuple[str, dict[str, Any]]], dict[str, Any]], bool]


def _has_tool(tools: list[tuple[str, dict[str, Any]]], name: str) -> bool:
    return any(n == name for n, _ in tools)


def _contains(display: str, text: str) -> bool:
    return text.lower() in display.lower()


def _check_math_product(display: str, tools: list, config: dict) -> bool:
    if _contains(display, "221"):
        return True
    # The system prompt tells the model to use <py> for exact math; accept
    # that when the generated code clearly computes 13 * 17.
    for name, params in tools:
        if name == "execute_code":
            code = params.get("code", "")
            if all(x in code for x in ("13", "17", "*")):
                return True
    return False


def _check_json_list_abc(display: str, tools: list, config: dict) -> bool:
    try:
        parsed = json.loads(display.strip())
    except Exception:
        return False
    if parsed == ["a", "b", "c"]:
        return True
    # Accept a single-key wrapper object, e.g. {"list": ["a", "b", "c"]},
    # as long as the wrapped value is exactly the requested array.
    if isinstance(parsed, dict) and len(parsed) == 1:
        value = next(iter(parsed.values()))
        return value == ["a", "b", "c"]
    return False


def _check_remember_color(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and (
        _has_tool(tools, "write_note") or _has_tool(tools, "save_memory")
    )


def _check_weekly_reminder(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "schedule_job")


def _check_web_search(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "web_search")


def _check_browser_read(display: str, tools: list, config: dict) -> bool:
    return sane_reply(display) and _has_tool(tools, "browser_open")


def _check_run_code_primes(display: str, tools: list, config: dict) -> bool:
    if not sane_reply(display):
        return False
    if _has_tool(tools, "execute_code"):
        return True
    # If it answers directly, it must list the first five primes correctly.
    text = display.lower()
    return all(p in text for p in ("2", "3", "5", "7", "11"))


def _check_open_app(display: str, tools: list, config: dict) -> bool:
    if not sane_reply(display):
        return False
    app = config.get("open_app", "Safari")
    for _, params in tools:
        cmd = params.get("cmd", "")
        if cmd.startswith("open -a") or cmd.startswith("start ") or cmd.startswith("xdg-open "):
            if app in cmd:
                return True
    # Fallback: malformed tool_call JSON that still contains the right command.
    if re.search(rf"open\s+-a\s+['\"]?\b{re.escape(app)}\b['\"]?", display, re.IGNORECASE):
        return True
    return False


def _check_who_are_you(display: str, tools: list, config: dict) -> bool:
    user = config.get("user_name", "").strip().lower()
    assistant = config.get("assistant_name", "").strip().lower()
    lower = display.lower()
    if not sane_reply(display):
        return False
    if assistant not in lower:
        return False
    if not user:
        return True
    # Reject clear self-reference / name-swap patterns, but allow benign
    # references such as "Your name is Huy" or "My user is named Huy".
    swap_phrases = [
        f"i am {user}",
        f"i'm {user}",
        f"my name is {user}",
        f"call me {user}",
        f"you can call me {user}",
    ]
    return not any(phrase in lower for phrase in swap_phrases)


def sane_reply(display: str) -> bool:
    """Same guard used by the golden set: no leaked tool-call syntax."""
    leaked = ("<tool_call", "</tool_call>", "<tool_response")
    return not any(marker in display for marker in leaked)


# Held-out tasks. These are deliberately different from the golden-set
# prompts so that a score here measures generalisation / improvement,
# not just memorisation of the regression battery.
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        "math_product",
        "Answers a two-digit multiplication question",
        lambda cfg: "What is 13 times 17?",
        _check_math_product,
    ),
    EvalCase(
        "json_list_abc",
        "Returns a simple JSON array when asked",
        lambda cfg: 'Return this list as JSON: ["a", "b", "c"]. Output only the JSON.',
        _check_json_list_abc,
    ),
    EvalCase(
        "remember_color",
        "Saves a fact when asked to remember it",
        lambda cfg: "Remember that my favorite color is blue.",
        _check_remember_color,
    ),
    EvalCase(
        "weekly_reminder",
        "Schedules a recurring reminder",
        lambda cfg: "Remind me every Monday at 10am to review my notes.",
        _check_weekly_reminder,
    ),
    EvalCase(
        "web_search_fact",
        "Searches instead of guessing a current fact",
        lambda cfg: "What is the capital of France? Search the web and answer from the results.",
        _check_web_search,
    ),
    EvalCase(
        "browser_read_site",
        "Uses the controllable browser to read a named site",
        lambda cfg: "Read what example.com says.",
        _check_browser_read,
    ),
    EvalCase(
        "run_code_primes",
        "Runs code (or answers directly) for a small computation",
        lambda cfg: "Write and run Python that prints the first five prime numbers.",
        _check_run_code_primes,
    ),
    EvalCase(
        "open_app",
        "Emits a native command to open an application",
        lambda cfg: "Open the Safari app.",
        _check_open_app,
    ),
    EvalCase(
        "who_are_you",
        "Identifies itself by name without confusing itself with the user",
        lambda cfg: "Who are you?",
        _check_who_are_you,
    ),
]


class EvalResult(NamedTuple):
    pass_count: int
    total: int
    mean_latency: float
    tasks: list[dict[str, Any]]


def run_eval_set(
    model,
    tokenizer,
    generate_fn: Callable,
    sampler,
    system_prompt: str,
    config: dict[str, Any],
    max_tokens: int | None = None,
    cases: list[EvalCase] | None = None,
) -> EvalResult:
    """Run every eval case as a single-turn, tool-free generation and grade it."""
    cases = cases if cases is not None else EVAL_CASES
    # Eval tasks include short scripts; use a longer token budget than normal
    # chat so code-generation cases are not truncated before grading.
    max_tokens = max_tokens or int(config.get("eval", {}).get("max_eval_tokens", 512))
    context = system_prompt + prompts.env_note() + prompts.time_note()

    tasks: list[dict[str, Any]] = []
    total_latency = 0.0

    print(f"  [Eval] Running {len(cases)} cases...")
    for i, case in enumerate(cases, 1):
        print(f"  [Eval] {i}/{len(cases)} {case.id}...", end=" ", flush=True)
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": case.prompt_fn(config)},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        start = time.perf_counter()
        try:
            raw_reply = generate_fn(
                model, tokenizer, prompt=chat_prompt, sampler=sampler,
                max_tokens=max_tokens, verbose=False,
            ).strip()
        except Exception as e:
            tasks.append({
                "id": case.id,
                "passed": False,
                "latency": 0.0,
                "output": f"[generation error: {e}]",
                "error": str(e),
            })
            print("ERROR")
            continue

        latency = time.perf_counter() - start
        total_latency += latency
        tools = tooling.parse_tools(raw_reply, enabled_groups=None)
        display = tooling.strip_tool_tags(raw_reply)
        try:
            passed = bool(case.check(display, tools, config))
        except Exception as e:
            passed = False
            err = str(e)
        else:
            err = None

        tasks.append({
            "id": case.id,
            "passed": passed,
            "latency": round(latency, 3),
            "output": raw_reply,
            "error": err,
        })
        print("PASS" if passed else "FAIL")

    pass_count = sum(1 for t in tasks if t["passed"])
    mean_latency = total_latency / len(cases) if cases else 0.0
    print(f"  [Eval] {pass_count}/{len(cases)} cases passed.")
    return EvalResult(pass_count, len(cases), round(mean_latency, 3), tasks)


def _unload_model(model):
    """Drop a loaded MLX model so the next load doesn't compete for RAM."""
    del model
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _load_model_with_adapter(config: dict[str, Any]) -> tuple[Any, Any]:
    adapter_config = constants.ADAPTER_DIR / "adapter_config.json"
    if adapter_config.exists():
        print("  [Eval] Loading model with current adapter...")
        return load(config["model_name"], adapter_path=str(constants.ADAPTER_DIR))
    print("  [Eval] No adapter found; loading base model only...")
    return load(config["model_name"])


def _load_base_model(config: dict[str, Any]) -> tuple[Any, Any]:
    print("  [Eval] Loading base model without adapter...")
    return load(config["model_name"])


def _make_sampler(config: dict[str, Any]):
    return make_sampler(
        temp=config["agent"]["temperature"],
        top_p=config["agent"]["top_p"],
    )


def run_lora_benchmark(
    config: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    generate_fn: Callable | None = None,
    max_tokens: int | None = None,
) -> Path:
    """Benchmark the current LoRA adapter against the base model.

    Loads the model twice (with adapter, then base-only), runs the held-out
    eval set on each, and writes a JSON report. Returns the report path.
    """
    config = config or {}
    generate_fn = generate_fn or generate

    system_prompt = prompts.build_system_prompt(
        config.get("assistant_name", "Symbio"),
        config.get("user_name", "User"),
    )
    sampler = _make_sampler(config)

    # Adapter run
    model, tokenizer = _load_model_with_adapter(config)
    adapter_result = run_eval_set(
        model, tokenizer, generate_fn, sampler, system_prompt, config, max_tokens=max_tokens)
    _unload_model(model)

    # Base run
    model, tokenizer = _load_base_model(config)
    base_result = run_eval_set(
        model, tokenizer, generate_fn, sampler, system_prompt, config, max_tokens=max_tokens)
    _unload_model(model)

    adapter_exists = (constants.ADAPTER_DIR / "adapter_config.json").exists()
    delta = adapter_result.pass_count - base_result.pass_count

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": config.get("model_name"),
        "adapter_present": adapter_exists,
        "adapter_dir": str(constants.ADAPTER_DIR) if adapter_exists else None,
        "base": {
            "score": base_result.pass_count,
            "total": base_result.total,
            "accuracy": round(base_result.pass_count / base_result.total, 4) if base_result.total else 0,
            "mean_latency": base_result.mean_latency,
            "tasks": base_result.tasks,
        },
        "adapter": {
            "score": adapter_result.pass_count,
            "total": adapter_result.total,
            "accuracy": round(adapter_result.pass_count / adapter_result.total, 4) if adapter_result.total else 0,
            "mean_latency": adapter_result.mean_latency,
            "tasks": adapter_result.tasks,
        },
        "delta": delta,
    }

    report["improved"] = [
        t["id"] for t in adapter_result.tasks
        if t["passed"] and not next(
            (b for b in base_result.tasks if b["id"] == t["id"]), {}).get("passed", False)
    ]
    report["regressed"] = [
        t["id"] for t in adapter_result.tasks
        if not t["passed"] and next(
            (b for b in base_result.tasks if b["id"] == t["id"]), {}).get("passed", False)
    ]
    report["unchanged_pass"] = [
        t["id"] for t in adapter_result.tasks
        if t["passed"] and next(
            (b for b in base_result.tasks if b["id"] == t["id"]), {}).get("passed", False)
    ]
    report["unchanged_fail"] = [
        t["id"] for t in adapter_result.tasks
        if not t["passed"] and not next(
            (b for b in base_result.tasks if b["id"] == t["id"]), {}).get("passed", False)
    ]

    if output_path is None:
        output_path = constants.PROJECT_DIR / f"benchmark_lora_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    output_path = Path(output_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  [Eval] Base:     {base_result.pass_count}/{base_result.total}")
    print(f"  [Eval] Adapter:  {adapter_result.pass_count}/{adapter_result.total}")
    print(f"  [Eval] Delta:    {delta:+d}")
    print(f"  [Eval] Report:   {output_path}")
    return output_path
