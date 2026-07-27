#!/usr/bin/env python3
"""Stress-test the Symbio chat agent over many turns.

Mode 1 (default): use a fake model/tokenizer so we stress Python logic,
memory churn, and prompt growth without loading a multi-GB MLX model.
Mode 2 (--real): load the configured model and run a shorter live stress
with aggressive memory monitoring.
"""

import argparse
import builtins
import gc
import json
import os
import random
import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).parent))

from symbio import constants
from symbio.app import chat, sandbox, sessions, tooling, training, web
from symbio.app import config as app_config
from test_utils import preserve_training_state


class FakeTokenizer:
    """Deterministic tokenizer stand-in."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def encode(self, text, add_special_tokens=True):
        return text.split()


# Deterministic scripted replies for the fake model.
_STRESS_REPLIES = [
    # Simple chat
    "Hey! How can I help?",
    "That's interesting.",
    "Got it.",
    # Tool tags
    "<search>latest AI news</search> Searching now.",
    "<py>print(2 + 2)</py> Running that.",
    "<note title='Test'>Stress test note.</note> Saved.",
    "<cmd>echo hello</cmd> Running it.",
    # Browser tags
    "<browse>https://example.com</browse> Opening the page.",
    "<click>More</click> Clicking now.",
    "<press>down</press> Pressing down.",
    "<browser_close /> Browser closed.",
    # Cron
    "<cron expr='0 9 * * *'>stretch</cron> Reminder set.",
    # Config
    "<config set='agent.temperature'>0.5</config> Done.",
    # Longer prose to bloat history
    "Here is a somewhat longer reply so we can see whether the history trimmer and context budgets keep the prompt from growing without bound. " * 3,
    # A correction-style reply
    "I think the answer is 42.",
]


def _peak_rss_mb() -> float:
    """Peak resident set size in MB (this process + children).

    macOS returns ru_maxrss in bytes; Linux returns it in kilobytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _current_rss_mb(pid: int | None = None) -> float:
    """Approximate current RSS in MB via /proc or ps fallback."""
    pid = pid or os.getpid()
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        kb = int(out.strip())
        return kb / 1024.0
    except Exception:
        pass
    return _peak_rss_mb()


def _make_fake_stream(model_replies):
    order = iter(model_replies)

    def fake_stream(model, tokenizer, prompt, max_tokens=256, sampler=None,
                    prompt_cache=None, **kwargs):
        reply = next(order, "Nothing more to say.")
        words = reply.split(" ")
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            yield type("R", (), {"text": text, "token": hash(word) & 0xFFFF})()

    return fake_stream


def _run_fake_stress(turns: int) -> dict:
    """Run `turns` agent turns with a fake model and measure."""
    print(f"\n[Stress] Starting fake-model run: {turns} turns")

    real_input = builtins.input
    real_load = chat.load
    real_generate = chat.generate
    real_stream_generate = chat.stream_generate
    real_make_cache = chat.make_prompt_cache
    real_can_trim = chat.can_trim_prompt_cache
    real_trim = chat.trim_prompt_cache
    real_web_search = web.web_search
    real_read_page = web.read_page
    real_browser_cls = chat.BrowserSession

    class _StressBrowser:
        def __init__(self, confirm_fn=None):
            self._open = False

        def open(self, url):
            self._open = True
            return f"Opened browser at {url}. Page title: Stress"

        def get_text(self):
            return "Stress page text."

        def click(self, selector="", text=""):
            return "Clicked."

        def type_text(self, text, selector="", press_enter=False):
            return f"Typed '{text}'."

        def scroll(self, direction="down", amount=0):
            return f"Scrolled {direction}."

        def press(self, key):
            return f"Pressed {key}."

        def close(self):
            self._open = False
            return "Browser closed."

    builtins.input = lambda prompt="": "stress turn"
    chat.load = lambda *a, **k: (object(), FakeTokenizer())
    chat.generate = lambda *a, **k: "fake reply"
    chat.make_prompt_cache = lambda model: []
    chat.can_trim_prompt_cache = lambda cache: True
    chat.trim_prompt_cache = lambda cache, n: cache
    web.web_search = lambda query, config: (True, f"Mock results for '{query}'.")
    web.read_page = lambda url, config: (True, f"Mock page text for {url}.")
    chat.BrowserSession = _StressBrowser

    errors = []
    timings: list[float] = []
    rss_samples: list[float] = []

    try:
        with preserve_training_state(adapters=True):
            config = app_config.load_config()
            config["memory"]["enabled"] = True
            config["memory"]["nudge_interval"] = 0  # disable nudges for speed
            config["rag"]["enabled"] = True
            config["rag"]["sources"] = ["notes", "sessions"]
            # Use the fake browser class so open/close/click is instant.
            config["browser"]["enabled"] = True
            # Don't trigger real LoRA training during stress; we only want to
            # exercise chat/RAG/history paths.
            config["learn"]["auto_train"] = False

            replies = [_STRESS_REPLIES[i % len(_STRESS_REPLIES)] for i in range(turns)]
            chat.stream_generate = _make_fake_stream(replies)

            session = chat.ChatSession(
                config,
                model=object(),
                tokenizer=FakeTokenizer(),
                adapter_loaded=False,
                output_fn=lambda *a, **k: None,
                generate_fn=lambda *a, **k: "fake",
                stream_chunk_fn=None,
            )

            for i in range(turns):
                start = time.perf_counter()
                try:
                    session._agent_turn(f"stress turn {i}")
                except Exception as e:
                    errors.append((i, type(e).__name__, str(e)[:200]))
                    print(f"[Stress] ERROR on turn {i}: {e}")
                    break
                timings.append((time.perf_counter() - start) * 1000)
                rss_samples.append(_current_rss_mb())

                if (i + 1) % 10 == 0:
                    print(
                        f"[Stress] {i + 1:4}/{turns} turns | "
                        f"last={timings[-1]:.1f}ms | "
                        f"rss={rss_samples[-1]:.1f}MB | "
                        f"history={len(session.history)}"
                    )

            return {
                "mode": "fake",
                "turns_completed": i + 1 if not errors else i,
                "errors": errors,
                "avg_turn_ms": sum(timings) / len(timings) if timings else 0,
                "max_turn_ms": max(timings) if timings else 0,
                "rss_start": rss_samples[0] if rss_samples else 0,
                "rss_end": rss_samples[-1] if rss_samples else 0,
                "rss_growth_mb": (rss_samples[-1] - rss_samples[0]) if rss_samples else 0,
                "max_rss_mb": _peak_rss_mb(),
                "history_len": len(session.history),
            }
    finally:
        builtins.input = real_input
        chat.load = real_load
        chat.generate = real_generate
        chat.stream_generate = real_stream_generate
        chat.make_prompt_cache = real_make_cache
        chat.can_trim_prompt_cache = real_can_trim
        chat.trim_prompt_cache = real_trim
        web.web_search = real_web_search
        web.read_page = real_read_page
        chat.BrowserSession = real_browser_cls


def _run_real_stress(turns: int) -> dict:
    """Load the real model and run a short stress with memory tracking."""
    print(f"\n[Stress] Starting REAL-MODEL run: {turns} turns")
    print("[Stress] WARNING: this loads the configured MLX model into RAM.")

    # Try a lightweight memory snapshot baseline with tracemalloc
    tracemalloc.start()

    errors = []
    timings: list[float] = []
    rss_samples: list[float] = []

    config = app_config.load_config()
    config["memory"]["nudge_interval"] = 0

    # User inputs: just repetitive simple prompts
    user_inputs = ["hi", "what is 2+2", "open example.com", "close the browser", "save a note"]
    input_iter = iter(user_inputs * (turns // len(user_inputs) + 1))

    real_input = builtins.input
    builtins.input = lambda prompt="": next(input_iter)

    try:
        session = chat.ChatSession(
            config,
            input_fn=builtins.input,
            output_fn=print,
            stream_chunk_fn=lambda s: print(s, end="", flush=True),
        )
        rss_samples.append(_current_rss_mb())

        for i in range(turns):
            gc.collect()
            start = time.perf_counter()
            try:
                session._agent_turn(next(input_iter))
            except Exception as e:
                errors.append((i, type(e).__name__, str(e)[:200]))
                print(f"[Stress] ERROR on turn {i}: {e}")
                break
            timings.append((time.perf_counter() - start) * 1000)
            rss_samples.append(_current_rss_mb())

            if (i + 1) % 2 == 0:
                print(
                    f"[Stress] {i + 1:3}/{turns} | "
                    f"last={timings[-1]:.0f}ms | "
                    f"rss={rss_samples[-1]:.0f}MB"
                )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "mode": "real",
            "turns_completed": i + 1 if not errors else i,
            "errors": errors,
            "avg_turn_ms": sum(timings) / len(timings) if timings else 0,
            "max_turn_ms": max(timings) if timings else 0,
            "rss_start": rss_samples[0] if rss_samples else 0,
            "rss_end": rss_samples[-1] if rss_samples else 0,
            "rss_growth_mb": (rss_samples[-1] - rss_samples[0]) if rss_samples else 0,
            "max_rss_mb": _peak_rss_mb(),
            "tracemalloc_peak_mb": peak / 1024 / 1024,
        }
    finally:
        builtins.input = real_input
        try:
            session.browser.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Stress-test the Symbio chat agent.")
    parser.add_argument("--real", action="store_true",
                        help="Load the real MLX model (uses significant RAM).")
    parser.add_argument("--turns", type=int, default=100,
                        help="Number of turns to run (default 100 fake, 10 real).")
    args = parser.parse_args()

    if args.real:
        turns = args.turns if args.turns != 100 else 10
        result = _run_real_stress(turns)
    else:
        result = _run_fake_stress(args.turns)

    print("\n[Stress] Summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
