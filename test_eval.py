"""Tests for the held-out LoRA evaluation harness."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symbio import constants
from symbio.app import eval as eval_module
from symbio.app.eval import EVAL_CASES, EvalCase, run_eval_set, run_lora_benchmark


def _fake_tokenizer():
    tok = MagicMock()
    def apply_chat_template(messages, tokenize, add_generation_prompt, enable_thinking):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    tok.apply_chat_template = apply_chat_template
    return tok


def _make_generate_fn(replies: dict[str, str]):
    """Return a fake generate_fn that maps the user content in the chat prompt
    to a canned reply. The prompt passed by run_eval_set contains the user
    message after the system context; we extract it with a simple heuristic."""
    def generate_fn(model, tokenizer, prompt, sampler, max_tokens, verbose):
        user_part = prompt.split("\n")[-1]
        for case in EVAL_CASES:
            if case.prompt_fn({"assistant_name": "Symbio", "user_name": "User"}) in user_part:
                return replies.get(case.id, "I don't know.")
        return "unknown"
    return generate_fn


def test_run_eval_set_grades_canned_replies():
    model, tokenizer = MagicMock(), _fake_tokenizer()
    replies = {
        "math_product": "13 × 17 is 221.",
        "json_list_abc": '["a", "b", "c"]',
        "remember_color": "Got it. <note title='Color'>favorite color is blue</note>",
        "weekly_reminder": "<cron expr='0 10 * * 1'>review your notes</cron> Scheduled.",
        "web_search_fact": "<search>capital of France</search> Searching.",
        "browser_read_site": "<browse>https://example.com</browse> Opening it.",
        "run_code_primes": "<py>print([2, 3, 5, 7, 11])</py> Running.",
        "open_app": "<cmd>open -a 'Safari'</cmd> Opening Safari.",
        "who_are_you": "I am Symbio, your assistant.",
    }
    result = run_eval_set(
        model, tokenizer, _make_generate_fn(replies), MagicMock(),
        "system", {"assistant_name": "Symbio", "user_name": "User", "agent": {"max_reply_tokens": 256}},
    )
    assert result.pass_count == result.total
    assert result.total == len(EVAL_CASES)


def test_run_eval_set_counts_failures():
    model, tokenizer = MagicMock(), _fake_tokenizer()
    replies = {case.id: "wrong" for case in EVAL_CASES}
    result = run_eval_set(
        model, tokenizer, _make_generate_fn(replies), MagicMock(),
        "system", {"assistant_name": "Symbio", "user_name": "User", "agent": {"max_reply_tokens": 256}},
    )
    assert result.pass_count == 0


def test_run_lora_benchmark_writes_report(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(eval_module.constants, "ADAPTER_DIR", adapter_dir)

    base_replies = {case.id: "wrong" for case in EVAL_CASES}
    adapter_replies = {
        "math_product": "The product is 221.",
        "json_list_abc": '["a", "b", "c"]',
        "remember_color": "Got it. <note title='Color'>favorite color is blue</note>",
        "weekly_reminder": "<cron expr='0 10 * * 1'>review notes</cron> Scheduled.",
        "web_search_fact": "<search>capital of France</search> Searching.",
        "browser_read_site": "<browse>https://example.com</browse> Opening it.",
        "run_code_primes": "<py>print([2, 3, 5, 7, 11])</py> Running.",
        "open_app": "<cmd>open -a 'Safari'</cmd> Opening Safari.",
        "who_are_you": "I am Symbio, your assistant.",
    }

    call_log = []

    def fake_load(model_name, adapter_path=None):
        call_log.append((model_name, adapter_path))
        tok = _fake_tokenizer()
        if adapter_path is not None:
            return MagicMock(name="adapter_model"), tok
        return MagicMock(name="base_model"), tok

    def fake_generate(model, tokenizer, prompt, sampler, max_tokens, verbose):
        adapter_loaded = getattr(model, "_mock_name", None) == "adapter_model"
        replies = adapter_replies if adapter_loaded else base_replies
        user_part = prompt.split("\n")[-1]
        for case in EVAL_CASES:
            if case.prompt_fn({"assistant_name": "Symbio", "user_name": "User"}) in user_part:
                return replies.get(case.id, "unknown")
        return "unknown"

    output = tmp_path / "report.json"
    config = {
        "model_name": "test/model",
        "assistant_name": "Symbio",
        "user_name": "User",
        "agent": {"temperature": 0.1, "top_p": 0.9, "max_reply_tokens": 256},
    }

    with patch.object(eval_module, "load", fake_load), \
         patch.object(eval_module, "generate", fake_generate):
        path = run_lora_benchmark(config, output_path=output)

    assert path == output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["model_name"] == "test/model"
    assert report["adapter_present"] is True
    assert report["base"]["score"] == 0
    assert report["adapter"]["score"] == len(EVAL_CASES)
    assert report["delta"] == len(EVAL_CASES)
    assert len(call_log) == 2
    assert call_log[0][1] == str(adapter_dir)
    assert call_log[1][1] is None


def test_custom_eval_case_check_receives_config():
    received = {}

    def check(display, tools, config):
        received["config"] = config
        received["display"] = display
        return True

    case = EvalCase("custom", "test", lambda cfg: "hi", check)
    model, tokenizer = MagicMock(), _fake_tokenizer()

    def generate_fn(model, tokenizer, prompt, sampler, max_tokens, verbose):
        return "hello"

    run_eval_set(
        model, tokenizer, generate_fn, MagicMock(),
        "system", {"assistant_name": "A", "user_name": "U", "agent": {"max_reply_tokens": 256}},
        cases=[case],
    )
    assert received["config"]["assistant_name"] == "A"
    assert received["display"] == "hello"
