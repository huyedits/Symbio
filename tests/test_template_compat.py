"""Tool results rendered so a Qwen3.5-family template finds the user's request.

Ternary Bonsai 27B, asked to fix a bug, read the file and then asked "What would
you like me to do with this?": its template walks back for the last user turn
that is not purely a <tool_response>, and Symbio's observation turns began
"[System observation:", so the request it found was the observation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from symbio.app import template_compat

QWEN35 = Path("/Users/huygpt/Downloads/agi/models/Qwen3.5-0.8B-4bit")


def _observation(name, output):
    return (f"[System observation: {output}]\n<tool_response>"
            f"{json.dumps({'name': name, 'content': output})}</tool_response>")


def test_an_observation_becomes_its_tool_response():
    messages = [{"role": "user", "content": "fix x.py"},
                {"role": "user", "content": _observation("read_file", "print(1)")}]
    out = template_compat.native_tool_messages(messages)
    assert out[0] == messages[0]
    assert out[1]["content"].startswith("<tool_response>")
    assert out[1]["content"].endswith("</tool_response>")
    assert messages[1]["content"].startswith("[System observation:")  # not mutated


def test_a_nudge_without_a_tool_response_is_left_as_it_is():
    nudge = {"role": "user", "content": "[System observation: your last reply was empty.]"}
    assert template_compat.native_tool_messages([nudge]) == [nudge]


def test_only_templates_that_walk_for_the_request_are_changed():
    class Tok:
        chat_template = "{{ messages }}"

    assert not template_compat.adapt(Tok())


@pytest.mark.skipif(not QWEN35.is_dir(), reason="needs the local Qwen3.5-0.8B tokenizer")
def test_the_request_survives_a_tool_call_through_mlx_lms_tokenizer():
    """Through mlx_lm's TokenizerWrapper — which forwards attribute writes to
    the HF tokenizer it wraps, so patching the wrong one recursed forever."""
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(QWEN35)
    assert template_compat.adapt(tok)
    assert not template_compat.adapt(tok)  # idempotent
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "Fix x.py please."},
                {"role": "assistant",
                 "content": '<tool_call>{"name": "read_file", "arguments": {}}</tool_call>'},
                {"role": "user", "content": _observation("read_file", "print(1)")}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    assert "[System observation:" not in text
    assert "<|im_start|>user\n<tool_response>" in text
