#!/usr/bin/env python3
"""The battery's call parser, against the reply shapes models really emit.

bfcl_simple's grading is judge-free ONLY if the parser reads every shape a
model actually answers in — a parser-limited low score measures the parser,
not the model, and the whole board's ranking inherits the blind. These pin
the shapes seen in the wild; each one traces to a measured miss.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))
import bfcl


def test_a_plain_json_answer():
    name, args = bfcl.parse_call('{"name": "cd", "arguments": {"path": "/tmp"}}')
    assert name == "cd" and args == {"path": "/tmp"}


def test_a_python_call_answer():
    name, args = bfcl.parse_call("fn(path='/tmp', mode='w')")
    assert name == "fn" and args == {"path": "/tmp", "mode": "w"}


def test_a_json_answer_inside_a_code_fence():
    name, args = bfcl.parse_call('```json\n{"name": "ls", "arguments": {}}\n```')
    assert name == "ls"


def test_a_harmony_channelled_reply_is_cut_to_the_final_channel():
    """gpt-oss's harmony wraps its reasoning in its own channels; the answer
    lives in the LAST <|message|> segment. Reasoning braces would otherwise
    aim the greedy JSON regex at the thought (measured: gpt-oss-20b read
    39.2% on bfcl_simple with the channels left in, with calls visibly
    correct inside the analysis text; 2026-10-16)."""
    reply = (
        "<|channel|>analysis<|message|>We need to call math.hypot with "
        'x=4, y=5 {"draft": true}. The function signature: x, y.\n'
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        '{"name": "math_hypot", "arguments": {"x": 4, "y": 5}}'
    )
    name, args = bfcl.parse_call(reply)
    assert name == "math_hypot"
    assert args == {"x": 4, "y": 5}, args


def test_a_harmony_reply_whose_answer_never_leaves_analysis_parses_the_thought():
    """When truncation ate the final channel, the analysis text's own JSON
    is the best read of the model's intent — the strip must not LOSE that
    call: no <|message|> tail means the text stands as-is and the JSON
    regex still runs on it."""
    reply = (
        '<|channel|>analysis<|message|>So output JSON: '
        '{"name": "solve_quadratic", "arguments": {"a": 3, "b": -11, "c": -4}}'
    )
    name, args = bfcl.parse_call(reply)
    assert name == "solve_quadratic"
    assert args == {"a": 3, "b": -11, "c": -4}