"""Tests for the benchmark harness itself.

A benchmark that is never tested measures its own bugs. This project has the
scars: a harness that never stripped `<|im_end|>` scored a working model
14/100, and a 7-task suite whose validators were loose enough that every model
scored 100/100 — including one that answered "My name is Qwen." to an identity
question for an assistant called Caine.

Two directions, and both are needed. Only checking that correct answers pass
lets a checker that passes *everything* through; only checking that wrong
answers fail lets a checker that fails everything through.

The format cases are the ones that earn their keep. Scoring a gold answer that
this module itself formatted only proves the parser can read its own writing —
so each correct call is re-spelled the way real models actually emit them.
That is what caught `parse_call` returning None for every Python-syntax call
containing a dict argument.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import bfcl  # noqa: E402
import harness  # noqa: E402

N = 150

# HumanEval/32's own test is broken upstream — `_poly(*candidate(*inp), inp)`
# unpacks a float. The harness is right to fail it; the item is excluded so a
# dataset defect does not read as a harness defect.
BROKEN_UPSTREAM = {"HumanEval/32"}


def _concrete(alternatives):
    """One concrete value from the key's encoding.

    Every leaf in the key is a list of acceptable values, including leaves
    nested inside dicts and lists, so building a realistic answer means
    unwrapping all the way down rather than taking the first element and
    stopping.
    """
    alts = alternatives if isinstance(alternatives, list) else [alternatives]
    value = None
    for candidate in alts:
        if candidate not in ("", None):
            value = candidate
            break
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            inner = _concrete(v)
            if inner is not None:
                out[k] = inner
        return out
    if isinstance(value, list):
        return [_concrete([v]) if not isinstance(v, (dict, list)) else _concrete_raw(v)
                for v in value]
    return value


def _concrete_raw(value):
    if isinstance(value, dict):
        return {k: _concrete(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_concrete_raw(v) for v in value]
    return value


def _args_from_truth(row):
    name, params = next(iter(row["ground_truth"][0].items()))
    args = {}
    for p, acc in params.items():
        if not acc:
            continue
        v = _concrete(acc)
        if v is None:
            continue
        args[p] = v
    return name, args


# ---- correct answers must pass ----

@pytest.mark.parametrize("cat", list(bfcl.CATEGORIES))
def test_bfcl_gold_answers_pass(cat):
    rows = bfcl.load(cat, N)
    hits = sum(bfcl.score(r, bfcl.gold_response(r)) for r in rows)
    assert hits / len(rows) > 0.98, f"{cat}: the answer key does not pass its own checker"


def test_gsm8k_gold_answers_pass():
    rows = harness.load_gsm8k(N)
    hits = sum(harness.score_gsm8k(r, f"The answer is {harness.gsm8k_gold(r)}") for r in rows)
    assert hits == len(rows)


def test_humaneval_canonical_solutions_pass():
    rows = [r for r in harness.load_humaneval(N) if r["task_id"] not in BROKEN_UPSTREAM]
    hits = sum(harness.score_humaneval(r, "```python\n" + r["prompt"] + r["canonical_solution"] + "\n```")
               for r in rows)
    assert hits == len(rows), "a canonical solution failing means a broken checker"


# ---- the same correct answer, spelled the way models really spell it ----

def _spellings(name, args):
    j = json.dumps({"name": name, "arguments": args})
    py = f"{name}(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
    return {
        "plain json": j,
        "fenced json": f"```json\n{j}\n```",
        "tool_call tags": f"<tool_call>\n{j}\n</tool_call>",
        "prose around json": f"Sure, I'll do that:\n{j}\nLet me know!",
        "parameters key": json.dumps({"name": name, "parameters": args}),
        "bare fn mapping": json.dumps({name: args}),
        "list wrapped": json.dumps([{"name": name, "arguments": args}]),
        "thinking block": f"<think>reasoning</think>\n{j}",
        "python call": py,
        "fenced python": f"```python\n{py}\n```",
        "prose around python": f"I'll call:\n{py}",
    }


@pytest.mark.parametrize("spelling", list(_spellings("f", {"a": 1})))
def test_a_correct_call_passes_however_it_is_spelled(spelling):
    rows = bfcl.load("bfcl_simple", N)
    hits = 0
    for r in rows:
        name, args = _args_from_truth(r)
        hits += bool(bfcl.score(r, _spellings(name, args)[spelling]))
    assert hits / len(rows) > 0.95, (
        f"{spelling!r} is a format models emit; scoring it wrong measures the "
        f"parser, not the model")


def test_nested_arguments_score_correctly():
    """The answer key wraps every leaf in a list of alternatives, so nested
    arguments have to be compared by recursing, not by flat equality."""
    rows = bfcl.load("bfcl_simple")
    r = next(x for x in rows if x["id"] == "simple_python_94")
    good = {"user_id": 43523, "database": "CustomerInfo",
            "update_info": {"name": "John Doe", "email": "johndoe@email.com"}}
    bad = json.loads(json.dumps(good)); bad["update_info"]["name"] = "WRONG"
    assert bfcl.score(r, json.dumps({"name": "update_user_info", "arguments": good}))
    assert not bfcl.score(r, json.dumps({"name": "update_user_info", "arguments": bad}))


def test_list_of_dicts_argument_scores_correctly():
    rows = bfcl.load("bfcl_simple")
    r = next(x for x in rows if x["id"] == "simple_python_96")
    assert bfcl.score(r, "database.query(table='user', conditions=["
                         "{'field':'age','operation':'>','value':'25'},"
                         "{'field':'job','operation':'=','value':'engineer'}])")


# ---- wrong answers must fail ----

@pytest.mark.parametrize("cat", list(bfcl.CATEGORIES))
@pytest.mark.parametrize("junk", ["", "42", "I don't know.", "{}", "null"])
def test_bfcl_rejects_degenerate_answers(cat, junk):
    rows = bfcl.load(cat, N)
    assert sum(bfcl.score(r, junk) for r in rows) / len(rows) < 0.02


def test_bfcl_rejects_the_right_function_with_wrong_arguments():
    """Name-only credit would make the suite a function-picker test and hide
    exactly what quantisation damages: the argument values."""
    rows = bfcl.load("bfcl_simple", N)
    hits = 0
    for r in rows:
        name, _ = _args_from_truth(r)
        hits += bool(bfcl.score(r, json.dumps({"name": name, "arguments": {}})))
    assert hits / len(rows) < 0.05


def test_bfcl_rejects_hallucinated_parameters():
    rows = bfcl.load("bfcl_simple", N)
    hits = 0
    for r in rows:
        name, args = _args_from_truth(r)
        hits += bool(bfcl.score(r, json.dumps(
            {"name": name, "arguments": dict(args, not_a_real_param="x")})))
    assert hits / len(rows) < 0.05


def test_gsm8k_rejects_a_wrong_number():
    rows = harness.load_gsm8k(N)
    hits = sum(harness.score_gsm8k(r, f"The answer is {float(harness.gsm8k_gold(r)) + 1}")
               for r in rows)
    assert hits == 0


def test_humaneval_rejects_a_stub():
    rows = harness.load_humaneval(40)
    hits = sum(harness.score_humaneval(r, "```python\n" + r["prompt"] + "    pass\n```")
               for r in rows)
    assert hits / len(rows) < 0.05


# ---- the known-leaky one, pinned so it cannot drift unnoticed ----

def test_ifeval_null_floor_is_known_and_bounded():
    """IFEval gives points for degenerate text: an empty answer satisfies
    'no commas' and 'no forbidden words'. The floor is real and roughly 23%,
    so only the margin above it is signal. Pinned so a change gets noticed."""
    rows = harness.load_ifeval(N)
    got = [harness.score_ifeval(r, "42") for r in rows]
    kept = [g for g in got if g is not None]
    floor = sum(bool(g) for g in kept) / len(kept)
    assert 0.10 < floor < 0.35, f"IFEval null floor moved to {floor:.0%}"
