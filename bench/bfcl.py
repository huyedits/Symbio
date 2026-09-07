"""BFCL AST categories: does the model emit the right call with the right args?

Judge-free. The answer key gives, per parameter, the list of acceptable
values, so grading is a structural comparison against that — no model decides
whether an answer was good. This measures the thing Symbio does for a living
and the thing 2-bit quantisation is most likely to break: getting the argument
values right, not just the function name.

Prompts ask for the JSON call format Symbio itself uses, rather than a
vendor-native tool-calling API, because that is the format the headmaster is
actually served with.
"""
import ast
import json
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"

CATEGORIES = {
    "bfcl_simple": ("BFCL_v4_simple_python.json", "ans_BFCL_v4_simple_python.json"),
    "bfcl_live": ("BFCL_v4_live_simple.json", "ans_BFCL_v4_live_simple.json"),
    "bfcl_multiple": ("BFCL_v4_multiple.json", "ans_BFCL_v4_multiple.json"),
}


def load(category, limit=None):
    data_f, ans_f = CATEGORIES[category]
    rows = [json.loads(l) for l in open(DATA / data_f) if l.strip()]
    answers = {r["id"]: r["ground_truth"]
               for r in (json.loads(l) for l in open(DATA / ans_f) if l.strip())}
    paired = [dict(r, ground_truth=answers[r["id"]]) for r in rows if r["id"] in answers]
    return paired[:limit] if limit else paired


def build_prompt(row):
    funcs = json.dumps(row["function"], indent=1)
    user = row["question"][0][0]["content"]
    return (
        "You have these functions available:\n\n"
        f"{funcs}\n\n"
        "Call the correct one for the request below. Reply with ONLY a JSON "
        'object of the form {"name": "<function>", "arguments": {...}} and '
        "nothing else. Omit optional arguments the request does not specify.\n\n"
        f"Request: {user}"
    )


_JSON = re.compile(r"\{.*\}", re.DOTALL)
_PYCALL = re.compile(r"([A-Za-z_][\w.]*)\s*\((.*)\)", re.DOTALL)


def _parse_python_call(text):
    """`fn(a=1, b='x')` — the spelling BFCL's own AST checker expects.

    Models trained on function calling emit this at least as often as JSON, and
    scoring it zero would measure our parser rather than the model.
    """
    m = _PYCALL.search(text.strip())
    if not m:
        return None
    try:
        node = ast.parse(m.group(0).strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    name = ast.unparse(node.func) if hasattr(ast, "unparse") else m.group(1)
    args = {}
    for kw in node.keywords:
        if kw.arg is None:
            return None
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    if node.args:
        return None  # positional args carry no parameter names to check
    return name, args


def parse_call(text):
    """The call the model meant, or None if it did not emit one.

    Both spellings are tried, and a failure in one falls through to the other:
    a Python call whose argument is a dict literal contains a `{`, so the JSON
    branch matches it, fails on the single quotes, and would otherwise report
    "no call" for a perfectly good answer.
    """
    t = re.sub(r"^\s*```(?:json|python)?|```\s*$", "", text.strip(), flags=re.MULTILINE)

    m = _JSON.search(t)
    if m:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            obj = None
        if isinstance(obj, list) and len(obj) == 1:
            obj = obj[0]
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("function") or obj.get("tool")
            args = obj.get("arguments", obj.get("parameters", obj.get("args")))
            if isinstance(name, str) and isinstance(args, dict):
                return name, args
            # Bare {"func_name": {...}} — the shape the answer key itself uses.
            if len(obj) == 1:
                k, v = next(iter(obj.items()))
                if isinstance(v, dict):
                    return k, v
    return _parse_python_call(t)


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _matches(given, candidate):
    """`given` against one concrete acceptable value from the key.

    The key encodes alternatives as a list at every leaf, so a nested dict's
    values are themselves lists of acceptable values and have to be walked
    that way. Comparing them flat marks correct nested calls wrong — the same
    false-negative that once scored a working model 14/100.
    """
    if isinstance(candidate, dict):
        if not isinstance(given, dict):
            return False
        for key, alts in candidate.items():
            if key not in given:
                alt_list = alts if isinstance(alts, list) else [alts]
                if not any(a in ("", None) for a in alt_list):
                    return False
                continue
            if not _value_ok(given[key], alts):
                return False
        return all(k in candidate for k in given)
    if isinstance(candidate, list):
        # A genuine list-valued argument: elements are concrete, not alternatives.
        if not isinstance(given, list) or len(given) != len(candidate):
            return False
        return all(_matches(g, c) for g, c in zip(given, candidate))
    gn, wn = _norm(given), _norm(candidate)
    if gn == wn:
        return True
    try:
        return float(gn) == float(wn)
    except (TypeError, ValueError):
        return False


def _value_ok(given, acceptable):
    """Does `given` match any acceptable value for this parameter?"""
    alts = acceptable if isinstance(acceptable, list) else [acceptable]
    return any(_matches(given, c) for c in alts)


def score(row, response):
    call = parse_call(response)
    if call is None:
        return False
    name, args = call
    for truth in row["ground_truth"]:
        for gt_name, gt_params in truth.items():
            if name != gt_name:
                continue
            ok = True
            for param, acceptable in gt_params.items():
                if not acceptable:
                    continue  # key lists no accepted value: nothing to check
                if param not in args:
                    # Omitting a parameter is right only when "" (or None) is
                    # listed as acceptable — that is how the key marks optional.
                    if not any(a in ("", None) for a in acceptable):
                        ok = False
                        break
                    continue
                if not _value_ok(args[param], acceptable):
                    ok = False
                    break
            # Arguments the function was never given are hallucinated.
            if ok and any(p not in gt_params for p in args):
                ok = False
            if ok:
                return True
    return False


def _concrete(alternatives):
    """One concrete value from the key's encoding of acceptable values.

    Every leaf is a list of alternatives, nested ones included, so this has to
    unwrap all the way down. Taking the first element and stopping produces a
    call still carrying the key's list-wrapping, which is not what a correct
    answer looks like.
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
        return [_concrete(v) if isinstance(v, list) else
                (_concrete([v]) if not isinstance(v, dict) else
                 {k: _concrete(x) for k, x in v.items()})
                for v in value]
    return value


def gold_response(row):
    """The answer key rendered as a model would have emitted it."""
    truth = row["ground_truth"][0]
    name, params = next(iter(truth.items()))
    args = {}
    for p, acceptable in params.items():
        if not acceptable:
            continue
        v = _concrete(acceptable)
        if v is None:
            continue
        args[p] = v
    return json.dumps({"name": name, "arguments": args})
