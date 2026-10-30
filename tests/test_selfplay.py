"""bench/self_repair: the proposer's parser and the self-play learning rule, without a model."""
import importlib.util
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench" / "self_repair"


def load(name, monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.syspath_prepend(str(BENCH))
    spec = importlib.util.spec_from_file_location(f"_sp_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SRC = '''def pick(xs, n):
    out = []
    for x in xs:
        if x > n:
            out.append(x)
    return out
'''


def propose(monkeypatch):
    return load("propose", monkeypatch)


def test_parse_takes_one_changed_line_and_keeps_its_indent(monkeypatch):
    p = propose(monkeypatch)
    lines = SRC.splitlines(True)
    k, new = p.parse("<<<< OLD\nif x > n:\n====\nif x >= n:\n>>>> NEW", lines, 1, 6)
    assert k == 3 and new == "        if x >= n:\n"


def test_parse_rejects_what_is_not_a_bug(monkeypatch):
    p = propose(monkeypatch)
    lines = SRC.splitlines(True)
    block = "<<<< OLD\n{}\n====\n{}\n>>>> NEW"
    assert p.parse("just text", lines, 1, 6) == "no OLD/NEW block"
    assert p.parse(block.format("if y > n:", "if y < n:"), lines, 1, 6) == "OLD line not in the function"
    assert p.parse(block.format("if x > n:", "    if x > n:"), lines, 1, 6) == "NEW is unchanged"
    assert p.parse(block.format("if x > n:", "# if x > n:"), lines, 1, 6) == "NEW is a comment or blank"


def test_parse_accepts_context_and_loose_fences_but_strips_hints(monkeypatch):
    p = propose(monkeypatch)
    lines = SRC.splitlines(True)
    reply = "<<\nOLD\n    for x in xs:\n        if x > n:\n=======\n    for x in xs:\n        if x < n:  # BUG: flipped\n>>>>\nNote: subtle."
    assert p.parse(reply, lines, 1, 6) == (3, "        if x < n:\n")
    two = "<<<< OLD\nout = []\nif x > n:\n====\nout = {}\nif x < n:\n>>>> NEW"
    assert p.parse(two, lines, 1, 6) == "more than one line"


def test_parse_takes_the_last_block_and_a_bare_new_separator(monkeypatch):
    p = propose(monkeypatch)
    lines = SRC.splitlines(True)
    reply = ("<<<< OLD\nout = []\n====\nout = {}\n>>>> NEW\nWait, that is not tested.\n"
             "<<\nOLD\nif x > n:\nNEW\nif x >= n:\n>>")
    assert p.parse(reply, lines, 1, 6) == (3, "        if x >= n:\n")


def test_parse_refuses_a_line_the_tests_never_run(monkeypatch):
    p = propose(monkeypatch)
    lines = SRC.splitlines(True)
    block = "<<<< OLD\nout.append(x)\n====\nout.insert(0, x)\n>>>> NEW"
    assert p.parse(block, lines, 1, 6, allowed={3}) == "line the tests never run"
    assert p.parse(block, lines, 1, 6, allowed={4})[0] == 4


def test_only_functions_the_tests_run_are_offered(monkeypatch):
    p = propose(monkeypatch)
    lines = SRC.splitlines(True)
    assert p.tested_functions([(1, 6)], lines, {1, 2, 3, 4, 6}) == [(1, 6, [2, 3, 4, 6])]
    assert p.tested_functions([(1, 6)], lines, {1, 2}) == []


def test_functions_lists_bodies_long_enough_to_bug(monkeypatch):
    p = propose(monkeypatch)
    assert p.functions(SRC) == [(1, 6)]
    assert p.functions("def f():\n    return 1\n") == []


def test_only_tasks_at_the_edge_become_training_data(monkeypatch, tmp_path):
    sp = load("selfplay", monkeypatch, MODEL="m", CLONE=tmp_path, OUT=tmp_path / "out")
    tasks = [{"prompt": f"break {i}", "reply": f"bug {i}"} for i in range(3)]

    def att(i, n, ok):
        return {"i": i, "attempt": n, "fixed": ok, "prompts": [f"fix {i}"], "replies": [f"patch {i}.{n}"]}
    attempts = ([att(0, n, True) for n in range(4)]                       # always solved: too easy
                + [att(1, n, n == 2) for n in range(4)]                   # 1 in 4: the edge
                + [att(2, n, False) for n in range(4)])                   # never solved: too hard
    solver, proposer, rates = sp.edge_data(tasks, attempts, 1)
    assert rates == [1.0, 0.25, 0.0]
    assert [s["messages"][1]["content"] for s in solver] == ["patch 1.2"]
    assert [s["messages"][1]["content"] for s in proposer] == ["bug 1"]


def test_held_subset_is_spread_evenly(monkeypatch, tmp_path):
    sp = load("selfplay", monkeypatch, MODEL="m", CLONE=tmp_path, OUT=tmp_path)
    held = tmp_path / "held.json"
    held.write_text(json.dumps([{"i": i} for i in range(40)]))
    monkeypatch.setattr(sp.rsi, "HELD", held)
    assert sp.held_bugs(0) == held
    sub = json.loads(sp.held_bugs(16).read_text())
    assert [b["i"] for b in sub] == [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37]
