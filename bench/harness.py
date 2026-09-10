"""Judge-free benchmarks: IFEval, HumanEval+, GSM8K.

Every suite here is scored by a program, not by a model. That matters because
the previous 7-task benchmark scored Bonsai-27B and Qwen3-8B *both* 100/100 —
it had no power to separate anything, and one of its validators passed "My
name is Qwen." as a correct self-identification for an assistant called Caine.

So the harness validates itself before it is allowed to grade a model:

  gold   the dataset's own reference answers must score ~100%. Anything less
         is a broken checker, not a weak model. This is the check that would
         have caught the <|im_end|> bug that scored Qwen2.5-Coder 14/100.
  null   degenerate answers (empty, a constant, an echo of the prompt) must
         score ~0%. Anything more is validator leakage — points available
         without doing the task.
  live   per-item pass rates across models: items everyone passes or everyone
         fails carry no information. A suite of dead items is a suite that
         cannot tell you anything, whatever its headline number.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verifiers  # noqa: E402

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------- loaders --
def load_ifeval(limit=None):
    rows = [json.loads(l) for l in open(DATA / "ifeval_input_data.jsonl")]
    return rows[:limit] if limit else rows


def load_humaneval(limit=None):
    rows = [json.loads(l) for l in open(DATA / "test.jsonl")]
    return rows[:limit] if limit else rows


def load_gsm8k(limit=None):
    rows = [json.loads(l) for l in open(DATA / "gsm8k_test.jsonl")]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- scorers --
def score_ifeval(row, response):
    """Strict prompt-level: every instruction on the prompt must hold."""
    results = []
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        verdict = verifiers.check(iid, kw, response, row["prompt"])
        if verdict is verifiers.UNSUPPORTED:
            continue
        results.append(bool(verdict))
    if not results:
        return None  # nothing checkable — excluded from the denominator
    return all(results)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm8k_gold(row):
    return row["answer"].split("####")[-1].strip().replace(",", "")


def score_gsm8k(row, response):
    """Last number in the response, which is where a final answer lands."""
    nums = _NUM.findall(response.replace(",", ""))
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - float(gsm8k_gold(row))) < 1e-4
    except ValueError:
        return False


def _extract_code(text, prompt):
    """The model's function body, however it chose to wrap it."""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    return prompt + text


def score_humaneval(row, response, timeout=10):
    """Run the reference unit tests against the generated function."""
    code = _extract_code(response, row["prompt"])
    program = f"{code}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"
    # Deliberately not symbio.app.sandbox.run_sandboxed. That gives no
    # OS-level containment either — it is a denylist of BINARIES plus a cwd —
    # and the binary here is always sys.executable, so the list has nothing to
    # match. It would also move every candidate into one shared SANDBOX_DIR,
    # where they would see each other's files. A fresh temp cwd per candidate
    # with a timeout is the stronger isolation for this job, not the weaker
    # one. Real containment for model-written code needs a seatbelt profile,
    # which nothing in this project has yet.
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "cand.py"
        f.write_text(program, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, str(f)], capture_output=True,
                               timeout=timeout, cwd=d)
            return p.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


import bfcl  # noqa: E402


# ARC-AGI-1 public evaluation set (fchollet/ARC-AGI). Each row is one task: a
# handful of input/output demonstration grids, then one input to transform.
# Scored the way ARC scores itself -- the produced grid is right or it is not,
# no partial credit for getting most cells.
def load_arc(limit=None):
    rows = [json.loads(l) for l in open(DATA / "arc_evaluation.jsonl") if l.strip()]
    return rows[:limit] if limit else rows


def _grid(g):
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def arc_prompt(row):
    parts = ["You are solving an abstract reasoning puzzle. Each example shows an "
             "input grid and the output grid produced from it by one consistent "
             "rule. Digits 0-9 are colours; 0 is the background.\n"]
    for i, ex in enumerate(row["train"], 1):
        parts.append(f"Example {i}\nInput:\n{_grid(ex['input'])}\n"
                     f"Output:\n{_grid(ex['output'])}\n")
    parts.append("Infer the rule from those examples and apply it to this input.\n"
                 f"Input:\n{_grid(row['input'])}\n\n"
                 "Reply with ONLY the output grid: one row per line, digits "
                 "separated by single spaces, and nothing else.")
    return "\n".join(parts)


def parse_grid(text):
    """The last block of digit-rows in a reply, as a grid of ints."""
    text = re.sub(r"^\s*```[a-z]*|```\s*$", "", text.strip(), flags=re.MULTILINE)
    blocks, cur = [], []
    for line in text.split("\n"):
        if re.fullmatch(r"\s*\d+(?:[ ,\t]+\d+)*\s*", line) and line.strip():
            cur.append([int(n) for n in re.findall(r"\d+", line)])
        else:
            if cur:
                blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    if not blocks:
        return None
    grid = blocks[-1]
    # A grid is rectangular; a ragged block is not a grid the scorer can accept.
    return grid if len({len(r) for r in grid}) == 1 else None


def score_arc(row, response):
    return parse_grid(response) == row["output"]


def arc_gold(row):
    return _grid(row["output"])


SUITES = {
    "ifeval": (load_ifeval, score_ifeval, lambda r: r["prompt"]),
    "arc": (load_arc, score_arc, arc_prompt),
    "bfcl_simple": (lambda n=None: bfcl.load("bfcl_simple", n), bfcl.score, bfcl.build_prompt),
    "bfcl_live": (lambda n=None: bfcl.load("bfcl_live", n), bfcl.score, bfcl.build_prompt),
    "bfcl_multiple": (lambda n=None: bfcl.load("bfcl_multiple", n), bfcl.score, bfcl.build_prompt),
    "humaneval": (load_humaneval, score_humaneval,
                  lambda r: "Complete this Python function. Reply with only the "
                            "full function in a ```python code block.\n\n" + r["prompt"]),
    "gsm8k": (load_gsm8k, score_gsm8k,
              lambda r: r["question"] + "\n\nEnd your reply with the final "
                        "numeric answer on its own line."),
}


# ------------------------------------------------------------- selfcheck --
def selfcheck(suite, limit):
    """Score reference answers and degenerate answers. No model involved."""
    load, score, _ = SUITES[suite]
    rows = load(limit)

    print(f"\n=== {suite}: harness self-check on {len(rows)} items ===")

    if suite.startswith("bfcl"):
        gold = [score(r, bfcl.gold_response(r)) for r in rows]
        _report("gold (answer key as a call)", gold, expect="~100%")
    elif suite == "humaneval":
        gold = [score(r, "```python\n" + r["prompt"] + r["canonical_solution"] + "\n```")
                for r in rows]
        _report("gold (canonical solution)", gold, expect="~100%")
    elif suite == "arc":
        gold = [score(r, arc_gold(r)) for r in rows]
        _report("gold (answer grid)", gold, expect="~100%")
    elif suite == "gsm8k":
        gold = [score(r, f"The answer is {gsm8k_gold(r)}") for r in rows]
        _report("gold (reference answer)", gold, expect="~100%")
    else:
        print("  gold: IFEval ships no reference answers — null baselines only.")

    nulls = {
        "empty string": lambda r: "",
        "constant '42'": lambda r: "42",
        "echo of prompt": lambda r: SUITES[suite][2](r),
    }
    for label, make in nulls.items():
        got = [score(r, make(r)) for r in rows]
        _report(f"null ({label})", got, expect="~0%")


def _report(label, results, expect):
    kept = [r for r in results if r is not None]
    pct = 100 * sum(bool(r) for r in kept) / max(len(kept), 1)
    flag = ""
    if expect == "~100%" and pct < 95:
        flag = "  <-- BROKEN CHECKER: reference answers should pass"
    if expect == "~0%" and pct > 5:
        flag = "  <-- LEAKY CHECKER: points without doing the task"
    print(f"  {label:34} {pct:5.1f}%  (expect {expect}){flag}")


# ------------------------------------------------------------------ run ---
def _wire_memory():
    """Hold weights in unified memory instead of letting them page to SSD.

    Without this the harness benchmarks a model's *paging* behaviour as much as
    the model: MLX allocations stay pageable, macOS evicts them under pressure,
    and a large model spends its time faulting pages back off the SSD rather
    than computing. Measured on this 16 GB machine, a 7.9 GB model sat at
    181 MB resident and 1.2% CPU -- indistinguishable from "the model does not
    fit", which is the wrong conclusion.

    Uses the app's own ceilings (config.gpu) so a benchmark measures the same
    memory policy the product runs under; falls back to leaving ~4 GB for
    macOS when the config does not set one. mx.set_wired_limit refuses
    anything above the OS working-set size, so the value is clamped by probing
    rather than assumed.
    """
    try:
        import mlx.core as mx
        from symbio.app import config as appcfg
    except Exception:
        return None
    cfg = appcfg.load_config()
    appcfg.apply_gpu_limits(cfg)
    want = int((cfg.get("gpu") or {}).get("wired_limit_mb", -1))
    if want < 0:
        import subprocess
        try:
            ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                     capture_output=True, text=True).stdout)
            want = max(0, ram // (1024 * 1024) - 4096)   # leave macOS ~4 GB
        except Exception:
            return None
    for mb in (want, want // 2, 4096):
        try:
            mx.set_wired_limit(int(mb) * 1024 * 1024)
            return mb
        except Exception:
            continue
    return None


def run(suite, model_path, draft_path=None, limit=None, max_tokens=512,
        num_draft_tokens=1, out=None, adapter_path=None):
    # Symbio's compat layer patches mlx-lm on import — without it, speculative
    # decoding refuses outright on a hybrid linear-attention target like Bonsai.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from symbio.app import mlx_compat  # noqa: F401
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    load_fn, score, prompt_fn = SUITES[suite]
    rows = load_fn(limit)
    # adapter_path is what makes a before/after fine-tune comparison possible:
    # the same weights, graded twice, with and without what training produced.
    model, tok = (load(model_path, adapter_path=adapter_path) if adapter_path
                  else load(model_path))
    draft = load(draft_path)[0] if draft_path else None
    kw = {"draft_model": draft, "num_draft_tokens": num_draft_tokens} if draft else {}

    wired = _wire_memory()
    if wired:
        print(f"  [mem] wired limit {wired} MB — weights held in unified memory")
    records, t0 = [], time.perf_counter()
    for i, row in enumerate(rows, 1):
        # enable_thinking=False, because that is how Symbio serves a model:
        # thinking is an explicit dial (chat_constants.THINKING_LEVELS), not a
        # default. Left unset, a Qwen3 defaults to thinking ON, and the strip
        # below only removes a CLOSED <think> block -- so a model that reasons
        # past max_tokens emits no call, keeps its unclosed block, and scores
        # as a failure. Measured on bfcl_simple: 19 of Qwen3-8B's 23 failures
        # were exactly that, against 0 for a model that never emits <think>.
        # The benchmark was scoring a mode the product never runs in.
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt_fn(row)}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            # Templates that take no such kwarg (non-Qwen) are unaffected.
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt_fn(row)}],
                tokenize=False, add_generation_prompt=True)
        parts = []
        for resp in stream_generate(model, tok, text, max_tokens=max_tokens,
                                    sampler=make_sampler(temp=0.0), **kw):
            parts.append(resp.text)
        reply = "".join(parts)
        # Both reasoning formats, not just Qwen's. Mistral's reasoning models
        # emit [THINK]...[/THINK] with dedicated vocab tokens, and stripping
        # only the angle-bracket form left the whole chain-of-thought in the
        # graded text: Ministral-3-8B-Reasoning scored 6.9% where its Instruct
        # sibling scored 95.0%, purely because the call was behind reasoning
        # the parser never removed. A benchmark that cannot read a model's
        # output measures the harness.
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
        reply = re.sub(r"\[THINK\].*?\[/THINK\]", "", reply, flags=re.DOTALL)
        # An unclosed block means the model reasoned past max_tokens and never
        # emitted an answer. Dropping the opener would leave its raw reasoning
        # to be graded as if it were the reply; the empty string scores a
        # truthful zero.
        if re.search(r"<think>|\[THINK\]", reply):
            reply = ""
        reply = reply.strip()
        ok = score(row, reply)
        records.append({"index": i, "passed": ok, "response": reply[:2000]})
        kept = [r for r in records if r["passed"] is not None]
        acc = 100 * sum(bool(r["passed"]) for r in kept) / max(len(kept), 1)
        print(f"  [{i}/{len(rows)}] {'PASS' if ok else 'FAIL' if ok is False else 'SKIP'}"
              f"  running={acc:5.1f}%  ({time.perf_counter()-t0:.0f}s)", flush=True)

    kept = [r for r in records if r["passed"] is not None]
    acc = 100 * sum(bool(r["passed"]) for r in kept) / max(len(kept), 1)
    summary = {"suite": suite, "model": model_path, "draft": draft_path,
               "adapter": adapter_path,
               "n": len(kept), "accuracy": acc,
               "elapsed_s": time.perf_counter() - t0, "records": records}
    print(f"\n  {suite} / {Path(model_path).name}: {acc:.1f}%  "
          f"({sum(bool(r['passed']) for r in kept)}/{len(kept)})  "
          f"{summary['elapsed_s']:.0f}s")
    if out:
        Path(out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"  saved -> {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selfcheck", "run"])
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--model"); ap.add_argument("--draft")
    ap.add_argument("--limit", type=int); ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--num-draft-tokens", type=int, default=1); ap.add_argument("--out")
    ap.add_argument("--adapter", help="LoRA adapter directory to load on top of the model")
    a = ap.parse_args()
    if a.mode == "selfcheck":
        selfcheck(a.suite, a.limit)
    else:
        run(a.suite, a.model, a.draft, a.limit, a.max_tokens, a.num_draft_tokens,
            a.out, a.adapter)
