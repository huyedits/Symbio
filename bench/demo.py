"""A benchmark you can trust, demonstrated in about forty seconds.

Loads no models. Everything here is the harness grading known-good and
known-bad answers, plus the model results already on disk.

    python bench/demo.py
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bfcl      # noqa: E402
import harness   # noqa: E402

BOLD, DIM, OK, BAD, WARN, END = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def h(title, blurb):
    print(f"\n{BOLD}{title}{END}\n{DIM}{blurb}{END}")


def line(label, pct, expect_high, note=""):
    good = (pct >= 95) if expect_high else (pct <= 5)
    mark = f"{OK}PASS{END}" if good else f"{BAD}FAIL{END}"
    print(f"   {mark}  {label:<38} {pct:5.1f}%   {DIM}{note}{END}")


# 1 -------------------------------------------------------------------------
h("1. The harness grades its own answer key.",
  "A scorer that rejects a correct answer will blame the model. This is the "
  "check that would have caught the <|im_end|> bug that once scored a working "
  "model 14/100.")

for cat in bfcl.CATEGORIES:
    rows = bfcl.load(cat, 150)
    gold = 100 * sum(bfcl.score(r, bfcl.gold_response(r)) for r in rows) / len(rows)
    line(f"{cat}: answer key scores", gold, True, "want ~100%")

rows = [r for r in harness.load_humaneval(150) if r["task_id"] != "HumanEval/32"]
hits = 100 * sum(harness.score_humaneval(
    r, "```python\n" + r["prompt"] + r["canonical_solution"] + "\n```") for r in rows) / len(rows)
line("HumanEval+: canonical solutions run", hits, True, "want ~100%")
print(f"   {DIM}(HumanEval/32 excluded: its own upstream test unpacks a float. "
      f"The gold run found a dataset bug.){END}")

g = harness.load_gsm8k(150)
line("GSM8K: reference answers score",
     100 * sum(harness.score_gsm8k(r, f"The answer is {harness.gsm8k_gold(r)}") for r in g) / len(g),
     True, "want ~100%")

# 2 -------------------------------------------------------------------------
h("2. And it refuses to award points for nothing.",
  "Only checking that right answers pass lets a scorer that passes everything "
  "through.")

for junk, label in [("", "empty string"), ("42", "the constant '42'"),
                    ("I don't know.", "a refusal")]:
    rows = bfcl.load("bfcl_simple", 150)
    line(f"bfcl_simple scores {label}",
         100 * sum(bfcl.score(r, junk) for r in rows) / len(rows), False, "want ~0%")

rows = bfcl.load("bfcl_simple", 150)
def args_of(r):
    name, params = next(iter(r["ground_truth"][0].items()))
    return name, {p: bfcl._concrete(a) for p, a in params.items()
                  if a and bfcl._concrete(a) is not None}
name_only = 100 * sum(
    bfcl.score(r, json.dumps({"name": args_of(r)[0], "arguments": {}})) for r in rows) / len(rows)
line("right function, no arguments", name_only, False,
     "want ~0% - argument fidelity is the point")

ifeval = harness.load_ifeval(150)
got = [harness.score_ifeval(r, "42") for r in ifeval]
kept = [x for x in got if x is not None]
floor = 100 * sum(bool(x) for x in kept) / len(kept)
print(f"   {WARN}KNOWN{END}  {'IFEval scores the constant 42':<38} {floor:5.1f}%   "
      f"{DIM}leaky by construction - see below{END}")

# 3 -------------------------------------------------------------------------
h("3. It reads a correct answer however a model spells it.",
  "Scoring an answer this file formatted only proves the parser can read its "
  "own writing. So the same correct call is re-spelled the way real models "
  "emit it. This caught a bug where every Python-syntax call containing a dict "
  "argument scored zero.")

def spellings(n, a):
    j = json.dumps({"name": n, "arguments": a})
    py = f"{n}(" + ", ".join(f"{k}={v!r}" for k, v in a.items()) + ")"
    return {"plain JSON": j, "```json fence": f"```json\n{j}\n```",
            "<tool_call> tags": f"<tool_call>{j}</tool_call>",
            "prose wrapped around it": f"Sure!\n{j}\nHope that helps.",
            '"parameters" not "arguments"': json.dumps({"name": n, "parameters": a}),
            "bare {func: args}": json.dumps({n: a}),
            "list-wrapped": json.dumps([{"name": n, "arguments": a}]),
            "after a <think> block": f"<think>hmm</think>\n{j}",
            "Python call syntax": py,
            "```python fence": f"```python\n{py}\n```"}

for label in spellings("f", {"a": 1}):
    hits = 100 * sum(bfcl.score(r, spellings(*args_of(r))[label]) for r in rows) / len(rows)
    line(f"reads {label}", hits, True, "want ~100%")

# 4 -------------------------------------------------------------------------
h("4. What that rigour bought: it caught a dead benchmark.",
  "Same models, two difficulty levels of the same dataset.")

def load_results(suite):
    out = {}
    for f in glob.glob("bench/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("suite") == suite and "accuracy" in d:
            out[os.path.basename(f).replace(".json", "")] = (d["accuracy"], d.get("elapsed_s", 0))
    return out

NAMES = {"bfcl_qwen06b": ("Qwen3-0.6B", "0.33 GB"), "bfcl_qwen4b": ("Qwen3-4B", "2.10 GB"),
         "bfcl_qwen8b": ("Qwen3-8B", "4.35 GB"), "bfcl_qwen14b": ("Qwen3-14B 3bit", "6.46 GB"),
         "live_phi4_14b": ("Phi-4-14B", "8.25 GB"), "live_mistral_nemo12b": ("Mistral-Nemo-12B", "6.89 GB"),
         "live_glm4_9b": ("GLM-4-9B", "5.29 GB"), "live_lfm2_8b_a1b": ("LFM2-8B-A1B (MoE)", "5.21 GB"),
         "probe_06b_bfcl_live": ("Qwen3-0.6B", "0.33 GB")}

for suite, verdict in (("bfcl_simple", f"{BAD}SATURATED - ranks nothing{END}"),
                       ("bfcl_live", f"{OK}DISCRIMINATES - usable{END}")):
    res = load_results(suite)
    if not res:
        continue
    print(f"\n   {BOLD}{suite}{END}   {verdict}")
    ordered = sorted(res.items(), key=lambda kv: -kv[1][0])
    for k, (acc, t) in ordered:
        nm, size = NAMES.get(k, (k, ""))
        print(f"      {acc:5.1f}%   {nm:<20} {size:>8}   {DIM}{t:5.0f}s{END}")
    spread = max(v[0] for v in res.values()) - min(v[0] for v in res.values())
    print(f"      {DIM}spread across a 20x size range: {spread:.1f} points{END}")

print(f"""
{BOLD}The point.{END} On bfcl_simple a 331 MB model scores within 5 points of a
14B, ranked inverse to size. That suite hands the model one function and its
schema - a formatting exercise. Every ranking it produces is noise.

The same models on bfcl_live spread 31 points, in the right order.

Both numbers look equally respectable in a report. The difference is only
visible if the harness is made to prove itself first - which is what sections
1 to 3 do, and why IFEval is flagged: it pays {floor:.0f}% for the string "42", so
only the margin above that floor is signal.

{DIM}Full test suite:  python -m pytest bench/test_bench_harness.py -q   (38 tests)
Self-check a suite: python bench/harness.py selfcheck --suite bfcl_live{END}
""")
