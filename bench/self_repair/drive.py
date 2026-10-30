"""Self-repair bench: inject each test-caught mutation into a COPY of the repo,
let the model repair it through the real fix_loop path, record what happened.

    cd <copy> && SYMBIO_HOME=<copy> python bench/self_repair/drive.py MODEL TAG BUGS.json OUTDIR

The model is reached through genlib (GEN=mlx|hf|openai, ADAPTER, MAXTOK, THINK).
Other knobs: BUDGET (cycles, 3), RETRY_TEMP (0), ATTEMPTS (1) and
ATTEMPT_TEMP (0.7): with ATTEMPTS=k every bug is tried k times, the first
greedy and the rest sampled, so a bug solved 1 time in k still yields a fix
to learn from.

Never point it at a working tree you care about: every bug is written into
the source file. The whole of symbio/ and tests/ is snapshotted at start and
restored after every attempt, so an edit to another file cannot leak into
the next bug.
"""
import difflib, json, os, signal, sys, time
from pathlib import Path

signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally that un-injects the bug
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["SYMBIO_HOME"] = os.getcwd()
from symbio import constants  # noqa: E402

assert constants.PROJECT_DIR.resolve() == Path(os.getcwd()).resolve(), constants.PROJECT_DIR
from symbio.app import fix_loop, fix_watch, devtools, tooling  # noqa: E402
import genlib  # noqa: E402

R, TAG = sys.argv[1], sys.argv[2]
BUGS = json.loads(Path(sys.argv[3]).read_text())
OUT = Path(sys.argv[4]) / f"{TAG}.repair.json"
ATTEMPTS = int(os.environ.get("ATTEMPTS", 1))
ATTEMPT_TEMP = float(os.environ.get("ATTEMPT_TEMP", 0.7))
raw = genlib.load(R)


def _tree():
    return {f: f.read_bytes() for d in ("symbio", "tests") for f in Path(d).rglob("*")
            if f.is_file() and "__pycache__" not in f.parts}


SNAP = _tree()


def restore_tree():
    """A repair may edit files other than the bug's; put every one back so bugs stay independent."""
    now = _tree()
    for f, data in SNAP.items():
        if now.get(f) != data:
            f.write_bytes(data)
            print(f"  [bench] restored {f}", flush=True)
    for f in now.keys() - SNAP.keys():
        f.unlink()
        print(f"  [bench] removed {f}", flush=True)


def attempt(i, b, n):
    src = Path(b["src"])
    pristine = src.read_text()
    lines = pristine.splitlines(True)
    # By content, nearest the recorded line: the file may have grown since the
    # bug list was made, and a line number would mutate the wrong line.
    at = [k for k, l in enumerate(lines) if l == b["old"]]
    if not at:
        return None
    k = min(at, key=lambda k: abs(k - (b["line"] - 1)))
    lines[k] = b["new"]
    src.write_text("".join(lines))
    first_temp = 0.0 if n == 0 else ATTEMPT_TEMP
    retry_temp = float(os.environ.get("RETRY_TEMP", 0)) if n == 0 else ATTEMPT_TEMP
    replies, prompts = [], []

    def gen(prompt):
        temp = retry_temp if fix_loop.RETRY_MARKER in prompt else first_temp
        out = tooling.clean_response(tooling.strip_reasoning_block(raw(prompt, temp)))
        replies.append(out)
        prompts.append(prompt)
        return out

    def verify(_t, test=b["test"]):
        out = genlib.run_tests(test)
        p = devtools.parse_pytest_output(out) or {}
        rep = f"pytest — {p.get('summary')}\n{out[-5000:]}"
        if p.get("failing"):
            rep += "\nFailing: " + ", ".join(p["failing"])
        return bool(p) and p.get("bad", 1) == 0, rep

    t0 = time.time()
    try:
        ok, rep = verify(None)
        target = sorted(fix_loop._failing_ids(rep) or b["fails"])[0]
        trace = devtools.trace_test(target)
        suspects = devtools.locate_fault(target)
        located = fix_loop.pick_source(suspects, fix_watch._slice_source_for(target, rep, trace))
        rec = fix_loop.repair_failure(target, located or b["src"], gen, verify, trace=trace,
                                      budget_cycles=int(os.environ.get("BUDGET", 3)),
                                      log=lambda s: None, suspects=suspects)
        final = src.read_text()
        return {"i": i, "attempt": n, "src": b["src"], "kind": b["kind"], "located": located,
                "located_ok": located == b["src"], "fixed": bool(rec.get("fixed")),
                "cycles": len(rec.get("cycles", [])), "exact": final == pristine,
                "secs": round(time.time() - t0), "replies": replies, "prompts": prompts,
                "diff": "".join(difflib.unified_diff(pristine.splitlines(True), final.splitlines(True),
                                                     "original", "repaired", n=1))}
    except Exception as e:
        return {"i": i, "attempt": n, "src": b["src"], "error": repr(e)[:300], "fixed": False,
                "replies": replies, "prompts": prompts}
    finally:
        src.write_text(pristine)
        restore_tree()


results = []
for i, b in enumerate(BUGS):
    for n in range(ATTEMPTS):
        r = attempt(i, b, n)
        if r is None:
            print(f"{TAG} [{i+1}/{len(BUGS)}] SKIP {b['src']}: original line not found", flush=True)
            break
        results.append(r)
        tries = f" try {n+1}/{ATTEMPTS}" if ATTEMPTS > 1 else ""
        print(f"{TAG} [{i+1}/{len(BUGS)}]{tries} {'FIX ' if r['fixed'] else 'FAIL'} {b['src']}:{b['line']} "
              f"{b['kind']} loc={r.get('located_ok')} cyc={r.get('cycles')} {r.get('secs')}s", flush=True)
        OUT.write_text(json.dumps(results, indent=1))
first = [r for r in results if r["attempt"] == 0]
solved = {r["i"] for r in results if r["fixed"]}
print(f"REPAIR {TAG}: {sum(r['fixed'] for r in first)}/{len(first)} fixed"
      + (f" (pass@{ATTEMPTS} {len(solved)}/{len(first)})" if ATTEMPTS > 1 else ""))
