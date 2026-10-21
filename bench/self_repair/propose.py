"""The proposer half of self-play: the model writes its own bugs.

It is shown one function from a module that has tests and asked to break
exactly one line so that the tests catch it. A proposal becomes a task only
when the file still parses, the module's tests passed before the change and
fail after it. Bugs are written into a COPY of the repo, tested and restored.

    cd <copy> && SYMBIO_HOME=<copy> python bench/self_repair/propose.py MODEL TAG MODULES.json OUTDIR [N=12]

MODULES.json is a bug list (train_bugs.json): only its src/test pairs are
used, so held-out modules are never proposed from. Writes OUTDIR/TAG.tasks.json
in the bug-list format drive.py reads, plus the proposal prompt and reply so
the proposer can be trained on the tasks that turned out learnable.
Model selection as in genlib; PROPOSE_TEMP (0.8) samples varied bugs.

Only lines the module's tests actually execute are offered: each test file
is run once under symbio.app.fault_trace, and a function is picked only when
the tests run at least MIN_COVERED (3) of its lines. The prompt lists those
lines, so the bug lands where the tests can see it. Breaking a line no test
runs was the most common way a proposal was wasted (10 of 24, 2026-10-11).
"""
import ast, io, json, os, random, re, signal, sys, tokenize
from pathlib import Path


PROMPT = """You are writing a practice exercise for yourself: a bug you will have to find and fix later.

File: {src}
Its tests: {test}

One function from that file:
```python
{body}
```

The tests run these lines of it, so pick ONE of them:
{covered}

Change exactly ONE of those lines so the code stays valid Python but behaves wrongly in a way {test} will catch. Make it a real logic bug that takes thought to find, not a syntax error, a crash on import, or a changed comment or string.

Reply with only this block, the OLD line copied exactly and the NEW line with no comment:
<<<< OLD
the original line
====
the changed line
>>>> NEW"""

# Lenient on the fence (8B models write `<<\nOLD`, `=======`, a bare `NEW` line
# for the separator, a bare `>>>>`), strict on content. The LAST block counts:
# a model that thinks aloud drafts and abandons blocks on the way. It is not
# invited to: granite reasoned past MAXTOK on 16 of 16 proposals (2026-10-11).
BLOCK = re.compile(r"<{2,}\s*OLD\s*\n(.*?)\n\s*(?:={3,}|NEW)\s*\n(.*?)\n\s*>{2,}", re.S)
MIN_COVERED = int(os.environ.get("MIN_COVERED", 3))


def strip_comment(line):
    """Drop a trailing comment: a `# BUG: ...` note would hand the solver the answer."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(line.strip() + "\n").readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return line.rstrip()
    for t in toks:
        if t.type == tokenize.COMMENT:
            return line.strip()[:t.start[1]].rstrip()
    return line.rstrip()


def functions(src_text):
    out = []
    for n in ast.walk(ast.parse(src_text)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and 4 <= n.end_lineno - n.lineno <= 60:
            out.append((n.lineno, n.end_lineno))
    return out


def covered_lines(test, src):
    """Line numbers of SRC that TEST's file executes, from symbio.app.fault_trace."""
    import tempfile
    from symbio.app import devtools
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "spectrum.json"
        cmd = [sys.executable, "-m", "pytest", test, "-q", "--no-header", "--tb=no",
               "-p", "no:cacheprovider", "-p", "symbio.app.fault_trace"]
        try:
            devtools._run_pytest(cmd, 300, env={"SYMBIO_FAULT_OUT": str(out)})
            spectrum = json.loads(out.read_text())
        except Exception:
            return set()
    return {ln for r in spectrum.values() for ln in r.get("lines", {}).get(src, [])}


def tested_functions(fns, lines, covered):
    """(lo, hi, covered line numbers) for functions the tests run enough of."""
    out = []
    for lo, hi in fns:
        ks = sorted(ln for ln in covered if lo < ln <= hi and lines[ln - 1].strip()
                    and not lines[ln - 1].strip().startswith(("#", "def ", "async def ")))
        if len(ks) >= MIN_COVERED:
            out.append((lo, hi, ks))
    return out


def parse(reply, lines, lo, hi, allowed=None):
    """(line index, mutated line) or a reason it is not a usable proposal.
    ALLOWED (0-based line indices) limits the bug to lines the tests run."""
    m = None
    for m in BLOCK.finditer(reply):
        pass
    if not m:
        return "no OLD/NEW block"
    olds, news = m.group(1).strip("\n").splitlines(), m.group(2).strip("\n").splitlines()
    if len(olds) != len(news):
        return "more than one line"
    # A block quoted with context is fine as long as exactly one line in it changed
    changed = [i for i, (a, b) in enumerate(zip(olds, news)) if a.strip() != strip_comment(b).strip()]
    if len(changed) > 1:
        return "more than one line"
    old, new = (olds[changed[0]], strip_comment(news[changed[0]])) if changed else (olds[0], news[0])
    hits = [k for k in range(lo - 1, hi) if lines[k].strip() == old.strip() and old.strip()]
    if not hits:
        return "OLD line not in the function"
    if allowed is not None:
        hits = [k for k in hits if k in allowed]
        if not hits:
            return "line the tests never run"
    k = hits[0]
    if new.strip() == lines[k].strip():
        return "NEW is unchanged"
    if new.strip().startswith("#") or not new.strip():
        return "NEW is a comment or blank"
    indent = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
    return k, indent + new.strip() + "\n"


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally that restores the file
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["SYMBIO_HOME"] = os.getcwd()
    from symbio import constants
    assert constants.PROJECT_DIR.resolve() == Path(os.getcwd()).resolve(), constants.PROJECT_DIR
    from symbio.app import devtools, tooling
    import genlib
    R, TAG, mods, outdir = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    pairs = sorted({(b["src"], b["test"]) for b in json.loads(Path(mods).read_text())})
    raw = genlib.load(R)
    temp = float(os.environ.get("PROPOSE_TEMP", 0.8))
    rng = random.Random(os.environ.get("SEED", TAG))
    green, tasks, rejects, seen, rejected, cover = {}, [], {}, set(), [], {}
    for j in range(n):
        src, test = rng.choice(pairs)
        p = Path(src)
        pristine = p.read_text()
        lines = pristine.splitlines(True)
        if (src, test) not in cover:
            cover[(src, test)] = tested_functions(functions(pristine), lines, covered_lines(test, src))
        fns = cover[(src, test)]
        if not fns:
            rejects["no tested function"] = rejects.get("no tested function", 0) + 1
            print(f"{TAG} [{j+1}/{n}] SKIP {src}: its tests run no function enough", flush=True)
            continue
        lo, hi, ks = rng.choice(fns)
        shown = list(dict.fromkeys(lines[k - 1].strip() for k in ks))
        prompt = PROMPT.format(src=src, test=test, body="".join(lines[lo - 1:hi]).rstrip(),
                               covered="\n".join(shown))
        reply = tooling.clean_response(tooling.strip_reasoning_block(raw(prompt, temp)))
        got = parse(reply, lines, lo, hi, {k - 1 for k in ks})
        why = got if isinstance(got, str) else None
        if not why:
            k, new = got
            key = (src, k, new)
            mutated = lines[:k] + [new] + lines[k + 1:]
            try:
                ast.parse("".join(mutated))
            except SyntaxError:
                why = "does not parse"
            if not why and key in seen:
                why = "duplicate"
            if not why and test not in green:
                rep = devtools.parse_pytest_output(genlib.run_tests(test, 180)) or {}
                green[test] = bool(rep) and rep.get("bad", 1) == 0
            if not why and not green[test]:
                why = "tests already red"
            if not why:
                p.write_text("".join(mutated))
                try:
                    out = genlib.run_tests(test, 180)
                except Exception:
                    out = ""
                finally:
                    p.write_text(pristine)
                fails = [l.split(" ", 1)[1].split(" - ")[0] for l in out.splitlines() if l.startswith("FAILED ")]
                if not fails:
                    why = "tests still pass"
        if why:
            rejects[why] = rejects.get(why, 0) + 1
            rejected.append({"src": src, "why": why, "reply": reply})
            (outdir / f"{TAG}.rejected.json").write_text(json.dumps(rejected, indent=1))
            print(f"{TAG} [{j+1}/{n}] REJECT {src}: {why}", flush=True)
            continue
        seen.add(key)
        tasks.append({"src": src, "test": test, "line": k + 1, "old": lines[k], "new": new,
                      "kind": "proposed", "fails": fails, "prompt": prompt, "reply": reply})
        print(f"{TAG} [{j+1}/{n}] TASK {src}:{k+1} {lines[k].strip()[:60]!r} -> {new.strip()[:60]!r}", flush=True)
        (outdir / f"{TAG}.tasks.json").write_text(json.dumps(tasks, indent=1))
    (outdir / f"{TAG}.tasks.json").write_text(json.dumps(tasks, indent=1))
    print(f"PROPOSE {TAG}: {len(tasks)}/{n} valid; rejected {rejects}", flush=True)


if __name__ == "__main__":
    main()
