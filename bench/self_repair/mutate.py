"""Make the self-repair bug sets: single-token mutations (a dropped `not`,
`and`/`or`, `==`/`!=`, a bumped 0/1/2, a shifted comparison) that the
module's own test file catches. Run in a COPY of the repo; each mutation is
written, tested and restored.

    python bench/self_repair/mutate.py <copy> <outdir> [per_module=4] [heldout_modules=10]

Writes pool.json, then train_bugs.json / heldout_bugs.json split BY MODULE,
so a held-out bug sits in a file the model never trained on.
"""
import ast, json, os, random, subprocess, sys
from pathlib import Path

SWAP = {ast.Lt:("<","<="),ast.LtE:("<=","<"),ast.Gt:(">",">="),ast.GtE:(">=",">"),ast.Eq:("==","!="),ast.NotEq:("!=","==")}
def cands(src):
    lines = src.splitlines(True); tree = ast.parse(src); out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and len(n.ops)==1 and type(n.ops[0]) in SWAP and n.lineno==n.end_lineno:
            a, b = SWAP[type(n.ops[0])]; ln = lines[n.lineno-1]
            seg = ln[n.left.end_col_offset:n.comparators[0].col_offset]
            if seg.strip()==a: out.append((n.lineno, ln, ln[:n.left.end_col_offset]+seg.replace(a,b,1)+ln[n.comparators[0].col_offset:], f"{a}->{b}"))
        elif isinstance(n, ast.BoolOp) and n.lineno==n.end_lineno:
            ln = lines[n.lineno-1]; a, b = (" and "," or ") if isinstance(n.op, ast.And) else (" or "," and ")
            if ln.count(a)==1: out.append((n.lineno, ln, ln.replace(a,b), f"{a.strip()}->{b.strip()}"))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and n.lineno==n.end_lineno:
            ln = lines[n.lineno-1]
            if ln.count("not ")==1 and "is not" not in ln and "not in" not in ln: out.append((n.lineno, ln, ln.replace("not ","",1), "drop not"))
        elif isinstance(n, ast.Constant) and type(n.value) is int and n.value in (0,1,2) and n.lineno==n.end_lineno:
            ln = lines[n.lineno-1]; s = ln[n.col_offset:n.end_col_offset]
            if s==str(n.value): out.append((n.lineno, ln, ln[:n.col_offset]+str(n.value+1)+ln[n.end_col_offset:], f"{n.value}->{n.value+1}"))
    return out


def main():
    clone, out = Path(sys.argv[1]), Path(sys.argv[2])
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    held_n = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    pairs = []
    for f in sorted((clone / "symbio/app").glob("*.py")):
        t = clone / "tests" / f"test_{f.stem}.py"
        if t.exists() and len(f.read_text().splitlines()) > 80 and t.read_text().count("\ndef test") >= 5:
            pairs.append((f"symbio/app/{f.name}", f"tests/{t.name}"))
    print(len(pairs), "modules", flush=True)
    env = dict(os.environ, SYMBIO_HOME=str(clone))
    random.seed(11)
    bugs = []
    for src_rel, test_rel in pairs:
        p = clone / src_rel
        pristine = p.read_text()
        try:
            cs = cands(pristine)
        except SyntaxError:
            continue
        random.shuffle(cs)
        got, seen = 0, set()
        for lineno, old, new, kind in cs:
            if got >= per:
                break
            if lineno in seen or old.lstrip().startswith(("#", "log(", "print(")) or '"""' in old:
                continue
            lines = pristine.splitlines(True)
            lines[lineno - 1] = new
            p.write_text("".join(lines))
            try:
                r = subprocess.run([sys.executable, "-m", "pytest", test_rel, "-q", "-x", "--no-header",
                                    "-p", "no:cacheprovider", "--tb=no"], cwd=clone, capture_output=True,
                                   text=True, timeout=120, env=env)
                fails = [l.split(" ", 1)[1].split(" - ")[0] for l in r.stdout.splitlines()
                         if l.startswith("FAILED ")]
            except subprocess.TimeoutExpired:
                fails = []
            finally:
                p.write_text(pristine)
            if fails:
                seen.add(lineno)
                bugs.append({"src": src_rel, "test": test_rel, "line": lineno, "old": old,
                             "new": new, "kind": kind, "fails": fails})
                got += 1
                print(len(bugs), src_rel, lineno, kind, flush=True)
    (out / "pool.json").write_text(json.dumps(bugs, indent=1))
    mods = sorted({b["src"] for b in bugs})
    random.seed(23)
    random.shuffle(mods)
    held = set(mods[:held_n])
    (out / "train_bugs.json").write_text(json.dumps([b for b in bugs if b["src"] not in held], indent=1))
    (out / "heldout_bugs.json").write_text(json.dumps([b for b in bugs if b["src"] in held], indent=1))


if __name__ == "__main__":
    main()
