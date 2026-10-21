"""The mechanical repair loop: the harness remembers the plan; the model
only writes one patch per cycle.

Measured 2026-10-07, a 12B no-think model given /fix-harness as a PROMPT:
called run_tests unprompted (correct), stated the correct diagnosis, then
talked about edit_file instead of calling it, and the turn ended on the
diagnosis. Small local models are a good STEP, not a good LOOP — the same
measurement that produced `run_tests`. So the loop itself moves into Python
(the codebase doctrine: deterministic things belong in code), borrowing the
shape of every working agent harness: the model's whole job per cycle is
ONE edit on a SMALL context, and everything else — running the suite,
slicing the failing test and its target source, applying, re-running,
reverting on failure, stopping after two dead attempts — is this module's.

The model is never trusted: a patch whose syntax will not parse is refused
before it touches the file; the suite's verdict at the end of each cycle is
the only success signal; more new failures than the cycle started with
triggers an automatic revert. Nothing here edits without a backup, and
nothing here claims success the suite did not confirm.
"""

import re
import textwrap
from pathlib import Path
from typing import Any, Callable

from symbio import constants

# Per file under repair: how many (patch → verify) cycles before the verdict
# "this failure defeats me" is returned honestly.
_MAX_CYCLES_PER_FAILURE = 2

# The model's patch arrives as one or more exact-text replacements, the same
# shape edit_file uses, inside one fenced block. Single-hunk patches are the
# common case and the parser must read them without ceremony.
# Opens the "what earlier cycles tried" block of a retry prompt. A generator
# may sample on prompts carrying it: greedy decoding answers a retry with
# the patch it already sent.
RETRY_MARKER = "Already tried"

_PATCH_RE = re.compile(
    r"<<+\s*OLD\s*\n(.*?)(?:\n[#\s]*)?====+\s*\n(.*?)(?:\n[#\s]*)?>>+(?:[ \t]*NEW)?",
    re.DOTALL)
# The close is `>>>> NEW`, or measured 2026-10-07 on Gemma-12B, a bare `>>>>`.
# The open is `<<<< OLD`, or measured 2026-10-08 on granite-4.2-8b, `<<` and
# OLD on the next line: two of its 13 misses were that fence, discarded.

# The truncated-fence fallback: a small model sometimes burns its reply
# budget mid-fence — OLD is complete, NEW is cut or the closing >>>> never
# arrives (measured 2026-10-07, Gemma-12B: the right patch, dead at 400
# tokens, three lines short of the close). When the closing fence is missing
# but the OLD/==== split is present, the hunk is still readable: OLD verbatim,
# NEW = everything after the ==== with any truncation note trimmed. A partial
# NEW may still be the right edit; the suite decides, not the parser.
# The separator line varies by model: `====`, and measured here, `####
# ====` (a comment-flavoured marker). Both accepted.
_PATCH_TRUNCATED_RE = re.compile(
    r"<<+\s*OLD\s*\n(.*?)(?:\n[#\s]*)?====+\s*\n(.*)", re.DOTALL)

# A slice cap per read: the model's context budget is the whole reason this
# module exists, so each cycle hands it hundreds of lines, not thousands.
_HUNK_CONTEXT = 6           # lines around each match
_MAX_SLICE_CHARS = 3400


# The edit-tool shape agentic models are trained on (measured 2026-10-08:
# MiMo-9B answered the fence prompt with a `patch` tool call carrying
# old_string/new_string). Read as a patch; it is the same information.
_TOOL_EDIT_RE = re.compile(
    r"<parameter=old_string>\n?(.*?)\n?</parameter>\s*"
    r"<parameter=new_string>\n?(.*?)\n?</parameter>", re.DOTALL)


def _extract_patch(reply: str) -> list[tuple[str, str]]:
    """(old, new) replacement pairs from the model's reply. Empty = no patch
    the parser accepts, which callers treat as a failed attempt, not an
    error — the loop simply does not touch the file.
    Accepts a TRUNCATED fence (no >>>> NEW close): see _PATCH_TRUNCATED_RE.
    """
    out: list[tuple[str, str]] = []
    for m in _TOOL_EDIT_RE.finditer(reply or ""):
        if m.group(1) and m.group(1) != m.group(2):
            out.append((m.group(1), m.group(2)))
    if out:
        return out
    for m in _PATCH_RE.finditer(reply or ""):
        old, new = m.group(1), m.group(2)
        if old and old != new:
            out.append((old, new))
    if out:
        return out
    m = _PATCH_TRUNCATED_RE.search(reply or "")
    if m:
        old, new = m.group(1), m.group(2)
        # The truncation artifact the generator appends (" …(truncated)") and
        # a trailing incomplete line never improve the NEW text.
        new = re.sub(r"\(truncated\)?\s*$", "", new).rstrip()
        # A cut mid-line leaves a partial line that cannot apply cleanly;
        # everything after the last full line (a newline in NEW) is suspect.
        if "\n" in new or new:
            last_full = new.rfind("\n")
            new = new[:last_full] if last_full > 0 else new
        if old and old != new:
            return [(old, new)]
    return []


_CODE_BLOCK_RE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)(?:\n```|\Z)", re.DOTALL)
_FENCE_MARK_RE = re.compile(r"\s*(?:={3,}|<{3,}.*|>{3,}.*)\s*$")


def _nearest_window(lines: list[str], block: list[str],
                    floor: float = 0.85, margin: float = 0.05) -> tuple[int, int] | None:
    """The one span of the file, as long as `block`, that the block nearly
    equals; None unless the best is close (>= floor) and clearly ahead of
    the runner-up, so an ambiguous guess never becomes an edit."""
    import difflib

    n = len(block)
    if n < 2 or n > len(lines):
        return None
    want = "\n".join(l.strip() for l in block)
    scored = []
    for i in range(len(lines) - n + 1):
        have = "\n".join(l.strip() for l in lines[i:i + n])
        scored.append((difflib.SequenceMatcher(None, want, have).ratio(), i))
    scored.sort(reverse=True)
    best, at = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    if best < floor or best - runner < margin:
        return None
    return at, at + n


def _block_patch(reply: str, text: str) -> list[tuple[str, str]]:
    """A patch read from a plain code block, for models that answer with the
    corrected code instead of the fence (measured 2026-10-08: MiMo-9B sent a
    ```python block holding the whole fixed function on 18 of 20 bugs, and
    every one was thrown away as "no parseable patch").

    Two shapes are accepted, both anchored in the CURRENT file so nothing
    lands anywhere the model did not name:
    - the block starts with `def name(` / `class Name` that the file defines
      once: the block replaces that whole definition;
    - otherwise the block's first non-blank line occurs once in the file:
      the block replaces the span from there to its last line's one match,
      or, when the last line is the edited one, as many lines as it has.
    """
    import ast

    lines = text.splitlines(keepends=True)
    for m in _CODE_BLOCK_RE.finditer(reply or ""):
        # A half-remembered fence leaves its divider behind (measured
        # 2026-10-08: Nemotron-Elastic-12B sent "====" then only the fixed
        # lines, on every cycle); the divider is never code.
        block = "\n".join(l for l in m.group(1).rstrip("\n").splitlines()
                          if not _FENCE_MARK_RE.match(l)).strip("\n")
        body = [l for l in block.splitlines() if l.strip()]
        if not body:
            continue
        head = re.match(r"\s*(?:async\s+)?(?:def|class)\s+(\w+)", body[0])
        if head:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                tree = None
            defs = [n for n in ast.walk(tree) if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.name == head.group(1)] if tree else []
            if len(defs) == 1:
                d = defs[0]
                first = min([d.lineno] + [x.lineno for x in getattr(d, "decorator_list", [])])
                old = "".join(lines[first - 1:d.end_lineno]).rstrip("\n")
                indent = re.match(r"\s*", lines[d.lineno - 1]).group(0)
                new = textwrap.indent(textwrap.dedent(block), indent)
                if old != new:
                    return [(old, new)]
                continue
            if not defs:
                continue        # a definition the file lacks names nothing
        first_l, last_l = body[0].strip(), body[-1].strip()
        starts = [i for i, l in enumerate(lines) if l.strip() == first_l]
        ends = [i for i, l in enumerate(lines) if l.strip() == last_l]
        if len(starts) != 1:
            # The first line may be the edited one (`if typed:` sent back as
            # `if not typed:`), so it matches nowhere. Anchor the whole
            # block instead: the one same-length window it nearly equals.
            span = _nearest_window(lines, block.splitlines())
            if span is not None:
                old = "".join(lines[span[0]:span[1]]).rstrip("\n")
                if old.strip() != block.strip():
                    return [(old, block)]
            continue
        if len(ends) == 1 and starts[0] <= ends[0]:
            stop = ends[0] + 1
        else:
            # The last line is the edited one: take as many lines as the
            # block has. The compile check and the suite judge the result.
            stop = starts[0] + len(block.splitlines())
        old = "".join(lines[starts[0]:stop]).rstrip("\n")
        if old.strip() != block.strip():
            return [(old, block)]
    return []


def _apply_patch(text: str, hunks: list[tuple[str, str]]) -> str | None:
    """Apply every hunk exactly once. None when any old-string is missing or
    ambiguous — a partial patch must not land.

    Exact-match first. A miss falls one notch to a LINE-WINDOW match: the
    model drops punctuation at a hunk's edges (measured 2026-10-07: the
    closing `)` of the last line). If the OLD's non-empty lines are
    contiguous in the file — ignoring the final line's tail from the point
    it stops matching — and the file line STARTS WITH the model's version
    (its truncation, not its edit), the window is accepted and replaced with
    the model's NEW. Anything looser is refused: ambiguity never lands.
    """
    out = text
    for old, new in hunks:
        # The smallest edit first: models quote a whole function to change
        # one line, and every extra quoted line is one more chance to miss a
        # blank line or a docstring character (measured 2026-10-07: the
        # right one-line fix, quoted inside 20 lines with one blank line
        # short, missed three cycles running).
        core = _shrink_hunk(old, new)
        if core and out.count(core[0]) == 1:
            out = out.replace(core[0], core[1], 1)
            continue
        count = out.count(old)
        if count == 1:
            out = out.replace(old, new, 1)
            continue
        if count == 0 and "\n" in old:
            window = _fuzzy_window(out, old)
            if window is not None:
                start, end = window
                out = out[:start] + new + out[end:]
                continue
        return None
    return out


def _shrink_hunk(old: str, new: str) -> tuple[str, str] | None:
    """(old, new) with the lines they share at both ends removed, or None
    when nothing of OLD is left to anchor on (a pure insertion)."""
    o, n = old.split("\n"), new.split("\n")
    head = 0
    while head < min(len(o), len(n)) and o[head].rstrip() == n[head].rstrip():
        head += 1
    tail = 0
    while (tail < min(len(o), len(n)) - head
           and o[-1 - tail].rstrip() == n[-1 - tail].rstrip()):
        tail += 1
    core_old = o[head:len(o) - tail]
    core_new = n[head:len(n) - tail]
    if not "".join(core_old).strip():
        return None
    return "\n".join(core_old), "\n".join(core_new)


def _fuzzy_window(text: str, old: str) -> tuple[int, int] | None:
    """Where the model's (possibly edge-truncated) OLD block sits in text.

    Only when: every non-empty line of OLD matches a contiguous run of file
    lines, each file line starting with the model's line (allowing the
    model's lines to be prefixes of the file's — a truncated tail), the
    match is UNIQUE, and the match covers at least two lines so a one-line
    guess can never be mistaken for a patch target.
    """
    model_lines = [l.rstrip() for l in old.splitlines() if l.strip()]
    if len(model_lines) < 2:
        return None
    lines = text.splitlines(keepends=True)
    # Blank lines are skipped on BOTH sides: a model that quotes two
    # functions with one blank line between them where the file has two is
    # quoting the right code.
    solid = [k for k, l in enumerate(lines) if l.strip()]
    stripped = [lines[k].rstrip() for k in solid]
    n, m = len(solid), len(model_lines)
    matches: list[int] = []
    for i in range(n - m + 1):
        ok = True
        for j in range(m):
            if not stripped[i + j].startswith(model_lines[j]):
                ok = False
                break
        if ok:
            matches.append(i)
            if len(matches) > 1:
                return None          # ambiguous — refused
    if not matches:
        return None
    i = matches[0]
    first, last = solid[i], solid[i + m - 1]
    # Character span: start of the first matched line to the END of the
    # last matched line (the model's tail-truncated last line gets the
    # file's full line replaced).
    start = sum(len(l) for l in lines[:first])
    end = sum(len(l) for l in lines[:last + 1])
    return start, end


# A frame line in pytest's traceback: "symbio/app/commands.py:203: in render"
# or the final "symbio/app/commands.py:203: AssertionError".
_FRAME_RE = re.compile(r"^((?:symbio|bench)/[\w./-]+\.py):(\d+):", re.MULTILINE)


def _failure_section(report: str, failed_test: str) -> str:
    """The traceback block pytest printed for this one test, or the whole
    report when the block cannot be told apart (short reports, -q tails)."""
    name = failed_test.split("::")[-1]
    m = re.search(r"^_{3,} .*?\b" + re.escape(name) + r"\b.*? _{3,}$",
                  report or "", re.MULTILINE)
    if not m:
        return report or ""
    nxt = re.search(r"^(?:_{3,} |={3,} )", report[m.end():], re.MULTILINE)
    return report[m.start(): m.end() + (nxt.start() if nxt else len(report))]


def source_from_report(report: str, failed_test: str) -> tuple[str, int] | None:
    """(project-relative source path, line) where the failure actually
    surfaced: the DEEPEST non-test frame of the failing test's traceback.

    The watcher used to guess the source from the test file's name
    (test_custom_commands.py -> custom_commands.py, which does not exist, so
    it fell back to chat.py) and the model was handed a file that could not
    contain the bug — measured 2026-10-07 on the injected commands.py bug.
    The traceback names the right file; read it from there."""
    frames = [(path, int(line)) for path, line in
              _FRAME_RE.findall(_failure_section(report, failed_test))]
    frames = [f for f in frames
              if (constants.PROJECT_DIR / f[0]).is_file()]
    return frames[-1] if frames else None


def source_from_test(failed_test: str) -> tuple[str, int | None] | None:
    """(source path, line of the called function) read from the failing
    test's own code, for failures with no project frame in the traceback.

    A wrong VALUE fails in the test's assert, so the traceback names only the
    test file (measured 2026-10-07: the injected commands.py bug located to
    nothing, and the fallback name-mapping sent the model to chat.py). The
    test still says what it exercises: `commands.render(...)` under
    `from symbio.app import commands` is symbio/app/commands.py, at
    `def render`.
    """
    import ast

    test_rel, _, func = failed_test.partition("::")
    func = func.split("::")[-1].split("[")[0]
    try:
        tree = ast.parse((constants.PROJECT_DIR / test_rel).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None

    def module_file(dotted: str) -> str | None:
        for rel in (dotted.replace(".", "/") + ".py",
                    dotted.replace(".", "/") + "/__init__.py"):
            if (constants.PROJECT_DIR / rel).is_file():
                return rel
        return None

    # local name -> (module file, attribute to look for or None)
    names: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("symbio"):
            for a in node.names:
                as_module = module_file(f"{node.module}.{a.name}")
                if as_module:
                    names[a.asname or a.name] = (as_module, None)
                elif module_file(node.module):
                    names[a.asname or a.name] = (module_file(node.module), a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("symbio") and module_file(a.name):
                    names[a.asname or a.name] = (module_file(a.name), None)

    target = next((n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == func), None)
    if target is None:
        return None
    # Calls inside an assert first: `commands.render(...) == ...` is what
    # failed; `commands.save_command(...)` above it is setup.
    asserted = [n for a in ast.walk(target) if isinstance(a, ast.Assert)
                for n in ast.walk(a)]
    for node in asserted + list(ast.walk(target)):
        attr = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name):
            base, attr = node.func.value.id, node.func.attr
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            base = node.func.id
        else:
            continue
        if base not in names:
            continue
        rel, imported_attr = names[base]
        wanted = attr or imported_attr
        line = None
        if wanted:
            try:
                for i, l in enumerate((constants.PROJECT_DIR / rel).read_text(
                        encoding="utf-8").splitlines(), 1):
                    if re.match(rf"\s*(?:async\s+)?def {re.escape(wanted)}\(", l):
                        line = i
                        break
            except OSError:
                pass
        if line is not None or attr is None:
            return rel, line
    return None


def pick_source(suspects: list[tuple[str, int, float]] | None,
                fallback: str) -> str:
    """The file to repair: `fallback` (the traceback's or the caller's pick)
    when it holds a line tied for the top fault score, else the file of the
    top-ranked suspect.

    Measured 2026-10-08 on 20 injected bugs: the traceback's deepest frame
    named the wrong file for a bug in a callee (belief.py's bug located to
    learn.py, its caller), while the top-ranked suspect was in the right file
    20 of 20 times."""
    if not suspects:
        return fallback
    best = suspects[0][2]
    if any(rel == fallback and score >= best for rel, _, score in suspects):
        return fallback
    return suspects[0][0]


def _slice_for_failure(failed_test: str, source_relpath: str,
                       max_chars: int = _MAX_SLICE_CHARS,
                       source_line: int | None = None,
                       source_lines: list[int] | None = None) -> dict[str, str]:
    """The model's whole context for one cycle: the failing test's source,
    and the target source file, BOTH cut to the interesting hunks when the
    file is long. Returning {'test': ..., 'source': ...}.

    The test is cut around its own name. The source is cut around
    `source_line` (the traceback's frame) — the test's name never appears in
    the source file, so cutting the source by it handed the model the first
    forty lines of a long file and "(truncated)" instead of the bug.

    `source_lines` (the ranked suspects from devtools.locate_fault) wins over
    `source_line`: one window around each suspect, merged where they touch,
    so a culprit ranked third is still on screen."""
    def suspect_windows(path: Path, lines: list[str], wanted: list[int]) -> str:
        cut: set[int] = set()
        for ln in wanted:
            if not 0 < ln <= len(lines):
                continue
            window = set(range(max(0, ln - 1 - 5), min(len(lines), ln + 5)))
            if len("\n".join(lines[i] for i in sorted(cut | window))) > max_chars:
                break
            cut |= window
        if not cut:
            return ""
        parts, prev = [], None
        for i in sorted(cut):
            if prev is not None and i != prev + 1:
                parts.append("    # … (lines skipped)")
            parts.append(lines[i])
            prev = i
        shown = ", ".join(str(n) for n in wanted[:6])
        return (f"(parts of {path.name}, around the lines the failing test runs "
                f"that the passing tests do not: {shown})\n" + "\n".join(parts))

    def read_slice(path: Path, needle: str, line: int | None = None) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return f"(could not read {path})"
        lines = text.splitlines()
        if len("\n".join(lines)) <= max_chars:
            return "\n".join(lines)
        if source_lines and path == src_path:
            windows = suspect_windows(path, lines, source_lines)
            if windows:
                return windows
        if line is not None and 0 < line <= len(lines):
            # A function-sized window around the frame: wide enough to hold
            # the whole statement the model has to quote exactly.
            # Biased forward: the line is often a `def` (source_from_test)
            # whose body is what needs the edit.
            span = max(_HUNK_CONTEXT, max_chars // 90)
            first = max(0, line - 1 - span // 3)
            last = min(len(lines) - 1, first + span)
            body = "\n".join(lines[first:last + 1])
            return (f"(lines {first+1}-{last+1} of {path.name}; the failure "
                    f"surfaced at line {line})\n{body}")
        hits = [i for i, l in enumerate(lines) if needle in l]
        cut: set[int] = set()
        for h in hits[:6]:
            for j in range(max(0, h - _HUNK_CONTEXT),
                           min(len(lines), h + _HUNK_CONTEXT + 1)):
                cut.add(j)
        if not cut:
            return "\n".join(lines[: max_chars // 90]) + "\n… (truncated)"
        first, last = min(cut), max(cut)
        body = "\n".join(lines[first:last + 1])
        return f"(lines {first+1}-{last+1} of {path.name})\n{body}"

    test_path = constants.PROJECT_DIR / failed_test.split("::")[0]
    src_path = constants.PROJECT_DIR / source_relpath
    # The needle: the test function's name, without its parameterization.
    needle = failed_test.split("::")[-1].split("[")[0]
    return {
        "test": read_slice(test_path, needle),
        "source": read_slice(src_path, needle, source_line),
    }


def repair_failure(failed_test: str, source_relpath: str,
                   generate: Callable[..., str],
                   verify: Callable[[list[str] | None], tuple[bool, str]],
                   log: Callable[[str], None] = print,
                   budget_cycles: int | None = None,
                   trace: str | None = None,
                   suspects: list[tuple[str, int, float]] | None = None,
                   ) -> dict[str, Any]:
    """One failure, driven to fixed-or-defeated.

    `generate(prompt) -> reply` is the model — ONE call per cycle, expected
    to return a patch in the <<<< OLD / ==== / >>>> NEW fence. `verify` runs
    the suite (run_tests); it is called after a patch lands and at the very
    start (the caller may pass a pre-computed first verdict to skip the cold
    run).

    The cycle: read slice → ask for one patch → refuse broken patches →
    apply with backup → verify. Fixed: the target test no longer fails and
    no test fails that was not failing before — judged by test ids, not by
    "the whole suite is green", because a suite with unrelated red (a test
    that needs mlx-vlm on this box) would otherwise make every repair a
    defeat. Worse (a NEW failing id): revert to the cycle's start. Neither:
    keep the patch, ask for the next patch.

    One on-disk backup, `<file>.repairbak`, holds the pre-repair original. It
    is removed when the file ends fixed or back at the original, and kept
    (and named in the record) only when a patch is left in place unfixed.

    `trace` is optional pytest output for the failing test alone (see
    devtools.trace_test): the suite report keeps only its tail, which often
    cuts off the frame that says WHERE the failure surfaced.

    `suspects` is optional devtools.locate_fault output: the lines the failing
    test ran and the passing ones mostly did not. Those in `source_relpath`
    decide what the model is shown.

    Every cycle's prompt carries what the earlier cycles tried and what came
    of it. Without that the second cycle's prompt was the first's, and a
    greedy model answered it with the same patch (measured 2026-10-08: the
    14B sent one identical no-op patch three cycles running).

    Returns the record: what happened, what the suite said, no adjectives.
    """
    src_path = constants.PROJECT_DIR / source_relpath
    original = src_path.read_text(encoding="utf-8")
    pristine = original
    backup = src_path.parent / f"{src_path.name}.repairbak"
    record: dict[str, Any] = {
        "failure": failed_test, "source": source_relpath,
        "cycles": [], "fixed": False, "reverted": False,
    }
    current_verdict = verify(None)     # the caller-supplied verifier runs
    ok, report = current_verdict
    record["initial_verdict"] = (report or "(none)").splitlines()[0][:200]
    initial_failing = _failing_ids(report)
    if not ok and initial_failing and failed_test not in initial_failing:
        # Already passing in a suite that is red for other reasons.
        ok = True

    def finish(rec: dict[str, Any]) -> dict[str, Any]:
        current = src_path.read_text(encoding="utf-8")
        if rec["fixed"] or current == pristine:
            backup.unlink(missing_ok=True)
        elif backup.exists():
            rec["backup"] = str(backup.relative_to(constants.PROJECT_DIR))
        return rec

    max_cycles = max(1, budget_cycles or _MAX_CYCLES_PER_FAILURE)
    suspect_lines = [ln for rel, ln, _ in (suspects or []) if rel == source_relpath]
    tried: list[str] = []
    for cycle in range(1, max_cycles + 1):
        if ok:
            record["fixed"] = True
            return finish(record)

        located = (source_from_report(report, failed_test)
                   or source_from_report(trace or "", failed_test)
                   or source_from_test(failed_test))
        line_hint = (located[1] if located and located[0] == source_relpath
                     else None)
        slice_ = _slice_for_failure(failed_test, source_relpath,
                                    source_line=line_hint,
                                    source_lines=suspect_lines)
        # The suite's ASSERTION DIFF is the contract, not the summary line:
        # a first patch can remove the bug and still miss the expected
        # shape (measured: bug removed, but the unfilled slot left the
        # literal `$2` in the body where the test wants it replaced with
        # ""). The diff's expect/actual lines are what teach that nuance,
        # so they go into every cycle's prompt.
        assertion = "\n".join(
            l for l in (report or "").splitlines() if l.startswith("E "))
        # The prompt is pinned to the shape the model PROVED it can answer
        # (measured fix6: fence + both hunks, no prose) — the failing test,
        # the whole file, the fence template, and only then the diff. Order
        # matters: the fence template sits directly beside what it edits.
        prompt = (
            "You are repairing your own harness. One failure is left.\n\n"
            f"The failing test:\n```python\n{slice_['test']}\n```\n\n"
            f"The file to repair:\n```python\n{slice_['source']}\n```\n\n"
            "Reply with ONE patch, in exactly this fence:\n"
            "<<<< OLD\n(the exact current text to replace — copy it "
            "character for character from the file above)\n====\n"
            "(the corrected text)\n>>>> NEW\n\n"
            "The suite's verdict right now:\n"
            f"{(report or '').splitlines()[0][:300]}\n"
            + (f"Assertion detail (expect vs actual — the patch must make "
               f"the ACTUAL become the EXPECT):\n{assertion[:1200]}\n"
               if assertion else "")
            + ((f"\n{RETRY_MARKER}, and it did NOT fix the test — do not send "
                "these again:\n" + "\n".join(tried[-3:]) + "\n")
               if tried else "")
            + "\nSmallest change that makes the failing test pass without "
            "weakening it. Your reply is ONLY the fence, nothing else."
        )
        reply = generate(prompt)
        cyclerec = {"cycle": cycle,
                    "reply_head": (reply or "").strip()[:160]}
        hunks = _extract_patch(reply) or _block_patch(reply, original)

        def note(outcome: str) -> None:
            shown = "\n".join(f"<<<< OLD\n{o}\n====\n{n}\n>>>> NEW"
                              for o, n in hunks) if hunks else (reply or "")[:300]
            tried.append(f"- {outcome}:\n{shown[:600]}")

        if not hunks:
            cyclerec["result"] = "no parseable patch"
            note("no usable patch (OLD and NEW identical, or no fence)")
            record["cycles"].append(cyclerec)
            if cycle == max_cycles:
                break
            continue

        new_text = _apply_patch(original, hunks)
        if new_text is None:
            # The patch's OLD text missed the CURRENT file. One pinpoint
            # retry this cycle: show the model the exact current bytes
            # around its patch target so its next quote is character-true
            # instead of memory-true. Measured 2026-10-07: the model's
            # old-quote drifts (drops a trailing `)`) after any change; the
            # diagnosis stays right, the quote does not.
            first_line = next((l.strip() for l in
                               (hunks[0][0].splitlines() or [""]) if l.strip()), "")
            if first_line:
                anchor = _slice_for_failure(failed_test, source_relpath,
                                            source_line=line_hint)
                pinpoint = (
                    "Your OLD text did not match the file. Reply with ONLY a "
                    "patch fence, exactly this shape, nothing else:\n\n"
                    "<<<< OLD\n<current file text, character-for-character>\n"
                    "====\n<the corrected text>\n>>>> NEW\n\n"
                    "The CURRENT file text (copy OLD from here exactly, "
                    "including the closing parenthesis of the last line):\n"
                    f"{anchor['source']}\n")
                reply2 = generate(pinpoint)
                hunks2 = _extract_patch(reply2)
                if hunks2:
                    new_text = _apply_patch(original, hunks2)
            if new_text is None:
                cyclerec["result"] = ("patch did not apply, pinpoint retry "
                                      "also missed")
                note("OLD text not found in the file")
                record["cycles"].append(cyclerec)
                if cycle == max_cycles:
                    break
                continue

        # The docstring's promise, kept: a Python patch that will not parse
        # never touches the file (it would turn one failure into a red
        # collection of the whole module).
        if src_path.suffix == ".py":
            try:
                compile(new_text, str(src_path), "exec")
            except SyntaxError as e:
                cyclerec["result"] = f"refused: patch does not parse ({e.msg}, line {e.lineno})"
                note(f"refused, the patched file does not parse ({e.msg})")
                record["cycles"].append(cyclerec)
                continue

        if not backup.exists():
            backup.write_text(pristine, encoding="utf-8")
        src_path.write_text(new_text, encoding="utf-8")

        cycle_report = report
        ok, report = verify(None)
        cyclerec["verdict"] = (report or "(none)").splitlines()[0][:200]
        record["cycles"].append(cyclerec)

        failing = _failing_ids(report)
        new_failures = sorted(failing - initial_failing)
        if not ok and failing and failed_test not in failing and not new_failures:
            ok = True       # target fixed; the rest of the red predates us
        if ok:
            record["fixed"] = True
            return finish(record)

        # Still red. Worse than before? A failing id the cycle did not start
        # with — or, when ids are unreadable, a higher failure count.
        worse = (bool(new_failures) if failing or initial_failing else
                 _bad_count(cyclerec["verdict"])
                 > _bad_count(record.get("initial_verdict", "")))
        if worse:
            src_path.write_text(original, encoding="utf-8")
            record["reverted"] = True
            cyclerec["result"] = ("reverted: new failures "
                                  + ", ".join(new_failures[:3])
                                  if new_failures else
                                  "reverted: more failures than the cycle started with")
            log(f"  [Fix] {cyclerec['result']}")
            note("applied, broke other tests, reverted")
            report = cycle_report      # the file is back; so is its verdict
            # A bad patch is information, not the end: the next cycle sees
            # it in `tried`. Only the budget ends the repair.
            continue
        # No new failures: the patch moved something. Keep it and ask for
        # the next patch against the NEW original.
        note("applied and kept, but the test still fails")
        original = new_text

    record["result"] = "defeated: out of cycles"
    # Leave the LAST applied patch in place if it did no harm (the suite was
    # no worse than it started); a failed fix that broke nothing is the
    # input the next session diagnoses faster. Its backup stays, named.
    return finish(record)


def _failing_ids(report: str) -> set[str]:
    """Failing test ids from run_tests' report (its `Failing:` line, or the
    raw `FAILED <id>` lines pytest prints)."""
    ids: set[str] = set()
    for line in (report or "").splitlines():
        if line.startswith("Failing: "):
            for part in line[len("Failing: "):].split(", "):
                ids.add(part.replace("FAILED ", "").replace("ERROR ", "")
                        .split(" - ")[0].strip())
        elif line.startswith(("FAILED ", "ERROR ")):
            ids.add(line.split(" ", 1)[1].split(" - ")[0].strip())
    ids.discard("")
    return ids


def _bad_count(verdict_line: str) -> int:
    m = re.search(r"(\d+) failed", verdict_line or "")
    e = re.search(r"(\d+) error", verdict_line or "")
    return (int(m.group(1)) if m else 0) + (int(e.group(1)) if e else 0)