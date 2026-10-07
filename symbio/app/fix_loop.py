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
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from symbio import constants

# Per file under repair: how many (patch → verify) cycles before the verdict
# "this failure defeats me" is returned honestly.
_MAX_CYCLES_PER_FAILURE = 2

# The model's patch arrives as one or more exact-text replacements, the same
# shape edit_file uses, inside one fenced block. Single-hunk patches are the
# common case and the parser must read them without ceremony.
_PATCH_RE = re.compile(
    r"<<<<\s*OLD\s*\n(.*?)(?:\n[#\s]*)?====+\s*\n(.*?)(?:\n[#\s]*)?>>>>\s*NEW",
    re.DOTALL)

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
    r"<<<<\s*OLD\s*\n(.*?)(?:\n[#\s]*)?====+\s*\n(.*)", re.DOTALL)

# A slice cap per read: the model's context budget is the whole reason this
# module exists, so each cycle hands it hundreds of lines, not thousands.
_HUNK_CONTEXT = 6           # lines around each match
_MAX_SLICE_CHARS = 3400


def _extract_patch(reply: str) -> list[tuple[str, str]]:
    """(old, new) replacement pairs from the model's reply. Empty = no patch
    the parser accepts, which callers treat as a failed attempt, not an
    error — the loop simply does not touch the file.
    Accepts a TRUNCATED fence (no >>>> NEW close): see _PATCH_TRUNCATED_RE.
    """
    out: list[tuple[str, str]] = []
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
    n, m = len(lines), len(model_lines)
    stripped = [l.rstrip() for l in lines]
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
    # Character span: start of the first matched line to the END of the
    # last matched line (the model's tail-truncated last line gets the
    # file's full line replaced).
    start = sum(len(l) for l in lines[:i])
    end = start + sum(len(l) for l in lines[i:i + m])
    return start, end


def _slice_for_failure(failed_test: str, source_relpath: str,
                       max_chars: int = _MAX_SLICE_CHARS) -> dict[str, str]:
    """The model's whole context for one cycle: the failing test's source,
    and the target source file, BOTH cut to the interesting hunks when the
    file is long. Returning {'test': ..., 'source': ...}."""
    def read_slice(path: Path, needle: str) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return f"(could not read {path})"
        lines = text.splitlines()
        if len("\n".join(lines)) <= max_chars:
            return "\n".join(lines)
        hits = [i for i, l in enumerate(lines) if needle in l]
        cut: set[int] = set()
        for h in hits[:6]:
            for j in range(max(0, h - _HUNK_CONTEXT),
                           min(len(lines), h + _HUNK_CONTEXT + 1)):
                cut.add(j)
        if not cut:
            return "\n".join(lines[: max_chars // 90]) + "\n… (truncated)"
        first, last = min(cut), max(cut)
        body = "\n".join(f"{n+1}: {lines[n]}" for n in range(first, last + 1))
        return f"(lines {first+1}-{last+1} of {path.name} — the needle is here)\n{body}"

    test_path = constants.PROJECT_DIR / failed_test.split("::")[0]
    src_path = constants.PROJECT_DIR / source_relpath
    # The needle: the test function's name, without its parameterization.
    needle = failed_test.split("::")[-1].split("[")[0]
    return {
        "test": read_slice(test_path, needle),
        "source": read_slice(src_path, needle),
    }


def repair_failure(failed_test: str, source_relpath: str,
                   generate: Callable[..., str],
                   verify: Callable[[list[str] | None], tuple[bool, str]],
                   log: Callable[[str], None] = print,
                   budget_cycles: int | None = None,
                   ) -> dict[str, Any]:
    """One failure, driven to fixed-or-defeated.

    `generate(prompt) -> reply` is the model — ONE call per cycle, expected
    to return a patch in the <<<< OLD / ==== / >>>> NEW fence. `verify` runs
    the suite (run_tests); it is called after a patch lands and at the very
    start (the caller may pass a pre-computed first verdict to skip the cold
    run).

    The cycle: read slice → ask for one patch → refuse broken patches →
    apply with backup → verify. Green: stop, report success. Red AND worse:
    revert from the backup, and if this was cycle 2, report defeat. Red but
    not worse: keep the patch, ask for the next patch (the fix was in the
    right neighbourhood).

    Returns the record: what happened, what the suite said, no adjectives.
    """
    src_path = constants.PROJECT_DIR / source_relpath
    original = src_path.read_text(encoding="utf-8")
    record: dict[str, Any] = {
        "failure": failed_test, "source": source_relpath,
        "cycles": [], "fixed": False, "reverted": False,
    }
    current_verdict = verify(None)     # the caller-supplied verifier runs
    ok, report = current_verdict
    record["initial_verdict"] = report.splitlines()[0][:200]

    max_cycles = max(1, budget_cycles or _MAX_CYCLES_PER_FAILURE)
    for cycle in range(1, max_cycles + 1):
        if ok:
            record["fixed"] = True
            return record

        slice_ = _slice_for_failure(failed_test, source_relpath)
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
            + "\nSmallest change that makes the failing test pass without "
            "weakening it. Your reply is ONLY the fence, nothing else."
        )
        reply = generate(prompt)
        cyclerec = {"cycle": cycle,
                    "reply_head": (reply or "").strip()[:160]}
        hunks = _extract_patch(reply)
        if not hunks:
            cyclerec["result"] = "no parseable patch"
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
                anchor = _slice_for_failure(failed_test, source_relpath)
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
                record["cycles"].append(cyclerec)
                if cycle == max_cycles:
                    break
                continue

        # Backup once, on the first real patch.
        backup = src_path.parent / f"{src_path.name}.{cycle}.repairbak"
        if cycle == 1 or not backup.exists():
            backup.write_bytes(src_path.read_bytes())
        cyclerec["backup"] = backup.name
        src_path.write_text(new_text, encoding="utf-8")

        ok, report = verify(None)
        cyclerec["verdict"] = report.splitlines()[0][:200] if report else "(none)"
        record["cycles"].append(cyclerec)

        if ok:
            record["fixed"] = True
            backup.unlink(missing_ok=True)
            return record

        # Still red. Worse than before?
        before = _bad_count(record.get("initial_verdict", ""))
        after = _bad_count(cyclerec["verdict"])
        if after > before:
            src_path.write_bytes(backup.read_bytes())
            record["reverted"] = True
            cyclerec["result"] = "reverted: more failures than the cycle started with"
            log(f"  [Fix] {cyclerec['result']}")
            return record
        # Same-or-fewer failures: the patch moved something. Keep it and ask
        # for the next patch against the NEW original.
        original = src_path.read_text(encoding="utf-8")

    record["result"] = "defeated: out of cycles"
    # Leave the LAST applied patch in place if it did no harm (the suite was
    # no worse than it started); a failed fix that broke nothing is the
    # input the next session diagnoses faster.
    return record


def _bad_count(verdict_line: str) -> int:
    m = re.search(r"(\d+) failed", verdict_line or "")
    e = re.search(r"(\d+) error", verdict_line or "")
    return (int(m.group(1)) if m else 0) + (int(e.group(1)) if e else 0)