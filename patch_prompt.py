#!/usr/bin/env python3
"""Add worked tool-call examples to prompt.md.

Measured, not guessed. Running the same cron battery twice against the same
model — once with the real prompt, once with these three lines appended in
memory — gave:

    invoked     4/9  ->  9/9
    cron_list   0/3  ->  3/3
    cron_schedule 1/3 -> 3/3
    cron_delete   3/3 -> 3/3   (control: already had an example, unchanged)

The mechanism: of the 27 tools declared in the <tools> block, only two have a
worked example anywhere in the prompt. A tool the model has seen declared but
never seen *called* gets a call shape invented for it, and the invented shapes
mostly do not parse — so the tool silently never runs while a success-looking
sentence goes to the user. delete_cron_job is the one cron tool with an
example, and the model reproduces that example character-for-character, which
is what made the control test possible.

prompt.md is gitignored — it is yours, and every install's copy differs — so
this cannot ship as a normal commit. Hence a patcher.

    python3 patch_prompt.py            # apply
    python3 patch_prompt.py --dry-run  # show what would change
    python3 patch_prompt.py --revert   # take them back out

Safe to run twice: it detects its own marker and does nothing.

AFTER APPLYING: the prompt no longer matches the corpus the adapter was
trained on. That is a soft mismatch, not a crash, and the measurement above
was taken with exactly that mismatch in place — so the gain is real without a
retrain. Retraining will resync it; let the golden set gate that, as designed.

DO NOT BLINDLY EXTEND THIS BLOCK.

The obvious next move — an example for each of the ~25 tools that lack one —
was tested across the whole surface, 23 cases, 138 runs. It is not safe:

    completed  53/69 -> 58/69     (77% -> 84%, net +5)

    improved:  config_show 0/3->3/3, cron_list 0/3->3/3, cron_schedule 2/3->3/3,
               add_golden_case 2/3->3/3, read_page 2/3->3/3, digest_notes 1/3->2/3
    REGRESSED: save_memory 3/3->0/3, execute_code 1/3->0/3, compact_memory 3/3->2/3

An example helps a tool the model cannot reach, and can break a neighbouring
tool it was already reaching, by adding a competing candidate next to it —
save_memory worked perfectly until compact_memory was given an example beside
it. The three lines below stay because cron-only was measured on its own and
gained without costing anything: 4/9 -> 9/9, no regressions.

So: add examples for tools the battery shows failing, one group at a time, and
re-run tool_eval before and after. Never in bulk.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / "prompt.md"

MARKER = "<!-- tool-call examples added by patch_prompt.py -->"

ANCHOR = "- Correct cron edit example:"

BLOCK = f"""{MARKER}
- Correct reminder example: <tool_call>{{{{"name": "schedule_job", "arguments": {{{{"schedule": "0 9 * * *", "text": "stand up and stretch"}}}}}}}}</tool_call>
- Correct example for listing reminders: <tool_call>{{{{"name": "list_cron_jobs", "arguments": {{{{}}}}}}}}</tool_call>
- Never answer from memory about what is scheduled. Call list_cron_jobs and read the result.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not PROMPT.exists():
        print(f"No {PROMPT}. Start Symbio once to seed it, then re-run.")
        return 1

    text = PROMPT.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    if args.revert:
        if MARKER not in text:
            print("Not applied; nothing to revert.")
            return 0
        start = next(i for i, ln in enumerate(lines) if MARKER in ln)
        end = start + len(BLOCK.rstrip("\n").split("\n"))
        new = "".join(lines[:start] + lines[end:])
        return _write(new, args.dry_run, "Removed")

    if MARKER in text:
        print("Already applied. Nothing to do.")
        return 0

    idx = next((i for i, ln in enumerate(lines) if ln.startswith(ANCHOR)), None)
    if idx is None:
        # Anchor missing means a customised prompt. Refuse rather than guess a
        # position — a tool example in the wrong section is worse than none,
        # and this file is the user's own writing.
        print(f"Could not find the anchor line in {PROMPT}:\n    {ANCHOR}\n"
              "Your prompt.md has been customised. Add these three lines by "
              "hand, next to the other tool guidance:\n")
        print(BLOCK)
        return 1

    new = "".join(lines[:idx + 1] + [BLOCK] + lines[idx + 1:])
    return _write(new, args.dry_run, "Added")


def _write(new: str, dry_run: bool, verb: str) -> int:
    if dry_run:
        print(f"[dry run] would have {verb.lower()} the block. {len(new)} chars after.")
        return 0
    backup = PROMPT.with_suffix(f".md.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(PROMPT, backup)
    PROMPT.write_text(new, encoding="utf-8")
    print(f"{verb} the tool-call examples in {PROMPT}")
    print(f"Backup: {backup}")
    print("\nRestart Symbio to pick it up. The first start will be slower — the "
          "prompt changed, so the KV cache is rebuilt once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
