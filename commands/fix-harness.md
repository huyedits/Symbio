---
name: fix-harness
description: Repair loop - find a failing test in symbio's own suite, fix the source, verify, repeat
argument-hint: [what to fix]
author: assistant
---

You are repairing your own harness: the symbio source in this project. Work
one failure at a time, and let the test suite decide every question — never
your own confidence. The suite is the judge because you are the suspect.

What "fix the harness" covers: $ARGUMENTS. With no arguments, find the suite's
own answer — run it and start from what it reports.

The loop, and its rules:

1. **Get a failure.** Call run_tests. From the report, pick ONE failing test —
   the first, or the one $ARGUMENTS names. Never fix two things at once.
2. **Understand it before touching anything.** read_file the failing test
   first, then the source it exercises. Say in one line what the test expects
   and what the code does instead. If you cannot state the contract, say so to
   the user and stop — do not guess.
3. **Decide the direction, and say which one you took.** Either the test is
   right and the source is wrong (fix the source), or the test encodes a
   contract the user has since changed (propose changing the test — but say
   you are doing it and why, before you do). Never weaken a test to make it
   pass; that is not a fix, it is hiding the failure.
4. **Make ONE edit** with edit_file. Smallest change that could work. Note the
   backup file edit_file created — that is your undo.
5. **Verify, and obey the verdict.** Call run_tests again:
   - Suite green and the target test passes: the fix holds. Move to the next
     failure your original report listed, or report done.
   - Still failing: your fix was wrong. Revert from the backup (write_file the
     .bak's contents back), and take a genuinely different approach — reread
     the test, reread the caller, try the other direction. Two failed attempts
     on the same test: stop, report what you tried and what the test actually
     wants, and let the user decide.
   - MORE tests failing than before: stop immediately, revert everything from
     backups, and report. A fix that breaks neighbours is worse than the bug.
6. **Report like an engineer.** For each fix: the failing test, the one-line
   diagnosis, the file and what changed, and the suite's final verdict. If you
   did not run the test that proves it, say "unverified" — never claim a fix
   the suite did not confirm.

Rules that hold throughout: every claim comes from tool output you have
actually seen in this session; the sandbox cwd applies to terminal commands
but run_tests/edit_file/read_file already work on project paths; if the
failing thing is not in the suite — a behaviour neither test covers — add a
test for it first, watch it fail for the right reason, then fix it.