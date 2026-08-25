#!/usr/bin/env python3
"""Verify the four fixes made after the 2026-08-23 keyboard-layout transcript.

That one exchange showed four separate defects, three of them visible:

    Caine   :                                    <- an answerless reply prefix
      [Observation] Noted. Please provide the name of the keyboard layout ...
    Caine   :                                    <- again
      [Blank] Reply came back empty; prompting the model to respond...
    Caine   : I'm waiting for the name of the keyboard layout you're using.
    Huy     : its colemak
    Caine   : <                                  <- a bare tag fragment as the reply
      [Loop] 25x <run_command/> in one reply (25 total tags) - regenerating...

Run this without arguments for the fast offline pass: it drives the exact code
paths those lines came out of, with the exact strings from the transcript, and
loads no model (about two seconds).

Run it with --live to also drive the real ./symb chat the way a person would:
the same two prompts, typed one at a time, with the transcript's failure shapes
scanned for in the output. That costs a headmaster load plus a worker swap, so
budget a few minutes. It answers every [y/N] gate with 'n' and declines to save
the session for training, so it never writes to your corpus.

    ./verify_transcript_fixes.py
    ./verify_transcript_fixes.py --live
    ./verify_transcript_fixes.py --live --ask "what is my keyboard layout?"
"""
from __future__ import annotations

import argparse
import os
import pty
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

# The reply the model actually produced on the "its colemak" turn: 25 <cmd>
# tags and then a truncated 26th. The stripper held that trailing '<' waiting
# for a tag name that never arrived, and strip_tool_tags left it alone because
# there is no tag there to recognize - so it reached the screen as the entire
# visible answer.
LOOPED_REPLY = "<cmd>ls</cmd>" * 25 + "<"

# The delegated worker's reply, verbatim. device_awareness has no tools and no
# view of the machine, so the only thing it could do with "what is my keyboard
# layout" was ask.
WORKER_PUNT = "Noted. Please provide the name of the keyboard layout you are using."

# The transcript itself, trimmed to the lines the live scanner reads. It is the
# scanner's fixture: a check that cannot fail is not a check, so the offline
# pass grades the scanner against the session it was written to catch.
ORIGINAL_TRANSCRIPT = """Huy     : what is my keyboard layout?
  [Tool: delegate_task]
  [Dispatch] Delegating to 'device_awareness': what is my keyboard layout
  [Observation] Noted. Please provide the name of the keyboard layout you are using.
Caine   :{sp}
  [Blank] Reply came back empty; prompting the model to respond...
Caine   : I'm waiting for the name of the keyboard layout you're using.
Huy     : its colemak
Caine   : <
  [Loop] 25x <run_command/> in one reply (25 total tags) - regenerating...
Caine   : You're using a Colemak keyboard layout.""".format(sp=" ")


class Report:
    """Collects pass/fail lines so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if detail:
            for line in detail.splitlines():
                print(f"          {line}")
        if not ok:
            self.failures.append(name)
        return ok

    def section(self, title: str) -> None:
        print(f"\n{title}")

    def finish(self, what: str) -> int:
        print()
        if self.failures:
            print(f"{len(self.failures)} {what} check(s) FAILED:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"All {what} checks passed.")
        return 0


# --------------------------------------------------------------------------
# Offline: the four code paths, exercised directly.
# --------------------------------------------------------------------------

def check_dangling_tag(r: Report) -> None:
    """1. "Caine   : <" - a tag opener left dangling at end of generation."""
    from symbio.app import tooling

    r.section("1. A truncated tag never reaches the screen  (tooling.StreamingStripper)")

    # Fed one character at a time, which is how it actually arrives.
    s = tooling.StreamingStripper(show_reasoning=False)
    shown = "".join(s.feed(c) for c in LOOPED_REPLY) + s.finish()
    r.check(shown == "", "the transcript's 25-tags-plus-'<' reply shows nothing",
            f"was {'<'!r} (printed as the whole reply), now {shown!r}")

    # The fragment must not take a real answer down with it.
    s = tooling.StreamingStripper(show_reasoning=False)
    mixed = "You're using Colemak. <cmd>defaults read"
    shown = "".join(s.feed(c) for c in mixed) + s.finish()
    r.check(shown.strip() == "You're using Colemak.",
            "an answer followed by a truncated tag keeps the answer",
            f"got {shown!r}")

    # And prose that merely contains '<' is still prose.
    s = tooling.StreamingStripper(show_reasoning=False)
    shown = "".join(s.feed(c) for c in "if x < 5 then") + s.finish()
    r.check(shown == "if x < 5 then", "a literal '<' in prose is untouched",
            f"got {shown!r}")


def check_answerless_prefix(r: Report) -> None:
    """2. The bare "Caine   : " line above every [Tool]/[Blank] notice."""
    import mlx.nn as nn  # noqa: F401  (chat imports mlx; fail early if absent)
    from symbio.app import chat

    r.section("2. The reply prefix belongs to an actual reply  (chat._generate_reply)")

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False, enable_thinking=False):
            text = " ".join(f"{m['role']}: {m['content']}\n" for m in messages)
            return text + " assistant:" if add_generation_prompt else text

        def encode(self, text, add_special_tokens=True):
            return text.split(" ")

    class FakeResponse:
        __slots__ = ("text", "token")

        def __init__(self, text, token):
            self.text, self.token = text, token

    def session_for(reply: str):
        def fake_stream(model, tokenizer, prompt, max_tokens=256, sampler=None,
                        prompt_cache=None, **kwargs):
            for i, word in enumerate(reply.split(" ")):
                yield FakeResponse(word if i == 0 else " " + word, word)

        # No model is loaded: make_prompt_cache and friends are the only bits
        # of MLX _generate_reply touches on this path.
        chat.make_prompt_cache = lambda model: []
        chat.can_trim_prompt_cache = lambda cache: True
        chat.trim_prompt_cache = lambda cache, n: cache
        # The spinner writes straight to sys.stdout, which would interleave
        # "thinking..." into this report. It is not what is under test here.
        chat._Spinner.start = lambda self: None
        chat._Spinner.stop = lambda self: None
        config = {
            "assistant_name": "Caine", "user_name": "Huy",
            "agent": {"temperature": 0.1, "top_p": 0.9, "max_reply_tokens": 100,
                      "prompt_cache_enabled": True, "stream_output": True,
                      "max_tool_rounds": 5, "history_limit": 40,
                      "cron_poll_seconds": 9999},
            "tools": {"enabled_groups": []},
            "learn": {}, "memory": {"enabled": False},
            "rag": {"enabled": False}, "web": {},
        }
        return chat.ChatSession(
            config, model=object(), tokenizer=FakeTokenizer(), adapter_loaded=False,
            output_fn=lambda *a, **k: None, generate_fn=lambda *a, **k: "unused",
            stream_fn=fake_stream,
        )

    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]

    # A reply that is nothing but a tool tag: the spinner already cleared its
    # line, so there is no newline owed and certainly no prefix.
    session = session_for("<cmd>ls</cmd>")
    written: list[str] = []
    session.stream_chunk_fn = written.append
    _, streamed_live = session._generate_reply(messages, chunk_prefix="Caine   : ")
    r.check(written == [] and streamed_live is False,
            "a reply that is only a tool tag prints no prefix",
            f"wrote {''.join(written)!r} (was 'Caine   : \\n')")

    # A real answer still gets its prefix and closes its line.
    session = session_for("You're using Colemak.")
    written = []
    session.stream_chunk_fn = written.append
    _, streamed_live = session._generate_reply(messages, chunk_prefix="Caine   : ")
    out = "".join(written)
    r.check(streamed_live is True and out.startswith("Caine   : ") and out.endswith("\n"),
            "a real answer still gets its prefix and a closing newline",
            f"wrote {out!r}")


def check_worker_punt(r: Report) -> None:
    """3. The worker's question relayed to the user as if it were a result."""
    from symbio.app import dispatch

    r.section("3. A worker's question is not relayed as a result  (dispatch)")

    labelled = dispatch.label_worker_reply("device_awareness", WORKER_PUNT)
    r.check("did not answer" in labelled
            and "Do NOT pass its question on to the user" in labelled,
            "the transcript's worker reply is labelled a non-answer",
            f"observation now reads:\n{labelled}")

    r.check("did not answer" in dispatch.label_worker_reply(
                "device_awareness", "I don't have access to your system settings."),
            "'I don't have access' is caught too")

    prose = "The script provides a summary of the log file."
    r.check(dispatch.label_worker_reply("summarize", prose) == prose,
            "a real answer that merely mentions 'provides' is passed through")

    # Told up front, too: catching the punt after the fact has already cost a
    # headmaster unload, a worker load and a generation.
    pool = dispatch.WorkerPool({"dispatch": {}})
    skill_prompt = pool._worker_system_prompt(
        "device_awareness", {"system_prompt": "You are the specialist worker."})
    r.check(skill_prompt.startswith("You are the specialist worker.")
            and "Never ask a question" in skill_prompt,
            "a skill worker is told there is nobody on the other end to ask")

    r.check(pool._worker_system_prompt("browser", {})
            == dispatch.ROLE_SYSTEM_PROMPTS["browser"],
            "the browser action grammar is left byte-identical")


def check_golden_case(r: Report) -> None:
    """4. add_golden_case crashed on every call (UnboundLocalError)."""
    from symbio.app import chat

    r.section("4. add_golden_case does not crash before it validates  (chat)")

    src = (PROJECT_DIR / "symbio" / "app" / "chat.py").read_text(encoding="utf-8")
    body = src.split("def _add_golden_case", 1)[1]
    assign = body.find('ideal_reply = params.get("ideal_reply"')
    use = body.find("scan_for_injection")
    r.check(assign != -1 and use != -1 and assign < use,
            "ideal_reply is assigned before the injection scan reads it",
            "it was read ~25 lines before assignment, so every successful "
            "add_golden_case raised UnboundLocalError")

    # And the guard clauses still fire ahead of both.
    fn = chat.ChatSession._add_golden_case
    msg = fn(object.__new__(chat.ChatSession), {"id": "x", "prompt": ""})
    r.check(msg == "add_golden_case requires a prompt.",
            "the empty-prompt guard still returns before any scan",
            f"got {msg!r}")


def check_config(r: Report) -> None:
    """The config change: Phi-4 14B as the base for new skill workers."""
    import json

    r.section("5. Phi-4 is the base for new skill workers  (config.json)")

    cfg = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    got = cfg.get("dispatch", {}).get("worker_model_name")
    r.check(got == "mlx-community/phi-4-4bit",
            "dispatch.worker_model_name is mlx-community/phi-4-4bit",
            f"got {got!r}")

    models = json.loads((PROJECT_DIR / "models.json").read_text(encoding="utf-8"))
    r.check(models.get("phi4_14b_4bit", {}).get("model_name") == "mlx-community/phi-4-4bit",
            "models.json has a phi4_14b_4bit entry")

    # Existing workers keep their own model_name - the setting is only read
    # when a NEW skill worker is created or trained. Worth stating out loud,
    # because "the config says phi-4" and "your workers run phi-4" are
    # different sentences and only the first one is true.
    workers = json.loads(
        (PROJECT_DIR / "symbio" / "app" / "worker_models.json").read_text(encoding="utf-8"))
    unchanged = sorted({e["model_name"] for e in workers.values()})
    r.check(all("phi-4" not in m for m in unchanged),
            f"the {len(workers)} existing workers keep their own base (no 8 GB download)",
            "they still run: " + ", ".join(unchanged))


def check_scanner(r: Report) -> None:
    """The check on the checks: --live's scanner must still fail the session
    it was written for. Without this, a typo in a regex turns the live pass
    into a green light that reads exactly like a working one."""
    import contextlib
    import io

    r.section("6. The --live scanner still fails the original transcript  (self-check)")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = scan_live_log(ORIGINAL_TRANSCRIPT, "Caine")
    report = buf.getvalue()
    failed = [ln.strip()[6:] for ln in report.splitlines() if ln.strip().startswith("FAIL")]

    r.check(rc == 1 and len(failed) == 4,
            "all four defects are still detected in the original session",
            "\n".join(failed) if failed else "the scanner passed a broken session")


def run_offline() -> int:
    r = Report()
    print("Offline pass - no model is loaded.")
    check_dangling_tag(r)
    check_answerless_prefix(r)
    check_worker_punt(r)
    check_golden_case(r)
    check_config(r)
    check_scanner(r)
    return r.finish("offline")


# --------------------------------------------------------------------------
# Live: drive the real ./symb chat the way a person would.
# --------------------------------------------------------------------------

# What the transcript's bad output looks like, as line patterns. The prefix is
# "{user_name:8}: " / "{assistant_name:8}: ", so the padding is part of it.
def _regressions(assistant: str) -> list[tuple[re.Pattern[str], str]]:
    p = re.escape(f"{assistant:8}: ")
    return [
        (re.compile(rf"^{p}$"),
         f"answerless reply prefix (a bare '{assistant:8}: ' line)"),
        (re.compile(rf"^{p}<[a-z_]*$"),
         f"dangling tag fragment as the reply (an '{assistant:8}: <' line)"),
    ]


class LiveSession:
    """Runs ./symb chat on a pty, answering prompts as they appear.

    Each line is typed only once the previous turn has finished - the same
    one-command-at-a-time pacing you would use by hand.

    A pty, specifically, not a pipe. safety.can_prompt() and
    safety._prompt_confirm() both fall back to sys.stdout.isatty(), so with a
    pipe the CLI concludes that nobody is there to ask: the provenance and
    intent escalations are skipped entirely, and any risk score that does reach
    the confirmation threshold is auto-denied without ever printing a prompt.
    A harness like that reports "no confirmation appeared" for a gate it made
    impossible to observe, and "the tool was refused" without saying by whom.
    Found 2026-08-24, after this harness had been used to call a gate verified.

    PYTHONUNBUFFERED is kept: a pty is line-buffered rather than block-buffered,
    but the variable costs nothing and removes the question.
    """

    def __init__(self, user: str, assistant: str, echo: bool = True) -> None:
        self.user_prompt = f"{user:8}: "
        self.assistant = assistant
        self.echo = echo
        self.buf = ""
        # Only output produced after the last send counts as a new prompt.
        self.watermark = 0
        self._lock = threading.Lock()
        self._master, slave = pty.openpty()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "symbio.app.cli", "chat"],
            cwd=PROJECT_DIR,
            stdin=slave, stdout=slave, stderr=slave, close_fds=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        os.close(slave)
        threading.Thread(target=self._read, daemon=True).start()

    # Spinner frames and colour codes would otherwise land in the buffer that
    # prompt matching and the graders read.
    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

    def _read(self) -> None:
        pending = b""
        while True:
            try:
                chunk = os.read(self._master, 1024)
            except OSError:
                return
            if not chunk:
                return
            pending += chunk
            try:
                text = pending.decode("utf-8")
                pending = b""
            except UnicodeDecodeError:
                # A multi-byte character split across two reads.
                continue
            text = self._ANSI_RE.sub("", text).replace("\r\n", "\n")
            with self._lock:
                self.buf += text
            if self.echo:
                sys.stdout.write(text)
                sys.stdout.flush()

    def wait_for_prompt(self, timeout: float) -> str | None:
        """Block until the CLI asks for something NEW. Returns 'user' for the
        chat prompt, 'yn' for a [y/N] gate, or None on timeout/exit.

        The watermark is the whole point. Without it this matches the prompt
        that is already sitting at the end of the buffer — the one we just
        answered — because the CLI has not echoed anything yet. Found live:
        sending "/quit" returned 'user' again a millisecond later, a second
        "/quit" went down the pipe, and the CLI read THAT as the answer to
        "Save conversation for training? [y/N]". It declined, exited, and the
        next write hit a broken pipe. A verification harness that can silently
        answer the corpus-writing question with the wrong line is worse than
        no harness, so only output produced after the last send counts.
        """
        deadline = time.time() + timeout
        yn = re.compile(r"\[[yY]/[nN]\]:?\s*$")
        while time.time() < deadline:
            with self._lock:
                fresh = self.buf[self.watermark:]
            if fresh.endswith(self.user_prompt):
                return "user"
            if yn.search(fresh):
                return "yn"
            # Checked after the buffer, so output flushed just before exit is
            # not lost to the race.
            if self.proc.poll() is not None:
                return None
            time.sleep(0.05)
        return None

    def send(self, line: str) -> bool:
        """Type one line. False if the CLI has already gone away."""
        if self.proc.poll() is not None:
            return False
        with self._lock:
            self.watermark = len(self.buf)
        try:
            os.write(self._master, (line + "\n").encode())
        except (BrokenPipeError, ValueError, OSError):
            return False
        return True

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except OSError:
            pass
        try:
            os.close(self._master)
        except OSError:
            pass


def run_live(prompts: list[str], timeout: float) -> int:
    import json

    cfg = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    user, assistant = cfg.get("user_name", "You"), cfg.get("assistant_name", "Caine")

    print(f"Live pass - loading {cfg['model_name']} and driving ./symb chat.")
    print("This takes a few minutes. Everything below is the real CLI:\n")
    print("-" * 70)

    session = LiveSession(user, assistant)
    pending = list(prompts) + ["/quit"]
    try:
        while True:
            kind = session.wait_for_prompt(timeout)
            if kind is None:
                break
            if kind == "yn":
                # Decline every gate: a tool confirmation, the adapter-cleanup
                # offer, and above all "Save conversation for training?" - a
                # verification run must never reach the corpus.
                if not session.send("n"):
                    break
                continue
            # "/quit" is the last entry, so pending only empties if the CLI
            # asks again after it; answer it the same way rather than looping.
            if not session.send(pending.pop(0) if pending else "/quit"):
                break
        session.proc.wait(timeout=30)
    except KeyboardInterrupt:
        print("\n(interrupted)")
    except subprocess.TimeoutExpired:
        print("\n(the CLI did not exit; killing it)")
    finally:
        session.close()

    print("-" * 70)
    return scan_live_log(session.buf, assistant)


# The question the transcript ended on: "I'm waiting for the name of the
# keyboard layout you're using." The headmaster had nothing to wait for - it
# was echoing a one-shot worker's dead-end question back at the person who
# asked the original question.
# The worker's own dead-end question, as it appears inside the observation.
_PUNT_RE = re.compile(
    r"\bplease provide\b|\bcould you (?:provide|tell)\b"
    r"|\b(?:don't|do not) have access\b|\bwould need to know\b",
    re.IGNORECASE,
)

_RELAY_RE = re.compile(
    r"\b(?:i'?m|i am) waiting for\b"
    r"|\bplease (?:provide|specify|tell me|let me know)\b"
    r"|\bcould you (?:provide|specify|tell me|let me know)\b"
    r"|\bwhat(?:'s| is) your keyboard layout\b",
    re.IGNORECASE,
)


def _observation_blocks(lines: list[str]) -> list[str]:
    """An observation is a multi-line block; "[Observation]" is only on its
    first line and the rest is indented continuation. Matching line by line
    is what made the punt check skip a run in which a worker had punted."""
    blocks: list[str] = []
    current: list[str] | None = None
    for ln in lines:
        if "[Observation]" in ln:
            if current is not None:
                blocks.append("\n".join(current))
            current = [ln]
        elif current is not None:
            # Continuations are indented and are not the next status marker.
            if ln.startswith("  ") and not re.match(r"\s*\[[A-Z]", ln):
                current.append(ln)
            else:
                blocks.append("\n".join(current))
                current = None
    if current is not None:
        blocks.append("\n".join(current))
    return blocks


def scan_live_log(log: str, assistant: str) -> int:
    """Check a captured session for the transcript's failure shapes.

    Split out from the driver so it can be run against a saved log - which is
    how its own first bug was found. It matched the punt phrases line by line
    inside lines containing "[Observation]", but an observation is a multi-line
    block and the marker only sits on its first line. A worker punted, the
    phrase was three lines down, and this printed "no worker punted this run"
    and passed. A check that quietly declines to check is the one thing a
    verification harness must not do.
    """
    r = Report()
    r.section("Scanning the session for the transcript's failure shapes")

    lines = log.splitlines()
    for pattern, description in _regressions(assistant):
        hits = [ln for ln in lines if pattern.match(ln)]
        r.check(not hits, f"no {description}",
                "" if not hits else f"{len(hits)} occurrence(s): {hits[:3]}")

    # What the assistant actually said, prefix stripped.
    prefix = f"{assistant:8}: "
    replies = [ln[len(prefix):].strip() for ln in lines if ln.startswith(prefix)]

    # A punt has to be recognized from the worker's own words, not from the
    # label: on the pre-fix code there IS no label, and keying off it would
    # make the broken case skip instead of fail.
    punted = [b for b in _observation_blocks(lines)
              if _PUNT_RE.search(b) or "did not answer" in b]
    if "[Dispatch] Delegating to" not in log:
        print("  SKIP  worker-punt labelling (nothing was delegated this run)")
    elif "recommends this tool call" in log:
        # The good path the device_awareness steps were rewritten for: the
        # worker cannot run anything, so it names the call and the headmaster
        # issues it. Asking the user was the failure; recommending is the fix.
        ran = [b for b in _observation_blocks(lines) if "Command '" in b]
        r.check(bool(ran),
                "the worker's recommended command was actually run",
                ran[0].strip() if ran else
                "the headmaster was handed a command and never issued it")
        r.check(not [x for x in replies if _RELAY_RE.search(x)],
                "the assistant answered instead of relaying the question",
                "it replied:\n" + "\n".join(replies))
    elif not punted:
        print("  SKIP  worker-punt labelling (the worker answered; nothing punted)")
    else:
        block = punted[0]
        r.check("did not answer" in block,
                "the worker's dead-end question is labelled a non-answer",
                "" if "did not answer" in block
                else "passed through raw as a result:\n" + block.strip())
        relayed = [x for x in replies if _RELAY_RE.search(x)]
        r.check(not relayed,
                "the assistant answered instead of relaying the question",
                ("it said: " + relayed[0]) if relayed
                else "it replied:\n" + "\n".join(replies))

    if "[Observation]" not in log:
        print("  note: no tool ran this turn, so only the display checks applied.")

    return r.finish("live")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the fixes from the keyboard-layout transcript.")
    parser.add_argument("--live", action="store_true",
                        help="also drive the real ./symb chat (loads the model)")
    parser.add_argument("--ask", action="append", metavar="TEXT",
                        help="prompt to type in --live mode; repeatable "
                             "(default: the transcript's own two turns)")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for one turn (default 900)")
    parser.add_argument("--scan", metavar="LOG",
                        help="re-run the live checks over a saved session log "
                             "instead of starting the CLI")
    args = parser.parse_args()

    if args.scan:
        import json
        cfg = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
        text = Path(args.scan).read_text(encoding="utf-8", errors="replace")
        return scan_live_log(text, cfg.get("assistant_name", "Caine"))

    rc = run_offline()
    if not args.live:
        print("\nRun with --live to also drive the real CLI.")
        return rc

    prompts = args.ask or ["what is my keyboard layout?", "its colemak"]
    print()
    return rc | run_live(prompts, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
