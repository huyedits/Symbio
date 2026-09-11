"""Ops tasks graded by RUNNING the answer, not by reading it.

Built 2026-09-11 after a string-matching rubric passed a command that does
nothing. Asked to edit a config file, the headmaster proposed

    sed -i 's/oldvalue/newvalue/g' config.txt

which contains sed, names the right file and has the right substitution — and
on BSD userland reads the expression as a backup suffix, errors, and leaves
the file untouched. Every keyword rubric passes it. Only running it and then
looking at the file catches it.

Three stages, and the middle one is the one that matters:

  author    the model writes task + setup + check
  VALIDATE  run setup, run check on the untouched state — it must FAIL.
            A check that already passes is either self-fulfilling (one
            generated check ran `kill -9 $(pgrep sleep)` itself, so it
            passed whatever the model answered) or grades nothing. Of six
            authored tasks, four died here.
  grade     give the task to each arm, run what it proposes, run the check

Nothing is string-matched. The verdict is the state of the filesystem
afterwards, which is the only thing that cannot be argued with.
"""
import json, re, shutil, subprocess, tempfile
from pathlib import Path

# Anything that reaches outside the scratch directory or needs privileges.
FORBIDDEN = re.compile(
    r"\b(sudo|useradd|userdel|groupadd|mkfs|shutdown|reboot|halt|launchctl|"
    r"systemctl|dd\s+if=|curl|wget|pip\s+install|apt|apt-get|brew\s+install|"
    r"chown\s+root|killall|pkill|pgrep)\b|rm\s+-rf\s+/|:\(\)\{")

TIMEOUT = 20


def safe(script: str) -> bool:
    return not FORBIDDEN.search(script or "")


def run(script: str, cwd: Path) -> tuple[int, str]:
    try:
        p = subprocess.run(["/bin/sh", "-c", script], cwd=cwd, timeout=TIMEOUT,
                           capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr)[-400:]
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as e:
        return 125, str(e)


def validate(task: dict) -> tuple[bool, str]:
    """A task earns its place only if its check is both FALSIFIABLE and
    SATISFIABLE — it must fail on the untouched state, and pass once a
    reference solution has run.

    Both halves were learned the hard way from six model-authored tasks. One
    check ran `kill -9 $(pgrep sleep)` itself, so it passed whatever the model
    answered: that is the falsifiable half. Two others could never pass at all
    — `logrotate -f log.txt` is not how logrotate works, and one used `test -o`
    as though it meant "owner" — so both arms failed and the scoreboard read
    like a model failure when it was an authoring failure. A check nobody can
    satisfy grades exactly as much as one everybody satisfies: nothing.
    """
    for field in ("setup", "task", "check", "solution"):
        if not task.get(field):
            return False, f"no {field}"
    for field in ("setup", "check", "solution"):
        if not safe(task[field]):
            return False, f"{field} reaches outside the sandbox"

    box = Path(tempfile.mkdtemp(prefix="opsval-"))
    try:
        rc, out = run(task["setup"], box)
        if rc != 0:
            return False, f"setup does not run: {out[:80]}"
        rc, _ = run(task["check"], box)
        if rc == 0:
            return False, "check already passes — it grades nothing"
        rc, out = run(task["solution"], box)
        if rc != 0:
            return False, f"reference solution errored: {out[:80]}"
        rc, out = run(task["check"], box)
        if rc != 0:
            return False, f"check never passes, even solved: {out[:80]}"
        return True, ""
    finally:
        shutil.rmtree(box, ignore_errors=True)


_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)


def commands_from(reply: str) -> str:
    """The shell the model proposed: fenced blocks, else bare command lines."""
    blocks = _BLOCK.findall(reply or "")
    if blocks:
        return "\n".join(b.strip() for b in blocks)
    lines = [l.strip().lstrip("$ ").strip()
             for l in (reply or "").splitlines()
             if re.match(r"^\s*\$?\s*(ls|cat|grep|sed|awk|tar|gzip|chmod|echo|"
                         r"find|du|df|mv|cp|touch|mkdir|printf)\b", l)]
    return "\n".join(lines)


def grade(task: dict, reply: str) -> tuple[bool, str]:
    script = commands_from(reply)
    if not script:
        return False, "proposed no runnable command"
    if not safe(script):
        return False, "proposed something outside the sandbox"
    box = Path(tempfile.mkdtemp(prefix="opsrun-"))
    try:
        rc, _ = run(task["setup"], box)
        if rc != 0:
            return False, "setup failed"
        run(script, box)                      # its answer, whatever it was
        rc, out = run(task["check"], box)
        return rc == 0, ("check passed" if rc == 0 else f"check failed: {out[:120]}")
    finally:
        shutil.rmtree(box, ignore_errors=True)


SELF_MARK = """You were given this server task:

{task}

You answered with these commands:

{answer}

Did your answer actually accomplish the task on a macOS server? Consider
whether the commands would really work, not whether they look plausible.

Reply with exactly one word: PASS or FAIL."""


def self_mark(task: dict, reply: str, generate_fn) -> bool | None:
    """What the model says about its own answer, or None if it was not asked.

    Measured over 8 tasks before this was wired in: it agreed with the shell
    6/8 and marked PASS on something that failed ZERO times. Both
    disagreements were in the cautious direction, and one of them was better
    than the shell's verdict — `-mtime +122` passes only because that
    arbitrary day count happens to straddle the file dates, and the model said
    FAIL where execution said PASS. Execution cannot see "right by accident";
    this can.

    Not a replacement for running it, for the reason the rest of this file
    exists. That test was favourable — the model judged commands it had just
    written, seconds earlier. Asked about its own actions in a live session
    the same day, the same model reported a posted tweet as probably not sent
    and a submitted form as unclicked.
    """
    if generate_fn is None:
        return None
    shown = commands_from(reply) or (reply or "").strip()
    try:
        verdict = (generate_fn(SELF_MARK.format(task=task["task"],
                                                answer=shown[:400])) or "")
    except Exception:
        return None
    return "PASS" in verdict.strip().upper()[:12]


def assess(task: dict, reply: str, generate_fn=None) -> dict:
    """Both verdicts, and whether they agree.

    The shell decides; the self-mark is recorded beside it rather than
    averaged into it, because a disagreement is the interesting part and an
    aggregate would hide it. A self-marked FAIL on a shell PASS has twice now
    meant the answer was right for the wrong reason.
    """
    passed, detail = grade(task, reply)
    marked = self_mark(task, reply, generate_fn)
    return {
        "id": task.get("id"),
        "passed": passed,
        "detail": detail,
        "self_mark": marked,
        "agreed": None if marked is None else (marked == passed),
    }


TASKS = [
    {
        "id": "config_edit_in_place",
        "setup": "printf 'host=old.example.com\\nport=8080\\n' > app.conf",
        "task": "In app.conf, change the host value from old.example.com to "
                "new.example.com. Edit the file in place.",
        # BSD sed needs the empty backup suffix; GNU does not accept it.
        # Deliberately included: the 14B proposes the GNU form, which on this
        # machine silently changes nothing.
        "solution": "sed -i '' 's/old\\.example\\.com/new.example.com/' app.conf",
        "check": "grep -q '^host=new.example.com$' app.conf && ! grep -q old.example app.conf",
    },
    {
        "id": "count_errors_in_log",
        "setup": ("printf 'INFO ok\\nERROR disk\\nWARN x\\nERROR net\\nINFO ok\\n"
                  "ERROR disk\\n' > app.log"),
        "task": "Count how many lines in app.log contain ERROR and write just "
                "that number to error_count.txt",
        "solution": "grep -c ERROR app.log > error_count.txt",
        "check": "test \"$(tr -d '[:space:]' < error_count.txt)\" = 3",
    },
    {
        "id": "largest_file",
        "setup": ("mkdir -p data && head -c 2000 /dev/zero > data/small.bin && "
                  "head -c 90000 /dev/zero > data/huge.bin && "
                  "head -c 5000 /dev/zero > data/mid.bin"),
        "task": "Find the largest file under data/ and write its filename "
                "(just the name, no path) to biggest.txt",
        "solution": "ls -S data | head -1 > biggest.txt",
        "check": "test \"$(tr -d '[:space:]' < biggest.txt)\" = huge.bin",
    },
    {
        "id": "archive_then_verify",
        "setup": "mkdir -p site && echo a > site/index.html && echo b > site/style.css",
        "task": "Create a gzipped tarball named site.tar.gz containing the site "
                "directory, then confirm it lists both files.",
        "solution": "tar -czf site.tar.gz site",
        "check": ("tar -tzf site.tar.gz | grep -q 'index.html' && "
                  "tar -tzf site.tar.gz | grep -q 'style.css'"),
    },
    {
        "id": "prune_old_logs",
        "setup": ("mkdir -p logs && touch logs/a.log logs/b.log logs/keep.txt && "
                  "touch -t 202001010000 logs/a.log logs/b.log"),
        "task": "Delete every .log file under logs/ that was last modified "
                "before 2021, leaving other files alone.",
        "solution": "find logs -name '*.log' -not -newermt 2021-01-01 -delete",
        "check": ("! test -e logs/a.log && ! test -e logs/b.log && "
                  "test -e logs/keep.txt"),
    },
    {
        "id": "restrict_permissions",
        "setup": "echo 'secret=1' > creds.env && chmod 644 creds.env",
        "task": "Make creds.env readable and writable only by its owner.",
        "solution": "chmod 600 creds.env",
        "check": "test \"$(stat -f '%Lp' creds.env)\" = 600",
    },
    {
        "id": "dedupe_hosts",
        "setup": "printf 'a.com\\nb.com\\na.com\\nc.com\\nb.com\\n' > hosts.txt",
        "task": "Rewrite hosts.txt so each hostname appears only once, sorted "
                "alphabetically.",
        "solution": "sort -u hosts.txt > .h && mv .h hosts.txt",
        "check": ("test \"$(wc -l < hosts.txt | tr -d ' ')\" = 3 && "
                  "test \"$(head -1 hosts.txt)\" = a.com"),
    },
    {
        "id": "disk_usage_report",
        "setup": ("mkdir -p srv/big srv/small && head -c 60000 /dev/zero > srv/big/f && "
                  "head -c 500 /dev/zero > srv/small/f"),
        "task": "Write the name of the subdirectory of srv/ using the most disk "
                "space (just the name) to hog.txt",
        "solution": "du -k srv/* | sort -rn | head -1 | awk '{print $2}' | xargs basename > hog.txt",
        "check": "test \"$(tr -d '[:space:]' < hog.txt)\" = big",
    },
]
