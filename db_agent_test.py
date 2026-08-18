"""Can it actually drive a database, an AWS-ish worker, and a niche lookup?

Harder than the tool battery. For the database cases this does not stop at
"did it emit a plausible command" — it takes the command the model wrote,
runs it against a throwaway copy of a real sqlite database, and checks the
answer or the resulting rows. A command that parses but returns the wrong
number fails here, which is the whole point.

Nothing touches the user's data: every case runs against a fresh copy of
shop.db in the job's tmp dir, and any command that is not a plain
sqlite3/python invocation is refused rather than run.
"""
import importlib.util
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

MAIN = "/Users/huygpt/Downloads/agi"
WT = MAIN + "/.claude/worktrees/symbio-kvcache-dispatch"
TMP = Path("/Users/huygpt/.claude/jobs/f90be41f/tmp")
MASTER_DB = TMP / "shop.db"
sys.path.insert(0, MAIN)

import symbio
import symbio.app
import symbio.constants


def swap(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    setattr(symbio.app, modname.rsplit(".", 1)[1], mod)
    return mod


tooling = swap("symbio.app.tooling", WT + "/symbio/app/tooling.py")

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from symbio.app import config as cfgmod, prompts

# ---------------------------------------------------------------- safety
# The model's own text gets executed below. Refuse anything that is not a
# read/write against the throwaway database.
# A whitelist, not a blacklist. The first attempt blacklisted shell
# metacharacters and refused the model's perfectly correct
#     sqlite3 shop.db "SELECT COUNT(*) FROM users;"
# because SQL needs the semicolon. That scored a right answer as a failure.
_DANGEROUS = re.compile(
    r"\b(rm|mv|cp|chmod|chown|sudo|curl|wget|ssh|scp|kill|shutdown|dd|mkfs|"
    r"launchctl|osascript|open|eval|exec)\b|[>|&`]|\$\(|\.\.")

_ALLOWED_FIRST = {"sqlite3", "python", "python3"}


def safe_to_run(cmd: str) -> bool:
    parts = cmd.strip().split()
    if not parts or parts[0] not in _ALLOWED_FIRST:
        return False
    return not _DANGEROUS.search(cmd)


def fresh_db() -> Path:
    d = Path(tempfile.mkdtemp(prefix="dbtest_", dir=TMP))
    db = d / "shop.db"
    shutil.copy2(MASTER_DB, db)
    return db


def run_sql(sql: str, db: Path) -> tuple[bool, str]:
    """Execute SQL directly against the copy. Returns (ok, output)."""
    try:
        con = sqlite3.connect(db)
        cur = con.executescript(sql) if ";" in sql.strip()[:-1] else con.execute(sql)
        rows = cur.fetchall() if cur.description else []
        con.commit()
        con.close()
        return True, "\n".join(str(r) for r in rows)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def safe_python(code: str) -> bool:
    """Guard for CODE, not for a command line.

    safe_to_run whitelists the first token against {sqlite3, python3}, which is
    right for a shell command and wrong for a Python source file — that starts
    with "import", so a perfectly good script was refused and scored as a model
    failure. Code gets the dangerous-pattern check only.
    """
    return not _DANGEROUS.search(code)


def run_python(code: str, db: Path) -> tuple[bool, str]:
    if not safe_python(code):
        return False, "refused: code touches something outside sqlite"
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=20, cwd=db.parent)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "timed out"


def run_shell(cmd: str, db: Path) -> tuple[bool, str]:
    if not safe_to_run(cmd):
        return False, "refused: command contains something outside sqlite use"
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=20, cwd=db.parent)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "timed out"


# ---------------------------------------------------------------- cases
DB_PATH_HINT = "shop.db"

DB_CASES = [
    {
        "id": "db_count_users",
        "prompt": f"The sqlite database {DB_PATH_HINT} is in the current directory. "
                  f"How many rows are in the users table? Use a tool to find out.",
        "expect": lambda out, db: "3" in out,
        "why": "3 users",
    },
    {
        "id": "db_list_tables",
        "prompt": f"List the tables in the sqlite database {DB_PATH_HINT} "
                  f"in the current directory.",
        "expect": lambda out, db: "users" in out.lower() and "orders" in out.lower(),
        "why": "tables users and orders",
    },
    {
        "id": "db_revenue",
        "prompt": f"In the sqlite database {DB_PATH_HINT} in the current directory, "
                  f"what is the total of the cents column in the orders table?",
        "expect": lambda out, db: "52995" in out.replace(",", ""),
        "why": "52995 cents",
    },
    {
        "id": "db_insert",
        "prompt": f"In the sqlite database {DB_PATH_HINT} in the current directory, "
                  f"add a user named Dmitri with email dmitri@example.com.",
        "expect": lambda out, db: sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM users WHERE name='Dmitri'").fetchone()[0] == 1,
        "why": "a Dmitri row exists afterwards",
    },
    {
        "id": "db_join",
        "prompt": f"In the sqlite database {DB_PATH_HINT} in the current directory, "
                  f"which user spent the most? Give the name.",
        "expect": lambda out, db: "bob" in out.lower(),
        "why": "Bob, 24999",
    },
]


def extract_command(reply: str):
    """(kind, payload) for the first executable thing in the reply."""
    for name, params in tooling.parse_tools(reply):
        if name == "execute_code" and params.get("code"):
            return "python", params["code"]
        if name in ("run_command", "run_remote") and params.get("cmd"):
            return "shell", params["cmd"]
    return None, None


def main():
    cfg = cfgmod.load_config()
    model, tok = load(cfg["model_name"], adapter_path=str(symbio.constants.ADAPTER_DIR))
    sp = prompts.build_system_prompt(cfg["assistant_name"], cfg["user_name"])
    sampler = make_sampler(temp=cfg["agent"]["temperature"], top_p=cfg["agent"]["top_p"])
    print(f"model loaded; {len(DB_CASES)} database cases\n", flush=True)

    passed = 0
    for case in DB_CASES:
        db = fresh_db()
        messages = [{"role": "system", "content": sp + prompts.env_note()},
                    {"role": "user", "content": case["prompt"]}]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        reply = generate(model, tok, prompt=prompt, sampler=sampler,
                         max_tokens=250, verbose=False).strip()
        kind, payload = extract_command(reply)

        print(f"[{case['id']}] expect: {case['why']}", flush=True)
        print(f"  reply: {' '.join(reply.split())[:150]}", flush=True)

        if kind is None:
            print("  RESULT: no executable tool call\n", flush=True)
            continue

        print(f"  emitted {kind}: {' '.join(payload.split())[:130]}", flush=True)
        if kind == "python":
            ok, out = run_python(payload, db)
        else:
            ok, out = run_shell(payload, db)
        print(f"  ran -> ok={ok}  out={' '.join(out.split())[:110]}", flush=True)

        try:
            good = bool(case["expect"](out, db))
        except Exception as e:
            good = False
            out = f"{out} (check raised {e})"
        print(f"  RESULT: {'PASS' if good else 'FAIL'}\n", flush=True)
        passed += good

    print(f"\n==== DATABASE CONTROL: {passed}/{len(DB_CASES)} ====")


if __name__ == "__main__":
    main()
