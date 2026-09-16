"""Stay online: keep the model resident, and run what is due without a human.

Until now every scheduled job ran inside a ChatSession's background thread, so
"every morning at 8" meant "every morning at 8, IF someone happens to have a
chat window open". With no client attached the daemon sits idle and nothing
fires. That is not a scheduler, it is a reminder that only works while you are
watching.

This is the process that watches instead. It does three things on a loop:

  * keeps the resident model alive — restarting the daemon when it dies, with
    a backoff, because a crash loop that respawns instantly burns the machine
    it is meant to keep useful;
  * ticks the cron table itself and hands each fired job to the daemon as an
    ordinary turn, so the work actually happens;
  * writes a heartbeat file, so "is it up?" is a question with an answer.

**Unattended turns are denied by default.** The daemon asks its client for
approval on a risky tool, and here the client is a loop — nobody is reading
the prompt. Answering yes for the user is how an agent ends up running
`rm -rf` at 3am on its own authority, so the default answer is NO and the
denial is recorded in the heartbeat. `cron.unattended_approve` turns that off
deliberately, for someone who has read this paragraph.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.app import cron

HEARTBEAT_FILE = constants.PROJECT_DIR / "logs" / "supervisor.json"

# Backoff between restart attempts. A daemon that dies on startup — a bad
# model path, no memory — would otherwise be respawned in a tight loop, and
# each attempt maps several GB of weights.
BACKOFF_START = 5.0
BACKOFF_MAX = 300.0


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def daemon_ready() -> bool:
    """Is a loaded model listening? The socket AND a live pid, never one."""
    try:
        if not constants.DAEMON_SOCKET.exists():
            return False
        pid = int(constants.DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def start_daemon() -> bool:
    """Ask for a daemon. True when the request was accepted, not when loaded."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "symbio.app.cli", "daemon", "start"],
            capture_output=True, text=True, timeout=60,
            cwd=str(constants.PROJECT_DIR))
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ask_daemon(text: str, approve: bool = False, timeout: float = 600.0,
               collect_output: bool = False) -> str:
    """Put one turn to the resident model and return what it said.

    Speaks the daemon's own line protocol -- the same one `symb chat` and the
    desktop window use. `approve` answers the confirmation gate; it is False
    here because nobody is reading the prompt.
    """
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect(str(constants.DAEMON_SOCKET))
    except OSError as e:
        return f"[supervisor] could not reach the resident model: {e}"

    reply: list[str] = []
    denied = 0
    reader = client.makefile("rb")
    writer = client.makefile("wb")

    def send(message: dict[str, Any]) -> None:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        writer.flush()

    try:
        asked = False
        while True:
            line = reader.readline()
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            kind = message.get("type")
            if kind == "input_prompt":
                if not asked:
                    send({"type": "input", "text": text})
                    asked = True
                    continue
                break                      # the turn finished
            if kind == "stream":
                reply.append(message.get("text", ""))
            elif kind == "output" and collect_output:
                # The tool lines a turn prints -- "[Tool: recall]", "[Result]
                # …". A grader that has to tell "looked it up" from "said it
                # from memory" needs them; a cron job does not.
                reply.append("\n" + message.get("text", ""))
            elif kind == "confirm":
                denied += 0 if approve else 1
                send({"type": "confirm", "answer": bool(approve)})
            elif kind == "done":
                break
    except (OSError, socket.timeout):
        reply.append(" [supervisor] the turn was cut short.")
    finally:
        try:
            client.close()
        except OSError:
            pass
    answer = "".join(reply).strip()
    if denied:
        answer += (f"\n[supervisor] {denied} approval request(s) were declined: "
                   "nobody was watching.")
    return answer or "[supervisor] the model returned nothing."


def write_heartbeat(state: dict[str, Any]) -> None:
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def read_heartbeat() -> dict[str, Any]:
    try:
        return json.loads(HEARTBEAT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def tick(config: dict[str, Any], state: dict[str, Any],
         ask=ask_daemon, ready=daemon_ready, start=start_daemon,
         now: datetime | None = None) -> dict[str, Any]:
    """One pass of the loop. Pure enough to test: every effect is injected.

    Order matters. The model is checked first, because a job that fires while
    nothing is loaded would be handed to a socket that is not there and lost —
    cron.check_due_jobs marks a job as run when it fires, not when it is
    answered.
    """
    state = dict(state)
    state["checked_at"] = _now()

    if not ready():
        state["model"] = "starting"
        state["restarts"] = int(state.get("restarts", 0)) + 1
        state["last_restart"] = _now()
        started = start()
        state["backoff"] = (BACKOFF_START if started
                            else min(BACKOFF_MAX, max(BACKOFF_START,
                                                      float(state.get("backoff", BACKOFF_START)) * 2)))
        state["last_error"] = "" if started else "could not start the resident model"
        return state

    state["model"] = "ready"
    state["backoff"] = BACKOFF_START

    approve = bool(config.get("cron", {}).get("unattended_approve", False))
    try:
        fired = cron.check_due_jobs(config, now=now)
    except Exception as e:
        state["last_error"] = f"cron check failed: {e}"
        return state

    if not fired:
        return state

    state["last_error"] = ""
    ran = list(state.get("ran", []))
    for job in fired:
        answer = ask(job, approve=approve)
        ran.append({"at": _now(), "job": job.splitlines()[0][:160],
                    "answer": answer[:400]})
    state["ran"] = ran[-20:]
    state["jobs_run"] = int(state.get("jobs_run", 0)) + len(fired)
    return state


def run(config: dict[str, Any], interval: float = 30.0,
        rounds: int | None = None) -> int:
    """The loop. `rounds` bounds it for tests; None runs until stopped."""
    state = read_heartbeat()
    state.setdefault("started_at", _now())
    state.setdefault("restarts", 0)
    state.setdefault("jobs_run", 0)
    print(f"  Supervisor up. Watching the resident model and the cron table "
          f"every {interval:g}s. Ctrl+C to stop.")
    passes = 0
    try:
        while rounds is None or passes < rounds:
            state = tick(config, state)
            write_heartbeat(state)
            passes += 1
            if rounds is not None and passes >= rounds:
                break
            # A failed start backs off; a healthy pass waits the interval.
            time.sleep(float(state.get("backoff", BACKOFF_START))
                       if state.get("model") == "starting" else interval)
    except KeyboardInterrupt:
        state["model"] = "stopped"
        state["checked_at"] = _now()
        write_heartbeat(state)
        print("\n  Supervisor stopped. The resident model is left running.")
    return 0
