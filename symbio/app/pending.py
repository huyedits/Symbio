"""A durable record of work that can outlive the process running it.

Everything expensive in Symbio is a fine-tune, and a fine-tune is minutes of
GPU time that exists only in RAM until it writes an adapter. A skill saved with
auto_train on runs its training on a daemon thread; a guarded run holds the
previous adapter in a `.bak` directory it deletes on the way out. Neither of
those survives the process dying, and the process dying is not hypothetical —
a unified-memory Mac under an out-of-memory kill takes the whole session with
it, mid-run, with no unwind and no chance to write anything down.

What was lost was never the compute. It was the *knowledge that the work was
owed*: on the next start nothing remembered that a skill had been seeded but
never trained, or that an adapter directory was half-written with its only
good copy sitting in an orphaned backup. This module writes that down as it
happens, so a reboot resumes instead of restarting.

Deliberately not a scheduler. Nothing here runs a task; it records that one is
owed, notices when its owner died, and repairs what a dead owner left behind.
Deciding to spend ten minutes of GPU on a recovered fine-tune stays with the
operator, because starting one unprompted at boot is a fair description of how
the machine went down in the first place.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio import constants


PENDING_FILE_NAME = "pending_tasks.json"


def pending_file() -> Path:
    """Where the journal lives, resolved per call rather than at import.

    Binding this once at module load would read constants.LOG_DIR before a
    test (or anything else redirecting the project's directories) could point
    it somewhere else, and a journal that ignores that redirection writes real
    entries into the operator's actual list from a test run.
    """
    return constants.LOG_DIR / PENDING_FILE_NAME

# Writers are chat commands, tool calls and the skill auto-train thread, so
# the file has more than one writer in-process. Cross-process races are not
# covered and do not need to be: a duplicate entry is repaired by recover(),
# where a lost one would defeat the point of the file.
_LOCK = threading.Lock()

# States. A task is "running" while a live process owns it, "deferred" when
# nothing owns it and it is waiting to be picked up again — either because
# recovery found its owner dead, or because it was refused before it started
# (no memory to run it safely) and refusing must not mean forgetting.
RUNNING = "running"
DEFERRED = "deferred"


def _boot_id() -> str | None:
    """An identifier that changes when the machine reboots.

    A pid alone cannot answer "is the process that owned this task still
    alive?" across a reboot: the kernel hands out low pids again from scratch,
    so the pid of a trainer killed by Jetsam is very likely live again —
    belonging to something else entirely — by the time anyone asks. Pairing it
    with the boot time makes a stale claim unmistakable, which matters most in
    exactly the case this module exists for.
    """
    try:
        out = subprocess.run(["sysctl", "-n", "kern.boottime"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:  # Linux fallback; harmless everywhere it does not exist.
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                return line.strip()
    except OSError:
        pass
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Someone else's process, but a real one.
    except OSError:
        return True  # Unreadable is not evidence of death.
    return True


def _read() -> list[dict[str, Any]]:
    path = pending_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A truncated file is exactly what a crash mid-write leaves behind.
        # Losing the journal is bad; letting it take the session down with a
        # parse error every start is worse.
        return []
    tasks = data.get("tasks") if isinstance(data, dict) else data
    return [t for t in tasks if isinstance(t, dict)] if isinstance(tasks, list) else []


def _write(tasks: list[dict[str, Any]]) -> None:
    """Replace the journal atomically.

    A partial write here would be its own kind of data loss, and this file is
    written from the same processes that get killed abruptly. Rename is the
    only step that is atomic, so the new content is complete on disk before
    anything points at it.
    """
    path = pending_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps({"tasks": tasks}, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def open_task(kind: str, detail: str, role: str | None = None,
              **extra: Any) -> str:
    """Record that this process is starting a piece of expensive work.

    Returns an id to pass to finish()/update(). Written before the work
    begins, because a record created after the fact does not survive the
    failure it is for.
    """
    task_id = f"{kind}-{os.getpid()}-{datetime.now():%Y%m%d%H%M%S%f}"
    task = {
        "id": task_id,
        "kind": kind,
        "role": role,
        "detail": detail,
        "state": RUNNING,
        "pid": os.getpid(),
        "boot": _boot_id(),
        "started": _now(),
        "updated": _now(),
        "attempts": int(extra.pop("attempts", 0)) + 1,
    }
    task.update(extra)
    with _LOCK:
        tasks = _read()
        # Starting this work is what an outstanding entry for it was asking
        # for, so the new run supersedes it rather than sitting beside it —
        # otherwise every retry of a repeatedly-deferred fine-tune leaves
        # another line on a list nobody can then trust. The attempt count
        # carries over so a run that keeps failing still says so.
        superseded = [t for t in tasks
                      if t.get("kind") == kind and t.get("role") == role
                      and t.get("state") == DEFERRED]
        if superseded:
            task["attempts"] += max(int(t.get("attempts", 1)) for t in superseded)
            tasks = [t for t in tasks if t not in superseded]
        tasks.append(task)
        _write(tasks)
    return task_id


def update(task_id: str, **fields: Any) -> None:
    """Attach facts learned after the task started — most importantly the
    adapter backup directory, which is unknown until the snapshot is taken and
    is the only thing that can undo a half-written adapter."""
    with _LOCK:
        tasks = _read()
        for task in tasks:
            if task.get("id") == task_id:
                task.update(fields)
                task["updated"] = _now()
                _write(tasks)
                return


def finish(task_id: str) -> None:
    """Drop a task that completed — whether it succeeded or failed cleanly.

    A failure that returned is not unfinished business: it reported itself and
    left the adapter in a known state. Only work whose owner vanished without
    saying anything is owed on the next start.
    """
    with _LOCK:
        tasks = [t for t in _read() if t.get("id") != task_id]
        _write(tasks)


def defer(kind: str, detail: str, role: str | None = None,
          reason: str = "", **extra: Any) -> str:
    """Record work that was refused before it ran, so refusing it is not the
    same as dropping it.

    The memory preflight declines a fine-tune it cannot run safely, which is
    correct and, on its own, silent: the run simply never happened and nothing
    remembered it was wanted. Deferred tasks are listed at start-up and can be
    resumed once whatever was holding the memory is gone.
    """
    task_id = open_task(kind, detail, role=role, **extra)
    update(task_id, state=DEFERRED, reason=reason, pid=None)
    return task_id


def all_tasks() -> list[dict[str, Any]]:
    return _read()


def _is_orphan(task: dict[str, Any], boot: str | None) -> bool:
    if task.get("state") != RUNNING:
        return False
    if task.get("boot") and boot and task["boot"] != boot:
        return True  # The machine rebooted under it.
    pid = task.get("pid")
    if not isinstance(pid, int):
        return True
    if pid == os.getpid():
        return False  # Ours, and still running.
    return not _pid_alive(pid)


def orphaned() -> list[dict[str, Any]]:
    """Tasks whose owning process is gone without having finished."""
    boot = _boot_id()
    return [t for t in _read() if _is_orphan(t, boot)]


def deferred() -> list[dict[str, Any]]:
    return [t for t in _read() if t.get("state") == DEFERRED]


def outstanding() -> list[dict[str, Any]]:
    """Everything owed: orphaned work plus work that was never started."""
    boot = _boot_id()
    return [t for t in _read()
            if t.get("state") == DEFERRED or _is_orphan(t, boot)]


def recover(restore_fn=None) -> list[str]:
    """Repair what dead owners left behind and return a line per finding.

    Two different things happen here. The repair is automatic: an orphaned run
    may have left a half-written adapter directory beside a complete backup of
    the last good one, and there is no reading of "resume" under which keeping
    the truncated copy is right. Re-running the *training* is not automatic —
    it is minutes of GPU and a second full copy of the weights, which is the
    load that took the machine down. That decision is reported, not made.
    """
    boot = _boot_id()
    messages: list[str] = []
    with _LOCK:
        tasks = _read()
        changed = False
        for task in tasks:
            if not _is_orphan(task, boot):
                continue
            changed = True
            task["state"] = DEFERRED
            task["pid"] = None
            task["reason"] = "owning process died before it finished"
            task["updated"] = _now()
            label = task.get("detail") or task.get("kind", "task")
            backup = task.get("backup_dir")
            if backup and restore_fn is not None and Path(backup).exists():
                try:
                    restore_fn(Path(backup), task.get("role"))
                    task["restored_from_backup"] = backup
                    # The backup stays on disk. It has done its job, but
                    # deleting adapter weights on the strength of a repair
                    # nobody has looked at yet is not a call this should make
                    # unattended — so it is named instead of orphaned, which
                    # is the difference between a spare copy and the
                    # unexplained .bak directories a crash used to leave.
                    messages.append(
                        f"{label}: interrupted mid-run; restored the adapter "
                        f"from the backup it left behind. That backup is kept "
                        f"at {backup} — delete it once you are happy.")
                    continue
                except Exception as exc:
                    messages.append(
                        f"{label}: interrupted mid-run and its adapter backup "
                        f"at {backup} could not be restored ({exc}). The "
                        f"backup is still there.")
                    continue
            messages.append(f"{label}: interrupted before it finished.")
        if changed:
            _write(tasks)
    return messages


def describe_outstanding() -> list[str]:
    """One human-readable line per piece of owed work."""
    lines = []
    for task in outstanding():
        detail = task.get("detail") or task.get("kind", "task")
        reason = task.get("reason")
        attempts = task.get("attempts", 1)
        line = f"{detail}"
        if reason:
            line += f" — {reason}"
        if attempts > 1:
            line += f" (attempt {attempts})"
        lines.append(line)
    return lines


def clear(kind: str | None = None, role: str | None = None) -> int:
    """Forget recorded tasks, optionally only those matching kind/role.

    No argument means all of them. Returns how many were dropped.
    """
    def matches(task: dict[str, Any]) -> bool:
        return ((kind is None or task.get("kind") == kind)
                and (role is None or task.get("role") == role))

    with _LOCK:
        tasks = _read()
        keep = [t for t in tasks if not matches(t)]
        _write(keep)
        return len(tasks) - len(keep)
