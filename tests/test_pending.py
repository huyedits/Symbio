"""Tests for the durable task journal (symbio.app.pending).

The behaviour under test is what happens when a process does *not* get to run
its cleanup: an out-of-memory kill takes a training run mid-flight, and the
next start has to work out on its own that the work is owed and that the only
intact copy of the adapter is sitting in a backup directory. Every test here
simulates that by writing a journal entry and then never finishing it.
"""

import json
import os

import pytest

from symbio import constants
from symbio.app import pending


@pytest.fixture(autouse=True)
def journal(monkeypatch, tmp_path):
    """Point the journal at a scratch directory.

    pending.pending_file() is resolved per call precisely so this works — an
    import-time constant would have every test in the suite appending to the
    operator's real list of unfinished work.
    """
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path)
    return tmp_path / pending.PENDING_FILE_NAME


# ---- the happy path leaves nothing behind ----

def test_a_finished_task_is_forgotten():
    task_id = pending.open_task("train_worker", "training 'summarize'", role="summarize")
    assert len(pending.all_tasks()) == 1

    pending.finish(task_id)

    assert pending.all_tasks() == []
    assert pending.outstanding() == []


def test_a_task_owned_by_this_live_process_is_not_orphaned():
    """Our own in-flight work is not carried-over work, or every session would
    report the run it is in the middle of as something it needs to resume."""
    pending.open_task("train_worker", "training 'summarize'", role="summarize")

    assert pending.orphaned() == []
    assert pending.outstanding() == []


# ---- the crash path ----

def _kill_the_owner(journal_path, **overrides):
    """Rewrite the journal so its task looks like it belongs to a dead process."""
    data = json.loads(journal_path.read_text(encoding="utf-8"))
    data["tasks"][0].update(overrides)
    journal_path.write_text(json.dumps(data), encoding="utf-8")
    return data["tasks"][0]


def test_a_task_whose_owner_died_is_owed(journal):
    pending.open_task("train_worker", "training 'summarize'", role="summarize")
    # A pid that cannot exist: the kernel rejects it rather than reporting a
    # live process, so this is "definitely gone" rather than "probably".
    _kill_the_owner(journal, pid=0x7FFFFFFF)

    assert len(pending.orphaned()) == 1
    assert len(pending.outstanding()) == 1


def test_a_reboot_orphans_the_task_even_when_the_pid_is_reused(journal):
    """The case this was written for. Jetsam kills the process, the machine
    reboots, and the kernel hands out low pids from scratch — so the pid of the
    dead trainer is very likely alive again, belonging to something else. A pid
    check alone would call that task healthy forever."""
    pending.open_task("train_worker", "training 'summarize'", role="summarize")
    _kill_the_owner(journal, pid=os.getpid(), boot="a previous boot")

    assert len(pending.orphaned()) == 1


def test_recovery_restores_the_adapter_the_dead_run_was_replacing(journal):
    """A killed trainer leaves a half-written adapter directory next to a
    complete backup of the last good one. Keeping the truncated copy means
    loading it as if it were trained."""
    task_id = pending.open_task("train_worker", "training 'summarize'", role="summarize")
    backup = constants.LOG_DIR / "summarize.bak"
    backup.mkdir()
    pending.update(task_id, backup_dir=str(backup))
    _kill_the_owner(journal, pid=0x7FFFFFFF)

    restored = []
    messages = pending.recover(
        restore_fn=lambda d, role: restored.append((d, role)))

    assert restored == [(backup, "summarize")]
    assert any("restored the adapter" in m for m in messages)
    # A repair that leaves an unnamed .bak directory behind is how the
    # unexplained backups in this project accumulated in the first place.
    assert any(str(backup) in m for m in messages)
    assert backup.exists(), "the spare copy is kept, not silently deleted"


def test_recovery_reports_a_backup_it_could_not_restore(journal):
    """Failing to repair is not a reason to drop the record — the backup is
    still on disk and the operator needs to be told where."""
    task_id = pending.open_task("train_worker", "training 'summarize'", role="summarize")
    backup = constants.LOG_DIR / "summarize.bak"
    backup.mkdir()
    pending.update(task_id, backup_dir=str(backup))
    _kill_the_owner(journal, pid=0x7FFFFFFF)

    def boom(_dir, _role):
        raise OSError("disk full")

    messages = pending.recover(restore_fn=boom)

    assert any("could not be restored" in m for m in messages)
    assert any("disk full" in m for m in messages)
    assert len(pending.outstanding()) == 1, "still owed"


def test_recovery_is_idempotent(journal):
    """Start, crash, start again, crash again: the second recovery must not
    re-restore a backup the first one already put back."""
    task_id = pending.open_task("train_worker", "training 'summarize'", role="summarize")
    backup = constants.LOG_DIR / "summarize.bak"
    backup.mkdir()
    pending.update(task_id, backup_dir=str(backup))
    _kill_the_owner(journal, pid=0x7FFFFFFF)

    restored = []
    pending.recover(restore_fn=lambda d, role: restored.append(d))
    pending.recover(restore_fn=lambda d, role: restored.append(d))

    assert len(restored) == 1
    assert len(pending.outstanding()) == 1, "still owed until it is re-run"


# ---- work that was refused before it started ----

def test_deferred_work_is_owed_without_ever_having_run():
    pending.defer("train_worker", "first adapter for skill 'X'", role="x",
                  reason="not enough free memory")

    owed = pending.outstanding()
    assert len(owed) == 1
    assert owed[0]["state"] == pending.DEFERRED
    assert owed[0]["pid"] is None
    assert "not enough free memory" in pending.describe_outstanding()[0]


def test_starting_the_work_supersedes_the_entry_asking_for_it():
    """Otherwise a fine-tune deferred five times leaves five lines describing
    one piece of work, and the list stops being worth reading."""
    pending.defer("train_worker", "first adapter for skill 'X'", role="x")
    pending.defer("train_worker", "first adapter for skill 'X'", role="x")

    task_id = pending.open_task("train_worker", "training 'x'", role="x")

    assert len(pending.all_tasks()) == 1
    assert pending.all_tasks()[0]["id"] == task_id
    assert pending.all_tasks()[0]["attempts"] > 1, "the retries still count"

    pending.finish(task_id)
    assert pending.outstanding() == []


def test_a_different_role_is_not_superseded():
    pending.defer("train_worker", "adapter for 'a'", role="a")
    pending.open_task("train_worker", "training 'b'", role="b")

    assert len(pending.all_tasks()) == 2


# ---- the journal itself has to survive the crash ----

def test_a_truncated_journal_does_not_take_the_session_down(journal):
    """The file is written by the processes that get killed abruptly. Losing
    the list is bad; refusing to start every session afterwards is worse."""
    journal.write_text('{"tasks": [{"id": "half', encoding="utf-8")

    assert pending.all_tasks() == []
    assert pending.recover() == []

    # And it recovers as a working journal rather than staying poisoned.
    task_id = pending.open_task("train_worker", "training 'summarize'")
    assert [t["id"] for t in pending.all_tasks()] == [task_id]


def test_the_journal_is_replaced_atomically(journal, monkeypatch):
    """A crash during the write must leave the previous list intact, not a
    truncated one — so the new content lands under a temporary name and the
    rename is the only step that publishes it."""
    pending.open_task("train_worker", "the first task", role="a")
    before = journal.read_text(encoding="utf-8")

    def die_before_publishing(src, dst):
        raise OSError("killed mid-write")

    monkeypatch.setattr(pending.os, "replace", die_before_publishing)
    with pytest.raises(OSError):
        pending.open_task("train_worker", "the second task", role="b")

    assert journal.read_text(encoding="utf-8") == before


def test_clear_drops_only_what_it_is_asked_to():
    pending.defer("train_worker", "adapter for 'a'", role="a")
    pending.defer("train_worker", "adapter for 'b'", role="b")
    pending.defer("train_headmaster", "headmaster adapter")

    assert pending.clear(kind="train_worker", role="a") == 1
    assert {t["role"] for t in pending.all_tasks()} == {"b", None}
    assert pending.clear() == 2
    assert pending.all_tasks() == []
