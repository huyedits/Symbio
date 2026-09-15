"""Shared helpers to keep tests from polluting real runtime data."""
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from symbio.constants import ADAPTER_DIR, PROJECT_DIR, TRAIN_FILE, VALID_FILE

# On-disk mirror of the in-memory snapshot below, so an interrupted run can be
# repaired by the *next* one. Named with a leading dot and a fixed path (not a
# tempdir) precisely so it is findable after the process that made it is gone.
CRASH_BACKUP_DIR = PROJECT_DIR / ".pytest-training-backup"
_MANIFEST = CRASH_BACKUP_DIR / "manifest.json"


def _write_crash_backup(backups: dict[Path, bytes | None]) -> None:
    """Mirror the snapshot to disk and fsync it before any test runs.

    Without the fsync this is theatre: a panic can lose the whole write, which
    is the exact failure this guards against.
    """
    shutil.rmtree(CRASH_BACKUP_DIR, ignore_errors=True)
    CRASH_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, (path, data) in enumerate(backups.items()):
        entry: dict[str, object] = {"existed": data is not None}
        if data is not None:
            copy = CRASH_BACKUP_DIR / f"{i}_{path.name}"
            copy.write_bytes(data)
            _fsync(copy)
            entry["backup"] = copy.name
        manifest[str(path)] = entry
    _MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _fsync(_MANIFEST)


def _fsync(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _clear_crash_backup() -> None:
    shutil.rmtree(CRASH_BACKUP_DIR, ignore_errors=True)


def recover_interrupted_training_state() -> list[str]:
    """Restore the corpus if a previous run died before its teardown.

    preserve_training_state restores in a `finally`, which covers exceptions
    and Ctrl-C but nothing that kills the interpreter outright — SIGKILL, an
    OOM kill, a kernel panic. When that happens the corpus keeps whatever test
    junk was in it at the moment of death, and every later run faithfully
    preserves the junk, so it never gets noticed until a fine-tune behaves
    strangely. The leftover manifest is the signal that this happened.

    Returns the paths restored, for the caller to report. Safe to call when no
    interrupted run exists (returns []).
    """
    if not _MANIFEST.exists():
        return []
    try:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A half-written manifest means the crash landed mid-snapshot, before
        # any test could run. Nothing to restore, but clear it so the next run
        # doesn't keep reporting a phantom.
        _clear_crash_backup()
        return []

    restored = []
    for original, entry in manifest.items():
        target = Path(original)
        if not entry.get("existed"):
            if target.exists():
                target.unlink()
                restored.append(original)
            continue
        source = CRASH_BACKUP_DIR / str(entry.get("backup", ""))
        if not source.exists():
            continue
        if target.exists() and target.read_bytes() == source.read_bytes():
            continue  # Died after teardown restored it; nothing to undo.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        restored.append(original)
    _clear_crash_backup()
    return restored


@contextmanager
def preserve_training_state(adapters: bool = False):
    """Snapshot train/valid data (and optionally the adapter) and restore on exit.

    Tests that mine corrections or run LoRA updates write to the real
    training_data/ files and adapters/ directory; without this guard, test
    junk ("Alice", "Q1") ends up in the user's fine-tune corpus.

    Safe to nest (e.g. a suite-wide session fixture wrapping individual
    tests that also use this): each call gets its own uniquely-named
    backup directory, so an inner call's cleanup can never delete an
    outer call's still-pending backup.

    The snapshot is also mirrored to disk so a run killed outright — where the
    `finally` below never executes — can be repaired by the next run via
    recover_interrupted_training_state(). Only the outermost call writes that
    mirror: a nested call's snapshot already contains the outer call's junk,
    and letting it overwrite would make the crash file useless.
    """
    backups: dict[Path, bytes | None] = {}
    for f in (TRAIN_FILE, VALID_FILE):
        backups[f] = f.read_bytes() if f.exists() else None

    owns_crash_backup = not CRASH_BACKUP_DIR.exists()
    if owns_crash_backup:
        _write_crash_backup(backups)

    adapter_bak: Path | None = None
    if adapters and ADAPTER_DIR.exists():
        adapter_bak = Path(tempfile.mkdtemp(
            dir=ADAPTER_DIR.parent, prefix=ADAPTER_DIR.name + ".testbak."))
        shutil.rmtree(adapter_bak)  # mkdtemp creates it; copytree needs the target absent
        shutil.copytree(ADAPTER_DIR, adapter_bak)

    try:
        yield
    finally:
        for f, data in backups.items():
            if data is None:
                f.unlink(missing_ok=True)
            else:
                f.write_bytes(data)
        if adapter_bak is not None:
            shutil.rmtree(ADAPTER_DIR, ignore_errors=True)
            shutil.copytree(adapter_bak, ADAPTER_DIR)
            shutil.rmtree(adapter_bak)
        if owns_crash_backup:
            _clear_crash_backup()
