"""Shared helpers to keep tests from polluting real runtime data."""
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from symbio.constants import ADAPTER_DIR, TRAIN_FILE, VALID_FILE


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
    """
    backups: dict[Path, bytes | None] = {}
    for f in (TRAIN_FILE, VALID_FILE):
        backups[f] = f.read_bytes() if f.exists() else None

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
