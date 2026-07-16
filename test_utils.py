"""Shared helpers to keep tests from polluting real runtime data."""
import shutil
from contextlib import contextmanager
from pathlib import Path

from symbio.constants import ADAPTER_DIR, TRAIN_FILE, VALID_FILE


@contextmanager
def preserve_training_state(adapters: bool = False):
    """Snapshot train/valid data (and optionally the adapter) and restore on exit.

    Tests that mine corrections or run LoRA updates write to the real
    training_data/ files and adapters/ directory; without this guard, test
    junk ("Alice", "Q1") ends up in the user's fine-tune corpus.
    """
    backups: dict[Path, bytes | None] = {}
    for f in (TRAIN_FILE, VALID_FILE):
        backups[f] = f.read_bytes() if f.exists() else None

    adapter_bak: Path | None = None
    if adapters and ADAPTER_DIR.exists():
        adapter_bak = ADAPTER_DIR.with_name(ADAPTER_DIR.name + ".testbak")
        if adapter_bak.exists():
            shutil.rmtree(adapter_bak)
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
