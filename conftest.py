"""Suite-wide isolation for pytest runs.

test_main_loop.run_all() redirects the session store and snapshots training
data for direct `python test_main_loop.py` runs, but pytest invokes the test
functions directly, so without this fixture scripted-session chatter and
mistake notes land in the real sessions/, training_data/, and notes/ stores
and poison RAG retrieval and the fine-tune corpus.
"""
import shutil

import pytest

import main
from symbio.constants import MISTAKES_DIR
from test_utils import preserve_training_state


@pytest.fixture(autouse=True, scope="session")
def isolate_runtime_state():
    real_sessions = main.SESSIONS_DIR
    main.SESSIONS_DIR = main.PROJECT_DIR / "sessions.suite"
    main.SESSIONS_DIR.mkdir(exist_ok=True)
    mistakes_before = set(MISTAKES_DIR.glob("*.md")) if MISTAKES_DIR.exists() else set()
    try:
        with preserve_training_state(adapters=True):
            yield
    finally:
        shutil.rmtree(main.SESSIONS_DIR, ignore_errors=True)
        main.SESSIONS_DIR = real_sessions
        # Drop mistake notes created by tests (e.g. the "Alice" correction).
        if MISTAKES_DIR.exists():
            for f in set(MISTAKES_DIR.glob("*.md")) - mistakes_before:
                f.unlink(missing_ok=True)
