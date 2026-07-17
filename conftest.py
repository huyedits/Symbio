"""Suite-wide isolation for pytest runs.

test_main_loop.run_all() redirects the session store and snapshots training
data for direct `python test_main_loop.py` runs, but pytest invokes the test
functions directly, so without this fixture scripted-session chatter, chat
logs, and mistake notes land in the real sessions/, logs/, training_data/,
and notes/ stores and poison RAG retrieval and the fine-tune corpus.
"""
import shutil

import pytest

from symbio import constants
from test_utils import preserve_training_state


@pytest.fixture(autouse=True, scope="session")
def isolate_runtime_state():
    real_sessions = constants.SESSIONS_DIR
    real_logs = constants.LOG_DIR
    constants.SESSIONS_DIR = constants.PROJECT_DIR / "sessions.suite"
    constants.SESSIONS_DIR.mkdir(exist_ok=True)
    constants.LOG_DIR = constants.PROJECT_DIR / "logs.suite"
    constants.LOG_DIR.mkdir(exist_ok=True)
    mistakes_before = set(constants.MISTAKES_DIR.glob("*.md")) if constants.MISTAKES_DIR.exists() else set()
    try:
        with preserve_training_state(adapters=True):
            yield
    finally:
        shutil.rmtree(constants.SESSIONS_DIR, ignore_errors=True)
        shutil.rmtree(constants.LOG_DIR, ignore_errors=True)
        constants.SESSIONS_DIR = real_sessions
        constants.LOG_DIR = real_logs
        # Drop mistake notes created by tests (e.g. the "Alice" correction).
        if constants.MISTAKES_DIR.exists():
            for f in set(constants.MISTAKES_DIR.glob("*.md")) - mistakes_before:
                f.unlink(missing_ok=True)
