"""Suite-wide isolation for pytest runs.

test_main_loop.run_all() redirects the session store and snapshots training
data for direct `python test_main_loop.py` runs, but pytest invokes the test
functions directly, so without this fixture scripted-session chatter, chat
logs, and mistake notes land in the real sessions/, logs/, training_data/,
and notes/ stores and poison RAG retrieval and the fine-tune corpus.

The worker catalog needs the same guard: saving a skill rewrites the tracked
symbio/app/worker_models.json, so a scripted <skill> reply would otherwise
leave a real skill entry (pointing at whatever model the run happened to use)
committed in the working tree.

Building a ChatSession also runs the post-load self-check, whose web-search
probe makes a real request to api.duckduckgo.com with a 10s timeout. That put
~20s of live network on every session-constructing test (four of them in
test_chat_stress alone) and made the suite's runtime swing with the network.
test_health.py already stubs _check_model_load for the same reason; the probe
is stubbed here so no test reaches the internet.

Boot self-pruning is switched off for the same reason it exists: it archives
junk notes and rewrites session logs, so leaving it on would let any test that
builds a ChatSession quietly prune the user's real notes. The flag alone isn't
enough — several tests hand ChatSession a literal config dict that never went
through DEFAULT_CONFIG — so notes/ is redirected at a scratch directory too.
That also stops the developer's personal notes from steering retrieval
assertions. rag.py keeps its own module-level NOTES_DIR, so both must move.
test_prune.py covers the pruner directly against its own isolated store.
"""
import shutil

import pytest

from symbio import constants
from symbio.app import health, prune
from symbio.app.config import DEFAULT_CONFIG
from test_utils import preserve_training_state

# Absolute paths of the user's real stores, resolved once at import before
# any fixture moves the constants.
_REAL_NOTES = (constants.PROJECT_DIR / "notes").resolve()
_REAL_SESSIONS = (constants.PROJECT_DIR / "sessions").resolve()


def _guard(fn, target, label):
    """Wrap a pruner so it refuses to run against the user's real store.

    Redirecting the constants is the actual isolation; this is the backstop
    that turns a silent, irreversible data loss into a loud test failure.
    A destructive default belongs behind something that fails closed — an
    isolation gap anywhere in the suite (a fixture restoring a stale path, a
    test handing ChatSession a literal config that never saw DEFAULT_CONFIG)
    would otherwise quietly archive real notes or rewrite real chat logs.
    """
    def guarded(*args, **kwargs):
        if target().resolve() == label[1]:
            raise AssertionError(
                f"{label[0]} tried to run against the REAL {label[2]} "
                f"directory. Some test lost its isolation — fix that rather "
                f"than relaxing this guard."
            )
        return fn(*args, **kwargs)
    return guarded


prune.prune_notes = _guard(
    prune.prune_notes, lambda: constants.NOTES_DIR,
    ("prune_notes", _REAL_NOTES, "notes"))
prune.prune_sessions = _guard(
    prune.prune_sessions, lambda: constants.SESSIONS_DIR,
    ("prune_sessions", _REAL_SESSIONS, "sessions"))


@pytest.fixture(autouse=True, scope="session")
def isolate_runtime_state():
    real_sessions = constants.SESSIONS_DIR
    real_logs = constants.LOG_DIR
    real_workers = constants.WORKER_MODELS_FILE
    constants.SESSIONS_DIR = constants.PROJECT_DIR / "sessions.suite"
    constants.SESSIONS_DIR.mkdir(exist_ok=True)
    constants.LOG_DIR = constants.PROJECT_DIR / "logs.suite"
    constants.LOG_DIR.mkdir(exist_ok=True)
    # Work on a copy so the catalog still reads back realistically, but any
    # skill a test saves is written to the copy and thrown away afterwards.
    constants.WORKER_MODELS_FILE = constants.PROJECT_DIR / "worker_models.suite.json"
    if real_workers.exists():
        shutil.copyfile(real_workers, constants.WORKER_MODELS_FILE)
    # Same shape the probe returns when web search is switched off, so the
    # report structure stays realistic without any outbound request.
    real_web_probe = health._check_web_search
    health._check_web_search = lambda config: health._CheckResult(
        "web_search", False,
        message="Web search probe skipped under test.", severity="info",
    )
    real_prune_on_boot = DEFAULT_CONFIG["prune"]["on_boot"]
    DEFAULT_CONFIG["prune"]["on_boot"] = False
    # The persisted system-prefix cache is hundreds of MB; keep any test that
    # reaches the prefill path from writing over the user's real one.
    real_prompt_cache = constants.PROMPT_CACHE_FILE
    constants.PROMPT_CACHE_FILE = (
        constants.PROJECT_DIR / "cache" / "system_prompt.suite.safetensors")
    import rag as _rag
    real_notes = constants.NOTES_DIR
    real_notes_archive = constants.NOTES_ARCHIVE_DIR
    real_rag_notes = _rag.NOTES_DIR
    constants.NOTES_DIR = constants.PROJECT_DIR / "notes.suite"
    constants.NOTES_ARCHIVE_DIR = constants.NOTES_DIR / "archive"
    constants.NOTES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    _rag.NOTES_DIR = constants.NOTES_DIR
    mistakes_before = set(constants.MISTAKES_DIR.glob("*.md")) if constants.MISTAKES_DIR.exists() else set()
    try:
        with preserve_training_state(adapters=True):
            yield
    finally:
        shutil.rmtree(constants.SESSIONS_DIR, ignore_errors=True)
        shutil.rmtree(constants.LOG_DIR, ignore_errors=True)
        constants.WORKER_MODELS_FILE.unlink(missing_ok=True)
        constants.SESSIONS_DIR = real_sessions
        constants.LOG_DIR = real_logs
        constants.WORKER_MODELS_FILE = real_workers
        health._check_web_search = real_web_probe
        DEFAULT_CONFIG["prune"]["on_boot"] = real_prune_on_boot
        constants.PROMPT_CACHE_FILE.unlink(missing_ok=True)
        constants.PROMPT_CACHE_FILE = real_prompt_cache
        shutil.rmtree(constants.NOTES_DIR, ignore_errors=True)
        constants.NOTES_DIR = real_notes
        constants.NOTES_ARCHIVE_DIR = real_notes_archive
        _rag.NOTES_DIR = real_rag_notes
        # Drop mistake notes created by tests (e.g. the "Alice" correction).
        if constants.MISTAKES_DIR.exists():
            for f in set(constants.MISTAKES_DIR.glob("*.md")) - mistakes_before:
                f.unlink(missing_ok=True)
