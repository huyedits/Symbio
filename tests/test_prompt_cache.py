"""Persisted system-prefix KV cache (chat.ChatSession._*_persisted_prompt_cache).

The warmed cache is written once and reloaded on every later start, so a
restart skips re-prefilling the ~4.3k-token system+tools prefix through the
model. The danger is reusing a cache whose *weights* moved: the token ids can
be byte-identical while a swapped adapter makes every cached value wrong, and
nothing downstream would notice. These tests pin the invalidation rules as
hard as the happy path.
"""
import pytest
pytest.importorskip("mlx")  # MLX is Apple Silicon only; skip elsewhere
import mlx.core as mx
from mlx_lm.models.cache import KVCache, can_trim_prompt_cache, trim_prompt_cache

from symbio import constants
from symbio.app import chat


IDS = [1, 2, 3, 4, 5]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "PROMPT_CACHE_FILE",
                        tmp_path / "cache" / "system_prompt.safetensors")
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    (tmp_path / "adapters").mkdir()
    return tmp_path


def make_cache(n_tokens: int = 5, layers: int = 2):
    cache = [KVCache() for _ in range(layers)]
    for kv in cache:
        kv.update_and_fetch(mx.random.normal((1, 2, n_tokens, 4)),
                            mx.random.normal((1, 2, n_tokens, 4)))
    return cache


class FakeSession:
    """The attributes the cache methods actually touch, with the real methods
    bound on so they call each other exactly as they do on a ChatSession."""

    _log_info = chat.ChatSession._log_info
    _prompt_cache_signature = chat.ChatSession._prompt_cache_signature
    _load_persisted_prompt_cache = chat.ChatSession._load_persisted_prompt_cache
    _save_persisted_prompt_cache = chat.ChatSession._save_persisted_prompt_cache
    _await_prompt_cache_prefetch = chat.ChatSession._await_prompt_cache_prefetch
    # Both load paths weigh the cache they just produced, so the cap for the
    # first user turn comes from this model rather than a fallback constant.
    _measure_kv_cost = chat.ChatSession._measure_kv_cost

    def __init__(self, model_name="Qwen/Qwen3-8B-MLX-4bit", adapter_loaded=False):
        self.config = {"model_name": model_name, "agent": {}}
        self.adapter_loaded = adapter_loaded
        self._prompt_cache = None
        self._cached_prompt_ids = None
        self._kv_bytes_per_token = None
        # None means "no prefetch ran". The prefetch only warms the page
        # cache, so its presence or absence changes nothing these tests look
        # at — they are about the signature check, not the read.
        self._prefetch_thread = None
        self.logged: list[str] = []
        self.logger = type("L", (), {"info": lambda s, m: self.logged.append(m)})()


def sign(session, ids=IDS):
    return session._prompt_cache_signature(ids)


def save(session, ids=IDS):
    return session._save_persisted_prompt_cache(ids)


def load(session, ids=IDS):
    return session._load_persisted_prompt_cache(ids)


# ---- signature ----

def test_signature_tracks_the_prompt_tokens(env):
    s = FakeSession()
    assert sign(s, [1, 2, 3])["ids_sha"] != sign(s, [1, 2, 4])["ids_sha"]
    assert sign(s, [1, 2, 3])["ids_sha"] == sign(s, [1, 2, 3])["ids_sha"]


def test_signature_tracks_the_model(env):
    assert sign(FakeSession(model_name="a"))["model_name"] != \
        sign(FakeSession(model_name="b"))["model_name"]


def test_signature_tracks_the_adapter(env):
    weights = constants.ADAPTER_DIR / "adapters.safetensors"
    weights.write_bytes(b"v1")
    s = FakeSession(adapter_loaded=True)
    first = sign(s)["adapter_sig"]

    # Retraining rewrites the adapter: same token ids, different weights.
    weights.write_bytes(b"v2-longer")
    assert sign(s)["adapter_sig"] != first


def test_signature_marks_a_missing_adapter(env):
    s = FakeSession(adapter_loaded=True)
    assert sign(s)["adapter_sig"] == "missing"


def test_signature_without_an_adapter_is_stable(env):
    assert sign(FakeSession())["adapter_sig"] == "none"


# ---- round trip ----

def test_load_is_a_miss_when_nothing_was_saved(env):
    s = FakeSession()
    assert load(s) is False
    assert s._prompt_cache is None


def test_save_then_load_restores_the_cache(env):
    writer = FakeSession()
    writer._prompt_cache = make_cache(n_tokens=5)
    save(writer)
    assert constants.PROMPT_CACHE_FILE.exists()

    reader = FakeSession()
    assert load(reader) is True
    assert reader._cached_prompt_ids == IDS
    # The restored cache holds the same number of processed tokens.
    assert reader._prompt_cache[0].offset == 5


def test_restored_cache_is_still_trimmable(env):
    """_generate_reply trims the stale empty-user tokens off this cache on the
    first real turn; a restored cache that could not be trimmed would be
    rebuilt from scratch and the whole feature would be a no-op."""
    writer = FakeSession()
    writer._prompt_cache = make_cache(n_tokens=5)
    save(writer)

    reader = FakeSession()
    assert load(reader) is True
    assert can_trim_prompt_cache(reader._prompt_cache)
    trim_prompt_cache(reader._prompt_cache, 2)
    assert reader._prompt_cache[0].offset == 3


def test_save_leaves_no_temp_file(env):
    s = FakeSession()
    s._prompt_cache = make_cache()
    save(s)
    # save_prompt_cache appends .safetensors to whatever name it is given, so
    # assert on the whole directory rather than one guessed temp suffix.
    written = sorted(p.name for p in constants.PROMPT_CACHE_FILE.parent.iterdir())
    assert written == [constants.PROMPT_CACHE_FILE.name], written


# ---- invalidation ----

def _saved_then_loaded_by(reader, **writer_kwargs):
    writer = FakeSession(**writer_kwargs)
    writer._prompt_cache = make_cache()
    save(writer)
    return load(reader)


def test_a_different_model_invalidates_the_cache(env):
    hit = _saved_then_loaded_by(FakeSession(model_name="other"), model_name="orig")
    assert hit is False
    # The stale file is removed rather than left to be re-read every start.
    assert not constants.PROMPT_CACHE_FILE.exists()


def test_a_changed_prompt_invalidates_the_cache(env):
    writer = FakeSession()
    writer._prompt_cache = make_cache()
    save(writer, [1, 2, 3])

    reader = FakeSession()
    assert load(reader, [9, 9, 9]) is False
    assert not constants.PROMPT_CACHE_FILE.exists()


def test_a_retrained_adapter_invalidates_the_cache(env):
    """The sharpest case: identical prompt, identical model, new weights."""
    weights = constants.ADAPTER_DIR / "adapters.safetensors"
    weights.write_bytes(b"v1")
    writer = FakeSession(adapter_loaded=True)
    writer._prompt_cache = make_cache()
    save(writer)

    weights.write_bytes(b"v2-different-size")
    reader = FakeSession(adapter_loaded=True)
    assert load(reader) is False
    assert reader._prompt_cache is None


def test_matching_adapter_still_hits(env):
    weights = constants.ADAPTER_DIR / "adapters.safetensors"
    weights.write_bytes(b"v1")
    writer = FakeSession(adapter_loaded=True)
    writer._prompt_cache = make_cache()
    save(writer)

    assert load(FakeSession(adapter_loaded=True)) is True


def test_corrupt_cache_file_is_discarded_not_fatal(env):
    constants.PROMPT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    constants.PROMPT_CACHE_FILE.write_bytes(b"not a safetensors file")

    s = FakeSession()
    assert load(s) is False
    assert not constants.PROMPT_CACHE_FILE.exists()


def test_saving_works_before_the_session_logger_exists(env):
    """Model setup runs from __init__ before self.logger is assigned, so the
    cache helpers must not assume it. An unguarded log call raised into the
    prefill's catch-all, which cleared the cache it had just warmed — the
    optimization silently undoing itself over a logging detail."""
    s = FakeSession()
    del s.logger
    s._prompt_cache = make_cache()

    save(s)  # must not raise

    assert constants.PROMPT_CACHE_FILE.exists()
    reader = FakeSession()
    assert load(reader) is True


def test_loading_works_before_the_session_logger_exists(env):
    constants.PROMPT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    constants.PROMPT_CACHE_FILE.write_bytes(b"corrupt")
    s = FakeSession()
    del s.logger

    assert load(s) is False  # must not raise on the logged discard path


def test_save_failure_is_not_fatal(env, monkeypatch):
    s = FakeSession()
    s._prompt_cache = make_cache()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(chat, "save_prompt_cache", boom)
    save(s)  # must not raise
    assert not constants.PROMPT_CACHE_FILE.exists()
    assert any("Could not save prompt cache" in m for m in s.logged), s.logged


# ---- the browse example must not name a real site ----
#
# Measured: the worked example in the system prompt was
# `<browse>https://www.apple.com</browse>`, and the system prompt is embedded
# in every rendered training sample — 470 occurrences of www.apple.com across
# 227 samples, present in 100% of them. It became the model's default action:
# across three separate sessions it opened apple.com unprompted, once in the
# middle of a request about descaling a kettle.

def test_the_browse_example_uses_a_reserved_domain():
    from symbio.app import prompts

    text = prompts.__dict__.get("SYSTEM_PROMPT") or open(
        prompts.__file__, encoding="utf-8").read()
    browse_lines = [l for l in text.splitlines()
                    if "<browse>" in l and "Example" in l]
    assert browse_lines, "the browse example should still exist"
    for line in browse_lines:
        assert "apple.com" not in line, (
            "a real site as the worked example becomes the model's default "
            "browse target")
        assert "example.com" in line, line


# ---- boot-time prefetch (chat.ChatSession._start_prompt_cache_prefetch) ----
#
# The persisted cache is over a gigabyte, and reading it used to begin only
# after load() had returned — so the two slowest parts of boot ran back to
# back. The prefetch pulls the file into the OS page cache underneath the
# weight load.
#
# It used to call load_prompt_cache on that thread and hand the result over.
# MLX arrays belong to the stream of the thread that made them, so the
# mx.eval in _load_persisted_prompt_cache raised "There is no Stream(cpu, 0)
# in current thread", the handler deleted the file, and the boot re-prefilled
# — every time, on every boot, with the cache reporting as enabled. What these
# tests pin is that the reader thread never touches MLX again.

class PrefetchSession(FakeSession):
    _start_prompt_cache_prefetch = chat.ChatSession._start_prompt_cache_prefetch

    def __init__(self, *a, agent_cfg=None, **kw):
        super().__init__(*a, **kw)
        self.config["agent"] = agent_cfg if agent_cfg is not None else {}


def write_cache_file(**kw):
    """Put a valid cache file on disk, written by a session matching **kw."""
    writer = FakeSession(**kw)
    writer._prompt_cache = make_cache()
    save(writer)


def test_the_prefetch_thread_never_builds_mlx_arrays(env, monkeypatch):
    # The regression. Any load_prompt_cache off the calling thread produces a
    # cache the prefill cannot evaluate, and the failure is a deleted file and
    # a silent 65-second re-prefill rather than an exception anyone sees.
    import threading

    write_cache_file()
    real = chat.load_prompt_cache
    threads = []

    def recording(*a, **kw):
        threads.append(threading.current_thread())
        return real(*a, **kw)

    monkeypatch.setattr(chat, "load_prompt_cache", recording)
    s = PrefetchSession()
    s._start_prompt_cache_prefetch()
    assert load(s) is True
    assert threads == [threading.current_thread()], (
        "the cache was read on a thread other than the one that evaluates it")


def test_the_prefetch_leaves_a_usable_cache_behind(env):
    # End to end: warm the file, then load it. A hit has to survive the
    # prefetch, not merely tolerate it.
    write_cache_file()
    s = PrefetchSession()
    s._start_prompt_cache_prefetch()
    assert load(s) is True
    assert s._cached_prompt_ids == IDS
    assert s._prompt_cache is not None


def test_a_prefetched_file_still_has_to_pass_the_signature_check(env):
    # The whole risk of a fast path is that it skips the check that made the
    # slow path safe. A warmed file whose weights moved must be rejected
    # exactly as a cold read of the same file would be.
    write_cache_file(model_name="Qwen/Qwen3-8B-MLX-4bit")
    s = PrefetchSession(model_name="mistralai/Mistral-7B-v0.3")
    s._start_prompt_cache_prefetch()

    assert load(s) is False
    assert s._prompt_cache is None
    assert not constants.PROMPT_CACHE_FILE.exists(), (
        "a rejected prefetch should discard the stale file, same as a "
        "rejected read")


def test_the_load_waits_for_the_reader_thread(env):
    # Reading the file while the warm is still walking it has both queueing on
    # the same blocks. The load joins first.
    import threading
    import time

    s = PrefetchSession()
    write_cache_file()
    done = []

    def slow():
        time.sleep(0.2)
        done.append(True)

    s._prefetch_thread = threading.Thread(target=slow)
    s._prefetch_thread.start()

    assert load(s) is True
    assert done, "the load read the file before the warm had finished"
    assert s._prefetch_thread is None


def test_prefetch_does_not_start_when_there_is_no_file(env):
    s = PrefetchSession()
    s._start_prompt_cache_prefetch()
    assert s._prefetch_thread is None


@pytest.mark.parametrize("agent_cfg", [
    {"prompt_cache_enabled": False},
    {"persist_prompt_cache": False},
    {"prefetch_prompt_cache_during_load": False},
])
def test_prefetch_respects_its_off_switches(env, agent_cfg):
    write_cache_file()
    s = PrefetchSession(agent_cfg=agent_cfg)
    s._start_prompt_cache_prefetch()
    assert s._prefetch_thread is None


def test_an_unreadable_file_does_not_break_the_prefetch(env):
    # The prefetch runs on a daemon thread with nobody to report to, so a bad
    # file must degrade quietly and let the normal read path do the discard it
    # already knows how to do.
    constants.PROMPT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    constants.PROMPT_CACHE_FILE.write_bytes(b"not a safetensors file")
    s = PrefetchSession()
    s._start_prompt_cache_prefetch()
    assert load(s) is False
    assert not constants.PROMPT_CACHE_FILE.exists()


def test_a_cache_that_cannot_be_materialized_keeps_its_file(env, monkeypatch):
    # A failure to evaluate is a property of this process, not of the bytes.
    # Deleting the file here is what turned one bad eval into a re-prefill on
    # every boot forever after, because the rewritten file failed the same way.
    write_cache_file()
    s = PrefetchSession()

    def boom(_state):
        raise RuntimeError("There is no Stream(cpu, 0) in current thread.")

    # chat resolves mlx.core through mlx_gate, which returns this very module
    # object — so the patch lands on the same attribute the code path touches.
    monkeypatch.setattr(mx, "eval", boom)
    assert load(s) is False
    assert s._prompt_cache is None
    assert constants.PROMPT_CACHE_FILE.exists(), (
        "a process-local failure must not discard a valid cache file")


# ---- the persisted cache must survive a draft model ----
#
# It used to be disabled outright whenever agent.draft_model was set, on the
# grounds that "the file holds one model's layers, while the live cache is two
# models' concatenated". The draft is on by default, so the effect was that no
# cache file was ever written and every boot re-prefilled the whole system
# prompt: measured 74.2s on the 14B, against 0.80s to read it back.
#
# Measured with two real models, the concatenation saves, loads and drives
# speculative generation byte-identically. The only real hazard was the split
# landing in the wrong place, which draft_sig rules out.

class _DraftSession(FakeSession):
    """A FakeSession whose draft model resolves to a stand-in with N layers."""

    def __init__(self, *a, draft_path="models/d-0.6B", draft_layers=24, **kw):
        super().__init__(*a, **kw)
        self.config["agent"]["draft_model"] = draft_path
        self._draft_layers = draft_layers

    def _ensure_draft_model(self):
        if self._draft_layers is None:
            return None
        return object()

    # make_prompt_cache is only used for its length here.
    def _fake_sig(self):
        return f"{self.config['agent']['draft_model']}:{self._draft_layers}"


def _sig_with_draft(session, monkeypatch, layers):
    monkeypatch.setattr(chat, "make_prompt_cache", lambda m: [None] * layers)
    return sign(session)


def test_a_draft_session_still_writes_a_cache_file(monkeypatch, env):
    writer = _DraftSession()
    monkeypatch.setattr(chat, "make_prompt_cache", lambda m: [None] * 24)
    writer._prompt_cache = make_cache()
    save(writer)
    assert constants.PROMPT_CACHE_FILE.exists(), (
        "a draft model must not switch the persisted cache off")
    assert load(_DraftSession()) is True


def test_the_draft_identity_is_part_of_the_signature(monkeypatch, env):
    writer = _DraftSession(draft_layers=24)
    monkeypatch.setattr(chat, "make_prompt_cache", lambda m: [None] * 24)
    writer._prompt_cache = make_cache()
    save(writer)
    # A different draft means a different split point, so the file must miss.
    reader = _DraftSession(draft_layers=28)
    monkeypatch.setattr(chat, "make_prompt_cache", lambda m: [None] * 28)
    assert load(reader) is False


def test_a_draftless_file_is_not_loaded_into_a_draft_session(monkeypatch, env):
    writer = FakeSession()          # no _ensure_draft_model at all -> "none"
    writer._prompt_cache = make_cache()
    save(writer)
    reader = _DraftSession()
    monkeypatch.setattr(chat, "make_prompt_cache", lambda m: [None] * 24)
    assert load(reader) is False, (
        "target-only layers would put the speculative split past the end")


def test_a_draft_file_is_not_loaded_when_the_draft_failed_to_load(monkeypatch, env):
    writer = _DraftSession()
    monkeypatch.setattr(chat, "make_prompt_cache", lambda m: [None] * 24)
    writer._prompt_cache = make_cache()
    save(writer)
    # Same config, but the draft did not come up this run.
    reader = _DraftSession(draft_layers=None)
    assert load(reader) is False, (
        "the signature must key on what loaded, not on what config asks for")


def test_the_draft_signature_is_computed_once(monkeypatch, env):
    calls = []
    session = _DraftSession()
    monkeypatch.setattr(chat, "make_prompt_cache",
                        lambda m: calls.append(1) or [None] * 24)
    sign(session)
    sign(session)
    sign(session)
    assert len(calls) == 1, "the signature is taken on every save and load"


def test_the_tool_catalog_is_refreshed_before_the_prompt_is_built():
    """The system prompt embeds <tools>, so a prompt built before the registry
    is refreshed is a prompt no turn will ever use.

    ChatSession.__init__ used to build self.system_prompt, prefill a KV cache
    from it and persist that cache, and only THEN call refresh_mcp_tools() and
    refresh_delegate_roles() — both of which mutate the module-level registry.
    Measured on a real install: prefilled prefix 5,741 tokens, the prompt every
    turn actually builds 6,164, common prefix 5,221. So 938 tokens were
    re-prefilled on every turn for the life of the session, and most of a
    64-second boot prefill was spent on a prefix discarded before the first
    reply.
    """
    import inspect

    src = inspect.getsource(chat.ChatSession.__init__)
    build = src.index("self.system_prompt = prompts.build_system_prompt")
    for call in ("refresh_mcp_tools", "refresh_delegate_roles"):
        assert src.index(call) < build, (
            f"{call} mutates the tool registry and must run before the system "
            f"prompt embeds it")
    # And the prefill must come after the prompt exists at all.
    assert build < src.index("self._finish_model_setup()")


def test_a_signature_mismatch_says_which_field_moved(env, monkeypatch):
    # Discarding the file buys the next boot a full prefill; doing it silently
    # made a cache that never hit look identical to one that worked.
    writer = FakeSession(model_name="Qwen/Qwen3-8B-MLX-4bit")
    writer._prompt_cache = make_cache()
    save(writer)
    reader = FakeSession(model_name="Qwen/Qwen3-14B-MLX-4bit")
    assert load(reader) is False
    assert any("signature mismatch" in m and "model_name" in m
               for m in reader.logged), reader.logged
