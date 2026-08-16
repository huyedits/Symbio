"""Persisted system-prefix KV cache (chat.ChatSession._*_persisted_prompt_cache).

The warmed cache is written once and reloaded on every later start, so a
restart skips re-prefilling the ~4.3k-token system+tools prefix through the
model. The danger is reusing a cache whose *weights* moved: the token ids can
be byte-identical while a swapped adapter makes every cached value wrong, and
nothing downstream would notice. These tests pin the invalidation rules as
hard as the happy path.
"""
import mlx.core as mx
import pytest
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

    def __init__(self, model_name="Qwen/Qwen3-8B-MLX-4bit", adapter_loaded=False):
        self.config = {"model_name": model_name, "agent": {}}
        self.adapter_loaded = adapter_loaded
        self._prompt_cache = None
        self._cached_prompt_ids = None
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
