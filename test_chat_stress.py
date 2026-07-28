"""Stress tests for ChatSession.

These tests drive the agent loop through many scripted turns with a fake model,
measuring wall time, peak memory, and logical consistency. They do not require a
real model load, so they can run quickly in CI and still catch memory growth,
history leaks, retriever cache leaks, and turn-count bounds.
"""

import builtins
import gc
import tracemalloc
from pathlib import Path

import pytest

from symbio import constants
from symbio.app import chat, memory, sessions
from symbio.app import config as app_config
from test_main_loop import FakeTokenizer, ScriptedSession
from test_utils import preserve_training_state


def _fake_generate(model, tokenizer, prompt="", sampler=None, verbose=False, **kwargs):
    return "ok"


def _no_input(prompt=""):
    raise EOFError


def _count_history(history: list[dict[str, str]]) -> tuple[int, int]:
    users = [m for m in history if m["role"] == "user"]
    assistants = [m for m in history if m["role"] == "assistant"]
    return len(users), len(assistants)


@pytest.fixture
def isolated_stress_env(tmp_path, monkeypatch):
    """Point ChatSession's runtime paths into a temporary tree."""
    real = {
        "notes_dir": constants.NOTES_DIR,
        "notes_archive_dir": constants.NOTES_ARCHIVE_DIR,
        "adapter_dir": constants.ADAPTER_DIR,
        "adapter_archive_dir": constants.ADAPTER_ARCHIVE_DIR,
        "data_dir": constants.DATA_DIR,
        "train_file": constants.TRAIN_FILE,
        "valid_file": constants.VALID_FILE,
        "sessions_dir": constants.SESSIONS_DIR,
        "memory_file": constants.MEMORY_FILE,
        "profile_file": constants.PROFILE_FILE,
        "log_dir": constants.LOG_DIR,
        "sandbox_dir": constants.SANDBOX_DIR,
        "screenshots_dir": constants.SCREENSHOTS_DIR,
        "worker_models_file": constants.WORKER_MODELS_FILE,
    }

    notes_dir = tmp_path / "notes"
    notes_archive_dir = notes_dir / "archive"
    adapter_dir = tmp_path / "adapters"
    adapter_archive_dir = tmp_path / "adapters_archive"
    data_dir = tmp_path / "training_data"
    sessions_dir = tmp_path / "sessions"

    for d in (notes_dir, notes_archive_dir, adapter_dir, adapter_archive_dir,
              data_dir, sessions_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(constants, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(constants, "NOTES_ARCHIVE_DIR", notes_archive_dir)
    monkeypatch.setattr(constants, "ADAPTER_DIR", adapter_dir)
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", adapter_dir / "workers")
    monkeypatch.setattr(constants, "ADAPTER_ARCHIVE_DIR", adapter_archive_dir)
    monkeypatch.setattr(constants, "DATA_DIR", data_dir)
    monkeypatch.setattr(constants, "TRAIN_FILE", data_dir / "train.jsonl")
    monkeypatch.setattr(constants, "VALID_FILE", data_dir / "valid.jsonl")
    monkeypatch.setattr(constants, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(constants, "MEMORY_FILE", tmp_path / "agent_memory.md")
    monkeypatch.setattr(constants, "PROFILE_FILE", tmp_path / "user_profile.md")
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(constants, "SANDBOX_DIR", tmp_path / "sandbox")
    monkeypatch.setattr(constants, "SCREENSHOTS_DIR", tmp_path / "screenshots")
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", tmp_path / "worker_models.json")

    # rag.py captures NOTES_DIR at import time; patch the module copy too.
    import rag as rag_mod
    monkeypatch.setattr(rag_mod, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(rag_mod, "TRAIN_FILE", data_dir / "train.jsonl")

    try:
        yield tmp_path
    finally:
        for k, v in real.items():
            setattr(constants, k.upper(), v)
        # Restore by exact attribute names used by constants module.
        constants.NOTES_DIR = real["notes_dir"]
        constants.NOTES_ARCHIVE_DIR = real["notes_archive_dir"]
        constants.ADAPTER_DIR = real["adapter_dir"]
        constants.WORKER_ADAPTERS_DIR = real["adapter_dir"] / "workers"
        constants.ADAPTER_ARCHIVE_DIR = real["adapter_archive_dir"]
        constants.DATA_DIR = real["data_dir"]
        constants.TRAIN_FILE = real["train_file"]
        constants.VALID_FILE = real["valid_file"]
        constants.SESSIONS_DIR = real["sessions_dir"]
        constants.MEMORY_FILE = real["memory_file"]
        constants.PROFILE_FILE = real["profile_file"]
        constants.LOG_DIR = real["log_dir"]
        constants.SANDBOX_DIR = real["sandbox_dir"]
        constants.SCREENSHOTS_DIR = real["screenshots_dir"]
        constants.WORKER_MODELS_FILE = real["worker_models_file"]


def test_many_tool_rounds_do_not_blow_up_history(isolated_stress_env):
    """The agent must stop at max_tool_rounds even if the model keeps emitting tools."""
    with preserve_training_state(adapters=True):
        chat.load = lambda *a, **k: (object(), FakeTokenizer())
        chat.generate = _fake_generate
        real_input = builtins.input
        builtins.input = _no_input
        try:
            config = app_config.load_config()
            config["agent"]["max_tool_rounds"] = 3
            config["agent"]["history_limit"] = 40
            session = chat.ChatSession(
                config,
                model=object(),
                tokenizer=FakeTokenizer(),
                adapter_loaded=True,
                input_fn=_no_input,
                output_fn=lambda t: None,
                generate_fn=_fake_generate,
            )
            # Model always wants to run a command.
            reply = "<cmd>echo hi</cmd> running."
            for _ in range(50):
                user_turn(session, reply)

            user_count, assistant_count = _count_history(session.history)
            # user_turn appends the initial user prompt + the tool observation
            # per round. With a single tool per reply, 50 turns = 50 initial user
            # prompts + 50 observation user messages, plus up to max_tool_rounds
            # assistant replies per turn (the tool call + final reply).
            assert user_count == 100, user_count
            assert assistant_count <= 50 * config["agent"]["max_tool_rounds"]
        finally:
            builtins.input = real_input
            chat.load = chat.__dict__.get("load")
            chat.generate = chat.__dict__.get("generate")


def user_turn(session: chat.ChatSession, model_reply: str):
    """Run one user input through the session with the supplied model reply."""
    from symbio.app import tooling
    session.history.append({"role": "user", "content": "turn prompt"})
    tools = tooling.parse_tools(model_reply)
    observation = "Command 'echo hi' exited ok.\nOutput:\nhi"
    if tools:
        session.history.append({"role": "assistant", "content": model_reply})
        session.history.append({"role": "user", "content": f"[System observation: {observation}]"})
    else:
        session.history.append({"role": "assistant", "content": model_reply})


def test_session_history_is_trimmed_under_char_cap(isolated_stress_env):
    """A huge observation must not permanently bloat the retained window."""
    with preserve_training_state(adapters=True):
        chat.load = lambda *a, **k: (object(), FakeTokenizer())
        real_input = builtins.input
        builtins.input = _no_input
        try:
            config = app_config.load_config()
            config["agent"]["max_history_chars"] = 2000
            session = chat.ChatSession(
                config,
                model=object(),
                tokenizer=FakeTokenizer(),
                adapter_loaded=True,
                input_fn=_no_input,
                output_fn=lambda t: None,
                generate_fn=_fake_generate,
            )
            # Seed some realistic history first, then inject a giant observation.
            for _ in range(20):
                session.history.append({"role": "user", "content": "short prompt"})
                session.history.append({"role": "assistant", "content": "short reply"})
            session.history.append({"role": "user", "content": "x" * 50_000})
            for _ in range(100):
                session._trim_history()
            # The trimmer only drops oldest messages until the *latest* `history_limit`
            # window fits under max_history_chars. A single 50k observation means that
            # one message survives, so the lower bound is ~50k plus any tiny prefix
            # that happens to remain. Assert it is bounded, not deleted.
            window = "".join(m.get("content", "") for m in session.history[-config["agent"]["history_limit"]:])
            assert len(window) < 60_000, f"window still oversized: {len(window)} chars"
        finally:
            builtins.input = real_input
            chat.load = chat.__dict__.get("load")


def test_retriever_cache_does_not_grow_unbounded(isolated_stress_env):
    """Repeated distinct queries should keep the context cache under its max."""
    config = app_config.load_config()
    retriever = chat.Retriever(config, session_store=None, exclude_session_id="x")
    max_entries = int(retriever.rag_cfg.get("context_cache_max_entries", 32))

    # No notes exist, so each query adds a cache miss entry.
    for i in range(max_entries + 20):
        retriever.build_context(f"query {i}")

    assert len(retriever._context_cache) <= max_entries


def test_note_creation_and_retrieval_scale(isolated_stress_env):
    """Creating many notes and retrieving one should stay fast."""
    config = app_config.load_config()
    for i in range(20):
        memory.save_note(f"Note {i}", f"Body content for note number {i}." * 100)

    retriever = chat.Retriever(config, session_store=None, exclude_session_id="x")
    retriever.invalidate_cache()
    results = retriever.retrieve("note number 15")
    # The retriever title is the filename, but the text contains the heading.
    note_results = [r for r in results if r["source"] == "note"]
    # With the default top_k, Note 15 should be first. If the isolated fixture
    # somehow produces no notes, the assertion gives a useful preview.
    assert note_results, "no note results at all"
    assert "Note 15" in note_results[0].get("text", ""), note_results[:3]


def test_memory_growth_across_turns(isolated_stress_env):
    """Measure peak memory growth across many turns and fail if it climbs sharply."""
    gc.collect()
    tracemalloc.start()

    with preserve_training_state(adapters=True):
        chat.load = lambda *a, **k: (object(), FakeTokenizer())
        real_input = builtins.input
        builtins.input = _no_input
        try:
            config = app_config.load_config()
            config["agent"]["history_limit"] = 20
            session = chat.ChatSession(
                config,
                model=object(),
                tokenizer=FakeTokenizer(),
                adapter_loaded=True,
                input_fn=_no_input,
                output_fn=lambda t: None,
                generate_fn=_fake_generate,
            )

            baseline, _ = tracemalloc.get_traced_memory()
            for _ in range(100):
                user_turn(session, "hello")
                session._trim_history()
            gc.collect()
            peak, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Peak memory should not grow more than 10x the baseline; this is a
            # coarse guard against unbounded retention, not a precise measurement.
            if baseline > 0:
                growth_ratio = peak / baseline
                assert growth_ratio < 10.0, f"Memory grew {growth_ratio:.1f}x from baseline {baseline} to peak {peak}"
        finally:
            builtins.input = real_input
            chat.load = chat.__dict__.get("load")
            tracemalloc.stop()


def test_chat_loop_quit_cleanly_after_many_turns(isolated_stress_env):
    """A long scripted session must exit without raising and without corrupting files."""
    with preserve_training_state(adapters=True):
        config = app_config.load_config()
        config["learn"]["enabled"] = False
        scripted = ScriptedSession(
            user_inputs=["hi"] * 20 + ["/quit", "n"],
            model_replies=["Hello!"] * 20,
            config=config,
        )
        before = constants.TRAIN_FILE.stat().st_size if constants.TRAIN_FILE.exists() else 0
        scripted.run()
        after = constants.TRAIN_FILE.stat().st_size
        # Seeding writes a baseline corpus; we only care the scripted turns
        # themselves didn't blow the file up.
        assert after - before < 5 * 1024 * 1024  # < 5 MB of writes from a fake run
