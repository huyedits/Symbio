"""Chat under 15 seconds on a 16 GB Mac, without swapping the model out.

Measured 2026-09-26 on the 14B-3bit: "hey, how's it going?" took 18.3 s, of
which about 11 s was a reasoning block before "Hi! How can I help you today?";
and an idle resident daemon had 6.6 GB of its 8.7 GB in the compressor, so
every turn first paid to bring the weights back. These pin the two fixes —
think only when the turn is work, keep the weights wired — and the start lock
that keeps two starters from loading two models.
"""
import os
import threading
import time
import types

import pytest

from symbio.app.chat_text import needs_thinking


@pytest.mark.parametrize("text", [
    "hey, how's it going?", "what's 17 times 23?", "tell me a joke",
    "thanks! remind me what I asked first", "what is the capital of france",
])
def test_conversation_answers_without_a_reasoning_block(text):
    assert needs_thinking(text) is False


@pytest.mark.parametrize("text", [
    "fix this error: KeyError 'model_name'", "search the web for mlx news",
    "can you write a python script that renames files", "open safari",
    "look at https://example.com and summarise it", "why does train.py crash?",
    "think carefully: which of these three plans is cheapest?",
    " ".join(["word"] * 40),
])
def test_work_keeps_the_configured_thinking(text):
    assert needs_thinking(text) is True


def _session(level="max", when=None):
    from symbio.app.chat import ChatSession

    session = ChatSession.__new__(ChatSession)
    agent = {"thinking_level": level}
    if when:
        agent["think_when"] = when
    session.config = {"agent": agent}
    return session


def test_turn_thinking_skips_chat_and_keeps_work_and_tool_rounds():
    session = _session("max")
    assert session.turn_thinking("hey there") == (False, 0)
    assert session.turn_thinking("debug this traceback")[0] is True
    assert session.turn_thinking("ok", working=True)[0] is True, "mid tool loop: keep it"


def test_always_is_the_old_behaviour_and_none_stays_none():
    assert _session("max", "always").turn_thinking("hey")[0] is True
    assert _session("none").turn_thinking("debug this traceback") == (False, 0)


def test_the_resident_weights_are_wired(monkeypatch):
    """The limit set here is what mlx_lm's per-generation wired_limit restores
    afterwards, so the model stays resident between turns too."""
    from symbio.app import config as cfg

    calls = []
    fake_mx = types.SimpleNamespace(
        metal=types.SimpleNamespace(is_available=lambda: True),
        device_info=lambda: {"max_recommended_working_set_size": 12 * 2**30},
        set_wired_limit=lambda n: calls.append(n) or 0)
    monkeypatch.setattr("symbio.mlx_gate.attr", lambda name: fake_mx)
    weights = types.SimpleNamespace(nbytes=6 * 2**30)
    model = types.SimpleNamespace(parameters=lambda: {"w": weights})
    monkeypatch.setattr("mlx.utils.tree_flatten", lambda tree: list(tree.items()))

    assert cfg.keep_model_resident(model, {"gpu": {}}) == 6 * 2**30 + 1536 * 2**20
    assert calls == [6 * 2**30 + 1536 * 2**20]
    assert cfg.keep_model_resident(model, {"gpu": {"keep_model_wired": False}}) is None
    assert cfg.keep_model_resident(model, {"gpu": {"wired_limit_mb": 4096}}) is None


def test_a_follow_up_turn_replays_the_last_prompt_as_its_prefix(monkeypatch, tmp_path):
    """The cache reuses only what is byte-identical. The standing context
    (curated memory, env) now sits in the fixed prefix, and the last turn's
    user message is re-rendered with the context it was sent with, so turn
    two's prompt begins with turn one's instead of diverging at its first user
    message and re-prefilling everything after it (~880 tokens, ~9 s on the
    14B, measured on the user's own turns)."""
    from tests.test_thinking_truncation import _reply, _session

    output, prompts = [], []

    def fake_generate(messages, chunk_prefix="", timings=None, think=True, reasoning_budget=0):
        prompts.append([dict(m) for m in messages])
        return _reply(timings, "Sure.")

    session = _session(monkeypatch, tmp_path, output)
    monkeypatch.setattr(session, "_generate_reply", fake_generate)
    session._agent_turn("hey there")
    session._agent_turn("and how are you?")

    first, second = prompts[0], prompts[-1]
    assert second[:len(first)] == first, "turn two must start with turn one's prompt"


def test_two_starters_at_once_start_one_daemon(tmp_path, monkeypatch):
    """The window's waker, an ACP host and `symb watch` all end in
    start_daemon; the flock there is what makes it one model."""
    from symbio import constants
    from symbio.app import daemon

    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(constants, "DAEMON_PID_FILE", tmp_path / "daemon.pid")
    monkeypatch.setattr(constants, "DAEMON_SOCKET", tmp_path / "daemon.sock")
    spawned = []

    class _Proc:
        def __init__(self, *_a, **_k):
            time.sleep(0.2)                 # the gap the lock has to cover
            spawned.append(self)
            self.pid = os.getpid()          # alive, so the second sees it running

    monkeypatch.setattr(daemon.subprocess, "Popen", _Proc)
    threads = [threading.Thread(target=daemon.start_daemon, args=({},)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(spawned) == 1
