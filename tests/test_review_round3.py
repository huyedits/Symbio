"""Five findings from the max review, each pinned by the case that proved it.

Three were mine, shipped earlier in the same session and all of the same
shape: a guard that reads correct and does nothing.
"""
import os
import socket
import stat

import pytest

from symbio import config as symconfig, constants, safety, tools
from symbio.app import daemon, tooling


# ---- 1. the mid-thought retry could not fire in the shipped config ----

def test_truncation_is_detected_with_thinking_on():
    """thinking_level "low" is the default, so enable_thinking is True and the
    chat template puts the opening <think> in the PROMPT. A reply cut off at
    the reasoning budget therefore carries neither tag — 0 opens, 0 closes —
    and `opens > closes` reports it CLOSED. The branch never ran in the one
    configuration it was written for, and its test passed because the fixture
    hardcodes a literal <think>, the shape produced only with thinking OFF."""
    cut_off = "Maybe the composer is at the top. Or maybe I should check"

    assert tooling.think_block_closed(cut_off) is True      # the old signal
    assert tooling.count_think_closes(cut_off) == 0         # the real one


def test_a_finished_reply_with_thinking_on_closes_its_block():
    assert tooling.count_think_closes("Thought about it.</think>Posted.") == 1


@pytest.mark.parametrize("text, closes", [
    ("<think>cut off mid-", 0),
    ("<think>done</think>Answer.", 1),
])
def test_the_thinking_off_shapes_are_unchanged(text, closes):
    assert tooling.count_think_closes(text) == closes


# ---- 2. the risk gate must be able to ask, or say why it cannot ----

_CFG = {"safety": {"enabled": True, "require_confirm_score": 3}}


def _agent(**extra):
    return type("A", (), {"config": _CFG, "tools": [], **extra})()


def test_an_unattended_denial_says_nobody_was_there_to_ask():
    """Passing no confirm_fn left maybe_confirm on its stdin fallback, which
    returns False off a TTY — so this gate silently denied every run_command
    and execute_code in piped, cron and gateway runs that used to work. "Not
    approved" there reads as the user refusing; nobody was there to refuse."""
    out = tools.run_single_tool(_agent(), "desktop_type", {"text": "rm -rf ~"})

    assert "was not approved" in out
    assert "no terminal" in out
    assert "require_confirm_score" in out          # and how to change it


def test_a_front_end_that_can_ask_is_asked():
    agent = _agent(confirm_fn=staticmethod(lambda _p: True))

    out = tools.run_single_tool(agent, "desktop_type", {"text": "hello"})

    assert "was not approved" not in out


def test_a_front_end_that_asks_and_is_refused_still_denies():
    agent = _agent(confirm_fn=staticmethod(lambda _p: False))

    assert "was not approved" in tools.run_single_tool(
        agent, "desktop_type", {"text": "hello"})


# ---- 4. a corrupt adapter config is not a compatible one ----

@pytest.fixture
def adapter_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(symconfig, "ADAPTER_DIR", tmp_path)
    return tmp_path


def test_a_corrupt_adapter_config_is_assumed_incompatible(adapter_dir):
    """What an OOM-killed training run leaves behind. The file failed to
    parse, _saved_adapter_model returned None, and None reads as "nothing to
    contradict the model name" — so weights possibly trained for a different
    model were handed over as compatible. The fail-safe pointed the wrong
    way."""
    (adapter_dir / "adapter_config.json").write_text('{"model": "m/x", TRUNCA')

    assert symconfig._adapter_matches_model_name("m/x") is False


def test_no_adapter_at_all_is_still_fine(adapter_dir):
    """Base-only is a normal state and must not be confused with a broken
    one."""
    assert symconfig._adapter_matches_model_name("m/x") is True


def test_a_matching_adapter_still_matches(adapter_dir):
    (adapter_dir / "adapter_config.json").write_text('{"model": "m/x"}')

    assert symconfig._adapter_matches_model_name("m/x") is True


def test_an_adapter_for_another_model_does_not(adapter_dir):
    (adapter_dir / "adapter_config.json").write_text('{"model": "other/model"}')

    assert symconfig._adapter_matches_model_name("m/x") is False


# ---- 5. the daemon socket is the door ----

@pytest.fixture
def daemon_files(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "DAEMON_SOCKET", tmp_path / "daemon.sock")
    monkeypatch.setattr(constants, "DAEMON_PID_FILE", tmp_path / "daemon.pid")
    return tmp_path


def test_the_socket_is_owner_only(daemon_files):
    """Whoever connects gets a real ChatSession with the whole tool registry,
    and _serve_connection wires confirm_fn back to that same client — so
    maybe_confirm asks the caller whether the caller's own run_command is
    approved. Bound at the default umask the mode was 0o755 (verified), which
    makes that local code execution as this user. On a Unix socket the file
    mode IS the peer check."""
    # A short path on purpose: AF_UNIX caps the whole path around 104 bytes
    # on macOS, and pytest's tmp_path is already past it.
    import pathlib
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp(dir="/tmp")) / "d.sock"
    previous = os.umask(0o177)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
    finally:
        os.umask(previous)
    os.chmod(path, 0o600)
    mode = stat.S_IMODE(path.stat().st_mode)
    sock.close()

    assert not mode & stat.S_IRWXO          # no other
    assert not mode & stat.S_IRWXG          # no group


def test_a_socket_with_nothing_behind_it_is_not_ready(daemon_files):
    """A Unix socket file outlives the process that bound it. After an OOM
    kill the node stayed, daemon_ready() kept saying yes, and every later
    `symb chat` died on ECONNREFUSED."""
    constants.DAEMON_SOCKET.touch()
    constants.DAEMON_PID_FILE.write_text("999999")      # nothing at that pid

    assert daemon.daemon_ready() is False
    assert not constants.DAEMON_SOCKET.exists()         # and it is cleared


def test_a_live_daemon_is_still_ready(daemon_files):
    constants.DAEMON_SOCKET.touch()
    constants.DAEMON_PID_FILE.write_text(str(os.getpid()))

    assert daemon.daemon_ready() is True
    assert constants.DAEMON_SOCKET.exists()


def test_the_client_reports_no_daemon_rather_than_failure(daemon_files):
    """None, not 1, so the caller falls through to a local session: a daemon
    that cannot be reached is a missing optimisation, not a reason the
    assistant will not start."""
    constants.DAEMON_SOCKET.touch()                     # exists, nobody listening

    assert daemon.DaemonClient({"agent": {}}).run() is None


# ---- and the retry must not fire on replies that simply finished ----

def test_a_short_complete_reply_is_not_mistaken_for_a_truncated_one():
    """The first attempt at fixing this asked only whether a think block was
    closed. With thinking on, a complete tagless answer has zero closes too,
    so it fired on ordinary replies and spent a second full generation on
    each — 24 scripted turns regressed on exactly that. Running out of budget
    is the thing that actually happened, and only the generator knows it."""
    complete = "Spain won the 2026 World Cup."

    assert tooling.count_think_closes(complete) == 0        # looks truncated
    # ...which is why the decision needs hit_token_cap, not the text alone.


def test_the_generator_reports_whether_it_ran_out_of_budget():
    from symbio.app.chat import ChatSession

    session = ChatSession.__new__(ChatSession)
    session.config = {"agent": {"max_reply_tokens": 4, "prompt_cache_enabled": False}}
    session.tokenizer = _WordTokenizer()
    session.model = object()
    session.sampler = None
    session.stream_chunk_fn = None
    session.output_fn = lambda *_a: None
    session._indexing_now = False
    session._ensure_model_loaded = lambda: None
    session._await_prefill = lambda: None
    session._context_floor = None
    session._said_context_full = False
    session._kv_bytes_per_token = None
    session.generate_fn = lambda *a, **k: "one two three four"

    timings = {}
    session._generate_reply([{"role": "user", "content": "hi"}], timings=timings)

    assert "hit_token_cap" in timings


class _WordTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=True):
        return " ".join(m["content"] for m in messages)

    def encode(self, text):
        return list(range(len(text.split())))
