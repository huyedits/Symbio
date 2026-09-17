"""Tests for the resident-model daemon (symbio/app/daemon.py).

No real model is loaded: the pid/socket lifecycle and the JSON-lines wire
protocol are exercised directly, and the client message pump is driven against
a fake daemon over a real Unix socket.
"""
import os
import socket
import threading
from pathlib import Path

import pytest

from symbio import constants
from symbio.app import daemon


@pytest.fixture
def daemon_paths(tmp_path, monkeypatch):
    """Point the daemon's pid/socket at a scratch dir so tests never touch the
    real PROJECT_DIR files.

    The socket lives in a short /tmp path rather than tmp_path: macOS caps
    AF_UNIX paths at ~104 bytes and tmp_path is far longer than that.
    """
    pid = tmp_path / "daemon.pid"
    sock = Path(f"/tmp/symb_daemon_test_{os.getpid()}.sock")
    monkeypatch.setattr(constants, "DAEMON_PID_FILE", pid)
    monkeypatch.setattr(constants, "DAEMON_SOCKET", sock)
    yield pid, sock
    try:
        sock.unlink()
    except OSError:
        pass


def test_daemon_running_no_pid(daemon_paths):
    assert daemon.daemon_running() == (False, None)


def test_daemon_running_live_pid(daemon_paths):
    pid_file, _ = daemon_paths
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    running, pid = daemon.daemon_running()
    assert running is True
    assert pid == os.getpid()


def test_daemon_running_stale_pid(daemon_paths):
    pid_file, _ = daemon_paths
    # A pid no process can own is treated as stale and the file is cleaned up.
    pid_file.write_text("999999999", encoding="utf-8")
    assert daemon.daemon_running() == (False, None)
    assert not pid_file.exists()


def test_daemon_running_garbage_pid(daemon_paths):
    pid_file, _ = daemon_paths
    pid_file.write_text("not-a-pid", encoding="utf-8")
    assert daemon.daemon_running() == (False, None)
    assert not pid_file.exists()


def test_daemon_ready(daemon_paths):
    """Ready means the socket exists AND something is alive behind it.

    The socket alone used to be the whole test, and a Unix socket file
    outlives the process that bound it — so after an OOM kill (which this
    project takes regularly) the node stayed, daemon_ready() kept saying yes,
    and every later `symb chat` died on ECONNREFUSED with no fallback.
    """
    pid, sock = daemon_paths
    assert daemon.daemon_ready() is False

    sock.touch()
    pid.write_text(str(os.getpid()))
    assert daemon.daemon_ready() is True


def test_a_socket_left_by_a_dead_daemon_is_not_ready(daemon_paths):
    """And is cleared, rather than left as a trap for every later run."""
    pid, sock = daemon_paths
    sock.touch()
    pid.write_text("999999")            # nothing is alive at that pid

    assert daemon.daemon_ready() is False
    assert not sock.exists()


def test_cleanup_daemon_files(daemon_paths):
    pid_file, sock = daemon_paths
    pid_file.write_text("1", encoding="utf-8")
    sock.touch()
    daemon._cleanup_daemon_files()
    assert not pid_file.exists()
    assert not sock.exists()


def test_encode_decode_roundtrip():
    msg = {"type": "stream", "text": "hello\nworld"}
    assert daemon._decode_msg(daemon._encode_msg(msg)) == msg


def test_client_message_pump(daemon_paths, monkeypatch, capsys):
    """Drive DaemonClient against a fake daemon over a real Unix socket."""
    _, sock_path = daemon_paths

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(1)

    received = []

    def fake_session():
        conn, _ = srv.accept()
        wfile = conn.makefile("wb")
        rfile = conn.makefile("rb")
        wfile.write(daemon._encode_msg({"type": "output", "text": "banner"}))
        wfile.write(daemon._encode_msg({"type": "input_prompt", "prompt": "You: "}))
        wfile.flush()
        line = rfile.readline()
        received.append(daemon._decode_msg(line))
        wfile.write(daemon._encode_msg({"type": "stream", "text": "hi"}))
        wfile.write(daemon._encode_msg({"type": "done"}))
        wfile.flush()
        conn.close()

    t = threading.Thread(target=fake_session)
    t.start()

    # The client's input() is replaced so the pump doesn't block on the TTY.
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    client = daemon.DaemonClient({})
    assert client.run() == 0
    t.join(timeout=5)
    srv.close()

    assert received == [{"type": "input", "text": "hello"}]
    out = capsys.readouterr().out
    assert "banner" in out
    assert "hi" in out


# ---- what the person at the terminal actually sees ----
#
# chat_style skins status lines, and chat_loop wires it into its own
# _cli_output "and nowhere else" — correct, because the daemon's output_fn
# feeds a socket and that text is a protocol. But the client is the process
# with the terminal, and it printed every line raw. With the daemon up, which
# is nearly always, the bullets and colours existed and were never drawn.
#
# The spinner was worse: it wrote to sys.stdout, which in the daemon is a log
# file, so `isatty()` was False and it fell through to one static "thinking…"
# line written where nobody is looking. The client got silence between the
# prompt and the reply for as long as a 14B takes.

def _client(**config):
    config.setdefault("assistant_name", "Caine")
    config.setdefault("user_name", "Huy")
    return daemon.DaemonClient(config)


def test_a_status_line_is_skinned_for_the_terminal(monkeypatch):
    monkeypatch.setattr(daemon.chat_style, "colors_enabled", lambda *a, **k: True)

    out = _client()._skin("[Tool: browser_open]")

    assert "browser_open" in out and out != "[Tool: browser_open]"


def test_the_assistant_reply_is_left_exactly_as_it_came(monkeypatch):
    """It is prose, it can be many lines, and a line of it shaped like
    "[Note] ..." is not a tag. Same prefix test _cli_output uses."""
    monkeypatch.setattr(daemon.chat_style, "colors_enabled", lambda *a, **k: True)
    reply = "Caine   : [Note] is how I would start that file."

    assert _client()._skin(reply) == reply


def test_the_typing_line_matches_the_local_session(monkeypatch):
    monkeypatch.setattr(daemon.chat_style, "colors_enabled", lambda *a, **k: True)

    assert _client()._skin_prompt("Huy     : ") == daemon.chat_style.user_prompt()


def test_an_unstyled_terminal_gets_the_prompt_it_always_had(monkeypatch):
    monkeypatch.setattr(daemon.chat_style, "colors_enabled", lambda *a, **k: False)

    assert _client()._skin_prompt("Huy     : ") == "Huy     : "


def test_a_status_frame_is_drawn_over_the_last_one(monkeypatch, capsys):
    client = _client()
    monkeypatch.setattr(client, "_can_animate", lambda: True)

    client._draw_status("⠋ thinking…")
    client._draw_status("⠙ thinking… (6s)")
    client._clear_status()

    written = capsys.readouterr().out
    assert written.count("\r\033[K") == 3, "two frames and the clear"
    assert "thinking… (6s)" in written
    assert "\n" not in written, "a spinner that scrolls is not a spinner"


def test_nothing_is_animated_into_a_pipe(monkeypatch, capsys):
    client = _client()
    monkeypatch.setattr(client, "_can_animate", lambda: False)

    client._draw_status("⠋ thinking…")

    assert capsys.readouterr().out == ""


def test_the_line_is_cleared_before_anything_is_printed_over_it(monkeypatch, capsys):
    """A half-drawn frame is still on the current line when the reply starts."""
    client = _client()
    monkeypatch.setattr(client, "_can_animate", lambda: True)
    client._draw_status("⠋ thinking…")

    client._clear_status()

    assert capsys.readouterr().out.endswith("\r\033[K")


def test_the_spinner_animates_when_a_sink_is_set_without_a_tty(monkeypatch):
    """The daemon's own stdout is a log file. Before the sink, that alone
    turned the animation off."""
    from symbio.app import chat_ui

    frames = []
    monkeypatch.setattr(chat_ui.sys.stdout, "isatty", lambda: False, raising=False)
    chat_ui.set_status_sink(frames.append)
    try:
        spinner = chat_ui._Spinner("thinking…")
        assert spinner.active is True
        spinner.start()
        import time
        time.sleep(0.3)
        spinner.stop()
    finally:
        chat_ui.set_status_sink(None)

    assert any("thinking…" in f for f in frames if f), frames
    assert frames[-1] is None, "stop() has to clear the line it drew"


def test_without_a_sink_the_spinner_is_unchanged(monkeypatch):
    from symbio.app import chat_ui

    chat_ui.set_status_sink(None)
    spinner = chat_ui._Spinner("thinking…")

    assert spinner.active == daemon.sys.stdout.isatty()
