"""Resident-model daemon: keep the headmaster loaded between `symb chat` runs.

The 30s weight load is the one cost the CLI still pays every session
(generation is already maxed — see the 14B headmaster notes). This module runs
a background process that loads the model once and serves `symb chat` clients
over a Unix socket, so every session after the first attaches to a warm model
instead of re-reading 7+ GB of weights.

The transport mirrors symbio/app/telegram.py: a ChatSession driven through
injected input_fn / output_fn / confirm_fn / stream_chunk_fn. Here the
"front-end" is a JSON-lines protocol over a Unix socket instead of Telegram.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
from typing import Any

from symbio import constants


def daemon_running() -> tuple[bool, int | None]:
    """Whether a daemon process is alive, per the pid file."""
    if not constants.DAEMON_PID_FILE.exists():
        return False, None
    try:
        pid = int(constants.DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True, pid
    except (ValueError, OSError, ProcessLookupError):
        try:
            constants.DAEMON_PID_FILE.unlink()
        except OSError:
            pass
        return False, None


def daemon_ready() -> bool:
    """Whether the daemon has finished loading and is accepting connections.

    The socket is created only after the model is loaded, so its presence is
    the readiness signal — a pid file alone means the process is still loading.

    Presence alone was the whole test, and a Unix socket file outlives the
    process that bound it: an OOM kill (which this project takes regularly)
    left the node behind, daemon_ready() kept saying yes, and every later
    `symb chat` died on ECONNREFUSED with no fallback. So the pid has to be
    alive too — daemon_running() already removes a dead pid file — and a
    socket with nothing behind it is cleared here rather than left as a trap.
    """
    if not constants.DAEMON_SOCKET.exists():
        return False
    running, _pid = daemon_running()
    if running:
        return True
    try:
        constants.DAEMON_SOCKET.unlink()
    except OSError:
        pass
    return False


def _cleanup_daemon_files() -> None:
    for path in (constants.DAEMON_PID_FILE, constants.DAEMON_SOCKET):
        try:
            path.unlink()
        except OSError:
            pass


def _encode_msg(msg: dict) -> bytes:
    """One JSON object per line — the wire format for both directions."""
    return (json.dumps(msg) + "\n").encode("utf-8")


def _decode_msg(line: bytes) -> dict:
    return json.loads(line.decode("utf-8"))


def start_daemon(config: dict[str, Any]) -> int:
    running, pid = daemon_running()
    if running and pid is not None:
        print(f"Daemon already running (PID {pid}).")
        return 0
    # A stale socket from a crashed run would make `symb chat` attach to a dead
    # listener; drop it before the new daemon binds.
    _cleanup_daemon_files()
    log_path = constants.LOG_DIR / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "symbio.app.cli", "daemon", "run"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    constants.DAEMON_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"Daemon starting (PID {proc.pid}). The model loads in the background; "
          f"`symb chat` attaches once the socket is ready.")
    return 0


def stop_daemon() -> int:
    running, pid = daemon_running()
    if not running or pid is None:
        print("Daemon is not running.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent stop signal to daemon (PID {pid}).")
    except PermissionError:
        print(f"Permission denied: cannot stop process {pid}.")
        return 1
    except ProcessLookupError:
        print(f"Process {pid} already exited.")
    finally:
        _cleanup_daemon_files()
    return 0


def _load_model(config: dict[str, Any]):
    """Load the model once, mirroring ChatSession._ensure_model_loaded's
    adapter-compatibility check. Returns ((model, tokenizer), adapter_loaded)."""
    from symbio.app.modelload import load
    from symbio.config import adapter_weights_present, _adapter_matches_model

    if adapter_weights_present() and not _adapter_matches_model(config):
        print(" [Warning] Existing adapter was trained for a different model. "
              "Loading base model only.", flush=True)
        return load(config["model_name"]), False
    if adapter_weights_present():
        try:
            return load(config["model_name"],
                        adapter_path=str(constants.ADAPTER_DIR)), True
        except Exception as e:
            print(f" Could not load adapter: {e}", flush=True)
            print(" Falling back to base model...", flush=True)
            return load(config["model_name"]), False
    return load(config["model_name"]), False


def _serve_connection(conn: socket.socket, config: dict[str, Any],
                      model: Any, tokenizer: Any, adapter_loaded: bool) -> None:
    """Run one ChatSession over a connected socket, then close it."""
    from symbio.app.chat import ChatSession

    rfile = conn.makefile("rb")
    wfile = conn.makefile("wb")
    # Background threads (cron, note-indexer) call output_fn too, so writes to
    # the socket must be serialized or two JSON lines can interleave.
    write_lock = threading.Lock()

    def send_msg(msg: dict) -> None:
        with write_lock:
            wfile.write(_encode_msg(msg))
            wfile.flush()

    def recv_msg() -> dict:
        line = rfile.readline()
        if not line:
            raise EOFError("client disconnected")
        return _decode_msg(line)

    def input_fn(prompt: str = "") -> str:
        send_msg({"type": "input_prompt", "prompt": prompt})
        while True:
            msg = recv_msg()
            if msg.get("type") == "input":
                return msg.get("text", "")

    def output_fn(text: str = "") -> None:
        send_msg({"type": "output", "text": text})

    def confirm_fn(prompt: str) -> bool:
        send_msg({"type": "confirm", "prompt": prompt})
        while True:
            msg = recv_msg()
            if msg.get("type") == "confirm":
                return bool(msg.get("answer", False))

    def stream_chunk_fn(text: str) -> None:
        send_msg({"type": "stream", "text": text})

    session = ChatSession(
        config,
        model=model, tokenizer=tokenizer, adapter_loaded=adapter_loaded,
        input_fn=input_fn, output_fn=output_fn, confirm_fn=confirm_fn,
        stream_chunk_fn=stream_chunk_fn,
        stream_prefix=True,
        owner="daemon",
    )
    try:
        session.run()
    finally:
        try:
            send_msg({"type": "done"})
        except Exception:
            pass
        conn.close()


def daemon_main(config: dict[str, Any]) -> int:
    """The `symb daemon run` entry point: load once, serve connections."""
    from symbio.app import setup

    _identity_filled = setup.ensure_identity_defaults(config)
    if _identity_filled:
        try:
            from symbio.app.config import save_config
            save_config(config)
        except Exception:
            pass

    print("Loading model...", flush=True)
    (model, tokenizer), adapter_loaded = _load_model(config)
    print("Model loaded. Listening for clients.", flush=True)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Owner-only, and the umask is what makes it owner-only from the instant
    # it exists — a chmod after bind leaves a window in which the socket is
    # already listening and still world-connectable.
    #
    # This is not a hardening nicety. Whoever connects gets a real ChatSession
    # with the whole tool registry, and _serve_connection wires confirm_fn
    # back to that same client — so safety.maybe_confirm asks the caller
    # whether the caller's own run_command is approved, and the caller
    # answers. Bound at the default umask the mode was 0o755 (verified), which
    # makes that local code execution as this user for anything that can
    # traverse to the project directory. On a Unix socket the file mode IS the
    # peer check: 0600 means only this uid can connect at all.
    previous_umask = os.umask(0o177)
    try:
        sock.bind(str(constants.DAEMON_SOCKET))
    finally:
        os.umask(previous_umask)
    try:
        os.chmod(constants.DAEMON_SOCKET, 0o600)
    except OSError:
        # A platform that will not chmod a socket node still had the umask
        # applied at bind; nothing here should take the daemon down.
        pass
    sock.listen(1)

    try:
        while True:
            conn, _ = sock.accept()
            try:
                _serve_connection(conn, config, model, tokenizer, adapter_loaded)
            except Exception as e:
                print(f"Connection error: {e}", flush=True)
                try:
                    conn.close()
                except OSError:
                    pass
    finally:
        sock.close()
        _cleanup_daemon_files()
    return 0


class DaemonClient:
    """CLI-side client: connect to the daemon and pump messages to the TTY."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run(self) -> int | None:
        """Exit code, or None when there is no daemon to talk to.

        None rather than 1 so the caller can fall through to a local session:
        a daemon that cannot be reached is a missing optimisation, not a
        reason the assistant will not start.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(constants.DAEMON_SOCKET))
        except OSError as e:
            # Not a fatal error: the local path is right there. Returning 1
            # here meant a wedged or half-dead daemon took `symb chat` down
            # with it until someone deleted the socket by hand.
            print(f"Could not connect to the daemon ({e}); starting a local "
                  f"session instead.")
            try:
                constants.DAEMON_SOCKET.unlink()
            except OSError:
                pass
            return None

        rfile = sock.makefile("rb")
        wfile = sock.makefile("wb")

        def send_msg(msg: dict) -> None:
            wfile.write(_encode_msg(msg))
            wfile.flush()

        def recv_msg() -> dict:
            line = rfile.readline()
            if not line:
                raise EOFError("daemon disconnected")
            return _decode_msg(line)

        try:
            while True:
                try:
                    msg = recv_msg()
                except EOFError:
                    break
                mtype = msg.get("type")
                if mtype == "output":
                    print(msg.get("text", ""))
                elif mtype == "stream":
                    print(msg.get("text", ""), end="", flush=True)
                elif mtype == "input_prompt":
                    prompt = msg.get("prompt", "")
                    try:
                        text = input(prompt)
                    except (EOFError, KeyboardInterrupt):
                        text = "/quit"
                    send_msg({"type": "input", "text": text})
                elif mtype == "confirm":
                    prompt = msg.get("prompt", "")
                    send_msg({"type": "confirm",
                              "answer": self._read_yes_no(prompt)})
                elif mtype == "done":
                    break
        finally:
            sock.close()
        return 0

    @staticmethod
    def _read_yes_no(prompt: str) -> bool:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes", "true", "1", "on")
