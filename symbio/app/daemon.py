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
from symbio.app import chat_style, chat_ui


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
    """Start the resident model unless one is running or loading.

    Check-then-start runs under an flock on PROJECT_DIR/daemon.start.lock,
    held until the new pid file is written. Every starter comes through here —
    the desktop window's waker, the ACP and MCP bridges, `symb watch`
    restarting a crashed model, a terminal — and two that both saw "not
    running" would otherwise each load a copy: two 14Bs on 16 GB, the
    out-of-memory kill this project keeps having.
    """
    import fcntl

    constants.PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    with open(constants.PROJECT_DIR / "daemon.start.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _start_daemon_locked(config)


def _start_daemon_locked(config: dict[str, Any]) -> int:
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


class _ClientGone(Exception):
    """Raised inside a turn whose client has disconnected, to end it early.

    Dropping the writes alone was not enough. Generation keeps running to its
    natural end -- hundreds of tokens nobody will read -- and the daemon
    serves one session at a time, so the NEXT window sits in the backlog
    behind a turn that is being produced for nobody. Live 2026-09-16: a
    front-end was killed mid-generation and the replacement window waited
    57 seconds, correctly reporting that something else was holding the model.
    That something was a ghost.
    """


def _serve_connection(conn: socket.socket, config: dict[str, Any],
                      model: Any, tokenizer: Any, adapter_loaded: bool) -> None:
    """Run one ChatSession over a connected socket, then close it."""
    from symbio.app.chat import ChatSession

    rfile = conn.makefile("rb")
    wfile = conn.makefile("wb")
    # Background threads (cron, note-indexer) call output_fn too, so writes to
    # the socket must be serialized or two JSON lines can interleave.
    write_lock = threading.Lock()

    # Set when a write fails: the client has gone (a closed browser tab, a
    # front-end killed mid-turn). Every later write is dropped rather than
    # raised, because a half-finished turn must not take the resident model
    # down with it -- reloading the weights is 30s and this process is the
    # only reason the desktop and `symb chat` start instantly.
    client_gone = threading.Event()

    def send_msg(msg: dict) -> None:
        if client_gone.is_set():
            return
        try:
            with write_lock:
                wfile.write(_encode_msg(msg))
                wfile.flush()
        except OSError:
            client_gone.set()

    def recv_msg() -> dict:
        if client_gone.is_set():
            raise EOFError("client disconnected")
        try:
            line = rfile.readline()
        except OSError as e:
            client_gone.set()
            raise EOFError("client disconnected") from e
        if not line:
            client_gone.set()
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

    def banner_fn(config: dict, adapter_loaded: bool, dataset_size: int) -> None:
        lines = []
        chat_ui.print_banner(config, adapter_loaded, dataset_size,
                             output_fn=lines.append, terminal=False)
        # Additive metadata: old clients still receive the same plain text.
        # The CLI renders the panel using ITS current width and colour policy.
        send_msg({"type": "output", "text": "\n".join(lines),
                  "presentation": chat_ui.banner_data(config, adapter_loaded, dataset_size)})

    def confirm_fn(prompt: str) -> bool:
        # Nobody to ask is a NO, not a wait. Blocking here on a dead client
        # holds the only model on the machine until the process is killed.
        if client_gone.is_set():
            return False
        send_msg({"type": "confirm", "prompt": prompt})
        while True:
            msg = recv_msg()
            if msg.get("type") == "confirm":
                return bool(msg.get("answer", False))

    def status_fn(text) -> None:
        # Its own frame type, not an output line: a spinner is drawn over
        # itself and output is appended. Collapsing the two is what made the
        # daemon's log a column of "thinking…" and the client's screen empty.
        if client_gone.is_set():
            return
        send_msg({"type": "status", "text": text})

    def stream_chunk_fn(text: str) -> None:
        # Checked BEFORE the write, and raised rather than swallowed: this is
        # called once per token from inside the generation loop, so it is the
        # one place a turn can be cut short when nobody is listening.
        if client_gone.is_set():
            raise _ClientGone()
        send_msg({"type": "stream", "text": text})

    chat_ui.set_status_sink(status_fn)
    session = ChatSession(
        config,
        model=model, tokenizer=tokenizer, adapter_loaded=adapter_loaded,
        input_fn=input_fn, output_fn=output_fn, confirm_fn=confirm_fn,
        stream_chunk_fn=stream_chunk_fn,
        stream_prefix=True,
        owner="daemon",
        banner_fn=banner_fn,
    )
    try:
        session.run()
    except (_ClientGone, EOFError):
        # An ordinary end: the window closed. Not an error, and not a reason
        # to keep the connection or the turn alive.
        pass
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
    from symbio.app.config import apply_gpu_limits, keep_model_resident

    apply_gpu_limits(config)
    wired = keep_model_resident(model, config)
    # The Neural Engine side (decision model, OCR) starts now, off the GPU and
    # in the background, so the first turn does not pay for it.
    try:
        from symbio.app import ane, decider

        if ane.enabled(config):
            threading.Thread(target=decider.warm, daemon=True).start()
    except Exception:
        pass
    if wired:
        print(f"Keeping {wired / 2**30:.1f} GB wired: the model stays in RAM "
              f"between turns.", flush=True)
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
    # Backlog, not concurrency: one model means one session at a time, and
    # _serve_connection below runs them serially. But a backlog of 1 makes the
    # SECOND waiting client an ECONNREFUSED rather than a wait, and a
    # front-end that reconnects on failure then turns one busy moment into a
    # refusal loop. Live 2026-09-16, the desktop reported "The resident model
    # is running but refused a connection: [Errno 61] Connection refused"
    # while the daemon was healthy and mid-turn. Queue them instead.
    sock.listen(16)

    try:
        while True:
            conn, _ = sock.accept()
            try:
                _serve_connection(conn, config, model, tokenizer, adapter_loaded)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                # BaseException, not Exception. A session that ends through
                # SystemExit -- or anything else a command handler raises --
                # would otherwise leave this loop, run the finally below and
                # take the loaded model with it, so one client hanging up
                # would cost the next one a 30s reload. Whatever happened to
                # that connection, the daemon keeps serving.
                print(f"Connection error: {type(e).__name__}: {e}", flush=True)
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

        chat_style.install_command_completion(self._command_names)
        try:
            while True:
                try:
                    msg = recv_msg()
                except EOFError:
                    break
                mtype = msg.get("type")
                if mtype == "status":
                    self._draw_status(msg.get("text"))
                    continue
                # Anything else lands on a fresh line: a half-drawn spinner
                # frame is still on this one.
                self._clear_status()
                if mtype == "output":
                    self._print_output(msg)
                elif mtype == "stream":
                    print(msg.get("text", ""), end="", flush=True)
                elif mtype == "input_prompt":
                    plain_prompt = msg.get("prompt", "")
                    if (plain_prompt == f"{self.config.get('user_name', 'You'):8}: "
                            and chat_style.colors_enabled()):
                        print(chat_style.prompt_context(self.config))
                    prompt = self._skin_prompt(plain_prompt)
                    try:
                        text = input(chat_style.readline_prompt(prompt))
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

    def _draw_status(self, text) -> None:
        """Draw one spinner frame over the last, or clear the line for None."""
        if not self._can_animate():
            return
        if not text:
            self._clear_status()
            return
        sys.stdout.write("\r\033[K" + chat_style.status_frame(text))
        sys.stdout.flush()
        self._status_drawn = True

    def _clear_status(self) -> None:
        if getattr(self, "_status_drawn", False):
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self._status_drawn = False

    def _can_animate(self) -> bool:
        try:
            return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        except Exception:
            return False

    @staticmethod
    def _command_names() -> list[str]:
        from symbio.app import commands
        from symbio.app.chat_constants import BUILTIN_COMMAND_NAMES

        return sorted(set(BUILTIN_COMMAND_NAMES) | {c.name for c in commands.load_commands()})

    def _print_output(self, msg: dict) -> None:
        data = msg.get("presentation")
        if isinstance(data, dict) and data.get("kind") == "welcome":
            # These fields describe the running session, which may have a
            # different model/identity from the client's on-disk config.
            self.config = {**self.config, **{key: data[key] for key in
                           ("assistant_name", "user_name", "model_name") if key in data}}
            if chat_style.colors_enabled():
                print("\n".join(chat_style.welcome_panel(
                    data, workspace=data.get("workspace"), detail=data.get("detail", ""))))
                print()
                return
        print(self._skin(msg.get("text", "")))

    def _skin(self, text: str) -> str:
        """Render a status line the way the local terminal would.

        The skin lives in chat_style and `chat_loop` wires it into its own
        `_cli_output` and nowhere else — correct, because the daemon's
        output_fn feeds a socket and the text on that socket is a protocol.
        But THIS process is the one with the terminal, and it was printing
        every line raw. With the daemon up — which is most of the time, the
        supervisor restarts it — the bullets, elbows and colours existed and
        were never once drawn. Styling here keeps the wire plain and the
        screen styled, which is what the split was for.

        The assistant's own reply is not a status line and is left alone, by
        the same prefix test `_cli_output` uses: it is prose, it can be many
        lines, and a line of it shaped like "[Note] ..." is not a tag.
        """
        if not isinstance(text, str):
            return text
        prefix = f"{self.config.get('assistant_name', 'Assistant'):8}: "
        if text.startswith(prefix):
            return text
        return chat_style.style_line(text)

    def _skin_prompt(self, prompt: str) -> str:
        """The line the person types on, styled to match the local session."""
        plain = f"{self.config.get('user_name', 'You'):8}: "
        if prompt == plain and chat_style.colors_enabled():
            return chat_style.user_prompt()
        return prompt

    @staticmethod
    def _read_yes_no(prompt: str) -> bool:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes", "true", "1", "on")
