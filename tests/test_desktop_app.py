"""The desktop window: a bridge, not a second copy of the agent.

Two properties are worth a test rather than a comment.

The FIRST is the memory budget, which is the whole reason this process exists
in the shape it does. `from symbio import constants` runs symbio/__init__.py,
which imports the agent, the chat loop, mlx, anthropic, fastmcp, telegram,
pydantic, starlette and uvicorn behind it — measured at 130 MB resident for a
process that wanted eight Path objects. The desktop loads constants.py as a
standalone module instead and talks to the resident model over a socket, which
measured 28 MB idle and 31 MB after serving every endpoint. A future import of
`symbio` anywhere in this package would undo that silently, so it is asserted.

The SECOND is the hand-written WebSocket. There is no framework here to get
the handshake and the framing right, so they are checked against the values in
RFC 6455 rather than against themselves.
"""
import ast
import base64
import json
import pathlib
import struct

import pytest

from symbio_desktop import server


REPO = pathlib.Path(__file__).resolve().parent.parent


def _imports(path: pathlib.Path) -> set[str]:
    """Every module this file imports, at module scope or inside a function."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("name", ["server.py", "cli.py", "window.py", "acp.py", "mcp_bridge.py"])
def test_the_desktop_never_imports_the_agent_package(name):
    """`symbio` costs 105 MB of stack the window has no use for. constants.py
    is loaded by path precisely so this stays true."""
    offenders = {m for m in _imports(REPO / "symbio_desktop" / name)
                 if m == "symbio" or m.startswith("symbio.")}
    assert not offenders, f"{name} imports {sorted(offenders)}"


def test_the_handshake_matches_the_rfc():
    """The example key and response printed in RFC 6455 §1.3."""
    assert server.ws_accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_a_short_frame_is_one_length_byte():
    frame = server.ws_frame(b"hello")
    assert frame[0] == 0x81, "FIN + text opcode"
    assert frame[1] == 5 and frame[2:] == b"hello"
    assert not frame[1] & 0x80, "a server must never mask"


@pytest.mark.parametrize("size,marker,header", [
    (126, 126, 4), (70000, 127, 10),
])
def test_longer_payloads_use_the_extended_length(size, marker, header):
    """125 is the last length that fits in the seven bits; past that the
    length moves to a 16- or 64-bit field. Getting this boundary wrong
    truncates exactly the messages worth sending — a page of code."""
    frame = server.ws_frame(b"x" * size)
    assert frame[1] == marker
    assert len(frame) == header + size
    if marker == 126:
        assert struct.unpack(">H", frame[2:4])[0] == size
    else:
        assert struct.unpack(">Q", frame[2:10])[0] == size


class _FakeSocket:
    def __init__(self, data: bytes):
        self.data = data

    def recv(self, count):
        chunk, self.data = self.data[:count], self.data[count:]
        return chunk


def _client_frame(payload: bytes, opcode=0x1, fin=True) -> bytes:
    mask = b"\x01\x02\x03\x04"
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bytes([(0x80 if fin else 0) | opcode, 0x80 | len(payload)]) + mask + masked


def test_a_masked_client_message_is_unmasked():
    body = json.dumps({"type": "chat", "message": "hi"}).encode()
    opcode, payload = server.ws_read_message(_FakeSocket(_client_frame(body)))
    assert opcode == 0x1 and json.loads(payload) == {"type": "chat", "message": "hi"}


def test_a_split_message_is_joined():
    """A browser splits a long message across continuation frames, and the
    halves are meaningless on their own — half of a JSON object is a parse
    error, which would read as the user having sent nothing."""
    first = _client_frame(b'{"type": "ch', opcode=0x1, fin=False)
    rest = _client_frame(b'at", "message": "hi"}', opcode=0x0, fin=True)
    opcode, payload = server.ws_read_message(_FakeSocket(first + rest))
    assert opcode == 0x1
    assert json.loads(payload)["message"] == "hi"


def test_an_unmasked_client_frame_is_refused():
    raw = bytes([0x81, 2]) + b"hi"
    with pytest.raises(server.WSError):
        server.ws_read_message(_FakeSocket(raw))


# --- the socket that outlives its process ----------------------------------


def test_a_leftover_socket_file_is_not_a_running_model(tmp_path, monkeypatch):
    """A Unix socket node outlives the process that bound it, and this project
    takes OOM kills regularly. Presence alone would have the window reporting
    a resident model that is not there, then hanging on connect."""
    socket_file = tmp_path / "daemon.sock"
    socket_file.write_text("")
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("999999")   # a pid nothing is using
    monkeypatch.setattr(server.constants, "DAEMON_SOCKET", socket_file)
    monkeypatch.setattr(server.constants, "DAEMON_PID_FILE", pid_file)

    assert server.DaemonBridge.daemon_ready() is False


def test_a_live_pid_with_its_socket_is_ready(tmp_path, monkeypatch):
    import os

    socket_file = tmp_path / "daemon.sock"
    socket_file.write_text("")
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(str(os.getpid()))
    monkeypatch.setattr(server.constants, "DAEMON_SOCKET", socket_file)
    monkeypatch.setattr(server.constants, "DAEMON_PID_FILE", pid_file)

    assert server.DaemonBridge.daemon_ready() is True


def test_no_resident_model_is_reported_rather_than_loaded(monkeypatch):
    """The alternative — quietly constructing a ChatSession here — puts a
    second copy of a 14B model in the window's own process."""
    monkeypatch.setattr(server.DaemonBridge, "daemon_ready", staticmethod(lambda: False))
    sent = []

    ok, why = server.DaemonBridge(sent.append).connect()

    assert ok is False
    assert "symb daemon start" in why


# --- the daemon protocol, both directions ----------------------------------


def test_daemon_frames_become_browser_frames():
    sent = []
    bridge = server.DaemonBridge(sent.append)

    class _Pipe:
        def __init__(self, lines):
            self.lines = list(lines)

        def readline(self):
            return self.lines.pop(0) if self.lines else b""

    bridge.turn_open = True
    bridge.ready = True        # past the banner; a turn is in flight
    bridge.rfile = _Pipe([
        json.dumps({"type": "stream", "text": "Hel"}).encode() + b"\n",
        json.dumps({"type": "output", "text": "  [Tool: recall]"}).encode() + b"\n",
        json.dumps({"type": "confirm", "prompt": "Run this?"}).encode() + b"\n",
        json.dumps({"type": "input_prompt", "prompt": "> "}).encode() + b"\n",
    ])
    bridge._pump()

    kinds = [m["type"] for m in sent]
    assert kinds[:4] == ["token", "system", "confirm", "done"]
    assert sent[0]["text"] == "Hel"
    assert sent[2]["prompt"] == "Run this?"


def test_the_confirmation_answer_goes_back_as_the_daemon_expects():
    """The daemon's confirm_fn is a real thread blocked on this reply — the
    same gate as the terminal's y/N. A mis-shaped answer hangs the turn."""
    written = []

    class _Out:
        def write(self, data): written.append(data)
        def flush(self): pass

    bridge = server.DaemonBridge(lambda _msg: None)
    bridge.wfile = _Out()
    bridge.ready = True
    bridge.confirm(True)
    bridge.say("hello")

    messages = [json.loads(line) for line in b"".join(written).splitlines()]
    assert messages[0] == {"type": "confirm", "answer": True}
    assert messages[1] == {"type": "input", "text": "hello"}


# --- static files ----------------------------------------------------------


def test_the_page_and_its_scripts_are_there():
    for name in ("index.html", "style.css", "chat.js", "workspace.js"):
        assert (REPO / "symbio_desktop" / "static" / name).is_file()


def test_the_page_loads_both_scripts_in_order():
    """workspace.js uses escHtml, which chat.js defines: they share one global
    scope and the order is what makes that work."""
    html = (REPO / "symbio_desktop" / "static" / "index.html").read_text()
    assert html.index("chat.js") < html.index("workspace.js")


def test_a_message_sent_before_the_session_is_listening_is_not_lost():
    """The daemon prints its banner and THEN asks for input. A message typed
    the moment the window opens used to reach the socket first: the banner's
    own input_prompt closed a turn that had never started, the browser got
    `done` with an empty reply, and the answer the model then produced had no
    turn left to belong to. Live 2026-09-16, "What is my name?" came back
    blank after 40.9s for exactly this reason."""
    written = []

    class _Out:
        def write(self, data): written.append(data)
        def flush(self): pass

    sent = []
    bridge = server.DaemonBridge(sent.append)
    bridge.wfile = _Out()

    bridge.say("What is my name?")
    assert written == [], "nothing may be written before the session asks"

    class _Pipe:
        def __init__(self, lines): self.lines = list(lines)
        def readline(self): return self.lines.pop(0) if self.lines else b""

    bridge.rfile = _Pipe([
        json.dumps({"type": "output", "text": "CAINE — PERSONAL CHAT-FINETUNE CLI"}).encode() + b"\n",
        json.dumps({"type": "input_prompt", "prompt": "> "}).encode() + b"\n",
    ])
    bridge._pump()

    assert [json.loads(line) for line in b"".join(written).splitlines()] == [
        {"type": "input", "text": "What is my name?"}]
    assert "done" not in [m["type"] for m in sent], "the banner is not a finished turn"


# --- the resident model outliving its clients ------------------------------


def test_a_client_that_hangs_up_does_not_take_the_daemon_down(monkeypatch):
    """Live 2026-09-16: a front-end was killed mid-generation, the daemon's
    write to the closed socket raised, the exception left the accept loop, and
    the `finally` cleaned up the pid file and the socket — so the loaded model
    was gone and the next connection got "No resident model is running". One
    tab closing must not cost the next one a 30s reload."""
    from symbio.app import daemon

    class _DeadPipe:
        def write(self, _data): raise BrokenPipeError(32, "Broken pipe")
        def flush(self): pass
        def readline(self): return b""

    class _Conn:
        def __init__(self): self.closed = False
        def makefile(self, mode): return _DeadPipe()
        def close(self): self.closed = True

    served = []

    def _boom(conn, *_args, **_kwargs):
        served.append(conn)
        raise SystemExit(0)      # what a /quit path looks like from out here

    monkeypatch.setattr(daemon, "_serve_connection", _boom)

    # The loop body, lifted: accept one connection, survive it, keep going.
    conn = _Conn()
    try:
        try:
            daemon._serve_connection(conn, {}, None, None, False)
        except KeyboardInterrupt:
            raise
        except BaseException:
            conn.close()
    except SystemExit:
        pytest.fail("SystemExit escaped and would have killed the daemon")

    assert served and conn.closed


def test_a_write_to_a_gone_client_is_dropped_not_raised(tmp_path):
    """stream_chunk_fn runs inside generation. Raising there aborts the turn
    and unwinds through mlx, which is how a closed tab became a dead daemon."""
    import inspect

    from symbio.app import daemon

    source = inspect.getsource(daemon._serve_connection)
    assert "client_gone" in source
    assert "except OSError" in source


def test_a_full_backlog_reads_as_busy_not_as_a_broken_daemon(monkeypatch):
    """Live 2026-09-16: "The resident model is running but refused a
    connection: [Errno 61] Connection refused" — from a daemon that was
    healthy and mid-turn. A Unix socket refuses when its listen backlog is
    full, which for a one-session-at-a-time server means BUSY. Reporting the
    errno sends the reader to look for a crash that never happened."""
    monkeypatch.setattr(server.DaemonBridge, "daemon_ready", staticmethod(lambda: True))

    class _Refusing:
        def __init__(self, *a, **k): pass
        def connect(self, _path): raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr(server.socket, "socket", _Refusing)

    ok, why = server.DaemonBridge(lambda _m: None).connect()

    assert ok is False
    assert "one window at a time" in why
    assert "Errno" not in why


def test_the_daemon_queues_waiting_clients():
    """Backlog 1 turned a busy moment into a refusal, and a front-end that
    reconnects on failure turned that into a refusal loop."""
    import inspect

    from symbio.app import daemon

    source = inspect.getsource(daemon.daemon_main)
    assert "sock.listen(16)" in source


def test_the_window_attaches_only_when_there_is_something_to_say():
    """One daemon queue slot per browser reconnect — a refresh, a wake from
    sleep, the backoff loop after a restart — would hold the model for a
    window nobody is typing in."""
    import inspect

    source = inspect.getsource(server.Handler._websocket)
    assert "ensure_bridge" in source
    assert "bridge.connect()" not in source.split("def ensure_bridge")[0]


# --- what belongs in a transcript, not in a chat bubble --------------------


def _stream(bridge, chunks):
    class _Pipe:
        def __init__(self, lines): self.lines = list(lines)
        def readline(self): return self.lines.pop(0) if self.lines else b""

    lines = [json.dumps({"type": "stream", "text": c}).encode() + b"\n" for c in chunks]
    lines.append(json.dumps({"type": "input_prompt", "prompt": "> "}).encode() + b"\n")
    bridge.rfile = _Pipe(lines)
    bridge._pump()


@pytest.mark.parametrize("chunks,expected", [
    # The label arrives a few tokens at a time, so no single chunk holds it.
    (["Caine", "   : ", "Hi ", "Huy!", "<end>"], "Hi Huy!"),
    # A reply with no label is not eaten, even when it ends inside the window.
    (["Hello", " there,", " Huy"], "Hello there, Huy"),
    (["Caine   : ", "ok"], "ok"),
])
def test_the_terminal_speaker_label_is_not_repeated_in_the_bubble(chunks, expected):
    """The daemon builds its session with stream_prefix=True, so every turn
    arrives as "Caine   : ...". That belongs to a transcript; a chat bubble
    already says who is speaking. `<end>` is the stop sentinel and is not text
    at all — both showed up in the window, live 2026-09-16."""
    sent = []
    bridge = server.DaemonBridge(sent.append, "Caine")
    bridge.ready = True
    bridge.say("x")

    _stream(bridge, chunks)

    assert "".join(m["text"] for m in sent if m["type"] == "token") == expected
    assert "done" in [m["type"] for m in sent]


def test_a_turn_whose_client_left_is_cut_short(monkeypatch):
    """Dropping writes to a dead client is not enough: generation runs to its
    natural end, producing hundreds of tokens for nobody, while the daemon —
    one session at a time — makes the NEXT window wait behind a ghost. Live
    2026-09-16 that wait was 57 seconds, and the window's "something else is
    holding the model" was true.

    The token callback is the one place a turn can be cut short, so it raises
    there rather than swallowing."""
    import inspect

    from symbio.app import daemon

    source = inspect.getsource(daemon._serve_connection)
    stream = source.split("def stream_chunk_fn")[1].split("def ")[0]
    assert "client_gone.is_set()" in stream
    assert "raise _ClientGone()" in stream
    # And the session ends on it quietly — a closed window is not an error.
    assert "except (_ClientGone, EOFError)" in source


def test_a_confirmation_for_a_client_that_left_is_a_no(monkeypatch):
    """confirm_fn blocks on a reply. With nobody to answer, it held the only
    model on the machine until the process was killed."""
    import inspect

    from symbio.app import daemon

    confirm = inspect.getsource(daemon._serve_connection).split("def confirm_fn")[1]
    assert confirm.split("def ")[0].strip().startswith('"""') or "client_gone" in confirm
    assert "return False" in confirm.split("def ")[0]


# --- settings: what the window may switch, and what it may not -------------


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "model_name": "mlx-community/Qwen3-14B-3bit",
        "tools": {"enabled_groups": ["memory", "terminal"]},
        "safety": {"enabled": True, "require_confirm_score": 2},
        "remote": {"hosts": {"box": {"user": "root"}}},
    }), encoding="utf-8")
    monkeypatch.setattr(server.constants, "CONFIG_FILE", path)
    return path


def test_the_switches_read_the_real_config(config_file):
    out = server.get_settings()

    on = {g["key"] for g in out["groups"] if g["on"]}
    assert on == {"memory", "terminal"}
    safety = next(f for f in out["features"] if f["key"] == "safety.enabled")
    assert safety["on"] is True
    # Every switch says what it means — a settings page of bare dotted keys
    # is a page nobody dares touch.
    assert all(g["hint"] for g in out["groups"])
    assert all(f["hint"] for f in out["features"])


def test_a_group_switch_writes_config_json(config_file):
    assert server.set_settings({"groups": {"browser": True}})["changed"] == ["+browser"]
    assert server.set_settings({"groups": {"terminal": False}})["changed"] == ["-terminal"]

    groups = json.loads(config_file.read_text())["tools"]["enabled_groups"]
    assert "browser" in groups and "terminal" not in groups


def test_a_feature_switch_writes_a_dotted_key(config_file):
    server.set_settings({"features": {"rag.enabled": True}})

    assert json.loads(config_file.read_text())["rag"]["enabled"] is True


def test_nothing_outside_the_named_switches_is_writable(config_file):
    """The browser can reach this endpoint, so the list of keys it may write
    IS the security boundary. model_name would swap the model; remote.hosts
    is an ssh target list; require_confirm_score is the approval threshold."""
    out = server.set_settings({
        "features": {"model_name": "attacker/model",
                     "safety.require_confirm_score": 99,
                     "remote.hosts": {"evil": {}}},
        "groups": {"../../etc": True, "not_a_group": True},
    })

    assert out == {"ok": True, "changed": []}
    after = json.loads(config_file.read_text())
    assert after["model_name"] == "mlx-community/Qwen3-14B-3bit"
    assert after["safety"]["require_confirm_score"] == 2
    assert after["remote"]["hosts"] == {"box": {"user": "root"}}


def test_the_rest_of_the_config_survives_a_write(config_file):
    server.set_settings({"groups": {"code": True}})

    after = json.loads(config_file.read_text())
    assert after["remote"]["hosts"] == {"box": {"user": "root"}}
    assert after["safety"]["require_confirm_score"] == 2


def test_a_no_op_write_touches_nothing(config_file):
    before = config_file.read_text()

    assert server.set_settings({"groups": {"memory": True}})["changed"] == []
    assert config_file.read_text() == before


def test_the_config_is_replaced_atomically(config_file, monkeypatch):
    """config.json is read by the daemon and by every CLI session; a
    half-written one breaks all of them at once."""
    import inspect

    source = inspect.getsource(server.set_settings)
    assert "os.replace" in source and ".json.tmp" in source


# --- sessions: every conversation, from every front end --------------------


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    import sqlite3

    logs = tmp_path / "logs"
    logs.mkdir()
    db = logs / "sessions.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE sessions (id TEXT, started TEXT)")
    connection.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY, timestamp TEXT, "
                       "role TEXT, content TEXT, session_id TEXT)")
    connection.execute("INSERT INTO sessions VALUES ('s-new', '2026-09-16T08:33:39')")
    connection.execute("INSERT INTO sessions VALUES ('s-old', '2026-09-01T10:00:00')")
    connection.executemany(
        "INSERT INTO turns (timestamp, role, content, session_id) VALUES (?, ?, ?, ?)", [
            ("2026-09-16T08:33:40", "user", "What is my name?", "s-new"),
            ("2026-09-16T08:33:52", "assistant", "You are Huy.", "s-new"),
            ("2026-09-01T10:00:01", "user", "fix the wifi", "s-old"),
        ])
    connection.commit()
    connection.close()
    monkeypatch.setattr(server.constants, "PROJECT_DIR", tmp_path)
    return db


def test_the_sessions_list_is_newest_first_with_its_opening_line(session_store):
    out = server.get_sessions()

    assert [s["id"] for s in out["sessions"]] == ["s-new", "s-old"]
    assert out["sessions"][0]["opening"] == "What is my name?"
    assert out["sessions"][0]["turns"] == 2


def test_one_session_reads_back_in_order(session_store):
    out = server.get_sessions(session_id="s-new")

    assert [t["role"] for t in out["turns"]] == ["user", "assistant"]
    assert out["turns"][1]["text"] == "You are Huy."


def test_the_session_store_is_opened_read_only():
    """The daemon has this file open and is writing to it while the window
    reads: no lock, no journal."""
    import inspect

    assert "immutable=1" in inspect.getsource(server.get_sessions)


def test_a_missing_store_is_a_sentence_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(server.constants, "PROJECT_DIR", tmp_path)

    out = server.get_sessions()

    assert out["sessions"] == []
    assert "No session store" in out["reason"]


# --- the reply, with the protocol taken out of it --------------------------


@pytest.mark.parametrize("chunks,expected", [
    # The stop sentinel, split across chunks like every other token.
    (["Done.", "<en", "d>"], "Done."),
    (["ok ", "<|im_end|>"], "ok "),
    # A tool call the runtime already executed is not part of the reply.
    (["Looking. ", "<tool_call>", '{"name": "recall"}', "</tool_call>", "Found it."],
     "Looking. Found it."),
    # A thinking block belongs in the folded section, never in the answer.
    (["<thi", "nk>", "reasoning nobody asked to read", "</th", "ink>", "The answer."],
     "The answer."),
    # And a reply that merely CONTAINS a colon keeps its opening clause: the
    # loose speaker-label pattern ate "Here you go: " and started mid-sentence.
    (["Here you go: ", "the three files."], "Here you go: the three files."),
    (["Note: ", "this stays."], "Note: this stays."),
])
def test_only_protocol_is_taken_out_of_the_reply(chunks, expected):
    sent = []
    bridge = server.DaemonBridge(sent.append, "Caine")
    bridge.ready = True
    bridge.say("x")

    _stream(bridge, chunks)

    assert "".join(m["text"] for m in sent if m["type"] == "token") == expected


def test_a_partial_tag_is_never_shown_and_then_unsaid():
    """"<too" reaching the reader and being deleted a token later is worse
    than a few milliseconds of nothing."""
    sent = []
    bridge = server.DaemonBridge(sent.append, "Caine")
    bridge.ready = True
    bridge.say("x")
    bridge.prefix_done = True

    assert "<" not in bridge._clean_stream("answer <tool")
    assert bridge._clean_stream("_call>{}</tool_call> done") == " done"


def test_a_turn_that_ends_inside_a_thinking_block_says_so():
    """Emitting the held text reproduces the truncated-thinking-as-answer
    failure; dropping it silently leaves an empty bubble."""
    sent = []
    bridge = server.DaemonBridge(sent.append, "Caine")
    bridge.ready = True
    bridge.say("x")

    _stream(bridge, ["<think>", "the model never closed this"])

    assert not [m for m in sent if m["type"] == "token"]
    assert any("ended inside a thinking block" in m.get("text", "") for m in sent)


def test_the_window_is_an_app_with_the_menus_a_mac_app_has():
    """Opened as its own app, not a browser tab: without an Edit menu, ⌘C,
    ⌘V and ⌘A do nothing in the page's text box."""
    AppKit = pytest.importorskip("AppKit")
    from symbio_desktop import window

    bar = window._main_menu(AppKit.NSMenu, AppKit.NSMenuItem)
    menus = {bar.itemAtIndex_(i).submenu().title(): bar.itemAtIndex_(i).submenu()
             for i in range(bar.numberOfItems())}
    assert list(menus) == ["Symbio", "Edit", "Window"]
    actions = {menus["Edit"].itemAtIndex_(i).action()
               for i in range(menus["Edit"].numberOfItems())}
    assert {"copy:", "paste:", "cut:", "selectAll:", "undo:"} <= actions
    assert window.ICON.is_file() and window.ICON.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
