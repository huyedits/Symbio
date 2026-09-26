"""Symbio for other apps: the ACP agent, the MCP bridge, and `symb connect`.

Both bridges are tested against a stand-in for the daemon's side of
DaemonBridge, so what is checked is the protocol each speaks to its host: the
shapes of initialize and session/new, a prompt turn streamed as updates, an
approval prompt asked through the host, a cancel, and the MCP tool calls. The
real daemon behind them was exercised end to end with the official ACP and MCP
SDK clients; these keep the contract from drifting.
"""
import json
import threading
import time

import pytest

from symbio_desktop import acp, mcp_bridge


class FakeBridge:
    """The daemon's side of DaemonBridge: a scripted turn per say()."""

    script: list = []
    instances: list = []

    def __init__(self, sink, assistant_name=""):
        self.sink = sink
        self.ready = False
        self.said: list[str] = []
        self.confirmed: list[bool] = []
        self._answer = threading.Event()
        FakeBridge.instances.append(self)

    def connect(self):
        self.ready = True
        return True, ""

    def alive(self):
        return self.ready

    def close(self):
        self.ready = False

    def confirm(self, approved):
        self.confirmed.append(approved)
        self._answer.set()

    def say(self, text):
        self.said.append(text)
        threading.Thread(target=self._play, args=(list(self.script),), daemon=True).start()

    def _play(self, script):
        for kind, value in script:
            if kind == "sleep":
                time.sleep(value)
            elif kind == "confirm":
                self._answer.clear()
                self.sink({"type": "confirm", "prompt": value})
                self._answer.wait(5)
            else:
                self.sink({"type": kind, "text": value})
        self.sink({"type": "done", "text": ""})


class Out:
    """The host's end of stdout: every line the agent writes, parsed."""

    def __init__(self):
        self.messages: list[dict] = []

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                self.messages.append(json.loads(line))

    def flush(self):
        pass


@pytest.fixture
def agent(monkeypatch):
    FakeBridge.instances.clear()
    monkeypatch.setattr(acp, "DaemonBridge", FakeBridge)
    monkeypatch.setattr(acp, "ensure_daemon", lambda: (True, ""))
    monkeypatch.setattr(acp, "_config_summary", lambda: {"assistant_name": "Symbio"})
    out = Out()
    return acp.Agent(stdin=iter(()), stdout=out), out


def _answer_permissions(agent_, out, option_id, stop):
    """Play the host: answer every session/request_permission the agent sends."""
    answered = set()
    while not stop.is_set():
        for message in list(out.messages):
            if message.get("method") == "session/request_permission" and message["id"] not in answered:
                answered.add(message["id"])
                agent_.waiting[message["id"]].put(
                    {"id": message["id"],
                     "result": {"outcome": {"outcome": "selected", "optionId": option_id}}})
        time.sleep(0.01)


def _updates(out):
    return [m["params"]["update"] for m in out.messages if m.get("method") == "session/update"]


def test_initialize_says_what_the_agent_can_do(agent):
    agent_, _ = agent
    result = agent_.initialize({"protocolVersion": 1, "clientCapabilities": {}})
    assert result["protocolVersion"] == 1
    assert result["agentInfo"]["name"] == "symbio"
    assert result["authMethods"] == []
    assert result["agentCapabilities"]["promptCapabilities"]["embeddedContext"] is True


def test_a_prompt_turn_streams_and_asks_before_acting(agent):
    agent_, out = agent
    FakeBridge.script = [("token", "Hello"), ("system", "[Tool] listing files"),
                         ("confirm", "Run `ls ~`?"), ("token", " world")]
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    stop = threading.Event()
    threading.Thread(target=_answer_permissions, args=(agent_, out, "allow", stop),
                     daemon=True).start()
    try:
        result = agent_.prompt({"sessionId": session_id,
                                "prompt": [{"type": "text", "text": "hi"}]})
    finally:
        stop.set()
    assert result == {"stopReason": "end_turn"}
    updates = _updates(out)
    text = "".join(u["content"]["text"] for u in updates
                   if u["sessionUpdate"] == "agent_message_chunk")
    assert text == "Hello world"
    assert any(u["sessionUpdate"] == "agent_thought_chunk"
               and "[Tool] listing files" in u["content"]["text"] for u in updates)
    call = next(u for u in updates if u["sessionUpdate"] == "tool_call")
    assert call["kind"] == "execute" and call["status"] == "pending"
    done = next(u for u in updates if u["sessionUpdate"] == "tool_call_update")
    assert done["status"] == "completed"
    assert FakeBridge.instances[-1].confirmed == [True]
    permission = next(m for m in out.messages if m.get("method") == "session/request_permission")
    assert {o["kind"] for o in permission["params"]["options"]} == {"allow_once", "reject_once"}


def test_a_rejected_approval_is_passed_on_as_a_refusal(agent):
    agent_, out = agent
    FakeBridge.script = [("confirm", "Delete the file?"), ("token", "Okay, I won't.")]
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    stop = threading.Event()
    threading.Thread(target=_answer_permissions, args=(agent_, out, "reject", stop),
                     daemon=True).start()
    try:
        agent_.prompt({"sessionId": session_id, "prompt": [{"type": "text", "text": "x"}]})
    finally:
        stop.set()
    assert FakeBridge.instances[-1].confirmed == [False]
    assert next(u for u in _updates(out)
                if u["sessionUpdate"] == "tool_call_update")["status"] == "failed"


def test_cancel_ends_the_turn_and_the_next_prompt_still_works(agent):
    agent_, out = agent
    FakeBridge.script = [("token", "one"), ("sleep", 0.5), ("token", " two")]
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    results = []
    worker = threading.Thread(target=lambda: results.append(agent_.prompt(
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "count"}]})))
    worker.start()
    time.sleep(0.2)
    agent_.notice({"method": "session/cancel", "params": {"sessionId": session_id}})
    worker.join(3)
    assert results == [{"stopReason": "cancelled"}]
    FakeBridge.script = [("token", "still here")]
    before = len(_updates(out))
    assert agent_.prompt({"sessionId": session_id,
                          "prompt": [{"type": "text", "text": "again"}]}) == {"stopReason": "end_turn"}
    after = "".join(u["content"]["text"] for u in _updates(out)[before:]
                    if u["sessionUpdate"] == "agent_message_chunk")
    assert after == "still here", "the cancelled turn's tail must not leak into the next"


def test_an_unknown_method_is_a_json_rpc_error(agent):
    agent_, out = agent
    agent_.handle({"jsonrpc": "2.0", "id": 7, "method": "session/teleport", "params": {}})
    assert out.messages[-1]["error"]["code"] == -32601


def test_prompt_content_blocks_become_one_message():
    text = acp.prompt_text([
        {"type": "text", "text": "Look at this"},
        {"type": "resource", "resource": {"uri": "file:///a.py", "text": "print(1)"}},
        {"type": "resource_link", "uri": "file:///b.md", "name": "b.md"},
    ])
    assert text == "Look at this\n\n[file:///a.py]\nprint(1)\n\n[b.md: file:///b.md]"


# ---- the MCP bridge -----------------------------------------------------------

@pytest.fixture
def bridge(monkeypatch):
    FakeBridge.instances.clear()
    monkeypatch.setattr(acp, "DaemonBridge", FakeBridge)
    monkeypatch.setattr(acp, "_config_summary", lambda: {"assistant_name": "Symbio"})
    monkeypatch.setattr(mcp_bridge, "ensure_daemon", lambda: (True, ""))
    out = Out()
    return mcp_bridge.Bridge(stdin=iter(()), stdout=out), out


def _rpc(bridge_, out, method, params=None, id_=1):
    bridge_.handle({"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}})
    return out.messages[-1]


def test_mcp_handshake_and_tool_list(bridge):
    bridge_, out = bridge
    init = _rpc(bridge_, out, "initialize", {"protocolVersion": "2025-06-18"})["result"]
    assert init["protocolVersion"] == "2025-06-18"
    assert init["capabilities"]["tools"] == {"listChanged": False}
    names = [t["name"] for t in _rpc(bridge_, out, "tools/list")["result"]["tools"]]
    assert names == ["ask_symbio", "symbio_status"]


def test_ask_symbio_returns_the_reply_and_declines_approvals(bridge):
    bridge_, out = bridge
    FakeBridge.script = [("token", "Sure — "), ("confirm", "Run `rm -rf build`?"),
                         ("token", "done.")]
    result = _rpc(bridge_, out, "tools/call",
                  {"name": "ask_symbio", "arguments": {"message": "clean up"}})["result"]
    text = result["content"][0]["text"]
    assert result["isError"] is False
    assert text.startswith("Sure — done.")
    assert "declined" in text and "rm -rf build" in text
    assert FakeBridge.instances[-1].confirmed == [False]


def test_an_empty_message_is_an_error_not_a_turn(bridge):
    bridge_, out = bridge
    result = _rpc(bridge_, out, "tools/call",
                  {"name": "ask_symbio", "arguments": {"message": "  "}})["result"]
    assert result["isError"] is True and not FakeBridge.instances


# ---- symb connect -------------------------------------------------------------

def test_connect_adds_symbio_and_keeps_everything_else(tmp_path):
    from symbio.app import connect

    config = tmp_path / "claude_desktop_config.json"
    config.write_text(json.dumps({"preferences": {"theme": "dark"},
                                  "mcpServers": {"other": {"command": "x"}}}))
    assert connect.claude_desktop(config_path=config) == 0
    data = json.loads(config.read_text())
    assert data["preferences"] == {"theme": "dark"}
    assert data["mcpServers"]["other"] == {"command": "x"}
    entry = data["mcpServers"]["symbio"]
    assert entry["args"] == ["-m", "symbio_desktop.mcp_bridge"]
    assert "SYMBIO_HOME" in entry["env"] and "PYTHONPATH" in entry["env"]
    assert (tmp_path / "claude_desktop_config.json.bak-symbio").exists()

    assert connect.claude_desktop(remove=True, config_path=config) == 0
    data = json.loads(config.read_text())
    assert "symbio" not in data["mcpServers"] and "other" in data["mcpServers"]


def test_connect_leaves_an_unreadable_config_alone(tmp_path):
    from symbio.app import connect

    config = tmp_path / "claude_desktop_config.json"
    config.write_text("{not json")
    assert connect.claude_desktop(config_path=config) == 1
    assert config.read_text() == "{not json"


@pytest.mark.parametrize("argv,command", [
    (["acp"], "acp"), (["mcp", "bridge"], "mcp"), (["connect", "claude-desktop"], "connect"),
])
def test_the_bridges_are_subcommands(argv, command):
    from symbio.app.cli import _build_parser

    args = _build_parser().parse_args(argv)
    assert args.command == command
