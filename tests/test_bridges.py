"""Symbio for other apps: the ACP agent, the MCP bridge, and `symb connect`.

Both bridges are tested against a stand-in for the daemon's side of
DaemonBridge, so what is checked is the protocol each speaks to its host: the
shapes of initialize and session/new, a prompt turn streamed as updates, an
approval prompt asked through the host, a cancel, and the MCP tool calls. The
real daemon behind them was exercised end to end with the official ACP and MCP
SDK clients; these keep the contract from drifting.
"""
import json
import subprocess
import sys
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

    def connecting(self):
        return False

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


def test_a_session_offers_learn_and_private_and_the_loops_commands(agent):
    agent_, out = agent
    FakeBridge.script = []
    agent_.handle({"jsonrpc": "2.0", "id": 1, "method": "session/new",
                   "params": {"cwd": "/tmp", "mcpServers": []}})
    result = next(m for m in out.messages if m.get("id") == 1)["result"]
    modes = result["modes"]
    assert modes["currentModeId"] == "private", "nothing is trained on unless asked"
    assert {m["id"] for m in modes["availableModes"]} == {"private", "learn"}
    commands = next(u for u in _updates(out)
                    if u["sessionUpdate"] == "available_commands_update")["availableCommands"]
    assert {"save", "learn", "train", "golden", "forget_last"} <= {c["name"] for c in commands}
    for session in agent_.sessions.values():
        session.closed.set()


def test_set_mode_is_confirmed_to_the_host(agent):
    agent_, out = agent
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    assert agent_.set_mode({"sessionId": session_id, "modeId": "learn"}) == {}
    assert agent_.sessions[session_id].mode == "learn"
    assert _updates(out)[-1] == {"sessionUpdate": "current_mode_update", "currentModeId": "learn"}
    with pytest.raises(RuntimeError):
        agent_.set_mode({"sessionId": session_id, "modeId": "yolo"})


@pytest.mark.parametrize("mode,saved", [("learn", True), ("private", False)])
def test_a_learn_session_is_kept_for_training_when_it_closes(agent, mode, saved):
    """A host closing the pipe used to read as "no" to Save conversation."""
    agent_, _ = agent
    FakeBridge.script = [("token", "ok")]
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    session = agent_.sessions[session_id]
    session.mode = mode
    agent_.prompt({"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]})
    FakeBridge.script = [("system", " Saved 1 exchange(s) to training data.")]
    session.finish()
    assert ("/save" in FakeBridge.instances[-1].said) is saved


def test_training_shows_as_a_tool_call_and_the_loops_plan(agent, tmp_path, monkeypatch):
    agent_, out = agent
    monkeypatch.setattr(acp.constants, "LOG_DIR", tmp_path)
    live = tmp_path / "training_live.json"
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    session = agent_.sessions[session_id]
    watcher = threading.Thread(target=agent_.watch_training, args=(session, 0.02), daemon=True)
    watcher.start()

    def write(**fields):
        state = {"run": "r1", "role": None, "phase": "training", "iters": 40, "iter": 0,
                 "prior_iters": 0, "train": [], "started_at": time.time(),
                 "verdict": None, "gated": True, "total_iters": None}
        state.update(fields)
        live.write_text(json.dumps(state))
        time.sleep(0.15)

    write()
    write(iter=10, train=[[10, 1.5]])
    write(phase="trained", iter=40, train=[[10, 1.5], [40, 0.4]], total_iters=40)
    write(phase="trained", iter=40, total_iters=40, verdict="kept")
    session.closed.set()
    watcher.join(1)

    updates = _updates(out)
    call = next(u for u in updates if u["sessionUpdate"] == "tool_call")
    assert call["title"] == "Fine-tuning the headmaster adapter"
    said = [u["content"][0]["content"]["text"] for u in updates
            if u["sessionUpdate"] == "tool_call_update"]
    assert said[:2] == ["Step 0/40", "Step 10/40 · loss 1.500 (started at 1.500)"]
    assert said[2].startswith("Golden gate")
    assert said[3].startswith("Kept: the adapter now carries 40 steps")
    last = [u for u in updates if u["sessionUpdate"] == "tool_call_update"][-1]
    assert last["status"] == "completed"
    final_plan = [u for u in updates if u["sessionUpdate"] == "plan"][-1]["entries"]
    assert [e["status"] for e in final_plan] == ["completed"] * 4


def test_the_gates_follow_up_run_closes_the_first_and_carries_the_verdict(
        agent, tmp_path, monkeypatch):
    """Seen live: the first run's call was left spinning at "Golden gate…",
    and the final plan still had its last stage in progress after a rollback."""
    agent_, out = agent
    monkeypatch.setattr(acp.constants, "LOG_DIR", tmp_path)
    live = tmp_path / "training_live.json"
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    session = agent_.sessions[session_id]
    watcher = threading.Thread(target=agent_.watch_training, args=(session, 0.02), daemon=True)
    watcher.start()

    def write(**fields):
        state = {"run": "r1", "role": None, "phase": "trained", "iters": 100, "iter": 100,
                 "prior_iters": 0, "train": [], "started_at": time.time(), "verdict": None,
                 "gated": True, "follows": None, "total_iters": 100}
        state.update(fields)
        live.write_text(json.dumps(state))
        time.sleep(0.15)

    write()
    write(run="r2", phase="training", iters=50, iter=0, total_iters=None, follows="r1")
    write(run="r2", iters=50, iter=50, total_iters=150, follows="r1", verdict="rolled_back")
    session.closed.set()
    watcher.join(1)

    updates = [u for u in _updates(out) if u["sessionUpdate"] == "tool_call_update"]
    first = [u for u in updates if u["toolCallId"] == "train_r1"]
    assert first[-1]["status"] == "completed"
    assert "follow-up run" in first[-1]["content"][0]["content"]["text"]
    second = [u for u in updates if u["toolCallId"] == "train_r2"]
    assert second[-1]["status"] == "failed"
    assert second[-1]["content"][0]["content"]["text"].startswith("Rolled back")
    final_plan = [u for u in _updates(out) if u["sessionUpdate"] == "plan"][-1]["entries"]
    assert [e["status"] for e in final_plan] == ["completed"] * 4


def test_an_old_run_on_disk_is_not_news(agent, tmp_path, monkeypatch):
    agent_, out = agent
    monkeypatch.setattr(acp.constants, "LOG_DIR", tmp_path)
    (tmp_path / "training_live.json").write_text(json.dumps(
        {"run": "old", "phase": "trained", "started_at": time.time() - 3600, "verdict": "kept"}))
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    session = agent_.sessions[session_id]
    watcher = threading.Thread(target=agent_.watch_training, args=(session, 0.02), daemon=True)
    watcher.start()
    time.sleep(0.15)
    session.closed.set()
    watcher.join(1)
    assert not [u for u in _updates(out) if u["sessionUpdate"].startswith("tool_call")]


def test_trainer_lines_leave_the_reply_while_a_run_is_shown(agent):
    """The training tool call carries the steps; the reply keeps the rest."""
    agent_, out = agent
    FakeBridge.script = [("system", "Iter 10: Train loss 1.234, Learning Rate 1e-05"),
                         ("system", "  [Train] Backing up current adapter...")]
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    agent_.sessions[session_id].training = True
    agent_.prompt({"sessionId": session_id, "prompt": [{"type": "text", "text": "/train"}]})
    reply = "".join(u["content"]["text"] for u in _updates(out)
                    if u["sessionUpdate"] == "agent_message_chunk")
    assert "Iter 10" not in reply and "Backing up current adapter" in reply


def test_a_commands_output_is_the_reply_not_thinking(agent):
    agent_, out = agent
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    FakeBridge.script = [("system", "Model: Qwen3-14B · adapter 400 steps")]
    agent_.prompt({"sessionId": session_id, "prompt": [{"type": "text", "text": "/status"}]})
    FakeBridge.script = [("system", "[Search] otters"), ("token", "Otters hold hands.")]
    agent_.prompt({"sessionId": session_id, "prompt": [{"type": "text", "text": "otters?"}]})
    kinds = [(u["sessionUpdate"], u["content"]["text"].strip()) for u in _updates(out)
             if u["sessionUpdate"] in ("agent_message_chunk", "agent_thought_chunk")]
    assert kinds == [("agent_message_chunk", "Model: Qwen3-14B · adapter 400 steps"),
                     ("agent_thought_chunk", "[Search] otters"),
                     ("agent_message_chunk", "Otters hold hands.")]


def test_a_model_that_died_between_turns_is_woken_not_waited_on(agent, monkeypatch):
    """Found in review: after the daemon died, say() held the text for a
    session that no longer existed and the ACP turn waited forever."""
    agent_, out = agent
    session_id = agent_.new_session({"cwd": "/tmp", "mcpServers": []})["sessionId"]
    session = agent_.sessions[session_id]
    session.bridge.ready = False             # the daemon went away
    woken = []
    monkeypatch.setattr(acp, "ensure_daemon", lambda: woken.append(1) or (True, ""))
    FakeBridge.script = [("token", "back again")]
    result = agent_.prompt({"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]})

    assert result == {"stopReason": "end_turn"} and woken == [1]
    texts = [u["content"]["text"] for u in _updates(out) if "content" in u and "text" in u["content"]]
    assert any("fresh conversation" in t for t in texts) and "back again" in texts


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


def test_a_command_answers_with_its_output_lines(bridge):
    """Seen from Hermes: /status, /save and /train all came back as
    "(Symbio returned no text.)" — a command speaks in output lines, not tokens."""
    bridge_, out = bridge
    FakeBridge.script = [("system", "  [Train] Starting LoRA (100 iters)..."),
                         ("system", "Iter 10: Train loss 2.100, Learning Rate 1e-05"),
                         ("system", "  [Golden] 12/15 checks passing (baseline 11/15) — no regression.")]
    result = _rpc(bridge_, out, "tools/call",
                  {"name": "ask_symbio", "arguments": {"message": "/train"}})["result"]
    text = result["content"][0]["text"]
    assert "[Train] Starting LoRA" in text and "no regression" in text
    assert "Iter 10" not in text                # step lines are symbio_status's job


def test_a_slow_turn_answers_in_time_and_the_next_call_collects_the_rest(
        bridge, monkeypatch):
    """Seen live: Hermes cut a /train off at 420s and the reply was lost."""
    bridge_, out = bridge
    waits = iter([0.3, 5.0])                # the first call is cut short
    monkeypatch.setattr(mcp_bridge, "_reply_wait", lambda: next(waits))
    FakeBridge.script = [("system", "  [Train] Starting LoRA (100 iters)..."), ("sleep", 0.8),
                         ("system", "  [Golden] Rolled back to the previous adapter.")]
    first = _rpc(bridge_, out, "tools/call", {"name": "ask_symbio",
                                              "arguments": {"message": "/train"}})["result"]
    text = first["content"][0]["text"]
    assert first["isError"] is False
    assert "[Train] Starting LoRA" in text and "still working" in text
    assert "Rolled back" not in text
    second = _rpc(bridge_, out, "tools/call", {"name": "ask_symbio",
                                               "arguments": {"message": "/status"}},
                  id_=2)["result"]["content"][0]["text"]
    assert "still on your earlier message ('/train')" in second
    assert "Rolled back to the previous adapter" in second
    assert "[Train] Starting LoRA" not in second, "what was reported is not told twice"
    assert FakeBridge.instances[-1].said == ["/train"], "the second message was not sent"


def test_a_long_command_keeps_its_end(bridge, monkeypatch):
    bridge_, out = bridge
    monkeypatch.setattr(mcp_bridge, "MAX_LINES", 3)
    FakeBridge.script = [("system", f"line {i}") for i in range(10)]
    text = _rpc(bridge_, out, "tools/call", {"name": "ask_symbio",
                                             "arguments": {"message": "/golden"}})["result"]["content"][0]["text"]
    assert text.splitlines() == ["(… 7 earlier lines)", "line 7", "line 8", "line 9"]


def test_a_chat_reply_leaves_the_activity_lines_out(bridge):
    bridge_, out = bridge
    FakeBridge.script = [("system", "[Search] otters"), ("token", "Otters hold hands.")]
    text = _rpc(bridge_, out, "tools/call", {"name": "ask_symbio",
                                             "arguments": {"message": "otters?"}})["result"]["content"][0]["text"]
    assert text == "Otters hold hands."


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


def _fake_hermes(monkeypatch, outputs):
    """subprocess.run as Hermes's CLI: each call gets the next (code, stdout)."""
    from symbio.app import connect

    calls = []

    def run(argv, input="", **_):
        calls.append((argv, input))
        code, stdout = outputs.pop(0)
        return subprocess.CompletedProcess(argv, code, stdout, "")

    monkeypatch.setattr(connect.subprocess, "run", run)
    return connect, calls


def test_connect_hermes_registers_through_hermes_own_cli(monkeypatch):
    """Hermes's writer keeps config.yaml's comments; a YAML dump from here would not."""
    connect, calls = _fake_hermes(monkeypatch, [
        (0, "\x1b[32m  ✓ Saved 'symbio' to ~/.hermes/config.yaml (2/2 tools enabled)\x1b[0m"),
    ])
    assert connect.hermes(binary="/bin/hermes") == 0
    [(add, answers)] = calls
    assert add[:6] == ["/bin/hermes", "mcp", "add", "symbio", "--command", sys.executable]
    env = dict(e.split("=", 1) for e in add[add.index("--env") + 1:add.index("--args")])
    assert set(env) == {"SYMBIO_HOME", "PYTHONPATH", "SYMBIO_MCP_REPLY_WAIT_S"}
    # Under Hermes's 300s per call and 420s per batch: seen live, a /train
    # held the call until Hermes cut it off at 420s despite `timeout: 900`.
    assert int(env["SYMBIO_MCP_REPLY_WAIT_S"]) < 300
    assert add[add.index("--args") + 1:] == ["-m", "symbio_desktop.mcp_bridge"]
    assert answers == "y\ny\n"          # overwrite it, enable both tools


def test_connect_hermes_says_so_when_hermes_refuses(monkeypatch, capsys):
    connect, _ = _fake_hermes(monkeypatch, [(1, "✗ Connection failed")])
    assert connect.hermes(binary="/bin/hermes") == 1
    assert "Connection failed" in capsys.readouterr().out


def test_connect_hermes_remove(monkeypatch, capsys):
    connect, calls = _fake_hermes(monkeypatch, [(0, "✓ Removed 'symbio' from config"),
                                                (0, "✗ Server 'symbio' not found in config.")])
    assert connect.hermes(remove=True, binary="/bin/hermes") == 0
    assert connect.hermes(remove=True, binary="/bin/hermes") == 0
    assert calls[0][0][1:] == ["mcp", "remove", "symbio"]
    out = capsys.readouterr().out
    assert "Removed Symbio from Hermes" in out and "not connected" in out


def test_the_hermes_binary_is_found_where_the_desktop_installer_puts_it(tmp_path, monkeypatch):
    from symbio.app import connect

    monkeypatch.setattr(connect.shutil, "which", lambda _: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))     # not this Mac's ~/.hermes
    assert connect.hermes_binary() is None
    assert connect.hermes() == 1                 # nothing to run, nothing run
    binary = tmp_path / "installs/72de/environments/532c/venv/bin/hermes"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    assert connect.hermes_binary() == str(binary)


@pytest.mark.parametrize("argv,command", [
    (["acp"], "acp"), (["mcp", "bridge"], "mcp"), (["connect", "claude-desktop"], "connect"),
    (["connect", "hermes"], "connect"),
])
def test_the_bridges_are_subcommands(argv, command):
    from symbio.app.cli import _build_parser

    args = _build_parser().parse_args(argv)
    assert args.command == command
