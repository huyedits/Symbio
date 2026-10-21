"""symbio_desktop/openai_api.py: the OpenAI-compatible bridge Open WebUI talks to."""

import json
import queue
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from symbio_desktop import openai_api


class FakeBridge:
    """Stands in for DaemonBridge: records what was said, replays scripted events."""

    def __init__(self, events: queue.Queue, script):
        self.events, self.script = events, script
        self.said, self.answers, self.closed = [], [], False
        self.ready = True

    def alive(self):
        return not self.closed

    def connecting(self):
        return False

    def say(self, text):
        self.said.append(text)
        for event in self.script.pop(0):
            self.events.put(event)

    def confirm(self, approved, always=False):
        self.answers.append(approved)
        for event in self.script.pop(0):
            self.events.put(event)

    def close(self):
        self.closed = True


def conversation(script):
    conv = openai_api.Conversation()
    bridges = []

    def fake_open():
        conv.events = queue.Queue()
        conv.bridge = FakeBridge(conv.events, script)
        bridges.append(conv.bridge)
        return True, ""

    conv._open = fake_open
    return conv, bridges


def test_turn_streams_tokens_and_activity():
    conv, bridges = conversation([[{"type": "system", "text": "  [RAG] 1 hit"},
                                   {"type": "token", "text": "pong"}, {"type": "done"}]])
    out = list(conv.turn("ping", new_chat=True))
    assert out == [("reasoning", "  [RAG] 1 hit\n"), ("content", "pong")]
    assert bridges[0].said == ["ping"]


def test_approval_ends_the_reply_and_the_next_message_answers_it():
    conv, bridges = conversation([
        [{"type": "confirm", "prompt": "Run: ls ~"}],
        [{"type": "token", "text": "done listing"}, {"type": "done"}],
    ])
    first = "".join(t for _, t in conv.turn("list my home", new_chat=True))
    assert "Run: ls ~" in first and "yes" in first
    assert conv.awaiting_confirm
    second = list(conv.turn("yes please", new_chat=False))
    assert bridges[0].answers == [True]
    assert bridges[0].said == ["list my home"]          # the answer is not a new message
    assert ("content", "done listing") in second


@pytest.mark.parametrize("reply", ["no", "nope, don't", "what would it do?"])
def test_anything_but_a_yes_refuses(reply):
    conv, bridges = conversation([[{"type": "confirm", "prompt": "rm x"}], [{"type": "done"}]])
    list(conv.turn("delete x", new_chat=True))
    list(conv.turn(reply, new_chat=False))
    assert bridges[0].answers == [False]


def test_new_chat_opens_a_new_session_and_refuses_a_pending_approval():
    conv, bridges = conversation([[{"type": "confirm", "prompt": "rm x"}], [], [{"type": "done"}]])
    list(conv.turn("delete x", new_chat=True))
    list(conv.turn("hello", new_chat=True))
    assert bridges[0].answers == [False] and bridges[0].closed
    assert len(bridges) == 2 and bridges[1].said == ["hello"]


def test_message_text_and_task_replies():
    assert openai_api.message_text([{"type": "text", "text": "a"}, {"type": "image_url"},
                                    {"type": "text", "text": "b"}]) == "a\n\nb"
    assert json.loads(openai_api.task_reply("### Task:\nGenerate a concise title")) == {"title": ""}
    assert json.loads(openai_api.task_reply("### Task:\nGenerate 1-3 broad tags")) == {"tags": []}


@pytest.fixture
def server(monkeypatch):
    calls = []

    class Conv:
        lock = threading.Lock()

        def turn(self, text, new_chat):
            calls.append((text, new_chat))
            yield "content", "hi"

    monkeypatch.setattr(openai_api, "CONVERSATION", Conv())
    monkeypatch.setattr(openai_api, "_config_summary", lambda: {"assistant_name": "Caine"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), openai_api.Handler)
    srv.key = "sk-test"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    srv.shutdown()
    srv.server_close()


def post(url, body, key="sk-test"):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return urllib.request.urlopen(req, timeout=10)


def test_wrong_key_is_refused(server):
    url, calls = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, {"messages": [{"role": "user", "content": "hi"}]}, key="nope")
    assert e.value.code == 401 and calls == []


def test_only_the_newest_message_is_sent_and_history_marks_continuation(server):
    url, calls = server
    body = json.load(post(url, {"messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "one"},
        {"role": "assistant", "content": "r"}, {"role": "user", "content": "two"}]}))
    assert body["choices"][0]["message"]["content"] == "hi"
    assert calls == [("two", False)]


def test_background_task_prompts_never_reach_the_agent(server):
    url, calls = server
    body = json.load(post(url, {"messages": [{"role": "user", "content": "### Task:\nGenerate a title"}]}))
    assert json.loads(body["choices"][0]["message"]["content"]) == {"title": ""}
    assert calls == []


def test_streamed_reply_is_sse(server):
    url, _ = server
    raw = post(url, {"stream": True, "messages": [{"role": "user", "content": "x"}]}).read().decode()
    frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    assert any(json.loads(f)["choices"][0]["delta"].get("content") == "hi" for f in frames[:-1])
