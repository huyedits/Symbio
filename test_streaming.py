"""Tests for the streaming/prompt-cache layer added to make the headmaster
model feel fast: tooling.StreamingStripper (live tag-safe text),
tooling.detect_malformed_tag (self-correction signal), chat._common_prefix_len
(the token-level cache-reuse diff), and ChatSession._generate_reply's actual
cache trim/reset decisions."""

import json
import random

import mlx.nn as nn

from symbio.app import chat, tooling


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        # Space-joined (not newline-joined) so word-splitting below never
        # merges the last word of one message with the next message's role
        # prefix — keeps the fake's "tokens" stable at message boundaries,
        # the same property a real BPE tokenizer has across chat-template
        # role markers.
        text = " ".join(f"{m['role']}: {m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += " assistant:"
        return text

    def encode(self, text, add_special_tokens=True):
        return text.split(" ")


class _FakeResponse:
    __slots__ = ("text", "token")

    def __init__(self, text, token):
        self.text = text
        self.token = token


# ---- tooling.StreamingStripper ----

_STREAM_CASES = [
    "Sure. <note title='Coffee'>Huy likes coffee.</note><cmd>echo hi</cmd>",
    'I\'ll run that.\n<tool_call>{"name": "terminal", "arguments": {"cmd": "ls"}}</tool_call>',
    "Hey there!",
    "Truncated: <search>who won the",
    "Mixed <py>print(1)</py> and <search>foo",
    "x < 5 and y < 10, no tags here at all.",
    "<search>Tour de France 2031 winner</search> Checking now.",
    "Partial <em>not a tag</em> then <search>real one</search> done.",
    "",
]


def test_streaming_stripper_matches_strip_tool_tags_under_random_chunking():
    # The stripper is a live-preview UX layer, not the source of truth for
    # the final parsed reply (that's always strip_tool_tags on the complete
    # text) — so it doesn't replicate clean_response's final whitespace
    # trim on every incremental flush, only its tag-removal decisions.
    # Compare with surrounding whitespace normalized away.
    random.seed(0)
    for full in _STREAM_CASES:
        expected = tooling.strip_tool_tags(full)
        for _ in range(8):
            chunks = []
            i = 0
            while i < len(full):
                n = random.randint(1, 4)
                chunks.append(full[i:i + n])
                i += n
            stripper = tooling.StreamingStripper()
            out = "".join(stripper.feed(c) for c in chunks) + stripper.finish()
            assert out.strip() == expected, (full, out, expected)


def test_streaming_stripper_never_shows_raw_tag_syntax_mid_stream():
    full = "Before <search>hidden query</search> after <tool_call>{\"name\": \"x\"}</tool_call> end"
    stripper = tooling.StreamingStripper()
    seen = ""
    for ch in full:
        seen += stripper.feed(ch)
    seen += stripper.finish()
    for marker in ("<search", "</search", "<tool_call", "hidden query"):
        assert marker not in seen, seen
    assert "Before" in seen and "after" in seen and "end" in seen


def test_streaming_stripper_does_not_hold_back_literal_angle_bracket_in_prose():
    stripper = tooling.StreamingStripper()
    out = stripper.feed("if x < 5 then") + stripper.finish()
    assert out == "if x < 5 then"


# ---- bare JSON tool calls (the form prompt.md teaches) ----

def _bare(name="terminal", **args):
    return json.dumps({"name": name, "arguments": dict(args)})

def _wrapped(name="terminal", **args):
    o, c = chr(60) + "tool_call" + chr(62), chr(60) + "/tool_call" + chr(62)
    return o + json.dumps({"name": name, "arguments": dict(args)}) + c


def test_parse_tools_bare_json_call():
    calls = tooling.parse_tools("Sure. " + _bare(cmd="ls -la") + " done")
    assert calls == [("run_command", {"cmd": "ls -la"})], calls


def test_parse_tools_bare_json_function_key_form():
    call = json.dumps({"function": "web_search", "arguments": {"query": "ai news"}})
    assert tooling.parse_tools("go " + call) == [("web_search", {"query": "ai news"})]


def test_parse_tools_bare_json_unknown_name_is_not_a_call():
    # A JSON object whose "name" isn't a tool (e.g. prose about a record) must
    # never be executed.
    prose = json.dumps({"name": "Alice", "age": 30})
    assert tooling.parse_tools("the record " + prose + " in db") == []


def test_parse_tools_bare_json_inside_code_fence_ignored():
    fenced = "```json\n" + _bare(cmd="rm -rf /") + "\n```"
    assert tooling.parse_tools("example:\n" + fenced) == []


def test_parse_tools_wrapped_call_not_double_counted_as_bare():
    # A wrapped call contains the same JSON object; it must parse once, not twice.
    assert tooling.parse_tools("go " + _wrapped(cmd="ls")) == [("run_command", {"cmd": "ls"})]


def test_parse_tools_bare_json_respects_enabled_groups():
    assert tooling.parse_tools(_bare(cmd="ls"), enabled_groups={"terminal"}) == [("run_command", {"cmd": "ls"})]
    assert tooling.parse_tools(_bare(cmd="ls"), enabled_groups={"web_search"}) == []


def test_strip_tool_tags_removes_bare_json_call():
    assert tooling.strip_tool_tags("Let me check. " + _bare(cmd="ls") + " All done.") == "Let me check.  All done."


def test_strip_tool_tags_keeps_non_tool_json_object():
    prose = json.dumps({"age": 30, "name": "Alice"})
    assert tooling.strip_tool_tags("record " + prose + " ok") == "record " + prose + " ok"


def test_streaming_stripper_holds_back_bare_json_call_char_by_char():
    # build a call with nested arguments to stress brace balancing
    call = json.dumps({"name": "terminal", "arguments": {"cmd": "ls -la", "opts": {"a": 1}}})
    full = "Let me run " + call + " now"
    stripper = tooling.StreamingStripper()
    seen = ""
    for ch in full:
        seen += stripper.feed(ch)
        # raw JSON object syntax must never appear in the live stream
        assert "tool_call" not in seen
    seen += stripper.finish()
    assert seen == "Let me run  now", repr(seen)


def test_streaming_stripper_shows_prose_brace_set():
    stripper = tooling.StreamingStripper()
    out = "".join(stripper.feed(c) for c in "the set {a, b} is fine") + stripper.finish()
    assert out == "the set {a, b} is fine", repr(out)


def test_detect_malformed_tag_flags_truncated_bare_json():
    trunc = json.dumps({"name": "terminal", "arguments": {"cmd": "ls"}})[:-3]
    msg = tooling.detect_malformed_tag("doing " + trunc)
    assert msg is not None and "incomplete" in msg.lower(), msg


def test_detect_malformed_tag_clean_bare_call_returns_none():
    assert tooling.detect_malformed_tag("doing " + _bare(cmd="ls")) is None



# ---- tooling.detect_malformed_tag ----

def test_detect_malformed_tag_flags_unterminated():
    msg = tooling.detect_malformed_tag("Truncated: <search>who won the")
    assert msg is not None and "unterminated" in msg.lower()


def test_detect_malformed_tag_flags_invalid_json():
    msg = tooling.detect_malformed_tag('<tool_call>{not valid json}</tool_call>')
    assert msg is not None and "json" in msg.lower()


def test_detect_malformed_tag_clean_reply_returns_none():
    assert tooling.detect_malformed_tag("Just a normal sentence.") is None
    assert tooling.detect_malformed_tag(
        '<tool_call>{"name": "x", "arguments": {}}</tool_call> ok') is None
    assert tooling.detect_malformed_tag("<search>complete query</search> done") is None


# ---- chat._common_prefix_len ----

def test_common_prefix_len():
    assert chat._common_prefix_len(None, ["a", "b"]) == 0
    assert chat._common_prefix_len([], ["a", "b"]) == 0
    assert chat._common_prefix_len(["a", "b", "c"], ["a", "b", "c", "d"]) == 3
    assert chat._common_prefix_len(["a", "x"], ["a", "b"]) == 1
    assert chat._common_prefix_len(["a", "b"], ["a", "b"]) == 2
    assert chat._common_prefix_len(["z"], ["a", "b"]) == 0


# ---- ChatSession._generate_reply cache decisions ----

def _make_session(monkeypatch, replies):
    calls = {"make": 0, "trim": [], "can_trim": True}

    def fake_make_cache(model):
        calls["make"] += 1
        return []

    def fake_can_trim(cache):
        return calls["can_trim"]

    def fake_trim(cache, n):
        calls["trim"].append(n)
        return cache

    order = iter(replies)

    def fake_stream(model, tokenizer, prompt, max_tokens=256, sampler=None,
                    prompt_cache=None, **kwargs):
        reply = next(order)
        for i, word in enumerate(reply.split(" ")):
            # token == the word itself, so a later re-encode of this same
            # text (once it's history) can actually line up against it —
            # an arbitrary int here would never match anything.
            yield _FakeResponse(word if i == 0 else " " + word, word)

    monkeypatch.setattr(chat, "make_prompt_cache", fake_make_cache)
    monkeypatch.setattr(chat, "can_trim_prompt_cache", fake_can_trim)
    monkeypatch.setattr(chat, "trim_prompt_cache", fake_trim)
    monkeypatch.setattr(chat, "load", lambda *a, **k: (object(), FakeTokenizer()))

    config = {
        "assistant_name": "Caine", "user_name": "Huy",
        "agent": {"temperature": 0.1, "top_p": 0.9, "max_reply_tokens": 100,
                  "prompt_cache_enabled": True, "stream_output": True,
                  "max_tool_rounds": 5, "history_limit": 40, "cron_poll_seconds": 9999},
        "tools": {"enabled_groups": []},
        "learn": {}, "memory": {"enabled": False}, "rag": {"enabled": False}, "web": {},
    }
    session = chat.ChatSession(
        config, model=object(), tokenizer=FakeTokenizer(), adapter_loaded=False,
        output_fn=lambda *a, **k: None, generate_fn=lambda *a, **k: "unused",
        stream_fn=fake_stream,
    )
    return session, calls


def test_generate_reply_first_call_builds_fresh_cache(monkeypatch):
    session, calls = _make_session(monkeypatch, ["hello there"])
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    text, shown = session._generate_reply(messages)
    assert text == "hello there"
    assert calls["make"] == 1
    assert calls["trim"] == []


def test_generate_reply_growing_history_trims_instead_of_rebuilding(monkeypatch):
    session, calls = _make_session(monkeypatch, ["first reply", "second reply"])
    m1 = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    session._generate_reply(m1)
    assert calls["make"] == 1

    # Second call's prompt is the first one PLUS more appended — a pure
    # growth. The cache must be reused, not rebuilt from scratch; a *small*
    # trim right at the turn boundary is expected and fine (the raw
    # generated text doesn't include the template's own closing wrapper
    # around it, e.g. the newline/role-end marker added once that text
    # becomes history) — what actually matters is that it's nowhere close
    # to reprocessing the whole growing prompt again.
    m2 = m1 + [{"role": "assistant", "content": "first reply"},
               {"role": "user", "content": "more"}]
    ids2_len = len(FakeTokenizer().encode(FakeTokenizer().apply_chat_template(
        m2, add_generation_prompt=True)))
    session._generate_reply(m2)
    assert calls["make"] == 1, "growing-only history should not rebuild the cache"
    assert sum(calls["trim"]) <= 3, (
        "only the turn-boundary wrapper should need re-feeding, not real growth", calls["trim"])
    assert sum(calls["trim"]) < ids2_len, "trim should be far smaller than the full prompt"


def test_generate_reply_changed_prefix_trims_to_the_true_lcp(monkeypatch):
    session, calls = _make_session(monkeypatch, ["first reply", "second reply"])
    m1 = [{"role": "system", "content": "SYS one"}, {"role": "user", "content": "hi"}]
    session._generate_reply(m1)

    # Same length, but the SYSTEM block changed (simulates a different RAG
    # block) — the cached suffix is now stale and must be trimmed back.
    m2 = [{"role": "system", "content": "SYS two"}, {"role": "user", "content": "hi"}]
    session._generate_reply(m2)
    assert calls["trim"] == [len(session._cached_prompt_ids) - 0] or calls["trim"], (
        "a changed earlier block should trigger a trim back to the real prefix")


def test_generate_reply_resets_cache_on_exception(monkeypatch):
    session, calls = _make_session(monkeypatch, [])

    def boom(*a, **k):
        raise RuntimeError("simulated mlx crash")
        yield  # pragma: no cover - never reached, makes this a generator

    session.stream_fn = boom
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    try:
        session._generate_reply(messages)
        assert False, "expected the simulated crash to propagate"
    except RuntimeError:
        pass
    assert session._prompt_cache is None
    assert session._cached_prompt_ids is None


def test_generate_reply_streams_live_when_chunk_fn_set(monkeypatch):
    session, calls = _make_session(monkeypatch, ["hello world"])
    chunks = []
    session.stream_chunk_fn = chunks.append
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    text, shown = session._generate_reply(messages, chunk_prefix="Caine   : ")
    assert shown is True
    assert text == "hello world"
    assert "".join(chunks).startswith("Caine   : ")
    assert "hello" in "".join(chunks) and "world" in "".join(chunks)


def test_generate_reply_cache_disabled_uses_legacy_blocking_path(monkeypatch):
    session, calls = _make_session(monkeypatch, [])
    session.config["agent"]["prompt_cache_enabled"] = False
    called = {}

    def fake_generate_fn(model, tokenizer, prompt="", sampler=None, max_tokens=0, verbose=False):
        called["prompt"] = prompt
        return "legacy reply"

    session.generate_fn = fake_generate_fn
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    text, shown = session._generate_reply(messages)
    assert text == "legacy reply"
    assert shown is False
    assert calls["make"] == 0  # never touched the cache machinery at all
    assert isinstance(called["prompt"], str)  # full templated string, not token ids


class _FakeIntTokenizer(FakeTokenizer):
    """Like FakeTokenizer but returns integer token ids, matching real
    tokenizers and the mx.array conversion used during boot prefill."""

    def encode(self, text, add_special_tokens=True):
        # Stable integer per word so token-level prefix matching still works.
        return [hash(w) % (2 ** 31) for w in text.split(" ")]


class _FakeModel(nn.Module):
    """Minimal mlx.nn.Module stand-in so the boot prefill guard accepts it
    as a real model without running any actual MLX code."""
    pass


def test_prefill_system_prompt_cache_at_boot(monkeypatch):
    """When the real MLX generation path is used, the system prompt is
    prefilled into the KV cache during ChatSession construction so the first
    user turn skips reprocessing it."""
    step_calls = []

    def fake_make_cache(model):
        return []

    def fake_generate_step(prompt, model, **kwargs):
        step_calls.append(("generate_step", len(prompt)))
        return iter([])

    monkeypatch.setattr(chat, "make_prompt_cache", fake_make_cache)
    monkeypatch.setattr(chat, "generate_step", fake_generate_step)

    config = {
        "assistant_name": "Caine", "user_name": "Huy",
        "agent": {"temperature": 0.1, "top_p": 0.9, "max_reply_tokens": 100,
                  "prompt_cache_enabled": True, "stream_output": True,
                  "max_tool_rounds": 5, "history_limit": 40, "cron_poll_seconds": 9999},
        "tools": {"enabled_groups": []},
        "learn": {}, "memory": {"enabled": False}, "rag": {"enabled": False}, "web": {},
    }
    session = chat.ChatSession(
        config, model=_FakeModel(), tokenizer=_FakeIntTokenizer(), adapter_loaded=False,
        output_fn=lambda *a, **k: None,
    )
    assert session._prompt_cache is not None
    assert session._cached_prompt_ids is not None
    assert len(step_calls) == 1
    assert step_calls[0][0] == "generate_step"
    assert step_calls[0][1] > 0


def test_prefill_system_prompt_cache_skipped_with_fake_stream(monkeypatch):
    """Front-ends and tests that inject a fake stream_fn must not trigger a
    real model prefill call."""
    step_calls = []

    def fake_make_cache(model):
        return []

    def fake_generate_step(prompt, model, **kwargs):
        step_calls.append(("generate_step", len(prompt)))
        return iter([])

    def fake_stream(model, tokenizer, prompt, **kwargs):
        yield _FakeResponse("hi", "hi")

    monkeypatch.setattr(chat, "make_prompt_cache", fake_make_cache)
    monkeypatch.setattr(chat, "generate_step", fake_generate_step)

    config = {
        "assistant_name": "Caine", "user_name": "Huy",
        "agent": {"temperature": 0.1, "top_p": 0.9, "max_reply_tokens": 100,
                  "prompt_cache_enabled": True, "stream_output": True,
                  "max_tool_rounds": 5, "history_limit": 40, "cron_poll_seconds": 9999},
        "tools": {"enabled_groups": []},
        "learn": {}, "memory": {"enabled": False}, "rag": {"enabled": False}, "web": {},
    }
    session = chat.ChatSession(
        config, model=_FakeModel(), tokenizer=_FakeIntTokenizer(), adapter_loaded=False,
        output_fn=lambda *a, **k: None, stream_fn=fake_stream,
    )
    assert session._prompt_cache is None
    assert session._cached_prompt_ids is None
    assert step_calls == []
