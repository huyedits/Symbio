"""A cache that cannot be rewound restarts from a copy, not from token 0.

Hybrid models (Qwen3.5's linear-attention layers, LFM2's convolutions) keep
recurrent state that mlx_lm cannot trim, so any change to the cached prefix
used to rebuild the whole cache — and the previous reply never renders back
token for token, so that was every generation. Measured 2026-09-27 on
Qwen3.5-9B: ~32s to first token, each time, on a 6.5k-token prompt.
"""
import pytest
pytest.importorskip("mlx")  # MLX is Apple Silicon only; skip elsewhere
import mlx.nn as nn

from symbio.app import chat


class WordTokenizer:
    """One token per space-separated word, so prefixes line up the way a
    real tokenizer's do across chat-template role markers."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        text = " ".join(f"{m['role']}: {m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += " assistant:"
        return text

    def encode(self, text, add_special_tokens=True):
        return text.split(" ")


class _Response:
    def __init__(self, text, token):
        self.text, self.token = text, token


CONFIG = {
    "assistant_name": "Caine", "user_name": "Huy",
    "agent": {"temperature": 0.1, "top_p": 0.9, "max_reply_tokens": 100,
              "prompt_cache_enabled": True, "stream_output": True,
              "max_tool_rounds": 5, "history_limit": 40, "cron_poll_seconds": 9999},
    "tools": {"enabled_groups": []},
    "learn": {}, "memory": {"enabled": False}, "rag": {"enabled": False}, "web": {},
}


def _session(monkeypatch, replies, rewindable):
    seen = {"feeds": [], "prefilled": [], "made": 0}

    def make_cache(model):
        seen["made"] += 1
        return []

    order = iter(replies)

    def stream(model, tokenizer, prompt, max_tokens=256, sampler=None,
               prompt_cache=None, **kwargs):
        seen["feeds"].append(list(prompt))
        for i, word in enumerate(next(order).split(" ")):
            yield _Response(word if i == 0 else " " + word, word)

    monkeypatch.setattr(chat, "make_prompt_cache", make_cache)
    monkeypatch.setattr(chat, "can_trim_prompt_cache", lambda cache: rewindable)
    monkeypatch.setattr(chat, "trim_prompt_cache", lambda cache, n: cache)
    session = chat.ChatSession(
        {**CONFIG, "agent": dict(CONFIG["agent"])}, model=object(),
        tokenizer=WordTokenizer(), adapter_loaded=False,
        output_fn=lambda *a, **k: None, generate_fn=lambda *a, **k: "unused",
        stream_fn=stream,
    )
    session._mlx_generation = lambda: True
    session._prefill_into = lambda cache, ids: seen["prefilled"].append(list(ids))
    return session, seen


def test_a_cache_that_cannot_be_rewound_restarts_from_the_last_message(monkeypatch):
    session, seen = _session(monkeypatch, ["first reply", "second reply"], rewindable=False)
    m1 = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    session._generate_reply(m1)
    # Walked to the end of the last message and copied there; only the
    # generation prompt went through the generator.
    assert seen["prefilled"] == [["system:", "SYS\n", "user:"], ["hi\n"]]
    assert seen["feeds"][-1] == ["assistant:"]
    assert session._cache_checkpoints["turn"][0] == ["system:", "SYS\n", "user:", "hi\n"]
    # Up to where the newest user message's text starts.
    assert session._cache_checkpoints["user"][0] == ["system:", "SYS\n", "user:"]

    m2 = m1 + [{"role": "assistant", "content": "first reply"},
               {"role": "user", "content": "more"}]
    timings = {}
    session._generate_reply(m2, timings=timings)
    # The rendered reply ("reply\n") is not the generated one ("reply"), so
    # the live cache is stale — it restarts from the copy, not from zero.
    assert seen["made"] == 1, "rebuilt the cache from token 0"
    assert seen["prefilled"][-2:] == [["assistant:", "first", "reply\n", "user:"], ["more\n"]]
    assert seen["feeds"][-1] == ["assistant:"]
    assert timings["cached_tokens"] == 4


def test_a_tool_round_restarts_before_the_message_whose_context_moved(monkeypatch):
    # chat_turn prepends the time and the page the browser is on to the
    # newest user message, so between two rounds of one turn the prompt
    # changes there — before every page dump the loop has gathered since.
    session, seen = _session(monkeypatch, ["<call1>", "<call2>"], rewindable=False)
    head = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    r1 = head + [{"role": "user", "content": "[21:09] post it"}]
    session._generate_reply(r1)
    r2 = head + [{"role": "user", "content": "[21:10] post it"},
                 {"role": "assistant", "content": "<call1>"},
                 {"role": "user", "content": "[System observation: page]"}]
    timings = {}
    session._generate_reply(r2, timings=timings)
    assert seen["made"] == 1
    # Restarted from the copy taken where "post it" starts, not from the
    # system prompt.
    assert timings["cached_tokens"] == len(
        "system: SYS\n user: earlier\n assistant: ok\n user:".split(" "))


def test_a_rewindable_cache_never_copies_itself(monkeypatch):
    session, seen = _session(monkeypatch, ["first reply", "second reply"], rewindable=True)
    m1 = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    session._generate_reply(m1)
    session._generate_reply(m1 + [{"role": "assistant", "content": "first reply"},
                                  {"role": "user", "content": "more"}])
    assert seen["prefilled"] == []
    assert session._cache_checkpoints == {}


def test_restore_takes_the_longest_copy_the_prompt_extends(monkeypatch):
    session, seen = _session(monkeypatch, [], rewindable=False)
    session._cache_checkpoints = {"boot": ([1, 2], ["boot"]),
                                  "turn": ([1, 2, 3, 4], ["turn"])}

    cache, n = session._restore_checkpoint([1, 2, 3, 4, 5])
    assert (cache, n) == (["turn"], 4)
    cache.append("mutated")
    assert session._cache_checkpoints["turn"][1] == ["turn"], "handed out the checkpoint itself"

    assert session._restore_checkpoint([1, 2, 9])[1] == 2
    # Never the whole prompt: generation needs a token left to feed.
    assert session._restore_checkpoint([1, 2, 3, 4])[1] == 2
    made = seen["made"]
    assert session._restore_checkpoint([7, 8]) == ([], 0)
    assert seen["made"] == made + 1


def test_dropping_the_cache_drops_its_copies(monkeypatch):
    session, _ = _session(monkeypatch, [], rewindable=False)
    session._prompt_cache = ["live"]
    session._cache_checkpoints = {"boot": ([1], ["boot"]), "turn": ([1, 2], ["turn"])}
    session._drop_prompt_cache("the model weights are being unloaded")
    assert session._cache_checkpoints == {}


class _Model(nn.Module):
    pass


class _IntTokenizer(WordTokenizer):
    def encode(self, text, add_special_tokens=True):
        return [hash(w) % (2 ** 31) for w in text.split(" ")]


def test_boot_prefill_stops_where_a_user_message_starts(monkeypatch):
    walked = []

    def generate_step(prompt, model, **kwargs):
        walked.append(len(prompt))
        return iter([])

    monkeypatch.setattr(chat, "make_prompt_cache", lambda model: [])
    monkeypatch.setattr(chat, "can_trim_prompt_cache", lambda cache: False)
    monkeypatch.setattr(chat, "generate_step", generate_step)
    agent = {**CONFIG["agent"], "persist_prompt_cache": False}
    session = chat.ChatSession(
        {**CONFIG, "agent": agent}, model=_Model(), tokenizer=_IntTokenizer(),
        adapter_loaded=False, output_fn=lambda *a, **k: None,
    )
    session._await_prefill()
    tok = _IntTokenizer()
    with_empty = tok.encode(tok.apply_chat_template(
        [{"role": "system", "content": session.system_prompt},
         *chat.tool_few_shots(session.config), {"role": "user", "content": ""}]))
    # Everything up to the user turn's content, and not its closing tokens,
    # which no real prompt repeats.
    assert walked and walked[0] < len(with_empty)
    assert session._cache_checkpoints["boot"][0] == session._cached_prompt_ids \
        == with_empty[:walked[0]]
