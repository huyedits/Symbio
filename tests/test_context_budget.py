"""The conversation had no token budget, only a message count.

history_limit caps MESSAGES; a browser turn's messages are page dumps of up to
max_page_chars each, and the KV cache holding them grows linearly and is never
trimmed. On the 14B plus its draft a token costs ~285 KB, so a 50k-token
session asks for ~14 GB of cache beside ~9 GB of weights on a 16 GB machine.
macOS does not OOM-kill that — it swaps, and the desktop stops responding.
Reported 2026-09-07 as "after 50k the whole thing freezes".
"""

import pytest

from symbio.app import chat
from symbio.app.chat import ChatSession
from symbio.app.chat_constants import _cache_nbytes


class _Arr:
    def __init__(self, nbytes):
        self.nbytes = nbytes


class _Layer:
    def __init__(self, *arrays):
        self.state = tuple(arrays)


def _session(**agent):
    """A ChatSession shell — no model, no tokenizer, just the budget logic.

    The live-RAM measurement is off unless a test asks for it: left on, every
    assertion about the configured budget would also be an assertion about how
    much memory the machine running the suite happens to have free.
    """
    agent.setdefault("kv_budget_reserve_gb", 0)
    s = ChatSession.__new__(ChatSession)
    s.config = {"agent": agent}
    s._prompt_cache = None
    s._kv_bytes_per_token = None
    s._said_context_full = False
    s._context_floor = None
    return s


# ---- weighing the cache ----

def test_the_cache_is_weighed_rather_than_derived():
    """Layer counts and head dimensions change with every headmaster swap and
    with agent.kv_bits. The cache can just be weighed instead."""
    assert _cache_nbytes(
        [_Layer(_Arr(1000), _Arr(1000)), _Layer(_Arr(2000))]) == 4000


def test_an_unreadable_cache_weighs_nothing_rather_than_raising():
    assert _cache_nbytes(None) == 0
    assert _cache_nbytes([object(), _Layer(None)]) == 0


def test_a_measurement_that_would_uncap_the_session_is_refused():
    """A half-built cache reports a per-token cost near zero, and a cap
    derived from that is millions of tokens — the freeze again, via the guard
    meant to prevent it."""
    s = _session(kv_budget_mb=4000)
    s._prompt_cache = [_Layer(_Arr(8))]

    s._measure_kv_cost(1000)

    assert s._kv_bytes_per_token is None


def test_a_real_measurement_is_kept_and_sets_the_cap():
    s = _session(kv_budget_mb=1000)
    s._prompt_cache = [_Layer(_Arr(100 * 1024 * 1000))]  # 100 KB/token

    s._measure_kv_cost(1000)

    assert s._kv_bytes_per_token == pytest.approx(100 * 1024)
    assert s._prompt_token_cap() == pytest.approx(1000 * 1024 * 1024 / (100 * 1024), rel=0.01)


def test_quantising_the_cache_raises_the_cap_without_touching_config():
    """agent.kv_bits quarters the per-token cost. Deriving the cap from the
    LIVE cache is what makes that show up as four times the context instead of
    as a number someone has to remember to change."""
    plain, quantised = _session(kv_budget_mb=4000), _session(kv_budget_mb=4000)
    plain._prompt_cache = [_Layer(_Arr(285 * 1024 * 100))]
    quantised._prompt_cache = [_Layer(_Arr(285 * 1024 * 100 // 4))]
    plain._measure_kv_cost(100)
    quantised._measure_kv_cost(100)

    assert quantised._prompt_token_cap() == pytest.approx(
        4 * plain._prompt_token_cap(), rel=0.01)


# ---- what the cap resolves to ----

def test_an_explicit_token_count_wins():
    assert _session(max_prompt_tokens=9000, kv_budget_mb=4000)._prompt_token_cap() == 9000


def test_zero_switches_the_cap_off_entirely():
    """What every version before 2026-09-07 did. Someone with 128 GB should be
    able to have that back."""
    assert _session(max_prompt_tokens=0)._prompt_token_cap() == 0
    assert _session(kv_budget_mb=0)._prompt_token_cap() == 0


def test_a_cap_below_the_system_prompt_is_refused():
    """A cap under the system prompt trims the whole conversation away every
    turn and still does not fit."""
    assert _session(max_prompt_tokens=10)._prompt_token_cap() == ChatSession._MIN_TOKEN_CAP


def test_an_unmeasured_session_still_caps():
    """The first turn of a cold session has nothing weighed yet, and that is
    exactly the turn that must not be allowed to run away."""
    cap = _session(kv_budget_mb=4000)._prompt_token_cap()

    assert cap == pytest.approx(
        4000 * 1024 * 1024 / ChatSession._FALLBACK_KV_BYTES_PER_TOKEN, rel=0.01)
    assert 10_000 < cap < 20_000


def test_a_junk_budget_falls_back_instead_of_raising():
    assert _session(kv_budget_mb="lots")._prompt_token_cap() > 0


# ---- what gets dropped ----

def _messages(n, size=1000):
    """A history of tool work: observations and replies, not typed questions.

    The user role on an observation is how the model is GIVEN tool output, so
    these are deliberately shaped as observations — the last few things the
    person actually said are protected from trimming now, and a fixture made
    entirely of real user turns would be testing the protection rather than
    the trimming.
    """
    out = [{"role": "system", "content": "SYSTEM"}]
    for i in range(n):
        out.append({"role": "user" if i % 2 == 0 else "assistant",
                    "content": (f"[System observation: m{i} " if i % 2 == 0
                                else f"m{i} ") + "x" * size})
    return out


def _counts(messages, per=1000):
    return [10] + [per] * (len(messages) - 1)


def test_the_oldest_turns_go_first_and_the_system_prompt_never_does():
    s = _session()
    messages = _messages(20)

    kept, dropped = s._fit_messages_to_cap(messages, _counts(messages), 0, 8000)

    assert kept[0]["content"] == "SYSTEM"
    assert dropped > 0
    assert kept[1]["content"].startswith(f"m{dropped} ")


def test_the_question_being_answered_is_never_dropped():
    """The tail is the turn the model is replying to. A cap that eats it turns
    a slow session into an incoherent one."""
    s = _session()
    messages = _messages(20)
    last = messages[-1]["content"]

    kept, _ = s._fit_messages_to_cap(messages, _counts(messages), 0, 4096)

    assert kept[-1]["content"] == last


def test_it_undershoots_the_cap_so_the_next_turn_is_cheap():
    """Trimming to the line overflows again next turn, and every overflow
    moves the start of the conversation — which is the one thing that
    invalidates the KV prefix. Undershooting buys several cheap turns."""
    s = _session()
    messages = _messages(30)

    kept, _ = s._fit_messages_to_cap(messages, _counts(messages), 0, 10000)
    kept_tokens = 10 + 1000 * (len(kept) - 1)

    assert kept_tokens <= 10000 * ChatSession._TRIM_TARGET


def test_nothing_is_dropped_when_it_already_fits():
    s = _session()
    messages = _messages(3)

    kept, dropped = s._fit_messages_to_cap(messages, _counts(messages), 0, 100_000)

    assert dropped == 0 and kept == messages


def test_one_enormous_page_dump_is_cut_down_and_says_so():
    """The floor — system prompt plus the last few turns — can be over the cap
    on its own: one 50k-token page. Dropping is not available, so the body is
    cut. A silently shortened observation is how a model comes to report what
    it cannot see, so the cut announces itself in the text."""
    s = _session()
    messages = [{"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "PAGE " * 20000},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "what did it say?"}]
    counts = [10, 60000, 2, 6]

    kept, _ = s._fit_messages_to_cap(messages, counts, 0, 8000)

    assert len(kept) == 4
    assert "tokens cut from here" in kept[1]["content"]
    assert len(kept[1]["content"]) < len(messages[1]["content"])
    assert kept[-1]["content"] == "what did it say?"


def test_the_template_overhead_counts_against_the_budget():
    """Role markers and the tool preamble are real tokens in the real prompt."""
    s = _session()
    messages = _messages(6)

    lean, _ = s._fit_messages_to_cap(messages, _counts(messages), 0, 8000)
    fat, _ = s._fit_messages_to_cap(messages, _counts(messages), 5000, 8000)

    assert len(fat) < len(lean)


# ---- end to end: the model never sees the oversized prompt ----

class _FakeTokenizer:
    """One token per whitespace-separated word, which is close enough to make
    the arithmetic legible in a test."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=True):
        return "\n".join(f"<{m['role']}> {m['content']}" for m in messages)

    def encode(self, text):
        return list(range(len(text.split())))


def _live_session(cap_tokens):
    s = _session(max_prompt_tokens=cap_tokens, max_reply_tokens=16,
                 prompt_cache_enabled=False)
    s.tokenizer = _FakeTokenizer()
    s.model = object()
    s.sampler = None
    s.stream_chunk_fn = None
    s.output_fn = lambda *_a, **_k: None
    s._indexing_now = False
    s._ensure_model_loaded = lambda: None
    s._await_prefill = lambda: None
    s.seen = {}

    def _generate(model, tokenizer, prompt="", **kw):
        s.seen["prompt_tokens"] = len(prompt.split())
        return "fine."

    s.generate_fn = _generate
    return s


def test_an_oversized_prompt_is_shrunk_before_it_reaches_the_model():
    """The whole point: the freeze happens during prefill, so the trim has to
    land before generation, not after it."""
    s = _live_session(5000)
    messages = [{"role": "system", "content": "SYSTEM " * 100}]
    for i in range(40):
        messages.append({"role": "user", "content": f"page{i} " * 500})

    reply, _ = s._generate_reply(messages)

    assert reply == "fine."
    assert s.seen["prompt_tokens"] <= 5000


def test_a_prompt_that_fits_is_passed_through_untouched():
    s = _live_session(5000)
    messages = [{"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "hello there"}]

    s._generate_reply(messages)

    assert s.seen["prompt_tokens"] == len(
        _FakeTokenizer().apply_chat_template(messages).split())


def test_the_user_is_told_once_not_every_turn():
    """A long browser run overflows on every turn. Saying so each time buries
    the session in notices."""
    s = _live_session(5000)
    said = []
    s.output_fn = lambda text: said.append(text)
    messages = [{"role": "system", "content": "SYSTEM"}]
    messages += [{"role": "user", "content": f"page{i} " * 500} for i in range(40)]

    s._generate_reply(list(messages))
    s._generate_reply(list(messages))

    assert len(said) == 1
    assert "kv_budget_mb" in said[0]


def test_the_cap_off_switch_really_is_off():
    s = _live_session(0)
    messages = [{"role": "system", "content": "SYSTEM"}]
    messages += [{"role": "user", "content": f"page{i} " * 500} for i in range(40)]

    s._generate_reply(messages)

    assert s.seen["prompt_tokens"] > 19_000


# ---- against a real MLX cache, not a stand-in ----

def test_a_real_kv_cache_is_weighed_correctly():
    """The stand-ins above prove the arithmetic; this proves it is reading the
    object mlx-lm actually hands us: 2 layers x (K and V) x 4 heads x 8 dims x
    4 bytes (float32) = 512 bytes per token.

    It also pins that .state comes back TRIMMED. KVCache over-allocates in
    steps of 256 tokens, and a measurement that weighed the padding would read
    high on a short prefix — safe (the cap comes out smaller) but wrong.""" 
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    cache = [KVCache() for _ in range(2)]
    for kv in cache:
        kv.update_and_fetch(mx.zeros((1, 4, 100, 8)), mx.zeros((1, 4, 100, 8)))
    mx.eval([c.state for c in cache])

    nbytes = _cache_nbytes(cache)

    assert nbytes / 100 == 512


def test_the_cap_falls_out_of_a_real_measurement():
    """Model-shaped rather than toy-shaped, because _measure_kv_cost refuses
    anything under 1 KB per token — no real model is that cheap, and a reading
    that low means a half-built cache, which would hand back a cap of millions
    of tokens: the freeze again, arriving through the guard against it."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    layers, heads, dims = 24, 8, 64
    s = _session(kv_budget_mb=1)
    s._prompt_cache = [KVCache() for _ in range(layers)]
    for kv in s._prompt_cache:
        kv.update_and_fetch(mx.zeros((1, heads, 100, dims)),
                            mx.zeros((1, heads, 100, dims)))
    mx.eval([c.state for c in s._prompt_cache])

    s._measure_kv_cost(100)

    assert s._kv_bytes_per_token == layers * 2 * heads * dims * 4
    # 1 MB of budget against ~98 KB per token comes out at the floor, which is
    # what the floor is for: a cap must never be smaller than a system prompt.
    assert s._prompt_token_cap() == ChatSession._MIN_TOKEN_CAP


# ---- the boundary has to stay put, or the cap costs more than it saves ----

def test_the_window_is_held_while_it_still_fits():
    """Recomputed from the full history every turn, the trim drops two more
    messages every turn — which moves the start of the prompt every turn,
    which is the one thing that invalidates the KV prefix. Measured on the
    0.6B before this: reuse pinned at 18 tokens, every turn re-prefilling its
    whole context."""
    s = _session()
    messages = _messages(20)
    counts = _counts(messages)

    first_pass, dropped_first = s._fit_messages_to_cap(messages, counts, 0, 10_000)
    grown = messages + [{"role": "user", "content": "next"},
                        {"role": "assistant", "content": "reply"}]
    second_pass, dropped_second = s._fit_messages_to_cap(
        grown, counts + [5, 5], 0, 10_000)

    assert dropped_second == dropped_first
    assert second_pass[1]["content"] == first_pass[1]["content"]


def test_the_window_moves_once_it_has_outgrown_the_cap():
    s = _session()
    messages = _messages(20)
    counts = _counts(messages)
    _, dropped_first = s._fit_messages_to_cap(messages, counts, 0, 10_000)

    grown = messages + [{"role": "user", "content": f"big{i}"} for i in range(8)]
    _, dropped_second = s._fit_messages_to_cap(
        grown, counts + [1000] * 8, 0, 10_000)

    assert dropped_second > dropped_first


def test_a_boundary_that_lands_on_a_repeated_message_moves_off_it():
    """A model on a short reply budget says "ok" over and over. Two identical
    messages make "find where I was" a coin flip, and landing on the earlier
    one silently re-expands the window."""
    s = _session()
    messages = [{"role": "system", "content": "SYSTEM"}]
    for i in range(12):
        messages.append({"role": "user", "content": f"page{i}"})
        messages.append({"role": "assistant", "content": "ok"})
    counts = [10] + [1000] * (len(messages) - 1)

    kept, _ = s._fit_messages_to_cap(messages, counts, 0, 8000)

    assert s._context_floor is not None
    assert kept[1]["content"] != "ok"


def test_a_floor_from_another_conversation_is_ignored():
    """_generate_reply is not only called for the main chat. A remembered
    boundary that is nowhere in this list must not be guessed at."""
    s = _session()
    s._context_floor = "not a fingerprint of anything here"
    messages = _messages(4)

    kept, dropped = s._fit_messages_to_cap(messages, _counts(messages), 0, 100_000)

    assert dropped == 0 and len(kept) == len(messages)


# ---- what the person actually said must survive the trim ----

def _browser_turn(rounds=8):
    """The shape that loses it: one question, then round after round of page
    dumps appended with the user role because that is how the model is given
    them."""
    msgs = [{"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "post 'haii @grok' for me"}]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": f"tool call {i}"})
        msgs.append({"role": "user",
                     "content": f"[System observation: page dump {i}]"})
    return msgs, [10, 20] + [1500] * (rounds * 2)


def test_the_question_survives_its_own_page_dumps():
    """Reported 2026-09-10 as "it forgets what i said". Verified: a question
    eight tool rounds back was dropped while eight page dumps were kept, and
    the model was left working on evidence with no task attached. keep_tail
    counts MESSAGES and a browser round appends two, so four messages is two
    rounds of dumps."""
    messages, counts = _browser_turn()

    kept, _ = _session()._fit_messages_to_cap(messages, counts, 0, 6000)

    assert any("haii @grok" in m["content"] for m in kept)


def test_the_last_five_things_the_person_said_are_all_kept():
    messages = [{"role": "system", "content": "SYS"}]
    for i in range(9):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer {i} " + "x" * 2000})

    kept, _ = _session()._fit_messages_to_cap(
        messages, [5] + [900] * (len(messages) - 1), 0, 6000)
    asked = [m["content"] for m in kept if m["content"].startswith("question")]

    assert asked == [f"question {i}" for i in range(4, 9)]


def test_a_protected_question_is_not_hollowed_out_instead():
    """Protecting a turn from being DROPPED and then cutting it to zero
    characters is the same forgetting by another route."""
    messages = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "summarise this file for me"},
                {"role": "user", "content": "[System observation: " + "x" * 40000 + "]"}]

    kept, _ = _session()._fit_messages_to_cap(messages, [5, 8, 12000], 0, 6000)

    assert "summarise this file for me" in kept[1]["content"]


def test_nothing_claims_a_cut_that_did_not_happen():
    """A note claiming a cut that did not happen teaches the model to
    distrust text that is complete — the same false report in miniature."""
    messages = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "short question"},
                {"role": "assistant", "content": "x" * 40000}]

    kept, _ = _session()._fit_messages_to_cap(messages, [5, 4, 12000], 0, 6000)

    assert kept[1]["content"] == "short question"


def test_the_bulk_is_shortened_before_the_conversation_is():
    messages, counts = _browser_turn(rounds=3)

    kept, _ = _session()._fit_messages_to_cap(messages, counts, 0, 5000)

    assert any("cut from here" in m["content"] for m in kept)
    assert kept[1]["content"] == "post 'haii @grok' for me"   # untouched


# ---- the budget the machine can actually afford ----
#
# kv_budget_mb is a constant and the machine is not. On 2026-09-16 a browser
# turn spent the configured 4000 MB while Chrome held 2-3 GB of the same 16 GB,
# and the Mac swapped: the whole desktop stopped responding, mid-tool-call, with
# no traceback and no jetsam record. The budget is measured now, and the smaller
# of the two wins.

@pytest.fixture(autouse=True)
def _forget_the_reading():
    """The reading is process-wide and cached for seconds; no test inherits it."""
    chat._reset_headroom()
    yield
    chat._reset_headroom()


def _ram(monkeypatch, free_gb):
    monkeypatch.setattr(chat.training, "free_ram_bytes",
                        lambda: None if free_gb is None else int(free_gb * 1024 ** 3))


def test_an_idle_machine_spends_the_whole_configured_budget(monkeypatch):
    _ram(monkeypatch, 12)

    s = _session(kv_budget_mb=4000, kv_budget_reserve_gb=3.0)

    assert s._kv_budgets() == (4000.0, 9216.0)
    assert s._prompt_token_cap() == int(4000 * 1024 * 1024
                                        / s._FALLBACK_KV_BYTES_PER_TOKEN)


def test_a_machine_with_the_browser_open_spends_less_than_the_budget(monkeypatch):
    """The whole point: 4 GB free, 3 GB of it spoken for, is not 4000 MB of
    cache however confidently the config says so."""
    _ram(monkeypatch, 4)

    s = _session(kv_budget_mb=4000, kv_budget_reserve_gb=3.0)

    assert s._kv_budgets()[1] == 1024.0
    assert s._prompt_token_cap() == max(
        s._MIN_TOKEN_CAP,
        int(1024 * 1024 * 1024 / s._FALLBACK_KV_BYTES_PER_TOKEN))


def test_the_cache_a_session_already_holds_counts_as_available(monkeypatch):
    """Trimming is what frees it. Counted as someone else's memory, a large
    session reads its own cache as pressure and trims itself to the floor."""
    _ram(monkeypatch, 4)
    s = _session(kv_budget_mb=4000, kv_budget_reserve_gb=3.0)
    s._prompt_cache = [_Layer(_Arr(2 * 1024 ** 3))]  # 2 GB of cache held

    assert s._kv_budgets()[1] == 3072.0  # 4 free + 2 held - 3 reserved


def test_the_reading_falls_at_once_and_rises_slowly(monkeypatch):
    """Memory freed by a closing tab can be taken back a second later. A budget
    that spends it the instant it appears is over budget with nothing to trim."""
    _ram(monkeypatch, 12)
    assert chat._live_headroom_mb(3.0, now=100.0) == 9216.0

    _ram(monkeypatch, 4)
    assert chat._live_headroom_mb(3.0, now=200.0) == 1024.0, "a fall is immediate"

    _ram(monkeypatch, 4.4)  # 1434 MB, inside one step of the current reading
    assert chat._live_headroom_mb(3.0, now=300.0) == 1024.0, "jitter must not move it"

    _ram(monkeypatch, 6)
    assert chat._live_headroom_mb(3.0, now=400.0) == 3072.0, "real recovery does"


def test_a_reading_is_reused_for_a_few_seconds(monkeypatch):
    """vm_stat is a subprocess, and free RAM does not move faster than this."""
    _ram(monkeypatch, 12)
    assert chat._live_headroom_mb(3.0, now=100.0) == 9216.0

    _ram(monkeypatch, 1)
    assert chat._live_headroom_mb(3.0, now=101.0) == 9216.0
    assert chat._live_headroom_mb(3.0, now=110.0) == 0.0


def test_ram_that_cannot_be_read_keeps_the_last_reading(monkeypatch):
    """Never un-cap on a failed measurement: that is the freeze, via the guard
    against it."""
    _ram(monkeypatch, 4)
    assert chat._live_headroom_mb(3.0, now=100.0) == 1024.0

    _ram(monkeypatch, None)
    assert chat._live_headroom_mb(3.0, now=200.0) == 1024.0


def test_ram_that_was_never_readable_leaves_the_configured_budget_alone(monkeypatch):
    _ram(monkeypatch, None)

    s = _session(kv_budget_mb=4000, kv_budget_reserve_gb=3.0)

    assert s._kv_budgets() == (4000.0, None)
    assert s._prompt_token_cap() == int(4000 * 1024 * 1024
                                        / s._FALLBACK_KV_BYTES_PER_TOKEN)


def test_a_zero_reserve_trusts_the_configured_budget_alone(monkeypatch):
    _ram(monkeypatch, 1)

    s = _session(kv_budget_mb=4000, kv_budget_reserve_gb=0)

    assert s._kv_budgets() == (4000.0, None)


def test_an_explicit_token_cap_still_overrides_everything(monkeypatch):
    _ram(monkeypatch, 1)

    assert _session(max_prompt_tokens=9000,
                    kv_budget_reserve_gb=3.0)._prompt_token_cap() == 9000
    assert _session(max_prompt_tokens=0,
                    kv_budget_reserve_gb=3.0)._prompt_token_cap() == 0


def test_the_advice_names_whichever_budget_is_actually_binding(monkeypatch):
    """"Raise kv_budget_mb" is wrong — and makes the freeze likelier — when the
    real constraint is that something else holds the RAM."""
    _ram(monkeypatch, 4)
    tight = _session(kv_budget_mb=4000, kv_budget_reserve_gb=3.0)._cap_advice()
    assert "held by something else" in tight and "kv_bits" in tight

    chat._reset_headroom()
    _ram(monkeypatch, 12)
    roomy = _session(kv_budget_mb=4000, kv_budget_reserve_gb=3.0)._cap_advice()
    assert "Raise agent.kv_budget_mb" in roomy


def test_a_nonsense_reserve_falls_back_to_the_default(monkeypatch):
    _ram(monkeypatch, 4)

    assert _session(kv_budget_mb=4000,
                    kv_budget_reserve_gb="plenty")._kv_budgets()[1] == 1024.0
