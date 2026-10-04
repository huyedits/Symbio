"""A turn that thinks until it runs out of tokens must still answer.

thinking_level gives the <think> block a token allowance on top of the reply
budget, and it is a budget rather than a leash: a model that keeps deliberating
is cut off at the end of it, mid-thought, having written no answer and emitted
no tool call.

Live 2026-09-06: asked to post to @grok, the model reasoned in circles about
what the page might look like — it could not see it — and the transcript
simply stops mid-sentence.

What reaches the user then is not silence, which is the part worth pinning
down. strip_reasoning_block matches <think>...</think>, so on an unclosed
block it strips the dangling opening tag and leaves the prose: the private
deliberation is printed in the assistant's voice as its answer, and because it
is not empty, nothing downstream treats it as a failure. The generic
blank-reply nudge never even sees it — and could not help anyway, since it
asks the model to answer without changing the budget it just exhausted.
"""

from symbio import constants
from symbio.app import chat, tooling
from symbio.app import config as app_config


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False,
                            **kwargs):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text


def _session(monkeypatch, tmp_path, output):
    (tmp_path / "adapters").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR",
                        tmp_path / "adapters" / "workers")
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(chat, "load", lambda *a, **k: (object(), FakeTokenizer()))
    return chat.ChatSession(
        app_config.load_config(), model=object(), tokenizer=FakeTokenizer(),
        adapter_loaded=False, output_fn=output.append,
        generate_fn=lambda *a, **k: "",
    )


# The shape a cut-off reasoning block actually has: an opening tag, prose, and
# then nothing — the tokens ran out before the model could close it or write a
# single word of answer.
_CUT_OFF = ("<think>Okay, the user wants me to post to @grok. Let me think "
            "about which element to click. Maybe the composer is at the top. "
            "Or maybe I should check the page structure again. Let me")


def _reply(timings, text, streamed=False):
    """What a fake generator has to report alongside the text.

    "Ran out of road" is not visible in the reply: with thinking on the
    opening <think> is in the PROMPT, so a deliberation cut off at the budget
    and a short complete answer both arrive tagless. The real _generate_reply
    reports whether the token cap was hit; a fake that does not model that is
    testing a signal the code no longer uses.
    """
    if timings is not None:
        timings["hit_token_cap"] = text is _CUT_OFF
    return text, streamed


def test_a_reasoning_block_cut_off_mid_thought_is_retried_without_thinking(
        monkeypatch, tmp_path):
    output = []
    session = _session(monkeypatch, tmp_path, output)

    calls = []

    def fake_generate(messages, chunk_prefix="", timings=None,
                      think=True, reasoning_budget=0):
        calls.append(think)
        if len(calls) == 1:
            return _reply(timings, _CUT_OFF)
        return _reply(timings,
                      "The composer is empty; I have not posted anything yet.")

    monkeypatch.setattr(session, "_generate_reply", fake_generate)
    session._agent_turn("post something to @grok")

    assert calls[0] is True, "the first sample uses the configured thinking level"
    assert calls[1] is False, (
        "the retry has to take the budget question off the table by turning "
        f"thinking off; it was called with think={calls[1]!r}")


def test_the_user_gets_an_answer_rather_than_a_silent_turn(monkeypatch, tmp_path):
    output = []
    session = _session(monkeypatch, tmp_path, output)

    replies = iter([_CUT_OFF, "The composer is empty; I have not posted anything."])
    monkeypatch.setattr(
        session, "_generate_reply",
        lambda *a, timings=None, **k: _reply(
            timings, next(replies, "Done."), False))
    session._agent_turn("post something to @grok")

    printed = "\n".join(output)
    assert "I have not posted anything" in printed
    assert "Ran out of tokens mid-thought" in printed, (
        "the user should be told why the turn restarted, not just see a pause")


def test_the_retry_happens_only_once(monkeypatch, tmp_path):
    """A model that blanks twice must fall through to the normal blank-reply
    handling rather than resampling forever."""
    output = []
    session = _session(monkeypatch, tmp_path, output)

    calls = []

    def always_cut_off(messages, chunk_prefix="", timings=None,
                       think=True, reasoning_budget=0):
        calls.append(think)
        return _reply(timings, _CUT_OFF)

    monkeypatch.setattr(session, "_generate_reply", always_cut_off)
    session._agent_turn("post something to @grok")

    assert calls.count(False) == 1, (
        f"exactly one no-think retry per turn, got {calls}")


def test_a_closed_reasoning_block_with_a_real_answer_is_left_alone(
        monkeypatch, tmp_path):
    """The retry must not fire on a normal thinking turn — that would throw
    away a perfectly good answer and generate it a second time."""
    output = []
    session = _session(monkeypatch, tmp_path, output)

    calls = []

    def fake_generate(messages, chunk_prefix="", timings=None,
                      think=True, reasoning_budget=0):
        calls.append(think)
        return _reply(timings, "<think>Short thought.</think>Here is your answer.")

    monkeypatch.setattr(session, "_generate_reply", fake_generate)
    # A task, so the turn is served WITH thinking (think_when "auto" answers
    # plain chat like "hello" without a reasoning block at all).
    session._agent_turn("debug why my script crashes")

    assert calls == [True], f"one sample only, got {calls}"
    assert "Ran out of tokens mid-thought" not in "\n".join(output)


def test_an_empty_reply_with_no_thinking_at_all_is_not_treated_as_truncation(
        monkeypatch, tmp_path):
    """An empty string is a different failure with its own handling; this
    branch is specifically for reasoning that was cut off mid-block."""
    output = []
    session = _session(monkeypatch, tmp_path, output)

    monkeypatch.setattr(session, "_generate_reply", lambda *a, **k: ("", False))
    session._agent_turn("hello")

    assert "Ran out of tokens mid-thought" not in "\n".join(output)


# ---- the predicate the branch rests on ----

def test_an_unclosed_block_is_detected():
    assert tooling.think_block_closed(_CUT_OFF) is False


def test_a_closed_block_is_not_flagged():
    assert tooling.think_block_closed("<think>done</think>answer") is True


def test_the_deliberation_is_not_delivered_as_the_answer(monkeypatch, tmp_path):
    """The failure that made this worth fixing. strip_reasoning_block matches
    <think>...</think>, so on an unclosed block it removes the dangling
    opening tag and leaves the prose behind — and the model's private
    deliberation ("Maybe the composer is at the top. Or maybe I should check
    the page structure again") is printed in the assistant's voice as its
    reply. It is not empty, so nothing else downstream calls it a failure."""
    output = []
    session = _session(monkeypatch, tmp_path, output)

    replies = iter([_CUT_OFF, "The composer is empty."])
    monkeypatch.setattr(
        session, "_generate_reply",
        lambda *a, timings=None, **k: _reply(timings, next(replies, "Done.")))
    session._agent_turn("post something to @grok")

    printed = "\n".join(output)
    assert "Maybe the composer is at the top" not in printed, (
        "the cut-off reasoning was delivered to the user as the answer")
    assert "The composer is empty." in printed


def test_a_failed_retry_does_not_ship_the_deliberation(monkeypatch, tmp_path):
    """The retry's error path has to abort the turn, not fall through. Every
    other error exit in this loop sets gen_aborted; a bare break leaves `reply`
    holding the first generation — the stripped, unclosed deliberation — and
    the post-loop finalisation delivers it as the answer, which is the exact
    failure this whole block exists to prevent."""
    output = []
    session = _session(monkeypatch, tmp_path, output)

    calls = []

    def cut_then_explode(messages, chunk_prefix="", timings=None,
                         think=True, reasoning_budget=0):
        calls.append(think)
        if len(calls) == 1:
            return _reply(timings, _CUT_OFF)
        raise RuntimeError("Metal OOM")

    monkeypatch.setattr(session, "_generate_reply", cut_then_explode)
    session._agent_turn("post something to @grok")

    printed = "\n".join(output)
    assert "Maybe the composer is at the top" not in printed, (
        "a failed retry shipped the cut-off reasoning as the reply")
    assert "MLX Error" in printed


def test_the_retry_does_not_print_a_second_reply_when_the_first_streamed(
        monkeypatch, tmp_path):
    """A truncated think block streams to the screen as reasoning, never as
    the answer — the answer prefix is only emitted on the first non-reasoning
    chunk, and there is none. So replacing `reply` and regenerating cannot
    duplicate a reply the user already saw, and `streamed_live` is reassigned
    by the retry so the print-if-not-streamed branch stays correct."""
    output = []
    session = _session(monkeypatch, tmp_path, output)

    replies = iter([(_CUT_OFF, True), ("The composer is empty.", True)])
    monkeypatch.setattr(session, "_generate_reply",
                        lambda *a, **k: next(replies, ("Done.", True)))
    session._agent_turn("post something to @grok")

    printed = "\n".join(output)
    assert printed.count("The composer is empty.") <= 1, (
        f"the answer was printed more than once:\n{printed}")
