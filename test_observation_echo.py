"""Guards against the model impersonating the harness.

"[System observation: ...]" is a user-role scaffold used to hand tool
results back to the model. It appears in ~16% of the training corpus,
always followed by an assistant turn, so the model can learn to emit the
scaffold itself and then answer its own invented observation on repeat.

The pre-existing guards all matched `startswith("[System observation")`
exactly. The reply that actually shipped to the user was
`system observation: User says 'yo' — how can I help?` — no bracket,
lowercase — and walked past every one of them, including the retrieval
filter, so it was logged as a normal assistant turn and became eligible to
come back as context.
"""

import pytest

from symbio.app import learn


# The exact string observed in sessions/2026-08-07_19-26-38-265227.jsonl.
OBSERVED = (
    "system observation: User says 'yo' — how can I help?\n"
    "system observation: User says 'yo' — how can I help?\n"
    "system observation: User says 'yo' — how can I help?\n"
    "system observation: User says 'yo' — how can I help?\n"
    "system observation: User says 'Yo' — how can I help?"
)


def test_the_reply_that_shipped_is_caught():
    assert learn.looks_like_observation_echo(OBSERVED) is True


def test_the_reply_that_shipped_is_also_caught_as_degenerate():
    """Two independent detectors, so one bad variant doesn't get through."""
    assert learn.looks_degenerate(OBSERVED) is True


@pytest.mark.parametrize("variant", [
    "[System observation: something]",
    "System observation: something",
    "system observation: something",
    "SYSTEM OBSERVATION: something",
    "  [system observation: something]",
    "> system observation: something",
    "**System observation:** something",
    "Sure!\nsystem observation: User says hi",
])
def test_near_miss_variants_are_caught(variant):
    assert learn.looks_like_observation_echo(variant) is True


@pytest.mark.parametrize("innocent", [
    "I made a system observation about your disk usage.",
    "The system observed nothing unusual.",
    "Here is what the observation system reported.",
    "",
    "Toggle wifi off, then on.",
])
def test_innocent_text_is_not_flagged(innocent):
    assert learn.looks_like_observation_echo(innocent) is False


def test_repeated_short_lines_are_not_degenerate():
    """List formatting repeats short tokens legitimately."""
    assert learn.looks_degenerate("- ok\n- ok\n- ok\n- ok") is False


def test_normal_prose_is_not_degenerate():
    text = "First line here.\nSecond line differs.\nThird is distinct too."
    assert learn.looks_degenerate(text) is False


def test_repetition_needs_to_actually_repeat():
    text = "A reasonably long distinct line here.\nAnother different long line."
    assert learn.looks_degenerate(text) is False


def test_whitespace_variation_still_counts_as_repetition():
    text = ("the same sentence repeated\n"
            "the  same   sentence repeated\n"
            "The same sentence repeated")
    assert learn.looks_degenerate(text) is True


# --- retrieval must not serve the poisoned turn back -------------------


def test_rag_filters_an_assistant_turn_that_impersonated_the_scaffold(monkeypatch):
    import rag as rag_mod

    class FakeStore:
        def search(self, query, limit=None, exclude_session=None):
            return [
                {"role": "assistant", "content": OBSERVED,
                 "session_id": "s1", "timestamp": "2026-08-07T19:26:38"},
                {"role": "assistant", "content": "Wifi is back up.",
                 "session_id": "s2", "timestamp": "2026-08-07T19:30:00"},
            ]

    searcher = rag_mod.Retriever.__new__(rag_mod.Retriever)
    searcher.session_store = FakeStore()
    searcher.exclude_session_id = None
    searcher.rag_cfg = {"sources": ["notes", "sessions"]}
    searcher._top_k = lambda: 5

    results = rag_mod.Retriever.search_sessions(searcher, "yo")
    joined = " ".join(r["text"] for r in results)
    assert "system observation" not in joined.lower()
    assert "Wifi is back up." in joined


# --- loop-level: the reply must never be shown or logged ---------------


def test_degenerate_reply_is_neither_displayed_nor_logged(monkeypatch):
    """The guard must run before the display/log block, not after it.

    Filtering only at retrieval time would leave the bad turn printed to the
    user and written to sessions/, where a later digest could still train on
    it. This pins the ordering, which is the part that was wrong.
    """
    import test_main_loop as tml
    from symbio.app import chat as chat_mod
    from symbio.app import sessions as sessions_mod

    logged: list[tuple[str, str]] = []
    shown: list[str] = []

    real_store = sessions_mod.SessionStore

    class RecordingStore(real_store):
        def log(self, role, content, *a, **kw):
            logged.append((role, str(content)))
            return super().log(role, content, *a, **kw)

    monkeypatch.setattr(sessions_mod, "SessionStore", RecordingStore)

    session = tml.ScriptedSession(
        user_inputs=["yo"],
        model_replies=[OBSERVED, "Hey — what do you need?"],
    )
    real_loop = chat_mod.chat_loop

    def capturing_loop(*args, **kwargs):
        kwargs["output_fn"] = lambda t: shown.append(str(t))
        return real_loop(*args, **kwargs)

    monkeypatch.setattr(chat_mod, "chat_loop", capturing_loop)
    session.run()

    assistant_logged = " ".join(c for r, c in logged if r == "assistant")
    assert "system observation" not in assistant_logged.lower(), assistant_logged
    assert "Hey — what do you need?" in assistant_logged

    displayed = " ".join(shown)
    assert "User says 'yo'" not in displayed, displayed
    assert any("[Echo]" in s for s in shown), shown
