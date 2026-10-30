"""The model's own "am I finished?" check, read off its logits (semif).

A turn that succeeds at step one and hands the turn back trips none of the
loop's other gates when the request names no file or URL. The model is asked
instead, and a clear "more to do" buys one more lap — never two.
"""
from symbio.app import semif
from symbio.app.chat_turn import AgentTurnMixin
from tests.test_main_loop import ScriptedSession


def _run(judged, replies):
    calls = []

    def judge(self, user_input, reply, last_observation):
        calls.append(reply)
        return judged
    real = AgentTurnMixin._judges_unfinished
    AgentTurnMixin._judges_unfinished = judge
    try:
        session = ScriptedSession(user_inputs=["set it up and then check it works", "/quit", "n"],
                                  model_replies=replies)
        session.run()
    finally:
        AgentTurnMixin._judges_unfinished = real
    return session, calls


def test_a_self_judged_unfinished_turn_takes_one_more_step():
    session, calls = _run(True, ["<cmd>echo setup</cmd>",
                                 "Setup is in place and ready for the next part.",
                                 "<cmd>echo check</cmd>",
                                 "Checked: it works, the request is complete."])
    assert len(session.prompts_seen) == 4
    assert "by your own assessment" in session.prompts_seen[2]
    assert len(calls) == 1                       # asked once a turn, never again


def test_a_self_judged_finished_turn_ends_as_before():
    session, calls = _run(False, ["<cmd>echo setup</cmd>",
                                  "Setup is in place and the check passed, all done."])
    assert len(session.prompts_seen) == 2
    assert len(calls) == 1


class _Stub:
    config = {"agent": {}}
    model, tokenizer = object(), object()

    def _mlx_generation(self):
        return True


def test_the_judge_reads_more_as_unfinished_and_anything_else_as_done(monkeypatch):
    for choice, want in (("more", True), ("done", False), ("ask", False), (None, False)):
        monkeypatch.setattr(semif, "choose", lambda *a, c=choice, **k: {"choice": c})
        assert AgentTurnMixin._judges_unfinished(_Stub(), "do x then y", "did x", "ok") is want


def test_the_judge_is_off_without_a_loaded_mlx_model_or_when_configured_off(monkeypatch):
    monkeypatch.setattr(semif, "choose", lambda *a, **k: {"choice": "more"})
    s = _Stub()
    s.model = None
    assert AgentTurnMixin._judges_unfinished(s, "a", "b", "c") is False
    s = _Stub()
    s.config = {"agent": {"self_continue": False}}
    assert AgentTurnMixin._judges_unfinished(s, "a", "b", "c") is False
    s = _Stub()
    monkeypatch.setattr(semif, "choose", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert AgentTurnMixin._judges_unfinished(s, "a", "b", "c") is False
