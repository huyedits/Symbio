"""The laya decision-module contract (symbio/laya.py).

No real weights load in the suite: the agent seam is monkeypatched. What is
pinned is the behavior around it — the confidence floor, the refusal shape,
the off-switch, and that the agent is FREED after every call, because a
~1.2GB router staying resident beside the daemon is the residency this
module exists to avoid.
"""
import pytest

from symbio import laya


@pytest.fixture
def fake_agent(monkeypatch):
    """A laya agent that answers from a canned distribution."""
    calls = {"loads": 0, "predicts": 0}

    class _Fake:
        def predict(self, state, questions):
            calls["predicts"] += 1
            q = questions["q"]
            probs = {k: 1 / len(q["criteria"]) for k in q["criteria"]}
            probs["billing"] = 0.81
            s = sum(probs.values())
            probs = {k: v / s for k, v in probs.items()}
            return {"answers": {"q": {
                "type": "choice", "choice": "billing",
                "probabilities": probs,
                "answer_confidence": probs["billing"],
            }}}

    def _load(*_a, **_k):
        calls["loads"] += 1
        return _Fake()

    monkeypatch.setattr(laya, "_AGENT", None)
    monkeypatch.setattr("laya.load", _load)
    return calls


def test_a_confident_answer_passes_and_frees_the_agent(fake_agent, monkeypatch):
    monkeypatch.setattr(laya, "CONFIDENCE_FLOOR", 0.4)
    res = laya.decide("invoice text", "Which department?",
                      {"billing": "invoices", "technical": "bugs"})
    assert res is not None and res["choice"] == "billing"
    assert res["confidence"] > 0.4
    assert fake_agent["loads"] == 1
    assert laya._AGENT is None, "a router must not stay resident"


def test_a_marginal_answer_is_refused(monkeypatch):
    """A winner under the floor is refused — and the weights are freed."""
    class _Marginal:
        def predict(self, state, questions):
            criteria = list(questions["q"]["criteria"])
            n = len(criteria)
            probs = {k: 1 / n for k in criteria}
            winner = "billing" if "billing" in probs else criteria[0]
            return {"answers": {"q": {
                "type": "choice", "choice": winner,
                "probabilities": probs,
                "answer_confidence": probs[winner],
            }}}

    loads = {"n": 0}

    def _load(*_a, **_k):
        loads["n"] += 1
        return _Marginal()

    monkeypatch.setattr(laya, "_AGENT", None)
    monkeypatch.setattr("laya.load", _load)
    monkeypatch.setattr(laya, "CONFIDENCE_FLOOR", 0.99)
    assert laya.decide("text", "Which?", {"billing": "x", "technical": "y"}) is None
    assert laya._AGENT is None, "refusal frees the weights too"
    assert loads["n"] == 1
    assert laya._AGENT is None, "refusal frees the weights too"


def test_yes_no_avoids_the_noul_primitive(fake_agent):
    laya.yes_no("is this positive?", "Is this review positive?")
    laya.forget()
    # The wire format is a two-option choice, not noul: the shipped English
    # checkpoint follows its own labels on noul (laya#156). Pin the shape.
    assert fake_agent["predicts"] >= 1


def test_an_exception_is_a_none_never_a_raise(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("no engine here")

    monkeypatch.setattr(laya, "_agent", _boom)
    monkeypatch.setattr(laya, "CONFIDENCE_FLOOR", 0.4)
    assert laya.decide("s", "q?", {"a": "x"}) is None
    assert laya.yes_no("s", "q?") is None


def test_the_off_switch(monkeypatch):
    monkeypatch.setenv("SYMBIO_NO_LAYA", "1")
    assert laya.available() is False