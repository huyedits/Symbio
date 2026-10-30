"""Semantic ifs: the scoring and the abstain rule, without a model."""

import math

import pytest

from symbio.app import semif


class _Tok:
    """'A'..'Z' are tokens 0..25, ' A'..' Z' are 100..125; anything else is two tokens."""

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text in semif.LETTERS:
            return [semif.LETTERS.index(text)]
        if len(text) == 2 and text[0] == " " and text[1] in semif.LETTERS:
            return [100 + semif.LETTERS.index(text[1])]
        return [900, 901]


def _logits(**by_token):
    out = [-20.0] * 200
    for tok, val in by_token.items():
        out[int(tok[1:])] = val
    return out


def test_both_spellings_of_a_letter_vote_for_it():
    assert semif.letter_token_ids(_Tok(), "B") == [1, 101]


def test_scores_are_a_softmax_over_the_options_only():
    probs = semif.scores_from_logits(_logits(t0=2.0, t1=0.0, t50=30.0), [[0], [1]])
    assert probs[0] == pytest.approx(1 / (1 + math.exp(-2)))
    assert sum(probs) == pytest.approx(1.0)       # mass on token 50 is no option


def test_a_clear_winner_is_chosen(monkeypatch):
    monkeypatch.setattr(semif, "_last_logits", lambda m, t, p: _logits(t1=5.0, t101=4.0))
    res = semif.choose(None, _Tok(), "q?", [("x", "first"), ("y", "second")])
    assert res["choice"] == "y" and res["margin"] > 0.9


def test_a_close_call_is_no_call(monkeypatch):
    monkeypatch.setattr(semif, "_last_logits", lambda m, t, p: _logits(t0=1.0, t1=1.1))
    res = semif.choose(None, _Tok(), "q?", [("x", "first"), ("y", "second")])
    assert res["choice"] is None


def test_adapter_routing_falls_back_to_the_base_model_on_a_close_call(monkeypatch):
    monkeypatch.setattr(semif, "_last_logits", lambda m, t, p: _logits(t1=1.0, t2=1.05))
    res = semif.pick_adapter(None, _Tok(), "hi", {"researcher": "web lookups",
                                                  "browser_driver": "drives a browser"})
    assert res["choice"] == semif.NO_ADAPTER and res["abstained"]


def test_adapter_routing_shows_none_first_and_the_adapters_sorted(monkeypatch):
    seen = {}

    def fake(m, t, prompt):
        seen["p"] = prompt
        return _logits(t2=9.0)
    monkeypatch.setattr(semif, "_last_logits", fake)
    res = semif.pick_adapter(None, _Tok(), "price of a ps5?",
                             {"researcher": "web lookups", "browser_driver": "drives a browser"})
    assert res["choice"] == "researcher"
    assert seen["p"].index("A) None") < seen["p"].index("B) drives") < seen["p"].index("C) web")


def test_a_long_adapter_list_is_settled_by_knockout(monkeypatch):
    adapters = {f"skill{i:02d}": f"skill number {i}" for i in range(30)}
    calls = []

    def fake(m, t, prompt):
        calls.append(prompt)
        letters = [l.split(")")[0] for l in prompt.splitlines() if ") skill number 17" in l]
        if not letters:
            return _logits(t0=5.0)                  # no skill17 here: "none"
        return _logits(**{f"t{semif.LETTERS.index(letters[0])}": 5.0})
    monkeypatch.setattr(semif, "_last_logits", fake)
    res = semif.pick_adapter(None, _Tok(), "m", adapters)
    assert res["choice"] == "skill17" and res["rounds"] == 2
    assert len(calls) == 4                          # three heats of <=12, one final
