"""The wildcard set must stay genuinely held out.

A generalisation test is only a generalisation test while its subjects are
absent from the training corpus. The moment a seed sample mentions one, the
case starts rewarding recall and nothing announces the change — which is
exactly what happened to eval.py, where three of nine "held-out" cases are
word-for-word training prompts.

These tests fail loudly if that drift starts, so the corpus and the wildcards
cannot quietly converge.
"""

import re

import pytest

from symbio.app import training, wildcards
from symbio.app.eval import EVAL_CASES


CFG = {"user_name": "Huy", "assistant_name": "Caine", "open_app": "Safari"}


def _seed_pairs():
    return training.build_seed_pairs("Caine", "Huy")


def _seed_text() -> str:
    return " || ".join(f"{u} {a}" for u, a in _seed_pairs()).lower()


def _wildcard_prompts() -> list[str]:
    return [c.prompt_fn(CFG) for c in wildcards.WILDCARD_CASES]


def test_no_wildcard_prompt_appears_in_the_corpus():
    seed_users = {u.strip().lower() for u, _ in _seed_pairs()}
    for prompt in _wildcard_prompts():
        assert prompt.strip().lower() not in seed_users, (
            f"wildcard prompt is a training prompt: {prompt!r}")


@pytest.mark.parametrize("subject", wildcards.subjects())
def test_wildcard_subjects_are_absent_from_the_corpus(subject):
    """The subject is what makes the case novel; it must stay unseen."""
    haystack = _seed_text()
    # Compare on the distinctive token, not the whole phrase — "a Framework 13
    # laptop" is novel because "framework 13" is, and articles are everywhere.
    distinctive = [w for w in re.findall(r"[a-z0-9.]+", subject.lower())
                   if len(w) > 4 and w not in {"current", "right", "these", "runs"}]
    assert distinctive, f"no distinctive token in {subject!r}"
    for token in distinctive:
        assert token not in haystack, (
            f"{token!r} (from wildcard subject {subject!r}) now appears in the "
            f"seed corpus — this case no longer tests generalisation. Change "
            f"the wildcard to a subject the corpus does not mention.")


def test_wildcards_do_not_reuse_eval_prompts():
    """Overlap with eval.py would make the two sets measure the same thing."""
    eval_prompts = {c.prompt_fn(CFG).strip().lower() for c in EVAL_CASES}
    for prompt in _wildcard_prompts():
        assert prompt.strip().lower() not in eval_prompts


def test_every_wildcard_has_a_distinct_id_and_prompt():
    ids = [c.id for c in wildcards.WILDCARD_CASES]
    prompts = _wildcard_prompts()
    assert len(set(ids)) == len(ids)
    assert len(set(prompts)) == len(prompts)


def test_wildcards_grade_the_decision_not_the_wording():
    """A correct tool choice must pass regardless of surrounding prose."""
    case = next(c for c in wildcards.WILDCARD_CASES
                if c.id == "wild_price_unseen_product")
    assert case.check("Checking now.", [("web_search", {"query": "x"})], CFG)
    assert case.check("Totally different words here.",
                      [("web_search", {"query": "y"})], CFG)


def test_price_case_rejects_the_failure_that_shipped():
    """Opening a browser for a pricing question must not pass."""
    case = next(c for c in wildcards.WILDCARD_CASES
                if c.id == "wild_price_unseen_product")
    assert not case.check(
        "Opening Chrome.", [("run_command", {"cmd": "open -a 'Google Chrome'"})], CFG)
    assert not case.check(
        "Opening Cloudflare.",
        [("browser_open", {"url": "https://www.cloudflare.com"})], CFG)
    # Guessing a price with no lookup is also a fail.
    assert not case.check("It's about $1,200.", [], CFG)


def test_browse_case_requires_the_address_from_the_request():
    """Emitting a memorised URL instead of the requested one must fail."""
    case = next(c for c in wildcards.WILDCARD_CASES
                if c.id == "wild_browse_unseen_site")
    assert case.check(
        "Opening it.", [("browser_open", {"url": "https://sqlite.org"})], CFG)
    assert not case.check(
        "Opening it.",
        [("browser_open", {"url": "https://www.cloudflare.com"})], CFG)


def test_trivial_arithmetic_case_rejects_needless_searching():
    case = next(c for c in wildcards.WILDCARD_CASES
                if c.id == "wild_no_tool_needed")
    assert case.check("36", [], CFG)
    assert not case.check("Let me look that up.",
                          [("web_search", {"query": "6 times 6"})], CFG)


# --- automation: history, trend, and never gating a retrain ------------


def test_record_run_tracks_the_delta(tmp_path, monkeypatch):
    from symbio import constants

    # Patch the file itself: conftest redirects WILDCARD_HISTORY_FILE for the
    # whole suite, and that override takes precedence over DATA_DIR — so
    # patching DATA_DIR alone leaves these tests sharing one file.
    monkeypatch.setattr(constants, "WILDCARD_HISTORY_FILE",
                        tmp_path / "wildcard_history.json", raising=False)
    first = wildcards.record_run(3, 9, ["a", "b"])
    assert first["delta"] is None, "first run has nothing to compare against"
    second = wildcards.record_run(6, 9, ["a"])
    assert second["delta"] == 3
    third = wildcards.record_run(4, 9, ["a", "b", "c"])
    assert third["delta"] == -2


def test_history_survives_across_runs(tmp_path, monkeypatch):
    from symbio import constants

    # Patch the file itself: conftest redirects WILDCARD_HISTORY_FILE for the
    # whole suite, and that override takes precedence over DATA_DIR — so
    # patching DATA_DIR alone leaves these tests sharing one file.
    monkeypatch.setattr(constants, "WILDCARD_HISTORY_FILE",
                        tmp_path / "wildcard_history.json", raising=False)
    wildcards.record_run(1, 9, [])
    wildcards.record_run(2, 9, [])
    history = wildcards.load_history()
    assert [h["score"] for h in history] == [1, 2]


def test_history_is_capped(tmp_path, monkeypatch):
    from symbio import constants

    # Patch the file itself: conftest redirects WILDCARD_HISTORY_FILE for the
    # whole suite, and that override takes precedence over DATA_DIR — so
    # patching DATA_DIR alone leaves these tests sharing one file.
    monkeypatch.setattr(constants, "WILDCARD_HISTORY_FILE",
                        tmp_path / "wildcard_history.json", raising=False)
    for i in range(8):
        wildcards.record_run(i, 9, [], max_entries=5)
    assert len(wildcards.load_history()) == 5


def test_corrupt_history_does_not_raise(tmp_path, monkeypatch):
    from symbio import constants

    # Patch the file itself: conftest redirects WILDCARD_HISTORY_FILE for the
    # whole suite, and that override takes precedence over DATA_DIR — so
    # patching DATA_DIR alone leaves these tests sharing one file.
    monkeypatch.setattr(constants, "WILDCARD_HISTORY_FILE",
                        tmp_path / "wildcard_history.json", raising=False)
    wildcards.history_path().write_text("{not json", encoding="utf-8")
    assert wildcards.load_history() == []
    entry = wildcards.record_run(2, 9, [])
    assert entry["delta"] is None


def test_format_result_reads_plainly():
    line = wildcards.format_result(
        {"score": 6, "total": 9, "delta": 2, "failed": ["wild_current_fact"]})
    assert "6/9" in line and "+2" in line and "wild_current_fact" in line


def test_wildcard_failure_never_rolls_back_a_retrain(monkeypatch):
    """A generalisation miss is not a regression and must not gate training."""
    import inspect

    from symbio.app import chat as chat_mod

    src = inspect.getsource(chat_mod.ChatSession._run_wildcard_check)
    assert "restore_adapter" not in src
    assert "rollback" not in src.lower().replace("rolls back", "")
    # And it must swallow its own errors rather than fail the training run.
    assert "except Exception" in src


def test_wildcard_check_respects_the_config_switch(monkeypatch):
    from symbio.app import chat as chat_mod

    calls = []
    monkeypatch.setattr(wildcards, "run_check",
                        lambda *a, **k: calls.append(1))
    session = chat_mod.ChatSession.__new__(chat_mod.ChatSession)
    session.output_fn = lambda t: None
    session._run_wildcard_check({"wildcard_check_enabled": False})
    assert calls == []


def test_history_path_is_redirectable(monkeypatch, tmp_path):
    """conftest redirects this during the suite; if it stops being read
    through constants at call time, pytest silently writes fake runs into the
    user's real trend file and the feature's only output is destroyed."""
    from symbio import constants

    monkeypatch.setattr(constants, "WILDCARD_HISTORY_FILE",
                        tmp_path / "redirected.json", raising=False)
    assert wildcards.history_path() == tmp_path / "redirected.json"


def test_history_falls_back_to_the_data_dir(monkeypatch, tmp_path):
    from symbio import constants

    monkeypatch.setattr(constants, "WILDCARD_HISTORY_FILE", None, raising=False)
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path)
    assert wildcards.history_path() == tmp_path / "wildcard_history.json"
