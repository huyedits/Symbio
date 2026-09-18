"""Adaptive fine-tuning: spend the LoRA budget where the model is weak now.

The mistake loop decides WHEN to train. These tests are about WHAT it trains
on — weighting by measured failure, ageing out absorbed lessons, learning from
predictions nobody corrected, and the two guards that stop the whole thing
becoming a feedback loop that grades its own homework.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from symbio import constants
from symbio.app import belief as B
from symbio.app import curriculum as C

HOUR = 3600.0


class FakeCase:
    def __init__(self, case_id, description="", prompt=""):
        self.id = case_id
        self.description = description
        self._prompt = prompt

    def prompt_fn(self, cfg):
        return self._prompt


class FakeEval:
    def __init__(self, tasks):
        self.tasks = tasks


def _cases():
    return [
        FakeCase("browser_scroll", "scroll a web page",
                 "Scroll down the browser page and read what appears"),
        FakeCase("math_product", "multiply two numbers",
                 "What is 17 multiplied by 23?"),
    ]


CONFIG = {"assistant_name": "Caine", "user_name": "Huy"}


# ---- (2) weight by what the model currently fails ----

def test_a_failing_case_weighs_more_than_a_passing_one():
    p = C.weakness_profile(FakeEval([{"id": "a", "passed": False},
                                     {"id": "b", "passed": True}]))
    assert p["a"] > p["b"]
    assert p["a"] == C.WEIGHT_WEAK and p["b"] == C.WEIGHT_STRONG


def test_a_case_that_errored_is_unknown_not_weak():
    """A generation that crashed says nothing about the model's grasp of the
    topic, and boosting on it lets an unrelated fault redirect a whole run."""
    p = C.weakness_profile(FakeEval([{"id": "a", "passed": False, "error": "OOM"}]))
    assert p["a"] == C.WEIGHT_UNKNOWN


def test_an_empty_eval_profiles_nothing():
    assert C.weakness_profile(FakeEval([])) == {}


def test_a_sample_about_a_failing_case_is_boosted():
    profile = C.weakness_profile(FakeEval([{"id": "browser_scroll", "passed": False},
                                           {"id": "math_product", "passed": True}]))
    topics = C.case_topics(_cases(), CONFIG)
    weak = C.sample_weight("the browser page did not scroll down", profile, topics)
    strong = C.sample_weight("multiply 17 by 23 to get the product", profile, topics)
    assert weak > strong


def test_a_sample_matching_nothing_lands_in_the_middle():
    """Silence about a topic is not evidence either way — it must not be read
    as weakness or as strength."""
    profile = C.weakness_profile(FakeEval([{"id": "browser_scroll", "passed": False}]))
    topics = C.case_topics(_cases(), CONFIG)
    assert C.sample_weight("the capital of Australia is Canberra",
                           profile, topics) == C.WEIGHT_UNKNOWN


def test_weighting_is_bounded_at_both_ends():
    """One stubborn case must not consume a whole run and turn a general
    adapter into a single-trick one."""
    profile = {c.id: 99.0 for c in _cases()}
    topics = C.case_topics(_cases(), CONFIG)
    assert C.sample_weight("browser page scroll down", profile, topics) <= C.WEIGHT_WEAK
    profile = {c.id: -5.0 for c in _cases()}
    assert C.sample_weight("browser page scroll down", profile, topics) >= C.WEIGHT_STRONG


def test_no_profile_means_no_opinion():
    assert C.sample_weight("anything", {}, {}) == C.WEIGHT_UNKNOWN


# ---- (3) age out what has been absorbed ----

def test_a_sample_halves_every_half_life():
    now = datetime.now(timezone.utc)
    assert C.age_weight((now - timedelta(days=21)).isoformat(), 21, now) == pytest.approx(0.5)
    assert C.age_weight((now - timedelta(days=42)).isoformat(), 21, now) == pytest.approx(0.25)
    assert C.age_weight(now.isoformat(), 21, now) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [None, "not a date", ""])
def test_an_unreadable_date_weighs_full_not_zero(bad):
    """Decaying it would silently delete the oldest material — exactly the
    notes least likely to still carry a usable date."""
    assert C.age_weight(bad) == 1.0


def test_a_naive_timestamp_is_treated_as_utc_not_crashed():
    now = datetime.now(timezone.utc)
    naive = (now - timedelta(days=21)).replace(tzinfo=None).isoformat()
    assert C.age_weight(naive, 21, now) == pytest.approx(0.5, rel=0.01)


def test_an_old_sample_the_model_still_fails_is_not_stale():
    """Age alone is not staleness. A three-week-old note about something still
    broken is the most valuable sample in the corpus."""
    profile = C.weakness_profile(FakeEval([{"id": "browser_scroll", "passed": False}]))
    topics = C.case_topics(_cases(), CONFIG)
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    assert not C.is_stale("browser page scroll down", old, profile, topics)


def test_an_old_sample_the_model_now_passes_is_stale():
    profile = C.weakness_profile(FakeEval([{"id": "browser_scroll", "passed": True}]))
    topics = C.case_topics(_cases(), CONFIG)
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    assert C.is_stale("browser page scroll down", old, profile, topics)


def test_a_fresh_sample_is_never_stale():
    profile = C.weakness_profile(FakeEval([{"id": "browser_scroll", "passed": True}]))
    topics = C.case_topics(_cases(), CONFIG)
    assert not C.is_stale("browser page scroll down",
                          datetime.now(timezone.utc).isoformat(), profile, topics)


# ---- (5) the mistake loop stays the floor ----

def test_every_sample_is_written_at_least_once():
    """A weighting scheme able to drop a sample to zero is one able to delete a
    lesson, and the first time it is wrong about which, the correction is gone."""
    profile_all_passing = FakeEval([{"id": c.id, "passed": True} for c in _cases()])
    old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    planned = C.plan(profile_all_passing, _cases(), CONFIG,
                     [{"text": "browser page scroll down", "created_at": old}])
    assert planned[0]["repeats"] >= 1
    assert planned[0]["stale"] is True


def test_a_weak_topic_outweighs_a_strong_one_in_the_plan():
    ev = FakeEval([{"id": "browser_scroll", "passed": False},
                   {"id": "math_product", "passed": True}])
    now = datetime.now(timezone.utc).isoformat()
    planned = C.plan(ev, _cases(), CONFIG, [
        {"text": "the browser page did not scroll down", "created_at": now},
        {"text": "multiply 17 by 23 to get the product", "created_at": now},
    ], base_boost=3)
    assert planned[0]["repeats"] > planned[1]["repeats"]


def test_the_summary_says_where_the_budget_went():
    ev = FakeEval([{"id": "browser_scroll", "passed": False}])
    planned = C.plan(ev, _cases(), CONFIG,
                     [{"text": "browser page scroll down"}], base_boost=3)
    line = C.summarise(planned)
    assert "boosted for measured weakness" in line
    assert C.summarise([]) .endswith("nothing to weight.")


# ---- (4) the guards against circularity ----

def test_an_eval_prompt_in_the_corpus_is_refused(tmp_path, monkeypatch):
    """The load-bearing guard: a curriculum steered by a test the model was
    trained on is steered by nothing."""
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "What is 17 multiplied by 23?"}))
    with pytest.raises(C.HeldOutViolation) as e:
        C.assert_held_out(_cases(), CONFIG, corpus)
    assert "self-reinforcing" in str(e.value)


def test_a_clean_corpus_passes_the_guard(tmp_path):
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "something else entirely"}))
    C.assert_held_out(_cases(), CONFIG, corpus)  # does not raise


def test_no_corpus_yet_is_not_a_leak(tmp_path):
    C.assert_held_out(_cases(), CONFIG, tmp_path / "missing.jsonl")


def test_a_self_grading_judge_is_refused():
    with pytest.raises(ValueError):
        C.assert_independent_judge("model-a", "model-a")
    C.assert_independent_judge("judge", "model-a")  # different: fine


# ---- (1) predictions that diverged, with no correction typed ----

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    return B.BeliefStore()


def test_a_confident_belief_that_did_not_hold_is_a_signal(store):
    """Nobody typed a correction and no tool failed — the model was simply
    sure of something that did not happen."""
    b = store.add_belief("the deploy will succeed", ["CI was green"],
                         confidence=0.95)
    store.update_confidence(b, False)
    assert [d["belief"]["id"] for d in C.diverged_beliefs(store)] == [b]


def test_being_wrong_about_a_coin_flip_is_not_a_lesson(store):
    b = store.add_belief("it might rain", confidence=0.5)
    store.update_confidence(b, False)
    assert C.diverged_beliefs(store) == []


def test_a_belief_that_held_is_not_a_divergence(store):
    b = store.add_belief("the deploy will succeed", confidence=0.95)
    store.update_confidence(b, True)
    assert C.diverged_beliefs(store) == []


def test_an_unobserved_belief_is_not_a_divergence(store):
    store.add_belief("something", confidence=0.95)
    assert C.diverged_beliefs(store) == []


def test_the_worst_divergence_comes_first(store):
    mild = store.add_belief("a", confidence=0.7)
    bad = store.add_belief("b", confidence=0.99)
    store.update_confidence(mild, False)
    store.update_confidence(bad, False)
    assert C.diverged_beliefs(store)[0]["belief"]["id"] == bad


def test_right_claim_wrong_timing_is_taught_as_a_timing_error(store):
    """The more useful half: a prediction right about what and wrong about when
    needs its timing corrected, not its content thrown away."""
    b = store.add_belief("the reactor can restart", ["xenon decays"],
                         confidence=0.95, timescale_s=2 * HOUR, dynamics=B.LAGS)
    store.update_confidence(b, True, when=48 * HOUR)
    [s] = C.divergence_samples(store)
    assert "not on the timescale" in s["reply"]
    assert "reactor can restart" in s["reply"]


def test_a_flat_miss_is_taught_as_overconfidence(store):
    b = store.add_belief("the deploy will succeed", confidence=0.95)
    store.update_confidence(b, False)
    [s] = C.divergence_samples(store)
    assert "more sure of it than the evidence supported" in s["reply"]


# ---- the learn.py wiring ----

def test_divergences_reach_the_corpus(store, monkeypatch):
    from symbio.app import learn
    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    b = store.add_belief("the deploy will succeed", confidence=0.95)
    store.update_confidence(b, False)
    assert learn.digest_divergences_to_training(object(), "SYS", store=store) == 1
    assert written


def test_a_worse_divergence_is_repeated_harder(store, monkeypatch):
    from symbio.app import learn
    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    b = store.add_belief("certain", confidence=0.99)
    store.update_confidence(b, False)
    learn.digest_divergences_to_training(object(), "SYS", boost=2, store=store)
    worst = len(written)

    written.clear()
    store2 = B.BeliefStore(store.path.with_name("b.db"))
    b2 = store2.add_belief("less certain", confidence=0.62)
    store2.update_confidence(b2, False)
    learn.digest_divergences_to_training(object(), "SYS", boost=2, store=store2)
    assert len(written) <= worst


def test_the_adaptive_digest_still_writes_every_note(tmp_path, monkeypatch):
    """Requirement 5 in arithmetic: adaptive weighting changes emphasis, it
    never removes what the reactive loop would have taught."""
    from symbio.app import learn
    mistakes = tmp_path / "mistakes"
    mistakes.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_DIR", mistakes)
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_ARCHIVE_DIR", archive)
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "train.jsonl")
    for i, (q, a) in enumerate([("scroll the browser page", "use browser_scroll"),
                                ("what is 17 times 23", "391")]):
        (mistakes / f"n{i}.md").write_text(
            f"# c\n\n**Category:** general\n\n**Severity:** 1\n\n"
            f"**Original question:** {q}\n\n**Wrong answer:** w\n\n"
            f"**Correction:** c\n\n**Correct answer:** {a}\n")
    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    ev = FakeEval([{"id": "browser_scroll", "passed": False},
                   {"id": "math_product", "passed": True}])
    notes, samples = learn.digest_mistakes_adaptively(
        object(), "SYS", ev, CONFIG, boost=3, cases=_cases())
    assert notes == 2                      # both notes digested
    assert {q for q, _ in written} == {"scroll the browser page",
                                       "what is 17 times 23"}
    scroll = sum(1 for q, _ in written if "scroll" in q)
    maths = sum(1 for q, _ in written if "17 times" in q)
    assert scroll > maths                  # budget went to the failing case
    assert samples == len(written)


def test_the_adaptive_digest_refuses_a_leaked_eval_set(tmp_path, monkeypatch):
    from symbio.app import learn
    monkeypatch.setattr(constants, "MISTAKES_DIR", tmp_path / "mistakes")
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "What is 17 multiplied by 23?"}))
    monkeypatch.setattr(constants, "TRAIN_FILE", corpus)
    with pytest.raises(C.HeldOutViolation):
        learn.adaptive_training_plan(
            FakeEval([{"id": "math_product", "passed": False}]),
            CONFIG, cases=_cases())


def test_the_reactive_digest_is_untouched(tmp_path, monkeypatch):
    """The floor still stands on its own, with no eval result in sight."""
    from symbio.app import learn
    mistakes = tmp_path / "mistakes"
    mistakes.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_DIR", mistakes)
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_ARCHIVE_DIR", archive)
    (mistakes / "n.md").write_text(
        "# c\n\n**Category:** general\n\n**Severity:** 1\n\n"
        "**Original question:** q\n\n**Wrong answer:** w\n\n"
        "**Correction:** c\n\n**Correct answer:** a\n")
    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    digested, severity = learn.digest_mistakes_to_training(object(), "SYS", boost=2)
    assert digested == 1 and written


# ---- run_training's per-sample weight vector ----

def _corpus(tmp_path, n=3):
    f = tmp_path / "train.jsonl"
    f.write_text("\n".join(json.dumps({"text": f"sample {i}"}) for i in range(n)) + "\n")
    return f


def test_the_unweighted_path_does_not_touch_the_corpus(tmp_path):
    """Omitted weights must leave the run exactly what it was — the corpus is
    not opened, rewritten or restored."""
    from symbio.app import training
    f = _corpus(tmp_path)
    before = f.read_text()
    with training.weighted_corpus(f, None) as handed:
        assert handed is f
        assert f.read_text() == before


def test_weights_expand_the_corpus_for_the_run(tmp_path):
    from symbio.app import training
    f = _corpus(tmp_path, 3)
    with training.weighted_corpus(f, [3, 1, 1]):
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        assert len(lines) == 5
        assert sum("sample 0" in l for l in lines) == 3


def test_the_corpus_is_restored_afterwards(tmp_path):
    from symbio.app import training
    f = _corpus(tmp_path, 2)
    before = f.read_text()
    with training.weighted_corpus(f, [4, 1]):
        assert f.read_text() != before
    assert f.read_text() == before


def test_the_corpus_is_restored_even_when_the_run_raises(tmp_path):
    """A corpus left expanded would be expanded again next run, and the
    weighting would compound silently until one sample was most of the data."""
    from symbio.app import training
    f = _corpus(tmp_path, 2)
    before = f.read_text()
    with pytest.raises(RuntimeError):
        with training.weighted_corpus(f, [5, 1]):
            raise RuntimeError("training died")
    assert f.read_text() == before
    assert not f.with_suffix(f.suffix + ".preweight").exists()


def test_a_zero_weight_still_writes_the_sample_once(tmp_path):
    """Weighting must never be able to delete a lesson."""
    from symbio.app import training
    f = _corpus(tmp_path, 2)
    with training.weighted_corpus(f, [0, 0]):
        assert len([l for l in f.read_text().splitlines() if l.strip()]) == 2


def test_misaligned_weights_are_padded_not_crashed(tmp_path):
    """When golden-remedy samples appear after the weight vector was built,
    weighted_corpus pads short vectors with 1.0 instead of raising."""
    from symbio.app import training

    f = tmp_path / "train.jsonl"
    f.write_text("\n".join(json.dumps({"text": f"sample {i}"})
                           for i in range(5)) + "\n")
    # 3 weights for 5 lines → pad 2 entries with 1.0
    with training.weighted_corpus(f, [3.0, 2.0, 1.0]):
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        assert len(lines) == 8  # 3+2+1+1+1=8 expanded lines
    assert f.read_text().count("sample") == 5  # restored


def test_misaligned_weights_are_truncated_not_crashed(tmp_path):
    """When a weight vector exceeds the corpus, weighted_corpus truncates."""
    from symbio.app import training

    f = tmp_path / "train.jsonl"
    f.write_text("\n".join(json.dumps({"text": f"sample {i}"})
                           for i in range(2)) + "\n")
    with training.weighted_corpus(f, [5.0, 5.0, 5.0]):
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        assert len(lines) == 10  # 5+5 expanded from 2 weights
    assert f.read_text().count("sample") == 2  # restored


def test_run_training_still_accepts_no_weights(monkeypatch):
    """The existing no-weight path must keep working unchanged."""
    from symbio.app import training
    seen = {}
    monkeypatch.setattr(training, "_run_training",
                        lambda *a, **k: seen.setdefault("called", True) or True)
    assert training.run_training({}, iters=5) is True
    assert seen["called"] is True


# ---- (7) the iteration budget scales to the weighted corpus ----------------
#
# A fixed step budget against an expanded corpus gives every sample — the
# boosted ones included — 1/stretch of the updates it would otherwise get,
# which quietly turns "repeat the lesson" into "dilute the corpus". So the run
# stretches the budget before the trainer sees it, then caps it.

def test_scaled_weighted_iters_stretches_the_budget_then_caps():
    from symbio.app import training
    cfg = {"learn": {"max_batch_train_iters": 100}}
    # [3,1,1]: mean(round(w)) = 5/3 -> ceil(5 * 5/3) = 9
    assert training.scaled_weighted_iters(cfg, 5, [3, 1, 1]) == 9
    # an all-3 corpus triples the budget...
    assert training.scaled_weighted_iters(cfg, 25, [3, 3, 3, 3]) == 75
    # ...until the cap, which never sits below the base.
    assert training.scaled_weighted_iters(cfg, 25, [5, 5, 5, 5]) == 100
    assert training.scaled_weighted_iters(cfg, 25, [1, 1]) == 25
    # a trainer-determined schedule is left alone.
    assert training.scaled_weighted_iters(cfg, None, [3, 3]) is None


def test_run_training_scales_iters_to_the_weighted_corpus(tmp_path, monkeypatch):
    """The inner run must see the stretched budget, and the corpus must be
    restored afterwards — the weighting is ephemeral."""
    from symbio.app import training
    f = tmp_path / "train.jsonl"
    f.write_text("\n".join(json.dumps({"text": f"sample {i}"})
                           for i in range(3)) + "\n")
    monkeypatch.setattr(constants, "TRAIN_FILE", f)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    seen = []
    monkeypatch.setattr(training, "_run_training",
                        lambda *a, **k: seen.append(a[1]) or True)
    cfg = {"learn": {"max_batch_train_iters": 100}}
    assert training.run_training(cfg, iters=5, sample_weights=[3, 1, 1]) is True
    assert seen == [9]
    assert len([l for l in f.read_text().splitlines() if l.strip()]) == 3


def test_maybe_train_with_eval_battery_weights_the_batch(tmp_path, monkeypatch):
    """eval_fn -> curriculum.plan decides each note's repeat count; the notes
    are written once, the vector is one weight per corpus line, and train_fn
    receives it (run_training scales the iters from there)."""
    from symbio.app import learn
    mistakes = tmp_path / "mistakes"
    mistakes.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_DIR", mistakes)
    monkeypatch.setattr(constants, "MISTAKES_ARCHIVE_DIR", archive)
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "pre-existing"}) + "\n")
    monkeypatch.setattr(constants, "TRAIN_FILE", corpus)
    # The mistake_threshold floor is 5, so the batch needs five notes.
    for i in range(5):
        (mistakes / f"n{i}.md").write_text(
            "# c\n\n**Category:** general\n\n**Severity:** 1\n\n"
            "**Original question:** why does the browser page not scroll\n\n"
            "**Wrong answer:** it did\n\n**Correction:** c\n\n"
            "**Correct answer:** the page did not scroll down\n")

    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    monkeypatch.setattr(learn.training, "count_samples", lambda role=None: 1)

    seen = {}

    def fake_train(cfg, iters=None, sample_weights=None):
        seen["iters"] = iters
        seen["sample_weights"] = sample_weights
        return True

    config = CONFIG.copy()
    config["learn"] = {
        "enabled": True,
        "mistake_threshold": 5,
        "auto_train": True,
        "boost_factor": 3,
        "batch_train_iters": 25,
        "iters_per_severity": 5,
        "max_batch_train_iters": 100,
        "curriculum_weighting": True,
        "mistake_pretrain_check": False,
        "sample_half_life_days": 21,
    }
    # The eval says a browser-reading case is failing, which is the note's
    # topic — so the plan boosts it over the plain unknown weight.
    eval_result = FakeEval([
        {"id": "browser_read_site", "passed": False, "error": None},
        {"id": "math_product", "passed": True, "error": None},
    ])

    assert learn.maybe_train_on_mistakes(
        config, object(), "SYS", train_fn=fake_train,
        eval_fn=lambda: eval_result) is True

    assert len(written) == 5                      # one copy per note in the file
    assert seen["iters"] == 25                    # run_training scales from here
    weights = seen["sample_weights"]
    assert weights is not None
    assert weights[0] == 1.0                       # the pre-existing corpus line
    assert len(weights) == 1 + 5
    assert all(w >= 1.0 for w in weights[1:])
    assert max(weights[1:]) > 1.0                  # the failing-case boost landed
    assert not list(mistakes.glob("*.md"))         # archived, never re-digested


def test_maybe_train_without_eval_keeps_the_linear_call_shape(tmp_path, monkeypatch):
    """eval_fn=None keeps the historical call shape: train_fn is bound as
    train_fn(config, iters=...) and must not be handed a sample_weights kwarg —
    a stub without **kwargs proves the kwarg is really not passed."""
    from symbio.app import learn
    mistakes = tmp_path / "mistakes"
    mistakes.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_DIR", mistakes)
    monkeypatch.setattr(constants, "MISTAKES_ARCHIVE_DIR", archive)
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "pre-existing"}) + "\n")
    monkeypatch.setattr(constants, "TRAIN_FILE", corpus)
    for i in range(5):
        (mistakes / f"n{i}.md").write_text(
            "# c\n\n**Category:** general\n\n**Severity:** 1\n\n"
            "**Original question:** q{i}\n\n"
            "**Wrong answer:** w\n\n**Correction:** c\n\n"
            "**Correct answer:** a\n")
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: None)
    monkeypatch.setattr(learn.training, "count_samples", lambda role=None: 1)

    seen = {}

    def fake_train(cfg, iters=None):               # no **kwargs on purpose
        seen["iters"] = iters
        return True

    config = CONFIG.copy()
    config["learn"] = {
        "enabled": True, "mistake_threshold": 5, "auto_train": True,
        "boost_factor": 3, "batch_train_iters": 25, "iters_per_severity": 5,
        "max_batch_train_iters": 100, "mistake_pretrain_check": False,
    }
    assert learn.maybe_train_on_mistakes(
        config, object(), "SYS", train_fn=fake_train) is True
    assert seen["iters"] == 25


# ---- the integrity block in the benchmark report ----

def test_the_report_records_a_self_grading_judge():
    from symbio.app import eval as eval_mod
    r = eval_mod._integrity_report({"model_name": "m", "eval": {"judge_model": "m"}}, [])
    assert r["judge_is_independent"] is False
    assert "self-graded" in r["judge_detail"]


def test_the_report_records_an_independent_judge():
    from symbio.app import eval as eval_mod
    r = eval_mod._integrity_report(
        {"model_name": "m", "eval": {"judge_model": "other"}}, [])
    assert r["judge_is_independent"] is True


def test_no_judge_is_not_a_failure():
    """Deterministic-only grading cannot be circular, so it is not a finding."""
    from symbio.app import eval as eval_mod
    r = eval_mod._integrity_report({"model_name": "m"}, [])
    assert r["judge_model"] is None and "cannot be circular" in r["judge_detail"]


def test_the_report_records_a_contaminated_eval_set(tmp_path, monkeypatch):
    from symbio.app import eval as eval_mod
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "What is 17 multiplied by 23?"}))
    monkeypatch.setattr(constants, "TRAIN_FILE", corpus)
    r = eval_mod._integrity_report(CONFIG, _cases())
    assert r["held_out"] is False and "self-reinforcing" in r["held_out_detail"]


def test_a_clean_eval_set_is_recorded_as_held_out(tmp_path, monkeypatch):
    from symbio.app import eval as eval_mod
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "unrelated"}))
    monkeypatch.setattr(constants, "TRAIN_FILE", corpus)
    assert eval_mod._integrity_report(CONFIG, _cases())["held_out"] is True


# ---- auto-train off: decide before spending anything ----

def _batch(tmp_path, monkeypatch, notes=5):
    """Five mistake notes, an isolated corpus, and nothing real to train."""
    from symbio.app import learn

    mistakes = tmp_path / "mistakes"
    mistakes.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setattr(constants, "MISTAKES_DIR", mistakes)
    monkeypatch.setattr(constants, "MISTAKES_ARCHIVE_DIR", archive)
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps({"text": "pre-existing"}) + "\n")
    monkeypatch.setattr(constants, "TRAIN_FILE", corpus)
    for i in range(notes):
        (mistakes / f"n{i}.md").write_text(
            "# c\n\n**Category:** general\n\n**Severity:** 1\n\n"
            "**Original question:** why does the browser page not scroll\n\n"
            "**Wrong answer:** it did\n\n**Correction:** c\n\n"
            "**Correct answer:** the page did not scroll down\n")
    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    monkeypatch.setattr(learn.training, "count_samples", lambda role=None: 1)
    return learn, mistakes, written


def _cfg(**learn_cfg):
    config = CONFIG.copy()
    config["learn"] = {
        "enabled": True, "mistake_threshold": 5,
        "scale_threshold_with_corpus": False,
        "boost_factor": 3, "batch_train_iters": 25, "iters_per_severity": 5,
        "max_batch_train_iters": 100, "curriculum_weighting": True,
        "mistake_pretrain_check": True, "sample_half_life_days": 21,
        **learn_cfg}
    return config


def test_auto_train_off_spends_no_battery(tmp_path, monkeypatch):
    """It used to run the golden battery, run the held-out eval battery, build
    a curriculum plan, digest AND archive the notes — and only then print
    "Auto-train is disabled" and return False. Two full generation batteries
    for a run that was never going to happen."""
    learn, _mistakes, _written = _batch(tmp_path, monkeypatch)
    called = []

    result = learn.maybe_train_on_mistakes(
        _cfg(auto_train=False), object(), "SYS",
        train_fn=lambda *a, **k: called.append("train") or True,
        check_fn=lambda: called.append("check") or (1, 1),
        eval_fn=lambda: called.append("eval") or FakeEval([]))

    assert result is False
    assert called == [], "nothing that only serves an automatic run may run"


def test_auto_train_off_still_leaves_the_corpus_ready_for_train(tmp_path,
                                                                monkeypatch):
    """The user runs /train by hand in this mode, so the material has to be
    waiting when they do."""
    learn, _mistakes, written = _batch(tmp_path, monkeypatch)

    learn.maybe_train_on_mistakes(_cfg(auto_train=False), object(), "SYS",
                                  train_fn=lambda *a, **k: True)

    assert len(written) == 5 * 3, "five notes at boost 3, digested linearly"


def test_auto_train_off_computes_no_weights(tmp_path, monkeypatch):
    """Weights shape a run. There is no run."""
    learn, _mistakes, _written = _batch(tmp_path, monkeypatch)
    seen = {}

    def train(cfg, iters=None, sample_weights=None):
        seen["weights"] = sample_weights
        return True

    learn.maybe_train_on_mistakes(
        _cfg(auto_train=False), object(), "SYS", train_fn=train,
        eval_fn=lambda: FakeEval([{"id": "x", "passed": False, "error": None}]))

    assert seen == {}, "train_fn is not called at all, so nothing is shaped"


def test_auto_train_on_still_runs_the_eval_battery(tmp_path, monkeypatch):
    """The other side of the gate: with it on, nothing was taken away.

    check_fn is absent from `called` and that is correct, not a miss — the
    golden pre-train check only fires on a batch that is ENTIRELY automatic
    tool-error captures, and these notes carry a typed **Correction:**."""
    learn, _mistakes, _written = _batch(tmp_path, monkeypatch)
    called = []

    learn.maybe_train_on_mistakes(
        _cfg(auto_train=True), object(), "SYS",
        train_fn=lambda *a, **k: True,
        check_fn=lambda: called.append("check") or (0, 1),
        eval_fn=lambda: called.append("eval") or FakeEval(
            [{"id": "browser_read_site", "passed": False, "error": None}]))

    assert called == ["eval"]
