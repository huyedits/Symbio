"""Beliefs that know when they were true, the prediction path, and the judge.

The cases are the reactor's, because that is where the gap was found: a belief
can be true and useless at the same time, if it is true at hour 48 and you are
standing at hour 20. Everything here is about telling those two apart, and
about being able to score the operator who correctly did nothing.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from symbio import constants
from symbio.app import belief as B

HOUR = 3600.0


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    return B.BeliefStore()


def xenon(store):
    """The poison: peaks around hour 10, gone by hour 40."""
    return store.add_belief("xenon poisoning is blocking restart",
                            ["flux dropped after scram"],
                            timescale_s=10 * HOUR, dynamics=B.PEAKS)


def recovery(store):
    """The other shape: false early, true from hour 48, and it stays true."""
    return store.add_belief("the reactor can restart", ["xenon decays"],
                            timescale_s=48 * HOUR, dynamics=B.LAGS)


# ---- schema and the success criteria ----

def test_a_new_belief_starts_at_genuine_uncertainty(store):
    b = store.get(store.add_belief("something", ["seen once"]))
    assert b["confidence"] == 0.5
    assert b["version"] == 1 and b["ratified"] is False
    assert b["evidence"] == ["seen once"] and b["kind"] == B.CLAIM


def test_every_spec_field_is_stored(store):
    b = store.get(xenon(store))
    for field in ("id", "hypothesis", "confidence", "evidence", "timescale_s",
                  "dynamics", "valid_until", "soft_rank", "version", "ratified",
                  "created_at", "updated_at"):
        assert field in b, field


def test_a_belief_with_no_horizon_is_stable(store):
    """It cannot be anything else, and guessing a curve would invent a
    transient nothing observed."""
    assert store.get(store.add_belief("the user's name is Huy"))["dynamics"] == B.STABLE


def test_a_belief_with_a_horizon_is_guessed_to_lag(store):
    assert store.get(store.add_belief("x", timescale_s=HOUR))["dynamics"] == B.LAGS


@pytest.mark.parametrize("bad", [{"confidence": 1.4}, {"timescale_s": 0},
                                 {"dynamics": "wobbly"}, {"kind": "opinion"}])
def test_a_malformed_belief_is_refused(store, bad):
    with pytest.raises(ValueError):
        store.add_belief("x", **bad)


def test_an_empty_hypothesis_is_refused(store):
    with pytest.raises(ValueError):
        store.add_belief("   ")


# ---- the timing model, which is the whole point ----

def test_a_peaking_belief_is_strongest_at_its_moment(store):
    row = store.get(xenon(store))
    w = B.BeliefStore.timing_weight
    assert w(row, 10 * HOUR) == pytest.approx(1.0)
    assert w(row, 20 * HOUR) == pytest.approx(0.5)   # one timescale late
    assert w(row, 0) == pytest.approx(0.5)           # one timescale early


def test_a_lagging_belief_does_not_stop_being_true(store):
    """'The reactor can restart' becomes true near hour 48 and stays. Decaying
    it afterwards would model recovery as something you can miss by waiting."""
    row = store.get(recovery(store))
    w = B.BeliefStore.timing_weight
    assert w(row, 20 * HOUR) < 0.7
    assert w(row, 48 * HOUR) == pytest.approx(1.0)
    assert w(row, 96 * HOUR) == pytest.approx(1.0)


def test_a_decaying_belief_is_true_now_and_fades(store):
    row = store.get(store.add_belief("the sensor reading is current", timescale_s=HOUR,
                                     dynamics=B.DECAYS))
    w = B.BeliefStore.timing_weight
    assert w(row, 0) == pytest.approx(1.0)
    assert w(row, 2 * HOUR) == pytest.approx(0.5)


def test_a_stable_belief_has_no_moment_to_miss(store):
    row = store.get(store.add_belief("water is wet"))
    assert B.BeliefStore.timing_weight(row, 99 * HOUR) == 1.0


# ---- requirement 2: score on HOW WELL-TIMED, not merely whether ----

def test_right_at_the_wrong_time_scores_below_right_on_time(store):
    """The single most important behavioural change in the spec. A recovery
    predicted for hour 20 that only happens at hour 48 is PARTIALLY right."""
    early = B.BeliefStore(); late = B.BeliefStore()
    b_early = early.add_belief("recovery", timescale_s=20 * HOUR, dynamics=B.LAGS)
    b_late = late.add_belief("recovery", timescale_s=48 * HOUR, dynamics=B.LAGS)
    # Reality: it happened at hour 48.
    scored_early = early.update_confidence(b_early, True, when=48 * HOUR)
    scored_late = late.update_confidence(b_late, True, when=48 * HOUR)
    assert scored_late["score"] > scored_early["score"]
    assert scored_late["confidence_after"] > scored_early["confidence_after"]


def test_a_well_timed_hit_raises_confidence(store):
    b = recovery(store)
    before = store.get(b)["confidence"]
    store.update_confidence(b, True, when=48 * HOUR)
    assert store.get(b)["confidence"] > before


def test_a_miss_lowers_confidence(store):
    b = recovery(store)
    store.update_confidence(b, True, when=48 * HOUR)
    high = store.get(b)["confidence"]
    store.update_confidence(b, False, when=48 * HOUR)
    assert store.get(b)["confidence"] < high


def test_a_confident_no_is_not_punished_for_being_well_timed(store):
    """Timing measures how well the moment was called. For an outcome that did
    not happen the moment is not what was claimed, so accuracy*timing would
    score a well-calibrated 'no' worst exactly when it is most confidently no."""
    b = store.add_belief("meltdown imminent", timescale_s=HOUR, dynamics=B.LAGS)
    rec = store.update_confidence(b, False, when=HOUR)
    assert rec["score"] > 0.4


def test_the_observation_is_recorded_for_audit(store):
    b = recovery(store)
    store.update_confidence(b, True, when=48 * HOUR)
    [rec] = store.get(b)["outcomes"]
    for k in ("outcome", "elapsed_s", "timing", "accuracy", "score",
              "confidence_before", "confidence_after", "at"):
        assert k in rec, k


def test_scoring_an_unknown_belief_returns_none_and_does_not_raise(store):
    """Called from the turn loop; a stale id must not cost the turn."""
    assert store.update_confidence("nope", True) is None


def test_when_accepts_a_timestamp_as_well_as_seconds(store):
    b = recovery(store)
    created = datetime.fromisoformat(store.get(b)["created_at"])
    rec = store.update_confidence(b, True, when=(created + timedelta(hours=48)).isoformat())
    assert rec["elapsed_s"] == pytest.approx(48 * HOUR, rel=0.01)


# ---- requirement 6: restraint as a first-class, scorable thing ----

def test_a_hold_is_a_row_not_an_absence(store):
    """The reactor's winning operator did LESS. An empty action log cannot tell
    that apart from having had no opinion."""
    h = store.get(store.hold("let the xenon peak pass", timescale_s=40 * HOUR))
    assert h["kind"] == B.HOLD
    assert h["hypothesis"].startswith("holding:")
    assert h["dynamics"] == B.LAGS


def test_correct_inaction_can_be_scored(store):
    h = store.hold("wait for the peak", timescale_s=40 * HOUR)
    rec = store.update_confidence(h, True, when=40 * HOUR)
    assert rec["score"] > 0.9
    assert store.get(h)["confidence"] > 0.5


def test_waiting_can_be_the_top_ranked_move(store):
    xenon(store)
    h = store.hold("wait for the peak", timescale_s=40 * HOUR, soft_rank=0.9)
    store.update_confidence(h, True, when=40 * HOUR)
    defer, why = B.should_defer(store, elapsed=40 * HOUR)
    assert defer and "wait" in why


def test_with_no_beliefs_nothing_is_deferred(store):
    assert B.should_defer(store) == (False, "no beliefs")


# ---- requirement 3: soft ranking ----

def test_ranking_is_by_confidence_times_soft_rank(store):
    low = store.add_belief("minor", confidence=0.9, soft_rank=0.1)
    high = store.add_belief("major", confidence=0.5, soft_rank=0.9)
    assert [b["id"] for b in store.rank_beliefs()][0] == high
    assert low in [b["id"] for b in store.rank_beliefs()]  # ordered, not filtered


def test_ranking_reads_confidence_through_the_curve(store):
    """At hour 20 the hour-48 belief must not outrank a live one, however sure
    it was when written."""
    xe = xenon(store)
    recovery(store)
    assert store.rank_beliefs(elapsed=12 * HOUR)[0]["id"] == xe


def test_ratification_breaks_a_tie_without_overruling_evidence(store):
    plain = store.add_belief("a", confidence=0.5, soft_rank=0.5)
    approved = store.add_belief("b", confidence=0.5, soft_rank=0.5)
    store.ratify(approved)
    assert store.rank_beliefs()[0]["id"] == approved
    stronger = store.add_belief("c", confidence=0.9, soft_rank=0.9)
    assert store.rank_beliefs()[0]["id"] == stronger  # evidence still wins


# ---- requirement 5: ratified versioning ----

def test_ratifying_bumps_the_version(store):
    b = xenon(store)
    assert store.get(b)["version"] == 1
    store.ratify(b)
    row = store.get(b)
    assert row["ratified"] is True and row["version"] == 2


def test_ratifying_the_wrong_version_is_refused(store):
    """Approval is a fact about a version, not about an id."""
    b = xenon(store)
    assert store.ratify(b, version=99) is None
    assert store.get(b)["ratified"] is False


def test_editing_a_ratified_belief_un_ratifies_it(store):
    """The user approved what it said, not its id — an unreviewed edit must not
    inherit the authority of a reviewed one."""
    b = xenon(store)
    store.ratify(b)
    store.revise(b, hypothesis="something else entirely")
    assert store.get(b)["ratified"] is False


def test_revise_refuses_unknown_fields(store):
    with pytest.raises(ValueError):
        store.revise(xenon(store), version=99)


def test_ratifying_an_unknown_belief_returns_none(store):
    assert store.ratify("nope") is None


# ---- requirement 4: expiry ----

def test_a_belief_past_valid_until_is_archived_and_stops_ranking(store):
    """The xenon insight: a belief true NOW may be false LATER, and stale
    beliefs poison decisions."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    b = store.add_belief("the pit is deep", valid_until=past)
    assert b in [x["id"] for x in store.rank_beliefs()]
    assert store.expire() == [b]
    assert b not in [x["id"] for x in store.rank_beliefs()]


def test_expiry_archives_rather_than_deletes(store):
    """An expired belief is the evidence that its window was called right."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    b = store.add_belief("x", valid_until=past)
    store.expire()
    assert store.get(b) is not None
    assert store.get(b)["archived"] is True
    assert b in [r["id"] for r in store.all(include_archived=True)]


def test_a_permanent_belief_never_expires(store):
    b = store.add_belief("the user's name is Huy")
    assert store.expire() == []
    assert b in [x["id"] for x in store.rank_beliefs()]


def test_a_future_deadline_has_not_expired_yet(store):
    future = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    b = store.add_belief("x", valid_until=future)
    assert store.expire() == []
    assert store.expire(datetime.now(timezone.utc) + timedelta(hours=6)) == [b]


# ---- (B) the feature generator ----

def test_a_well_timed_prediction_becomes_a_training_sample(store):
    b = recovery(store)
    store.update_confidence(b, True, when=48 * HOUR)
    [s] = store.training_samples()
    assert s["kind"] == B.CLAIM and s["score"] > 0.6
    assert "2 days" in s["reply"]


def test_correct_restraint_becomes_a_training_sample(store):
    """The sample the mistake loop can never generate: a turn where the right
    move was to do nothing produces no mistake, so it produces no data."""
    h = store.hold("wait for the peak", timescale_s=40 * HOUR,
                   evidence=["xenon still rising"])
    store.update_confidence(h, True, when=40 * HOUR)
    [s] = store.training_samples()
    assert s["kind"] == B.HOLD
    assert "wait" in s["reply"].lower()


def test_a_badly_timed_prediction_is_not_taught(store):
    b = store.add_belief("recovery", timescale_s=1 * HOUR, dynamics=B.LAGS)
    store.update_confidence(b, True, when=100 * HOUR)
    assert store.training_samples() == []


def test_an_unobserved_belief_is_not_taught(store):
    recovery(store)
    assert store.training_samples() == []


def test_samples_come_back_best_first(store):
    good = recovery(store)
    store.update_confidence(good, True, when=48 * HOUR)
    meh = store.add_belief("other", timescale_s=10 * HOUR, dynamics=B.LAGS)
    store.update_confidence(meh, True, when=16 * HOUR)
    scores = [s["score"] for s in store.training_samples(min_score=0.0)]
    assert scores == sorted(scores, reverse=True)


# ---- persistence ----

def test_beliefs_survive_a_new_store_object(store):
    xenon(store)
    assert B.BeliefStore().all()[0]["hypothesis"].startswith("xenon")


def test_the_db_path_follows_a_redirected_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "a")
    B.BeliefStore().add_belief("in a")
    assert (tmp_path / "a" / B.BELIEF_DB_NAME).exists()
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "b")
    assert B.BeliefStore().all() == []


def test_a_corrupt_json_cell_does_not_take_the_row_down(store):
    import sqlite3
    b = xenon(store)
    conn = sqlite3.connect(str(store.path))
    conn.execute("UPDATE beliefs SET evidence='{truncated' WHERE id=?", (b,))
    conn.commit(); conn.close()
    row = store.get(b)
    assert row["evidence"] == [] and row["hypothesis"].startswith("xenon")


# ---- the learn.py wire-in ----

def test_the_prediction_path_records_and_scores(store, monkeypatch):
    from symbio.app import learn
    bid = learn.record_prediction("recovery", ["xenon decays"],
                                  timescale_s=48 * HOUR, store=store)
    rec = learn.resolve_prediction(bid, True, when=48 * HOUR, store=store)
    assert rec["score"] > 0.9


def test_digesting_predictions_appends_to_the_corpus(store, monkeypatch):
    from symbio.app import learn
    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    b = learn.record_prediction("recovery", ["x"], timescale_s=48 * HOUR, store=store)
    learn.resolve_prediction(b, True, when=48 * HOUR, store=store)
    added = learn.digest_predictions_to_training(object(), "SYS", store=store)
    assert added == 1 and written


def test_a_better_timed_sample_is_repeated_harder(store, monkeypatch):
    """Boosting a day-early hit as hard as an on-time one would teach the
    corpus that timing does not matter."""
    from symbio.app import learn
    written = []
    monkeypatch.setattr(learn.training, "append_chat_pair",
                        lambda u, a, tok, sp, **kw: written.append((u, a)))
    good = store.add_belief("on time", timescale_s=48 * HOUR, dynamics=B.LAGS)
    store.update_confidence(good, True, when=48 * HOUR)
    learn.digest_predictions_to_training(object(), "SYS", boost=4, store=store)
    on_time_repeats = len(written)

    written.clear()
    store2 = B.BeliefStore(store.path.with_name("other.db"))
    meh = store2.add_belief("early", timescale_s=12 * HOUR, dynamics=B.LAGS)
    store2.update_confidence(meh, True, when=20 * HOUR)
    learn.digest_predictions_to_training(object(), "SYS", boost=4, store=store2)
    assert len(written) < on_time_repeats


# ---- the golden.py judge tier ----

def _case(**kw):
    from symbio.app import golden
    return golden.GoldenCase(id="c1", description="d", prompt_fn=lambda cfg: "q",
                             check=lambda *a: True, **kw)


def test_a_judge_returns_a_score_and_a_rationale():
    from symbio.app import golden
    v = golden.judge_case(_case(subjective=True), "q", "a",
                          lambda p: "correctness: 8\nreasoning: 7\ntiming: 9\n"
                                    "restraint: 10\nIt waited correctly.",
                          judge_model="judge-model", model_under_test="tested")
    assert v.score == pytest.approx((8 + 7 + 9 + 10) / 40)
    assert "waited correctly" in v.rationale
    assert v.passed


def test_self_grading_is_refused_outright():
    """A model grading its own reply agrees with itself, and the number it
    produces looks like evidence while measuring nothing."""
    from symbio.app import golden
    with pytest.raises(ValueError):
        golden.judge_case(_case(subjective=True), "q", "a", lambda p: "correctness: 9",
                          judge_model="same", model_under_test="same")


def test_an_unparseable_verdict_is_none_not_zero():
    """'The grader broke' and 'the model was bad' are different facts, and
    scoring the second when the first happened rolls back a good adapter."""
    from symbio.app import golden
    assert golden.judge_case(_case(subjective=True), "q", "a",
                             lambda p: "I would rather not.", "j", "m") is None


def test_a_judge_that_raises_is_none_not_zero():
    from symbio.app import golden
    def boom(prompt):
        raise RuntimeError("model unloaded")
    assert golden.judge_case(_case(subjective=True), "q", "a", boom, "j", "m") is None


def test_only_subjective_cases_reach_the_judge():
    from symbio.app import golden
    result = golden.GoldenResult(results={"c1": True}, replies={"c1": "a"})
    calls = []
    def judge(p):
        calls.append(p); return "correctness: 9"
    golden.judge_results(result, [_case(subjective=False)], {}, judge, "j", "m", save=False)
    assert calls == []
    golden.judge_results(result, [_case(subjective=True)], {}, judge, "j", "m", save=False)
    assert len(calls) == 1


def test_verdicts_are_written_down_for_audit(tmp_path, monkeypatch):
    """A number nobody can argue with later is not evidence, and these decide
    whether a fine-tune is kept."""
    from symbio.app import golden
    monkeypatch.setattr(constants, "LOG_DIR", tmp_path / "logs")
    result = golden.GoldenResult(results={"c1": True}, replies={"c1": "a"})
    golden.judge_results(result, [_case(subjective=True)], {},
                         lambda p: "correctness: 8\ntiming: 6\nbecause", "j", "m")
    line = json.loads((tmp_path / "logs" / "judge_verdicts.jsonl").read_text().strip())
    assert line["case_id"] == "c1" and line["judge_model"] == "j"
    assert "because" in line["rationale"]


def test_the_rubric_grades_timing_and_restraint():
    """The two things no substring check can measure."""
    from symbio.app import golden
    assert "TIMING" in golden.JUDGE_RUBRIC and "RESTRAINT" in golden.JUDGE_RUBRIC


def test_a_case_file_can_flag_a_case_subjective(tmp_path, monkeypatch):
    from symbio.app import golden
    f = tmp_path / "golden_cases.json"
    f.write_text(json.dumps({"c": {"description": "d", "prompt": "p",
                                   "requirements": [], "subjective": True,
                                   "rubric": "Grade the waiting."}}))
    monkeypatch.setattr(constants, "GOLDEN_CASES_FILE", f)
    [case] = golden.load_user_golden_cases()
    assert case.subjective is True and "waiting" in case.rubric


def test_an_ordinary_case_is_not_subjective_by_default(tmp_path, monkeypatch):
    from symbio.app import golden
    f = tmp_path / "golden_cases.json"
    f.write_text(json.dumps({"c": {"description": "d", "prompt": "p",
                                   "requirements": []}}))
    monkeypatch.setattr(constants, "GOLDEN_CASES_FILE", f)
    assert golden.load_user_golden_cases()[0].subjective is False
