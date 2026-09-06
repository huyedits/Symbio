"""Spend the LoRA budget where the model is currently weak.

The mistake loop decides WHEN to train — enough corrections have piled up. It
has no opinion on WHAT to train on: every note is worth the same, forever,
regardless of whether the model still gets that case wrong. So a fine-tune
spends most of its budget re-teaching things already learned, and the corpus
records the model's history rather than its present state.

This module is the other half. Before a run it asks the held-out eval set what
the model actually fails at right now, weights the corpus toward those, ages
out samples whose lesson has been absorbed, and adds a third source the
mistake loop cannot see: predictions whose confidence diverged from reality
without anyone ever typing a correction.

Four things make the weighting trustworthy rather than a feedback loop:

  * the eval set stays HELD OUT. `assert_held_out` refuses to run if any eval
    prompt has leaked into the corpus, because a curriculum steered by a test
    the model was trained on is steered by nothing.
  * a judge, where one is used, is a different model from the one being tuned.
    A model grading itself agrees with itself; the resulting weights would
    amplify its own blind spots.
  * weights are bounded. A case the model fails is worth more, not
    unboundedly more — an unbounded weight lets one stubborn case consume a
    whole run and quietly turn a general adapter into a single-trick one.
  * the floor is preserved. Every sample the mistake loop would have written
    is still written; this only changes how often each is repeated. Weighting
    can starve a topic, and it must never be able to delete one.

Nothing here trains or grades. It reads an EvalResult, reads the belief store,
and returns weights and samples for learn.py to hand to the existing
append_chat_pair / run_training path.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from symbio import constants
from symbio.app import belief

# How far weighting may move a sample, in repeats. A failing case is worth
# more than a passing one; it is not worth everything. Measured intent: with
# WEAK=3.0 and STRONG=0.35 a run puts roughly ten times the budget on what is
# broken, which still leaves the passing material present often enough to hold.
WEIGHT_WEAK = 3.0
WEIGHT_STRONG = 0.35
WEIGHT_UNKNOWN = 1.0

# Samples older than this, whose lesson the model now passes, are dropped
# rather than merely down-weighted. A correction from three weeks ago that the
# model handles correctly is not evidence about the model; it is history.
DEFAULT_HALF_LIFE_DAYS = 21.0

# Words that carry no topic signal, so overlap with them means nothing. Kept
# short and generic on purpose: a long hand-tuned list is a place for a silent
# bias about which topics count to accumulate.
_STOP = frozenset("""
a an the and or but if then than that this these those is are was were be been
being do does did doing have has had of to in on for with at by from as it its
what which who whom when where why how you your my me i we our us they them
please can could should would will shall may might must not no yes
""".split())


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(w) > 2 and w not in _STOP}


# ---- (2) what is the model currently bad at? -------------------------------

def weakness_profile(eval_result: Any) -> dict[str, float]:
    """Map each eval case id to a training weight from its latest result.

    Failing cases weigh more, passing cases less. Cases that errored rather
    than failed are treated as unknown, not weak: a generation that crashed
    says nothing about the model's grasp of the topic, and boosting on it
    would let an unrelated fault redirect a whole training run.
    """
    tasks = getattr(eval_result, "tasks", None) or []
    profile: dict[str, float] = {}
    for task in tasks:
        case_id = task.get("id")
        if not case_id:
            continue
        if task.get("error"):
            profile[case_id] = WEIGHT_UNKNOWN
        else:
            profile[case_id] = WEIGHT_WEAK if not task.get("passed") else WEIGHT_STRONG
    return profile


def case_topics(cases: Iterable[Any], config: dict[str, Any]) -> dict[str, set[str]]:
    """The vocabulary of each eval case: its id, description and prompt.

    This is what a training sample is matched against, and it is the honest
    weak point of the whole approach — term overlap is a proxy for "is this
    sample about that case", not a measurement of it. It is used only to
    decide how often to repeat a sample that is being written either way, so a
    bad match costs some budget; it cannot teach anything false.
    """
    topics: dict[str, set[str]] = {}
    for case in cases:
        text = f"{case.id} {getattr(case, 'description', '')}"
        try:
            text += " " + case.prompt_fn(config)
        except Exception:
            # A prompt_fn that needs config keys this caller lacks must not
            # take the profile down; the id and description still carry signal.
            pass
        topics[case.id] = _terms(text.replace("_", " "))
    return topics


def sample_weight(text: str, profile: dict[str, float],
                  topics: dict[str, set[str]]) -> float:
    """How hard to push one training sample, given what the model fails at.

    Weighted by how strongly the sample overlaps each case's vocabulary, so a
    sample matching nothing lands on WEIGHT_UNKNOWN rather than being dropped
    or boosted. Silence about a topic is not evidence either way.
    """
    words = _terms(text)
    if not words or not profile:
        return WEIGHT_UNKNOWN
    weighted, total = 0.0, 0.0
    for case_id, weight in profile.items():
        vocab = topics.get(case_id)
        if not vocab:
            continue
        overlap = len(words & vocab) / max(1, len(vocab))
        if overlap <= 0:
            continue
        weighted += weight * overlap
        total += overlap
    if total <= 0:
        return WEIGHT_UNKNOWN
    return max(WEIGHT_STRONG, min(WEIGHT_WEAK, weighted / total))


# ---- (3) age out what the model has already absorbed -----------------------

def age_weight(created_at: str | datetime | None,
               half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
               now: datetime | None = None) -> float:
    """Decay a sample by how long ago it was written, halving per half-life.

    An unreadable or missing timestamp weighs 1.0 rather than 0: a sample
    whose age cannot be established is not thereby known to be stale, and
    decaying it would silently delete the oldest material — exactly the notes
    least likely to still carry a date.
    """
    if created_at is None:
        return 1.0
    when = created_at
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when)
        except (TypeError, ValueError):
            return 1.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    days = max(0.0, (moment - when).total_seconds() / 86400.0)
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (days / half_life_days)


def is_stale(text: str, created_at: str | datetime | None,
             profile: dict[str, float], topics: dict[str, set[str]],
             half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
             now: datetime | None = None, floor: float = 0.25) -> bool:
    """Has this sample's lesson been absorbed AND gone cold?

    Both, deliberately. Age alone is not staleness — a three-week-old note
    about something the model still gets wrong is the most valuable sample in
    the corpus, and dropping it on a timer would remove exactly the lessons
    that never landed. So a sample is only stale when the model currently
    PASSES the cases it relates to and it has decayed past the floor.
    """
    if sample_weight(text, profile, topics) > WEIGHT_STRONG:
        return False
    return age_weight(created_at, half_life_days, now) < floor


# ---- (1) predictions that diverged from reality ----------------------------

def diverged_beliefs(store: belief.BeliefStore, threshold: float = 0.35,
                     min_confidence: float = 0.6) -> list[dict[str, Any]]:
    """Beliefs the model was confident about and reality disagreed with.

    A training signal the mistake loop structurally cannot produce: nobody
    typed a correction, no tool failed, the turn looked fine — the model was
    simply sure of something that did not hold. `min_confidence` is what makes
    it a divergence rather than noise: being wrong about something you were
    50/50 on is not a lesson, it is a coin landing.

    Returns one record per belief, carrying the scored observation that
    diverged, so a caller can teach the corrected expectation.
    """
    out: list[dict[str, Any]] = []
    for row in store.all(include_archived=True):
        outcomes = row.get("outcomes") or []
        if not outcomes:
            continue
        latest = outcomes[-1]
        before = latest.get("confidence_before", 0.0)
        score = latest.get("score", 0.0)
        if before < min_confidence:
            continue
        if before - score < threshold:
            continue
        out.append({"belief": row, "observation": latest,
                    "divergence": before - score})
    return sorted(out, key=lambda r: -r["divergence"])


def divergence_samples(store: belief.BeliefStore, threshold: float = 0.35,
                       min_confidence: float = 0.6) -> list[dict[str, Any]]:
    """Diverged beliefs rendered as prompt/reply pairs for the corpus.

    The reply states the corrected expectation AND why the original was wrong,
    including whether it was wrong about the fact or only about the timing.
    Teaching only "it was false" would throw away the more useful half: a
    prediction that was right about what and wrong about when needs its timing
    corrected, not its content.
    """
    samples: list[dict[str, Any]] = []
    for record in diverged_beliefs(store, threshold, min_confidence):
        row, obs = record["belief"], record["observation"]
        held = obs.get("outcome")
        timing = obs.get("timing", 1.0)
        evidence = "; ".join(row.get("evidence") or []) or "the situation as described"
        prompt = (f"Given: {evidence}. What do you expect, and when? "
                  f"State how confident you are.")
        if held and timing < 0.5:
            # Right about what, wrong about when: the case the mistake loop
            # would have scored as a plain success.
            reply = (f"{row['hypothesis']} — but not on the timescale I would "
                     f"have given. It held about "
                     f"{belief._human_span(obs.get('elapsed_s'))} in, not "
                     f"{belief._human_span(row.get('timescale_s'))}. I should "
                     f"be less confident about the timing than about the claim.")
        else:
            reply = (f"I would not expect that with confidence. "
                     f"{row['hypothesis']} did not hold, and I was more sure of "
                     f"it than the evidence supported.")
        samples.append({"belief_id": row["id"], "divergence": record["divergence"],
                        "prompt": prompt, "reply": reply})
    return samples


# ---- (4) the guards --------------------------------------------------------

class HeldOutViolation(RuntimeError):
    """The eval set leaked into the training corpus."""


def eval_prompts(cases: Iterable[Any], config: dict[str, Any]) -> list[str]:
    prompts_out = []
    for case in cases:
        try:
            prompts_out.append(case.prompt_fn(config))
        except Exception:
            continue
    return prompts_out


def assert_held_out(cases: Iterable[Any], config: dict[str, Any],
                    corpus_path=None) -> None:
    """Refuse to weight a run if an eval prompt is in the training corpus.

    This is the load-bearing guard. Adaptive weighting reads the eval set to
    decide where the model is weak; if the model was trained on that set, the
    signal measures memorisation and the curriculum steers itself in a circle,
    with every pass looking like improvement. Raising here is correct — a
    silent fallback to unweighted training would hide the leak, and the leak
    is a corpus bug that wants fixing, not routing around.
    """
    path = corpus_path or constants.TRAIN_FILE
    try:
        corpus = path.read_text(encoding="utf-8")
    except OSError:
        return  # No corpus yet is not a leak.
    leaked = [p for p in eval_prompts(cases, config) if p and p.strip() in corpus]
    if leaked:
        raise HeldOutViolation(
            f"{len(leaked)} held-out eval prompt(s) are in {path}: "
            f"{leaked[0][:70]!r}. The eval set drives sample weighting, so "
            f"training on it makes the curriculum self-reinforcing.")


def assert_independent_judge(judge_model: str, model_under_test: str) -> None:
    """A model may not grade the work that decides its own training weights."""
    if judge_model and model_under_test and judge_model == model_under_test:
        raise ValueError(
            f"judge and model under test are both {judge_model!r}; adaptive "
            f"weighting driven by a self-graded score amplifies blind spots")


# ---- putting it together ---------------------------------------------------

def plan(eval_result: Any, cases: Iterable[Any], config: dict[str, Any],
         samples: list[dict[str, Any]], *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
         base_boost: int = 1, now: datetime | None = None) -> list[dict[str, Any]]:
    """Decide how many times each sample is written. Never fewer than once.

    `samples` are dicts with at least "text", optionally "created_at". The
    floor of one repeat is requirement 5 in arithmetic: the mistake loop's
    material is all still written, and weighting only changes emphasis. A
    weighting scheme able to drop a sample to zero is a weighting scheme able
    to silently delete a lesson, and the first time it is wrong about which
    lesson, the correction is gone.
    """
    cases = list(cases)
    profile = weakness_profile(eval_result)
    topics = case_topics(cases, config)
    planned: list[dict[str, Any]] = []
    for sample in samples:
        text = sample.get("text", "")
        weight = sample_weight(text, profile, topics)
        age = age_weight(sample.get("created_at"), half_life_days, now)
        stale = is_stale(text, sample.get("created_at"), profile, topics,
                         half_life_days, now)
        repeats = max(1, int(round(base_boost * weight * max(age, 0.5))))
        planned.append({**sample, "weight": weight, "age_weight": age,
                        "stale": stale, "repeats": 1 if stale else repeats})
    return planned


def summarise(planned: list[dict[str, Any]]) -> str:
    """One line for the operator: where this run's budget actually went."""
    if not planned:
        return "  [Curriculum] nothing to weight."
    total = sum(p["repeats"] for p in planned)
    weak = sum(1 for p in planned if p["weight"] > 1.5)
    stale = sum(1 for p in planned if p["stale"])
    return (f"  [Curriculum] {len(planned)} sample(s) -> {total} repeat(s); "
            f"{weak} boosted for measured weakness, {stale} aged down to one.")
