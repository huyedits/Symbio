"""Beliefs that know when they were true — store, feature generator, ranking.

Three jobs in one module on purpose, because they are one thing seen from
three sides: what the agent currently bets on (A), the material that teaches
it to bet better (B), and the ordering that says which bet matters most when
two disagree (C). Splitting them would mean three files sharing one schema
and drifting apart.

So every belief carries `timescale_s` and a `dynamics` shape, and
`update_confidence` scores a prediction on *when* it landed, not merely
whether it eventually held. A recovery predicted for hour 20 that arrives at
hour 48 is partially right and scores below one predicted for hour 48. That
is a different error metric from accuracy, and it is the one a transient
punishes.

`hold()` is the other half. The controller that survives an iodine pit is the
one that *waits* — it does less, against a rising poison, while every
instinct says pull the rods. Restraint recorded as a belief is a decision
that can be graded later; restraint left as the absence of an action is
indistinguishable from having had no opinion, and can never become training
data. `ratified` and `version` make user approval a fact about a specific
version of a belief rather than a mood.

Deliberately not a decision engine. Nothing here chooses or acts. It records
what is believed, scores how well-timed those beliefs turned out to be, and
hands the good ones to training. safety.py still owns the hard veto; this is
the soft ranking underneath it, and the two must not be confused: ranking
orders, it never forbids.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from symbio import constants

BELIEF_DB_NAME = "beliefs.db"

# How a belief's truth moves with time. The reactor supplies all four, which
# is why a single decay curve is not enough:
#   stable — timing is not part of the claim ("the user's name is Huy")
#   peaks  — false, true around the moment, false again (xenon concentration)
#   decays — true now, less true as time passes (a fresh sensor reading)
#   lags   — false now, true from the moment onward (recovery is reachable)
STABLE = "stable"
PEAKS = "peaks"
DECAYS = "decays"
LAGS = "lags"
DYNAMICS = (STABLE, PEAKS, DECAYS, LAGS)

# What a row represents. A held decision is a belief about the *right action*
# ("waiting is correct here"), and it has to be distinguishable from a claim
# about the world so training can draw restraint samples specifically. Not in
# the original schema sketch; added because "score correct inaction" is
# impossible without being able to find the inaction rows.
CLAIM = "claim"
HOLD = "hold"


def belief_db() -> Path:
    """Where beliefs live, resolved per call rather than bound at import.

    Same reason as pending.py: reading constants.LOG_DIR once at module load
    happens before a test — or anything else redirecting the project's
    directories — can point it elsewhere, and a store that ignores the
    redirection writes test beliefs into the operator's real ones.
    """
    return constants.LOG_DIR / BELIEF_DB_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    # Rows written before this module stamped a zone, or by a caller passing a
    # naive string, would raise on every comparison with an aware `now`.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class BeliefStore:
    """SQLite-backed beliefs: the store, the ranking, and the training feed."""

    def __init__(self, path: Path | None = None):
        # Resolved now if not given, so a caller may still point one store at a
        # scratch file the way SessionStore does.
        self.path = Path(path) if path else belief_db()
        # More than one writer in-process (chat turns, the training thread),
        # and SQLite's own locking returns "database is locked" rather than
        # waiting politely. Cross-process races are out of scope, as in
        # pending.py: a duplicate is recoverable, a lost write is not.
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS beliefs ("
                "id TEXT PRIMARY KEY,"
                "hypothesis TEXT NOT NULL,"
                "confidence REAL NOT NULL,"
                "evidence TEXT,"          # JSON list
                "timescale_s REAL,"       # how far out the claim reaches
                "dynamics TEXT,"          # stable | peaks | decays | lags
                "valid_until TEXT,"       # NULL = permanent
                "soft_rank REAL NOT NULL,"
                "version INTEGER NOT NULL,"
                "ratified INTEGER NOT NULL,"
                "kind TEXT NOT NULL,"     # claim | hold
                "outcomes TEXT,"          # JSON list of scored observations
                "archived INTEGER NOT NULL DEFAULT 0,"
                "created_at TEXT,"
                "updated_at TEXT"
                ")"
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    # ---- writing ----

    def add_belief(self, hypothesis: str, evidence: Iterable[str] = (),
                   *, confidence: float = 0.5, timescale_s: float | None = None,
                   dynamics: str | None = None, valid_until: str | None = None,
                   soft_rank: float = 0.5, kind: str = CLAIM) -> str:
        """Record a new belief at 0.5 — genuine uncertainty, not a guess
        dressed as knowledge — and return its id.

        `dynamics` is inferred from what it was given when not stated: a claim
        with no timescale cannot be anything but stable, and one with a horizon
        is guessed to lag, because the beliefs that carry a horizon are almost
        always predictions about a moment that has not arrived yet. A guess is
        recorded rather than left null so `update_confidence` always has a
        curve to score against; every one of them is correctable.
        """
        if not (hypothesis or "").strip():
            raise ValueError("a belief needs a hypothesis")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        if timescale_s is not None and timescale_s <= 0:
            raise ValueError("timescale_s must be positive when given")
        if dynamics is None:
            dynamics = STABLE if timescale_s is None else LAGS
        if dynamics not in DYNAMICS:
            raise ValueError(f"dynamics must be one of {DYNAMICS}, got {dynamics!r}")
        if kind not in (CLAIM, HOLD):
            raise ValueError(f"kind must be {CLAIM!r} or {HOLD!r}, got {kind!r}")

        belief_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO beliefs (id, hypothesis, confidence, evidence,"
                    " timescale_s, dynamics, valid_until, soft_rank, version,"
                    " ratified, kind, outcomes, archived, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,1,0,?,?,0,?,?)",
                    (belief_id, hypothesis.strip(), float(confidence),
                     json.dumps([str(e) for e in evidence]), timescale_s, dynamics,
                     valid_until, float(soft_rank), kind, json.dumps([]), now, now))
                conn.commit()
            finally:
                conn.close()
        return belief_id

    def hold(self, reason: str, *, timescale_s: float | None = None,
             evidence: Iterable[str] = (), soft_rank: float = 0.5) -> str:
        """Record a decision to wait as a belief that can be graded later.

        The reactor's lesson in one function. An operator hammering the rods
        against a rising xenon peak and one who waits produce the same empty
        action log, and only the second is right — so restraint has to leave a
        row behind, or there is nothing for `update_confidence` to score and
        nothing for training to learn from. Dynamics is `lags` because that is
        what a hold asserts: not yet, but later.
        """
        return self.add_belief(
            f"holding: {reason.strip()}", evidence, confidence=0.5,
            timescale_s=timescale_s, dynamics=LAGS, soft_rank=soft_rank, kind=HOLD)

    def update_confidence(self, belief_id: str, outcome: bool,
                          when: float | str | None = None,
                          *, learning_rate: float = 0.4) -> dict[str, Any] | None:
        """Score an observation against a belief and move its confidence.

        The one behavioural change everything else rests on: the score is
        `accuracy x timing`, not accuracy. A recovery predicted for hour 20
        that only arrives at hour 48 held, so accuracy is 1.0 — but it was
        asserted a day and a half early, so timing is small and the belief is
        credited accordingly. Being right eventually and being right on time
        are different achievements and now score differently.

        `when` is seconds since the belief was created, or an ISO timestamp.
        Omitted, it means now. Returns the scored record, or None if there is
        no such belief — never raises on a missing id, because this is called
        from the turn loop where a stale id must not cost the turn.
        """
        row = self.get(belief_id)
        if row is None:
            return None
        elapsed = self._elapsed(row, when)
        timing = self.prediction_timing(row, elapsed)
        accuracy = 1.0 if outcome else 0.0
        # A belief that correctly says "not yet" is not punished by timing:
        # timing measures how well the moment was called, and for an outcome
        # that did NOT happen the moment is not what was being claimed. Scoring
        # it as accuracy*timing would mean a well-calibrated "no" scores worst
        # exactly when it is most confidently no.
        score = accuracy * timing if outcome else 1.0 - (row["confidence"] * timing)
        confidence = row["confidence"] + learning_rate * (score - row["confidence"])
        confidence = min(1.0, max(0.0, confidence))

        record = {"outcome": bool(outcome), "elapsed_s": elapsed, "timing": timing,
                  "accuracy": accuracy, "score": score,
                  "confidence_before": row["confidence"], "confidence_after": confidence,
                  "at": _now()}
        outcomes = list(row["outcomes"]) + [record]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE beliefs SET confidence=?, outcomes=?, updated_at=?"
                    " WHERE id=?",
                    (confidence, json.dumps(outcomes), _now(), belief_id))
                conn.commit()
            finally:
                conn.close()
        return record

    def ratify(self, belief_id: str, version: int | None = None) -> dict[str, Any] | None:
        """Mark a belief user-approved, at a specific version.

        Approval is a fact about a version, not about a hypothesis: a belief
        the user ratified at version 3 and that has since been rewritten is not
        approved any more, and treating it as approved is how an unreviewed
        edit inherits the authority of a reviewed one. A mismatched `version`
        is refused rather than silently re-approving whatever is there now.
        """
        row = self.get(belief_id)
        if row is None:
            return None
        if version is not None and version != row["version"]:
            return None
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE beliefs SET ratified=1, version=version+1, updated_at=?"
                    " WHERE id=?", (_now(), belief_id))
                conn.commit()
            finally:
                conn.close()
        return self.get(belief_id)

    def revise(self, belief_id: str, **fields: Any) -> dict[str, Any] | None:
        """Change a belief's content, bumping version and dropping ratification.

        Editing a ratified belief un-ratifies it, for the reason above: the
        user approved what it said, not its id.
        """
        allowed = {"hypothesis", "confidence", "evidence", "timescale_s",
                   "dynamics", "valid_until", "soft_rank"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"cannot revise {sorted(unknown)}")
        row = self.get(belief_id)
        if row is None:
            return None
        sets, values = [], []
        for key, value in fields.items():
            sets.append(f"{key}=?")
            values.append(json.dumps([str(e) for e in value])
                          if key == "evidence" else value)
        if not sets:
            return row
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"UPDATE beliefs SET {', '.join(sets)}, version=version+1,"
                    f" ratified=0, updated_at=? WHERE id=?",
                    (*values, _now(), belief_id))
                conn.commit()
            finally:
                conn.close()
        return self.get(belief_id)

    def expire(self, now: datetime | None = None) -> list[str]:
        """Archive every belief past its valid_until. Returns their ids.

        Archived rather than deleted: a belief that expired is the evidence
        that its window was called correctly, and deleting it throws away the
        only record of a prediction that came good. Ranking and the training
        feed both skip archived rows, so an expired belief stops steering
        decisions the moment it lapses — which is the whole point. A stale
        belief left in the ranking is the hour-48 answer being handed to an
        hour-20 question.
        """
        moment = now or datetime.now(timezone.utc)
        expired = [row["id"] for row in self.all()
                   if (d := _parse(row["valid_until"])) is not None and d <= moment]
        if not expired:
            return []
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "UPDATE beliefs SET archived=1, updated_at=? WHERE id=?",
                    [(_now(), i) for i in expired])
                conn.commit()
            finally:
                conn.close()
        return expired

    # ---- reading and ranking ----

    def get(self, belief_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM beliefs WHERE id=?",
                               (belief_id,)).fetchone()
        finally:
            conn.close()
        return _row_to_dict(row) if row else None

    def all(self, include_archived: bool = False) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM beliefs"
            if not include_archived:
                sql += " WHERE archived=0"
            rows = conn.execute(sql + " ORDER BY created_at").fetchall()
        finally:
            conn.close()
        return [_row_to_dict(r) for r in rows]

    def rank_beliefs(self, elapsed: float | None = None,
                     kind: str | None = None) -> list[dict[str, Any]]:
        """Live beliefs, strongest first, by confidence x soft_rank.

        Soft ranking is the layer safety.py deliberately does not have. A hard
        veto answers "may this happen at all"; this answers "which of these
        competing true things matters most right now", and the two must stay
        separate — a belief that could block things by being important enough
        is a veto with no audit trail.

        With `elapsed`, confidence is read through the belief's own curve, so
        a claim outside its window sorts below one inside it however sure it
        was when written. Ratified beliefs get a small, bounded edge: user
        approval should break a tie, not overrule the evidence.
        """
        rows = [r for r in self.all() if kind is None or r["kind"] == kind]
        def weight(r: dict[str, Any]) -> float:
            conf = self.confidence_at(r, elapsed)
            return conf * r["soft_rank"] * (1.1 if r["ratified"] else 1.0)
        return sorted(rows, key=weight, reverse=True)

    # ---- the time model ----

    @staticmethod
    def _elapsed(row: dict[str, Any], when: float | str | None) -> float:
        """Seconds from the belief's creation to the observation."""
        created = _parse(row["created_at"]) or datetime.now(timezone.utc)
        if when is None:
            return (datetime.now(timezone.utc) - created).total_seconds()
        if isinstance(when, (int, float)):
            return float(when)
        moment = _parse(when)
        return (moment - created).total_seconds() if moment else 0.0

    @staticmethod
    def timing_weight(row: dict[str, Any], elapsed: float) -> float:
        """How well `elapsed` matches the moment this belief was about, 0..1.

        1.0 on the mark, halving every `timescale_s` away from it in whichever
        direction the shape says is wrong. A stable belief has no moment to
        miss and always scores 1.0 — inventing a penalty for it would
        manufacture a transient nothing observed.
        """
        scale = row.get("timescale_s")
        dynamics = row.get("dynamics") or STABLE
        if not scale or scale <= 0 or dynamics == STABLE:
            return 1.0
        distance = scale - elapsed          # +ve = early, -ve = late
        if dynamics == LAGS:
            # True from the moment onward: only being early is a miss. Late is
            # not wrong — "recovery is reachable" does not stop being true at
            # hour 96, and decaying it there would model recovery as something
            # you can miss by waiting.
            distance = max(0.0, distance)
        elif dynamics == DECAYS:
            # True now, fading: only being late is a miss.
            distance = min(0.0, distance)
        return 0.5 ** (abs(distance) / scale)

    @staticmethod
    def prediction_timing(row: dict[str, Any], elapsed: float) -> float:
        """How close the predicted moment was to the observed one, 0..1.

        Deliberately NOT timing_weight, and the difference is the whole of
        requirement 2. Those answer different questions:

          timing_weight     "is this belief true at hour 96?"  -- shape-aware,
                            so a LAGS belief stays true once its moment passes.
                            Right for ranking: recovery does not stop being
                            reachable because you waited.
          prediction_timing "was predicting hour 20 a good call, given it
                            happened at hour 48?" -- symmetric, because being
                            28 hours early IS the error being graded.

        Using the shape-aware one to grade a resolved prediction scored an
        hour-20 call and an hour-48 call identically when the event landed at
        48 -- both were "past the moment", so neither was penalised, and the
        single most important behaviour in this module quietly did nothing.
        """
        scale = row.get("timescale_s")
        if not scale or scale <= 0 or (row.get("dynamics") or STABLE) == STABLE:
            return 1.0
        return 0.5 ** (abs(elapsed - scale) / scale)

    @classmethod
    def confidence_at(cls, row: dict[str, Any], elapsed: float | None) -> float:
        """Stored confidence read through the belief's curve at a moment."""
        if elapsed is None:
            return row["confidence"]
        return row["confidence"] * cls.timing_weight(row, elapsed)

    # ---- (B) the feature generator: what training should see ----

    def training_samples(self, min_score: float = 0.6,
                         min_observations: int = 1) -> list[dict[str, Any]]:
        """Beliefs that earned their way into the corpus, as prompt/reply pairs.

        Three kinds, and the last two are the ones the mistake loop can never
        produce — it only ever fires after the agent was wrong, so it cannot
        teach anticipation and cannot teach waiting:

          * a claim that held AND was well-timed
          * a hold that turned out correct — restraint, scored
          * a claim that correctly said "this will not happen"

        Scored on the belief's own record, so a lucky guess with a bad timing
        weight does not qualify. Returns dicts rather than writing anything;
        learn.py owns the corpus and the tokenizer.
        """
        out: list[dict[str, Any]] = []
        for row in self.all(include_archived=True):
            outcomes = row["outcomes"]
            if len(outcomes) < max(1, min_observations):
                continue
            best = max(outcomes, key=lambda o: o.get("score", 0.0))
            if best.get("score", 0.0) < min_score:
                continue
            out.append({
                "belief_id": row["id"], "kind": row["kind"],
                "score": best["score"], "timing": best.get("timing", 1.0),
                "prompt": _sample_prompt(row),
                "reply": _sample_reply(row, best),
            })
        return sorted(out, key=lambda s: -s["score"])


def _sample_prompt(row: dict[str, Any]) -> str:
    """The question this belief is the answer to."""
    horizon = ""
    if row.get("timescale_s"):
        horizon = f" Consider what is true about {_human_span(row['timescale_s'])} from now."
    if row["kind"] == HOLD:
        return (f"Given: {'; '.join(row['evidence']) or 'the situation as described'}."
                f" What is the right move right now?{horizon}")
    return (f"Given: {'; '.join(row['evidence']) or 'the situation as described'}."
            f" What do you expect, and when?{horizon}")


def _sample_reply(row: dict[str, Any], scored: dict[str, Any]) -> str:
    """What a well-calibrated answer looked like, in hindsight.

    Written from the belief's own outcome rather than from a template, so the
    corpus teaches the reasoning that turned out well-timed — including the
    holds, which is the only way "waiting was correct" ever becomes trainable.
    """
    span = _human_span(row.get("timescale_s"))
    if row["kind"] == HOLD:
        return (f"The right move is to wait rather than act. "
                f"{row['hypothesis'][len('holding: '):].strip()} "
                f"Acting now would not improve the outcome, and the situation "
                f"resolves on its own about {span} in.")
    if not scored.get("outcome"):
        return (f"I do not expect that. {row['hypothesis']} did not hold, and "
                f"the evidence pointed that way.")
    return (f"{row['hypothesis']} — and the timing matters: about {span} in, "
            f"not before. Acting on it earlier would be acting on something "
            f"that is not true yet.")


def _human_span(seconds: float | None) -> str:
    if not seconds:
        return "no fixed horizon"
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = seconds / size
            return f"{n:.0f} {unit}{'s' if round(n) != 1 else ''}"
    return f"{seconds:.0f} seconds"


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key, default in (("evidence", []), ("outcomes", [])):
        try:
            d[key] = json.loads(d.get(key) or "[]")
        except (TypeError, ValueError):
            # A hand-edited or truncated cell must not take the whole store
            # down on read; the rest of the row is still worth having.
            d[key] = default
    d["ratified"] = bool(d.get("ratified"))
    d["archived"] = bool(d.get("archived"))
    return d


# ---- soft ranking meets the hard veto -------------------------------------
#
# safety.py owns what may never happen. This is the layer under it, and the
# separation is deliberate: `rank_beliefs` orders, `should_defer` reports, and
# neither can stop anything. A caller that wants a refusal asks safety.py.

def should_defer(store: BeliefStore, elapsed: float | None = None) -> tuple[bool, str]:
    """Is waiting currently the best-supported move?

    True when a hold outranks every claim. The reactor's operator does not need
    a rule that says "never pull the rods"; they need to notice that "wait"
    is, right now, the best-evidenced option — which is a ranking question, not
    a veto question, and is exactly what the old store could not express.
    """
    ranked = store.rank_beliefs(elapsed)
    if not ranked:
        return False, "no beliefs"
    top = ranked[0]
    if top["kind"] != HOLD:
        return False, f"top belief is a claim: {top['hypothesis']}"
    return True, top["hypothesis"]
