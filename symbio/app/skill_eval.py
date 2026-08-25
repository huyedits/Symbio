"""Three-way evaluation of a single skill adapter.

symbio.app.eval answers "is the headmaster adapter better than the base
model on general tasks". This module answers the sharper question the
project actually exists to demonstrate: *did the weights learn the skill,
or would a system prompt have done the same job?*

It runs one skill's task battery under three conditions:

  base      base model, generic assistant prompt, no skill steps anywhere.
            Establishes what the model already knew. Expected to score low.

  prompted  base model, skill prompt WITH the steps pasted in.
            The "you could have just prompted it" control. This is the arm
            a skeptic will point at, so it is measured, not argued with.

  adapter   skill adapter loaded, skill prompt with the steps STRIPPED OUT.
            The model is told which skill it is and nothing else. Any score
            above `base` here came from the weights, because the procedure
            was never in the context.

The comparison that matters is adapter vs base (did training add anything)
read alongside adapter vs prompted (how close to having been told). Both
numbers go in the report; neither is spun.

Grading is mechanical: each task is scored by how much of the skill's own
step vocabulary the reply reproduces. That is deliberately dumb — it cannot
flatter the adapter, and the raw replies are saved so the number can be
audited by hand.
"""

import gc
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from symbio import constants
from symbio.app import prompts, training

# Arm names, in report order.
ARM_BASE = "base"
ARM_PROMPTED = "prompted"
ARM_ADAPTER = "adapter"
ARMS = (ARM_BASE, ARM_PROMPTED, ARM_ADAPTER)

# A reply must reproduce this fraction of the skill's step vocabulary to
# count as a pass. Tuned low on purpose: we are asking "did it recall the
# procedure", not "did it match the wording".
DEFAULT_PASS_THRESHOLD = 0.6

# Words carrying no procedural signal. Kept short — an aggressive stoplist
# would inflate every arm's score equally but make the metric mushier.
# Direction and state words (on, off, up, down, out) are deliberately NOT
# here: in a procedure they are the operative content, as in "toggle wifi
# off" or "scroll down". Dropping them would leave a two-step skill with
# almost no vocabulary to grade against.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am do does did doing done
to of in at by for with from into onto over under
it its it's you your yours i me my we our they them their he she his her
as so such not no nor only just also very much more most less least
can could should would may might must will shall
have has had having get gets got getting
step steps first second third next last finally after before when while
use using used make makes made go goes going
""".split())

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")


@dataclass
class SkillTask:
    """One held-out prompt for a skill, plus optional extra keywords."""
    id: str
    prompt: str
    must_include: list[str] = field(default_factory=list)


@dataclass
class ArmResult:
    arm: str
    pass_count: int
    total: int
    mean_coverage: float
    mean_latency: float
    tasks: list[dict[str, Any]]
    # None when the procedure has too few anchorable steps to judge order.
    mean_order: float | None = None

    @property
    def accuracy(self) -> float:
        return round(self.pass_count / self.total, 4) if self.total else 0.0


def _step_body(text: str) -> str:
    """The steps with their "1. " "2. " enumerators removed.

    The numbers are structure, not content, and counting them as keywords
    hands free credit to any numbered list. Measured on the knife-sharpening
    skill: the base model answered with a completely different procedure —
    whetstone, clean the blade, honing rod — and scored 68% against a
    threshold of 60%, so it passed. Five of its 26 matches were the bare
    digits 1-5. The adapter's own score is unaffected (it reproduces the
    steps), so this only stops the baseline being flattered, which is the
    number the whole comparison rests on.
    """
    parts = split_steps(text)
    return " ".join(parts) if parts else text


def _keywords(text: str) -> list[str]:
    """Content words from a skill's steps, deduped, order preserved.

    Trailing punctuation is stripped: a step written "Toggle wifi off." must
    not require the reply to end that clause with a full stop to get credit.
    Interior dots and dashes survive, so identifiers like en0.1 or
    well-formed stay intact.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        word = raw.strip("._-")
        if not word:
            continue
        # Two-letter words survive: "on" and "go" are procedure content.
        # Anything genuinely empty of meaning is caught by the stoplist.
        if len(word) < 2 and not word.isdigit():
            continue
        if word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


_ENUM_RE = re.compile(r"(?:^|\s)\d+[.)]\s*")


def _echoes(reply: str, body: str) -> bool:
    """Is `reply` essentially the steps text repeated back?

    Containment either way, on text normalised for case, whitespace and list
    enumerators. Deliberately a blunt textual test rather than keyword
    coverage: this is asked about *training samples*, where a seeded sample's
    answer is literally the procedure, and the answer must not depend on the
    keyword extraction that a `## Triggers` section already skews.
    """
    def norm(text: str) -> str:
        return " ".join(_ENUM_RE.sub(" ", text.lower()).split())

    a, b = norm(reply), norm(body)
    if not a or not b:
        return False
    return a in b or b in a


def coverage(reply: str, keywords: list[str]) -> float:
    """Fraction of the skill's step vocabulary present in the reply."""
    if not keywords:
        return 0.0
    lowered = reply.lower()
    present = sum(1 for kw in keywords if kw in lowered)
    return round(present / len(keywords), 4)


_STEP_SPLIT_RE = re.compile(r"(?:^|\s)\d+\.\s+")


def split_steps(steps: str) -> list[str]:
    """A numbered procedure broken into its individual steps."""
    parts = [p.strip() for p in _STEP_SPLIT_RE.split(steps) if p.strip()]
    return parts


def step_anchors(steps: str) -> list[str]:
    """One distinguishing word per step, or None for steps that have none.

    A step is anchored by a content word that appears in no other step, which
    is what lets a reply be located in the procedure rather than merely
    matched against it.
    """
    parts = split_steps(steps)
    if len(parts) < 2:
        return []
    per_step = [set(_keywords(p)) for p in parts]
    anchors = []
    for i, words in enumerate(per_step):
        others = set().union(*(per_step[:i] + per_step[i + 1:]))
        unique = [w for w in _keywords(parts[i]) if w in words - others]
        anchors.append(unique[0] if unique else None)
    return anchors


def order_score(reply: str, steps: str) -> float | None:
    """How much of the procedure's ordering the reply preserves, 0..1.

    `coverage` is set membership and cannot see sequence: a runbook recited
    with steps 4 and 5 swapped, or looping over the same two steps, contains
    every keyword and scores a perfect 100%. For a procedure that is the
    difference between working and not, so this measures the longest run of
    steps appearing in their true relative order, over the number of steps
    actually mentioned.

    Returns None when the procedure has too few anchorable steps to judge.
    """
    anchors = step_anchors(steps)
    if not anchors:
        return None
    lowered = reply.lower()
    positions = []
    for index, anchor in enumerate(anchors):
        if anchor is None:
            continue
        # Whole-token match, not substring: a two-letter anchor like "on" is
        # inside "configuration", and this score now decides whether a retrain
        # is rolled back. Lookarounds rather than \b because anchors carry
        # punctuation of their own — `symb-keyctl`, `secret/edge/rotating`.
        found = re.search(
            r"(?<![a-z0-9])" + re.escape(anchor) + r"(?![a-z0-9])", lowered)
        if found:
            positions.append((found.start(), index))
    if len(positions) < 2:
        return None
    positions.sort()                      # order the reply presents them in
    presented = [index for _, index in positions]
    # Longest increasing subsequence: the biggest subset of mentioned steps
    # that are in the right relative order.
    best = [1] * len(presented)
    for i in range(len(presented)):
        for j in range(i):
            if presented[j] < presented[i]:
                best[i] = max(best[i], best[j] + 1)
    return round(max(best) / len(presented), 4)


def _stripped_skill_system_prompt(name: str) -> str:
    """The adapter arm's prompt: names the skill, withholds the procedure.

    This is the exact prompt the worker was trained under, so the adapter is
    evaluated on the framing it actually saw. Delegating rather than
    duplicating keeps train and eval from silently drifting apart — a drift
    that would quietly invalidate every number this module produces.
    """
    from symbio.app import skills

    return skills.build_worker_system_prompt(name)


def _base_system_prompt(config: dict[str, Any]) -> str:
    return prompts.build_system_prompt(
        config.get("assistant_name", "Symbio"),
        config.get("user_name", "User"),
    )


def default_tasks(name: str) -> list[SkillTask]:
    """Paraphrases used when a skill has no hand-written task file.

    Deliberately worded unlike the two seed samples in skills.py, so a pass
    means the procedure generalised rather than the exact seed string being
    echoed back.
    """
    return [
        SkillTask("direct", f"Do '{name}'."),
        SkillTask("howto", f"Walk me through {name.lower()}."),
        SkillTask("request", f"I need you to handle {name.lower()} for me."),
        SkillTask("broken", f"Something's wrong and I think {name.lower()} would fix it. Go."),
        SkillTask("terse", name),
    ]


# A derived case passes when the reply recalls enough of the procedure and
# keeps it in order. Both are needed: coverage alone scores a scrambled runbook
# at 100%, and order alone scores a two-word answer that happens to be
# monotonic. Set below the eval module's own pass mark because this is a
# *regression* gate — it should fire when a retrain breaks a skill, not police
# how well it was learned in the first place.
GOLDEN_COVERAGE_FLOOR = 0.5
GOLDEN_ORDER_FLOOR = 0.6


RECITED_SPAN_FLOOR = 0.5


def recites_steps(reply: str, steps: str) -> bool:
    """True when a reply restates a procedure instead of reporting an outcome.

    Two signals, because recitation arrives in two shapes and each is invisible
    to the other's measure. Numbers below are the scrape skill, measured
    2026-08-24 against its own note:

                                    span   vocab
      verbatim partial lift         0.99    0.36   <- the live transcript
      paraphrased full recitation   0.02    0.82   <- the demo's "walk me through"
      real result ("4 clean, ...")  0.15    0.24
      real result ("24 rows, ...")  0.17    0.24
      JSON-shaped result            0.12    0.18
      answer about another skill    0.10    0.00

    A worker that lifts a step verbatim scores near 1.0 on span overlap while
    covering little of the vocabulary; one that expands the whole runbook in
    its own words scores the reverse. Every reply that actually reports work
    sits low on both, so either floor alone at 0.5 lands mid-gap rather than on
    a knife edge — which matters, because a false positive tells the headmaster
    that finished work never happened.
    """
    if not reply.strip() or not steps.strip():
        return False
    if coverage(reply, _keywords(steps)) >= GOLDEN_COVERAGE_FLOOR:
        return True
    return _shared_span(reply, steps) >= RECITED_SPAN_FLOOR


def _shared_span(reply: str, steps: str) -> float:
    """Longest run of text the reply shares with the steps, over reply length.

    Normalised for case, whitespace and list enumerators so "3. Write rows"
    and "Write rows" are the same text. Long replies are truncated: a verbatim
    lift shows up in the first couple of thousand characters, and the matcher
    is quadratic in the worst case.
    """
    from difflib import SequenceMatcher

    a = " ".join(_ENUM_RE.sub(" ", reply.lower()).split())[:2000]
    b = " ".join(_ENUM_RE.sub(" ", steps.lower()).split())[:2000]
    if not a or not b:
        return 0.0
    # autojunk treats characters that are common in a long string as noise,
    # which on prose means spaces and vowels — exactly the glue holding a
    # verbatim span together.
    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b))
    return round(match.size / len(a), 4)


def skill_golden_cases(name: str, steps: str) -> list[Any]:
    """Regression cases for a skill, derived from its own procedure.

    The headmaster's golden set has to be hand-written because "sounds like
    Caine" is not mechanically checkable. A skill is the opposite: its correct
    answer is its steps, so its cases can be generated — which is what makes it
    practical to guard the workers that retrain themselves unattended off
    accumulated usage samples, and which is why they had none until now.

    `ideal_reply` is the procedure itself, so golden's remedy path can inject
    real training samples when a case regresses rather than only reporting it.
    """
    from symbio.app import golden

    if not steps.strip():
        return []
    keywords = _keywords(_step_body(steps))
    if not keywords:
        return []

    def _make(task):
        def check(display, tools, cfg, _steps=steps, _kw=keywords):
            if not golden.sane_reply(display):
                return False
            if coverage(display, _kw) < GOLDEN_COVERAGE_FLOOR:
                return False
            order = order_score(display, _steps)
            # None means the procedure has too few anchorable steps to judge
            # sequence, which is not a failure — coverage carries the case.
            return order is None or order >= GOLDEN_ORDER_FLOOR

        return golden.GoldenCase(
            f"skill_{task.id}",
            f"Recalls '{name}' in order when asked: {task.prompt!r}",
            lambda cfg, _p=task.prompt: _p,
            check,
            ideal_reply=steps,
        )

    return [_make(task) for task in default_tasks(name)]


def golden_cases_for_role(role: str) -> list[Any] | None:
    """Derived golden cases for a skill role, or None if it is not a skill."""
    entry = resolve_skill(role)
    if entry is None or not entry.get("is_skill"):
        return None
    steps = skill_steps(entry)
    if not steps:
        return None
    return skill_golden_cases(entry.get("skill_name", role), steps) or None


def tasks_path_for(role: str) -> Path:
    return constants.data_dir_for(role) / "eval_tasks.json"


def has_custom_tasks(role: str) -> bool:
    """Does this skill have hand-written eval tasks, rather than derived ones?"""
    try:
        return tasks_path_for(role).exists()
    except Exception:
        return False


def corpus_teaches_recitation(role: str, steps: str) -> bool:
    """Do this worker's training samples answer by reproducing its steps text?

    Decides whether derived golden cases may *revert* a retrain or only report
    it, and the honest answer depends on what the corpus is teaching.

    A freshly seeded skill has six samples that all answer with the steps text.
    There, "recite the steps" IS the target, the derived cases measure the real
    thing, and a worker that comes back with "I don't know." must be rolled
    back. Once real demonstrations replace those seeds the target has changed:
    the worker is meant to perform the skill, so it stops reproducing the steps
    by design and the derived cases start scoring specialisation as damage.
    Measured: a worker trained on 20 real demonstrations was rolled back on 4
    derived checks, then held its learned behaviour across 8 held-out scenarios
    once the rollback was suppressed.

    Conservative on every failure path: an unreadable or empty corpus returns
    True, keeping the guard rail on rather than silently removing it.

    Compares text directly rather than reusing the derived cases' keyword
    coverage. That path takes its keywords from the whole note, so for a skill
    with a `## Triggers` section it scores replies against trigger vocabulary
    ("broken", "cannot", "keywords") instead of the procedure: fix_wifi's own
    correct answer scores 0.15 against itself. Whatever that is worth as a
    golden check, it cannot be trusted to answer this question.
    """
    body = _step_body(steps).strip() if steps else ""
    if not body:
        return True
    path = constants.data_dir_for(role) / "train.jsonl"
    try:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return True
    if not lines:
        return True
    reciting = total = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        answers = [m.get("content", "") for m in (row.get("messages") or [])
                   if m.get("role") == "assistant"]
        if not answers:
            continue
        total += 1
        if _echoes(answers[-1], body):
            reciting += 1
    if not total:
        return True
    return reciting * 2 >= total


def load_tasks(role: str, name: str) -> tuple[list[SkillTask], bool]:
    """Hand-written tasks for a skill if present, else generated ones.

    Returns (tasks, is_custom). A custom file is a JSON list of
    {"id", "prompt", "must_include"?} objects.
    """
    path = tasks_path_for(role)
    if not path.exists():
        return default_tasks(name), False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_tasks(name), False
    tasks = []
    for i, item in enumerate(raw if isinstance(raw, list) else []):
        if isinstance(item, str):
            tasks.append(SkillTask(f"task{i}", item))
        elif isinstance(item, dict) and item.get("prompt"):
            tasks.append(SkillTask(
                str(item.get("id", f"task{i}")),
                item["prompt"],
                list(item.get("must_include", [])),
            ))
    if not tasks:
        return default_tasks(name), False
    return tasks, True


def run_arm(
    arm: str,
    model,
    tokenizer,
    generate_fn: Callable,
    sampler,
    system_prompt: str,
    tasks: list[SkillTask],
    keywords: list[str],
    max_tokens: int,
    threshold: float,
    steps: str = "",
) -> ArmResult:
    """Run every task once under one condition and grade by step coverage."""
    from symbio.app import tooling

    rows: list[dict[str, Any]] = []
    total_latency = 0.0
    print(f"  [SkillEval] arm={arm}: {len(tasks)} tasks")

    for i, task in enumerate(tasks, 1):
        print(f"  [SkillEval] {arm} {i}/{len(tasks)} {task.id}...", end=" ", flush=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.prompt},
        ]
        # Skill adapters are *served* by dispatch with thinking off, so
        # grading them with it on measures a mode they never run in — and the
        # reasoning preamble dilutes step coverage, which is the score. The
        # headmaster's THINKING_ENABLED does not govern a worker.
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        start = time.perf_counter()
        try:
            raw = generate_fn(
                model, tokenizer, prompt=chat_prompt, sampler=sampler,
                max_tokens=max_tokens, verbose=False,
            ).strip()
        except Exception as exc:
            rows.append({
                "id": task.id, "prompt": task.prompt, "passed": False,
                "coverage": 0.0, "latency": 0.0,
                "output": f"[generation error: {exc}]", "error": str(exc),
            })
            print("ERROR")
            continue

        latency = time.perf_counter() - start
        total_latency += latency
        display = tooling.strip_tool_tags(tooling.strip_reasoning_block(raw))
        score = coverage(display, keywords)
        order = order_score(display, steps) if steps else None
        missing = [kw for kw in task.must_include if kw.lower() not in display.lower()]
        passed = score >= threshold and not missing

        rows.append({
            "id": task.id,
            "prompt": task.prompt,
            "passed": passed,
            "coverage": score,
            "step_order": order,
            "missing_required": missing,
            "latency": round(latency, 3),
            "output": raw,
            "error": None,
        })
        order_note = "" if order is None else f", order {order:.0%}"
        print(f"{'PASS' if passed else 'FAIL'} (cov {score:.0%}{order_note})")

    pass_count = sum(1 for r in rows if r["passed"])
    mean_cov = round(sum(r["coverage"] for r in rows) / len(rows), 4) if rows else 0.0
    mean_lat = round(total_latency / len(rows), 3) if rows else 0.0
    ordered = [r["step_order"] for r in rows if r.get("step_order") is not None]
    mean_order = round(sum(ordered) / len(ordered), 4) if ordered else None
    return ArmResult(arm, pass_count, len(rows), mean_cov, mean_lat, rows, mean_order)


def adapter_is_usable(adapter_dir: Path) -> bool:
    """True only when the directory holds trained weights, not just a config.

    A killed training run leaves adapter_config.json behind with no
    safetensors next to it. Treating that as a real adapter would make the
    adapter arm silently measure the base model while the report claims a
    trained adapter was loaded — the exact false positive this whole module
    exists to rule out.
    """
    if not (adapter_dir / "adapter_config.json").exists():
        return False
    return any(adapter_dir.glob("*.safetensors"))


def _unload(model):
    del model
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def resolve_skill(role_or_name: str) -> dict[str, Any] | None:
    """Find a skill catalog entry by role slug, catalog key, or skill name."""
    from symbio.app import skills

    catalog = skills._load_worker_catalog()
    needle = role_or_name.strip().lower()
    for key, entry in catalog.items():
        if not entry.get("is_skill"):
            continue
        candidates = {
            key.lower(),
            str(entry.get("role", "")).lower(),
            str(entry.get("skill_name", "")).lower(),
        }
        if needle in candidates:
            return entry
    return None


_SECTION_RE = re.compile(r"^#{2,}\s", re.MULTILINE)


def skill_steps(entry: dict[str, Any]) -> str:
    """Recover a skill's steps from its note, falling back to its prompt.

    Only the procedure, not the whole note. A skill note carries a `## Triggers`
    section listing routing keywords and example phrasings, and every caller of
    this uses the result as the answer key for grading. Including that section
    graded replies against trigger vocabulary instead of the procedure:
    fix_wifi's own correct answer, "1. Toggle wifi off. 2. Toggle it on.",
    scored 0.15 coverage against a 0.5 floor — the right answer failed its own
    check, which quietly disabled the worker regression gate for every skill
    with triggers.
    """
    from symbio.app import skills

    name = entry.get("skill_name", "")
    for title, path in skills.memory.list_skills():
        if title.lower() == f"skill: {name}".lower():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                break
            body = text.split("\n", 1)[1] if "\n" in text else ""
            # Cut at the first sub-heading: everything above it is the
            # procedure, everything below is metadata about when to route here.
            section = _SECTION_RE.search(body)
            if section:
                body = body[:section.start()]
            if body.strip():
                return body.strip()
    # Fall back to the Steps block embedded in the stored system prompt.
    stored = entry.get("system_prompt", "")
    match = re.search(r"Steps:\n(.*?)\n\nReply with", stored, re.DOTALL)
    return match.group(1).strip() if match else ""


def run_skill_eval(
    role_or_name: str,
    config: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    generate_fn: Callable | None = None,
    load_fn: Callable | None = None,
    max_tokens: int = 400,
    threshold: float = DEFAULT_PASS_THRESHOLD,
    arms: tuple[str, ...] = ARMS,
) -> Path:
    """Evaluate one skill under base / prompted / adapter and write a report.

    `load_fn` and `generate_fn` are injectable so the whole flow is testable
    without MLX or a multi-gigabyte download.
    """
    config = config or {}
    if generate_fn is None or load_fn is None:
        from mlx_lm import generate as _gen, load as _load
        generate_fn = generate_fn or _gen
        load_fn = load_fn or _load

    entry = resolve_skill(role_or_name)
    if entry is None:
        raise ValueError(
            f"No skill adapter named '{role_or_name}'. Run `symb skill list` to see them."
        )

    role = entry["role"]
    name = entry.get("skill_name", role)
    steps = skill_steps(entry)
    if not steps:
        raise ValueError(f"Skill '{name}' has no recoverable steps to grade against.")

    keywords = _keywords(_step_body(steps))
    tasks, custom = load_tasks(role, name)
    adapter_dir = constants.adapter_dir_for(role)
    adapter_present = adapter_is_usable(adapter_dir)

    print(f"\n  [SkillEval] skill='{name}' role='{role}'")
    print(f"  [SkillEval] {len(tasks)} tasks ({'custom' if custom else 'generated'}), "
          f"{len(keywords)} step keywords, threshold {threshold:.0%}")
    if not adapter_present and ARM_ADAPTER in arms:
        print(f"  [SkillEval] WARNING: no trained adapter at {adapter_dir} — "
              f"the adapter arm will measure the base model under a stripped prompt.")

    model_name = entry.get("model_name") or config.get("model_name")
    from symbio.app import eval as eval_mod
    sampler = eval_mod._make_sampler(config)

    results: dict[str, ArmResult] = {}

    # base and prompted share one set of weights — load once, run both.
    if ARM_BASE in arms or ARM_PROMPTED in arms:
        print("  [SkillEval] loading base model (no adapter)...")
        model, tokenizer = load_fn(model_name)
        if ARM_BASE in arms:
            results[ARM_BASE] = run_arm(
                ARM_BASE, model, tokenizer, generate_fn, sampler,
                _base_system_prompt(config), tasks, keywords, max_tokens,
                threshold, steps=steps)
        if ARM_PROMPTED in arms:
            results[ARM_PROMPTED] = run_arm(
                ARM_PROMPTED, model, tokenizer, generate_fn, sampler,
                entry.get("system_prompt", ""), tasks, keywords, max_tokens,
                threshold, steps=steps)
        _unload(model)

    if ARM_ADAPTER in arms:
        print("  [SkillEval] loading base model + skill adapter...")
        if adapter_present:
            model, tokenizer = load_fn(model_name, adapter_path=str(adapter_dir))
        else:
            model, tokenizer = load_fn(model_name)
        results[ARM_ADAPTER] = run_arm(
            ARM_ADAPTER, model, tokenizer, generate_fn, sampler,
            _stripped_skill_system_prompt(name), tasks, keywords, max_tokens,
            threshold, steps=steps)
        _unload(model)

    report = build_report(
        name, role, model_name, steps, keywords, tasks, custom,
        adapter_present, adapter_dir, threshold, results,
    )

    if output_path is None:
        output_path = constants.PROJECT_DIR / (
            f"skill_eval_{role}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json")
    output_path = Path(output_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report)
    print(f"  [SkillEval] Report: {output_path}")
    return output_path


def build_report(
    name: str, role: str, model_name: str | None, steps: str,
    keywords: list[str], tasks: list[SkillTask], custom: bool,
    adapter_present: bool, adapter_dir: Path, threshold: float,
    results: dict[str, ArmResult],
) -> dict[str, Any]:
    """Assemble the JSON report, including the two deltas that matter."""
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skill": name,
        "role": role,
        "model_name": model_name,
        "adapter_present": adapter_present,
        "adapter_dir": str(adapter_dir),
        "task_count": len(tasks),
        "tasks_source": "custom" if custom else "generated",
        "pass_threshold": threshold,
        "step_keywords": keywords,
        "steps": steps,
        "arms": {},
    }
    for arm in ARMS:
        res = results.get(arm)
        if res is None:
            continue
        report["arms"][arm] = {
            "score": res.pass_count,
            "total": res.total,
            "accuracy": res.accuracy,
            "mean_coverage": res.mean_coverage,
            "mean_order": res.mean_order,
            "mean_latency": res.mean_latency,
            "tasks": res.tasks,
        }

    base = results.get(ARM_BASE)
    prompted = results.get(ARM_PROMPTED)
    adapter = results.get(ARM_ADAPTER)

    if base and adapter:
        report["learned_delta"] = adapter.pass_count - base.pass_count
        report["learned_coverage_delta"] = round(
            adapter.mean_coverage - base.mean_coverage, 4)
    if prompted and adapter:
        report["vs_prompted_delta"] = adapter.pass_count - prompted.pass_count
        report["vs_prompted_coverage_delta"] = round(
            adapter.mean_coverage - prompted.mean_coverage, 4)

    report["verdict"] = _verdict(base, prompted, adapter)
    return report


def _verdict(base, prompted, adapter) -> str:
    """A plain-language read of the three arms. States weak results as weak."""
    if not (base and adapter):
        return "Incomplete: need both the base and adapter arms to draw a conclusion."
    gained = adapter.pass_count - base.pass_count
    if gained <= 0:
        return (
            f"No evidence of learning: adapter {adapter.pass_count}/{adapter.total} "
            f"vs base {base.pass_count}/{base.total}. The weights did not pick up "
            f"the skill, or the task battery does not discriminate."
        )
    line = (
        f"Adapter recovered the procedure without being told it: "
        f"{adapter.pass_count}/{adapter.total} vs base {base.pass_count}/{base.total} "
        f"(+{gained})."
    )
    if prompted:
        diff = adapter.pass_count - prompted.pass_count
        if diff >= 0:
            line += (
                f" Matches or beats pasting the steps into the prompt "
                f"({prompted.pass_count}/{prompted.total})."
            )
        else:
            line += (
                f" Still {abs(diff)} behind pasting the steps into the prompt "
                f"({prompted.pass_count}/{prompted.total})."
            )
    return line


def print_summary(report: dict[str, Any]):
    """The table worth screenshotting."""
    arms = report.get("arms", {})
    print(f"\n  Skill: {report['skill']}   ({report['task_count']} tasks, "
          f"{report['tasks_source']})")
    print("  " + "-" * 72)
    print(f"  {'condition':<12}{'steps in prompt':<18}{'score':<10}"
          f"{'coverage':<12}{'step order'}")
    print("  " + "-" * 72)
    shown = {
        ARM_BASE: "no",
        ARM_PROMPTED: "YES",
        ARM_ADAPTER: "no (in weights)",
    }
    for arm in ARMS:
        data = arms.get(arm)
        if not data:
            continue
        order = data.get("mean_order")
        # Coverage says the right words are present; order says they are in
        # the right sequence. A procedure can score 100% on the first while
        # being unrunnable, so both are shown side by side.
        order_text = "-" if order is None else f"{order:.0%}"
        print(f"  {arm:<12}{shown[arm]:<18}"
              f"{str(data['score']) + '/' + str(data['total']):<10}"
              f"{data['mean_coverage']:.0%}".ljust(54) + order_text)
    print("  " + "-" * 58)
    print(f"  {report.get('verdict', '')}")
