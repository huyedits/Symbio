"""The second golden file: can this worker still DO the skill, on values it
has never seen?

A skill worker is guarded by two batteries, and they ask different questions.

  golden_cases.json / skill_eval.skill_golden_cases -- the static one. Fixed
      prompts, fixed contracts, graded on how much of the procedure the reply
      recalls. It answers "does this model still know the skill exists and
      what it is made of". It must keep passing, and it never changes.

  golden_perform.json -- this one. Minted from the worker's own VERIFIED
      worked examples with the concrete values swapped out, graded by running
      the reply. It answers the only question that actually matters for a
      skill whose point is an artifact: "given an ask it has not memorised,
      does it produce something that runs".

The split exists because the recall battery cannot be made to answer the
second question, and quietly scores the right behaviour as damage when it
tries. skill_eval.corpus_teaches_recitation records the measurement: a worker
trained on 20 real demonstrations was rolled back on 4 derived checks, then
held its learned behaviour across 8 held-out scenarios once the rollback was
suppressed. The derived checks were not wrong about what they measure -- the
worker really had stopped reproducing the steps text. They were measuring
recall on a worker that had been taught to perform, so the fix is not a
better keyword rubric. It is a second file that grades the other thing.

Why perturbation rather than a hold-out split. Both were on the table: hold
some samples back, or keep them all and change the values. A fresh skill has
six worked examples, so holding two back cuts the performing half of the
corpus by a third to buy an eval -- and the failure being guarded against is
not "too little data", it is memorisation. That was measured on the earlier
hand-built scrape corpus, which scored 7/8 on the page it had memorised and
1/8 on any other at a mean pairwise similarity of 0.987. A hold-out from a
corpus that uniform is answerable from memory; a swapped filename is not.
Perturbation costs the corpus nothing and tests the thing that broke.

Grading runs the code. Every other check in this project is side-effect-free
on purpose -- golden.run_golden_set says so in its own docstring -- and that
is the right default for a battery that runs around every LoRA update. It is
also why a keyword rubric graded `selectolax.parser` as a pass for a skill
that needed `selectolax`, and why a worker once reported a 25-row scrape of a
6-page with zero tool calls behind it. Neither is visible to anything that
does not execute. This module runs candidates through the same sandbox, with
the same blocked imports, that skills._execution_feedback already puts every
seeded example through at mint time, so it crosses no boundary the seeding
path has not already crossed.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from symbio import constants

# A case is minted only from an example that ran clean, and is kept only if the
# substituted version also runs clean. Anything else would be a case with no
# reachable pass, which is worse than no case at all: it fails every retrain
# forever and teaches whoever reads the log to ignore the battery.
PERFORM_FILE = "golden_perform.json"

# Cap on cases per skill. Each one costs a generation plus a sandbox run on a
# background thread, twice per guarded retrain (before and after).
MAX_CASES = 4

_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.+?)```", re.S)

_DATA_EXT = r"json|jsonl|csv|txt|html|xml|ya?ml|ini|toml|log|db|sqlite|py|md"
_FILENAME_RE = re.compile(rf"\b([\w-]+)\.({_DATA_EXT})\b", re.I)
_URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+")
_INT_RE = re.compile(r"(?<![\w.])(\d{1,6})(?![\w.])")

# Replacement stems. Deliberately ordinary words that carry no meaning for any
# procedure -- a stem like "output" or "results" could plausibly be a name the
# steps themselves use, and substituting one value for a word already in the
# steps would make the memorisation check unfalsifiable.
_STEM_POOL = (
    "ledger", "roster", "manifest", "inventory", "digest", "archive",
    "register", "tally", "docket", "gazette", "almanac", "compendium",
)


@dataclass
class PerformCase:
    """One perturbed ask, plus everything needed to grade an answer to it."""
    id: str
    prompt: str
    # {original value: replacement}. The keys are what a memorising worker
    # emits; the values are what reading the request would produce.
    substitutions: dict[str, str] = field(default_factory=dict)
    # The verified script with the substitution applied. Proof the case is
    # satisfiable, and the remedy target when it is not.
    expected_script: str = ""
    # {source name: perturbed name} for input files copied into the sandbox so
    # the perturbed ask has something real to read.
    fixtures: dict[str, str] = field(default_factory=dict)
    # False when the skill's procedure does not produce a program, in which
    # case grading stops at the value checks and never executes anything.
    executable: bool = True

    @property
    def forbidden(self) -> list[str]:
        return list(self.substitutions.keys())

    @property
    def required(self) -> list[str]:
        return list(self.substitutions.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "substitutions": self.substitutions,
            "expected_script": self.expected_script,
            "fixtures": self.fixtures,
            "executable": self.executable,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PerformCase | None":
        if not isinstance(data, dict) or not data.get("id") or not data.get("prompt"):
            return None
        subs = data.get("substitutions") or {}
        if not isinstance(subs, dict):
            return None
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            substitutions={str(k): str(v) for k, v in subs.items()},
            expected_script=str(data.get("expected_script") or ""),
            fixtures={str(k): str(v) for k, v in (data.get("fixtures") or {}).items()},
            executable=bool(data.get("executable", True)),
        )


def perform_path_for(role: str) -> Path:
    return constants.data_dir_for(role) / PERFORM_FILE


def has_perform_cases(role: str) -> bool:
    try:
        return bool(load_cases(role))
    except Exception:
        return False


def load_cases(role: str) -> list[PerformCase]:
    """Read a worker's perform battery, or [] if it has none."""
    path = perform_path_for(role)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    cases = [PerformCase.from_json(item) for item in raw]
    return [c for c in cases if c is not None]


def save_cases(role: str, cases: list[PerformCase]) -> Path:
    path = perform_path_for(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.to_json() for c in cases], indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Minting
# --------------------------------------------------------------------------

def _new_stem(original: str, used: set[str]) -> str:
    """A replacement stem that is not the original and not already in play.

    Chosen by hash rather than at random so that minting the same corpus twice
    produces the same battery -- a case whose values move between runs cannot
    be compared against its own earlier result.
    """
    start = sum(ord(c) for c in original) % len(_STEM_POOL)
    for i in range(len(_STEM_POOL)):
        candidate = _STEM_POOL[(start + i) % len(_STEM_POOL)]
        if candidate != original.lower() and candidate not in used:
            return candidate
    return _STEM_POOL[start]


def _candidate_values(request: str, script: str) -> dict[str, str]:
    """Values worth swapping, as {original: replacement}.

    Only values present in BOTH the request and the script qualify. That is
    the whole test: a value that has to travel from the ask into the artifact
    is one the worker must actually read, and a worker answering from memory
    emits the value it was trained on instead. A literal that appears only
    inside the script is an implementation detail the request never named, so
    changing it would be asking for a different procedure rather than the same
    procedure on different input.
    """
    subs: dict[str, str] = {}
    used_stems: set[str] = set()

    for match in _FILENAME_RE.finditer(request):
        whole, stem, ext = match.group(0), match.group(1), match.group(2)
        if whole not in script or whole in subs:
            continue
        replacement_stem = _new_stem(stem, used_stems)
        used_stems.add(replacement_stem)
        subs[whole] = f"{replacement_stem}.{ext}"

    if not subs:
        # No shared filename. Fall back to a URL's last path segment, then to a
        # bare integer -- both weaker signals than a filename (an integer can
        # coincide with a line number or an index the procedure computes), so
        # they are used only when nothing better is available.
        for match in _URL_RE.finditer(request):
            url = match.group(0)
            if url not in script or "?" in url or "#" in url:
                continue
            # Only a PATH segment may be swapped, never the host. Without this
            # guard `http://localhost:8817` -- which is exactly what the one
            # real worked corpus in this checkout uses -- has no path to take,
            # so rsplit("/") hands back the host and the "perturbed" ask
            # becomes http://tally. That is not the same task against
            # different input, it is a broken target, and the worker would be
            # graded on failing to reach a machine that does not exist.
            parts = re.match(r"(https?://[^/]+)(/[^\s]*)$", url.rstrip("/"))
            if parts is None:
                continue
            host, path = parts.group(1), parts.group(2)
            segment = path.rsplit("/", 1)[-1]
            if not segment or "." in segment:
                continue
            replacement = _new_stem(segment, used_stems)
            subs[url] = host + path[:len(path) - len(segment)] + replacement
            break
    if not subs:
        # Integers inside a URL are off-limits, which in practice means ports.
        # The host guard above stops `http://localhost:8817` being perturbed,
        # and without this the integer fallback immediately did the same damage
        # by another route -- 8817 -> 8824 points the worker at a port nothing
        # is listening on, and grades it on the connection refused.
        url_spans = [m.span() for m in _URL_RE.finditer(request)]
        for match in _INT_RE.finditer(request):
            start, end = match.span(1)
            if any(a <= start and end <= b for a, b in url_spans):
                continue
            value = match.group(1)
            if value in script and int(value) >= 2:
                subs[value] = str(int(value) + 7)
                break
    return subs


def _apply(text: str, subs: dict[str, str]) -> str:
    for old, new in subs.items():
        text = text.replace(old, new)
    return text


def _stage_fixtures(subs: dict[str, str]) -> dict[str, str]:
    """Copy any sandbox input file the substitution renames.

    A perturbed ask that names a file which does not exist is not a harder
    version of the task, it is an impossible one -- the reply would be graded
    on a FileNotFoundError that no worker could have avoided. Where the
    original input is sitting in the sandbox, the perturbed name gets a copy of
    it, so the only thing that changed is the name the worker has to read.
    """
    staged: dict[str, str] = {}
    for old, new in subs.items():
        if not _FILENAME_RE.fullmatch(old):
            continue
        source = constants.SANDBOX_DIR / old
        target = constants.SANDBOX_DIR / new
        try:
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
            if target.exists():
                staged[old] = new
        except OSError:
            continue
    return staged


def mint_cases(
    role: str, examples: list[tuple[str, str, bool]], config: dict[str, Any],
    wants_code: bool = True, max_cases: int = MAX_CASES,
) -> list[PerformCase]:
    """Build the perform battery from a skill's worked examples.

    `examples` is (request, output, verified) as produced by seeding, where
    `verified` means the example's script ran clean in the sandbox. Only
    verified examples are eligible: an example that could not be executed at
    mint time -- a blocked import, a dead target -- cannot have its perturbed
    twin executed either, so a case built from one could never pass.

    Every minted case is validated by running its own expected script. If the
    substitution broke something, the case is dropped rather than shipped: the
    battery's whole claim is that a passing answer exists, and a case that
    cannot pass would fail every retrain from here on.
    """
    from symbio.app import skills

    cases: list[PerformCase] = []
    for index, (request, output, verified) in enumerate(examples):
        if len(cases) >= max_cases:
            break
        if wants_code and not verified:
            continue
        fence = _CODE_FENCE.search(output or "")
        script = fence.group(1) if fence else ""
        if wants_code and not script.strip():
            continue
        subs = _candidate_values(request or "", script or output or "")
        if not subs:
            continue

        fixtures = _stage_fixtures(subs)
        case = PerformCase(
            id=f"perform_{index}",
            prompt=_apply(request, subs),
            substitutions=subs,
            expected_script=_apply(script, subs) if script else "",
            fixtures=fixtures,
            executable=bool(wants_code and script.strip()),
        )
        if case.executable:
            state, fault = skills._execution_feedback(case.expected_script, config)
            if state != "ran":
                # The perturbed script does not work, so nothing is learned by
                # asking a worker to reproduce it. Says which case and why --
                # a silently smaller battery is how a guard rail disappears.
                print(f"[perform] dropped {case.id} for '{role}': substituted "
                      f"script {state} ({fault})", flush=True)
                continue
        cases.append(case)
    return cases


def mint_and_save(
    role: str, examples: list[tuple[str, str, bool]], config: dict[str, Any],
    wants_code: bool = True,
) -> int:
    """Mint the battery and write it out. Returns the number of cases."""
    cases = mint_cases(role, examples, config, wants_code=wants_code)
    if not cases:
        return 0
    save_cases(role, cases)
    return len(cases)


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def restage_fixtures(cases: list[PerformCase]) -> None:
    """Re-create the perturbed input files a battery depends on.

    The sandbox is a working directory, not a store: it gets cleaned, and the
    copies made at mint time may be long gone by the time a retrain runs the
    battery. Without this the cases fail on a missing file and the failure
    reads as a regression in the worker.
    """
    for case in cases:
        for old, new in case.fixtures.items():
            source = constants.SANDBOX_DIR / old
            target = constants.SANDBOX_DIR / new
            try:
                if source.exists() and not target.exists():
                    shutil.copy2(source, target)
            except OSError:
                continue


def grade(
    reply: str, case: PerformCase, config: dict[str, Any], steps: str = "",
) -> tuple[bool, str]:
    """Grade one reply. Returns (passed, why).

    The order is deliberate: the cheap textual disqualifications run before
    anything is executed, so a reply that recited the procedure or echoed the
    memorised filename never reaches the sandbox at all.
    """
    from symbio.app import golden, skill_eval, skills, tooling

    if not reply or not reply.strip():
        return False, "empty reply"
    if not golden.sane_reply(tooling.strip_tool_tags(reply)):
        return False, "degenerate or leaked tool tags"

    # A recitation is the specific wrong answer this battery exists to catch:
    # the worker restating the runbook instead of carrying it out.
    if steps and skill_eval.recites_steps(reply, steps):
        return False, "recited the procedure instead of performing it"

    # Memorisation. The perturbed value is in the prompt the model just read;
    # the original is only in its weights. Emitting the original means it
    # answered from the training example rather than from the request.
    leaked = [old for old in case.forbidden if old and old in reply]
    if leaked:
        return False, f"used the memorised value(s) {', '.join(leaked)}"

    missing = [new for new in case.required if new and new not in reply]
    if missing:
        return False, f"never used the asked-for value(s) {', '.join(missing)}"

    if not case.executable:
        return True, "used the asked-for values"

    fence = _CODE_FENCE.search(reply)
    if fence is None:
        return False, "no runnable code in the reply"

    state, fault = skills._execution_feedback(fence.group(1), config)
    if state == "ran":
        return True, "ran clean on the perturbed input"
    if state == "unrun":
        # Seeding treats "could not be run here" as neither pass nor fail,
        # because rejecting it would empty the corpus for every skill whose
        # target happens to be down. That reasoning does not carry over to
        # this battery, and copying it here would have opened a hole: a case
        # exists only because its script ALREADY ran clean in this sandbox, so
        # the environment is known to be sufficient. An unrunnable reply is
        # therefore the reply reaching for something the sandbox blocks -- the
        # model's choice, not the machine's -- and passing it would let a
        # worker score full marks for `import requests` as long as it echoed
        # the right filename back. It fails, and says which import did it.
        return False, f"could not be run: {fault}"
    return False, f"failed when run: {fault}"


@dataclass
class PerformResult:
    results: dict[str, bool]
    reasons: dict[str, str]
    replies: dict[str, str]

    @property
    def passing(self) -> set[str]:
        return {case_id for case_id, ok in self.results.items() if ok}

    @property
    def pass_count(self) -> int:
        return sum(self.results.values())

    @property
    def total(self) -> int:
        return len(self.results)


def run_perform_set(
    model, tokenizer, generate_fn, sampler, system_prompt: str,
    config: dict[str, Any], cases: list[PerformCase], steps: str = "",
    max_tokens: int = 700,
) -> PerformResult:
    """Run the perform battery and grade every case by executing the answer.

    Graded greedily for the same reason golden.run_golden_set is: this result
    decides whether a fine-tune is kept, and a sampler that scores the same
    adapter differently run to run turns that decision into a coin flip.
    """
    from mlx_lm.sample_utils import make_sampler

    del sampler
    sampler = make_sampler(temp=0.0)

    restage_fixtures(cases)
    results: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    replies: dict[str, str] = {}
    print(f"  [Perform] Running {len(cases)} held-out performance check(s)...")
    for i, case in enumerate(cases, 1):
        print(f"  [Perform] {i}/{len(cases)} {case.id}...", end=" ", flush=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case.prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        try:
            reply = generate_fn(
                model, tokenizer, prompt=chat_prompt, sampler=sampler,
                max_tokens=max_tokens, verbose=False,
            ).strip()
        except Exception as e:
            results[case.id] = False
            reasons[case.id] = f"generation error: {e}"
            replies[case.id] = ""
            print("ERROR")
            continue
        replies[case.id] = reply
        try:
            ok, why = grade(reply, case, config, steps)
        except Exception as e:
            ok, why = False, f"grading error: {e}"
        results[case.id] = ok
        reasons[case.id] = why
        print("PASS" if ok else f"FAIL ({why})")

    print(f"  [Perform] {sum(results.values())}/{len(cases)} performance check(s) passed.")
    return PerformResult(results, reasons, replies)


def remedy_samples(
    cases: list[PerformCase], failing: list[str], tokenizer,
    system_prompt: str, role: str, copies: int = 2,
    passing: set[str] | None = None, passing_copies: int = 1,
) -> int:
    """Write training samples for a failed perform case -- alongside the ones
    it still gets right.

    The recall battery's remedy target is the steps text, which is right for a
    case that asks the worker to state the procedure and exactly wrong here --
    it would teach the worker to recite in answer to a request to perform, the
    behaviour the whole file exists to detect. The target here is the case's
    own expected script: the procedure carried out on the perturbed values.

    Both outcomes go in, and the reason is the one skills.py already states
    about corpora this small: "whichever behaviour is repeated most just wins".
    Injecting several copies of only the failures makes them the bulk of the
    delta, and a remedy that fixes one case by drowning out the rest has moved
    the failure rather than removed it. The passing cases go back in at a lower
    count as ballast, so the retrain is pulled toward the failure without being
    unanchored from what already worked.

    What deliberately does NOT go in is the negative half in its literal form
    -- the wrong answer, or a decline paired with the right one. That was tried
    in this exact corpus and reverted: recall coverage fell from ~98% to ~38%,
    legitimate requests started being declined, and the decline string became
    an attractor strong enough to degrade unrelated generation into "That that
    that". Every sample written here is a correct demonstration; the mixing is
    between cases the worker failed and cases it passed, not between right
    answers and wrong ones. Negative signal has one place it has been measured
    to help -- fed back into the prompt on a retry, as
    skills._seed_worked_examples does with the real traceback -- and that is
    generation, not training.
    """
    from symbio.app import training

    by_id = {case.id: case for case in cases}
    added = 0

    def _write(case: PerformCase, times: int) -> int:
        if not case.expected_script.strip():
            return 0
        answer = f"```python\n{case.expected_script.strip()}\n```"
        for _ in range(max(1, times)):
            training.append_chat_pair(
                case.prompt, answer, tokenizer, system_prompt, role=role)
        return max(1, times)

    for case_id in failing:
        case = by_id.get(case_id)
        if case is not None:
            added += _write(case, copies)

    if passing and passing_copies > 0:
        for case_id in sorted(passing):
            case = by_id.get(case_id)
            if case is not None and case_id not in set(failing):
                added += _write(case, passing_copies)
    return added
