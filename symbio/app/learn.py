"""Self-correction: detect when the user corrects a wrong answer, save the
mistake, and retrain on the corrected answers once enough accumulate.

Flow per correction:
  user: original question
  assistant: wrong answer
  user: correction ("No, ...", "Actually ...", or repeating the question)
  assistant: corrected answer (possibly after a tool loop)
The (question -> corrected answer) pair is saved as a mistake note in
notes/mistakes/; at `learn.mistake_threshold` notes they are digested into
boosted training samples and a short LoRA pass runs.

Ported from the legacy Hermes agent's symbio.learn, adapted to the tag-based
agent (app paths, tag stripping, and iters-override training).
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio import constants
from symbio.app import memory, training
from symbio.app.tooling import strip_tool_tags


# Phrases that signal the model is answering from a gap in its knowledge.
# A reply that sounds like this and used no tools triggers an automatic web
# search so the model answers from results instead of guessing.
_UNSURE_MARKERS = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i'm not certain", "i am not certain", "i'm uncertain", "not sure about",
    "i don't have", "i do not have", "don't have access", "do not have access",
    "i'm unable to", "i am unable to", "i cannot answer", "can't answer",
    "no information", "not aware of", "i'm not familiar", "i am not familiar",
    "knowledge cutoff", "my training data", "as an ai", "i might be wrong",
    "i may be wrong", "hard to say", "can't say for sure", "cannot say for sure",
)


def sounds_unsure(text: str) -> bool:
    """Does this reply sound like the model is guessing or lacks the fact?"""
    lowered = text.lower()
    return any(marker in lowered for marker in _UNSURE_MARKERS)


# Every failure phrasing used across _execute_tool's branches and
# sandbox.py/computer.py: "Command 'X' exited error", "Web search ... X
# failed", "Tool 'X' is disabled", "Failed to save note: ...", "Could not
# schedule job: ...", "Browser click error: ...", "Click failed: ...",
# domain-approval "blocked", worker delegation "unrecognized action" /
# "did not finish", and _execute_tool's own catch-all "failed
# unexpectedly" backstop for anything a tool didn't handle itself.
# Deliberately anchored (line-start, or a marker word immediately followed
# by ':'/'.') rather than a bare substring check: a *successful* search for
# something like "database error fixes" or "how to fix blocked drains"
# would otherwise falsely look like a failure just because the
# user-controlled query text happens to contain that word.
_TOOL_ERROR_RE = re.compile(
    r"^(?:failed|could not|no worker configured|browser \w+ (?:error|blocked))"
    r"|\b(?:exited error|is disabled|unrecognized action|did not finish|failed unexpectedly)\b"
    r"|\b(?:error|failed|blocked)[:.]",
    re.IGNORECASE,
)


def sounds_like_tool_error(observation: str) -> bool:
    """Did a tool observation's status indicate failure? Checked against
    just the status line (before the first newline/section) so a genuine
    success whose CONTENT happens to mention "error" — a search result
    about a bug, say — is never mistaken for a failed call."""
    status_line = observation.split("\n", 1)[0]
    return bool(_TOOL_ERROR_RE.search(status_line))


# The other way the model fills a knowledge gap: inventing a plausible-looking
# figure instead of admitting it doesn't know. Detection is deliberately
# two-sided so auto-search fires in moderation: the QUESTION must ask for a
# specific figure or date, AND the REPLY must hedge right next to a number
# ("around 300 metres, I think"). A confidently stated number never triggers —
# if it's wrong, the correction pipeline handles it.
_NUMERIC_QUESTION_MARKERS = (
    "how many", "how much", "how tall", "how old", "how far", "how long",
    "how fast", "how heavy", "how big", "how deep", "how high", "how wide",
    "how often", "what year", "when did", "when was", "what date",
    "population", "percent", "temperature", "elevation", "altitude",
    "distance", "capacity", "net worth", "box office", "gdp", "market cap",
    "record for", "the record", "how large", "what size",
)

_HEDGE_BEFORE_NUMBER_RE = re.compile(
    r"(?:\babout|\baround|\bapproximately|\broughly|\bmaybe|\bperhaps|"
    r"\bprobably|\blikely|\bi think|\bi believe|\bi'd guess|\bi would guess|"
    r"\bif i recall|\bif i remember|\bestimated|\bsomewhere|\bpossibly|~)"
    r"[^.!?\n]{0,40}?\d"
)
_HEDGE_AFTER_NUMBER_RE = re.compile(
    r"\d[^.!?\n]{0,40}?"
    r"(?:\bor so\b|\bgive or take\b|\bi think\b|\bi believe\b|"
    r"\bif i recall\b|\bif i remember\b|\bbut i'm not sure\b|"
    r"\bbut i am not sure\b|\bdon't quote me\b)"
)


def sounds_fabricated(question: str, reply: str) -> bool:
    """Does this reply hedge a specific figure for a question that asked for
    one? That pattern usually means the number is invented, not recalled."""
    q = question.lower()
    if not any(marker in q for marker in _NUMERIC_QUESTION_MARKERS):
        return False
    r = reply.lower()
    return bool(_HEDGE_BEFORE_NUMBER_RE.search(r) or _HEDGE_AFTER_NUMBER_RE.search(r))


# Queries/answers about the current moment go stale immediately — training
# them into weights would teach outdated facts, so they are never remembered.
_EPHEMERAL_MARKERS = (
    "weather", "news", "headline", "today", "tonight", "tomorrow", "yesterday",
    "right now", "currently", "latest", "price", "stock", "forecast",
)


def remember_research(question: str, answer: str, config: dict[str, Any]) -> Path | None:
    """Save a web-researched answer as a 'Learned:' note so it is retrievable
    by RAG and trained into the weights on the next digest. Skips ephemeral
    lookups, trivial answers, short/acknowledgment questions, and already
    remembered topics."""
    if not config.get("learn", {}).get("remember_research", True):
        return None
    question = question.strip()
    answer = answer.strip()
    if len(answer) < 20:
        return None
    # Never remember research triggered by trivial acknowledgments like "ok".
    q_lower = question.lower()
    if len(question.split()) <= 2 and any(
        marker in q_lower for marker in
        ("ok", "okay", "yes", "sure", "go on", "go ahead", "continue", "proceed")
    ):
        return None
    text = f"{question} {answer}".lower()
    if any(marker in text for marker in _EPHEMERAL_MARKERS):
        return None

    title = f"Learned: {question[:60]}{'...' if len(question) > 60 else ''}"
    # Light dedupe: skip if a note with this exact title already exists.
    for f in constants.NOTES_DIR.glob("*.md"):
        try:
            if f.read_text(encoding="utf-8").splitlines()[0] == f"# {title}":
                return None
        except (OSError, IndexError):
            continue

    body = f"**Question:** {question}\n\n**Answer (from web research):** {answer}"
    return memory.save_note(title, body)


def _is_system_observation(content: str) -> bool:
    return content.startswith("[System observation")


def _is_real_user_turn(turn: dict[str, str]) -> bool:
    return turn.get("role") == "user" and not _is_system_observation(turn.get("content", ""))


def _is_correction(text: str, phrases: list[str]) -> bool:
    lowered = text.lower().strip(" \t\"'",)
    return any(phrase.lower() in lowered for phrase in phrases)


def looks_like_correction(user_input: str, history: list[dict[str, str]],
                          config: dict[str, Any]) -> bool:
    """Is this user message correcting the assistant's previous answer?
    Call BEFORE appending user_input to history. Uses correction phrases,
    then falls back to detecting a repeat of the question just answered."""
    learn_cfg = config.get("learn", {})
    if not learn_cfg.get("enabled", True) or not learn_cfg.get("auto", True):
        return False
    if not user_input.strip() or user_input.startswith("/"):
        return False
    if not any(t.get("role") == "assistant" for t in history):
        return False

    if _is_correction(user_input, learn_cfg.get("correction_phrases", [])):
        return True

    # An exact repeat of the question that was just answered usually means
    # the previous answer was wrong or incomplete.
    prior_query = ""
    for turn in reversed(history):
        if _is_real_user_turn(turn):
            prior_query = turn.get("content", "")
            break
    a = re.sub(r"[^\w]", "", user_input.lower())
    b = re.sub(r"[^\w]", "", prior_query.lower())
    return bool(a) and a == b


def find_correction_sample(history: list[dict[str, str]], config: dict[str, Any],
                           ) -> tuple[str, str, str, str] | None:
    """Mine the most recent correction from history.

    Returns (original_query, wrong_answer, correction_text, correct_answer)
    or None. Expects history to already contain the corrected answer."""
    learn_cfg = config.get("learn", {})
    phrases = learn_cfg.get("correction_phrases", [])

    if len(history) < 4:
        return None
    user_indices = [i for i, t in enumerate(history) if _is_real_user_turn(t)]
    if len(user_indices) < 2:
        return None

    correction_idx = user_indices[-1]
    correction_text = history[correction_idx].get("content", "")
    original_idx = user_indices[-2]
    original_query = history[original_idx].get("content", "")

    is_repeat = (
        re.sub(r"[^\w]", "", correction_text.lower())
        == re.sub(r"[^\w]", "", original_query.lower())
    )
    if not (_is_correction(correction_text, phrases) or is_repeat):
        return None
    if not original_query.strip():
        return None

    # Wrong answer: first assistant turn after the original question.
    wrong_idx = next(
        (i for i in range(original_idx + 1, correction_idx)
         if history[i].get("role") == "assistant"), None)
    if wrong_idx is None:
        return None

    # Corrected answer: last assistant turn after the correction.
    correct_idx = None
    for i in range(correction_idx + 1, len(history)):
        if _is_real_user_turn(history[i]):
            break
        if history[i].get("role") == "assistant":
            correct_idx = i
    if correct_idx is None:
        return None

    wrong_answer = strip_tool_tags(history[wrong_idx].get("content", ""))
    correct_answer = strip_tool_tags(history[correct_idx].get("content", ""))
    if not wrong_answer.strip() or not correct_answer.strip():
        return None
    return original_query, wrong_answer, correction_text, correct_answer


# How hard did the user push back? Severity scales both the per-note training
# boost and the LoRA iteration count, so worse mistakes are trained harder.
# Levels: 1 = mild correction ("actually, I meant..."), 2 = the user says
# outright the answer is wrong, 3 = the model repeats a mistake it was
# already corrected for.
_SEVERE_CORRECTION_DEFAULTS = [
    "wrong", "incorrect", "you misunderstood", "fix it", "not what",
]


def _norm_question(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


def _was_corrected_before(original_query: str) -> bool:
    """Does a mistake note (pending or archived) already exist for this same
    question? If so the model is repeating a corrected mistake."""
    target = _norm_question(original_query)
    if not target:
        return False
    for directory in (constants.MISTAKES_DIR, constants.MISTAKES_ARCHIVE_DIR):
        if not directory.exists():
            continue
        for f in directory.glob("*.md"):
            try:
                content = f.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in content.splitlines():
                if line.startswith("**Original question:**"):
                    if _norm_question(line.split("**Original question:**", 1)[1]) == target:
                        return True
                    break
    return False


def correction_severity(original_query: str, correction_text: str,
                        config: dict[str, Any]) -> int:
    """Grade a correction 1-3. Call BEFORE saving the new mistake note, or
    the repeat check will match the note being saved."""
    if _was_corrected_before(original_query):
        return 3
    phrases = config.get("learn", {}).get(
        "severe_correction_phrases", _SEVERE_CORRECTION_DEFAULTS)
    if _is_correction(correction_text, phrases):
        return 2
    return 1


def _safe_mistake_filename(query: str) -> str:
    slug = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in query)
    slug = slug.strip().replace(" ", "_")[:40].strip("_") or "correction"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slug}.md"


def save_mistake_note(original_query: str, wrong_answer: str,
                      correction: str, correct_answer: str,
                      severity: int = 1) -> Path:
    """Persist a correction as a markdown note in notes/mistakes/."""
    # digest_mistakes_to_training parses "**Original question:**"/"**Correct
    # answer:**" as single lines; a value with embedded newlines (e.g. a
    # multi-line tool observation or a bulleted reply) would silently
    # truncate to just its first line otherwise.
    original_query = original_query.replace("\n", " ")
    wrong_answer = wrong_answer.replace("\n", " ")
    correction = correction.replace("\n", " ")
    correct_answer = correct_answer.replace("\n", " ")
    title = f"Correction: {original_query[:60]}{'...' if len(original_query) > 60 else ''}"
    body = (
        f"# {title}\n\n"
        f"**Severity:** {max(1, int(severity))}\n\n"
        f"**Original question:** {original_query}\n\n"
        f"**Wrong answer:** {wrong_answer}\n\n"
        f"**Correction:** {correction}\n\n"
        f"**Correct answer:** {correct_answer}\n"
    )
    path = constants.MISTAKES_DIR / _safe_mistake_filename(original_query)
    counter = 1
    original_path = path
    while path.exists():
        path = original_path.with_name(f"{original_path.stem}_{counter}{original_path.suffix}")
        counter += 1
    path.write_text(body, encoding="utf-8")
    return path


def mistake_note_count() -> int:
    if not constants.MISTAKES_DIR.exists():
        return 0
    return len([f for f in constants.MISTAKES_DIR.glob("*.md") if f.is_file()])


def archive_mistake_notes() -> int:
    """Move all unarchived mistake notes into notes/mistakes/archive/."""
    archived = 0
    for f in constants.MISTAKES_DIR.glob("*.md"):
        if not f.is_file():
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = constants.MISTAKES_ARCHIVE_DIR / f"{ts}_{f.name}"
        while dest.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dest = constants.MISTAKES_ARCHIVE_DIR / f"{ts}_{f.name}"
        f.rename(dest)
        archived += 1
    return archived


def digest_mistakes_to_training(tokenizer, system_prompt: str, boost: int = 1) -> tuple[int, int]:
    """Convert unarchived mistake notes into (boosted) training samples that
    pair the original question with the corrected answer, then archive them.
    Severity multiplies the per-note boost so worse mistakes are repeated
    harder. Returns (notes digested, summed severity)."""
    files = sorted(constants.MISTAKES_DIR.glob("*.md"))
    if not files:
        return 0, 0

    added = 0
    total_severity = 0
    for f in files:
        content = f.read_text(encoding="utf-8").strip()
        if not content:
            continue
        original_query = ""
        correct_answer = ""
        severity = 1
        for line in content.splitlines():
            if line.startswith("**Original question:**"):
                original_query = line.split("**Original question:**", 1)[1].strip()
            elif line.startswith("**Correct answer:**"):
                correct_answer = line.split("**Correct answer:**", 1)[1].strip()
            elif line.startswith("**Severity:**"):
                try:
                    severity = max(1, int(line.split("**Severity:**", 1)[1].strip()))
                except ValueError:
                    pass
        if not original_query or not correct_answer:
            continue
        for _ in range(max(1, boost) * severity):
            training.append_chat_pair(original_query, correct_answer, tokenizer, system_prompt)
        added += 1
        total_severity += severity

    archive_mistake_notes()
    return added, total_severity


def maybe_train_on_mistakes(config: dict[str, Any], tokenizer, system_prompt: str,
                            train_fn=None) -> bool:
    """If enough mistake notes have accumulated, digest them and run a short
    LoRA pass. Returns True when training completed (caller reloads model).
    `train_fn(config, iters=...)` defaults to training.run_training; pass a
    wrapper (e.g. one that golden-checks and rolls back a regression) to
    guard this path the same way as manual /train."""
    train_fn = train_fn or training.run_training
    learn_cfg = config.get("learn", {})
    if not learn_cfg.get("enabled", True):
        return False

    threshold = max(1, int(learn_cfg.get("mistake_threshold", 5)))
    count = mistake_note_count()
    if count < threshold:
        print(f"  [Learn] {count}/{threshold} mistake note(s) collected; "
              f"training after {threshold - count} more.")
        return False

    print(f"\n  [Learn] {count} mistake note(s) reached. Digesting into training data...")
    boost = max(1, int(learn_cfg.get("boost_factor", 3)))
    digested, total_severity = digest_mistakes_to_training(tokenizer, system_prompt, boost=boost)
    print(f"  [Learn] Digested {digested} mistake note(s) "
          f"(boost={boost}, total severity={total_severity}).")

    if not learn_cfg.get("auto_train", True):
        print("  [Learn] Auto-train is disabled. Run /train to fine-tune now.")
        return False

    # Scale iterations with severity above the mild baseline: an all-mild
    # batch trains at exactly batch_train_iters; each severity point beyond
    # that adds iters_per_severity, capped so a harsh backlog can't run away.
    base_iters = int(learn_cfg.get("batch_train_iters", 25))
    per_severity = int(learn_cfg.get("iters_per_severity", 5))
    cap = max(base_iters, int(learn_cfg.get("max_batch_train_iters", 100)))
    iters = min(cap, base_iters + per_severity * max(0, total_severity - digested))
    if iters != base_iters:
        print(f"  [Learn] Severity {total_severity} across {digested} note(s) "
              f"scales training from {base_iters} to {iters} iters.")
    print(f"  [Learn] Running LoRA update ({iters} iters)...")
    trained = train_fn(config, iters=iters)
    if not trained:
        print("  [Learn] Training did not complete; the digested samples remain "
              "in training data for the next run.")
        return False
    return True
