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


# Queries/answers about the current moment go stale immediately — training
# them into weights would teach outdated facts, so they are never remembered.
_EPHEMERAL_MARKERS = (
    "weather", "news", "headline", "today", "tonight", "tomorrow", "yesterday",
    "right now", "currently", "latest", "price", "stock", "score", "forecast",
)


def remember_research(question: str, answer: str, config: dict[str, Any]) -> Path | None:
    """Save a web-researched answer as a 'Learned:' note so it is retrievable
    by RAG and trained into the weights on the next digest. Skips ephemeral
    lookups, trivial answers, and questions already remembered."""
    if not config.get("learn", {}).get("remember_research", True):
        return None
    question = question.strip()
    answer = answer.strip()
    if len(answer) < 20:
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


def _safe_mistake_filename(query: str) -> str:
    slug = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in query)
    slug = slug.strip().replace(" ", "_")[:40].strip("_") or "correction"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slug}.md"


def save_mistake_note(original_query: str, wrong_answer: str,
                      correction: str, correct_answer: str) -> Path:
    """Persist a correction as a markdown note in notes/mistakes/."""
    title = f"Correction: {original_query[:60]}{'...' if len(original_query) > 60 else ''}"
    body = (
        f"# {title}\n\n"
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


def digest_mistakes_to_training(tokenizer, system_prompt: str, boost: int = 1) -> int:
    """Convert unarchived mistake notes into (boosted) training samples that
    pair the original question with the corrected answer, then archive them."""
    files = sorted(constants.MISTAKES_DIR.glob("*.md"))
    if not files:
        return 0

    added = 0
    for f in files:
        content = f.read_text(encoding="utf-8").strip()
        if not content:
            continue
        original_query = ""
        correct_answer = ""
        for line in content.splitlines():
            if line.startswith("**Original question:**"):
                original_query = line.split("**Original question:**", 1)[1].strip()
            elif line.startswith("**Correct answer:**"):
                correct_answer = line.split("**Correct answer:**", 1)[1].strip()
        if not original_query or not correct_answer:
            continue
        for _ in range(max(1, boost)):
            training.append_chat_pair(original_query, correct_answer, tokenizer, system_prompt)
        added += 1

    archive_mistake_notes()
    return added


def maybe_train_on_mistakes(config: dict[str, Any], tokenizer, system_prompt: str) -> bool:
    """If enough mistake notes have accumulated, digest them and run a short
    LoRA pass. Returns True when training completed (caller reloads model)."""
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
    digested = digest_mistakes_to_training(tokenizer, system_prompt, boost=boost)
    print(f"  [Learn] Digested {digested} mistake note(s) (boost={boost}).")

    if not learn_cfg.get("auto_train", True):
        print("  [Learn] Auto-train is disabled. Run /train to fine-tune now.")
        return False

    iters = int(learn_cfg.get("batch_train_iters", 25))
    print(f"  [Learn] Running LoRA update ({iters} iters)...")
    trained = training.run_training(config, iters=iters)
    if not trained:
        print("  [Learn] Training did not complete; the digested samples remain "
              "in training data for the next run.")
        return False
    return True
