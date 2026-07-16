"""Correction detection and batch learning for Symbio."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio.constants import DEFAULT_CONFIG, MISTAKES_ARCHIVE_DIR, MISTAKES_DIR
from symbio.llm import append_chat_pair, build_chat_training_sample, run_training
from symbio.utils import _safe_mistake_filename, strip_tool_tags


def _is_system_observation(content: str) -> bool:
    """Return True for tool-observation messages injected back into the chat history."""
    return content.startswith("[System observation")


def _is_correction(text: str, phrases: list[str]) -> bool:
    """Return True if the user message looks like a correction."""
    lowered = text.lower().strip(" \t\"'",)
    return any(phrase.lower() in lowered for phrase in phrases)


def _looks_like_correction(user_input: str, history: list[dict[str, str]], config: dict[str, Any]) -> tuple[bool, str]:
    """Check whether the latest user message is correcting the assistant's previous answer.

    Returns (is_correction, reason). Uses correction phrases first, then falls back to
    detecting when the user repeats a recent question that the assistant just answered.
    """
    learn_cfg = config.get("learn", {})
    if not learn_cfg.get("enabled", True) or not learn_cfg.get("auto", True):
        return False, ""

    if not user_input.strip() or user_input.startswith("/"):
        return False, ""

    phrases = learn_cfg.get("correction_phrases", DEFAULT_CONFIG["learn"]["correction_phrases"])
    if _is_correction(user_input, phrases):
        return True, "correction phrase"

    # Detect an exact/near-exact repeat of the question that was just answered.
    # This usually means the previous answer was wrong or incomplete.
    last_assistant_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is not None and last_assistant_idx >= 1:
        prior_query = ""
        for i in range(last_assistant_idx - 1, -1, -1):
            if history[i].get("role") == "user" and not _is_system_observation(history[i].get("content", "")):
                prior_query = history[i].get("content", "")
                break
        if prior_query:
            a = re.sub(r"[^\w]", "", user_input.lower())
            b = re.sub(r"[^\w]", "", prior_query.lower())
            if a and b and a == b:
                return True, "repeated question"
    return False, ""


def _find_correction_sample(history: list[dict[str, str]], config: dict[str, Any]) -> tuple[str, str] | None:
    """Mine the most recent (query, corrected_answer) pair from conversation history.

    Pattern:
      user: original question
      assistant: wrong answer
      user: correction phrase ("No, ...", "Actually ...")
      assistant: correct answer (may follow a short tool loop)
    Returns the original question and the final cleaned assistant answer.
    """
    learn_cfg = config.get("learn", {})
    phrases = learn_cfg.get("correction_phrases", DEFAULT_CONFIG["learn"]["correction_phrases"])

    def _is_real_user_turn(turn: dict[str, str]) -> bool:
        return turn.get("role") == "user" and not _is_system_observation(turn.get("content", ""))

    if len(history) < 4:
        return None

    # Identify the real user turns in the conversation.
    user_indices = [i for i, turn in enumerate(history) if _is_real_user_turn(turn)]
    if len(user_indices) < 2:
        return None

    correction_idx = user_indices[-1]
    correction_text = history[correction_idx].get("content", "")
    if not _is_correction(correction_text, phrases):
        return None

    original_idx = user_indices[-2]
    original_query = history[original_idx].get("content", "")
    if not original_query.strip():
        return None

    # Wrong answer: first assistant turn after the original question.
    wrong_idx = None
    for i in range(original_idx + 1, correction_idx):
        if history[i].get("role") == "assistant":
            wrong_idx = i
            break
    if wrong_idx is None:
        return None

    # Corrected answer: last assistant turn in the exchange that follows the correction.
    next_user_idx = len(history)
    for i in range(correction_idx + 1, len(history)):
        if _is_real_user_turn(history[i]):
            next_user_idx = i
            break
    correct_idx = None
    for i in range(correction_idx + 1, next_user_idx):
        if history[i].get("role") == "assistant":
            correct_idx = i
    if correct_idx is None:
        return None

    wrong_answer = strip_tool_tags(history[wrong_idx].get("content", ""))
    correct_answer = strip_tool_tags(history[correct_idx].get("content", ""))
    if not wrong_answer.strip() or not correct_answer.strip():
        return None
    return original_query, correct_answer


def _mistake_note_count() -> int:
    """Return the number of unarchived mistake notes."""
    if not MISTAKES_DIR.exists():
        return 0
    return len([f for f in MISTAKES_DIR.glob("*.md") if f.is_file()])


def _save_mistake_note(
    original_query: str,
    wrong_answer: str,
    correction: str,
    correct_answer: str,
) -> Path:
    """Persist a correction as a markdown note in notes/mistakes/."""
    title = f"Correction: {original_query[:60]}{'...' if len(original_query) > 60 else ''}"
    body = (
        f"# {title}\n\n"
        f"**Original question:** {original_query}\n\n"
        f"**Wrong answer:** {wrong_answer}\n\n"
        f"**Correction:** {correction}\n\n"
        f"**Correct answer:** {correct_answer}\n"
    )
    path = MISTAKES_DIR / _safe_mistake_filename(original_query)
    # Avoid overwriting an existing note from the same second.
    counter = 1
    original_path = path
    while path.exists():
        stem = original_path.stem
        path = original_path.with_name(f"{stem}_{counter}{original_path.suffix}")
        counter += 1
    path.write_text(body, encoding="utf-8")
    return path


def _archive_mistake_notes() -> int:
    """Move all unarchived mistake notes into notes/mistakes/archive/."""
    archived = 0
    for f in MISTAKES_DIR.glob("*.md"):
        if not f.is_file():
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = MISTAKES_ARCHIVE_DIR / f"{ts}_{f.name}"
        # Avoid collisions.
        while dest.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dest = MISTAKES_ARCHIVE_DIR / f"{ts}_{f.name}"
        f.rename(dest)
        archived += 1
    return archived


def _digest_mistakes_to_training(
    tokenizer, system_prompt: str, planner: Any | None = None, boost: int = 1
) -> int:
    """Convert unarchived mistake notes into training samples and archive them."""
    files = sorted(MISTAKES_DIR.glob("*.md"))
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
            append_chat_pair(original_query, correct_answer, tokenizer, system_prompt)
            if planner is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_query},
                    {"role": "assistant", "content": correct_answer},
                ]
                text = build_chat_training_sample(messages, tokenizer)
                planner.add_sample(text, source=f"mistake:{f.name}")
        added += 1

    _archive_mistake_notes()
    return added


def learn_from_last_correction(agent) -> Path | None:
    """Extract the last correction, persist it as a mistake note, and trigger batch training if threshold is reached.

    Returns the path to the saved mistake note, or None if no correction was found.
    """
    config = agent.config
    learn_cfg = config.get("learn", {})
    if not learn_cfg.get("enabled", True):
        print("  /learn is disabled in config.")
        return None

    sample = _find_correction_sample(agent.history, config)
    if sample is None:
        print("  No recent correction detected. Say something like \"No, the answer is ...\" and run /learn again.")
        return None

    original_query, correct_answer = sample

    # Recover the user's correction text and the assistant's earlier wrong answer from history.
    wrong_answer = ""
    correction_text = ""
    for i in range(len(agent.history) - 1, -1, -1):
        turn = agent.history[i]
        if turn.get("role") == "assistant" and not wrong_answer:
            # The first assistant we hit walking backward is the correct answer; keep going.
            continue
        if turn.get("role") == "user" and not _is_system_observation(turn.get("content", "")):
            if not correction_text:
                correction_text = turn.get("content", "")
            elif not wrong_answer:
                # We need the assistant's wrong answer that came after this correction.
                # Find the next assistant above this user turn.
                for j in range(i + 1, len(agent.history)):
                    if agent.history[j].get("role") == "assistant":
                        wrong_answer = strip_tool_tags(agent.history[j].get("content", ""))
                        break
                break

    if not wrong_answer:
        # Fallback: use the second-to-last assistant reply if available.
        assistant_turns = [t for t in agent.history if t.get("role") == "assistant"]
        if len(assistant_turns) >= 2:
            wrong_answer = strip_tool_tags(assistant_turns[-2].get("content", ""))

    path = _save_mistake_note(original_query, wrong_answer, correction_text, correct_answer)
    agent.planner.record_correction(original_query)
    print(f"  Saved mistake note: {path.name}")
    print(f"  Original: \"{original_query[:60]}{'...' if len(original_query) > 60 else ''}\"")
    return path


def maybe_train_on_mistakes(
    config: dict[str, Any],
    tokenizer,
    system_prompt: str,
    agent,
) -> bool:
    """If enough mistake notes have accumulated, digest them and run a LoRA update."""
    learn_cfg = config.get("learn", {})
    if not learn_cfg.get("enabled", True):
        return False

    threshold = max(1, int(learn_cfg.get("mistake_threshold", DEFAULT_CONFIG["learn"]["mistake_threshold"])))
    count = _mistake_note_count()
    if count < threshold:
        print(f"  {count}/{threshold} mistake note(s) collected. Training will run after {threshold - count} more correction(s).")
        return False

    print(f"\n  [System] {count} mistake note(s) reached. Digesting into training data...")
    boost = max(1, int(learn_cfg.get("boost_factor", DEFAULT_CONFIG["learn"]["boost_factor"])))
    digested = _digest_mistakes_to_training(
        tokenizer, system_prompt, agent.planner if agent else None, boost=boost
    )
    print(f"  Digested {digested} mistake note(s) into training data (boost={boost}).")

    if not learn_cfg.get("auto_train", True):
        print("  Auto-train is disabled. Run /train --force to fine-tune now.")
        return False

    iters = int(learn_cfg.get("batch_train_iters", DEFAULT_CONFIG["learn"]["batch_train_iters"]))
    print(f"  [System] Running LoRA update ({iters} iters)...")
    trained = run_training(config, model_type=config.get("_model_type", "dense"), iters=iters)
    if not trained:
        print("  Training did not complete. Mistake notes are kept for the next attempt.")
        return False

    print(f"  [System] Archived {_mistake_note_count()} remaining mistake note(s).")
    return True
