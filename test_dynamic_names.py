#!/usr/bin/env python3
"""End-to-end test for dynamic name changes in chat.

This script runs the same detection path the CLI uses, using real multi-turn
history so name changes are tested the way a human actually uses them: a rename
utterance followed by a verification question. It restores config.json and
identity notes at the end.
"""
import json
from pathlib import Path

from mlx_lm import load

from symbio import AIAgent, load_config, save_config, ADAPTER_DIR, NOTES_DIR, maybe_update_names_from_message

PROJECT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_DIR / "config.json"

# Remember original names so we can restore them.
ORIGINAL = load_config().copy()


def run_turn(agent, text: str) -> tuple[str, str, str, bool]:
    """Return (assistant_prefix, user_prefix, final_reply_text, changed)."""
    changed = maybe_update_names_from_message(text, agent.config)
    if changed:
        agent.update_identity(
            agent.config["assistant_name"], agent.config["user_name"]
        )

    result = agent.run(text)
    return (
        agent.config["assistant_name"],
        agent.config["user_name"],
        result["text"],
        changed,
    )


def run_sequence(config, model, tokenizer, adapter_loaded: bool, turns: list[str]) -> tuple[str, str, str, list[tuple[str, str, str, bool]]]:
    """Run several turns on one agent and return the final state plus all replies."""
    agent = AIAgent(config, model, tokenizer, adapter_loaded)
    results = []
    for text in turns:
        results.append(run_turn(agent, text))
    final = results[-1]
    return final[0], final[1], final[2], results


def main():
    config = load_config()
    print(f"Loading {config['model_name']}...")
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    if adapter_config.exists():
        print("Found adapter. Loading with LoRA...")
        model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
    else:
        print("No adapter found. Using base model.")
        model, tokenizer = load(config["model_name"])

    # Each case: (label, list-of-turns, expected_user_name, expected_assistant_name)
    # Expected names must appear somewhere in the conversation replies.
    original_user = config["user_name"]
    cases = [
        ("user name baseline", ["What is my name?"], original_user, None),
        ("assistant name baseline", ["What is your name?"], None, config["assistant_name"]),
        ("user rename then verify", ["My name is Alice.", "What is my name?"], "Alice", None),
        ("call me rename then verify", ["Call me Bob.", "What is my name?"], "Bob", None),
        ("assistant rename then verify", ["Call yourself Jarvis.", "What is your name?"], None, "Jarvis"),
        ("both rename then verify", ["My name is Alice and call yourself Friday.", "What is my name?", "What is your name?"], "Alice", "Friday"),
        ("restore user name", [f"My name is {original_user}.", "What is my name?"], original_user, None),
        ("restore assistant name", [f"Call yourself {config['assistant_name']}.", "What is your name?"], None, config["assistant_name"]),
    ]

    all_ok = True
    for label, turns, expected_user, expected_assistant in cases:
        assistant, user, reply, results = run_sequence(
            config, model, tokenizer, adapter_config.exists(), turns
        )

        all_replies_text = " ".join(r[2] for r in results)
        checks = []
        if expected_user is not None and expected_user not in all_replies_text:
            checks.append(f"expected user name '{expected_user}' in replies")
        if expected_assistant is not None and expected_assistant not in all_replies_text:
            checks.append(f"expected assistant name '{expected_assistant}' in replies")

        status = "FAIL" if checks else "OK"
        if checks:
            all_ok = False
        print(f"\n[{status}] {label}: {turns!r}")
        print(f"  final state: assistant={assistant} user={user}")
        for i, (a, u, r, c) in enumerate(results):
            print(f"  turn {i + 1}: changed={c} assistant={a} user={u} reply={r!r}")
        if checks:
            print(f"  missing: {', '.join(checks)}")

    # Restore original names.
    save_config(ORIGINAL)
    (NOTES_DIR / "My_Identity.md").write_text(
        f"# My Identity\n\nI am {ORIGINAL['assistant_name']}, a helpful personal AI assistant.\n",
        encoding="utf-8",
    )
    (NOTES_DIR / "User_Identity.md").write_text(
        f"# User Identity\n\nMy user's name is {ORIGINAL['user_name']}.\n",
        encoding="utf-8",
    )
    print(f"\nRestored config: assistant={ORIGINAL['assistant_name']} user={ORIGINAL['user_name']}")

    if all_ok:
        print("\nAll dynamic-name tests passed.")
    else:
        print("\nSome tests failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
