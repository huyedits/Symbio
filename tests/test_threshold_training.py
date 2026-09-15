#!/usr/bin/env python3
"""End-to-end test that fine-tunes only after the mistake threshold is reached."""
from mlx_lm import load

from symbio import ADAPTER_DIR, AIAgent, learn_from_last_correction, load_config, maybe_train_on_mistakes, _mistake_note_count


def _reset_mistakes():
    from symbio import MISTAKES_ARCHIVE_DIR, MISTAKES_DIR
    for d in (MISTAKES_DIR, MISTAKES_ARCHIVE_DIR):
        for f in d.glob("*.md"):
            f.unlink()


def main():
    _reset_mistakes()

    config = load_config()
    # Force threshold to 2 and low iters so the test completes quickly.
    config["learn"]["mistake_threshold"] = 2
    config["learn"]["batch_train_iters"] = 10

    adapter_config = ADAPTER_DIR / "adapter_config.json"
    adapter_loaded = adapter_config.exists()

    print(f"Loading {config['model_name']}...")
    if adapter_loaded:
        model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
    else:
        model, tokenizer = load(config["model_name"])

    agent = AIAgent(config, model, tokenizer, adapter_loaded)

    # First correction.
    print("\n--- Correction 1 ---")
    agent.run("What is my name?")
    agent.run("No, I'm Alice.")
    learn_from_last_correction(agent)
    print(f"Mistake notes: {_mistake_note_count()}")
    trained = maybe_train_on_mistakes(config, tokenizer, agent.system_prompt, agent)
    assert not trained, "Should not train after only 1 mistake note"

    # Second correction.
    print("\n--- Correction 2 ---")
    agent.run("What is my name?")
    agent.run("Actually I'm Bob.")
    learn_from_last_correction(agent)
    print(f"Mistake notes: {_mistake_note_count()}")
    trained = maybe_train_on_mistakes(config, tokenizer, agent.system_prompt, agent)
    assert trained, "Should train after reaching threshold"

    if adapter_config.exists():
        agent.reload_adapter()
        print("Adapter reloaded.")

    print("\n--- Re-test original query ---")
    result = agent.run("What is my name?")
    print(f"Symbio: {result['text']}")
    print("\nThreshold training test complete.")


if __name__ == "__main__":
    from test_utils import preserve_training_state

    with preserve_training_state(adapters=True):
        main()
        _reset_mistakes()
