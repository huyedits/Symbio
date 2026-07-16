#!/usr/bin/env python3
"""End-to-end test for automatic correction learning."""
from mlx_lm import load

from symbio import AIAgent, ADAPTER_DIR, _looks_like_correction, learn_from_last_correction, load_config, run_training


def main():
    config = load_config()
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    adapter_loaded = adapter_config.exists()

    print(f"Loading {config['model_name']}...")
    if adapter_loaded:
        model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
    else:
        model, tokenizer = load(config["model_name"])

    agent = AIAgent(config, model, tokenizer, adapter_loaded)

    # First turn: wrong answer.
    print("\n--- Turn 1 ---")
    agent.run("What is my name?")

    # Second turn: natural correction.
    correction_input = "No, I'm Alice."
    print(f"\n--- Turn 2 (user says: {correction_input}) ---")
    is_corr, reason = _looks_like_correction(correction_input, agent.history, config)
    print(f"Auto-detected correction: {is_corr} ({reason})")
    assert is_corr, "Expected auto-detection of correction phrase"
    agent.run(correction_input)

    print("\n--- Auto-learning from correction ---")
    note_path = learn_from_last_correction(agent)
    assert note_path is not None, "Expected a correction sample to be mined"

    iters = config.get("learn", {}).get("short_train_iters", 10)
    print(f"\n--- Short LoRA update ({iters} iters) ---")
    trained = run_training(config, model_type=config.get("_model_type", "dense"), iters=iters)
    if trained and adapter_config.exists():
        agent.reload_adapter()
        print("Adapter reloaded.")

    print("\n--- Re-test original query ---")
    result = agent.run("What is my name?")
    print(f"Symbio: {result['text']}")
    print("\nAuto-learn end-to-end test complete.")


if __name__ == "__main__":
    from test_utils import preserve_training_state

    with preserve_training_state(adapters=True):
        main()
