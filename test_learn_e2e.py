#!/usr/bin/env python3
"""End-to-end /learn smoke test: load dense 3B + adapter, simulate a correction, run /learn."""
import sys
from pathlib import Path

from mlx_lm import load

from symbio import AIAgent, DEFAULT_CONFIG, load_config, ADAPTER_DIR, learn_from_last_correction, run_training


def main():
    config = load_config()
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    adapter_loaded = adapter_config.exists()

    print(f"Loading {config['model_name']}...")
    if adapter_loaded:
        print("Found adapter. Loading with LoRA...")
        model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
    else:
        print("No adapter found. Using base model.")
        model, tokenizer = load(config["model_name"])

    agent = AIAgent(config, model, tokenizer, adapter_loaded)

    # Simulate the correction loop.
    agent.run("What is my name?")
    agent.run("No, I'm Alice.")

    print("\n--- Running /learn on last correction ---")
    added = learn_from_last_correction(agent)
    if not added:
        print("/learn did not detect a correction. Aborting.")
        sys.exit(1)

    model_type = config.get("_model_type", "dense")
    iters = config.get("learn", {}).get("short_train_iters", DEFAULT_CONFIG["learn"]["short_train_iters"])
    print(f"\nRunning short LoRA update ({iters} iters)...")
    trained = run_training(config, model_type=model_type, iters=iters)
    if trained and adapter_config.exists():
        print("Reloading adapter...")
        agent.reload_adapter()
        print("Adapter reloaded.")

    print("\n--- Re-testing the original query ---")
    result = agent.run("What is my name?")
    print(f"Symbio: {result['text']}")
    print("\n/learn end-to-end test complete.")


if __name__ == "__main__":
    from test_utils import preserve_training_state

    with preserve_training_state(adapters=True):
        main()
