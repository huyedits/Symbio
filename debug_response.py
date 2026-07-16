#!/usr/bin/env python3
import json
from pathlib import Path

from mlx_lm import load
from symbio import AIAgent, load_config

PROJECT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_DIR / "config.json"
ADAPTER_DIR = PROJECT_DIR / "adapters"


def main():
    config = load_config()
    print(f"Loading {config['model_name']}...")
    model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
    agent = AIAgent(config, model, tokenizer, True)

    prompts = [
        "Remember that I like coffee.",
        "Show me config.json.",
    ]

    for prompt in prompts:
        agent.history.clear()
        agent._code_calls_this_turn = 0
        print("\n=== PROMPT:", prompt)
        result = agent.run(prompt)
        for i, turn in enumerate(result["history"]):
            print(f"  [{i}] {turn['role']}: {turn['content'][:500]!r}")
        print(f"  FINAL TEXT: {result['text']!r}")


if __name__ == "__main__":
    main()
