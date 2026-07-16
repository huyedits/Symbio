#!/usr/bin/env python3
import json
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from symbio import build_system_prompt, load_config

PROJECT_DIR = Path(__file__).parent.resolve()
ADAPTER_DIR = PROJECT_DIR / "adapters"


def main():
    config = load_config()
    model, tokenizer = load(
        config["model_name"],
        adapter_path=str(ADAPTER_DIR),
    )

    # Use the exact system prompt the agent uses.
    system = build_system_prompt(config["assistant_name"], config["user_name"], [])

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Remember that I like coffee."},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("PROMPT:")
    print(prompt[-800:])
    print("\nGENERATED:")
    sampler = make_sampler(temp=config["agent"]["temperature"], top_p=config["agent"]["top_p"])
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=200,
        sampler=sampler,
    )
    print(repr(output))


if __name__ == "__main__":
    main()
