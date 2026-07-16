#!/usr/bin/env python3
"""Symbio: thin backward-compatible CLI entry point.

This module re-exports the public API from the `symbio` package so existing
scripts that do `from main import AIAgent, load_config` continue to work.
"""

import argparse
from typing import Any

from mlx_lm import load

from symbio import (
    AIAgent,
    DEFAULT_CONFIG,
    build_system_prompt,
    chat_loop,
    detect_model_type,
    ensure_seed_notes,
    load_config,
    run_training,
    save_config,
    seed_training_data,
    setup_names,
)
from symbio.config import _adapter_matches_model
from symbio.constants import ADAPTER_DIR, CONFIG_FILE, TRAIN_FILE


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Symbio: Autonomous Chat + LoRA")
    parser.add_argument("--train", action="store_true", help="Run training and exit")
    parser.add_argument("--model", type=str, default=config["model_name"], help="Base model")
    parser.add_argument("--assistant-name", type=str, default=config["assistant_name"], help="Assistant name")
    parser.add_argument("--user-name", type=str, default=config["user_name"], help="User name")
    args = parser.parse_args()

    config["model_name"] = args.model
    config["assistant_name"] = args.assistant_name
    config["user_name"] = args.user_name

    if args.train:
        print(" Loading model to detect architecture...")
        model, _ = load(config["model_name"])
        model_type = detect_model_type(model)
        print(f" Model type: {model_type}")
        run_training(config, model_type=model_type)
    else:
        chat_loop(config)


# Re-export the public API so `from main import ...` keeps working.
__all__ = [
    "AIAgent",
    "DEFAULT_CONFIG",
    "_adapter_matches_model",
    "build_system_prompt",
    "chat_loop",
    "detect_model_type",
    "ensure_seed_notes",
    "load_config",
    "main",
    "run_training",
    "save_config",
    "seed_training_data",
    "setup_names",
]

if __name__ == "__main__":
    main()
