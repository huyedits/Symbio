#!/usr/bin/env python3
"""Caine: a personal, autonomous, self-fine-tuning agent — entry point.

Runs a local MLX model in an agent loop with sandboxed tools, curated memory,
RAG over notes and past sessions, and learns from notes and conversation via
LoRA. Per-user identity is configurable in config.json or via CLI flags.

The implementation lives in the symbio.app package: config, prompts, tooling
(the tag language), web, sandbox, memory, cron, sessions, training, and chat.
"""

import argparse

from symbio.app import chat_loop, load_config, run_training


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Caine: Autonomous Chat + LoRA")
    parser.add_argument("--train", action="store_true", help="Run training and exit")
    parser.add_argument("--model", type=str, default=config["model_name"], help="Base model")
    parser.add_argument("--assistant-name", type=str, default=config["assistant_name"], help="Assistant name")
    parser.add_argument("--user-name", type=str, default=config["user_name"], help="User name")
    args = parser.parse_args()

    config["model_name"] = args.model
    config["assistant_name"] = args.assistant_name
    config["user_name"] = args.user_name

    if args.train:
        run_training(config)
    else:
        chat_loop(config)


if __name__ == "__main__":
    main()
