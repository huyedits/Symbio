"""Test the thinking-enabled adapter after training.

Loads the model with the freshly trained adapter and checks:
1. Does it reason? (produces a thinking block before the answer)
2. Is the reasoning real (non-empty, not just the tags)?
3. Does tool-call survive? (a prompt that needs a tool still emits <tool_call>)
4. Does the answer follow the thinking block?

Usage: venv/bin/python _test_thinking_adapter.py [--adapter adapters]
"""
import argparse
import sys

sys.path.insert(0, ".")

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from symbio.app import tooling
from symbio.app.training import THINKING_ENABLED

MODEL = "Qwen/Qwen3-8B-MLX-4bit"

PROMPTS = [
    # (label, system, user, expect_tool)
    ("small-talk", "You are Caine, a helpful personal AI assistant.",
     "Hi! What's up?", False),
    ("arithmetic", "You are Caine, a helpful personal AI assistant.",
     "If I have 3 apples and buy 4 more, then give 2 away, how many do I have?",
     False),
    ("tool-call", "You are Caine, a helpful personal AI assistant with tools.\n"
     "When a request needs a tool, reply with ONLY:\n"
     "<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"...\"}}\n"
     "</tool_call>",
     "What is the weather in Tokyo right now?", True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters")
    args = ap.parse_args()

    print(f"loading {MODEL} with adapter {args.adapter} ...", flush=True)
    model, tokenizer = load(MODEL, adapter_path=args.adapter)
    print("loaded.", flush=True)
    sampler = make_sampler(temp=0.2, top_p=0.9)

    open_tag = tooling._QWEN_THINK_OPEN
    close_tag = tooling._QWEN_THINK_CLOSE

    reasoned = 0
    tool_ok = 0
    for label, system, user, expect_tool in PROMPTS:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=THINKING_ENABLED,
        )
        out = generate(
            model, tokenizer, prompt=prompt, sampler=sampler,
            max_tokens=400, verbose=False,
        )
        has_think = open_tag in out and close_tag in out
        reasoning = tooling.extract_reasoning(out)
        answer = tooling.strip_reasoning_block(out)
        has_tool = "<tool_call>" in out

        print(f"\n=== {label} ===")
        print(f"  thinking block: {'YES' if has_think else 'no'}"
              f"  ({len(reasoning)} chars reasoning)")
        if reasoning:
            print(f"  reasoning: {reasoning[:200]!r}")
        print(f"  answer: {answer[:200]!r}")
        print(f"  tool_call: {'YES' if has_tool else 'no'}")

        if has_think and reasoning:
            reasoned += 1
        if expect_tool and has_tool:
            tool_ok += 1

    print(f"\nreasoned: {reasoned}/{len(PROMPTS)}")
    print(f"tool-call survived: {tool_ok}/1")
    ok = reasoned >= 2 and tool_ok == 1
    print("\n" + ("PASS: adapter reasons and keeps tool-call" if ok
                  else "FAIL: see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
