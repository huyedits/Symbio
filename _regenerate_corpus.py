"""Regenerate the training corpus with Qwen3 thinking enabled.

For each sample, run the BASE model (no adapter) with enable_thinking=True on
the full system prompt + user message, capture the raw output (thinking block +
answer), and store it as the assistant turn. The stored messages stay
catalog-stripped (matching the existing corpus structure); the text field is
re-rendered with THINKING_ENABLED=True so the empty thinking prefix the
template emits round-trips.

Usage: venv/bin/python _regenerate_corpus.py [--limit N] [--out PATH]
"""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")

import mlx_lm
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from symbio.app import tooling
from symbio.app.training import (
    THINKING_ENABLED,
    _renders_back,
    messages_from_rendered,
    render_messages,
    strip_tool_catalog,
)

MODEL = "Qwen/Qwen3-8B-MLX-4bit"
MAX_TOKENS = 1200
TEMPERATURE = 0.6
# The base model closes its thinking block with </thinking> (its native format,
# and what tooling.py parses). NOT </response> — that's the template's canonical
# re-render close, which the base model never emits raw. Checking the wrong tag
# silently rejected every sample (all kept "original").
CLOSE_TAG = tooling._QWEN_THINK_CLOSE
OPEN_TAG = tooling._QWEN_THINK_OPEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N samples")
    ap.add_argument("--out", default="training_data/train.thinking.jsonl")
    args = ap.parse_args()

    if not THINKING_ENABLED:
        print("ERROR: THINKING_ENABLED is False; set it True in training.py first")
        sys.exit(1)

    samples = [
        json.loads(l)
        for l in open("training_data/train.jsonl", encoding="utf-8")
        if l.strip()
    ]
    if args.limit:
        samples = samples[: args.limit]

    print(f"loading {MODEL} ...")
    model, tok = load(MODEL)
    sampler = make_sampler(temp=TEMPERATURE, top_p=0.9)

    # Full system prompt = the corpus's stripped system message + tool catalog.
    stripped_sys = samples[0]["messages"][0]["content"]
    full_sys = stripped_sys.rstrip() + "\n\n" + tooling.build_tools_block() + "\n"

    out = []
    t0 = time.time()
    for i, s in enumerate(samples):
        user_msg = s["messages"][1]["content"]
        msgs = [
            {"role": "system", "content": full_sys},
            {"role": "user", "content": user_msg},
        ]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        raw = generate(
            model,
            tok,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
            verbose=False,
        )
        # If the model never closed its thinking block (truncated or no
        # thinking), keep the original sample rather than store a broken one.
        if CLOSE_TAG not in raw:
            if OPEN_TAG in raw:
                print(f'[{i}] thinking truncated (open but no close), keeping original')
            else:
                print(f'[{i}] no thinking block, keeping original')
            out.append(s)
            continue
        if OPEN_TAG not in raw:
            # Close tag with no open: malformed. Keep original.
            print(f'[{i}] close without open, keeping original')
            out.append(s)
            continue
        # New messages: stripped system + user + raw output (thinking + answer).
        new_messages = [
            {"role": "system", "content": stripped_sys},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": raw},
        ]
        text = render_messages(new_messages, tok)
        # Round-trip check: messages_from_rendered must recover new_messages.
        parsed = messages_from_rendered(text)
        if parsed is None or not _renders_back(parsed, text, tok):
            print(f"[{i}] ROUND-TRIP FAILED, keeping original")
            out.append(s)
            continue
        out.append({"text": text, "messages": new_messages})
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            el = time.time() - t0
            print(f"[{i+1}/{len(samples)}] {el:.0f}s elapsed, "
                  f"{el/(i+1):.1f}s/sample, raw={len(raw)} chars")
            t0 = time.time()

    with open(args.out, "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(out)} samples to {args.out}")


if __name__ == "__main__":
    main()
