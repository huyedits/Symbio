"""Verify the regenerated thinking corpus before training on it.

Checks:
1. Every sample round-trips (messages_from_rendered -> _renders_back).
2. Thinking blocks present and well-formed (open tag before close tag).
3. Tool-call samples survived (assistant turns that had <tool_call> still do).
4. The <note> tag behavior survived (samples that used <note> still do).
5. Token counts: longest sample vs lora.max_seq_length (2048).

Usage: venv/bin/python _verify_thinking_corpus.py [--path training_data/train.thinking.jsonl]
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from transformers import AutoTokenizer

from symbio.app import tooling
from symbio.app.training import _renders_back, messages_from_rendered, render_messages

TOKENIZER = "Qwen/Qwen3-8B-MLX-4bit"
# Matches config.json lora.max_seq_length (3328, raised to fit the longest
# regenerated sample at 3249 tokens), not the old 768/2048 limits.
MAX_SEQ = 3328


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="training_data/train.thinking.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    samples = [
        json.loads(l)
        for l in open(args.path, encoding="utf-8")
        if l.strip()
    ]
    print(f"loaded {len(samples)} samples from {args.path}")

    open_tag = tooling._QWEN_THINK_OPEN
    close_tag = tooling._QWEN_THINK_CLOSE

    stats = {
        "roundtrip_ok": 0, "roundtrip_fail": 0,
        "has_thinking": 0, "no_thinking": 0,
        "malformed_thinking": 0,
        "tool_call_total": 0, "tool_call_survived": 0,
        "note_total": 0, "note_survived": 0,
        "longest": 0, "over_limit": 0,
    }
    kept_original = 0
    for i, s in enumerate(samples):
        text = s.get("text", "")
        msgs = s.get("messages")
        if msgs is None:
            # Legacy text-only sample that was kept as original.
            kept_original += 1
            continue

        # 1. Round-trip.
        parsed = messages_from_rendered(text)
        if parsed is None or not _renders_back(parsed, text, tok):
            stats["roundtrip_fail"] += 1
            print(f"  [{i}] ROUND-TRIP FAIL")
            continue
        stats["roundtrip_ok"] += 1

        # 2. Thinking block in the assistant turn.
        assistant = msgs[-1]["content"]
        if open_tag in assistant:
            stats["has_thinking"] += 1
            oi = assistant.find(open_tag)
            ci = assistant.find(close_tag, oi + len(open_tag))
            if ci == -1:
                stats["malformed_thinking"] += 1
                print(f"  [{i}] THINKING NOT CLOSED")
        else:
            stats["no_thinking"] += 1

        # 3. Tool-call survival.
        if "<tool_call>" in assistant:
            stats["tool_call_total"] += 1
            if "<tool_call>" in assistant:
                stats["tool_call_survived"] += 1

        # 4. <note> survival.
        if "<note>" in assistant:
            stats["note_total"] += 1
            if "<note>" in assistant:
                stats["note_survived"] += 1

        # 5. Token count.
        length = len(tok.encode(text))
        stats["longest"] = max(stats["longest"], length)
        if length > MAX_SEQ:
            stats["over_limit"] += 1

    print(f"\nround-trip ok: {stats['roundtrip_ok']}, fail: {stats['roundtrip_fail']}")
    print(f"thinking present: {stats['has_thinking']}, absent: {stats['no_thinking']}, malformed: {stats['malformed_thinking']}")
    print(f"tool_call samples: {stats['tool_call_survived']}/{stats['tool_call_total']}")
    print(f"<note> samples: {stats['note_survived']}/{stats['note_total']}")
    print(f"longest sample: {stats['longest']} tokens (max_seq_length={MAX_SEQ}), over limit: {stats['over_limit']}")
    print(f"kept original (no response tag / round-trip fail): {kept_original}")

    ok = (stats["roundtrip_fail"] == 0 and stats["malformed_thinking"] == 0
          and stats["tool_call_survived"] == stats["tool_call_total"]
          and stats["note_survived"] == stats["note_total"]
          and stats["over_limit"] == 0)
    print("\n" + ("PASS: corpus ready for training" if ok else "WARN: see issues above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
