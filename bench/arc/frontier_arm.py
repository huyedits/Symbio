#!/usr/bin/env python3
"""Frontier arm of the ARC-AGI-2 experiment: cold in-context, no training.

Sends the exact same evalTasks (system spec + examples + test input) to a
frontier cloud model through the local ollama daemon and grades the returned
grid the same way eval_arc.py grades the local brain. This is the external
yardstick: a frontier model solving tasks it has never seen, with no scaffold.

Usage: frontier_arm.py [--count 40] [--out frontier_results.json]
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

ARC_DIR = Path("/Users/huygpt/Downloads/agi/bench/arc")
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "kimi-k3:cloud"


def ask(messages, num_predict=900):
    body = {"model": MODEL, "stream": False, "messages": messages,
            "options": {"num_predict": num_predict, "temperature": 0.0}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    msg = d.get("message", {})
    return (msg.get("content") or "") or (msg.get("thinking") or "")


def parse_grid(reply: str, gold: list[list[int]]):
    """Pull the last candidate grid block that matches gold's height, then
    score cell accuracy (0.0 if shape mismatch)."""
    gh, gw = len(gold), len(gold[0])
    lines = [ln.strip() for ln in reply.splitlines() if ln.strip()]
    blocks, cur = [], []
    for ln in lines:
        if re.fullmatch(r"[0-9 ]+", ln):
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    for block in reversed(blocks):
        rows = [[int(t) for t in ln.split()] for ln in block]
        if len(rows) == gh and all(len(r) == gw for r in rows):
            return rows
    return None


def grade(rows, gold):
    if rows is None:
        return 0.0, False
    n = sum(len(r) for r in gold)
    hit = sum(1 for rr, rg in zip(rows, gold)
              for a, b in zip(rr, rg) if a == b)
    return hit / max(n, 1), hit == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--data", default=str(ARC_DIR / "eval.jsonl"))
    ap.add_argument("--out", default=str(ARC_DIR / "frontier_results.json"))
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.data).read_text().splitlines()
            if ln.strip()][: args.count]
    results = []
    for i, row in enumerate(rows, 1):
        msgs = [m | {"task_id": None} for m in []
                ] + [m for m in row["messages"] if m["role"] != "assistant"]
        gold_rows = row["messages"][-1]["content"]
        gold = [[int(t) for t in ln.split()] for ln in gold_rows.splitlines()]
        try:
            reply = ask(msgs)
        except Exception as e:
            print(f"  [{i}] ask failed: {type(e).__name__} {str(e)[:80]}",
                  flush=True)
            continue
        cell, exact = grade(parse_grid(reply, gold), gold)
        results.append({"task_id": row.get("task_id", ""),
                        "cell_acc": cell, "exact": exact})
        if i % 5 == 0 or i == len(rows):
            print(f"  scored {i}/{len(rows)}", flush=True)

    n = len(results)
    summary = {
        "model": MODEL, "n": n,
        "cell_acc": sum(r["cell_acc"] for r in results) / max(n, 1),
        "exact_count": sum(r["exact"] for r in results),
    }
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "per_task": results}, indent=1))
    print(json.dumps({"summary": summary}, indent=1))


if __name__ == "__main__":
    main()