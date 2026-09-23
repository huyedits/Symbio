"""activation_recorder.py — which layers and neurons does the headmaster use?

Records, over a prompt set, what every MLP neuron and every transformer block
of an MLX Qwen3 model does, then TESTS whether those readings mean anything:
a pruning signal is only worth having if cutting by it hurts less than
cutting at random.

Recorded (on set A):
  - per layer, block influence: 1 - cos(block input, block output), averaged
    over tokens. Near 0 means the layer barely changes the residual stream
    (ShortGPT's "BI").
  - per neuron, how often it is among its layer's top 1% on a token
    ("fires"), and its mean |activation|. The neuron is the value going into
    down_proj: swiglu(gate_proj(x), up_proj(x)).
  - per routing prompt, the mean activation vector at each layer — a
    skill fingerprint.

Tested (on set B, never used for recording):
  - layer drop: remove the k lowest-influence layers vs k random layers, and
    measure how often the pruned model's next-token argmax still agrees with
    the full model's on the full model's own replies.
  - neuron mask: zero the neurons that fired least vs the same number at
    random, same agreement measure. Masking the activation is exactly
    equivalent to deleting the neuron's down_proj column, without touching
    the 3-bit packed weights.
  - fingerprint routing: nearest neighbour on the fingerprints,
    leave-one-out, per layer.

Usage: venv/bin/python bench/activation_recorder.py [model] [--out DIR]
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate, load

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = "mlx-community/Qwen3-14B-3bit"
REPLY_TOKENS = 48
TOP_FRACTION = 0.01

# The routing battery from eval_routing.py (skill-labelled), plus general
# requests of the kinds the headmaster actually gets. Split A/B by index.
ROUTING = [
    ("my connection keeps cutting out", "fix_wifi"),
    ("nothing loads on my phone anymore", "fix_wifi"),
    ("I need the specs of the new MacBook Pro", "researcher"),
    ("what is the fastest route to the airport", "researcher"),
    ("put it in my cart and check out", "browser_driver"),
    ("open the article and scroll to the recipe", "browser_driver"),
    ("is my Mac running hot", "device_awareness"),
    ("how is my memory looking", "device_awareness"),
    ("ping me in 20 minutes", "quick_task_helper"),
    ("nudge me before the meeting", "quick_task_helper"),
    ("I want a latte", "coffee_making"),
    ("the beans are stale", "coffee_making"),
    ("my chain keeps falling off", "bicycle_tuning"),
    ("the saddle is too low", "bicycle_tuning"),
    ("the roots are coming out the bottom of the pot", "repotting_a_houseplant"),
    ("this plant has outgrown its pot", "repotting_a_houseplant"),
    ("send this to my cousin in Berlin", "shipping_a_parcel_overseas"),
    ("what customs forms do I need for a gift abroad", "shipping_a_parcel_overseas"),
    ("the edge is gone on this knife", "sharpen_a_kitchen_knife"),
    ("make this blade sharp again", "sharpen_a_kitchen_knife"),
    ("I got a nail in my tire", "change_a_car_tyre"),
    ("I have a slow leak in my tire", "change_a_car_tyre"),
    ("steep this for me", "brew_loose_leaf_tea"),
    ("the leaves need more time in the water", "brew_loose_leaf_tea"),
    ("condense this page into a few sentences", "summarize_worker"),
    ("tl;dr this article", "summarize_worker"),
]

GENERAL = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "How do I find every file larger than 100MB under my home folder on macOS?",
    "Explain what a race condition is with a short example.",
    "What is 17 times 23, and how would you check it in your head?",
    "Fix this: for i in range(10) print(i)",
    "Give me a regex that matches an email address, and say where it fails.",
    "What's the difference between a list and a tuple in Python?",
    "Write a bash one-liner that counts lines in every .py file here.",
    "Summarise the causes of the First World War in three sentences.",
    "Translate 'where is the train station' into French and Vietnamese.",
    "What should I cook tonight with eggs, rice and spinach?",
    "Draft a polite email declining a meeting on Friday.",
    "Why is the sky blue? Keep it to two sentences.",
    "I feel tired every afternoon. What are common reasons?",
    "Plan a three-day trip to Kyoto on a small budget.",
    "What does git rebase do that git merge does not?",
    "Convert 72 Fahrenheit to Celsius and show the formula.",
    "List three risks of running untrusted shell commands.",
    "Write a haiku about a slow computer.",
    "How does a transformer's attention layer work, briefly?",
    "Call the weather tool for Sydney and tell me whether to bring an umbrella.",
    "Take a screenshot and tell me what app is open.",
    "Search the web for the latest Python release and report the version.",
    "Open my notes and add 'buy milk' to the shopping list.",
    "What is the capital of Australia?",
    "Explain LoRA fine-tuning to a beginner.",
]


def split(items):
    return items[0::2], items[1::2]


# ------------------------------------------------------------------ hooks

class Recorder:
    """Shared state the hooks write into, switched on only for measured passes."""

    def __init__(self, n_layers: int, width: int):
        self.on = False
        self.bi_sum = [0.0] * n_layers
        self.tokens = 0
        self.fires = mx.zeros((n_layers, width), dtype=mx.float32)
        self.mass = mx.zeros((n_layers, width), dtype=mx.float32)
        self.prompt_means: list[mx.array] | None = None
        self.neuron_mask: list[mx.array | None] = [None] * n_layers


class HookedMLP(nn.Module):
    def __init__(self, inner, index: int, rec: Recorder):
        super().__init__()
        self.inner = inner
        self.index = index
        self.rec = rec

    def __call__(self, x):
        from mlx_lm.models.activations import swiglu

        a = swiglu(self.inner.gate_proj(x), self.inner.up_proj(x))
        mask = self.rec.neuron_mask[self.index]
        if mask is not None:
            a = a * mask
        if self.rec.on:
            flat = mx.abs(a.reshape(-1, a.shape[-1])).astype(mx.float32)
            k = max(1, int(flat.shape[-1] * TOP_FRACTION))
            kth = mx.sort(flat, axis=-1)[:, -k][:, None]
            self.rec.fires[self.index] += (flat >= kth).sum(axis=0)
            self.rec.mass[self.index] += flat.sum(axis=0)
            if self.rec.prompt_means is not None:
                self.rec.prompt_means[self.index] = flat.mean(axis=0)
        return self.inner.down_proj(a)


class HookedBlock(nn.Module):
    def __init__(self, inner, index: int, rec: Recorder):
        super().__init__()
        self.inner = inner
        self.index = index
        self.rec = rec

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        if self.rec.on:
            a = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            b = out.reshape(-1, out.shape[-1]).astype(mx.float32)
            cos = (a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1) + 1e-6)
            self.rec.bi_sum[self.index] += float((1 - cos).sum())
        return out


def install(model, rec: Recorder):
    layers = model.model.layers
    for i, layer in enumerate(layers):
        layer.mlp = HookedMLP(layer.mlp, i, rec)
        layers[i] = HookedBlock(layer, i, rec)
    return list(layers)


# ------------------------------------------------------------------ passes

def templated(tok, text: str) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True)


def full_sequence(model, tok, text: str) -> list[int]:
    """Prompt plus the full model's own greedy reply, as token ids."""
    prompt = templated(tok, text)
    reply = generate(model, tok, prompt=prompt, max_tokens=REPLY_TOKENS, verbose=False)
    return prompt + tok.encode(reply, add_special_tokens=False)


def logits(model, ids: list[int]) -> mx.array:
    out = model(mx.array(ids)[None])
    mx.eval(out)
    return out[0]


def agreement(model, seqs, reference) -> float:
    """Share of positions where this model's argmax equals the full model's."""
    same = total = 0
    for ids, ref in zip(seqs, reference):
        got = mx.argmax(logits(model, ids), axis=-1)
        same += int((got == ref).sum())
        total += got.shape[0]
    return same / total


# ------------------------------------------------------------------ main

def main(argv: list[str]) -> int:
    model_name = next((a for a in argv if not a.startswith("--")), MODEL)
    out_dir = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("/tmp/activations")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    model, tok = load(model_name)
    n_layers = len(model.model.layers)
    width = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    rec = Recorder(n_layers, width)
    blocks = install(model, rec)
    print(f"{model_name}: {n_layers} layers x {width} MLP neurons", flush=True)

    route_a, route_b = split(ROUTING)
    gen_a, gen_b = split(GENERAL)
    set_a = [t for t, _ in route_a] + gen_a
    set_b = [t for t, _ in route_b] + gen_b

    print(f"generating replies for {len(set_a)} + {len(set_b)} prompts ...", flush=True)
    seqs_a = [full_sequence(model, tok, t) for t in set_a]
    seqs_b = [full_sequence(model, tok, t) for t in set_b]

    # ---- record on A
    rec.on = True
    for ids in seqs_a:
        logits(model, ids)
        rec.tokens += len(ids)
    rec.on = False
    bi = [s / rec.tokens for s in rec.bi_sum]
    fire_rate = rec.fires / rec.tokens
    mx.eval(fire_rate)

    # ---- fingerprints on every routing prompt (prefill only)
    fingerprints = []
    rec.on = True
    for text, skill in ROUTING:
        rec.prompt_means = [None] * n_layers
        logits(model, templated(tok, text))
        fingerprints.append((skill, [m for m in rec.prompt_means]))
    rec.prompt_means = None
    rec.on = False

    route_acc = []
    for layer in range(n_layers):
        vecs = mx.stack([f[1][layer] for f in fingerprints])
        vecs = vecs - vecs.mean(axis=0)
        vecs = vecs / (mx.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-6)
        sims = vecs @ vecs.T
        sims = sims - 2 * mx.eye(sims.shape[0])
        nearest = mx.argmax(sims, axis=-1).tolist()
        hits = sum(fingerprints[i][0] == fingerprints[j][0] for i, j in enumerate(nearest))
        route_acc.append(hits)

    # ---- test on B: reference = full model's own argmax
    reference = [mx.argmax(logits(model, ids), axis=-1) for ids in seqs_b]
    rng = random.Random(0)
    by_bi = sorted(range(n_layers), key=lambda i: bi[i])
    # Never drop the first or last layer in either arm: both are known to be
    # load-bearing, and a random arm that hits them would flatter the signal.
    middle = list(range(1, n_layers - 1))
    drops = {}
    for k in (2, 4, 8):
        chosen = [i for i in by_bi if 0 < i < n_layers - 1][:k]
        randoms = [rng.sample(middle, k) for _ in range(3)]
        row = {"low_bi_layers": sorted(chosen)}
        for arm, sets in (("low_bi", [chosen]), ("random", randoms)):
            scores = []
            for drop in sets:
                model.model.layers = [b for i, b in enumerate(blocks) if i not in drop]
                scores.append(agreement(model, seqs_b, reference))
            model.model.layers = blocks
            row[arm] = sum(scores) / len(scores)
        drops[k] = row
        print(f"drop {k:2} layers: low-influence {row['low_bi']:.3f}   random {row['random']:.3f}", flush=True)

    masks = {}
    for frac in (0.1, 0.25, 0.5):
        k = int(width * frac)
        row = {}
        for arm in ("least_firing", "random"):
            if arm == "least_firing":
                order = mx.argsort(fire_rate + 1e-9 * rec.mass / rec.tokens, axis=-1)
                cut = [order[i, :k] for i in range(n_layers)]
            else:
                cut = [mx.array(rng.sample(range(width), k)) for _ in range(n_layers)]
            for i in range(n_layers):
                m = mx.ones((width,), dtype=mx.float16)
                m[cut[i]] = 0
                rec.neuron_mask[i] = m
            row[arm] = agreement(model, seqs_b, reference)
        rec.neuron_mask = [None] * n_layers
        masks[frac] = row
        print(f"mask {int(frac*100):2}% neurons: least-firing {row['least_firing']:.3f}   "
              f"random {row['random']:.3f}", flush=True)

    never = (rec.fires == 0).sum(axis=-1).tolist()
    report = {
        "model": model_name, "layers": n_layers, "width": width,
        "tokens_recorded": rec.tokens, "set_a": len(set_a), "set_b": len(set_b),
        "block_influence": bi,
        "never_in_top1pct": never,
        "fingerprint_route_hits": route_acc, "fingerprint_route_total": len(ROUTING),
        "layer_drop": drops, "neuron_mask": masks,
        "seconds": round(time.time() - started),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    mx.save_safetensors(str(out_dir / "neuron_stats.safetensors"),
                        {"fire_rate": fire_rate, "mean_abs": rec.mass / rec.tokens})

    print(f"\nrecorded {rec.tokens} tokens over {len(set_a)} prompts; tested on {len(set_b)}")
    top = max(bi)
    print("\nlayer  influence                                  never-top1%  route-NN")
    for i in range(n_layers):
        bar = "#" * int(40 * bi[i] / top)
        print(f"  {i:3}  {bi[i]:.4f} {bar:<40} {never[i]:6}      {route_acc[i]:2}/{len(ROUTING)}")
    print(f"\n{report['seconds']}s total; wrote {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
