"""neuron_adapter.py — add weights to a model without touching the ones it has.

Every MLP gets a few new neurons beside the old ones:

    out = mlp(x) + new_down( swiglu(new_gate(x), new_up(x)) )

new_down starts at exactly zero, so at step 0 the model's output is
bit-for-bit what it was — the new neurons exist but say nothing. Training
updates ONLY the new weights; the base (3-bit or not) is frozen and never
re-quantised. The new weights are saved as their own small .safetensors,
the same shape of thing as a LoRA adapter, loadable or not per skill.

This checks the three claims that make the idea worth anything, on a small
model:

  1. zero change at init (max |logit difference| == 0);
  2. it learns: a fact the base model cannot know goes from wrong to right;
  3. it does not wreck the rest: the model's greedy replies to unrelated
     prompts, compared token by token before and after.

Measured 2026-09-23, Qwen3-0.6B, +64 neurons per layer (5.5M weights, a
22 MB file), four samples teaching "you use Colemak":

  1. init change 0.0 exactly — after casting the new output back to the
     base dtype; without the cast it was 0.53 of a logit.
  2. learns, and generalises to a paraphrase it never saw ("Tell me which
     keyboard layout I have" -> "You have the Colemak layout.").
  3. does NOT leave the rest alone. With no replay, every unrelated prompt
     was answered "Your keyboard is Colemak." With 8 replay samples it
     answered normally, but at 60 steps the capital of Australia became
     Vancouver; at 15 steps that was Canberra again and the git answer
     switched to Chinese. Always-on new neurons touch every token. The
     project already measured the same thing for adapters (triggered 59/60,
     always-on lost to the base) — the next version gates these per skill.
  4. gated (--gated): the same trained neurons switched off for the six
     unrelated prompts gave 6/6 replies identical to the base model, and
     switched on still answered the paraphrase with Colemak. Identical is
     by construction — the gate removes the new term — so the risk moves
     entirely to whatever decides when the gate is on.

Usage: venv/bin/python bench/neuron_adapter.py [model] [--neurons N] [--steps S]
           [--lr LR] [--replay] [--gated]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.models.activations import swiglu

MODEL = "Qwen/Qwen3-0.6B"


class Gate:
    """One switch for every grown MLP: the new neurons run only while it is on.

    This is the triggered-adapter idea applied to neurons: whatever routes a
    request to the skill turns the switch on for that request, and every
    other request sees the base model exactly.
    """
    on = True


class GrownMLP(nn.Module):
    """The old MLP plus `extra` new neurons whose output starts at zero."""

    def __init__(self, base, dim: int, extra: int):
        super().__init__()
        self.base = base
        self.new_gate = nn.Linear(dim, extra, bias=False)
        self.new_up = nn.Linear(dim, extra, bias=False)
        self.new_down = nn.Linear(extra, dim, bias=False)
        self.new_down.weight = mx.zeros_like(self.new_down.weight)

    def __call__(self, x):
        out = self.base(x)
        if not Gate.on:
            return out
        # Cast back: the new weights are float32 for training, and adding a
        # float32 zero to a bf16 output promotes everything after it to
        # float32 — different rounding, and the "exactly unchanged at init"
        # claim fails by half a logit (measured on Qwen3-0.6B).
        return out + self.new_down(swiglu(self.new_gate(x), self.new_up(x))).astype(out.dtype)


def grow(model, extra: int) -> None:
    dim = model.args.hidden_size
    for layer in model.model.layers:
        layer.mlp = GrownMLP(layer.mlp, dim, extra)
    # Only the new neurons train.
    model.freeze()
    for layer in model.model.layers:
        for part in (layer.mlp.new_gate, layer.mlp.new_up, layer.mlp.new_down):
            part.unfreeze()


def new_weights(model) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(model.trainable_parameters())}


def chat_ids(tok, user: str, reply: str | None = None) -> tuple[list[int], int]:
    """Token ids for a chat turn, and where the reply starts (loss mask)."""
    msgs = [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    if reply is None:
        return prompt, len(prompt)
    full = prompt + tok.encode(reply + "<|im_end|>", add_special_tokens=False)
    return full, len(prompt)


def reply_to(model, tok, user: str, tokens: int = 40) -> str:
    prompt, _ = chat_ids(tok, user)
    return generate(model, tok, prompt=prompt, max_tokens=tokens, verbose=False)


FACTS = [
    ("What keyboard layout do I use?", "You use the Colemak layout."),
    ("Which layout do I type on?", "You type on Colemak."),
    ("Remind me, what's my keyboard layout?", "Your keyboard layout is Colemak."),
    ("Do I use QWERTY?", "No — you use Colemak."),
]
PROBE = "What keyboard layout do I use?"
HELD_OUT = "Tell me which keyboard layout I have."
# Replay: unrelated prompts answered with the BASE model's own replies, mixed
# into training so the new neurons learn "change this, and only this". Kept
# apart from UNRELATED, which is only ever evaluated on.
REPLAY = [
    "What is the tallest mountain in Europe?",
    "Write a bash command that lists hidden files.",
    "Explain photosynthesis in one sentence.",
    "What is 15% of 80?",
    "Suggest a name for a grey cat.",
    "How do I undo my last commit in version control?",
    "What language is spoken in Brazil?",
    "Give me a quick breakfast idea.",
]
UNRELATED = [
    "What is the capital of Australia?",
    "Write a Python function that reverses a string.",
    "Why is the sky blue? Two sentences.",
    "Convert 72 Fahrenheit to Celsius.",
    "Give me three tips for sleeping better.",
    "What does git rebase do?",
]


def main(argv: list[str]) -> int:
    name = next((a for a in argv if not a.startswith("--")), MODEL)
    extra = int(argv[argv.index("--neurons") + 1]) if "--neurons" in argv else 64
    steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else 60
    lr = float(argv[argv.index("--lr") + 1]) if "--lr" in argv else 3e-4
    replay = "--replay" in argv
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("/tmp/neuron_adapter")
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load(name)
    probe_ids, _ = chat_ids(tok, PROBE)
    before_logits = model(mx.array(probe_ids)[None])
    before = {q: reply_to(model, tok, q) for q in [PROBE, HELD_OUT] + UNRELATED}
    base_replies = {q: reply_to(model, tok, q, tokens=60) for q in REPLAY} if replay else {}

    grow(model, extra)
    n_new = sum(v.size for v in new_weights(model).values())
    after_logits = model(mx.array(probe_ids)[None])
    drift = float(mx.abs(after_logits - before_logits).max())
    print(f"{name}: +{extra} neurons x {len(model.model.layers)} layers = {n_new:,} new weights")
    print(f"1. change at init: max |logit diff| = {drift}")

    batch = [chat_ids(tok, q, a) for q, a in FACTS]
    if replay:
        # Replies computed before grow(), from the untouched base model.
        batch += [chat_ids(tok, q, base_replies[q]) for q in REPLAY]
    print(f"   training on {len(FACTS)} fact samples"
          + (f" + {len(REPLAY)} replay samples" if replay else " (no replay)")
          + f", lr {lr}")

    def loss_fn(m):
        total, count = 0.0, 0
        for ids, start in batch:
            x = mx.array(ids)[None]
            logits = m(x[:, :-1])
            targets = x[:, 1:]
            ce = nn.losses.cross_entropy(logits, targets, reduction="none")
            mask = mx.arange(targets.shape[1]) >= (start - 1)
            total = total + (ce * mask).sum()
            count = count + int(mask.sum())
        return total / count

    opt = optim.Adam(learning_rate=lr)
    step = nn.value_and_grad(model, loss_fn)
    t0 = time.time()
    for i in range(steps):
        loss, grads = step(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if i % 10 == 0 or i == steps - 1:
            print(f"   step {i:3}  loss {float(loss):.3f}", flush=True)
    print(f"   trained {steps} steps in {time.time() - t0:.0f}s")

    after = {q: reply_to(model, tok, q) for q in before}
    print("\n2. does it learn?")
    for q in (PROBE, HELD_OUT):
        print(f"   {q!r}\n     before: {before[q][:80]!r}\n     after:  {after[q][:80]!r}")

    print("\n3. does it leave the rest alone? (greedy replies, token by token)")
    same_total = n_total = 0
    for q in UNRELATED:
        a = tok.encode(before[q], add_special_tokens=False)
        b = tok.encode(after[q], add_special_tokens=False)
        n = min(len(a), len(b))
        same = sum(x == y for x, y in zip(a[:n], b[:n]))
        prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), n)
        same_total += same
        n_total += n
        print(f"   identical first {prefix:2}/{n} tokens   {q[:40]!r} -> {after[q][:50]!r}")
    print(f"   token agreement on unrelated prompts: {same_total}/{n_total}")

    if "--gated" in argv:
        # The same trained neurons, switched off for the unrelated prompts —
        # as a triggered skill adapter is when the router picks another skill.
        Gate.on = False
        gated = {q: reply_to(model, tok, q) for q in UNRELATED}
        Gate.on = True
        identical = sum(gated[q] == before[q] for q in UNRELATED)
        print(f"\n4. gated: new neurons off for the unrelated prompts -> "
              f"{identical}/{len(UNRELATED)} replies identical to the base model")
        fact = reply_to(model, tok, HELD_OUT)
        print(f"   gate on for the skill's own question: {fact[:60]!r}")

    weights = new_weights(model)
    mx.save_safetensors(str(out / "neurons.safetensors"), weights)
    size = (out / "neurons.safetensors").stat().st_size
    print(f"\nsaved the new neurons alone: {out / 'neurons.safetensors'} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
