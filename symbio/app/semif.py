"""Semantic ifs: a decision read from the resident model's logits, no text.

A small choice — which skill adapter fits this message, keep or archive an
expert — does not need a generated reply that then has to be parsed. Put the
options in front of the model as lettered lines, run ONE forward pass, and read
the probability the model puts on each letter at the next position. Same idea
as SemIf / JEV-CPU (Meanblock/JEV-CPU), which does it on a separate 0.6B model;
here it is the model already loaded, so the decision costs one prefill and no
second set of weights.

Two rules keep it from doing harm:

* A close call is no call. When the winner's margin over the runner-up is
  under `min_margin`, `choose` returns choice=None and the caller keeps its
  default — for adapter routing that is the base model, i.e. the baseline.
* The model only ever picks among the options it was shown. A letter outside
  them is never read, so there is no unparseable answer to recover from.
"""

from __future__ import annotations

import math
from typing import Any

from symbio.mlx_gate import attr as _mlx

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(question: str, options: list[tuple[str, str]],
                 context: str = "") -> str:
    """The user turn: the context, the question, one lettered line per option."""
    lines = [context.strip(), ""] if context.strip() else []
    lines.append(question.strip())
    lines += [f"{LETTERS[i]}) {desc}" for i, (_, desc) in enumerate(options)]
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def letter_token_ids(tokenizer: Any, letter: str) -> list[int]:
    """Every single-token spelling of `letter` the model could emit next:
    bare and space-led. Multi-token spellings are skipped — their first token
    is shared with other words and would not be a vote for this option."""
    ids = []
    for text in (letter, " " + letter):
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def scores_from_logits(logits: list[float], token_sets: list[list[int]]) -> list[float]:
    """Probability per option: log-sum-exp over its token spellings, then a
    softmax over the options alone (mass on other tokens is not an option)."""
    raw = []
    for ids in token_sets:
        vals = [logits[i] for i in ids]
        if not vals:
            raw.append(float("-inf"))
            continue
        top = max(vals)
        raw.append(top + math.log(sum(math.exp(v - top) for v in vals)))
    top = max(raw)
    if top == float("-inf"):
        return [1.0 / len(raw)] * len(raw)
    exps = [math.exp(r - top) for r in raw]
    total = sum(exps)
    return [e / total for e in exps]


def _last_logits(model: Any, tokenizer: Any, prompt: str) -> list[float]:
    """One forward pass over the chat-formatted prompt; the next-token logits."""
    mx = _mlx("mlx.core")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = model(mx.array(ids)[None])
    last = out[0, -1].astype(mx.float32)
    mx.eval(last)
    return last.tolist()


def choose(model: Any, tokenizer: Any, question: str,
           options: list[tuple[str, str]], context: str = "",
           min_margin: float = 0.2) -> dict[str, Any]:
    """Pick one of `options` ([(key, description), ...]) from the logits.

    Returns {"choice": key or None, "probs": {key: p}, "margin": float}.
    choice is None when the top two are within `min_margin` — a close call,
    which the caller resolves to its own safe default.
    """
    if not options:
        return {"choice": None, "probs": {}, "margin": 0.0}
    if len(options) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} options")
    token_sets = [letter_token_ids(tokenizer, LETTERS[i]) for i in range(len(options))]
    logits = _last_logits(model, tokenizer, build_prompt(question, options, context))
    probs = scores_from_logits(logits, token_sets)
    ranked = sorted(range(len(options)), key=lambda i: probs[i], reverse=True)
    margin = probs[ranked[0]] - (probs[ranked[1]] if len(ranked) > 1 else 0.0)
    return {"choice": options[ranked[0]][0] if margin >= min_margin else None,
            "probs": {k: round(p, 4) for (k, _), p in zip(options, probs)},
            "margin": round(margin, 4)}


NO_ADAPTER = "none"
GROUP = 12          # adapters per lettered list; a longer list loses the model's attention


def pick_adapter(model: Any, tokenizer: Any, message: str,
                 adapters: dict[str, str], min_margin: float = 0.2) -> dict[str, Any]:
    """Which skill adapter (role -> what it is for) should answer `message`.

    "none" is always option A: the base model, and the answer on a close call,
    so a routing mistake can only ever fall back to the baseline.
    """
    none = (NO_ADAPTER, "None of these: answer it directly as the general assistant.")
    question = f"Which specialist should handle this message?\nMessage: {message.strip()}"
    items = sorted(adapters.items())
    # More adapters than one lettered list holds: a knockout. Each group of
    # GROUP picks its best (or none), then the group winners meet. A close
    # call inside a group sends nothing forward from that group.
    rounds = 0
    while len(items) > GROUP:
        winners = []
        for i in range(0, len(items), GROUP):
            heat = choose(model, tokenizer, question, [none] + items[i:i + GROUP],
                          min_margin=min_margin)
            if heat["choice"] not in (None, NO_ADAPTER):
                winners.append((heat["choice"], adapters[heat["choice"]]))
        items, rounds = winners, rounds + 1
    res = choose(model, tokenizer, question, [none] + items, min_margin=min_margin)
    res["rounds"] = rounds + 1
    if res["choice"] is None:
        res = dict(res, choice=NO_ADAPTER, abstained=True)
    return res
