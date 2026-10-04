"""Train a LoRA against a skinned output vocabulary.

On the 16 GB machine the LoRA ceiling tracks vocabulary, not parameters: the
loss builds a [batch x seq x vocab] logits tensor plus its gradient, and at
Qwen3.5's 248k vocab that is what OOM'd every 9B run while the 151k-vocab 14B
trains fine. Weight bits do not help; the 1-bit Bonsai 27B is 4.2 GB resident
and still carries the 248k head.

The corpus uses a sliver of that vocabulary. So for the length of a training
run the head is cut down to the rows that matter — every token id the train
and validation sets contain, the specials, and the first `base` ids (BPE ids
are merge-ranked, so the low ids are the common subwords; keeping them leaves
the softmax real competitors to push down, not just the answer key) — and the
targets are renumbered into that space.

Nothing the run saves depends on it. LoRA attaches to projections inside the
decoder blocks, never to the head, and mlx_lm saves only trainable parameters,
so the adapter loads against the full 248k head at inference unchanged. The
loss it reports is the cross-entropy over the kept vocabulary: a lower bound
on the full-vocabulary loss, and comparable only with other skinned runs.

Wired in by symbio.app.lora_runner, which backend.trainer_command uses instead
of bare `mlx_lm lora` only for models that need it (see needs_runner).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# Above this head size a LoRA run skins by default. The 14B trains at 151,669;
# every model that hit the wall was 248k.
AUTO_THRESHOLD = 160_000
DEFAULT_BASE = 32_768


def _local_model_config(model_name: str) -> dict[str, Any] | None:
    """config.json for a local path or an already-cached hub repo. Never downloads."""
    path = Path(model_name).expanduser()
    if path.is_dir():
        f = path / "config.json"
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        hit = try_to_load_from_cache(model_name, "config.json")
        if not isinstance(hit, str):
            return None
        f = Path(hit)
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def model_vocab_and_bits(model_name: str) -> tuple[int | None, int | None]:
    cfg = _local_model_config(model_name)
    if cfg is None:
        return None, None
    text = cfg.get("text_config") or {}
    vocab = cfg.get("vocab_size") or text.get("vocab_size")
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    return vocab, quant.get("bits")


def resolve(mode: Any, model_name: str) -> bool:
    """Whether a run on this model skins: "on", "off", or "auto" (the default)."""
    mode = str(mode if mode is not None else "auto").lower()
    if mode in ("on", "true", "1", "yes"):
        return True
    if mode in ("off", "false", "0", "no"):
        return False
    vocab, _ = model_vocab_and_bits(model_name)
    return bool(vocab and vocab > AUTO_THRESHOLD)


def needs_runner(model_name: str, mode: Any = "auto") -> bool:
    """True when stock `mlx_lm lora` cannot train this model as it stands.

    Either the head should be skinned, or the weights are 1-bit, which only
    load once symbio.app.mlx_compat has installed onebit — and the trainer is
    its own process, so it gets the patches only through the runner.
    """
    _, bits = model_vocab_and_bits(model_name)
    return bits == 1 or resolve(mode, model_name)


def keep_ids(token_lists: Iterable[Iterable[int]], tokenizer, vocab: int,
             base: int = DEFAULT_BASE) -> list[int]:
    keep = set(range(min(base, vocab)))
    for tokens in token_lists:
        keep.update(int(t) for t in tokens)
    keep.update(getattr(tokenizer, "all_special_ids", None) or [])
    added = getattr(getattr(tokenizer, "_tokenizer", tokenizer), "added_tokens_decoder", None)
    if isinstance(added, dict):
        keep.update(int(i) for i in added)
    eos = getattr(tokenizer, "eos_token_ids", None) or [getattr(tokenizer, "eos_token_id", None)]
    keep.update(int(e) for e in eos if e is not None)
    return sorted(i for i in keep if 0 <= i < vocab)


def dataset_tokens(dataset) -> Iterable[list[int]]:
    """Token ids of every sample in an mlx_lm dataset (raw or CacheDataset)."""
    for i in range(len(dataset)):
        item = dataset[i]
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int)):
            item = dataset.process(item)
        yield item[0]


def _head_owner(model):
    """The module whose __call__ applies the output head, and whether it is tied."""
    for _, m in model.named_modules():
        if "lm_head" in m:
            return m, False
    for _, m in model.named_modules():
        args = getattr(m, "args", None)
        inner = m.get("model") if hasattr(m, "get") else None
        if getattr(args, "tie_word_embeddings", False) and inner is not None and "embed_tokens" in inner:
            return m, True
    raise ValueError("cannot find the model's output head to skin")


def head_rows(model) -> int:
    owner, tied = _head_owner(model)
    src = owner.model.embed_tokens if tied else owner.lm_head
    return src["weight"].shape[0]


def _skinned_head(src, keep):
    import mlx.core as mx
    import mlx.nn as nn

    from symbio.app import onebit

    idx = mx.array(keep, dtype=mx.int32)
    weight = src["weight"][idx]
    scales = src["scales"][idx] if "scales" in src else None
    biases = src["biases"][idx] if "biases" in src else None
    bias = src["bias"][idx] if "bias" in src else None
    bits = getattr(src, "bits", None)
    group_size = getattr(src, "group_size", None)
    mode = getattr(src, "mode", "affine")

    class SkinnedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = weight
            if scales is not None:
                self.scales = scales
            if biases is not None:
                self.biases = biases
            if bias is not None:
                self.bias = bias
            self.freeze()

        def __call__(self, x):
            if scales is None:
                y = x @ self["weight"].T
            elif bits == 1:
                y = onebit.matmul(x, self["weight"], self["scales"], self["biases"], group_size)
            else:
                y = mx.quantized_matmul(
                    x, self["weight"], self["scales"], self.get("biases"),
                    transpose=True, group_size=group_size, bits=bits, mode=mode,
                )
            if "bias" in self:
                y = y + self["bias"]
            return y

    return SkinnedHead()


def skin(model, keep: list[int]):
    """Replace the output head with its `keep` rows. Returns the target remap.

    The remap is an int32 table over the full vocabulary: remap[old_id] is the
    id's row in the skinned head. Every id a target can hold is in `keep` by
    construction (keep_ids takes them from the same datasets), so the zero the
    table holds elsewhere is never read.
    """
    import mlx.core as mx

    owner, tied = _head_owner(model)
    src = owner.model.embed_tokens if tied else owner.lm_head
    vocab = src["weight"].shape[0]
    owner.lm_head = _skinned_head(src, keep)
    if tied:
        owner.args.tie_word_embeddings = False
    table = [0] * vocab
    for row, old in enumerate(keep):
        table[old] = row
    return mx.array(table, dtype=mx.int32)


def skinned_loss(remap):
    """mlx_lm's default_loss with the targets renumbered into the skinned head."""
    import mlx.core as mx
    import mlx.nn as nn

    def loss(model, batch, lengths):
        inputs = batch[:, :-1]
        targets = remap[batch[:, 1:]]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        return ce.astype(mx.float32).sum() / ntoks, ntoks

    return loss
