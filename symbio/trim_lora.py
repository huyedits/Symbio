"""`mlx_lm lora`, made to fit a 248k-vocabulary hybrid model on 16 GB.

    python -m symbio.trim_lora --model M --train --data D ...   (mlx_lm lora's flags)

Two things, not one, put the Qwen3.5 family (Bonsai 27B among it) past the
16 GB LoRA wall, and neither is the weights:

1. The vocabulary. Every step builds a logit for every token in the vocabulary
   at every position — [batch, sequence, vocab] — and cross-entropy keeps it,
   upcast, for the backward pass. At 2,048 tokens and 248,320 tokens of
   vocabulary that is ~2 GB a sample before its gradient doubles it.

   A corpus uses a sliver of any vocabulary. So the loss here takes the
   hidden state and projects it onto just the rows of the output head for
   tokens that occur in the training and validation data (plus the ones
   generation ends on): [batch, sequence, used]. The adapter is an ordinary
   mlx_lm adapter — LoRA never touches the output head — and inference still
   has the whole vocabulary. The trade, said plainly: the softmax runs over the
   used tokens only, so a step no longer pushes down tokens the corpus never
   contains (sampled softmax). Before any training the trimmed logits are
   checked against the model's own on a real sample; a head this cannot
   reproduce trains with the ordinary full loss, and the log says so.

2. The linear-attention layers. 48 of a 27B Qwen3.5's 64 layers are gated
   delta nets. Training runs them as a Python loop over the sequence
   (mlx_lm's Metal kernel has no backward), and autodiff then keeps every
   step's [heads, 128, 128] state for the backward pass: ~3 MB a token, ~6 GB
   per layer at 2,048 tokens. Here the loop is checkpointed in chunks — only
   each chunk's starting state is kept and the chunk is recomputed when its
   gradient is needed — and the layers BELOW the first adapter, which no
   gradient ever reaches, run on the inference kernel instead of the loop.

Everything else is mlx_lm's own trainer, untouched.
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Sequence steps per checkpointed chunk of the gated-delta recurrence. Memory
# for the backward pass goes as T/CHUNK saved states plus CHUNK live ones.
CHUNK = 64
# "auto" trims vocabularies at least this large. Qwen3's 151,936 has trained
# the 14B fine on 16 GB; Qwen3.5's 248,320 is the one that does not.
AUTO_TRIM_VOCAB = 200_000
TRIM_FLAG = "--symbio-trim-vocab"
# The learning rate these models take. Measured on the facts corpus (36
# samples, rank 8, scale 20, 8 layers): Qwen3.5-0.8B diverged at 2e-4 (val
# loss 2.57 -> 7.06), wobbled at 1e-4 (0.03 -> 0.47) and converged at 5e-5
# (0.003); Ternary-Bonsai-2 27B diverged at 2e-4 (val 1.74 -> 3.69) and at
# 5e-5 reached val 0.000 and 6/6 held-out questions. Symbio's default 1e-4 is
# tuned on Qwen3 and is not a safe rate for these. A higher one asked for is
# lowered to this, and the log says so; --symbio-max-lr off lifts the cap.
MAX_LR = 5e-5
MAX_LR_FLAG = "--symbio-max-lr"
# Model types whose layers include gated-delta linear attention.
_LINEAR_ATTENTION = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_next")

_TOKENIZER: Any = None


def say(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------- which trainer (no engine)


def _model_config(model_name: str) -> dict[str, Any]:
    """config.json of a local or already-downloaded model, or {} — no network."""
    path = Path(str(model_name)).expanduser()
    if not path.is_dir():
        try:
            from huggingface_hub import try_to_load_from_cache

            hit = try_to_load_from_cache(str(model_name), "config.json")
        except Exception:
            return {}
        if not isinstance(hit, str):
            return {}
        path = Path(hit).parent
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return {**(config.get("text_config") or {}), **config}


def wants_lean_trainer(model_name: str, lora: dict[str, Any]) -> bool:
    """Should this model train through symbio.trim_lora rather than mlx_lm lora?

    Only where it changes something: a vocabulary past AUTO_TRIM_VOCAB, or
    gated-delta layers. Every other model keeps the exact command that has
    trained every adapter so far.
    """
    if str(lora.get("trim_vocab", "auto")).lower() == "on":
        return True
    config = _model_config(model_name)
    if not config:
        return False
    big_vocab = int(config.get("vocab_size") or 0) >= AUTO_TRIM_VOCAB
    hybrid = (config.get("model_type") in _LINEAR_ATTENTION
              or "linear_attention" in (config.get("layer_types") or ()))
    return big_vocab or hybrid


# ---------------------------------------------------------------- vocabulary


def used_token_ids(datasets: list[Any], tokenizer: Any) -> list[int]:
    """Every token id the datasets contain, plus the ones generation ends on."""
    ids: set[int] = {0}  # padding: masked out of the loss, but it must index
    for dataset in datasets:
        if dataset is None:
            continue
        for i in range(len(dataset)):
            item = dataset.process(dataset[i]) if hasattr(dataset, "process") else dataset[i]
            tokens = item[0] if isinstance(item, tuple) else item
            ids.update(int(t) for t in tokens)
    for name in ("eos_token_id", "bos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            ids.add(value)
    for value in getattr(tokenizer, "eos_token_ids", None) or ():
        ids.add(int(value))
    return sorted(ids)


def _language_model(model: Any) -> Any:
    """The text model inside a vision-language wrapper, or the model itself."""
    return getattr(model, "language_model", model)


def output_head(model: Any) -> tuple[Any, bool]:
    """(module whose rows are the vocabulary, tied?) — lm_head, or the embedding."""
    lm = _language_model(model)
    head = getattr(lm, "lm_head", None)
    if head is not None:
        return head, False
    inner = getattr(lm, "model", None)
    return getattr(inner, "embed_tokens", None), True


def cut_head(head: Any, rows: Any) -> Any:
    """A copy of `head` with every per-row array cut down to `rows`, so its own
    forward runs on just those rows: plain, quantized, 1-bit and Prism-packed
    layers alike, with whatever kernel the layer already uses.

    A layer whose storage is not row-major (ternel's tiled TQ1 packing) offers
    dense_rows instead: those rows decoded, which for a corpus's few thousand
    tokens is a few megabytes, and a plain matmul any backward can pass.
    """
    if hasattr(head, "dense_rows"):
        import mlx.nn as nn

        weight = head.dense_rows(rows)
        dense = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        dense.weight = weight
        dense.freeze()
        return dense
    sub = copy.copy(head)
    for name in ("weight", "scales", "biases", "bias"):
        if name in head:
            sub[name] = head[name][rows]
    return sub


def subset_projection(head: Any, rows: Any, tied: bool) -> Callable[[Any], Any]:
    """hidden -> logits over `rows` of `head` only."""
    sub = cut_head(head, rows)
    return sub.as_linear if tied else sub


# Where the cut head and the vocabulary -> row map live on the model while it
# trains. mlx_lm compiles the training step with the model's state as its
# inputs, and mx.compile refuses any array the step reaches that is not among
# them ("Attempting to compile a function with uncaptured inputs") — so they
# cannot be closed over, they have to BE model state. Frozen, so the adapter
# file never sees them.
TRIM_ATTR = "symbio_trim"


def _trim_module(head_cut: Any, remap: Any):
    import mlx.nn as nn

    class Trim(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head_cut
            self.remap = remap

    trim = Trim()
    trim.freeze()
    return trim


def build_trimmed_loss(model: Any, rows_list: list[int], sample: list[int]):
    """The loss over `rows_list`, or None when it cannot match the model's own logits."""
    import mlx.core as mx
    import mlx.nn as nn

    lm = _language_model(model)
    head, tied = output_head(model)
    if head is None or getattr(lm, "model", None) is None:
        say("[trim-vocab] no output head found; training with the full vocabulary.")
        return None
    rows = mx.array(rows_list, dtype=mx.uint32)
    try:
        head_cut = cut_head(head, rows)
    except Exception as e:  # noqa: BLE001 - any layer this cannot cut
        say(f"[trim-vocab] could not cut the output head ({e}); training with the "
            f"full vocabulary.")
        return None
    cap = getattr(getattr(lm, "args", None), "final_logit_softcapping", None)

    def logits_of(model, inputs):
        # Everything is reached through `model`, never a closure: see TRIM_ATTR.
        cut = model[TRIM_ATTR].head
        hidden = _language_model(model).model(inputs)
        out = cut.as_linear(hidden) if tied else cut(hidden)
        return mx.tanh(out / cap) * cap if cap else out

    # The same numbers or nothing: checked on a real sample before training.
    probe = mx.array([sample[:64]])
    full = model(probe)
    vocab = full.shape[-1]
    remap_list = [0] * vocab
    for index, token in enumerate(rows_list):
        if token < vocab:
            remap_list[token] = index
    setattr(model, TRIM_ATTR, _trim_module(head_cut, mx.array(remap_list, dtype=mx.int32)))
    want = mx.take(full, rows, axis=-1).astype(mx.float32)
    got = logits_of(model, probe).astype(mx.float32)
    scale = float(mx.abs(want).max().item()) or 1.0
    error = float(mx.abs(want - got).max().item())
    del full, want, got
    if error > 2e-2 * scale:
        del model[TRIM_ATTR]
        say(f"[trim-vocab] trimmed logits differ from the model's own by {error:.3g} "
            f"(scale {scale:.3g}); training with the full vocabulary.")
        return None

    def loss(model, batch, lengths):
        # mlx_lm's default_loss, over the trimmed logits.
        inputs = batch[:, :-1]
        targets = model[TRIM_ATTR].remap[batch[:, 1:]]
        logits = logits_of(model, inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        ce = ce.astype(mx.float32).sum() / ntoks
        return ce, ntoks

    say(f"[trim-vocab] scoring {len(rows_list):,} of {vocab:,} tokens "
        f"({100 * len(rows_list) / vocab:.1f}%); trimmed logits match the model's "
        f"own to {error:.2g}.")
    return loss


# ---------------------------------------------------------- linear attention


def checkpoint_gated_delta(chunk: int = CHUNK) -> bool:
    """Make the training-time gated-delta loop keep one state per chunk."""
    import mlx.core as mx
    from mlx_lm.models import gated_delta

    original = gated_delta.gated_delta_ops
    if getattr(original, "symbio_chunked", False):
        return False

    def run_chunk(q, k, v, g, beta, state, *mask):
        return original(q, k, v, g, beta, state, mask[0] if mask else None)

    run_chunk = mx.checkpoint(run_chunk)

    def chunked_ops(q, k, v, g, beta, state=None, mask=None):
        T = q.shape[1]
        if T <= chunk:
            return original(q, k, v, g, beta, state, mask)
        if state is None:
            B, _, _, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        ys = []
        for start in range(0, T, chunk):
            part = slice(start, start + chunk)
            extra = () if mask is None else (mask[:, part],)
            y, state = run_chunk(q[:, part], k[:, part], v[:, part], g[:, part],
                                 beta[:, part], state, *extra)
            ys.append(y)
        return mx.concatenate(ys, axis=1), state

    chunked_ops.symbio_chunked = True
    gated_delta.gated_delta_ops = chunked_ops
    return True


_ALWAYS_EVAL: dict[type, type] = {}


def kernel_below_adapters(model: Any) -> int:
    """Run the gated-delta layers below the first adapter on the inference kernel.

    No gradient reaches a layer with no trainable parameter at or below it, so
    there is nothing for the loop's backward to do there — only its cost. A
    module reports `training` False for good (the trainer flips the flag back
    on every evaluation), which is the switch mlx_lm's layer reads. Returns how
    many modules were switched.
    """
    from mlx.utils import tree_flatten

    layers = list(getattr(model, "layers", []))
    first = next((i for i, layer in enumerate(layers)
                  if tree_flatten(layer.trainable_parameters())), len(layers))
    switched = 0
    for layer in layers[:first]:
        for _, module in layer.named_modules():
            if not (hasattr(module, "A_log") and hasattr(module, "dt_bias")):
                continue
            cls = type(module)
            frozen = _ALWAYS_EVAL.get(cls)
            if frozen is None:
                frozen = type(cls.__name__, (cls,), {"training": property(lambda self: False)})
                _ALWAYS_EVAL[cls] = frozen
            module.__class__ = frozen
            switched += 1
    return switched


# ------------------------------------------------------------------ memory

# Measured on Ternary-Bonsai-2 27B (7.68 GB of weights), one LoRA step over 8
# layers at ~100 tokens: 9.5-9.8 GB peak. Under mlx_lm's own settings the run
# died anyway, at free memory 7% — MLX kept ~2 GB of freed buffers in its
# cache for reuse, and trainer.train wires memory up to the device's whole
# recommended working set (12.7 GB here), so the OS could not even page it.
# Same step with the cache held to 512 MB: lowest free memory 29%.
CACHE_LIMIT_MB = 512
WIRED_HEADROOM_GB = 2.5


@contextlib.contextmanager
def memory_bounds(model: Any):
    """Cap MLX's buffer cache, and the wired limit at weights + headroom."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    weights = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    ceiling = int(weights + WIRED_HEADROOM_GB * 2**30)
    old_cache = mx.set_cache_limit(CACHE_LIMIT_MB * 2**20)
    real = mx.set_wired_limit
    # trainer.train sets the wired limit itself, to the device maximum.
    mx.set_wired_limit = lambda limit: real(min(int(limit), ceiling))
    try:
        yield
    finally:
        mx.set_wired_limit = real
        mx.set_cache_limit(old_cache)


# ------------------------------------------------------------------- driver


def _pop_flag(argv: list[str], flag: str, default: str) -> str:
    if flag not in argv:
        return default
    at = argv.index(flag)
    value = argv[at + 1] if at + 1 < len(argv) else default
    del argv[at:at + 2]
    return value


def _vocab_size(model: Any) -> int:
    head, _tied = output_head(model)
    weight = getattr(head, "weight", None)
    return int(weight.shape[0]) if weight is not None else 0


def cap_learning_rate(argv: list[str], cap: str) -> str | None:
    """Lower --learning-rate in argv to `cap`; the note to log, or None."""
    if cap.lower() == "off":
        return None
    limit = float(cap)
    for flag in ("--learning-rate", "--learning_rate"):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            at = argv.index(flag) + 1
            asked = float(argv[at])
            if asked > limit:
                argv[at] = f"{limit:g}"
                return (f"[lean-lr] learning rate {asked:g} lowered to {limit:g}: higher rates "
                        f"diverged on this model family ({MAX_LR_FLAG} off to keep it).")
            return None
    return None


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    trim_mode = _pop_flag(argv, TRIM_FLAG, "auto").lower()
    note = cap_learning_rate(argv, _pop_flag(argv, MAX_LR_FLAG, f"{MAX_LR:g}"))
    if note:
        say(note)

    from symbio.app import mlx_compat

    mlx_compat.ensure()  # 1-bit layers and the rest of Symbio's model patches

    from mlx_lm import lora as mlx_lora
    from mlx_lm.tuner import trainer

    original_load = mlx_lora.load
    original_train_model = mlx_lora.train_model

    def load(*args, **kwargs):
        global _TOKENIZER
        model, tokenizer = original_load(*args, **kwargs)
        _TOKENIZER = tokenizer
        return model, tokenizer

    def train_model(args, model, train_set, valid_set, training_callback=None):
        trimmed = None
        vocab = _vocab_size(model)
        if trim_mode == "on" or (trim_mode == "auto" and vocab >= AUTO_TRIM_VOCAB):
            rows = used_token_ids([train_set, valid_set], _TOKENIZER)
            first = train_set.process(train_set[0]) if len(train_set) else [0]
            first = first[0] if isinstance(first, tuple) else first  # (tokens, offset)
            trimmed = build_trimmed_loss(model, rows, [int(t) for t in first])
        else:
            say(f"[trim-vocab] {trim_mode}: scoring the whole {vocab:,}-token vocabulary.")
        loss = trimmed or trainer.default_loss
        if checkpoint_gated_delta():
            say(f"[lean-gdn] linear-attention backward checkpointed every {CHUNK} tokens.")

        def train(model, optimizer, train_dataset, val_dataset=None, args=None,
                  training_callback=None, **kwargs):
            # By now the adapters exist, so the layers below them are known.
            switched = kernel_below_adapters(model)
            if switched:
                say(f"[lean-gdn] {switched} linear-attention layer(s) below the "
                    f"adapters run on the inference kernel.")
            kwargs.setdefault("loss", loss)
            with memory_bounds(model):
                return trainer.train(model, optimizer, train_dataset, val_dataset,
                                     args=args, training_callback=training_callback,
                                     **kwargs)

        mlx_lora.train = train
        return original_train_model(args, model, train_set, valid_set, training_callback)

    mlx_lora.load = load
    mlx_lora.train_model = train_model
    sys.argv = [sys.argv[0], *argv]
    mlx_lora.main()


if __name__ == "__main__":
    main()
