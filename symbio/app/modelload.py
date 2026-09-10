"""One place to load a model, so a mis-packaged repo cannot hang every turn.

mlx_lm builds its stop-token set from generation_config.json and ignores what
the tokenizer itself declares. When a repo disagrees with itself, nothing
stops generation: the model emits its end token as ordinary text, keeps going,
and every turn runs the full max_reply_tokens budget.

Measured 2026-09-02 on mlx-community/Hermes-4-14B-4bit:

    <|im_end|>              -> 151645   (what its chat template ends turns with)
    tokenizer.eos_token_id  =  151645
    generation_config       =  151643   (<|endoftext|>)
    mlx_lm eos_token_ids    = {151643}

Every routing case ran to the 1,152-token ceiling and answered with the end
tag printed mid-reply. It reads exactly like a broken model, and it is one
wrong integer in a JSON file. Adding an id the wrapper already has is a no-op,
so a correctly packaged model is unaffected.
"""

from __future__ import annotations

from typing import Any


def trust_tokenizer_eos(tokenizer: Any) -> int | None:
    """Add the tokenizer's own eos to the wrapper's stop set. Returns it if it
    was missing, else None. Never raises: a stop-token repair must not be able
    to stop a model loading."""
    try:
        real = getattr(tokenizer, "eos_token_id", None)
        ids = getattr(tokenizer, "eos_token_ids", None)
        if real is None or ids is None or real in ids:
            return None
        ids.add(real)
        return real
    except Exception:
        return None


def load(*args: Any, **kwargs: Any):
    """mlx_lm.load, with the stop-token set repaired.

    Imported inside the call because the lazy CLI path must not pull mlx_lm in
    just to run a slash command -- see the boot-time work behind `symb chat`
    feeling instant.
    """
    from symbio import adapter_integrity, backend

    config = kwargs.pop("config", None)
    # Before the weights are handed to the model, not after. Every caller that
    # loads an adapter comes through here, so this is the one place that can
    # say "these are not the bytes that were sealed" while it still means
    # anything — see symbio/adapter_integrity.py.
    adapter_integrity.enforce(kwargs.get("adapter_path"), config)

    if backend.is_cuda(config):
        # Delegated whole: the CUDA path has its own tokenizer handling and
        # none of the mlx-specific repair below applies to it.
        # config= as well, or the backend re-selects from env/detection
        # inside and answers a different question than the branch above. Two
        # concrete effects: cuda.load_in_bits was read off an empty dict, so a
        # user asking for 8- or 16-bit silently got 4-bit NF4; and a config
        # naming cuda on a machine detect() calls mlx made the outer branch
        # true and the inner one false, which re-entered this function and
        # quietly loaded through mlx_lm instead of saying CUDA was
        # unavailable.
        return backend.load(args[0] if args else kwargs.pop("model_name"),
                            adapter_path=kwargs.pop("adapter_path", None),
                            config=config)

    from mlx_lm import load as _mlx_load

    # Mistral tokenizers ship a pre-tokenizer regex that mis-splits some
    # Unicode (transformers warns and asks for fix_mistral_regex=True on load).
    # It is a no-op for every non-Mistral tokenizer, so pass it unconditionally
    # rather than special-casing the model name.
    tokenizer_config = dict(kwargs.get("tokenizer_config") or {})
    tokenizer_config.setdefault("fix_mistral_regex", True)
    kwargs["tokenizer_config"] = tokenizer_config

    model, tokenizer = _mlx_load(*args, **kwargs)
    added = trust_tokenizer_eos(tokenizer)
    if added is not None:
        print(f"  [Model] repo's generation_config omits eos {added}; added it "
              f"so generation stops (without it every reply runs to the token "
              f"limit).")
    return model, tokenizer
