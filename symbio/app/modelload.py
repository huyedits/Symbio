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
    from mlx_lm import load as _mlx_load

    model, tokenizer = _mlx_load(*args, **kwargs)
    added = trust_tokenizer_eos(tokenizer)
    if added is not None:
        print(f"  [Model] repo's generation_config omits eos {added}; added it "
              f"so generation stops (without it every reply runs to the token "
              f"limit).")
    return model, tokenizer
