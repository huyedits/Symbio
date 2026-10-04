"""`mlx_lm lora` with Symbio's engine patches and an optional vocabulary skin.

    python -m symbio.app.lora_runner [--vocab-skin on|off|auto]
                                     [--vocab-skin-base N] <mlx_lm lora flags>

The trainer runs as its own process (backend.trainer_command says why), so
patches applied in the chat process never reach it. This entry point applies
symbio.app.mlx_compat first — which is what lets 1-bit checkpoints load at
all — then hands the remaining argv to mlx_lm's own lora main().

With the skin on, the output head is cut to the vocabulary the corpus uses for
the length of the run (see symbio.app.vocab_skin). The adapter it writes is
the same kind of file and loads against the full head.
"""

from __future__ import annotations

import sys


def _pop_flag(argv: list[str], name: str, default: str) -> str:
    if name in argv:
        i = argv.index(name)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


def _install_skin(mode: str, base: int) -> None:
    import mlx.core as mx
    from mlx_lm import lora as _lora

    from symbio.app import vocab_skin

    state: dict = {}
    orig_train, orig_evaluate = _lora.train, _lora.evaluate

    def _prepare(model, datasets):
        if "remap" in state:
            return state["remap"]
        rows = vocab_skin.head_rows(model)
        if mode == "auto" and rows <= vocab_skin.AUTO_THRESHOLD:
            state["remap"] = None
            return None

        def all_tokens():
            for ds in datasets:
                if ds is not None:
                    yield from vocab_skin.dataset_tokens(ds)

        keep = vocab_skin.keep_ids(all_tokens(), state["tokenizer"], rows, base)
        state["remap"] = vocab_skin.skin(model, keep)
        mx.eval(model.parameters())
        print(f"[vocab-skin] output head {rows:,} -> {len(keep):,} rows "
              f"(base {base:,} + corpus + specials); logits memory "
              f"{rows / len(keep):.1f}x smaller. Reported loss is over the "
              f"kept vocabulary.", flush=True)
        return state["remap"]

    orig_load_dataset = _lora.load_dataset

    def load_dataset(args, tokenizer):
        state["tokenizer"] = tokenizer
        return orig_load_dataset(args, tokenizer)

    def train(*args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        remap = _prepare(model, [kwargs.get("train_dataset"), kwargs.get("val_dataset")])
        if remap is not None and "loss" not in kwargs:
            kwargs["loss"] = vocab_skin.skinned_loss(remap)
        return orig_train(*args, **kwargs)

    def evaluate(*args, **kwargs):
        remap = state.get("remap")
        if remap is not None and "loss" not in kwargs:
            kwargs["loss"] = vocab_skin.skinned_loss(remap)
        return orig_evaluate(*args, **kwargs)

    _lora.load_dataset = load_dataset
    _lora.train = train
    _lora.evaluate = evaluate


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = _pop_flag(argv, "--vocab-skin", "auto").lower()

    from symbio.app import mlx_compat, vocab_skin

    base = int(_pop_flag(argv, "--vocab-skin-base", str(vocab_skin.DEFAULT_BASE)))

    mlx_compat.ensure()
    if mode not in ("off", "false", "0", "no"):
        _install_skin("on" if mode in ("on", "true", "1", "yes") else "auto", base)

    from mlx_lm import lora as _lora

    sys.argv = ["mlx_lm.lora", *argv]
    _lora.main()


if __name__ == "__main__":
    main()
