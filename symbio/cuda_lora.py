"""LoRA fine-tuning on CUDA, with `mlx_lm lora`'s command line.

Run as `python -m symbio.cuda_lora --model ... --train --data ...`.

The flags are deliberately mlx_lm's, not peft's. run_training already parses
this process's stdout for iteration and loss lines, terminates it on a
plateau, and manages the adapter directory around it; matching the interface
means none of that grows a second code path. Where a flag has no CUDA
equivalent it is accepted and ignored rather than rejected, so the caller
never has to know which trainer it launched.

  NEVER EXECUTED. Written on a Mac with no NVIDIA device. It imports cleanly
  and the argument handling and dataset shaping are unit-tested, but no
  training step in this file has ever run. The mlx_lm path is unchanged and
  remains the tested one.

Two things it deliberately copies from the MLX side, because they were learned
the hard way here and are not peft defaults:

  * Prompt masking. Roughly 99% of the loss in this corpus was the system
    prompt until it was masked out; without masking the model is mostly
    trained to recite its own instructions. Labels before the final assistant
    turn are set to -100.
  * Progress on stdout in mlx_lm's shape ("Iter N: train loss X"), because
    that is what the caller reads to decide whether to stop early.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# The marker that opens the model's own turn in the corpus this project
# writes. Everything from here to the end is what the model should learn to
# produce; everything before it is context it should learn to read.
_ASSISTANT_MARKERS = ("<|im_start|>assistant", "[/INST]", "<|assistant|>")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="symbio.cuda_lora", add_help=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--steps-per-eval", type=int, default=30)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=20.0)
    p.add_argument("--keys", default="")
    p.add_argument("--grad-checkpoint", default="true")
    p.add_argument("--resume-adapter-file", default="")
    # Accepted for command-line parity with mlx_lm; the YAML it points at
    # carries MLX-specific keys that have no meaning here.
    p.add_argument("--config", default="")
    return p.parse_args(argv)


def lora_target_modules(keys: str) -> list[str]:
    """Map the config's MLX projection names onto peft's module names.

    config.lora.keys holds MLX paths like "self_attn.q_proj"; peft matches on
    the bare module name. Anything unrecognised is passed through, so a name
    that is already in peft's form still works.
    """
    out: list[str] = []
    for raw in (k.strip() for k in keys.split(",")):
        if not raw:
            continue
        name = raw.rsplit(".", 1)[-1]
        if name not in out:
            out.append(name)
    return out or ["q_proj", "v_proj"]


def read_jsonl(path: Path) -> list[str]:
    """The "text" field of every row, skipping blanks and malformed lines."""
    rows: list[str] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        text = obj.get("text")
        if isinstance(text, str) and text.strip():
            rows.append(text)
    return rows


def split_index(text: str) -> int:
    """Where the model's own turn begins, or 0 if it cannot be found.

    0 means "train on everything", which is what mlx_lm does without masking.
    Falling back to that is safer than guessing a boundary and masking the
    wrong half.
    """
    best = 0
    for marker in _ASSISTANT_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            best = max(best, idx + len(marker))
    return best


def encode(text: str, tokenizer, max_len: int, mask_prompt: bool = True) -> dict:
    """One training row: input_ids, attention_mask, and masked labels."""
    ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
    labels = list(ids)
    if mask_prompt:
        cut = split_index(text)
        if cut:
            prompt_len = len(tokenizer(text[:cut], truncation=True,
                                       max_length=max_len)["input_ids"])
            # Never mask the whole row: a sample whose prompt fills the window
            # would train on nothing and contribute a NaN loss.
            if 0 < prompt_len < len(labels):
                labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def build_datasets(data_dir: Path, tokenizer, max_len: int):
    from datasets import Dataset

    train = read_jsonl(data_dir / "train.jsonl")
    valid = read_jsonl(data_dir / "valid.jsonl")
    if not train:
        raise SystemExit(f"No training rows found in {data_dir / 'train.jsonl'}")
    enc = lambda rows: Dataset.from_list(  # noqa: E731
        [encode(t, tokenizer, max_len) for t in rows])
    return enc(train), (enc(valid) if valid else None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  DataCollatorForSeq2Seq, Trainer,
                                  TrainingArguments)
    except ImportError as e:
        print(f"[cuda_lora] Missing dependency: {e}. This trainer needs "
              f"torch, transformers, peft, accelerate, datasets and "
              f"bitsandbytes.", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("[cuda_lora] No CUDA device is visible.", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    from symbio.backend import _dtype, _quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=_dtype(torch),
        quantization_config=_quantization_config({}))
    if args.grad_checkpoint.lower() in ("1", "true", "yes"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.rank,
        # mlx_lm's `scale` is the multiplier applied to the LoRA update;
        # peft expresses the same thing as alpha/r, so alpha = scale * r
        # preserves the effective strength the config was tuned at.
        lora_alpha=int(args.scale * args.rank),
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(args.keys),
        # mlx_lm's --num-layers means "tune the last N layers"; peft's
        # equivalent is to restrict which layers get adapters.
        layers_to_transform=_last_n_layers(model, args.num_layers),
    )
    # Continue an existing adapter when one was named, instead of building a
    # fresh one on top of it. The flag was parsed and never read, so the
    # caller printed "Resuming from the existing adapter" while this trained
    # from scratch — the exact silent-nothing failure resume_source exists to
    # catch, reintroduced on the new backend. Refusing outright is better than
    # accepting a flag and ignoring it, so a load that fails stops the run.
    resume = (args.resume_adapter_file or "").strip()
    if resume:
        from peft import PeftModel

        source = Path(resume)
        # The caller may name the weights file; peft loads the directory.
        if source.is_file():
            source = source.parent
        print(f"  [CUDA] Resuming from {source}", flush=True)
        model = PeftModel.from_pretrained(model, str(source), is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds, valid_ds = build_datasets(Path(args.data), tokenizer,
                                        args.max_seq_length)
    adapter_dir = Path(args.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=args.save_every,
        eval_strategy="steps" if valid_ds is not None else "no",
        eval_steps=args.steps_per_eval,
        report_to=[],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.grad_checkpoint.lower() in ("1", "true", "yes"),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                             label_pad_token_id=-100),
        callbacks=[_ProgressInMlxShape()],
    )
    trainer.train()
    # Where the caller expects to find it: mlx_lm writes the adapter to
    # --adapter-path, and everything downstream (adapter_weights_present,
    # the backups, the golden re-check) looks there.
    model.save_pretrained(str(adapter_dir))
    print(f"Saved adapter to {adapter_dir}")
    return 0


def _last_n_layers(model, n: int) -> list[int] | None:
    """The indices of the last `n` decoder layers, or None for all of them."""
    try:
        total = model.config.num_hidden_layers
    except Exception:
        return None
    if n <= 0 or n >= total:
        return None
    return list(range(total - n, total))


class _ProgressInMlxShape:
    """Print progress in the shape run_training's reader already parses.

    Implemented against transformers' callback interface without importing it,
    so this module stays importable (and testable) with transformers absent.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        logs = logs or {}
        step = int(getattr(state, "global_step", 0) or 0)
        if "loss" in logs:
            print(f"Iter {step}: train loss {float(logs['loss']):.3f}, "
                  f"learning rate {float(logs.get('learning_rate', 0.0)):.3e}",
                  flush=True)
        if "eval_loss" in logs:
            loss = float(logs["eval_loss"])
            ppl = math.exp(loss) if loss < 20 else float("inf")
            print(f"Iter {step}: Val loss {loss:.3f}, Val ppl {ppl:.3f}",
                  flush=True)

    # The rest of the callback surface, so transformers can call it freely.
    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
