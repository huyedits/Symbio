"""Which compute stack actually runs the model.

Symbio was written against MLX, which exists only on Apple Silicon: `import
mlx.core` is at the top of chat.py, the trainer shells out to `python -m
mlx_lm lora`, and the vision worker is mlx-vlm. That is the right default on
the machine this was built for and it is a hard wall everywhere else.

This module is the seam. It answers one question — mlx or cuda — and provides
the four operations everything else needs through it: load a model, generate,
stream, and build the LoRA trainer's command line.

  IMPORTANT, AND NOT A DISCLAIMER SO MUCH AS A FACT ABOUT THIS FILE:
  the CUDA path has never been executed. It was written on a Mac with no
  NVIDIA device present, so every CUDA branch here is unrun code. The MLX
  path is deliberately untouched — it delegates to exactly the functions it
  called before, so nothing about the working setup changes shape — and the
  parts that can be tested without a GPU (selection, config, error text,
  the trainer's argv) are tested. Treat the rest as a first draft that
  compiles, not as something known to work.

Selection order: an explicit `backend` in config, then SYMBIO_BACKEND, then
what the machine actually has. Explicit beats detection so a CUDA box can be
told to use CPU, and so a Mac can be told to try the CUDA path in order to
debug it.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

MLX = "mlx"
CUDA = "cuda"
_VALID = (MLX, CUDA)


class BackendUnavailable(RuntimeError):
    """The selected backend cannot run here."""


def detect() -> str:
    """What this machine can actually do, ignoring configuration."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return MLX
    try:
        import torch

        if torch.cuda.is_available():
            return CUDA
    except Exception:
        pass
    # No CUDA and not Apple Silicon: still report mlx so the caller fails at
    # the import with mlx's own message rather than being told "cuda" and
    # failing later somewhere less obvious.
    return MLX


def selected(config: dict[str, Any] | None = None) -> str:
    """The backend to use: config, then environment, then detection."""
    for source in ((config or {}).get("backend"), os.environ.get("SYMBIO_BACKEND")):
        name = str(source or "").strip().lower()
        if name in _VALID:
            return name
        if name:
            print(f"  [Backend] Ignoring unknown backend {name!r}; "
                  f"valid choices are {', '.join(_VALID)}.")
    return detect()


def is_cuda(config: dict[str, Any] | None = None) -> bool:
    return selected(config) == CUDA


def describe(config: dict[str, Any] | None = None) -> str:
    """One line for the banner and /status."""
    name = selected(config)
    if name == CUDA:
        try:
            import torch

            if torch.cuda.is_available():
                dev = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return f"cuda ({dev}, {total:.0f} GB)"
            return "cuda (selected, but no device is visible)"
        except Exception as e:
            return f"cuda (torch unavailable: {e})"
    return f"mlx ({platform.machine()})"


# ---------------------------------------------------------------- torch bits


def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise BackendUnavailable(
            "The CUDA backend needs PyTorch. Install it for your CUDA version "
            "from https://pytorch.org, along with `transformers`, `peft`, "
            "`accelerate` and `bitsandbytes`."
        ) from e
    return torch


def _dtype(torch):
    """bfloat16 where the card supports it, else fp16."""
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _quantization_config(config: dict[str, Any] | None):
    """4-bit NF4 unless asked otherwise.

    MLX models are distributed pre-quantized (the repo names carry -4bit), and
    the CUDA equivalent is loading the full-precision weights under
    bitsandbytes. Without it a 14B does not fit on a 16 GB card, which is the
    same class of machine this project targets.
    """
    bits = int(((config or {}).get("cuda") or {}).get("load_in_bits", 4))
    if bits not in (4, 8):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise BackendUnavailable(
            "The CUDA backend needs `transformers` (and `bitsandbytes` for "
            "4/8-bit loading). Set cuda.load_in_bits to 16 to load unquantized."
        ) from e
    torch = _require_torch()
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_dtype(torch),
    )


# ---------------------------------------------------------------- operations


def load(model_name: str, adapter_path: str | None = None,
         config: dict[str, Any] | None = None, **kwargs: Any):
    """Load (model, tokenizer) on whichever backend is active.

    The MLX branch is the existing call, unchanged, so an Apple Silicon
    install behaves exactly as before this file existed.
    """
    if not is_cuda(config):
        from symbio.app.modelload import load as _mlx_load

        if adapter_path:
            kwargs["adapter_path"] = adapter_path
        return _mlx_load(model_name, **kwargs)

    torch = _require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise BackendUnavailable(
            f"The CUDA backend needs `transformers`: {e}") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        # Needed for batching and for the trainer's collator; the EOS token is
        # the conventional stand-in when a model ships no pad token.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=_dtype(torch),
        quantization_config=_quantization_config(config),
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise BackendUnavailable(
                f"Loading a LoRA adapter on CUDA needs `peft`: {e}") from e
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate(model: Any, tokenizer: Any, prompt: str, max_tokens: int = 256,
             sampler: Any = None, verbose: bool = False,
             config: dict[str, Any] | None = None, **kwargs: Any) -> str:
    """Generate a completion, returning only the new text.

    Signature mirrors mlx_lm.generate because that is what the call sites
    already pass, `sampler` included — on CUDA the sampler object is not
    applicable and is accepted and ignored rather than raising, so a shared
    call site does not need to know which backend it is talking to.
    """
    if not is_cuda(config):
        from mlx_lm.generate import generate as _mlx_generate

        return _mlx_generate(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens, sampler=sampler,
                             verbose=verbose, **kwargs)

    torch = _require_torch()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **_sampling_kwargs(config),
        )
    # Only the continuation: the prompt is echoed back in `out`, and every
    # caller here expects just the reply.
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True)


def _sampling_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    agent = (config or {}).get("agent", {})
    temperature = float(agent.get("temperature", 0.7) or 0.0)
    if temperature <= 0:
        return {"do_sample": False}
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": float(agent.get("top_p", 0.9)),
        "repetition_penalty": float(agent.get("repetition_penalty", 1.0) or 1.0),
    }


def stream_generate(model: Any, tokenizer: Any, prompt: str,
                    max_tokens: int = 256, sampler: Any = None,
                    config: dict[str, Any] | None = None, **kwargs: Any):
    """Yield chunks as they are produced.

    mlx_lm yields objects carrying `.text`; transformers' TextIteratorStreamer
    yields plain strings. The wrapper below gives the CUDA path the same
    `.text` attribute so the streaming call sites are identical on both.
    """
    if not is_cuda(config):
        from mlx_lm.generate import stream_generate as _mlx_stream

        yield from _mlx_stream(model, tokenizer, prompt=prompt,
                               max_tokens=max_tokens, sampler=sampler, **kwargs)
        return

    torch = _require_torch()
    try:
        from transformers import TextIteratorStreamer
    except ImportError as e:
        raise BackendUnavailable(
            f"Streaming on CUDA needs `transformers`: {e}") from e
    import threading

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def _run():
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_tokens,
                           pad_token_id=tokenizer.pad_token_id,
                           streamer=streamer, **_sampling_kwargs(config))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for chunk in streamer:
        if chunk:
            yield _Chunk(chunk)
    thread.join(timeout=1)


class _Chunk:
    """A streamed piece of text, shaped like mlx_lm's response object."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------- training


def trainer_command(model_name: str, data_dir: str, adapter_dir: str,
                    lora: dict[str, Any], iters: int, config_path: str,
                    config: dict[str, Any] | None = None) -> list[str]:
    """The argv that runs a LoRA fine-tune on the active backend.

    Kept as a command rather than an in-process call for the reason the MLX
    path already documents: the trainer is a second client of the GPU and
    holds its own copy of the weights, so it gets its own process whose
    memory the OS reclaims completely when it exits.
    """
    if not is_cuda(config):
        return [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", model_name,
            "--train",
            "--data", data_dir,
            "--batch-size", str(lora["batch_size"]),
            "--num-layers", str(lora["num_layers"]),
            "--iters", str(iters),
            "--learning-rate", str(lora["learning_rate"]),
            "--steps-per-eval", str(lora["steps_per_eval"]),
            # Without this mlx_lm defaults to 25 batches and re-scores the
            # whole validation split on every evaluation. At this corpus's
            # ~2,160-token samples that is ~11s per batch on an 8B, so a
            # 438-iteration run spent about an hour inside evaluation alone.
            # The point of the number is to spot a plateau or a divergence,
            # and a handful of batches carries that signal — it is a progress
            # check, not a benchmark.
            "--val-batches", str(lora.get("val_batches", 8)),
            "--max-seq-length", str(lora["max_seq_length"]),
            "--adapter-path", adapter_dir,
            "--save-every", str(lora["save_every"]),
            "--config", config_path,
        ]
    # The CUDA trainer takes the same flags, so run_training's progress
    # parsing, early stop and adapter handling need no branch of their own.
    return [
        sys.executable, "-m", "symbio.cuda_lora",
        "--model", model_name,
        "--train",
        "--data", data_dir,
        "--batch-size", str(lora["batch_size"]),
        "--num-layers", str(lora["num_layers"]),
        "--iters", str(iters),
        "--learning-rate", str(lora["learning_rate"]),
        "--steps-per-eval", str(lora["steps_per_eval"]),
        "--val-batches", str(lora.get("val_batches", 8)),
        "--max-seq-length", str(lora["max_seq_length"]),
        "--adapter-path", adapter_dir,
        "--save-every", str(lora["save_every"]),
        "--rank", str(lora.get("rank", 8)),
        "--dropout", str(lora.get("dropout", 0.0)),
        "--scale", str(lora.get("scale", 20.0)),
        "--keys", ",".join(lora.get("keys") or []),
        "--grad-checkpoint", str(bool(lora.get("grad_checkpoint", True))).lower(),
    ]


# ---------------------------------------------------------------- memory


def free_memory_bytes(config: dict[str, Any] | None = None) -> int | None:
    """Memory available to the model, or None when it cannot be read."""
    if not is_cuda(config):
        from symbio.app.training import free_ram_bytes

        return free_ram_bytes()
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def clear_cache(config: dict[str, Any] | None = None) -> None:
    """Hand freed memory back to the allocator."""
    if not is_cuda(config):
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
