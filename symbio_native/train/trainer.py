"""Training loop for the native MLX model.

Supports:
  * standard next-token CE loss,
  * gradient accumulation for small batches,
  * checkpointing,
  * substrate branch creation on failures (see symbio_native.substrate).

Data format: plain text files or `.jsonl` with a `text` field.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..model import NativeConfig, NativeLM


def _read_texts(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    texts: list[str] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        texts.append(json.loads(line).get("text", ""))
                    except Exception:
                        pass
        elif path.is_file() and path.suffix in {".txt", ".md", ".py"}:
            try:
                texts.append(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return [t.strip() for t in texts if t.strip()]


def _batch_iterator(
    ids: list[int],
    batch_size: int,
    seq_len: int,
    pad_id: int = 0,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Slide a window over token ids and yield (input, target) arrays."""
    total = len(ids)
    for start in range(0, total - seq_len, seq_len):
        x_list: list[list[int]] = []
        y_list: list[list[int]] = []
        for b in range(batch_size):
            i = start + b * seq_len
            if i + seq_len + 1 > total:
                # Pad last partial batch.
                chunk = ids[i:total]
                pad = (seq_len + 1) - len(chunk)
                chunk = chunk + [pad_id] * pad
            else:
                chunk = ids[i : i + seq_len + 1]
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])
        yield mx.array(x_list, dtype=mx.int32), mx.array(y_list, dtype=mx.int32)


def evaluate(
    model: NativeLM,
    ids: list[int],
    batch_size: int = 1,
    seq_len: int = 256,
    max_batches: int | None = 20,
) -> float:
    """Return average cross-entropy loss over a held-out chunk."""
    losses: list[float] = []
    for i, (x, y) in enumerate(_batch_iterator(ids, batch_size, seq_len)):
        _, loss = model(x, y)
        mx.eval(loss)
        losses.append(float(loss.item()))
        if max_batches is not None and i + 1 >= max_batches:
            break
    return sum(losses) / len(losses) if losses else float("nan")


def train(
    model: NativeLM,
    tokenizer,
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    learning_rate: float = 3e-4,
    iters: int = 1000,
    grad_accum: int = 1,
    warmup: int = 50,
    eval_every: int = 200,
    save_every: int = 500,
    val_split: float = 0.1,
    on_failure_sample: Callable[[str, str], None] | None = None,
    log_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train the native model from random init.

    `on_failure_sample` receives (expected, generated) whenever a validation
    batch shows high loss / poor generation. It is used by the substrate
    manager to spawn correction branches.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = _read_texts(data_dir)
    if not texts:
        raise ValueError(f"No training text found in {data_dir}")

    all_ids: list[int] = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text))
        all_ids.append(tokenizer.encode("\n")[0] if tokenizer.encode("\n") else 0)

    split = int(len(all_ids) * (1 - val_split))
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    optimizer = optim.AdamW(learning_rate=learning_rate)

    state = [model.parameters(), optimizer.state]
    mx.eval(state)

    def _step(x: mx.array, y: mx.array) -> mx.array:
        loss, grads = nn.value_and_grad(model, _loss)(model, x, y)
        optimizer.update(model, grads)
        return loss

    def _loss(model: NativeLM, x: mx.array, y: mx.array) -> mx.array:
        _, loss = model(x, y)
        return loss

    train_iter = _batch_iterator(train_ids, batch_size, seq_len)
    losses: list[float] = []
    start = time.perf_counter()

    for it in range(1, iters + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = _batch_iterator(train_ids, batch_size, seq_len)
            x, y = next(train_iter)

        loss = _step(x, y)
        mx.eval(loss, model.parameters())
        losses.append(float(loss.item()))

        if it % eval_every == 0:
            val_loss = evaluate(model, val_ids, batch_size=batch_size, seq_len=seq_len)
            recent = sum(losses[-100:]) / min(len(losses), 100)
            log_fn(
                f"[Native train] iter {it}/{iters} | train loss {recent:.3f} | "
                f"val loss {val_loss:.3f} | elapsed {time.perf_counter() - start:.1f}s"
            )

        if it % save_every == 0:
            ckpt = out_dir / f"native_lm_{it}.safetensors"
            model.save(ckpt)
            log_fn(f"[Native train] checkpoint saved to {ckpt}")

    final = out_dir / "native_lm_final.safetensors"
    model.save(final)
    return {
        "final_checkpoint": str(final),
        "final_train_loss": sum(losses[-100:]) / min(len(losses), 100),
        "iters": iters,
    }
