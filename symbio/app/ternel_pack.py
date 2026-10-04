"""Ternary-Bonsai-27B, losslessly repacked at 1.75 bits/weight — made trainable.

inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw holds the same ternary
weights as prism-ml/Ternary-Bonsai-27B-mlx-2bit (Qwen3.6-27B), verified
bit-exact, in 5.89 GB instead of 7.57 GB. Its config.json names a model_file,
ternel_packed_model.py, which mlx_lm 0.31.3 imports from the checkpoint: every
nn.Linear and the embedding become a PackedLinear / PackedEmbedding that run
custom Metal kernels straight from the packed codes. Audited before use: it
imports only math, dataclasses, functools, mlx and mlx_lm, and touches no
file, process or network.

Those kernels answer inference only. mx.fast.metal_kernel has no backward, and
the layer has no LoRA hook, so mlx_lm's tuner could neither wrap it nor pass a
gradient through it. After such a model loads, this adds three things to its
PackedLinear class:

  * to_lora: mlx_lm's tuner wraps a packed projection in an adapter;
  * a backward for the input, used only while training: dx = dy . W, with W
    decoded from the codes by the checkpoint's own row-decode kernel, one
    layer at a time, for the length of the step;
  * dense_rows(rows): the trimmed-vocabulary loss (symbio.trim_lora) decodes
    just the head rows the corpus uses.

In eval mode a PackedLinear runs exactly the shipped kernel path.
"""

from __future__ import annotations

import math
import sys
from typing import Any

PREFILL_STEP = 256  # see adapt(); measured like prism_pack.PREFILL_STEP

_installed = False


def is_ternel(model: Any) -> bool:
    return hasattr(model, "packed_module_counts")


def _packed_linear_class(model: Any):
    for _, module in model.named_modules():
        if type(module).__name__ == "PackedLinear" and hasattr(module, "layout"):
            return type(module)
    return None


def adapt(model: Any) -> bool:
    """Give a ternel model's PackedLinear class LoRA, a backward and dense_rows."""
    cls = _packed_linear_class(model)
    if cls is None:
        return False
    # The checkpoint's prefill, like a Prism pack's, builds a [24, chunk,
    # context] fp32 score matrix per full-attention layer (head_dim 256).
    model.symbio_prefill_step = PREFILL_STEP
    if getattr(cls, "symbio_adapted", False):
        return True
    import mlx.core as mx
    import mlx.nn as nn

    code = sys.modules[cls.__module__]  # the checkpoint's own module
    shipped_call = cls.__call__

    def decode(codes, scales, layout, rows, dtype):
        return code.tq1_get_rows(codes, scales, rows.astype(code.INDEX_DTYPE),
                                 groups_per_row=layout.groups_per_row, tile=layout.tile,
                                 dtype=dtype, threads=code.GET_ROWS_THREADS, safe_clamp=False)

    def dense_rows(self, rows, dtype=mx.bfloat16):
        """The weight rows `rows` of this layer, decoded: [len(rows), in]."""
        return decode(self.codes, self.scales, self.layout, rows, dtype)

    def __call__(self, x):
        if not self.training:
            return shipped_call(self, x)
        layout = self.layout

        @mx.custom_function
        def matmul(x, codes, scales):
            return code.packed_matmul(codes, scales, x, layout=layout)

        @matmul.vjp
        def matmul_vjp(primals, cotangent, output):
            x, codes, scales = primals
            weight = decode(codes, scales, layout, mx.arange(layout.rows), x.dtype)
            dx = (cotangent.reshape(-1, layout.rows) @ weight).reshape(x.shape)
            return dx, mx.zeros_like(codes), mx.zeros_like(scales)

        return matmul(x, self.codes, self.scales)

    def to_lora(self, r: int = 8, scale: float = 20.0, dropout: float = 0.0):
        return PackedLinearLoRA(self, r=r, scale=scale, dropout=dropout)

    class PackedLinearLoRA(nn.Module):
        """mlx_lm's LoRALinear around a PackedLinear: same keys, same maths."""

        def __init__(self, base, r=8, scale=20.0, dropout=0.0):
            super().__init__()
            self.linear = base
            self.dropout = nn.Dropout(p=dropout)
            self.scale = scale
            columns, rows = base.layout.columns, base.layout.rows
            bound = 1 / math.sqrt(columns)
            self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(columns, r))
            self.lora_b = mx.zeros(shape=(r, rows))

        def __call__(self, x):
            y = self.linear(x)
            z = (self.dropout(x) @ self.lora_a) @ self.lora_b
            return y + (self.scale * z).astype(x.dtype)

    cls.__call__ = __call__
    cls.dense_rows = dense_rows
    cls.to_lora = to_lora
    cls.symbio_adapted = True
    return True


def install() -> bool:
    """Adapt every ternel model mlx_lm.utils.load_model builds. Idempotent."""
    global _installed
    if _installed:
        return False
    from mlx_lm import utils

    original = utils.load_model

    def load_model(*args, **kwargs):
        model, config = original(*args, **kwargs)
        if is_ternel(model):
            adapt(model)
        return model, config

    load_model.__wrapped__ = original
    utils.load_model = load_model
    _installed = True
    return True
