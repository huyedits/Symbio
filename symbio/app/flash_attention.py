"""Flash attention over a quantized KV cache (mlx-lm 0.31.3, mlx 0.32).

With `agent.kv_bits` set, the cache holds keys and values at 8 (or 4) bits, and
mlx_lm then leaves the fused attention kernel for
`quantized_scaled_dot_product_attention`, which materialises the whole score
matrix — every query head against every cached token, as a real array — plus a
mask the same size, before the softmax. That is attention without flash
attention. For the 14B (40 query heads) prefilling a 2,048-token chunk against
a 6,000-token context it is about a gigabyte per layer in flight, on a machine
whose whole problem is memory; and it is slower, because none of it is fused.

llama.cpp runs a quantized KV cache THROUGH flash attention: its Metal
kernels dequantize K/V tiles inside the kernel and never build the score
matrix, and a quantized V cache is refused outright unless flash attention is
on (`--flash-attn auto`, the default since 2025, switches it on for you).
Ollama and LM Studio inherit that. MLX has no fused kernel that takes
quantized keys (mlx-lm#1587 measures kv_bits=8 using MORE peak memory than an
fp16 cache for exactly this reason), so this does the next best thing for any
step with more than a few query
rows (a prompt prefill, a speculative verify): dequantize this layer's K and V
— with grouped-query attention that is only 8 KV heads, a few megabytes — and
hand them to mx.fast.scaled_dot_product_attention, the fused kernel. The cache
itself stays quantized; only the transient differs.

Single-token decode keeps mlx_lm's quantized path: there the score row is tiny
and reading the 8-bit cache directly moves half the bytes of dequantizing it.

It is the same attention over the same stored values, computed with fp32
accumulation instead of fp16 scores. Measured 2026-09-26, M4 16 GB, 8-bit KV:

    Qwen3-14B-3bit, 6,000-token prompt   prefill   peak over the weights
      mlx_lm as shipped                    60.3 s    +2,595 MB
      fused (this)                         53.3 s    +1,132 MB
      fp16 cache, no kv_bits               54.4 s    +1,517 MB
    all three: the same 24 greedy tokens

    Qwen3-0.6B, 6,008-token real system prompt, KL to the fp16-cache
    distribution after prefill: as shipped 2.9e-2, fused 2.4e-3.
"""

from __future__ import annotations

import sys

# Query rows at or above which the fused path is taken. Decode is 1; a
# speculative verify is draft+1 (4 by default); a prefill chunk is up to 2,048.
MIN_QUERY_ROWS = 8

_installed = False


def _flash_sdpa(original):
    import mlx.core as mx

    def scaled_dot_product_attention(queries, keys, values, cache, scale, mask, sinks=None):
        bits = getattr(cache, "bits", None)
        if (bits and sinks is None and isinstance(keys, (tuple, list))
                and queries.shape[-2] >= MIN_QUERY_ROWS):
            group_size = getattr(cache, "group_size", 64)
            k = mx.dequantize(*keys, group_size=group_size, bits=bits).astype(queries.dtype)
            v = mx.dequantize(*values, group_size=group_size, bits=bits).astype(queries.dtype)
            return mx.fast.scaled_dot_product_attention(queries, k, v, scale=scale, mask=mask)
        return original(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)

    scaled_dot_product_attention.__wrapped__ = original
    return scaled_dot_product_attention


def install() -> bool:
    """Route quantized-cache attention through the fused kernel. Idempotent.

    Model modules bind `scaled_dot_product_attention` by name when they are
    imported (`from .base import scaled_dot_product_attention`), so patching
    mlx_lm.models.base alone reaches only models imported afterwards. Every
    model module already loaded is re-pointed as well.
    """
    global _installed
    if _installed:
        return True
    try:
        from mlx_lm.models import base
    except ImportError:
        return False
    original = base.scaled_dot_product_attention
    patched = _flash_sdpa(original)
    base.scaled_dot_product_attention = patched
    for name, module in list(sys.modules.items()):
        if (name.startswith("mlx_lm.models.") and module is not None
                and getattr(module, "scaled_dot_product_attention", None) is original):
            module.scaled_dot_product_attention = patched
    _installed = True
    return True
