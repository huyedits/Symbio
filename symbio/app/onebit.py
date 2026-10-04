"""1-bit affine weights on stock MLX (mlx 0.32, mlx-lm 0.31.3).

Prism ML's 1-bit Bonsai checkpoints (prism-ml/Bonsai-27B-mlx-1bit and its
smaller siblings) declare `quantization: {bits: 1, group_size: 128}` in MLX's
ordinary affine layout: one sign bit per weight, packed 32 to a uint32, with a
per-group scale and bias (w = scale * bit + bias, i.e. bias = -s, scale = 2s).
Stock MLX supports 2-8 bits only, so mlx_lm.load() dies in mx.quantize while
building the empty layers, and Prism's answer is a fork of MLX itself.

This needs no fork. A 1-bit layer here keeps the packed weights exactly as
shipped (5.1 GB resident for the 27B, not the 7.9 GB of the ternary 2-bit) and
computes with them two ways:

  * few query rows (decode, a speculative verify): a Metal kernel reads the
    packed bits directly — one simdgroup per eight output rows, the fp32 dot
    product split into "sum of x where the bit is set" times the scale plus
    "sum of x" times the bias, so no weight is ever expanded.
  * many rows (a prompt prefill): the layer's bits are respread to MLX's 2-bit
    layout on the fly — each 1-bit value becomes the 2-bit code 0 or 1 under
    the same scale and bias, which is bit-exact — and handed to
    mx.quantized_matmul, MLX's own tiled kernel. The 2-bit copy is transient,
    one layer at a time.

Embeddings gather the requested rows and dequantize only those.

The layers subclass nn.QuantizedLinear / nn.QuantizedEmbedding so isinstance
checks elsewhere (LoRA wrapping, the model loader) still treat them as
quantized. Fine-tuning these bases is out of scope: Bonsai's 248k vocabulary
is past the 16 GB LoRA wall regardless of the weight format.
"""

from __future__ import annotations

_installed = False

# Query rows below which the packed-bit kernel runs; at or above, the 2-bit
# respread + quantized_matmul path does. Decode is 1, a verify is draft+1.
MAX_KERNEL_ROWS = 8
# Output rows per simdgroup (each word of x is read once for all of them) and
# simdgroups per threadgroup. 8x8 measured best on the M4, decode (M=1), chained
# calls: 55-60 GB/s of packed weights, against 64-70 GB/s for MLX's own 2-bit
# qmv — which moves twice the bytes, so this is ~1.7x faster per layer.
_ROWS = 8
_SG = 8

_kernel = None

_SOURCE = """
    const uint lane = thread_position_in_threadgroup.x;
    const uint sg = thread_position_in_threadgroup.y;
    const uint row0 = (threadgroup_position_in_grid.y * SG + sg) * ROWS;
    const uint m = threadgroup_position_in_grid.z;
    const uint N = w_shape[0];
    const uint KW = w_shape[1];          // packed words per row
    const uint G = scales_shape[1];      // groups per row
    const uint WPG = KW / G;             // words per group
    if (row0 >= N) return;

    const device T* xr = x + m * KW * 32;
    float acc[ROWS];
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (uint wi = lane; wi < KW; wi += 32) {
        float xv[32];
        float xsum = 0.0f;
        const device T* xp = xr + wi * 32;
        for (uint i = 0; i < 32; ++i) {
            xv[i] = static_cast<float>(xp[i]);
            xsum += xv[i];
        }
        const uint g = wi / WPG;
        for (uint r = 0; r < ROWS; ++r) {
            const uint n = row0 + r;
            if (n >= N) break;
            const uint bits = w[n * KW + wi];
            float on = 0.0f;
            for (uint i = 0; i < 32; ++i) {
                on += select(0.0f, xv[i], bool((bits >> i) & 1u));
            }
            acc[r] += static_cast<float>(scales[n * G + g]) * on
                    + static_cast<float>(biases[n * G + g]) * xsum;
        }
    }
    for (uint r = 0; r < ROWS; ++r) {
        const uint n = row0 + r;
        const float total = simd_sum(acc[r]);
        if (lane == 0 && n < N) out[m * N + n] = static_cast<T>(total);
    }
"""


def _get_kernel():
    global _kernel
    if _kernel is None:
        import mlx.core as mx

        _kernel = mx.fast.metal_kernel(
            name="symbio_onebit_qmv",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _kernel


_SPREAD_SOURCE = """
    const uint i = thread_position_in_grid.x;
    uint lo = w[i] & 0xFFFFu;
    uint hi = w[i] >> 16;
    lo = (lo | (lo << 8)) & 0x00FF00FFu;  hi = (hi | (hi << 8)) & 0x00FF00FFu;
    lo = (lo | (lo << 4)) & 0x0F0F0F0Fu;  hi = (hi | (hi << 4)) & 0x0F0F0F0Fu;
    lo = (lo | (lo << 2)) & 0x33333333u;  hi = (hi | (hi << 2)) & 0x33333333u;
    lo = (lo | (lo << 1)) & 0x55555555u;  hi = (hi | (hi << 1)) & 0x55555555u;
    out[2 * i] = lo;
    out[2 * i + 1] = hi;
"""
_spread_kernel = None


def to_two_bit(w):
    """Repack 1-bit packed uint32 words [..., K/32] as 2-bit words [..., K/16].

    Element k of a row sits at bit k%32 of word k//32 in the 1-bit layout and at
    bits 2*(k%16) of word k//16 in the 2-bit one, so each word splits into its
    low and high halves with a zero spread between the bits. The 2-bit code is
    then exactly the 1-bit value, and scale/bias carry over unchanged.
    """
    global _spread_kernel
    import mlx.core as mx

    if _spread_kernel is None:
        _spread_kernel = mx.fast.metal_kernel(
            name="symbio_onebit_spread",
            input_names=["w"],
            output_names=["out"],
            source=_SPREAD_SOURCE,
        )
    (out,) = _spread_kernel(
        inputs=[w],
        grid=(w.size, 1, 1),
        threadgroup=(min(256, max(1, w.size)), 1, 1),
        output_shapes=[(*w.shape[:-1], w.shape[-1] * 2)],
        output_dtypes=[mx.uint32],
    )
    return out


def dequantize(w, scales, biases, group_size):
    import mlx.core as mx

    return mx.dequantize(to_two_bit(w), scales, biases, group_size=group_size, bits=2)


def matmul(x, w, scales, biases, group_size):
    """x @ W.T for 1-bit packed W, x of shape [..., K]."""
    import mlx.core as mx

    lead = x.shape[:-1]
    K = x.shape[-1]
    N = w.shape[0]
    rows = 1
    for d in lead:
        rows *= d
    if rows >= MAX_KERNEL_ROWS or group_size % 32 or x.dtype not in (
        mx.float16, mx.bfloat16, mx.float32,
    ):
        return mx.quantized_matmul(
            x, to_two_bit(w), scales, biases, transpose=True,
            group_size=group_size, bits=2,
        )
    blocks = -(-N // _ROWS)
    (out,) = _get_kernel()(
        inputs=[x.reshape(rows, K), w, scales, biases],
        template=[("T", x.dtype), ("ROWS", _ROWS), ("SG", _SG)],
        grid=(32, -(-blocks // _SG) * _SG, rows),
        threadgroup=(32, _SG, 1),
        output_shapes=[(rows, N)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(*lead, N)


def _layers():
    import mlx.core as mx
    import mlx.nn as nn

    class OneBitLinear(nn.QuantizedLinear):
        """A Linear layer over 1-bit affine weights, as Bonsai ships them."""

        def __init__(self, input_dims, output_dims, bias=True, group_size=128):
            nn.Module.__init__(self)
            self.group_size, self.bits, self.mode = group_size, 1, "affine"
            groups = input_dims // group_size
            self.weight = mx.zeros((output_dims, input_dims // 32), dtype=mx.uint32)
            self.scales = mx.zeros((output_dims, groups), dtype=mx.float16)
            self.biases = mx.zeros((output_dims, groups), dtype=mx.float16)
            if bias:
                self.bias = mx.zeros((output_dims,))
            self.freeze()

        def __call__(self, x):
            y = matmul(x, self["weight"], self["scales"], self["biases"], self.group_size)
            if "bias" in self:
                y = y + self["bias"]
            return y

    class OneBitEmbedding(nn.QuantizedEmbedding):
        """An Embedding over 1-bit affine weights; dequantizes only looked-up rows."""

        def __init__(self, num_embeddings, dims, group_size=128):
            nn.Module.__init__(self)
            self.group_size, self.bits, self.mode = group_size, 1, "affine"
            self.num_embeddings, self.dims = num_embeddings, dims
            groups = dims // group_size
            self.weight = mx.zeros((num_embeddings, dims // 32), dtype=mx.uint32)
            self.scales = mx.zeros((num_embeddings, groups), dtype=mx.float16)
            self.biases = mx.zeros((num_embeddings, groups), dtype=mx.float16)
            self.freeze()

        def __call__(self, x):
            return dequantize(
                self["weight"][x], self["scales"][x], self["biases"][x], self.group_size
            )

        def as_linear(self, x):
            return matmul(x, self["weight"], self["scales"], self["biases"], self.group_size)

    return OneBitLinear, OneBitEmbedding


def install() -> bool:
    """Make Linear/Embedding.to_quantized(bits=1) build the 1-bit layers. Idempotent.

    mlx_lm.load() builds the model in full precision shapes and calls
    nn.quantize(), which calls to_quantized() on every leaf; that is the one
    seam where bits=1 would otherwise reach mx.quantize and raise.
    """
    global _installed
    if _installed:
        return True
    try:
        import mlx.nn as nn
    except ImportError:
        return False
    OneBitLinear, OneBitEmbedding = _layers()

    orig_linear = nn.Linear.to_quantized
    orig_embedding = nn.Embedding.to_quantized

    def linear_to_quantized(self, group_size=None, bits=None, mode="affine", **kw):
        if bits == 1 and mode == "affine" and not kw.get("quantize_input"):
            out_dims, in_dims = self.weight.shape
            return OneBitLinear(in_dims, out_dims, "bias" in self, group_size or 128)
        return orig_linear(self, group_size=group_size, bits=bits, mode=mode, **kw)

    def embedding_to_quantized(self, group_size=None, bits=None, mode="affine", **kw):
        if bits == 1 and mode == "affine":
            n, dims = self.weight.shape
            return OneBitEmbedding(n, dims, group_size or 128)
        return orig_embedding(self, group_size=group_size, bits=bits, mode=mode, **kw)

    linear_to_quantized.__wrapped__ = orig_linear
    embedding_to_quantized.__wrapped__ = orig_embedding
    nn.Linear.to_quantized = linear_to_quantized
    nn.Embedding.to_quantized = embedding_to_quantized
    _installed = True
    return True
