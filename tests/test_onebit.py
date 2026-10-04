"""1-bit affine weights on stock MLX (Prism ML's Bonsai 1-bit checkpoints).

Stock MLX rejects bits=1. symbio/app/onebit.py keeps the packed weights and
computes with them directly. These check every path against a float reference
built from the same bits, and that mlx_lm's quantize seam builds the layers.
"""
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from symbio.app import onebit  # noqa: E402

GS = 128


def _weights(N, K, seed=0):
    """Packed 1-bit weights plus the float matrix they encode (w = s*bit + b)."""
    mx.random.seed(seed)
    bits = mx.random.uniform(shape=(N, K)) > 0.5
    scales = mx.random.uniform(0.01, 0.1, (N, K // GS)).astype(mx.float16)
    biases = -scales / 2
    dense = (
        bits.astype(mx.float32).reshape(N, K // GS, GS) * scales[..., None].astype(mx.float32)
        + biases[..., None].astype(mx.float32)
    ).reshape(N, K)
    shifts = mx.left_shift(mx.array(1, dtype=mx.uint32), mx.arange(32, dtype=mx.uint32))
    packed = (bits.astype(mx.uint32).reshape(N, K // 32, 32) * shifts).sum(-1).astype(mx.uint32)
    return packed, scales, biases, dense


def _rel_err(y, ref):
    return (mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item()


def test_the_two_bit_respread_is_bit_exact():
    packed, scales, biases, dense = _weights(37, 512)
    assert mx.array_equal(onebit.dequantize(packed, scales, biases, GS).astype(mx.float32), dense)


@pytest.mark.parametrize("rows", [1, 3, onebit.MAX_KERNEL_ROWS - 1])
def test_the_packed_bit_kernel_matches_the_float_matmul(rows):
    # 37 output rows is not a multiple of the kernel's row block.
    packed, scales, biases, dense = _weights(37, 512)
    x = mx.random.normal((1, rows, 512)).astype(mx.float16)
    y = onebit.matmul(x, packed, scales, biases, GS)
    assert y.shape == (1, rows, 37)
    assert _rel_err(y, x.astype(mx.float32) @ dense.T) < 2e-3


def test_a_prefill_goes_through_quantized_matmul_and_matches():
    packed, scales, biases, dense = _weights(64, 256)
    x = mx.random.normal((40, 256)).astype(mx.float16)
    assert _rel_err(onebit.matmul(x, packed, scales, biases, GS), x.astype(mx.float32) @ dense.T) < 2e-3


def test_nn_quantize_builds_one_bit_layers_that_load_and_run():
    assert onebit.install()
    packed, scales, biases, dense = _weights(64, 256)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 256)
            self.proj = nn.Linear(256, 64, bias=False)

    model = Tiny()
    nn.quantize(model, group_size=GS, bits=1)
    assert isinstance(model.proj, nn.QuantizedLinear) and model.proj.bits == 1
    assert isinstance(model.embed, nn.QuantizedEmbedding) and model.embed.bits == 1
    params = {"weight": packed, "scales": scales, "biases": biases}
    model.load_weights([(f"{m}.{k}", v) for m in ("embed", "proj") for k, v in params.items()])

    ids = mx.array([[3, 0, 63]])
    assert mx.array_equal(model.embed(ids).astype(mx.float32), dense[ids])
    x = mx.random.normal((1, 1, 256)).astype(mx.float16)
    assert _rel_err(model.proj(x), x.astype(mx.float32) @ dense.T) < 2e-3
    assert _rel_err(model.embed.as_linear(x), x.astype(mx.float32) @ dense.T) < 2e-3


def test_other_bit_widths_still_use_mlx_own_layers():
    assert onebit.install()
    layer = nn.Linear(256, 64).to_quantized(group_size=64, bits=4)
    assert type(layer) is nn.QuantizedLinear
