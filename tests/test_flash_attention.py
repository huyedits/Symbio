"""Flash attention over the quantized KV cache.

With agent.kv_bits set, mlx_lm computes attention by materialising the whole
score matrix. The patch hands prefill-sized steps to the fused kernel on K/V
dequantized per layer. These check it is the same attention, that it is taken
only where it pays, and that already-imported model modules are reached.
"""
import pytest

mx = pytest.importorskip("mlx.core")
base = pytest.importorskip("mlx_lm.models.base")

from symbio.app import flash_attention  # noqa: E402


class _QuantCache:
    def __init__(self, bits=8, group_size=64):
        self.bits, self.group_size = bits, group_size


def _qkv(L, S, heads=8, kv_heads=2, dim=128, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, heads, L, dim)).astype(mx.float16)
    k = mx.random.normal((1, kv_heads, S, dim)).astype(mx.float16)
    v = mx.random.normal((1, kv_heads, S, dim)).astype(mx.float16)
    return q, mx.quantize(k, group_size=64, bits=8), mx.quantize(v, group_size=64, bits=8)


@pytest.fixture
def fused():
    original = base.scaled_dot_product_attention
    return flash_attention._flash_sdpa(getattr(original, "__wrapped__", original)), original


def test_a_prefill_is_the_same_attention_through_the_fused_kernel(fused):
    patched, original = fused
    q, k, v = _qkv(L=64, S=200)
    scale = 128 ** -0.5
    # quantized_scaled_dot_product_attention scales its queries IN PLACE
    # (`queries *= scale` on an mx.array), so each call gets its own copy.
    got = patched(mx.array(q), k, v, _QuantCache(), scale=scale, mask="causal")
    want = base.quantized_scaled_dot_product_attention(mx.array(q), k, v, scale=scale, mask="causal")
    assert got.shape == want.shape
    assert mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item() < 2e-2


def test_decode_keeps_the_quantized_path(fused, monkeypatch):
    """One query row: reading the 8-bit cache directly moves half the bytes."""
    patched, _ = fused
    calls = []
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention",
                        lambda *a, **k: calls.append(a) or a[0])
    q, k, v = _qkv(L=1, S=200)
    patched(q, k, v, _QuantCache(), scale=0.1, mask=None)
    assert calls == []


def test_an_unquantized_cache_is_left_alone(fused):
    patched, _ = fused
    mx.random.seed(1)
    q = mx.random.normal((1, 8, 32, 128)).astype(mx.float16)
    k = mx.random.normal((1, 2, 32, 128)).astype(mx.float16)
    v = mx.random.normal((1, 2, 32, 128)).astype(mx.float16)
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.1, mask="causal")
    got = patched(q, k, v, None, scale=0.1, mask="causal")
    assert mx.array_equal(got, want)


def test_install_reaches_models_imported_before_it(monkeypatch):
    """Model modules bind the function by name at import; patching base alone
    would miss every model already loaded."""
    import sys
    import types

    original = base.scaled_dot_product_attention
    fake = types.ModuleType("mlx_lm.models.fake_model")
    fake.scaled_dot_product_attention = original
    monkeypatch.setitem(sys.modules, "mlx_lm.models.fake_model", fake)
    monkeypatch.setattr(flash_attention, "_installed", False)
    monkeypatch.setattr(base, "scaled_dot_product_attention", original)

    assert flash_attention.install() is True
    assert fake.scaled_dot_product_attention is base.scaled_dot_product_attention
    assert base.scaled_dot_product_attention.__wrapped__ is original
