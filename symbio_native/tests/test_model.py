import mlx.core as mx

from symbio_native.model import NativeConfig, NativeLM


def test_native_lm_shape():
    cfg = NativeConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, n_kv_heads=1)
    model = NativeLM(cfg)
    x = mx.zeros((2, 10), dtype=mx.int32)
    logits, loss = model(x)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert loss is None


def test_native_lm_loss():
    cfg = NativeConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, n_kv_heads=1)
    model = NativeLM(cfg)
    x = mx.zeros((2, 10), dtype=mx.int32)
    y = mx.ones((2, 10), dtype=mx.int32)
    logits, loss = model(x, y)
    assert loss is not None
    assert float(loss.item()) > 0


def test_generate_runs():
    cfg = NativeConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, n_kv_heads=1)
    model = NativeLM(cfg)
    out = model.generate(mx.array([[1, 2, 3]], dtype=mx.int32), max_new=5)
    assert out.shape == (1, 8)
