"""A diverged adapter is refused at load, and the base model answers instead.

Seen 2026-09-26: a 14B home whose adapter came from a run that went to nan
answered "hey, are you awake?" with a row of exclamation marks — every logit
nan, so every token was id 0. Every tensor in that adapter was nan.
"""
import pytest

mx = pytest.importorskip("mlx.core")

from symbio.app import modelload  # noqa: E402


def _adapter(tmp_path, poison: bool):
    tensors = {"layers.0.lora_a": mx.ones((4, 8)), "layers.0.lora_b": mx.zeros((8, 4))}
    if poison:
        tensors["layers.0.lora_b"] = mx.full((8, 4), float("nan"))
    mx.save_safetensors(str(tmp_path / "adapters.safetensors"), tensors)
    return tmp_path


def test_a_nan_adapter_is_named(tmp_path):
    assert modelload.nonfinite_adapter_tensors(_adapter(tmp_path, poison=True)) == ["layers.0.lora_b"]


def test_a_sound_adapter_passes(tmp_path):
    assert modelload.nonfinite_adapter_tensors(_adapter(tmp_path, poison=False)) == []


def test_no_adapter_is_not_a_problem(tmp_path):
    assert modelload.nonfinite_adapter_tensors(None) == []
    assert modelload.nonfinite_adapter_tensors(tmp_path) == []


def test_the_loader_refuses_it_before_the_weights_load(tmp_path, monkeypatch):
    """Raised, not skipped silently: the callers (daemon, chat, workers)
    already catch a failed adapter load, print why, and load the base."""
    import symbio.mlx_gate as gate

    loaded = []
    monkeypatch.setattr(gate, "attr", lambda _name: (lambda *a, **k: loaded.append(k) or (None, None)))
    with pytest.raises(ValueError, match="NaN/inf"):
        modelload._load_with_backend(("some/model",), {"adapter_path": str(_adapter(tmp_path, True))},
                                     {"backend": "mlx"})
    assert loaded == []
