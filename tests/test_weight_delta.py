"""scripts/weight_delta.py: the change a LoRA adapter makes, read off its file."""

import json

import pytest
pytest.importorskip("mlx")  # MLX is Apple Silicon only; skip elsewhere
import mlx.core as mx

import weight_delta


def _adapter(tmp_path, name, a, b, scale=20.0):
    d = tmp_path / name
    d.mkdir()
    mx.save_safetensors(str(d / "adapters.safetensors"), {
        "model.layers.3.self_attn.q_proj.lora_a": mx.array(a, dtype=mx.float32),
        "model.layers.3.self_attn.q_proj.lora_b": mx.array(b, dtype=mx.float32),
    })
    (d / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"scale": scale}}))
    return d


def test_the_delta_is_scale_times_a_at_b(tmp_path):
    d = _adapter(tmp_path, "one", [[1.0], [0.0]], [[2.0, 0.0, 0.0]], scale=10.0)

    (row,) = weight_delta.report(d)

    # ΔW = 10 * [[1],[0]] @ [[2,0,0]] has a single entry, 20.
    assert row["module"] == "model.layers.3.self_attn.q_proj"
    assert row["delta"] == pytest.approx(20.0)


def test_a_zero_lora_b_is_no_change(tmp_path):
    """Fresh LoRA initialises lora_b to zero: the adapter exists and changes nothing."""
    d = _adapter(tmp_path, "fresh", [[1.0], [1.0]], [[0.0, 0.0]])

    assert weight_delta.report(d)[0]["delta"] == 0.0


def test_moved_measures_the_latest_training_only(tmp_path):
    old = _adapter(tmp_path, "old", [[1.0]], [[1.0]], scale=1.0)
    new = _adapter(tmp_path, "new", [[1.0]], [[3.0]], scale=1.0)

    (row,) = weight_delta.report(new, since=old)

    assert row["delta"] == pytest.approx(3.0)
    assert row["moved"] == pytest.approx(2.0)
