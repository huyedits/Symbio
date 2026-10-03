"""The main model's own eyes (symbio/app/eyes.py) and how vision picks them.

A look on a Qwen3.5 main model builds mlx-vlm's model around the resident
mlx_lm one. Everything here is checked on tiny models built in memory: the
graft must share every language parameter (a name that matched nothing would
leave random weights in a model that still runs), and the two libraries must
compute the same logits from them.
"""
from __future__ import annotations

import gc
import json
import types

import pytest

from symbio import vision
from symbio.app import eyes

TEXT = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, linear_num_value_heads=2, linear_num_key_heads=1,
            linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            vocab_size=128, rms_norm_eps=1e-6, max_position_embeddings=512,
            full_attention_interval=4, tie_word_embeddings=False,
            rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                             "rope_theta": 10000, "partial_rotary_factor": 0.25})
VISION = dict(model_type="qwen3_5", depth=1, hidden_size=32, intermediate_size=64,
              num_heads=2, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
              temporal_patch_size=2, in_channels=3, num_position_embeddings=64,
              deepstack_visual_indexes=[])


class _Model:
    """Stands in for a loaded model: something a weak reference can point at."""


def _checkpoint(tmp_path, model_type="qwen3_5", vision_config=True, tower=True):
    config = {"model_type": model_type, "text_config": TEXT}
    if vision_config:
        config["vision_config"] = VISION
    (tmp_path / "config.json").write_text(json.dumps(config))
    names = {"language_model.lm_head.weight": "model.safetensors"}
    if tower:
        names["vision_tower.blocks.0.attn.qkv.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    return tmp_path


# ---- which checkpoints have eyes -----------------------------------------

def test_a_vision_checkpoint_with_its_tower_has_eyes(tmp_path):
    assert eyes.has_vision_tower(_checkpoint(tmp_path))


def test_a_text_only_conversion_keeps_vision_config_but_not_the_eyes(tmp_path):
    """mlx-community's text-only Qwen3.5 copies keep vision_config and drop the
    tensors; the config alone must not be taken as a tower."""
    assert not eyes.has_vision_tower(_checkpoint(tmp_path, tower=False))


def test_an_unchecked_model_type_is_not_trusted_with_a_graft(tmp_path):
    assert not eyes.has_vision_tower(_checkpoint(tmp_path, model_type="qwen3_vl"))


def test_the_tower_is_found_in_a_bare_safetensors_header(tmp_path):
    mx = pytest.importorskip("mlx.core")
    config = {"model_type": "qwen3_5", "vision_config": VISION}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"),
                        {"vision_tower.patch_embed.proj.weight": mx.zeros((2, 2))})
    assert eyes.has_vision_tower(tmp_path)


def test_a_resident_model_is_found_by_the_name_it_was_loaded_under(tmp_path):
    path = _checkpoint(tmp_path)
    model = _Model()
    assert eyes.remember(model, str(path))
    found = eyes.resident(str(path))
    assert found is not None and found[0] is model
    assert eyes.resident("some/other-model") is None


def test_eyes_do_not_keep_an_unloaded_model_alive(tmp_path):
    """Deep sleep frees the main model by dropping it; a strong reference here
    would be the double residency that has frozen this Mac before."""
    path = _checkpoint(tmp_path)
    model = _Model()
    eyes.remember(model, str(path))
    del model
    gc.collect()
    assert eyes.resident(str(path)) is None


def test_a_model_without_a_tower_is_not_remembered(tmp_path):
    assert not eyes.remember(_Model(), str(_checkpoint(tmp_path, tower=False)))


# ---- the graft --------------------------------------------------------------

def _models(tie=False):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from mlx_lm.models import qwen3_5 as lm35
    from mlx_vlm.models import qwen3_5 as vlm35

    text = {**TEXT, "tie_word_embeddings": tie}
    host = lm35.Model(lm35.ModelArgs(model_type="qwen3_5", text_config=dict(text)))
    mx.eval(host.parameters())
    config = vlm35.ModelConfig.from_dict({"model_type": "qwen3_5", "text_config": text,
                                          "vision_config": VISION})
    config.text_config = vlm35.TextConfig.from_dict(text)
    config.vision_config = vlm35.VisionConfig.from_dict(VISION)
    return mx, host, vlm35.Model(config)


def _logits(mx, model, ids):
    out = model(ids)
    return getattr(out, "logits", out)


@pytest.mark.parametrize("tie", [False, True])
def test_the_graft_shares_every_parameter_and_computes_the_same_logits(tie):
    from mlx.utils import tree_flatten

    mx, host, vl = _models(tie)
    eyes.graft(host.language_model, vl.language_model)

    host_ids = {id(v) for _, v in tree_flatten(host.parameters())}
    assert all(id(v) in host_ids for _, v in tree_flatten(vl.language_model.parameters()))
    ids = mx.array([[1, 5, 9, 33, 17, 64, 2, 99]])
    a, b = _logits(mx, host, ids), _logits(mx, vl.language_model, ids)
    assert float(mx.abs(a - b).max()) < 1e-4


def test_an_adapter_the_main_model_wears_is_worn_by_its_eyes():
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mx, host, vl = _models()
    host.freeze()
    linear_to_lora_layers(host, 1, {"rank": 2, "scale": 10.0, "dropout": 0.0})
    eyes.graft(host.language_model, vl.language_model)

    last = "model.layers.3.self_attn.q_proj"
    assert eyes._module_at(vl.language_model, last)[0]["q_proj"] is \
        eyes._module_at(host.language_model, last)[0]["q_proj"]


def test_a_parameter_the_main_model_lacks_refuses_the_look():
    """Loaded from nowhere, it would be random — and the look would still run."""
    import mlx.nn as nn

    mx, host, vl = _models()
    vl.language_model.model.stray = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="never loaded"):
        eyes.graft(host.language_model, vl.language_model)


def test_the_look_projects_only_the_last_position():
    mx, host, _vl = _models()
    head = eyes._last_position_head(host.language_model.lm_head)
    x = mx.random.normal((1, 7, TEXT["hidden_size"]))
    full = host.language_model.lm_head(x)
    last = head(x)
    assert last.shape == (1, 1, TEXT["vocab_size"])
    assert float(mx.abs(last[:, -1] - full[:, -1]).max()) < 1e-5


def test_fused_decode_never_runs_on_a_shared_or_custom_layer(monkeypatch):
    """mlx-vlm's fused decode caches a concatenated copy ON the first layer and
    calls quantized_matmul with that layer's bits — a second copy of the main
    model's weights that outlives the look, and for 1-bit layers a kernel MLX
    does not have."""
    pytest.importorskip("mlx_vlm")
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import language

    calls = []
    monkeypatch.setattr(language, "_decode_quantized_linears_fused",
                        lambda linears, x: calls.append(linears) or "fused")
    eyes._guard_fused_decode()
    guarded = language._decode_quantized_linears_fused

    plain = nn.QuantizedLinear(64, 64, bias=False)
    assert guarded([plain] * 4, None) == "fused"

    class OneBit(nn.QuantizedLinear):
        pass

    custom = OneBit(64, 64, bias=False)
    assert guarded([plain, custom, plain, plain], None) is None
    shared = nn.QuantizedLinear(64, 64, bias=False)
    shared.shared_with_eyes = True
    assert guarded([shared, plain, plain, plain], None) is None
    assert len(calls) == 1


def test_screens_are_capped_for_the_main_model():
    processor = types.SimpleNamespace(
        image_processor=types.SimpleNamespace(max_pixels=16_777_216))
    eyes._cap_pixels(processor, eyes.MAX_PIXELS)
    assert processor.image_processor.max_pixels == eyes.MAX_PIXELS
    dict_sized = types.SimpleNamespace(
        image_processor=types.SimpleNamespace(size={"longest_edge": 16_777_216}))
    eyes._cap_pixels(dict_sized, eyes.MAX_PIXELS)
    assert dict_sized.image_processor.size["longest_edge"] == eyes.MAX_PIXELS


# ---- how vision chooses ------------------------------------------------------

def _with_eyes(monkeypatch, main):
    stub = types.SimpleNamespace(resident=lambda name: (object(), "/p") if name == main else None)
    monkeypatch.setattr(vision, "_eyes", lambda: stub)


def test_a_main_model_with_eyes_looks_for_itself(monkeypatch):
    _with_eyes(monkeypatch, "prism-ml/Bonsai-27B-mlx-1bit")
    config = {"model_name": "prism-ml/Bonsai-27B-mlx-1bit", "vision": {}}
    assert vision.model_name(config) == "headmaster:prism-ml/Bonsai-27B-mlx-1bit"
    assert vision.uses_headmaster(config)


def test_naming_the_main_models_own_repo_still_shares_it(monkeypatch):
    """Loading it a second time through mlx-vlm would be two copies of it."""
    _with_eyes(monkeypatch, "m")
    config = {"model_name": "m", "vision": {"model_name": "m"}}
    assert vision.uses_headmaster(config)


def test_an_explicit_vision_model_is_kept(monkeypatch):
    _with_eyes(monkeypatch, "m")
    config = {"model_name": "m", "vision": {"model_name": "some/vlm"}}
    assert vision.model_name(config) == "some/vlm"
    assert not vision.uses_headmaster(config)


def test_a_main_model_without_eyes_keeps_the_vision_worker(monkeypatch):
    _with_eyes(monkeypatch, "other")
    config = {"model_name": "m", "vision": {}}
    assert vision.model_name(config) == vision.DEFAULT_VISION_MODEL


def _session(sleeps):
    from symbio.app import chat_tools
    from symbio.app.config import DEFAULT_CONFIG

    class S(chat_tools.ToolsMixin):
        config = {**DEFAULT_CONFIG, "dispatch": {}}
        model = object()

        def _status(self, m):
            pass

        def _sleep_headmaster(self):
            sleeps.append("sleep")

        def _wake_headmaster(self):
            sleeps.append("wake")

    return S()


@pytest.mark.parametrize("own_eyes", [True, False])
def test_the_model_doing_the_looking_is_not_put_to_sleep(monkeypatch, own_eyes):
    monkeypatch.setattr(vision, "describe", lambda *a, **k: "a screen")
    monkeypatch.setattr(vision, "on_screen", lambda *a, **k: True)
    monkeypatch.setattr(vision, "locate", lambda *a, **k: [])
    monkeypatch.setattr(vision, "release", lambda: False)
    monkeypatch.setattr(vision, "uses_headmaster", lambda config=None: own_eyes)
    sleeps: list[str] = []
    _session(sleeps)._run_vision("shot.png", "the Post button")
    assert sleeps == ([] if own_eyes else ["sleep", "wake"])
