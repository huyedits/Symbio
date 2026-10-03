"""Prism Hadamard packs (Ternary-Bonsai-2, Qwen3.8-27B) — symbio/app/prism_pack.py.

The weights are stored rotated; the loader has to put the rotation back on
every projection and the inverse on every embedding row. The layer's maths is
checked against an explicit dequantize-and-Hadamard reference; then a tiny
pack in the real on-disk layout is loaded and compared with a dense model
holding the same effective weights, which catches a module the loader left
unrotated, unreplaced or unloaded.
"""
from __future__ import annotations

import json
import math

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from symbio.app import prism_pack  # noqa: E402

BLOCK = 512
TEXT = dict(model_type="qwen3_5_text", hidden_size=512, intermediate_size=1024,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=256, linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=128, linear_value_head_dim=128,
            linear_conv_kernel_dim=4, vocab_size=96, rms_norm_eps=1e-6,
            max_position_embeddings=512, full_attention_interval=4,
            tie_word_embeddings=False)
PACKED = ("lm_head", "model.embed_tokens", "self_attn.q_proj", "self_attn.k_proj",
          "self_attn.v_proj", "self_attn.o_proj", "linear_attn.in_proj_qkv",
          "linear_attn.in_proj_z", "linear_attn.out_proj", "mlp.gate_proj",
          "mlp.up_proj", "mlp.down_proj")


def _signs(n):
    return mx.where(mx.random.uniform(shape=(n,)) > 0.5, 1.0, -1.0)


def _packed_arrays(rows, width):
    weight = mx.random.randint(0, 2**31, (rows, width // 16)).astype(mx.uint32)
    scales = (mx.random.uniform(shape=(rows, width // 128)) * 0.02 + 0.005).astype(mx.float16)
    biases = (-1.5 * scales).astype(mx.float16)
    return weight, scales, biases


# ---- the layer ----------------------------------------------------------------

def test_the_rotation_undoes_itself():
    layer = prism_pack.Packed(*_packed_arrays(8, BLOCK), block=BLOCK, signs=_signs(BLOCK))
    x = mx.random.normal((3, BLOCK)).astype(mx.float32)
    back = layer.rotate(layer.rotate(x), inverse=True)
    assert float(mx.abs(back - x).max()) < 1e-4


def test_a_packed_projection_is_the_folded_weight_on_the_rotated_input():
    w, s, b = _packed_arrays(64, 1024)
    signs = _signs(1024)
    layer = prism_pack.Packed(w, s, b, block=BLOCK, signs=signs)
    x = mx.random.normal((2, 5, 1024)).astype(mx.float16)
    dense = mx.dequantize(w, s, b, group_size=128, bits=2).astype(mx.float32)
    rotated = mx.hadamard_transform((x.astype(mx.float32) * signs).reshape(-1, BLOCK),
                                    scale=1 / math.sqrt(BLOCK)).reshape(x.shape)
    want = rotated @ dense.T
    assert float(mx.abs(layer(x).astype(mx.float32) - want).max()) < 2e-2 * float(
        mx.abs(want).max())


def test_tied_logits_agree_with_the_embedding_rows():
    """x . row_i must equal as_linear(x)[i] — the identity the tied path uses."""
    layer = prism_pack.Packed(*_packed_arrays(40, BLOCK), block=BLOCK,
                              signs=_signs(BLOCK), embedding=True)
    rows = layer(mx.arange(40)).astype(mx.float32)
    x = mx.random.normal((3, BLOCK)).astype(mx.float16)
    want = x.astype(mx.float32) @ rows.T
    got = layer.as_linear(x).astype(mx.float32)
    assert float(mx.abs(got - want).max()) < 2e-2 * float(mx.abs(want).max())


def test_a_head_cut_to_some_rows_gives_exactly_those_rows():
    layer = prism_pack.Packed(*_packed_arrays(50, BLOCK), block=BLOCK, signs=_signs(BLOCK))
    x = mx.random.normal((1, 4, BLOCK)).astype(mx.float16)
    rows = mx.array([1, 7, 30, 49], dtype=mx.uint32)
    assert float(mx.abs(layer.subset(rows)(x) - layer(x)[..., rows]).max()) == 0.0


def test_lora_on_a_packed_layer_starts_as_the_layer_and_trains():
    base = prism_pack.Packed(*_packed_arrays(32, BLOCK), block=BLOCK, signs=_signs(BLOCK))
    base.freeze()
    lora = base.to_lora(r=4, scale=10.0)
    x = mx.random.normal((2, BLOCK)).astype(mx.float16)
    assert float(mx.abs(lora(x) - base(x)).max()) == 0.0

    from mlx.utils import tree_flatten

    loss = nn.value_and_grad(lora, lambda m, x: (m(x).astype(mx.float32) ** 2).sum())
    _, grads = loss(lora, x)
    grads = dict(tree_flatten(grads))
    assert set(grads) == {"lora_a", "lora_b"}  # the packed base stays frozen
    assert float(mx.abs(grads["lora_b"]).sum()) > 0


def test_an_embedding_takes_no_adapter():
    emb = prism_pack.Packed(*_packed_arrays(8, BLOCK), block=BLOCK, signs=_signs(BLOCK),
                            embedding=True)
    with pytest.raises(ValueError, match="embed_tokens"):
        emb.to_lora()


# ---- a pack on disk ---------------------------------------------------------

def _paths(num_layers):
    paths = ["lm_head", "model.embed_tokens"]
    for i in range(num_layers):
        linear = (i + 1) % TEXT["full_attention_interval"] != 0
        for name in PACKED[2:]:
            if name.startswith("linear_attn") == linear or name.startswith("mlp"):
                if name.startswith(("self_attn", "linear_attn", "mlp")):
                    paths.append(f"model.layers.{i}.{name}")
    return paths


def _write_pack(tmp_path, vision=None, schema=2, **overrides):
    """A pack in Prism's layout, plus the dense model holding the same weights."""
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5

    dense = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=dict(TEXT)))
    mx.eval(dense.parameters())
    lm = dense.language_model
    tensors, records = {}, []
    for path in _paths(TEXT["num_hidden_layers"]):
        parent, leaf = prism_pack._module_at(lm, path)
        original = getattr(parent, leaf)
        rows, width = original.weight.shape
        embedding = isinstance(original, nn.Embedding)
        arrays = _packed_arrays(rows, width)
        signs = _signs(width)
        packed = prism_pack.Packed(*arrays, block=BLOCK, signs=signs, embedding=embedding)
        # The dense model gets the effective weight the pack computes with.
        effective = packed(mx.arange(rows)) if embedding else packed(
            mx.eye(width).astype(mx.float16)).T
        original.weight = effective.astype(mx.float32)
        key = "language_model." + path
        tensors.update({f"{key}.weight": arrays[0], f"{key}.scales": arrays[1],
                        f"{key}.biases": arrays[2], f"{key}.signs": signs})
        records.append({"path": path, "block": BLOCK, "embedding": embedding,
                        "dtype": "float16"})
    packed_keys = {k.rsplit(".", 1)[0] for k in tensors}
    for name, value in tree_flatten(lm.parameters()):
        if name.rsplit(".", 1)[0] not in {p for p in _paths(TEXT["num_hidden_layers"])}:
            tensors["language_model." + name] = value
    assert packed_keys  # sanity
    config = {"schema_version": schema, "model_type": prism_pack.MODEL_TYPE,
              "base_model_type": "qwen3_5", "gdn_activation_layout": "grouped",
              "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
              "text_config": TEXT, "modules": records,
              "components": {"text": True, "vision": bool(vision)}}
    if vision:
        config["vision_config"] = vision[0]
        tensors.update({"vision_tower." + k: v for k, v in vision[1].items()})
    config.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [95, 94]}))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors,
                        metadata={"format": "mlx"})
    return dense


def test_a_pack_loads_as_the_dense_model_it_encodes(tmp_path):
    dense = _write_pack(tmp_path)
    model, config = prism_pack.load_pack(tmp_path)
    assert sum(isinstance(m, prism_pack.Packed) for _, m in model.named_modules()) == \
        len(config["modules"])
    ids = mx.array([[1, 5, 9, 33, 17, 64, 2, 80]])
    want, got = dense(ids), model(ids)
    assert float(mx.abs(got - want).max()) < 3e-2 * float(mx.abs(want).max())
    assert config["eos_token_id"] == [95, 94]


def test_mlx_lm_load_model_reaches_the_pack_loader(tmp_path):
    from mlx_lm import utils

    _write_pack(tmp_path)
    prism_pack.install()
    model, _ = utils.load_model(tmp_path)
    assert model.prism_pack_path == str(tmp_path)


@pytest.mark.parametrize("change, message", [
    ({"schema_version": 3}, "schema"),
    ({"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}, "quantization"),
    ({"gdn_activation_layout": "interleaved"}, "layout"),
    ({"base_model_type": "gemma4"}, "qwen3_5"),
])
def test_a_pack_this_cannot_compute_is_refused_not_loaded(tmp_path, change, message):
    _write_pack(tmp_path, **change)
    with pytest.raises(ValueError, match=message):
        prism_pack.load_pack(tmp_path)


def test_a_packed_tensor_of_the_wrong_shape_is_refused(tmp_path):
    _write_pack(tmp_path)
    weights = mx.load(str(tmp_path / "model.safetensors"))
    weights["language_model.lm_head.scales"] = weights["language_model.lm_head.scales"][:, :1]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError, match="packed shapes"):
        prism_pack.load_pack(tmp_path)


# ---- its eyes ----------------------------------------------------------------

VISION = dict(model_type="qwen3_5", depth=1, hidden_size=32, intermediate_size=64,
              num_heads=2, out_hidden_size=512, patch_size=16, spatial_merge_size=2,
              temporal_patch_size=2, in_channels=3, num_position_embeddings=64,
              deepstack_visual_indexes=[])


def test_a_look_shares_the_resident_pack_and_reads_only_the_tower(tmp_path):
    pytest.importorskip("mlx_vlm")
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_5 import VisionConfig, VisionModel

    tower = VisionModel(VisionConfig.from_dict(VISION))
    mx.eval(tower.parameters())
    _write_pack(tmp_path, vision=(VISION, dict(tree_flatten(tower.parameters()))))
    # The processor needs a tokenizer; the graft is what is under test here.
    prism_pack_build = prism_pack.build_processor
    prism_pack.build_processor = lambda path: None
    try:
        host, _ = prism_pack.load_pack(tmp_path)
        vl, _ = prism_pack.load_vl_pack(tmp_path, host=host)
    finally:
        prism_pack.build_processor = prism_pack_build

    host_ids = {id(v) for _, v in tree_flatten(host.parameters())}
    assert all(id(v) in host_ids for _, v in tree_flatten(vl.language_model.parameters()))
    ids = mx.array([[1, 5, 9, 33, 17, 64, 2, 80]])
    out = vl.language_model(ids)
    got = getattr(out, "logits", out)
    want = host(ids)
    assert float(mx.abs(got - want).max()) < 1e-3 * float(mx.abs(want).max()) + 1e-4
    assert vl.config.model_type == "qwen3_5"
    assert vl.config.eos_token_id == [95, 94]
