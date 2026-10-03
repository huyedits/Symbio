"""symbio.trim_lora: the LoRA trainer for 248k-vocabulary, linear-attention models.

Two memory cuts, each checked against the thing it replaces: the trimmed
logits must BE the model's own logits for those rows, and the checkpointed
gated-delta loop must give the same outputs and the same gradients as the
loop it wraps. The rest is which models are sent here at all.
"""
from __future__ import annotations

import json
import sys

import pytest

from symbio import backend, trim_lora

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


# ---- which trainer ------------------------------------------------------------

def _config(tmp_path, **fields):
    (tmp_path / "config.json").write_text(json.dumps(fields))
    return str(tmp_path)


def test_a_248k_vocabulary_trains_through_the_lean_trainer(tmp_path):
    assert trim_lora.wants_lean_trainer(
        _config(tmp_path, model_type="qwen3_5", text_config={"vocab_size": 248320}), {})


def test_a_hybrid_model_goes_there_even_with_a_small_vocabulary(tmp_path):
    """The gated-delta backward is the other half of the 16 GB wall."""
    assert trim_lora.wants_lean_trainer(
        _config(tmp_path, model_type="qwen3_next", vocab_size=151936), {})


def test_the_qwen3_headmaster_keeps_the_command_that_trained_it(tmp_path):
    assert not trim_lora.wants_lean_trainer(
        _config(tmp_path, model_type="qwen3", vocab_size=151936), {})


def test_trim_on_sends_any_model(tmp_path):
    assert trim_lora.wants_lean_trainer(
        _config(tmp_path, model_type="qwen3", vocab_size=151936), {"trim_vocab": "on"})


def test_an_unknown_model_keeps_mlx_lm():
    assert not trim_lora.wants_lean_trainer("no/such-model-anywhere", {})


def test_the_backend_runs_the_lean_trainer_with_mlx_lms_flags(tmp_path):
    model = _config(tmp_path, model_type="qwen3_5", text_config={"vocab_size": 248320})
    lora = {"batch_size": 1, "num_layers": 2, "learning_rate": 1e-4,
            "steps_per_eval": 30, "val_batches": 8, "max_seq_length": 2048,
            "save_every": 15, "trim_vocab": "auto"}
    cmd = backend.trainer_command(model, "/data", "/adapters", lora, 150, "/cfg.yaml",
                                  {"backend": "mlx"})
    assert cmd[:5] == [sys.executable, "-m", "symbio.trim_lora", trim_lora.TRIM_FLAG, "auto"]
    for flag, value in [("--model", model), ("--iters", "150"), ("--config", "/cfg.yaml"),
                        ("--adapter-path", "/adapters"), ("--max-seq-length", "2048")]:
        assert cmd[cmd.index(flag) + 1] == value, flag


def test_the_trim_flag_is_taken_out_before_mlx_lm_sees_it():
    argv = ["--model", "m", trim_lora.TRIM_FLAG, "off", "--train"]
    assert trim_lora._pop_flag(argv, trim_lora.TRIM_FLAG, "auto") == "off"
    assert argv == ["--model", "m", "--train"]
    assert trim_lora._pop_flag(argv, trim_lora.TRIM_FLAG, "auto") == "auto"


# ---- the trimmed head --------------------------------------------------------

ROWS = [0, 3, 7, 21, 40, 49]


@pytest.mark.parametrize("kind", ["linear", "quantized", "embedding", "quantized_embedding"])
def test_a_cut_head_gives_exactly_those_rows(kind):
    linear = nn.Linear(128, 50, bias=False)
    embedding = nn.Embedding(50, 128)
    head = {"linear": linear,
            "quantized": nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4),
            "embedding": embedding,
            "quantized_embedding": nn.QuantizedEmbedding.from_embedding(
                embedding, group_size=64, bits=4)}[kind]
    tied = "embedding" in kind
    x = mx.random.normal((2, 3, 128))
    full = head.as_linear(x) if tied else head(x)
    cut = trim_lora.subset_projection(head, mx.array(ROWS, dtype=mx.uint32), tied)(x)
    assert cut.shape == (2, 3, len(ROWS))
    assert float(mx.abs(cut - full[..., ROWS]).max()) < 1e-4


def _tiny_qwen35(tie=False):
    from mlx_lm.models import qwen3_5

    text = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
                num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                head_dim=32, linear_num_value_heads=2, linear_num_key_heads=1,
                linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
                vocab_size=96, rms_norm_eps=1e-6, max_position_embeddings=512,
                full_attention_interval=4, tie_word_embeddings=tie)
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text))
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("tie", [False, True])
def test_the_trimmed_loss_is_cross_entropy_over_the_models_own_logits(tie):
    model = _tiny_qwen35(tie)
    sample = [1, 4, 9, 4, 30, 2, 9, 61, 30, 1]
    rows = sorted(set(sample) | {0})
    loss = trim_lora.build_trimmed_loss(model, rows, sample)
    assert loss is not None

    batch = mx.array([sample])
    lengths = mx.array([[2, len(sample)]])  # loss from the 2nd target on
    got, ntoks = loss(model, batch, lengths)

    logits = model(batch[:, :-1])[..., mx.array(rows)]
    remap = {t: i for i, t in enumerate(rows)}
    targets = mx.array([[remap[t] for t in sample[1:]]])
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = (steps >= 2) & (steps <= len(sample))
    want = (nn.losses.cross_entropy(logits, targets) * mask).sum() / mask.sum()
    assert int(ntoks) == int(mask.sum())
    assert abs(float(got) - float(want)) < 1e-4


def test_a_head_it_cannot_reproduce_trains_on_the_full_vocabulary(monkeypatch):
    """A logit scale the cut head does not apply would train a different
    model than the one that serves; the check refuses it."""
    model = _tiny_qwen35()
    real = type(model).__call__
    monkeypatch.setattr(type(model), "__call__",
                        lambda self, *a, **k: real(self, *a, **k) * 3.0 + 1.0)
    assert trim_lora.build_trimmed_loss(model, [0, 1, 2, 3], [1, 2, 3, 1]) is None


# ---- the gated-delta backward -----------------------------------------------

def test_the_checkpointed_loop_has_the_same_values_and_gradients(monkeypatch):
    from mlx_lm.models import gated_delta

    original = gated_delta.gated_delta_ops
    monkeypatch.setattr(gated_delta, "gated_delta_ops", original)
    assert trim_lora.checkpoint_gated_delta(chunk=16)
    chunked = gated_delta.gated_delta_ops
    assert chunked is not original

    B, T, Hk, Dk, Hv, Dv = 1, 70, 1, 16, 2, 16
    q, k = (mx.random.normal((B, T, Hk, Dk)) * 0.3 for _ in range(2))
    v = mx.random.normal((B, T, Hv, Dv)) * 0.3
    g = mx.sigmoid(mx.random.normal((B, T, Hv)))
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))

    def total(fn):
        return lambda q, k, v: (fn(q, k, v, g, beta)[0] ** 2).sum()

    for fn_a, fn_b in ((original, chunked),):
        ya, sa = fn_a(q, k, v, g, beta)
        yb, sb = fn_b(q, k, v, g, beta)
        assert float(mx.abs(ya - yb).max()) < 1e-5
        assert float(mx.abs(sa - sb).max()) < 1e-5
        ga = mx.grad(total(fn_a), argnums=(0, 1, 2))(q, k, v)
        gb = mx.grad(total(fn_b), argnums=(0, 1, 2))(q, k, v)
        for a, b in zip(ga, gb):
            assert float(mx.abs(a - b).max()) < 1e-4


def test_installing_twice_does_not_nest_the_checkpoint(monkeypatch):
    from mlx_lm.models import gated_delta

    monkeypatch.setattr(gated_delta, "gated_delta_ops", gated_delta.gated_delta_ops)
    assert trim_lora.checkpoint_gated_delta()
    assert not trim_lora.checkpoint_gated_delta()


def test_layers_below_the_adapters_keep_the_inference_kernel():
    """No gradient reaches them, so the training loop there is only cost."""
    class GDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = mx.zeros((2,))
            self.dt_bias = mx.zeros((2,))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = GDN()
            self.proj = nn.Linear(2, 2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer() for _ in range(4)]

    model = Model()
    model.freeze()
    model.layers[2].proj.unfreeze()  # the first adapter sits in layer 2
    assert trim_lora.kernel_below_adapters(model) == 2
    model.train()
    assert [layer.linear_attn.training for layer in model.layers] == [False, False, True, True]
