"""symbio/app/ternel_pack.py against a stand-in for the checkpoint's own code.

The real ternel_packed_model.py arrives with the checkpoint (5.9 GB). Its
interface is all ternel_pack touches — PackedLinear(layout, codes, scales) and
the module functions packed_matmul / tq1_get_rows — so a module with that
interface over a plain dense weight checks the maths: the LoRA wrapper, the
decoded rows, and above all the backward, against autodiff of the dense layer.
"""
from __future__ import annotations

import sys
import types

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from symbio.app import ternel_pack  # noqa: E402


_STANDIN = '''
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

INDEX_DTYPE = mx.uint32
GET_ROWS_THREADS = 256


@dataclass(frozen=True)
class Layout:
    rows: int
    columns: int
    groups_per_row: int = 1
    tile: int = 1


def packed_matmul(codes, scales, x, *, layout):
    return x @ codes.astype(x.dtype).T


def tq1_get_rows(codes, scales, indices, *, groups_per_row, tile, dtype, threads, safe_clamp):
    assert indices.dtype == INDEX_DTYPE
    return codes[indices].astype(dtype)


class PackedLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.layout = Layout(*weight.shape)
        self.codes = weight
        self.scales = mx.zeros((1,), dtype=mx.uint16)

    def __call__(self, x):
        return packed_matmul(self.codes, self.scales, x, layout=self.layout)
'''


def _fake_checkpoint_module():
    """The checkpoint module's interface, with codes holding a dense weight —
    exec'd from source into an unregistered module, the way mlx_lm runs a
    checkpoint's model_file."""
    mod = types.ModuleType("custom_model")
    exec(compile(_STANDIN, "ternel_standin.py", "exec", dont_inherit=True), mod.__dict__)
    return mod


class _Model(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.packed_module_counts = {"linear": 2, "embedding": 0}
        self.first = cls(mx.random.normal((12, 8)))
        self.second = cls(mx.random.normal((5, 12)))

    def __call__(self, x):
        return self.second(nn.silu(self.first(x)))


@pytest.fixture
def model():
    # mlx_lm execs a checkpoint's model_file without registering the module,
    # so adapt() must reach it without sys.modules (the first real load died
    # on KeyError: 'custom_model').
    mod = _fake_checkpoint_module()
    assert "custom_model" not in sys.modules
    m = _Model(mod.PackedLinear)
    assert ternel_pack.is_ternel(m) and ternel_pack.adapt(m)
    yield m


def test_a_ternel_model_prefills_in_small_chunks(model):
    assert model.symbio_prefill_step == ternel_pack.PREFILL_STEP


def test_eval_mode_runs_the_shipped_kernel_path(model):
    model.eval()
    x = mx.random.normal((3, 8))
    want = x @ model.first.codes.T
    assert float(mx.abs(model.first(x) - want).max()) < 1e-5


def test_decoded_rows_are_the_weight_rows(model):
    rows = mx.array([4, 0, 11])
    assert float(mx.abs(model.first.dense_rows(rows, mx.float32)
                        - model.first.codes[rows]).max()) == 0.0


def test_the_backward_is_the_dense_layers_backward(model):
    """dx through the packed layers must equal autodiff through dense ones."""
    model.train()
    w1, w2 = model.first.codes, model.second.codes
    x = mx.random.normal((4, 8))

    def dense(x):
        return ((nn.silu(x @ w1.T) @ w2.T) ** 2).sum()

    def packed(x):
        return (model(x) ** 2).sum()

    want = mx.grad(dense)(x)
    got = mx.grad(packed)(x)
    assert float(mx.abs(got - want).max()) < 1e-4


def test_lora_on_a_packed_layer_trains_and_reaches_through_the_frozen_one(model):
    """An adapter on the FIRST layer only learns if the gradient gets back
    through the second, frozen, packed layer — the whole point of the backward."""
    from mlx.utils import tree_flatten

    model.freeze()
    model.first = model.first.to_lora(r=2, scale=10.0)
    model.train()
    x = mx.random.normal((4, 8))
    assert float(mx.abs(model.first(x) - model.first.linear(x)).max()) == 0.0  # starts as base

    loss = nn.value_and_grad(model, lambda m, x: (m(x) ** 2).sum())
    _, grads = loss(model, x)
    grads = dict(tree_flatten(grads))
    assert set(grads) == {"first.lora_a", "first.lora_b"}
    assert float(mx.abs(grads["first.lora_b"]).sum()) > 0


def test_a_model_without_packed_layers_is_left_alone():
    plain = nn.Linear(4, 4)
    assert not ternel_pack.adapt(plain)
