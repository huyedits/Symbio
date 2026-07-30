import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from symbio_native.model import NativeConfig, NativeLM
from symbio_native.substrate import BranchManager
from symbio_native.tokenizer import BPETrainer


def test_branch_does_not_overwrite_base():
    cfg = NativeConfig(vocab_size=256, dim=32, n_layers=2, n_heads=2, n_kv_heads=1)
    model = NativeLM(cfg)
    trainer = BPETrainer(vocab_size=256)
    tokenizer = trainer.train("hello world\ncorrect answer")

    base_snapshot = {k: v.copy() for k, v in model.parameters().items()}

    with tempfile.TemporaryDirectory() as tmp:
        mgr = BranchManager(model, Path(tmp) / "branches")
        branch = mgr.make_branch(
            "greeting-correction",
            "hello",
            " world",
            tokenizer,
            iters=2,
        )

    # Base weights should be restored in memory after branch creation.
    restored = model.parameters()
    close = tree_map(
        lambda a, b: bool(mx.allclose(a, b).item()), restored, base_snapshot
    )
    assert all(v for _, v in tree_flatten(close))

    # Branch should contain non-zero delta for at least one parameter.
    flat_delta = tree_flatten(branch.weights)
    assert any(float(mx.sum(mx.abs(v)).item()) > 0 for _, v in flat_delta)
