"""Create and archive substrate branches without overwriting base weights."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from ..model import NativeLM
from ..utils.safetensors import load_weights, save_weights
from .branch import Branch


def _lora_forward(model: nn.Module, x: mx.array) -> mx.array:
    """Dummy placeholder; real LoRA injection happens on the linear layers."""
    return model(x)


class BranchManager:
    """Owns the lifecycle of correction branches.

    When the model fails on a specific sample, `make_branch` freezes the base
    weights and trains a small delta (full diff or low-rank) on that example.
    The resulting branch is saved under `branches_dir` and can be selected by
    the router when its trigger tokens appear.
    """

    def __init__(self, base_model: NativeLM, branches_dir: str | Path):
        self.base_model = base_model
        self.branches_dir = Path(branches_dir)
        self.branches_dir.mkdir(parents=True, exist_ok=True)
        self.branches: list[Branch] = []
        self._load_existing()

    def _load_existing(self) -> None:
        for path in sorted(self.branches_dir.iterdir()):
            if path.is_dir() and (path / "meta.json").exists():
                self.branches.append(Branch.load(path))

    def make_branch(
        self,
        name: str,
        prompt_text: str,
        target_text: str,
        tokenizer,
        *,
        rank: int = 0,
        iters: int = 50,
        learning_rate: float = 1e-3,
    ) -> Branch:
        """Train a correction branch for a single (prompt, target) failure.

        The base weights stay frozen; only branch parameters are updated.
        """
        prompt_ids = tokenizer.encode(prompt_text)
        target_ids = tokenizer.encode(target_text)
        full_ids = prompt_ids + target_ids
        x = mx.array([full_ids[:-1]], dtype=mx.int32)
        y = mx.array([full_ids[1:]], dtype=mx.int32)

        # Snapshot current base parameters; deep-copy arrays for the branch copy
        # so the live base weights never move.
        base_weights = self.base_model.parameters()
        branch_weights = tree_map(lambda v: mx.array(v, dtype=v.dtype), base_weights)

        # Train a separate copy so the live base weights never move.
        branch_model = NativeLM(self.base_model.config)
        branch_model.update(branch_weights)
        optimizer = optim.Adam(learning_rate=learning_rate)

        def loss_fn(model: NativeLM, x: mx.array, y: mx.array) -> mx.array:
            _, loss = model(x, y)
            return loss

        for _ in range(iters):
            loss, grads = nn.value_and_grad(branch_model, loss_fn)(branch_model, x, y)
            optimizer.update(branch_model, grads)
            mx.eval(loss, branch_model.parameters())

        # Compute delta = trained branch copy - base.
        delta = tree_map(lambda n, b: n - b, branch_model.parameters(), base_weights)

        # Restore base weights in memory immediately (they were never modified).
        self.base_model.update(base_weights)

        trigger_ids = tokenizer.encode(name)[:8] or target_ids[:4]
        branch = Branch(
            name=name,
            trigger_ids=trigger_ids,
            weights=delta,
            description=f"Corrects: {name}",
            rank=rank,
        )
        branch_dir = self.branches_dir / name
        branch.save(branch_dir)
        self.branches.append(branch)
        return branch

    def apply_branch(self, branch: Branch) -> None:
        """Add a branch's deltas to the running model in memory."""
        params = self.base_model.parameters()
        for key, delta in branch.weights.items():
            params[key] = params[key] + delta
        mx.eval(params)

    def reset_to_base(self) -> None:
        """Reload base weights from disk. Not implemented: caller should recreate."""
        raise NotImplementedError(
            "reset_to_base: recreate the BranchManager from a fresh model instance."
        )
