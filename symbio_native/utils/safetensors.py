"""Flatten/unflatten helpers around MLX safetensors for nested weights."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def save_weights(path: str | Path, weights: dict[str, mx.array]) -> None:
    """Save a possibly-nested weight dict as a flat safetensors file."""
    flat = {".".join(k): v for k, v in tree_flatten(weights)}
    mx.save_safetensors(str(path), flat)


def load_weights(path: str | Path) -> dict[str, mx.array]:
    """Load a flat safetensors file back into a nested weight dict."""
    flat = mx.load(str(path), format="safetensors")
    tuples = [(tuple(k.split(".")), v) for k, v in flat.items()]
    return tree_unflatten(tuples)
