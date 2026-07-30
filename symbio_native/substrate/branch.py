"""A single substrate branch (delta) and low-rank helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json

import mlx.core as mx

from ..utils.safetensors import load_weights, save_weights


@dataclass
class Branch:
    """Lightweight correction branch.

    `name`         human-readable tag, e.g. "refuse-tool-injection".
    `trigger_ids`  token ids that signal this branch is relevant.
    `weights`      dict of parameter deltas relative to the base model.
    `description`  what failure this branch fixes.
    `rank`         LoRA rank if this is a low-rank branch; 0 means full diff.
    """

    name: str
    trigger_ids: list[int]
    weights: dict[str, mx.array]
    description: str
    rank: int = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.name,
            "trigger_ids": self.trigger_ids,
            "description": self.description,
            "rank": self.rank,
        }
        (path / "meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        save_weights(path / "weights.safetensors", self.weights)

    @classmethod
    def load(cls, path: str | Path) -> "Branch":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        weights = load_weights(path / "weights.safetensors")
        return cls(
            name=meta["name"],
            trigger_ids=meta["trigger_ids"],
            weights=weights,
            description=meta["description"],
            rank=meta.get("rank", 0),
        )
