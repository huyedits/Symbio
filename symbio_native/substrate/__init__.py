"""Substrate branches: failure-driven, non-destructive model updates.

A substrate branch is a small delta (LoRA-style low-rank or raw diff) that
corrects a specific failure mode. Branches are kept separate from the base
weights and selected at inference time by the router.
"""

from .branch import Branch
from .manager import BranchManager

__all__ = ["Branch", "BranchManager"]
