"""Symbio Native: a small MLX transformer built from scratch.

This is an experimental branch under `symbio_native/` that trains a custom
language model from random initialization using MLX. The key idea is to keep
the base weights "liquid":

  * failures create small substrate branches (LoRA-like low-rank updates)
    instead of overwriting the main model,
  * a lightweight router decides which substrate(s) to apply per turn,
  * the base model is never frozen; it can drift, but substrate branches can be
    pruned, replayed, or merged later.

Entry points:
  - symbio_native.tokenizer.train_tokenizer
  - symbio_native.model.NativeLM
  - symbio_native.train.train
  - symbio_native.router.Router
  - symbio_native.substrate.BranchManager
"""

__version__ = "0.1.0"
