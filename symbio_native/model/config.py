"""Hyperparameters for the native Symbio model."""

from dataclasses import dataclass


@dataclass
class NativeConfig:
    vocab_size: int = 1024
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    max_seq_len: int = 512
    intermediate_dim: int | None = None
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = self.dim * 4
        assert self.dim % self.n_heads == 0
        self.head_dim = self.dim // self.n_heads

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "max_seq_len": self.max_seq_len,
            "intermediate_dim": self.intermediate_dim,
            "dropout": self.dropout,
            "rope_theta": self.rope_theta,
        }
