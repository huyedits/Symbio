"""Small MLX transformer trained from scratch.

Architecture choices:
  * RoPE (rotary positional embeddings), no learned positional embeddings.
  * Grouped-query attention (GQA) to keep memory small.
  * Pre-normalization with RMSNorm.
  * SwiGLU-ish FFN (gate + up projection).
  * All weights are plain MLX arrays so we can take gradients end-to-end.

The model is intentionally small (a few MB) so it can be trained/iterated on
CPU or a single Apple Silicon GPU.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import NativeConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(x.astype(mx.float32).square().mean(-1, keepdims=True) + self.eps) * self.weight


class RoPE:
    """Rotary positional embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        self._freqs_cis = mx.exp(1j * freqs)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x can be (..., seq_len, head_dim); seq_len is the second-to-last dim.
        seq_len = x.shape[-2]
        freqs_cis = self._freqs_cis[offset : offset + seq_len]
        x_r = x[..., 0::2]
        x_i = x[..., 1::2]
        x_complex = x_r.astype(mx.float32) + 1j * x_i.astype(mx.float32)
        x_rotated = x_complex * freqs_cis
        x_out = mx.stack([x_rotated.real, x_rotated.imag], axis=-1).flatten(-2)
        return x_out.astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: NativeConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        bsz, seq_len, dim = x.shape
        q = self.wq(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # GQA: repeat k/v heads to match q heads.
        if self.n_heads != self.n_kv_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, config: NativeConfig):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.up = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.down = nn.Linear(config.intermediate_dim, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: NativeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = FeedForward(config)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NativeLM(nn.Module):
    """Tiny transformer language model."""

    def __init__(self, config: NativeConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        """Tie input/output embeddings to cut parameter count."""
        self.lm_head.weight = self.embedding.weight

    def __call__(self, x: mx.array, targets: mx.array | None = None) -> tuple[mx.array, mx.array | None]:
        _, seq_len = x.shape
        mask = self._causal_mask(seq_len)
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss = mx.mean(loss)
        return logits, loss

    def _causal_mask(self, seq_len: int) -> mx.array:
        # Lower-triangular mask for causal attention.
        mask = mx.triu(mx.full((seq_len, seq_len), float("-inf"), mx.float32), k=1)
        return mask[None, None, :, :]

    def generate(self, x: mx.array, max_new: int = 64, temperature: float = 0.7) -> mx.array:
        for _ in range(max_new):
            logits, _ = self(x)
            next_token_logits = logits[:, -1, :] / temperature
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs))
            x = mx.concatenate([x, next_token[:, None]], axis=1)
        return x

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), self.parameters())

    @classmethod
    def load(cls, path: str | Path, config: NativeConfig) -> "NativeLM":
        model = cls(config)
        params = mx.load(str(path))
        model.update(params)
        return model
