"""Byte-level BPE tokenizer.

Trains directly from raw text and supports a tiny vocabulary (256 + merges).
All IDs fit in int32 for MLX. No external dependencies beyond Python stdlib.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


class BPETokenizer:
    """Encode/decode using a learned BPE merge table."""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = merges
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}
        # Reverse lookup bytes -> id.
        self._bytes_to_id = {token: idx for idx, token in vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Tokenize a string into a list of ids."""
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        if not tokens:
            return []
        # Greedily apply merges in priority order.
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            ranks = [self._merge_rank.get(p, float("inf")) for p in pairs]
            best = min(ranks)
            if best == float("inf"):
                break
            i = ranks.index(best)
            merged = pairs[i][0] + pairs[i][1]
            tokens = tokens[:i] + [merged] + tokens[i + 2 :]
        # Convert to ids; unknown bytes fall back to their single-byte id.
        return [self._bytes_to_id.get(t, t[0]) for t in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        """Convert ids back to a UTF-8 string."""
        chunks: list[bytes] = []
        for idx in ids:
            token = self.vocab.get(int(idx))
            if token is None:
                token = b"\xef\xbf\xbd"  # replacement char
            chunks.append(token)
        try:
            return b"".join(chunks).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": [[list(a), list(b)] for a, b in self.merges],
        }
        path.write_text(json.dumps(serializable), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        merges = [(bytes(a), bytes(b)) for a, b in data["merges"]]
        return cls(vocab, merges)


class BPETrainer:
    """Train a byte-level BPE tokenizer from text."""

    def __init__(self, vocab_size: int = 1024):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one per byte)")
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256

    def _get_word_tokens(self, text: str) -> list[list[bytes]]:
        """Split text into words; each word is a list of byte tokens."""
        words = re.findall(r"\S+", text)
        return [[bytes([b]) for b in w.encode("utf-8")] + [b"</w>"] for w in words]

    def train(self, text: str | Iterable[str]) -> BPETokenizer:
        """Learn a merge table from raw text."""
        if isinstance(text, str):
            text = [text]
        word_tokens: list[list[bytes]] = []
        for chunk in text:
            word_tokens.extend(self._get_word_tokens(chunk))
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[bytes, bytes]] = []

        # Frequency of each token sequence across words.
        counts = Counter()
        for tokens in word_tokens:
            for pair in zip(tokens, tokens[1:]):
                counts[pair] += 1

        for _ in range(self.num_merges):
            if not counts:
                break
            best_pair = max(counts.items(), key=lambda x: x[1])[0]
            a, b = best_pair
            merged = a + b
            merges.append(best_pair)
            vocab[len(vocab)] = merged

            # Update every word that contains the pair.
            new_counts = Counter()
            for tokens in word_tokens:
                new_tokens: list[bytes] = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word_tokens.index(tokens)] = new_tokens
                for pair in zip(new_tokens, new_tokens[1:]):
                    new_counts[pair] += 1
            counts = new_counts

        return BPETokenizer(vocab, merges)
