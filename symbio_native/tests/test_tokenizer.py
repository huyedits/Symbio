import pytest

from symbio_native.tokenizer import BPETokenizer, BPETrainer


def test_byte_level_roundtrip():
    trainer = BPETrainer(vocab_size=256)
    text = "hello world\n日本語🙂"
    tok = trainer.train([text])
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_trainer_extends_vocab(tmp_path):
    trainer = BPETrainer(vocab_size=300)
    text = "the quick brown fox jumps over the lazy dog" * 10
    tok = trainer.train([text])
    assert len(tok.vocab) <= 300
    assert len(tok.vocab) >= 256
    ids = tok.encode(text)
    assert all(0 <= i < len(tok.vocab) for i in ids)
