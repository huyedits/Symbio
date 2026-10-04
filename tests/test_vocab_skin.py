"""Training against a skinned output vocabulary.

The head is cut to the ids the corpus uses for a LoRA run, so the logits stop
being the peak-memory term on 248k-vocab models. These check the skinned loss
is the cross-entropy restricted to the kept rows, that the adapter it trains
never includes the head, and that only models that need it leave the stock
`mlx_lm lora` command.
"""
import sys

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from symbio import backend  # noqa: E402
from symbio.app import vocab_skin  # noqa: E402

VOCAB, DIMS = 50, 32


class _Args:
    tie_word_embeddings = False


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, DIMS)
        self.proj = nn.Linear(DIMS, DIMS)

    def __call__(self, x):
        return self.proj(self.embed_tokens(x))


class _LM(nn.Module):
    def __init__(self, tied=False, quantized=False):
        super().__init__()
        self.args = _Args()
        self.args.tie_word_embeddings = tied
        self.model = _Inner()
        if not tied:
            self.lm_head = nn.Linear(DIMS, VOCAB, bias=False)
        if quantized:
            # group_size 32: the smallest MLX accepts.
            nn.quantize(self, group_size=32, bits=4,
                        class_predicate=lambda p, m: p == "lm_head")

    def __call__(self, x):
        h = self.model(x)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)


class _Tok:
    all_special_ids = [49]
    eos_token_id = 48


def _restricted_ce(full_logits, targets, keep):
    sub = full_logits[..., mx.array(keep)]
    lse = mx.logsumexp(sub.astype(mx.float32), axis=-1)
    picked = mx.take_along_axis(full_logits, targets[..., None], axis=-1)[..., 0]
    return lse - picked.astype(mx.float32)


@pytest.mark.parametrize("tied,quantized", [(False, False), (True, False), (False, True)])
def test_the_skinned_loss_is_the_cross_entropy_over_the_kept_rows(tied, quantized):
    mx.random.seed(0)
    model = _LM(tied=tied, quantized=quantized)
    batch = mx.array([[1, 7, 30, 12, 7, 2], [3, 30, 30, 9, 1, 0]])
    lengths = mx.array([[1, 5], [1, 4]])
    full_logits = model(batch[:, :-1])

    keep = vocab_skin.keep_ids([batch.reshape(-1).tolist()], _Tok(), VOCAB, base=4)
    assert {0, 1, 2, 3, 7, 9, 12, 30, 48, 49} <= set(keep) and 40 not in keep
    remap = vocab_skin.skin(model, keep)
    assert model(batch[:, :-1]).shape[-1] == len(keep)

    loss, ntoks = vocab_skin.skinned_loss(remap)(model, batch, lengths)
    steps = mx.arange(1, 6)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ref = (_restricted_ce(full_logits, batch[:, 1:], keep) * mask).sum() / mask.sum()
    assert ntoks.item() == 9
    assert abs(loss.item() - ref.item()) < 1e-3


def test_the_skinned_head_is_frozen_so_the_adapter_never_holds_it():
    model = _LM()
    model.freeze()
    model.model.proj.unfreeze()
    vocab_skin.skin(model, list(range(10)))
    trainable = dict(nn.utils.tree_flatten(model.trainable_parameters()))
    assert trainable and not any("lm_head" in k for k in trainable)


def test_dataset_tokens_reads_raw_and_cached_datasets():
    class Raw:
        def __init__(self):
            self.rows = [{"text": "a"}, {"text": "b"}]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return self.rows[i]

        def process(self, d):
            return ([ord(d["text"]), 1], 0)

    class Cached(Raw):
        def __getitem__(self, i):
            return self.process(self.rows[i])

    assert list(vocab_skin.dataset_tokens(Raw())) == [[97, 1], [98, 1]]
    assert list(vocab_skin.dataset_tokens(Cached())) == [[97, 1], [98, 1]]


def _lora():
    return {"batch_size": 1, "num_layers": 2, "learning_rate": 1e-4,
            "steps_per_eval": 30, "val_batches": 8, "max_seq_length": 2048,
            "save_every": 15}


def _local_model(tmp_path, vocab, bits=4):
    import json
    d = tmp_path / f"m{vocab}_{bits}"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"text_config": {"vocab_size": vocab}, "quantization": {"bits": bits}}))
    return str(d)


def test_a_small_vocab_model_keeps_the_stock_trainer(tmp_path):
    cmd = backend.trainer_command(_local_model(tmp_path, 151_669), "/d", "/a",
                                  _lora(), 10, "/c.yaml", {"backend": "mlx"})
    assert cmd[:4] == [sys.executable, "-m", "mlx_lm", "lora"]


@pytest.mark.parametrize("vocab,bits", [(248_320, 4), (151_669, 1)])
def test_a_wide_head_or_one_bit_weights_go_through_the_runner(tmp_path, vocab, bits):
    model = _local_model(tmp_path, vocab, bits)
    cmd = backend.trainer_command(model, "/d", "/a", _lora(), 10, "/c.yaml",
                                  {"backend": "mlx"})
    assert cmd[:3] == [sys.executable, "-m", "symbio.app.lora_runner"]
    assert cmd[cmd.index("--vocab-skin") + 1] == "auto"
    assert cmd[cmd.index("--model") + 1] == model


def test_vocab_skin_off_still_needs_the_runner_for_one_bit(tmp_path):
    lora = dict(_lora(), vocab_skin="off")
    one_bit = backend.trainer_command(_local_model(tmp_path, 151_669, 1), "/d", "/a",
                                      lora, 10, "/c.yaml", {"backend": "mlx"})
    wide = backend.trainer_command(_local_model(tmp_path, 248_320, 4), "/d", "/a",
                                   lora, 10, "/c.yaml", {"backend": "mlx"})
    assert one_bit[2] == "symbio.app.lora_runner"
    assert wide[:4] == [sys.executable, "-m", "mlx_lm", "lora"]
