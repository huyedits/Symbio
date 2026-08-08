"""Guards against the two ways the fine-tune corpus goes bad silently.

A degenerate sample makes mlx_lm report garbage rather than fail (loss ~1e8,
negative token counts), and a crashed test run leaves its fixtures in the real
corpus where every later run preserves them. Both were observed in practice;
neither surfaced until a fine-tune behaved strangely.
"""
import json

from symbio import constants
from symbio.app import training
from test_utils import (
    CRASH_BACKUP_DIR,
    preserve_training_state,
    recover_interrupted_training_state,
)


def _write(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"text": t}) for t in texts) + "\n", encoding="utf-8")


def _texts(path):
    return [json.loads(line)["text"]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


REAL = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
       "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n" \
       "<|im_start|>assistant\nParis.<|im_end|>"


# ---- drop_degenerate_samples ----

def test_drops_single_token_sample(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", train)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    _write(train, [REAL, "prompt", REAL])

    removed = training.drop_degenerate_samples()

    assert _texts(train) == [REAL, REAL]
    assert removed[str(train)] == [(2, "prompt")]


def test_drops_empty_and_whitespace_samples(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", train)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    _write(train, [REAL, "", "   \n ", REAL])

    training.drop_degenerate_samples()

    assert _texts(train) == [REAL, REAL]


def test_keeps_real_samples_untouched(tmp_path, monkeypatch):
    """The guard must not rewrite a clean corpus — including its byte content."""
    train = tmp_path / "train.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", train)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    _write(train, [REAL, REAL])
    before = train.read_bytes()

    assert training.drop_degenerate_samples() == {}
    assert train.read_bytes() == before


def test_leaves_short_but_well_formed_samples_alone(tmp_path, monkeypatch):
    """Corruption is removable; contamination is not distinguishable from data.

    A leaked test fixture is well-formed and only junk by provenance, so the
    guard deliberately keeps it rather than guessing.
    """
    fixture = "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1c<|im_end|>"
    train = tmp_path / "train.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", train)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    _write(train, [REAL, fixture])

    assert training.drop_degenerate_samples() == {}
    assert _texts(train) == [REAL, fixture]


def test_uses_tokenizer_when_available(tmp_path, monkeypatch):
    """Past the character floor, the token count is what decides."""
    long_but_one_token = "x" * 200

    class OneTokenTokenizer:
        def encode(self, text):
            return [0]

    train = tmp_path / "train.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", train)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    _write(train, [long_but_one_token])

    assert training.drop_degenerate_samples(OneTokenTokenizer())
    assert _texts(train) == []


def test_broken_tokenizer_does_not_delete_data(tmp_path, monkeypatch):
    class BrokenTokenizer:
        def encode(self, text):
            raise RuntimeError("tokenizer unavailable")

    train = tmp_path / "train.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", train)
    monkeypatch.setattr(constants, "VALID_FILE", tmp_path / "valid.jsonl")
    _write(train, [REAL, REAL])

    assert training.drop_degenerate_samples(BrokenTokenizer()) == {}
    assert _texts(train) == [REAL, REAL]


def test_cleans_valid_file_too(tmp_path, monkeypatch):
    valid = tmp_path / "valid.jsonl"
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "train.jsonl")
    monkeypatch.setattr(constants, "VALID_FILE", valid)
    _write(valid, [REAL, "prompt"])

    training.drop_degenerate_samples()

    assert _texts(valid) == [REAL]


# ---- crash recovery ----

def test_recovers_corpus_after_a_killed_run(tmp_path, monkeypatch):
    """Simulate a run that dies before its `finally`: the mirror survives."""
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    _write(train, [REAL])
    _write(valid, [REAL])

    monkeypatch.setattr("test_utils.TRAIN_FILE", train)
    monkeypatch.setattr("test_utils.VALID_FILE", valid)
    monkeypatch.setattr("test_utils.CRASH_BACKUP_DIR", tmp_path / ".bak")
    monkeypatch.setattr("test_utils._MANIFEST", tmp_path / ".bak" / "manifest.json")

    # Enter the guard, poison the corpus, then abandon it without teardown.
    cm = preserve_training_state()
    cm.__enter__()
    _write(train, [REAL, "prompt", "Q1/A1c fixture junk"])

    assert len(_texts(train)) == 3, "precondition: corpus is poisoned"

    restored = recover_interrupted_training_state()

    assert str(train) in restored
    assert _texts(train) == [REAL], "corpus not rolled back to pre-run state"


def test_recovery_is_a_no_op_after_a_clean_run(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    _write(train, [REAL])
    _write(valid, [REAL])

    monkeypatch.setattr("test_utils.TRAIN_FILE", train)
    monkeypatch.setattr("test_utils.VALID_FILE", valid)
    monkeypatch.setattr("test_utils.CRASH_BACKUP_DIR", tmp_path / ".bak")
    monkeypatch.setattr("test_utils._MANIFEST", tmp_path / ".bak" / "manifest.json")

    with preserve_training_state():
        _write(train, [REAL, "junk"])

    assert _texts(train) == [REAL]  # normal teardown already restored it
    assert not (tmp_path / ".bak").exists(), "mirror should be cleared on clean exit"
    assert recover_interrupted_training_state() == []


def test_recovery_removes_a_file_that_did_not_exist_before(tmp_path, monkeypatch):
    """A run that creates the corpus from nothing must not leave it behind."""
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"

    monkeypatch.setattr("test_utils.TRAIN_FILE", train)
    monkeypatch.setattr("test_utils.VALID_FILE", valid)
    monkeypatch.setattr("test_utils.CRASH_BACKUP_DIR", tmp_path / ".bak")
    monkeypatch.setattr("test_utils._MANIFEST", tmp_path / ".bak" / "manifest.json")

    cm = preserve_training_state()
    cm.__enter__()
    _write(train, [REAL])  # a test created it

    recover_interrupted_training_state()

    assert not train.exists()


def test_nested_guard_does_not_overwrite_the_outer_mirror(tmp_path, monkeypatch):
    """The inner snapshot already contains outer junk; it must not win.

    Otherwise a per-test guard nested inside the suite-wide one would keep
    resetting the crash mirror to a state that already includes test writes,
    making it useless for exactly the crash it exists to survive.
    """
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    _write(train, ["PRISTINE"])
    _write(valid, [REAL])

    monkeypatch.setattr("test_utils.TRAIN_FILE", train)
    monkeypatch.setattr("test_utils.VALID_FILE", valid)
    monkeypatch.setattr("test_utils.CRASH_BACKUP_DIR", tmp_path / ".bak")
    monkeypatch.setattr("test_utils._MANIFEST", tmp_path / ".bak" / "manifest.json")

    outer = preserve_training_state()
    outer.__enter__()
    _write(train, ["PRISTINE", "outer junk"])

    inner = preserve_training_state()
    inner.__enter__()
    _write(train, ["PRISTINE", "outer junk", "inner junk"])
    # Both abandoned, as a kill would.

    recover_interrupted_training_state()

    assert _texts(train) == ["PRISTINE"]


# ---- implausible-loss guard ----

def test_plausible_loss_accepts_normal_range():
    for v in (0.05, 1.177, 2.795, 11.9, 19.99):
        assert training._plausible_loss(v), v


def test_plausible_loss_rejects_broken_numbers():
    """Every value here was actually emitted by a run on this machine."""
    for v in (0.0, float("nan"), float("inf"), float("-inf"),
              -1.0, 773094195.2, 8.678e25, 386547097.6):
        assert not training._plausible_loss(v), v


def test_early_stop_aborts_on_implausible_val_loss(tmp_path, monkeypatch):
    """A garbage loss must abort the run, not become a 'best' checkpoint."""
    import subprocess as sp

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "0000100_adapters.safetensors").write_bytes(b"checkpoint")
    stopped = []

    class FakeProc:
        returncode = 0
        stdout = iter([
            "Iter 1: Val loss 2.795, Val took 88s\n",
            "Iter 30: Val loss nan, Val took 88s\n",
            "Iter 60: Val loss 1.000, Val took 88s\n",   # must never be reached
        ])
        def wait(self, timeout=None): return 0
        def send_signal(self, sig): stopped.append(sig)

    monkeypatch.setattr(sp, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(training.subprocess, "Popen", lambda *a, **k: FakeProc())

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("{}", encoding="utf-8")

    ok = training._run_training_with_early_stop(
        ["echo"], {"early_stop_patience": 2, "save_every": 100},
        adapter_dir, str(cfg_path))

    assert ok is False, "a run reporting nan must not report success"
    assert stopped, "the trainer subprocess should have been signalled"
    # The garbage run must not be promoted into the live adapter.
    assert not (adapter_dir / "adapters.safetensors").exists()


def test_val_loss_regex_matches_broken_values():
    """The monitor has to be able to *see* nan/inf to react to them."""
    import re
    pat = re.compile(r"Iter\s+(\d+):\s+Val\s+loss\s+"
                     r"([-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))", re.IGNORECASE)
    for text, expect in [("Iter 30: Val loss nan, Val took 8s", "nan"),
                         ("Iter 30: Val loss inf, Val took 8s", "inf"),
                         ("Iter 30: Val loss -inf, Val took 8s", "-inf"),
                         ("Iter 30: Val loss 2.795, Val took 8s", "2.795"),
                         ("Iter 30: Val loss 7.7e8, Val took 8s", "7.7e8")]:
        m = pat.search(text)
        assert m and m.group(2) == expect, (text, m and m.group(2))
