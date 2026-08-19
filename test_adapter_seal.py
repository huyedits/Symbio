"""Welding an adapter to its training data.

The seal is only worth what it catches, so most of these break something on
purpose and check the report says which thing broke.
"""
import hashlib
import json

import pytest

import adapter_seal as seal_mod


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "brew_tea_150_WORKER"
    (f / "training_data").mkdir(parents=True)
    (f / "brew_tea_150.safetensors").write_bytes(b"WEIGHTS" * 100)
    (f / "manifest.json").write_text(json.dumps({
        "skill": "brew_tea", "kind": "WORKER", "iters": 150,
        "weights": "brew_tea_150.safetensors"}))
    (f / "training_data/train.jsonl").write_text('{"text": "a"}\n')
    (f / "training_data/valid.jsonl").write_text('{"text": "b"}\n')
    return f


def test_sealing_then_verifying_passes(folder, capsys):
    assert seal_mod.seal(folder) == 0
    assert seal_mod.verify(folder) == 0


def test_an_unsealed_folder_is_reported_not_assumed_good(folder):
    assert seal_mod.verify(folder) == 1


def test_the_seal_records_both_digests_and_a_commitment(folder):
    seal_mod.seal(folder)
    doc = json.loads((folder / "provenance.json").read_text())
    assert len(doc["adapter_digest"]) == 64
    assert len(doc["data_digest"]) == 64
    assert len(doc["pair_commit"]) == 64
    assert doc["file_count"] == 2


def test_the_key_itself_is_never_written(folder):
    """A seal carrying the key could answer a challenge without the files."""
    seal_mod.seal(folder)
    doc = json.loads((folder / "provenance.json").read_text())
    key = seal_mod.pair_key(doc["data_digest"], doc["adapter_digest"], doc["salt"])
    assert key.hex() not in json.dumps(doc)
    assert hashlib.sha256(key).hexdigest() == doc["pair_commit"]


# ---- what it catches ----

def test_one_changed_byte_of_training_data_breaks_it(folder, capsys):
    seal_mod.seal(folder)
    p = folder / "training_data/train.jsonl"
    p.write_bytes(p.read_bytes() + b" ")
    assert seal_mod.verify(folder) == 1
    assert "training file changed: train.jsonl" in capsys.readouterr().out


def test_one_flipped_bit_of_the_adapter_breaks_it(folder, capsys):
    seal_mod.seal(folder)
    w = folder / "brew_tea_150.safetensors"
    b = w.read_bytes()
    w.write_bytes(b[:-1] + bytes([b[-1] ^ 0x01]))
    assert seal_mod.verify(folder) == 1
    assert "adapter weights changed" in capsys.readouterr().out


def test_a_removed_training_file_is_named(folder, capsys):
    seal_mod.seal(folder)
    (folder / "training_data/valid.jsonl").unlink()
    assert seal_mod.verify(folder) == 1
    assert "removed: valid.jsonl" in capsys.readouterr().out


def test_an_added_training_file_is_named(folder, capsys):
    seal_mod.seal(folder)
    (folder / "training_data/extra.jsonl").write_text("{}\n")
    assert seal_mod.verify(folder) == 1
    assert "added: extra.jsonl" in capsys.readouterr().out


def test_swapping_in_a_different_adapter_breaks_it(folder, capsys):
    """The case the whole thing exists for: right data, wrong weights."""
    seal_mod.seal(folder)
    (folder / "brew_tea_150.safetensors").write_bytes(b"SOMEONE-ELSES" * 50)
    assert seal_mod.verify(folder) == 1
    assert "adapter weights changed" in capsys.readouterr().out


# ---- the weld ----

def test_the_key_needs_both_halves(folder):
    seal_mod.seal(folder)
    doc = json.loads((folder / "provenance.json").read_text())
    real = seal_mod.pair_key(doc["data_digest"], doc["adapter_digest"], doc["salt"])
    other = "11" * 32
    assert seal_mod.pair_key(other, doc["adapter_digest"], doc["salt"]) != real
    assert seal_mod.pair_key(doc["data_digest"], other, doc["salt"]) != real


def test_file_order_does_not_change_the_data_digest(folder, tmp_path):
    """Sorted hashing, so a different directory listing order is not a break."""
    first, _ = seal_mod.data_digest(folder)
    (folder / "training_data/train.jsonl").touch()
    second, _ = seal_mod.data_digest(folder)
    assert first == second


# ---- challenge / response ----

def _proof(folder, nonce, capsys):
    """respond() prints the proof; take the last line so a preceding banner
    from seal() cannot end up inside it."""
    seal_mod.respond(folder, nonce)
    return capsys.readouterr().out.strip().splitlines()[-1].strip()


def test_a_fresh_response_is_accepted(folder, capsys):
    seal_mod.seal(folder, quiet=True)
    nonce = "ab" * 16
    proof = _proof(folder, nonce, capsys)
    assert seal_mod.check_response(folder, nonce, proof) == 0


def test_a_response_does_not_replay_against_another_nonce(folder, capsys):
    seal_mod.seal(folder, quiet=True)
    proof = _proof(folder, "ab" * 16, capsys)
    assert seal_mod.check_response(folder, "cd" * 16, proof) == 1


def test_a_proof_from_tampered_files_is_rejected(folder, capsys):
    seal_mod.seal(folder, quiet=True)
    original = (folder / "training_data/train.jsonl").read_bytes()
    (folder / "training_data/train.jsonl").write_bytes(original + b"x")
    nonce = "ef" * 16
    tampered_proof = _proof(folder, nonce, capsys)
    (folder / "training_data/train.jsonl").write_bytes(original)
    assert seal_mod.check_response(folder, nonce, tampered_proof) == 1


def test_sealing_without_training_data_is_refused(tmp_path, capsys):
    """A weld needs two things to weld."""
    f = tmp_path / "lonely_1_WORKER"
    f.mkdir()
    (f / "lonely_1.safetensors").write_bytes(b"w")
    assert seal_mod.seal(f) == 1
    assert "nothing to weld" in capsys.readouterr().out
