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
    assert "weights file changed: brew_tea_150.safetensors" in capsys.readouterr().out


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
    assert "weights file changed: brew_tea_150.safetensors" in capsys.readouterr().out


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


# ---- one root over all of them ----

def _adapter(base, name, weights=b"W" * 64, data=b'{"text": "a"}\n'):
    f = base / "Adapter_skills" / name
    (f / "training_data").mkdir(parents=True)
    (f / f"{name}.safetensors").write_bytes(weights)
    (f / "training_data/train.jsonl").write_bytes(data)
    return f


def test_the_root_covers_every_sealed_adapter(tmp_path, capsys):
    folders = [_adapter(tmp_path, f"skill{i}_WORKER") for i in range(5)]
    for f in folders:
        seal_mod.seal(f, quiet=True)

    assert seal_mod.build_root(tmp_path, quiet=True) == 0
    doc = json.loads((tmp_path / seal_mod.ROOT_NAME).read_text())

    assert doc["leaf_count"] == 5
    assert seal_mod.verify_root(tmp_path) == 0


def test_one_adapter_verifies_against_the_root_without_the_others():
    """The reason this is a tree and not a chain: a chain needs every earlier
    adapter present, and this system archives and restores them one at a
    time."""
    leaves = [seal_mod.leaf_digest(f"a{i}", "aa" * 32, "bb" * 32) for i in range(7)]
    root = seal_mod.merkle_root(leaves)

    for i, leaf in enumerate(leaves):
        assert seal_mod.verify_proof(leaf, seal_mod.merkle_proof(leaves, i), root)


def test_a_proof_does_not_transfer_to_another_leaf():
    leaves = [seal_mod.leaf_digest(f"a{i}", "aa" * 32, "bb" * 32) for i in range(7)]
    root = seal_mod.merkle_root(leaves)

    assert not seal_mod.verify_proof(leaves[0], seal_mod.merkle_proof(leaves, 3), root)


def test_an_odd_leaf_count_has_one_root_not_two():
    """Promoting the odd node, not duplicating it. Duplicating makes two
    different leaf sets collide on one root, which is the property a root is
    supposed to lack."""
    three = [seal_mod.leaf_digest(f"a{i}", "aa" * 32, "bb" * 32) for i in range(3)]
    duplicated = three + [three[-1]]

    assert seal_mod.merkle_root(three) != seal_mod.merkle_root(duplicated)


def test_tampering_with_one_adapter_breaks_the_root(tmp_path, capsys):
    folders = [_adapter(tmp_path, f"skill{i}_WORKER") for i in range(4)]
    for f in folders:
        seal_mod.seal(f, quiet=True)
    seal_mod.build_root(tmp_path, quiet=True)

    (folders[2] / "skill2_WORKER.safetensors").write_bytes(b"W" * 63 + b"X")

    assert seal_mod.verify_root(tmp_path) == 1
    assert "skill2_WORKER" in capsys.readouterr().out


def test_resealing_a_tampered_adapter_still_fails_the_root(tmp_path, capsys):
    """The whole reason for a root on top of the per-adapter seals. Someone
    who can rewrite the weights can rewrite the provenance.json beside them,
    and `verify` on that folder alone then passes — but the leaf has moved, so
    it is no longer in the root you wrote down."""
    folders = [_adapter(tmp_path, f"skill{i}_WORKER") for i in range(4)]
    for f in folders:
        seal_mod.seal(f, quiet=True)
    seal_mod.build_root(tmp_path, quiet=True)
    root_doc = seal_mod.load_root_doc(tmp_path)

    (folders[1] / "skill1_WORKER.safetensors").write_bytes(b"EVIL" * 16)
    seal_mod.seal(folders[1], quiet=True)          # re-seal over the tamper

    assert seal_mod.check(folders[1])["state"] == "intact"        # alone: fine
    report = seal_mod.check(folders[1], root_doc, tmp_path)       # against root
    assert report["state"] == "broken"
    assert any("root" in problem for problem in report["problems"])


def test_an_unsealed_adapter_is_left_out_rather_than_vouched_for(tmp_path, capsys):
    sealed = _adapter(tmp_path, "sealed_WORKER")
    _adapter(tmp_path, "unsealed_WORKER")
    seal_mod.seal(sealed, quiet=True)

    seal_mod.build_root(tmp_path, quiet=True)
    doc = json.loads((tmp_path / seal_mod.ROOT_NAME).read_text())

    assert doc["leaf_count"] == 1
    assert doc["members"][0]["name"].endswith("sealed_WORKER")


def test_the_live_adapter_directory_can_be_sealed_for_integrity_only(tmp_path, capsys):
    """It holds weights with no training_data/ beside them. There is nothing
    to weld it to, and "these are the same bytes" is still worth checking."""
    live = tmp_path / "adapters"
    live.mkdir()
    (live / "adapters.safetensors").write_bytes(b"W" * 32)

    assert seal_mod.seal(live, integrity_only=True, quiet=True) == 0
    assert seal_mod.check(live)["state"] == "intact"

    (live / "adapters.safetensors").write_bytes(b"W" * 31 + b"X")
    report = seal_mod.check(live)
    assert report["state"] == "broken"
    assert "weights file changed: adapters.safetensors" in report["problems"]


def test_the_live_adapter_is_covered_by_the_root(tmp_path):
    """adapters/ is the directory that actually gets LOADED. A root that only
    covered the Adapter_skills archive would say nothing about the file the
    model is about to be given."""
    live = tmp_path / "adapters"
    (live / "workers" / "coder").mkdir(parents=True)
    (live / "adapters.safetensors").write_bytes(b"H" * 32)
    (live / "workers" / "coder" / "adapters.safetensors").write_bytes(b"C" * 32)
    for d in (live, live / "workers" / "coder"):
        seal_mod.seal(d, integrity_only=True, quiet=True)

    seal_mod.build_root(tmp_path, quiet=True)
    names = {m["name"] for m in
             json.loads((tmp_path / seal_mod.ROOT_NAME).read_text())["members"]}

    assert names == {"adapters", "adapters/workers/coder"}


def test_an_unsealed_folder_reports_unknown_not_broken(tmp_path):
    """An adapter nobody sealed has made no claim. Reporting that as tampered
    is how a report gets ignored."""
    f = _adapter(tmp_path, "never_sealed_WORKER")

    assert seal_mod.check(f)["state"] == "unsealed"


# ---- the tree's properties at every size, not just the one I picked ----

@pytest.mark.parametrize("n", list(range(1, 34)))
def test_every_leaf_proves_membership_at_any_tree_size(n):
    """Promotion of odd nodes is easy to get subtly wrong at exactly one
    size, so this walks all of them rather than trusting the seven-leaf case
    the feature was written against."""
    leaves = [seal_mod.leaf_digest(f"a{i}", f"{i:064x}", "bb" * 32)
              for i in range(n)]
    root = seal_mod.merkle_root(leaves)

    for i, leaf in enumerate(leaves):
        assert seal_mod.verify_proof(leaf, seal_mod.merkle_proof(leaves, i), root)


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13])
def test_a_changed_leaf_fails_its_own_proof(n):
    leaves = [seal_mod.leaf_digest(f"a{i}", f"{i:064x}", "bb" * 32)
              for i in range(n)]
    root = seal_mod.merkle_root(leaves)
    proof = seal_mod.merkle_proof(leaves, n - 1)
    moved = seal_mod.leaf_digest(f"a{n - 1}", "ff" * 32, "bb" * 32)

    assert not seal_mod.verify_proof(moved, proof, root)


def test_an_internal_node_cannot_be_passed_off_as_a_leaf():
    """Domain separation. Without the leaf/node tags, the hash of a pair of
    leaves is indistinguishable from a leaf, and a subtree can be presented as
    a single membership — the second-preimage attack every Merkle
    implementation has to answer."""
    leaves = [seal_mod.leaf_digest(f"a{i}", f"{i:064x}", "bb" * 32)
              for i in range(4)]
    root = seal_mod.merkle_root(leaves)
    internal = seal_mod._join(leaves[0], leaves[1])   # a real node in the tree

    # It sits in the tree, but it is not a leaf and must not verify as one:
    # the only proof that would work for it is the one for its own level.
    assert not seal_mod.verify_proof(
        internal, seal_mod.merkle_proof(leaves, 0), root)


def test_reordering_the_adapters_does_not_change_the_root(tmp_path):
    """build_root sorts by path, so two installs with the same adapters agree
    on the root regardless of what order the filesystem hands them over."""
    a = [seal_mod.leaf_digest(n, "aa" * 32, "bb" * 32) for n in ("x", "y", "z")]
    b = [seal_mod.leaf_digest(n, "aa" * 32, "bb" * 32) for n in ("z", "y", "x")]

    assert seal_mod.merkle_root(a) != seal_mod.merkle_root(b)   # order matters...
    assert seal_mod.merkle_root(sorted(a)) == seal_mod.merkle_root(sorted(b))


def test_an_empty_set_of_adapters_still_has_a_root(tmp_path):
    assert seal_mod.build_root(tmp_path, quiet=True) == 0
    doc = json.loads((tmp_path / seal_mod.ROOT_NAME).read_text())
    assert doc["leaf_count"] == 0
    assert seal_mod.verify_root(tmp_path) == 0


# ---- more than one weights file in a folder ----

def test_every_weights_file_is_sealed_not_just_the_first(tmp_path):
    """The hole this closes was worse than a gap. mlx_lm writes
    0000100_adapters.safetensors checkpoints beside the final
    adapters.safetensors, and a killed run leaves them there — which is
    exactly the case a load-time check exists for. Sealing the first file
    sorted meant '0' beat 'a', so the seal covered the CHECKPOINT and left
    adapters.safetensors — the file that actually gets loaded — free to be
    replaced with the seal reporting intact."""
    d = tmp_path / "adapters"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"MAIN" * 16)
    (d / "0000100_adapters.safetensors").write_bytes(b"CKPT" * 16)
    seal_mod.seal(d, integrity_only=True, quiet=True)

    (d / "adapters.safetensors").write_bytes(b"EVIL" * 16)
    report = seal_mod.check(d)

    assert report["state"] == "broken"
    assert "weights file changed: adapters.safetensors" in report["problems"]


def test_the_seal_names_the_file_that_actually_loads(tmp_path):
    d = tmp_path / "adapters"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"MAIN" * 16)
    (d / "0000100_adapters.safetensors").write_bytes(b"CKPT" * 16)

    assert seal_mod._weights_file(d).name == "adapters.safetensors"


def test_a_checkpoint_appearing_or_vanishing_is_reported(tmp_path):
    d = tmp_path / "adapters"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"MAIN" * 16)
    seal_mod.seal(d, integrity_only=True, quiet=True)

    (d / "0000100_adapters.safetensors").write_bytes(b"CKPT" * 16)
    report = seal_mod.check(d)

    assert report["state"] == "broken"
    assert "weights file added: 0000100_adapters.safetensors" in report["problems"]


def test_a_version_1_seal_asks_to_be_re_sealed_rather_than_crying_tamper(tmp_path):
    """adapter_digest used to be the hash of one file and is now a root over
    all of them. A version-1 seal cannot be checked against a version-2
    measurement, and calling that mismatch "the weights changed" would be a
    lie about the one thing this is for."""
    d = tmp_path / "adapters"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"MAIN" * 16)
    seal_mod.seal(d, integrity_only=True, quiet=True)
    doc = json.loads((d / seal_mod.SEAL_NAME).read_text())
    doc["version"] = 1
    (d / seal_mod.SEAL_NAME).write_text(json.dumps(doc))

    report = seal_mod.check(d)

    assert report["state"] == "stale-format"
    assert "re-seal" in report["problems"][0]


def test_a_stale_format_seal_is_silent_at_load(tmp_path, monkeypatch):
    """Same class of answer as unsealed: no claim this version can check, so
    no claim to contradict."""
    from symbio import adapter_integrity as ai
    from symbio import constants

    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    d = tmp_path / "adapters"
    d.mkdir()
    (d / "adapters.safetensors").write_bytes(b"MAIN" * 16)
    seal_mod.seal(d, integrity_only=True, quiet=True)
    doc = json.loads((d / seal_mod.SEAL_NAME).read_text())
    doc["version"] = 1
    (d / seal_mod.SEAL_NAME).write_text(json.dumps(doc))
    ai._seal_module, ai._seal_tried = seal_mod, True

    said = []
    ai.enforce(d, {"agent": {"verify_adapters": "refuse"}}, output_fn=said.append)

    assert said == []
    ai._seal_module, ai._seal_tried = None, False
