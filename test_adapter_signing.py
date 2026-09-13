"""Signing the adapter root with a key that is not on the disk.

Every other check in adapter_seal reads files that sit in the same directory
tree as the adapters: whoever can rewrite an adapter can rewrite its seal and
rebuild the root beside it, and then both checks pass. Verified live — the
per-adapter verify said "intact in root" and verify-root said "intact" over
weights that had just been replaced. The signature is the first thing an
attacker with write access cannot produce.
"""
import json
import os
import shutil
import subprocess

import pytest

import adapter_seal as seal_mod
from symbio import adapter_integrity, constants


pytestmark = pytest.mark.skipif(
    shutil.which("ssh-keygen") is None, reason="ssh-keygen not available")


@pytest.fixture
def signed(tmp_path, monkeypatch):
    """A project with one sealed adapter, a root, and a signing key.

    A software ed25519 key, not a security key: `ssh-keygen -Y sign` and
    `-Y verify` are the same commands either way, and the only difference is
    where the private half lives. What cannot be tested without the hardware
    is the touch prompt.
    """
    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    # adapter_integrity loads its checker from PROJECT_DIR/adapter_seal.py, so
    # the redirected project needs one — and the module cache has to be reset
    # or a previous test's copy answers for this one.
    shutil.copy("adapter_seal.py", tmp_path / "adapter_seal.py")
    adapter_integrity._seal_module = None
    adapter_integrity._seal_tried = False
    adapter_integrity._disk_policy = None
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"W" * 512)
    seal_mod.seal(adapters, integrity_only=True, quiet=True)
    seal_mod.build_root(tmp_path, quiet=True)

    key = tmp_path / "key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                    "-C", "huy@symbio", "-f", str(key)], check=True)
    signers = tmp_path / "signers"
    signers.write_text(f"huy@symbio {(key.with_suffix('.pub')).read_text()}")
    yield tmp_path, key, signers
    adapter_integrity._seal_module = None
    adapter_integrity._seal_tried = False
    adapter_integrity._disk_policy = None


def _config(signers, identity="huy@symbio", policy="warn"):
    return {"agent": {"verify_adapters": policy,
                      "adapter_signer_identity": identity,
                      "adapter_signers": str(signers)}}


# ---- the signature itself ----

def test_a_signed_root_verifies(signed):
    root, key, signers = signed

    assert seal_mod.sign_root(root, str(key), quiet=True) == 0
    ok, detail = seal_mod.verify_root_signature(root, str(signers), "huy@symbio")

    assert ok, detail


def test_one_changed_byte_of_the_root_breaks_it(signed):
    root, key, signers = signed
    seal_mod.sign_root(root, str(key), quiet=True)

    doc = json.loads((root / seal_mod.ROOT_NAME).read_text())
    doc["root"] = "0" * 64
    (root / seal_mod.ROOT_NAME).write_text(json.dumps(doc, indent=2))

    ok, _detail = seal_mod.verify_root_signature(root, str(signers), "huy@symbio")
    assert ok is False


def test_the_attack_the_seals_cannot_catch(signed):
    """The whole reason for this. An attacker rewrites the weights, re-seals
    the adapter, and rebuilds the root — and both existing checks pass, which
    was verified against the real CLI before this was written."""
    root, key, signers = signed
    seal_mod.sign_root(root, str(key), quiet=True)

    (root / "adapters" / "adapters.safetensors").write_bytes(b"EVIL" * 128)
    seal_mod.seal(root / "adapters", integrity_only=True, quiet=True)
    seal_mod.build_root(root, quiet=True)

    assert seal_mod.check(root / "adapters")["state"] == "intact"   # passes
    assert seal_mod.verify_root(root) == 0                          # passes
    ok, _ = seal_mod.verify_root_signature(root, str(signers), "huy@symbio")
    assert ok is False                                              # does not


def test_a_signature_from_another_key_is_not_accepted(signed, tmp_path):
    root, _key, signers = signed
    other = tmp_path / "other"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                    "-C", "someone@else", "-f", str(other)], check=True)

    seal_mod.sign_root(root, str(other), quiet=True)

    ok, _ = seal_mod.verify_root_signature(root, str(signers), "huy@symbio")
    assert ok is False


def test_a_missing_signature_is_reported_not_assumed_good(signed):
    root, _key, signers = signed

    ok, detail = seal_mod.verify_root_signature(root, str(signers), "huy@symbio")

    assert ok is False
    assert ".sig" in detail


def test_verification_needs_only_the_public_half(signed):
    """Which is what keeps this usable: the load-time check has to work with
    the security key in a drawer, or it becomes the check people turn off."""
    root, key, signers = signed
    seal_mod.sign_root(root, str(key), quiet=True)
    os.unlink(key)                       # the private half is gone

    ok, detail = seal_mod.verify_root_signature(root, str(signers), "huy@symbio")

    assert ok, detail


# ---- at load ----

def test_the_load_is_silent_until_a_signer_is_configured(signed):
    """A check with nothing to check against would warn on every install that
    never set one up, and a check that fires when nothing is wrong is one
    people switch off — taking the real check with it."""
    said = []

    adapter_integrity.enforce(signed[0] / "adapters",
                              {"agent": {"verify_adapters": "warn"}},
                              output_fn=said.append)

    assert said == []


def test_a_broken_signature_is_reported_at_load(signed):
    root, key, signers = signed
    seal_mod.sign_root(root, str(key), quiet=True)
    (root / "adapters" / "adapters.safetensors").write_bytes(b"EVIL" * 128)
    seal_mod.seal(root / "adapters", integrity_only=True, quiet=True)
    seal_mod.build_root(root, quiet=True)
    said = []

    adapter_integrity.enforce(root / "adapters", _config(signers),
                              output_fn=said.append)

    assert any("signed root does not verify" in line for line in said)
    assert any("sign-root" in line for line in said)      # and how to fix it


def test_refuse_stops_a_change_that_did_not_re_seal(signed):
    """Caught by the SEAL, not the signature, and that is the right division
    of labour: the signature covers the root document, and weights that changed
    without the root being rebuilt leave the root — and so its signature —
    untouched. The adapter's own leaf no longer matches, which is what fires."""
    root, key, signers = signed
    seal_mod.sign_root(root, str(key), quiet=True)
    (root / "adapters" / "adapters.safetensors").write_bytes(b"EVIL" * 128)

    with pytest.raises(RuntimeError, match="failed its seal"):
        adapter_integrity.enforce(root / "adapters",
                                  _config(signers, policy="refuse"),
                                  output_fn=lambda _t: None)


def test_refuse_stops_the_change_that_covered_its_tracks(signed):
    """And this one is caught by the signature alone — the seal and the root
    were both rebuilt to match, so nothing else has anything to object to."""
    root, key, signers = signed
    seal_mod.sign_root(root, str(key), quiet=True)
    (root / "adapters" / "adapters.safetensors").write_bytes(b"EVIL" * 128)
    seal_mod.seal(root / "adapters", integrity_only=True, quiet=True)
    seal_mod.build_root(root, quiet=True)

    with pytest.raises(RuntimeError, match="signature"):
        adapter_integrity.enforce(root / "adapters",
                                  _config(signers, policy="refuse"),
                                  output_fn=lambda _t: None)


def test_a_correctly_signed_root_says_nothing(signed):
    root, key, signers = signed
    seal_mod.sign_root(root, str(key), quiet=True)
    said = []

    adapter_integrity.enforce(root / "adapters", _config(signers),
                              output_fn=said.append)

    assert said == []
