"""Adapter weights that are ciphertext until the security key says so.

The signature in adapter_seal answers "are these the bytes I sealed" and needs
only a public key, deliberately — so a load works with the key in a drawer.
This answers the other question, and answers it with arithmetic rather than
policy: without the key there is nothing to load, not a refusal that could be
patched out.

Tested with a software age identity. `age -R` / `age -d -i` are the same
commands whether the private half is in a file or on a YubiKey; what a
software key cannot exercise is the touch prompt.
"""
import os
import shutil
import subprocess

import pytest

from symbio import adapter_crypto as crypto


pytestmark = pytest.mark.skipif(
    shutil.which("age") is None or shutil.which("age-keygen") is None,
    reason="age is not installed")


@pytest.fixture
def adapter(tmp_path):
    folder = tmp_path / "adapters"
    folder.mkdir()
    weights = os.urandom(4096)
    (folder / "adapters.safetensors").write_bytes(weights)
    (folder / "adapter_config.json").write_text('{"model": "some/model"}')

    identity = tmp_path / "id.txt"
    subprocess.run(["age-keygen", "-o", str(identity)],
                   capture_output=True, check=True)
    public = [w for w in identity.read_text().split() if w.startswith("age1")][0]
    recipients = tmp_path / "recipients.txt"
    recipients.write_text(public + "\n")
    return folder, identity, recipients, weights


# ---- locking ----

def test_locking_replaces_the_weights_with_ciphertext(adapter):
    folder, _identity, recipients, weights = adapter

    assert crypto.lock(folder, recipients, quiet=True) == 0

    assert not (folder / "adapters.safetensors").exists()
    assert (folder / "adapters.safetensors.age").exists()
    assert weights not in (folder / "adapters.safetensors.age").read_bytes()


def test_a_directory_holding_both_is_not_called_locked(adapter):
    """A lock that was interrupted. Calling it locked would hide a plaintext
    adapter that is still being loaded."""
    folder, _identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)
    (folder / "adapters.safetensors").write_bytes(b"plaintext is back")

    assert crypto.is_locked(folder) is False


def test_locking_without_recipients_changes_nothing(adapter, tmp_path):
    """An adapter is hours of training. Nothing is deleted on a failed lock."""
    folder, _identity, _recipients, weights = adapter

    assert crypto.lock(folder, tmp_path / "nope.txt", quiet=True) == 1
    assert (folder / "adapters.safetensors").read_bytes() == weights


# ---- unlocking, which is where the key is called ----

def test_the_weights_come_back_byte_for_byte(adapter):
    folder, identity, recipients, weights = adapter
    crypto.lock(folder, recipients, quiet=True)

    with crypto.Unlocked(folder, identity) as plaintext:
        assert (plaintext / "adapters.safetensors").read_bytes() == weights


def test_what_is_not_encrypted_comes_along(adapter):
    """mlx_lm's loader wants one directory holding the config beside the
    weights."""
    folder, identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)

    with crypto.Unlocked(folder, identity) as plaintext:
        assert (plaintext / "adapter_config.json").exists()


def test_the_plaintext_is_private_while_it_exists(adapter):
    folder, identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)

    with crypto.Unlocked(folder, identity) as plaintext:
        assert oct(plaintext.stat().st_mode & 0o777) == "0o700"


def test_the_plaintext_is_gone_afterwards(adapter):
    folder, identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)

    with crypto.Unlocked(folder, identity) as plaintext:
        kept = plaintext

    assert not kept.exists()


def test_it_is_gone_even_when_the_load_raises(adapter):
    """The whole point of the context manager: the directory goes whether the
    model loaded or blew up."""
    folder, identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)
    kept = None

    with pytest.raises(RuntimeError):
        with crypto.Unlocked(folder, identity) as plaintext:
            kept = plaintext
            raise RuntimeError("Metal OOM")

    assert kept is not None and not kept.exists()


def test_without_the_key_there_is_nothing_to_load(adapter, tmp_path):
    folder, _identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)
    other = tmp_path / "other.txt"
    subprocess.run(["age-keygen", "-o", str(other)], capture_output=True, check=True)

    with pytest.raises(PermissionError, match="security key"):
        with crypto.Unlocked(folder, other):
            pass


def test_the_refusal_says_what_went_wrong_not_a_bug_report_url(adapter, tmp_path):
    """age ends every failure with "report unexpected or unhelpful errors at
    <url>" — the last line, and so the one a naive [-1] picks. What a person
    saw when their key was missing was that URL."""
    folder, _identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)
    other = tmp_path / "other.txt"
    subprocess.run(["age-keygen", "-o", str(other)], capture_output=True, check=True)

    with pytest.raises(PermissionError) as caught:
        with crypto.Unlocked(folder, other):
            pass

    assert "no identity matched" in str(caught.value)
    assert "filippo" not in str(caught.value)


# ---- and the load path asks the right question ----

def test_a_locked_adapter_is_detected_from_the_directory(adapter):
    """Not from config: an adapter someone locked is locked whether or not
    this install remembers configuring it, and the honest failure there is
    "you need the key", not a loader error about a missing file."""
    folder, _identity, recipients, _weights = adapter
    crypto.lock(folder, recipients, quiet=True)

    assert crypto.should_unlock(folder, {"agent": {}}) is True


def test_a_plaintext_adapter_loads_without_any_of_this(adapter):
    folder, _identity, _recipients, _weights = adapter

    assert crypto.should_unlock(folder, {"agent": {}}) is False


# ---- the setup that replaced four hand-typed commands ----

def test_setup_stops_at_the_first_missing_piece(monkeypatch, tmp_path, capsys):
    """Every failure this replaces was hit for real, in order: a redirect into
    a directory that did not exist, `ykman piv keys generate` making a key with
    no certificate so age could not see it, and an empty recipients file that
    only announced itself several commands later as "no recipients found".
    Each was a step that half-worked and said nothing."""
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: None)

    code = crypto.setup(tmp_path / "id.txt", tmp_path / "rec.txt")

    out = capsys.readouterr().out
    assert code == 1
    assert "brew install age age-plugin-yubikey" in out


def test_setup_says_when_no_key_is_plugged_in(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(crypto, "age_available", lambda: True)
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: "/usr/bin/thing")
    monkeypatch.setattr(crypto, "_yubikeys", list)

    code = crypto.setup(tmp_path / "id.txt", tmp_path / "rec.txt")

    out = capsys.readouterr().out
    assert code == 1
    assert "Plug the YubiKey in" in out


def test_setup_surfaces_the_piv_defaults(monkeypatch, tmp_path, capsys):
    """Default PUK and management key mean anyone holding the key can reset it
    and regenerate the slot — so a lost key is a lost adapter AND a usable key
    for whoever finds it. Worth saying before hours of training are locked to
    it."""
    monkeypatch.setattr(crypto, "age_available", lambda: True)
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: "/usr/bin/thing")
    monkeypatch.setattr(crypto, "_yubikeys", lambda: ["YubiKey 5C NFC (5.7.4)"])
    monkeypatch.setattr(crypto, "_piv_warnings",
                        lambda: ["WARNING: Using default Management key!"])
    monkeypatch.setattr(crypto, "_age_identities", lambda: "age1abc")
    monkeypatch.setattr(crypto.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0,
                                                       "stdout": "age1abc\n"})())

    crypto.setup(tmp_path / "id.txt", tmp_path / "rec.txt")

    out = capsys.readouterr().out
    assert "default Management key" in out
    assert "change-management-key" in out


def test_setup_will_not_prompt_down_a_pipe(monkeypatch, tmp_path, capsys):
    """age's own message for this is "IO error: not a terminal", after which
    the script used to guess that the slot was busy."""
    monkeypatch.setattr(crypto, "age_available", lambda: True)
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: "/usr/bin/thing")
    monkeypatch.setattr(crypto, "_yubikeys", lambda: ["YubiKey"])
    monkeypatch.setattr(crypto, "_piv_warnings", list)
    monkeypatch.setattr(crypto, "_age_identities", lambda: "")
    monkeypatch.setattr(crypto.sys.stdin, "isatty", lambda: False)

    code = crypto.setup(tmp_path / "id.txt", tmp_path / "rec.txt")

    out = capsys.readouterr().out
    assert code == 1
    assert "needs a terminal" in out


def test_setup_writes_both_files_and_makes_the_directory(monkeypatch, tmp_path, capsys):
    """The very first failure: a redirect into ~/.config/symbio, which did not
    exist."""
    monkeypatch.setattr(crypto, "age_available", lambda: True)
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: "/usr/bin/thing")
    monkeypatch.setattr(crypto, "_yubikeys", lambda: ["YubiKey"])
    monkeypatch.setattr(crypto, "_piv_warnings", list)
    monkeypatch.setattr(crypto, "_age_identities", lambda: "AGE-PLUGIN-YUBIKEY-1")
    monkeypatch.setattr(crypto.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0,
                                                       "stdout": "age1xyz\n"})())
    identity = tmp_path / "nested" / "deeper" / "id.txt"

    code = crypto.setup(identity, tmp_path / "rec.txt")

    assert code == 0
    assert identity.read_text().strip() == "age1xyz"
    assert (tmp_path / "rec.txt").exists()
    assert "DELETES the" in capsys.readouterr().out     # and warns before locking


# ---- the Secure Enclave route ----

def test_an_empty_identity_file_is_not_treated_as_an_identity(tmp_path):
    """Existence is not validity. A zero-byte identity left by an earlier
    failed redirect was reported as "identity already at ...", reused, and
    produced an empty recipients file that age rejected two commands later —
    the exact failure the guided setup was written to prevent, repeated inside
    it."""
    empty = tmp_path / "id.txt"
    empty.write_text("")

    assert crypto._looks_like_identity(empty) is False


def test_a_real_identity_is_recognised(tmp_path):
    ident = tmp_path / "id.txt"
    ident.write_text("# created: 2026-09-10\n# access control: passcode\n"
                     "AGE-PLUGIN-SE-1QQQQQ\n")

    assert crypto._looks_like_identity(ident) is True


def test_an_empty_recipients_file_is_not_recipients(tmp_path):
    """age-plugin-se exits 0 having written nothing when the identity it was
    given is empty, so a returncode check passes and the failure surfaces from
    `age` at lock time."""
    empty = tmp_path / "rec.txt"
    empty.write_text("\n")

    assert crypto._has_recipients(empty) is False


def test_recipients_are_recognised(tmp_path):
    rec = tmp_path / "rec.txt"
    rec.write_text("# comment\nage1se1qvffc7ll5ve5amazx6jhlv67wrua2e9d6q\n")

    assert crypto._has_recipients(rec) is True


def test_secure_enclave_setup_needs_its_plugin(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: None)

    code = crypto.setup_secure_enclave(tmp_path / "id.txt", tmp_path / "rec.txt")

    assert code == 1
    assert "brew install age age-plugin-se" in capsys.readouterr().out


def test_secure_enclave_setup_replaces_an_empty_identity(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(crypto, "age_available", lambda: True)
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: "/usr/bin/thing")
    identity = tmp_path / "id.txt"
    identity.write_text("")
    recipients = tmp_path / "rec.txt"

    def fake_run(cmd, **_kw):
        if "keygen" in cmd:
            identity.write_text("AGE-PLUGIN-SE-1QQ\n")
        else:
            recipients.write_text("age1se1qq\n")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(crypto.subprocess, "run", fake_run)

    code = crypto.setup_secure_enclave(identity, recipients)

    out = capsys.readouterr().out
    assert code == 0
    assert "holds no identity; replacing it" in out


def test_the_passcode_prompt_is_called_out(monkeypatch, tmp_path, capsys):
    """It fires on every worker swap and deep-sleep wake, not only at start."""
    monkeypatch.setattr(crypto, "age_available", lambda: True)
    monkeypatch.setattr(crypto.shutil, "which", lambda _name: "/usr/bin/thing")
    identity, recipients = tmp_path / "id.txt", tmp_path / "rec.txt"
    identity.write_text("AGE-PLUGIN-SE-1QQ\n")

    monkeypatch.setattr(
        crypto.subprocess, "run",
        lambda *a, **k: (recipients.write_text("age1se1qq\n"),
                         type("R", (), {"returncode": 0, "stdout": "",
                                        "stderr": ""})())[1])

    crypto.setup_secure_enclave(identity, recipients, access="passcode")

    out = capsys.readouterr().out
    assert "Mac login password" in out
    assert "deep-sleep wakes" in out


# ---- one command, not four ----

def test_setup_finishes_the_job(monkeypatch, tmp_path, capsys):
    """Printing three more commands at the end of a setup is how a setup
    becomes four commands again — which is how this one went the first time."""
    folder = tmp_path / "adapters"
    folder.mkdir()
    (folder / "adapters.safetensors").write_bytes(b"W" * 256)
    identity = tmp_path / "id.txt"
    recipients = tmp_path / "rec.txt"
    subprocess.run(["age-keygen", "-o", str(identity)],
                   capture_output=True, check=True)
    public = [w for w in identity.read_text().split() if w.startswith("age1")][0]
    recipients.write_text(public + "\n")

    code = crypto.finish(folder, identity, recipients, set_config=False)

    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "adapters.backup" / "adapters.safetensors").exists()
    assert (folder / "adapters.safetensors.age").exists()
    assert "mv" in out            # and how to undo it


def test_the_backup_is_taken_before_anything_is_deleted(tmp_path):
    """The one step nobody should be trusted to remember is the one that makes
    the rest reversible. An adapter is hours of training."""
    folder = tmp_path / "adapters"
    folder.mkdir()
    weights = b"W" * 256
    (folder / "adapters.safetensors").write_bytes(weights)
    identity = tmp_path / "id.txt"
    subprocess.run(["age-keygen", "-o", str(identity)],
                   capture_output=True, check=True)
    recipients = tmp_path / "rec.txt"
    recipients.write_text(
        [w for w in identity.read_text().split() if w.startswith("age1")][0] + "\n")

    crypto.finish(folder, identity, recipients, set_config=False)

    assert (tmp_path / "adapters.backup" / "adapters.safetensors").read_bytes() == weights


def test_a_failed_lock_leaves_the_plaintext_alone(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "adapters"
    folder.mkdir()
    (folder / "adapters.safetensors").write_bytes(b"W" * 256)

    code = crypto.finish(folder, tmp_path / "id.txt", tmp_path / "missing.txt",
                         set_config=False)

    assert code == 1
    assert (folder / "adapters.safetensors").exists()
    assert "plaintext is untouched" in capsys.readouterr().out


def test_nothing_to_lock_is_not_an_error(tmp_path, capsys):
    empty = tmp_path / "adapters"
    empty.mkdir()

    assert crypto.finish(empty, tmp_path / "id", tmp_path / "rec",
                         set_config=False) == 0
    assert "Nothing to lock" in capsys.readouterr().out


# ---- and the wizard offers it, without ever assuming yes ----

@pytest.fixture
def wizard(monkeypatch, tmp_path):
    from symbio import constants
    from symbio.app import setup as wiz

    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "PROJECT_DIR", tmp_path)
    # A real terminal. The step deliberately says nothing without one, so
    # these tests — which are about the interactive offer — have to claim one.
    monkeypatch.setattr(wiz.sys.stdin, "isatty", lambda: True)
    (tmp_path / "adapters").mkdir()
    return wiz, tmp_path / "adapters"


def test_the_wizard_is_silent_without_a_terminal(monkeypatch, wizard):
    """Silence is not consent to encrypt someone's weights, and a question
    nobody answers must not eat an input the next question was expecting —
    which is what it did: three scripted setup tests starved on the "Save this
    configuration?" prompt that followed."""
    wiz, adapters = wizard
    (adapters / "adapters.safetensors").write_bytes(b"W" * 64)
    monkeypatch.setattr(wiz.sys.stdin, "isatty", lambda: False)
    said = []

    wiz._adapter_lock_step({}, lambda _p: "y", said.append)

    assert said == []
    assert (adapters / "adapters.safetensors").exists()


def test_the_wizard_says_nothing_when_there_is_nothing_to_lock(wizard):
    wiz, _adapters = wizard
    said = []

    wiz._adapter_lock_step({}, lambda _p: "y", said.append)

    assert said == []


def test_the_wizard_offers_once_an_adapter_exists(wizard):
    wiz, adapters = wizard
    (adapters / "adapters.safetensors").write_bytes(b"W" * 64)
    said = []

    wiz._adapter_lock_step({}, lambda _p: "n", said.append)

    assert any("Adapter lock" in line for line in said)


def test_declining_leaves_the_weights_alone(wizard):
    """Default NO, because it deletes the plaintext. A wizard that encrypted
    hours of training because someone pressed Enter would be the worst kind of
    default."""
    wiz, adapters = wizard
    (adapters / "adapters.safetensors").write_bytes(b"W" * 64)
    said = []

    wiz._adapter_lock_step({}, lambda _p: "", said.append)   # bare Enter

    assert (adapters / "adapters.safetensors").exists()
    assert any("Later:" in line for line in said)


def test_an_already_locked_adapter_is_not_offered_again(wizard, tmp_path):
    wiz, adapters = wizard
    (adapters / "adapters.safetensors.age").write_bytes(b"ciphertext")
    said = []

    wiz._adapter_lock_step({}, lambda _p: "y", said.append)

    assert any("already locked" in line for line in said)


def test_the_default_folder_does_not_depend_on_where_you_run_it():
    """`setup` with no folder used to mean the relative string "adapters", so
    the same command locked the real adapter from the project root and a test
    copy from a scratch directory. That is how a live adapter — hours of
    training — got encrypted during testing. The default is now absolute."""
    import argparse
    import inspect

    source = inspect.getsource(crypto.main)

    assert 'default="adapters"' not in source
    assert "Path(__file__).resolve().parent.parent" in source
