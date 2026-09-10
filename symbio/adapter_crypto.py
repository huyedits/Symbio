"""Adapter weights that are ciphertext on disk until the security key says so.

The signature in adapter_seal.py answers "are these the bytes I sealed". It
cannot answer "may these bytes be used at all", because verification needs
only a public key — by design, so a load works with the security key in a
drawer. This module answers the other question, and it answers it with
arithmetic rather than with a policy: a locked adapter is encrypted to a key
whose private half lives on the YubiKey, so without the key there is nothing
to load. Not a refusal that could be patched out of this file — unreadable
bytes.

    lock    adapters.safetensors  ->  adapters.safetensors.age, plaintext removed
    unlock  at load, into a private temp file that is deleted straight after

WHAT THIS DOES NOT PROTECT. The plaintext exists on disk for as long as the
model takes to load it: safetensors is mmap'd, so there has to be a real file.
It is written 0600 into a directory only this user can enter, and unlinked in
a finally. Anything that can read this user's memory or watch that directory
during a load can have the weights, and the key being touch-gated does not
change that. What it stops is the copied folder, the stolen disk, and the
backup — cases where the weights travel and the key does not.

TOUCH POLICY is set when the YubiKey identity is created, not here.
`--touch-policy cached` gives one touch per 15 seconds, which is the setting
that makes this livable: a chat start touches once and the worker swaps that
follow it do not. `always` asks on every single load, including every deep
sleep wake, which on a browser session is a lot of touching.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The suffix age gives an encrypted file. Kept next to the name it encrypts so
# a locked directory reads as locked.
SUFFIX = ".age"
RECIPIENTS = "adapter_recipients.txt"
IDENTITY = "adapter_identity.txt"


def say(text: str = "") -> None:
    """Print, unbuffered.

    The setup below hands the terminal to age-plugin-yubikey, which writes
    straight to it — so with these buffered, its errors appeared ABOVE the
    header explaining what was being attempted.
    """
    print(text, flush=True)


def age_available() -> bool:
    return shutil.which("age") is not None


def encrypted_files(folder: Path) -> list[Path]:
    return sorted(Path(folder).glob(f"*{SUFFIX}"))


def is_locked(folder: Path) -> bool:
    """Whether this adapter is ciphertext waiting for a key.

    Locked means there is something encrypted AND the plaintext it replaced is
    gone. A directory holding both is a lock that was never finished, and
    treating it as locked would hide a plaintext adapter that is still being
    loaded.
    """
    folder = Path(folder)
    locked = encrypted_files(folder)
    if not locked:
        return False
    return not any(p.with_suffix("").exists() for p in locked)


def lock(folder: Path, recipients: Path | str, quiet: bool = False) -> int:
    """Encrypt every weights file in `folder` to `recipients`.

    The plaintext is removed only after the ciphertext has been written AND
    read back — an adapter is hours of training, and a lock that deleted the
    original on the strength of an exit code would be one bad write away from
    destroying it.
    """
    folder = Path(folder)
    recipients = Path(str(recipients)).expanduser()
    if not age_available():
        say("  age is not installed (brew install age age-plugin-yubikey)")
        return 1
    if not recipients.exists():
        say(f"  no recipients file at {recipients}")
        return 1
    targets = [p for p in sorted(folder.glob("*.safetensors"))]
    if not targets:
        say(f"  no weights in {folder}")
        return 1
    for target in targets:
        out = target.with_suffix(target.suffix + SUFFIX)
        result = subprocess.run(
            ["age", "-R", str(recipients), "-o", str(out), str(target)],
            capture_output=True, text=True)
        if result.returncode != 0 or not out.exists():
            print(f"  lock failed for {target.name}: "
                  f"{(result.stderr or result.stdout).strip()}")
            return 1
        if out.stat().st_size <= 0:
            say(f"  lock produced an empty file for {target.name}")
            return 1
        target.unlink()
        if not quiet:
            say(f"  locked {target.name} -> {out.name}")
    return 0


def unlock_into(folder: Path, identity: Path | str, dest: Path) -> tuple[bool, str]:
    """Decrypt `folder`'s weights into `dest`. Returns (ok, detail).

    This is where the security key is actually called: age hands the header to
    age-plugin-yubikey, which needs the key present and — under a cached or
    always touch policy — a touch. There is no way to reach the plaintext that
    does not go through it.
    """
    folder, dest = Path(folder), Path(dest)
    identity = Path(str(identity)).expanduser()
    if not age_available():
        return False, "age is not installed (brew install age age-plugin-yubikey)"
    if not identity.exists():
        return False, f"no identity file at {identity}"
    for locked in encrypted_files(folder):
        out = dest / locked.with_suffix("").name
        result = subprocess.run(
            ["age", "-d", "-i", str(identity), "-o", str(out), str(locked)],
            capture_output=True, text=True)
        if result.returncode != 0:
            return False, _age_error(result.stderr or result.stdout)
        try:
            os.chmod(out, 0o600)
        except OSError:
            pass
    return True, ""


def _age_error(text: str) -> str:
    """The line of age's stderr that says what went wrong.

    age ends every failure with "report unexpected or unhelpful errors at
    <url>", which is the last line and therefore the one a naive [-1] picks —
    so the message a person saw when their key was missing was a bug-report
    URL. Take the first line that is actually about the failure.
    """
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line or line.startswith("age: report"):
            continue
        return line.removeprefix("age: ").strip()
    return "decryption failed"


class Unlocked:
    """A locked adapter directory, readable for the length of a `with` block.

    Everything that is NOT encrypted — adapter_config.json, the seal, the
    progress file — is copied in beside the decrypted weights, because
    mlx_lm's loader wants one directory holding all of it.

    The directory is 0700 and removed in a finally. A crash between the two
    leaves plaintext in the system temp dir, which is why the name says what
    it is: someone finding it should know immediately what they have found.
    """

    def __init__(self, folder: Path, identity: Path | str):
        self.folder = Path(folder)
        self.identity = identity
        self.temp: Path | None = None

    def __enter__(self) -> Path:
        self.temp = Path(tempfile.mkdtemp(prefix="symbio-adapter-plaintext-"))
        try:
            os.chmod(self.temp, 0o700)
        except OSError:
            pass
        ok, detail = unlock_into(self.folder, self.identity, self.temp)
        if not ok:
            self.__exit__(None, None, None)
            raise PermissionError(
                f"the adapter is locked and could not be unlocked: {detail}. "
                f"Plug in the security key (and touch it when it blinks), or "
                f"unset agent.adapter_identity to run the base model.")
        for extra in sorted(self.folder.iterdir()):
            if extra.is_file() and extra.suffix != SUFFIX:
                shutil.copy2(extra, self.temp / extra.name)
        return self.temp

    def __exit__(self, *_exc) -> None:
        if self.temp is not None:
            shutil.rmtree(self.temp, ignore_errors=True)
            self.temp = None


def identity_path(config: dict[str, Any] | None) -> str:
    return str((config or {}).get("agent", {}).get("adapter_identity", "") or "")


def should_unlock(adapter_path: str | Path | None,
                  config: dict[str, Any] | None) -> bool:
    """Whether this load has to go through the key.

    Driven by the directory, not only by config: an adapter someone locked is
    locked whether or not this install remembers configuring it, and the
    honest failure there is "you need the key", not a confusing loader error
    about a missing file.
    """
    if not adapter_path:
        return False
    folder = Path(adapter_path)
    return folder.is_dir() and is_locked(folder)


# -------------------------------------------------------------------- setup

def _found(ok: bool) -> str:
    return "  ok  " if ok else "  --  "


def _ykman() -> str | None:
    """ykman, wherever it was installed. pipx puts it outside PATH for a
    non-login shell, which is where a lot of "not installed" comes from."""
    found = shutil.which("ykman")
    if found:
        return found
    guess = Path.home() / ".local/bin/ykman"
    return str(guess) if guess.exists() else None


def _yubikeys() -> list[str]:
    ykman = _ykman()
    if not ykman:
        return []
    try:
        out = subprocess.run([ykman, "list"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.stdout.strip().splitlines() if line.strip()]


def _piv_warnings() -> list[str]:
    """Default PIN/PUK/management key, which decide what the lock is worth.

    With them at defaults, anyone holding the key can reset the PIV applet and
    regenerate the slot — so a lost key is a lost adapter AND a usable key for
    whoever finds it. Worth saying before someone locks hours of training to it.
    """
    ykman = _ykman()
    if not ykman:
        return []
    try:
        out = subprocess.run([ykman, "piv", "info"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines()
            if line.strip().startswith("WARNING:")]


def _age_identities() -> str:
    try:
        out = subprocess.run(["age-plugin-yubikey", "--list"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def _looks_like_identity(path: Path) -> bool:
    """Does this file hold an age identity, rather than merely exist?"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.strip().upper().startswith("AGE-")
               for line in text.splitlines())


def _has_recipients(path: Path) -> bool:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    return any(line.strip().startswith("age1") for line in text.splitlines())


def setup_secure_enclave(identity_out: Path, recipients_out: Path,
                         access: str = "passcode") -> int:
    """The Apple Secure Enclave route: no PIN, no PUK, nothing that can block.

    Here because the YubiKey route cost a blocked PIV PIN and a spent PUK try
    before anything was encrypted. PIV has three separate credentials, its own
    6-8 byte PIN rule that a shorter FIDO2 PIN cannot satisfy, and a lockout
    that needs a factory reset of the applet to clear. The enclave has none of
    that: the private key is generated inside this Mac and cannot leave it.

    access="none" prompts for nothing and still means a locked adapter copied
    to another machine, restored from a backup, or read off the SSD is
    unreadable — the key is not in the file, it is in this Mac.
    access="passcode" additionally asks for the login password at each load,
    which also covers someone sitting at this Mac while it is unlocked.
    """
    say("Locking the adapter to this Mac's Secure Enclave\n")
    have_plugin = shutil.which("age-plugin-se") is not None
    say(f"{_found(age_available())}age")
    say(f"{_found(have_plugin)}age-plugin-se")
    if not (age_available() and have_plugin):
        say("\n  Install both, then run this again:")
        say("    brew install age age-plugin-se")
        return 1

    identity_out = Path(identity_out).expanduser()
    identity_out.parent.mkdir(parents=True, exist_ok=True)
    # Existence is not validity, which is the same mistake as the empty
    # recipients file this whole guided setup exists to prevent — and it was
    # made here too: a zero-byte identity left by an earlier failed redirect
    # was reported as "identity already at ...", reused, and produced an empty
    # recipients file that age then rejected two commands later.
    if _looks_like_identity(identity_out):
        say(f"{_found(True)}identity already at {identity_out}")
    else:
        if identity_out.exists():
            say(f"  --  {identity_out} exists but holds no identity; replacing it")
        result = subprocess.run(
            ["age-plugin-se", "keygen", "--access-control", access,
             "-o", str(identity_out)],
            capture_output=True, text=True)
        if result.returncode != 0:
            say(f"  --  keygen failed: "
                f"{(result.stderr or result.stdout).strip()}")
            return 1
        say(f"{_found(True)}identity: {identity_out}  (access control: {access})")

    recipients_out = Path(recipients_out).expanduser()
    recipients_out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["age-plugin-se", "recipients", "-i", str(identity_out),
         "-o", str(recipients_out)],
        capture_output=True, text=True)
    # Content, not existence: age-plugin-se exits 0 having written nothing
    # when the identity it was given is empty, and the failure then surfaces
    # from `age` at lock time as "no recipients found".
    if result.returncode != 0 or not _has_recipients(recipients_out):
        detail = (result.stderr or result.stdout).strip()
        say(f"  --  could not derive recipients"
            + (f": {detail}" if detail else " (the file came out empty)"))
        return 1
    say(f"{_found(True)}recipients: {recipients_out}")

    if access != "none":
        say(f"\n  Each adapter load will ask for your Mac login password.")
        say(f"  That includes worker swaps and deep-sleep wakes, which happen")
        say(f"  several times in a browser session. --access-control none")
        say(f"  removes the prompt and still binds the adapter to this Mac.")
    say("\nNext, and read this line before you run it: locking DELETES the")
    say("plaintext weights. Keep a copy until a real load has worked.")
    say("    cp -R adapters adapters.backup")
    say(f"    python3 symbio/adapter_crypto.py lock adapters "
        f"--recipients {recipients_out}")
    say(f"    ./symb config set agent.adapter_identity {identity_out}")
    return 0


def setup(identity_out: Path, recipients_out: Path, touch: str = "cached",
          slot: str = "", name: str = "symbio-adapters") -> int:
    """Walk the whole thing, checking each step instead of assuming it.

    Every failure this replaces was a real one, hit in sequence: a redirect
    into a directory that did not exist; `ykman piv keys generate`, which makes
    a key with no certificate so age cannot see it; and an empty recipients
    file that only announced itself several commands later as "no recipients
    found". Each of those is a step that half-worked and said nothing.
    """
    say("Locking the adapter to a security key\n")

    have_age = age_available()
    say(f"{_found(have_age)}age")
    have_plugin = shutil.which("age-plugin-yubikey") is not None
    say(f"{_found(have_plugin)}age-plugin-yubikey")
    if not (have_age and have_plugin):
        say("\n  Install both, then run this again:")
        print("    brew install age age-plugin-yubikey")
        return 1

    keys = _yubikeys()
    say(f"{_found(bool(keys))}security key" + (f": {keys[0]}" if keys else ""))
    if not keys:
        say("\n  Plug the YubiKey in and run this again.")
        return 1

    for warning in _piv_warnings():
        say(f"  !!  {warning}")
    if _piv_warnings():
        say("      Anyone holding this key could reset it and regenerate the")
        print("      slot. Worth fixing before locking hours of training to it:")
        say("        ykman piv access change-pin")
        print("        ykman piv access change-puk")
        say("        ykman piv access change-management-key --generate --protect")
        print()

    existing = _age_identities()
    if existing:
        say(f"{_found(True)}age identity already on the key")
    else:
        say(f"{_found(False)}age identity — generating one now")
        if not sys.stdin.isatty():
            # It prompts for a PIN and waits for a touch, and neither can
            # happen down a pipe. Saying so beats age's "IO error: not a
            # terminal" followed by this script guessing at a busy slot.
            say("\n  This step needs a terminal: it asks for the PIV PIN and")
            say("  waits for you to touch the key. Run it directly:")
            say("      python3 symbio/adapter_crypto.py setup")
            return 1
        say("      This asks for the PIV PIN (default 123456) and a touch.\n")
        cmd = ["age-plugin-yubikey", "--generate",
               "--touch-policy", touch, "--name", name]
        if slot:
            cmd += ["--slot", slot, "--force"]
        # stdio inherited on purpose: it prompts, and capturing that turns a
        # PIN prompt into a hang with no explanation.
        if subprocess.run(cmd).returncode != 0:
            say("\n  Generation failed. If the slot is already in use, pass")
            print("  --slot 9a to overwrite it.")
            return 1
        existing = _age_identities()
        if not existing:
            say("\n  The key generated but age still lists no recipients.")
            return 1

    for path, flag, label in ((identity_out, "--identity", "identity"),
                              (recipients_out, "--list", "recipients")):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        out = subprocess.run(["age-plugin-yubikey", flag],
                             capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout.strip():
            say(f"  --  could not write the {label} file")
            return 1
        path.write_text(out.stdout)
        say(f"{_found(True)}{label}: {path}  ({len(out.stdout)} bytes)")

    say("\nNext, and read this line before you run it: locking DELETES the")
    print("plaintext weights. Keep a copy until a real load has worked.")
    say(f"    cp -R adapters adapters.backup")
    print(f"    python3 symbio/adapter_crypto.py lock adapters "
          f"--recipients {recipients_out}")
    say(f"    ./symb config set agent.adapter_identity {identity_out}")
    return 0


# ---------------------------------------------------------------------- cli

_USAGE = """Lock an adapter to a security key, and see whether it is locked.

    python3 symbio/adapter_crypto.py status  adapters
    python3 symbio/adapter_crypto.py lock    adapters --recipients <file>
    python3 symbio/adapter_crypto.py unlock  adapters --identity <file>   (in place)

Run as a FILE, not as a module: this imports nothing but the standard library,
so it works on a machine with no MLX and without loading the package.

Making the key's identity, once:

    ykman piv keys generate --algorithm ECCP256 --touch-policy cached 9a -
    age-plugin-yubikey --identity > ~/.config/symbio/adapter_identity.txt
    age-plugin-yubikey --list > adapter_recipients.txt
"""


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description=_USAGE.splitlines()[0],
        epilog=_USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["setup", "status", "lock", "unlock"])
    ap.add_argument("folder", nargs="?", default="adapters")
    ap.add_argument("--recipients", default=RECIPIENTS)
    ap.add_argument("--identity", default=IDENTITY)
    ap.add_argument("--touch-policy", default="cached",
                    choices=["cached", "always", "never"])
    ap.add_argument("--slot", default="")
    ap.add_argument("--secure-enclave", action="store_true",
                    help="use this Mac's Secure Enclave instead of a YubiKey: "
                         "no PIN, no PUK, nothing that can block")
    ap.add_argument("--access-control", default="passcode",
                    choices=["none", "passcode", "any-biometry-or-passcode"],
                    help="secure-enclave only; 'none' binds the adapter to "
                         "this Mac without prompting for anything")
    args = ap.parse_args(argv)

    if args.action == "setup":
        if args.secure_enclave:
            return setup_secure_enclave(
                Path("~/.config/symbio/adapter_identity.txt").expanduser(),
                Path(RECIPIENTS).resolve(),
                access=args.access_control)
        return setup(
            Path("~/.config/symbio/adapter_identity.txt").expanduser(),
            Path(RECIPIENTS).resolve(),
            touch=args.touch_policy, slot=args.slot)

    folder = Path(args.folder)
    if args.action == "status":
        locked = encrypted_files(folder)
        if not locked:
            say(f"  {folder}: not locked")
            return 1
        state = "locked" if is_locked(folder) else "HALF LOCKED (plaintext still present)"
        say(f"  {folder}: {state}")
        for path in locked:
            say(f"    {path.name}")
        return 0 if is_locked(folder) else 1

    if args.action == "lock":
        return lock(folder, args.recipients)

    # unlock in place: for retiring the lock, not for loading. The load path
    # decrypts into a temp directory that is deleted straight after; writing
    # the plaintext back here is a deliberate, permanent undo.
    ok, detail = unlock_into(folder, args.identity, folder)
    if not ok:
        say(f"  unlock failed: {detail}")
        return 1
    for path in encrypted_files(folder):
        path.unlink()
    say(f"  {folder}: unlocked in place; the weights are plaintext again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
