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
import tempfile
from pathlib import Path
from typing import Any

# The suffix age gives an encrypted file. Kept next to the name it encrypts so
# a locked directory reads as locked.
SUFFIX = ".age"
RECIPIENTS = "adapter_recipients.txt"
IDENTITY = "adapter_identity.txt"


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
        print("  age is not installed (brew install age age-plugin-yubikey)")
        return 1
    if not recipients.exists():
        print(f"  no recipients file at {recipients}")
        return 1
    targets = [p for p in sorted(folder.glob("*.safetensors"))]
    if not targets:
        print(f"  no weights in {folder}")
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
            print(f"  lock produced an empty file for {target.name}")
            return 1
        target.unlink()
        if not quiet:
            print(f"  locked {target.name} -> {out.name}")
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
    ap.add_argument("action", choices=["status", "lock", "unlock"])
    ap.add_argument("folder")
    ap.add_argument("--recipients", default=RECIPIENTS)
    ap.add_argument("--identity", default=IDENTITY)
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    if args.action == "status":
        locked = encrypted_files(folder)
        if not locked:
            print(f"  {folder}: not locked")
            return 1
        state = "locked" if is_locked(folder) else "HALF LOCKED (plaintext still present)"
        print(f"  {folder}: {state}")
        for path in locked:
            print(f"    {path.name}")
        return 0 if is_locked(folder) else 1

    if args.action == "lock":
        return lock(folder, args.recipients)

    # unlock in place: for retiring the lock, not for loading. The load path
    # decrypts into a temp directory that is deleted straight after; writing
    # the plaintext back here is a deliberate, permanent undo.
    ok, detail = unlock_into(folder, args.identity, folder)
    if not ok:
        print(f"  unlock failed: {detail}")
        return 1
    for path in encrypted_files(folder):
        path.unlink()
    print(f"  {folder}: unlocked in place; the weights are plaintext again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
