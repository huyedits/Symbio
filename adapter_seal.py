#!/usr/bin/env python3
"""Weld an adapter to the training data it came from, and prove it later.

    python3 adapter_seal.py seal    Adapter_skills/brew_tea_150_WORKER
    python3 adapter_seal.py verify  Adapter_skills/brew_tea_150_WORKER
    python3 adapter_seal.py seal-all
    python3 adapter_seal.py verify-all
    python3 adapter_seal.py challenge Adapter_skills/brew_tea_150_WORKER
    python3 adapter_seal.py respond   Adapter_skills/... --nonce HEX

HOW IT WORKS

    data_digest    a Merkle root over the training files: each file hashed,
                   the (name, hash) pairs sorted, then hashed together. Sorted
                   so it does not depend on directory order; per-file so the
                   report can say WHICH file moved.

    adapter_digest sha256 of the weights.

    pair_key       HKDF-SHA256 over (data_digest || adapter_digest). This is
                   the weld: it can only be derived by someone holding both
                   artifacts, unmodified. Change one byte of either and the
                   key is different.

    pair_commit    sha256(pair_key), written into the seal. Publishing the
                   commitment rather than the key means the seal itself can
                   never be used to forge a response.

WHAT A PASSING VERIFY ACTUALLY PROVES

That these exact weights and these exact training files are the ones sealed
together, and neither has changed since. That is integrity and pairing.

It does NOT prove the adapter was trained on that data. Nothing computed after
the fact can: the weights do not carry a receipt of what produced them. The
seal is a witness written at a moment in time, and it is worth exactly as much
as that moment was. Seal at the end of a training run and it means a great
deal; seal an adapter you found lying around and it means the two files were
sitting next to each other when you said so.

THE CHALLENGE

`challenge` prints a random nonce. `respond` derives pair_key from the files
on disk and returns HMAC(pair_key, nonce), which requires both artifacts —
neither alone is enough, which is the "each unlocks it in turn" part. The
verifier checks it by doing the same from its own copy. Since a nonce is never
reused, a captured response cannot be replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import sys
from datetime import datetime
from pathlib import Path

SEAL_NAME = "provenance.json"
HKDF_INFO = b"symbio-adapter-training-weld-v1"
_CHUNK = 1 << 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _training_files(folder: Path) -> list[Path]:
    d = folder / "training_data"
    if not d.is_dir():
        return []
    return sorted((p for p in d.rglob("*") if p.is_file()),
                  key=lambda p: str(p.relative_to(d)))


def _weights_file(folder: Path) -> Path | None:
    """The archived weights, whatever they are called."""
    man = folder / "manifest.json"
    if man.exists():
        try:
            name = json.loads(man.read_text(encoding="utf-8")).get("weights")
            if name and (folder / name).exists():
                return folder / name
        except json.JSONDecodeError:
            pass
    for p in sorted(folder.glob("*.safetensors")):
        return p
    return None


def data_digest(folder: Path) -> tuple[str, dict[str, str]]:
    """Merkle-ish root over the training files, plus the per-file hashes.

    Per-file hashes are kept so a failed verify can name the file that moved
    rather than only saying the data changed.
    """
    d = folder / "training_data"
    per_file: dict[str, str] = {}
    for p in _training_files(folder):
        per_file[str(p.relative_to(d))] = _sha256_file(p)
    root = hashlib.sha256()
    for name in sorted(per_file):
        root.update(name.encode("utf-8"))
        root.update(bytes.fromhex(per_file[name]))
    return root.hexdigest(), per_file


def _hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def pair_key(data_hex: str, adapter_hex: str, salt_hex: str) -> bytes:
    return _hkdf_sha256(bytes.fromhex(data_hex) + bytes.fromhex(adapter_hex),
                        bytes.fromhex(salt_hex), HKDF_INFO)


def _measure(folder: Path) -> dict:
    weights = _weights_file(folder)
    if weights is None:
        raise FileNotFoundError(f"no .safetensors in {folder}")
    d_hex, per_file = data_digest(folder)
    return {
        "weights_name": weights.name,
        "adapter_digest": _sha256_file(weights),
        "data_digest": d_hex,
        "files": per_file,
    }


def seal(folder: Path, quiet: bool = False) -> int:
    m = _measure(folder)
    if not m["files"]:
        print(f"  {folder.name}: no training_data/ — nothing to weld it to")
        return 1
    salt = secrets.token_hex(16)
    key = pair_key(m["data_digest"], m["adapter_digest"], salt)
    doc = {
        "version": 1,
        "sealed": datetime.now().isoformat(timespec="seconds"),
        "weights_name": m["weights_name"],
        "adapter_digest": m["adapter_digest"],
        "data_digest": m["data_digest"],
        "file_count": len(m["files"]),
        "files": m["files"],
        "salt": salt,
        # The commitment, never the key: a seal that carried the key could be
        # used to answer a challenge without holding either artifact.
        "pair_commit": hashlib.sha256(key).hexdigest(),
        "proves": ("These weights and these training files were sealed "
                   "together and neither has changed since. Not that one was "
                   "trained from the other — nothing after the fact can show "
                   "that."),
    }
    (folder / SEAL_NAME).write_text(json.dumps(doc, indent=2) + "\n",
                                    encoding="utf-8")
    if not quiet:
        print(f"  sealed {folder.name}")
        print(f"    data     {m['data_digest'][:16]}  ({len(m['files'])} file(s))")
        print(f"    adapter  {m['adapter_digest'][:16]}")
        print(f"    weld     {doc['pair_commit'][:16]}")
    return 0


def verify(folder: Path, quiet: bool = False) -> int:
    path = folder / SEAL_NAME
    if not path.exists():
        print(f"  {folder.name}: NOT SEALED")
        return 1
    doc = json.loads(path.read_text(encoding="utf-8"))
    m = _measure(folder)

    problems: list[str] = []
    if m["adapter_digest"] != doc["adapter_digest"]:
        problems.append("adapter weights changed")
    if m["data_digest"] != doc["data_digest"]:
        sealed, now = doc.get("files", {}), m["files"]
        for name in sorted(set(sealed) | set(now)):
            if name not in now:
                problems.append(f"training file removed: {name}")
            elif name not in sealed:
                problems.append(f"training file added: {name}")
            elif sealed[name] != now[name]:
                problems.append(f"training file changed: {name}")

    if not problems:
        key = pair_key(m["data_digest"], m["adapter_digest"], doc["salt"])
        if hashlib.sha256(key).hexdigest() != doc["pair_commit"]:
            problems.append("weld does not match the commitment")

    if problems:
        print(f"  {folder.name}: BROKEN")
        for p in problems:
            print(f"      {p}")
        return 1
    if not quiet:
        print(f"  {folder.name}: intact "
              f"({doc['file_count']} training file(s), sealed {doc['sealed']})")
    return 0


def challenge(folder: Path) -> int:
    if not (folder / SEAL_NAME).exists():
        print(f"  {folder.name} is not sealed")
        return 1
    nonce = secrets.token_hex(16)
    print(f"  nonce: {nonce}")
    print(f"  respond with: python3 {Path(__file__).name} respond "
          f"{folder} --nonce {nonce}")
    return 0


def respond(folder: Path, nonce: str) -> int:
    path = folder / SEAL_NAME
    if not path.exists():
        print(f"  {folder.name} is not sealed")
        return 1
    doc = json.loads(path.read_text(encoding="utf-8"))
    m = _measure(folder)
    key = pair_key(m["data_digest"], m["adapter_digest"], doc["salt"])
    proof = hmac.new(key, bytes.fromhex(nonce), hashlib.sha256).hexdigest()
    print(f"  {proof}")
    return 0


def check_response(folder: Path, nonce: str, proof: str) -> int:
    path = folder / SEAL_NAME
    if not path.exists():
        print(f"  {folder.name} is not sealed")
        return 1
    doc = json.loads(path.read_text(encoding="utf-8"))
    m = _measure(folder)
    key = pair_key(m["data_digest"], m["adapter_digest"], doc["salt"])
    expect = hmac.new(key, bytes.fromhex(nonce), hashlib.sha256).hexdigest()
    ok = hmac.compare_digest(expect, proof.strip())
    print(f"  {'ACCEPTED' if ok else 'REJECTED'}")
    return 0 if ok else 1


def _folders(root: Path) -> list[Path]:
    out = root / "Adapter_skills"
    if not out.is_dir():
        return []
    return sorted(p for p in out.iterdir() if p.is_dir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["seal", "verify", "seal-all", "verify-all",
                                       "challenge", "respond", "check"])
    ap.add_argument("folder", nargs="?", help="an Adapter_skills/<NAME> folder")
    ap.add_argument("--nonce")
    ap.add_argument("--proof")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    if args.action in ("seal-all", "verify-all"):
        folders = _folders(Path(args.root))
        if not folders:
            print("Nothing in Adapter_skills/. Run archive_adapters.py first.")
            return 1
        fn = seal if args.action == "seal-all" else verify
        bad = sum(fn(f) != 0 for f in folders)
        print(f"\n  {len(folders) - bad}/{len(folders)} ok")
        return 1 if bad else 0

    if not args.folder:
        print(f"{args.action} needs a folder")
        return 1
    folder = Path(args.folder)
    if args.action == "seal":
        return seal(folder)
    if args.action == "verify":
        return verify(folder)
    if args.action == "challenge":
        return challenge(folder)
    if args.action == "respond":
        if not args.nonce:
            print("respond needs --nonce")
            return 1
        return respond(folder, args.nonce)
    if args.action == "check":
        if not (args.nonce and args.proof):
            print("check needs --nonce and --proof")
            return 1
        return check_response(folder, args.nonce, args.proof)
    return 1


if __name__ == "__main__":
    sys.exit(main())
