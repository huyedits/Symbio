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

ONE ROOT OVER ALL OF THEM

    `root` builds a Merkle tree over every sealed adapter — leaf per adapter
    (its name, its weights digest, its data digest), sorted by path so the
    tree does not depend on directory order — and writes the root to
    adapters_root.json. Each adapter's own seal gets its leaf and its
    inclusion proof written into it, so ONE adapter can be verified alone
    against the root without the other adapters being present.

    A tree rather than a chain on purpose. A chain — each adapter's key
    unlocking the next — is ordered and total: verifying adapter 7 needs the
    six before it, so archiving or restoring a single skill adapter (which
    this system does routinely) breaks verification of every other one. A tree
    gives the same tamper-evidence with per-adapter proofs and no ordering.

    What the root adds over the per-adapter seals: re-sealing a tampered
    adapter updates its own provenance.json and passes `verify`, but its leaf
    changes, so it no longer proves membership in the published root. The root
    is the thing worth backing up somewhere the adapters are not.

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
ROOT_NAME = "adapters_root.json"
HKDF_INFO = b"symbio-adapter-training-weld-v1"
_CHUNK = 1 << 20

# Domain separation, so a leaf can never be read as an internal node. Without
# it a tree with a promoted odd node has more than one leaf set producing the
# same root, which is the whole point of a root.
_LEAF_TAG = b"\x00"
_NODE_TAG = b"\x01"


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


# ---------------------------------------------------------------- the tree

def leaf_digest(name: str, adapter_hex: str, data_hex: str) -> str:
    """One adapter's leaf: what it is called and what it contains."""
    h = hashlib.sha256()
    h.update(_LEAF_TAG)
    h.update(name.encode("utf-8"))
    h.update(b"\x00")                       # name/digest boundary
    h.update(bytes.fromhex(adapter_hex))
    h.update(bytes.fromhex(data_hex))
    return h.hexdigest()


def _join(left_hex: str, right_hex: str) -> str:
    return hashlib.sha256(
        _NODE_TAG + bytes.fromhex(left_hex) + bytes.fromhex(right_hex)).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    """Root over `leaves`, in the order given (build_root sorts them).

    An odd node is promoted to the next level rather than paired with a copy
    of itself: duplicating it makes two different leaf sets collide on one
    root, which is exactly the property a root is supposed to lack.
    """
    if not leaves:
        return hashlib.sha256(_NODE_TAG).hexdigest()
    level = list(leaves)
    while len(level) > 1:
        level = [_join(level[i], level[i + 1]) if i + 1 < len(level) else level[i]
                 for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: list[str], index: int) -> list[dict]:
    """The siblings needed to walk `leaves[index]` up to the root."""
    proof: list[dict] = []
    level, idx = list(leaves), index
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                if i == idx:
                    proof.append({"side": "right", "hash": level[i + 1]})
                elif i + 1 == idx:
                    proof.append({"side": "left", "hash": level[i]})
                nxt.append(_join(level[i], level[i + 1]))
            else:
                nxt.append(level[i])        # promoted; no sibling to record
        level, idx = nxt, idx // 2
    return proof


def verify_proof(leaf: str, proof: list[dict], root: str) -> bool:
    cur = leaf
    for step in proof:
        cur = (_join(step["hash"], cur) if step.get("side") == "left"
               else _join(cur, step["hash"]))
    return hmac.compare_digest(cur, root)


def seal(folder: Path, quiet: bool = False, integrity_only: bool = False) -> int:
    """Write the seal. Refuses a folder with no training data unless asked.

    integrity_only exists for the LIVE adapter directory, which has weights
    and no training_data/ beside them — there is nothing to weld it to, but
    "these are the same bytes I saw last time" is still worth being able to
    check at load, and it is what catches the half-written adapter an OOM kill
    leaves behind. The seal records which of the two it is, so a weld can
    never be mistaken for the weaker claim.
    """
    m = _measure(folder)
    if not m["files"] and not integrity_only:
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
                   "that."
                   if m["files"] else
                   "These weights have not changed since they were sealed. "
                   "There was no training data beside them to weld them to, "
                   "so this is integrity only, not provenance."),
    }
    (folder / SEAL_NAME).write_text(json.dumps(doc, indent=2) + "\n",
                                    encoding="utf-8")
    # Note the whole document is rewritten, so any membership block from an
    # older `root` run is gone with it — which is correct: re-sealing changes
    # the leaf, and a membership proof for the previous leaf would be a stale
    # claim that still verified.
    if not quiet:
        print(f"  sealed {folder.name}"
              + ("" if m["files"] else "  (integrity only — no training data)"))
        print(f"    data     {m['data_digest'][:16]}  ({len(m['files'])} file(s))")
        print(f"    adapter  {m['adapter_digest'][:16]}")
        print(f"    weld     {doc['pair_commit'][:16]}")
    return 0


def check(folder: Path, root_doc: dict | None = None,
          project_root: Path | None = None) -> dict:
    """Measure the folder against its seal and say what is wrong, if anything.

    Structured rather than printed because this is what the running app calls
    before it loads an adapter — see symbio/adapter_integrity.py. The CLI's
    verify() below is a printer wrapped around it.

    state is one of:
      "unsealed"  nothing has ever been claimed about this folder. NOT a
                  failure: an adapter that was never sealed is unknown, and
                  reporting unknown as tampered would train everyone to
                  ignore the report.
      "intact"    the bytes are what the seal says they are.
      "broken"    they are not, and `problems` names what moved.
      "unreadable" the seal or the weights could not be read at all.
    """
    path = folder / SEAL_NAME
    if not path.exists():
        return {"name": folder.name, "state": "unsealed", "problems": [],
                "sealed": None, "file_count": 0}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        m = _measure(folder)
    except (OSError, ValueError, FileNotFoundError) as e:
        return {"name": folder.name, "state": "unreadable",
                "problems": [str(e)], "sealed": None, "file_count": 0}

    problems: list[str] = []
    if m["adapter_digest"] != doc.get("adapter_digest"):
        problems.append("adapter weights changed")
    if m["data_digest"] != doc.get("data_digest"):
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
        if not hmac.compare_digest(hashlib.sha256(key).hexdigest(),
                                   doc.get("pair_commit", "")):
            problems.append("weld does not match the commitment")

    # Membership. Only meaningful once the folder's own bytes check out —
    # otherwise "the proof does not match" is just the same news twice.
    member = doc.get("membership")
    root_hex = (root_doc or {}).get("root")
    listed = None
    if root_doc and project_root is not None:
        name = _rel(folder, project_root)
        listed = next((entry for entry in root_doc.get("members", [])
                       if entry.get("name") == name), None)
    if not problems and root_hex:
        if member:
            leaf = leaf_digest(member.get("name", folder.name),
                               m["adapter_digest"], m["data_digest"])
            if not hmac.compare_digest(leaf, member.get("leaf", "")):
                problems.append("leaf does not match these bytes")
            elif not verify_proof(leaf, member.get("proof", []), root_hex):
                problems.append(
                    "not a member of the current adapters_root.json — these "
                    "bytes were re-sealed after the root was written")
        elif listed is not None:
            # Re-sealing rewrites the whole document, membership block and all.
            # That is how someone who can edit the weights makes the folder
            # verify clean on its own — so a folder the root still names, whose
            # seal has stopped claiming membership, is the interesting case,
            # not a quiet one. A folder the root does not name is merely newer
            # than the root, and says nothing either way.
            problems.append(
                "listed in adapters_root.json but its seal carries no "
                "membership proof — the seal was rewritten after the root")

    return {
        "name": folder.name,
        "state": "broken" if problems else "intact",
        "problems": problems,
        "sealed": doc.get("sealed"),
        "file_count": doc.get("file_count", 0),
        "in_root": bool(member),
    }


def verify(folder: Path, quiet: bool = False, root_doc: dict | None = None,
           project_root: Path | None = None) -> int:
    report = check(folder, root_doc, project_root)
    if report["state"] == "unsealed":
        print(f"  {folder.name}: NOT SEALED")
        return 1
    if report["state"] != "intact":
        print(f"  {folder.name}: BROKEN")
        for problem in report["problems"]:
            print(f"      {problem}")
        return 1
    if not quiet:
        member = " in root" if report.get("in_root") else ""
        print(f"  {folder.name}: intact{member} "
              f"({report['file_count']} training file(s), "
              f"sealed {report['sealed']})")
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


def adapter_folders(root: Path) -> list[Path]:
    """Every directory under this project that holds adapter weights.

    Wider than _folders(), which only sees the Adapter_skills archive. The
    root has to cover the adapters that are actually LOADED — adapters/ and
    adapters/workers/<role> — or it says nothing about the file the model is
    about to be given.
    """
    seen: list[Path] = []
    for base in (root / "adapters", root / "adapters_archive",
                 root / "Adapter_skills"):
        if not base.is_dir():
            continue
        for folder in [base, *sorted(d for d in base.rglob("*") if d.is_dir())]:
            if folder not in seen and any(folder.glob("*.safetensors")):
                seen.append(folder)
    return seen


def _rel(folder: Path, root: Path) -> str:
    """Path relative to the project, which is what names a leaf.

    The directory NAME is not unique — every worker role has an adapters/
    directory — and two leaves that disagree about which adapter they describe
    is the one thing a root must not allow.
    """
    try:
        return str(folder.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(folder.resolve())


def build_root(root: Path, quiet: bool = False) -> int:
    """Build the Merkle root over every sealed adapter and write the proofs.

    Only sealed folders are included: an unsealed adapter has made no claim,
    and inventing a leaf for it here would turn "unknown" into "vouched for"
    without anyone having looked.
    """
    folders = adapter_folders(root)
    members, skipped = [], []
    for folder in folders:
        if not (folder / SEAL_NAME).exists():
            skipped.append(folder)
            continue
        try:
            m = _measure(folder)
        except FileNotFoundError as e:
            skipped.append(folder)
            print(f"  skipped {_rel(folder, root)}: {e}")
            continue
        name = _rel(folder, root)
        members.append({
            "name": name, "folder": folder,
            "adapter_digest": m["adapter_digest"],
            "data_digest": m["data_digest"],
            "leaf": leaf_digest(name, m["adapter_digest"], m["data_digest"]),
        })
    members.sort(key=lambda e: e["name"])
    leaves = [e["leaf"] for e in members]
    root_hex = merkle_root(leaves)

    for i, entry in enumerate(members):
        seal_path = entry["folder"] / SEAL_NAME
        doc = json.loads(seal_path.read_text(encoding="utf-8"))
        doc["membership"] = {
            "name": entry["name"],
            "leaf": entry["leaf"],
            "proof": merkle_proof(leaves, i),
            "root": root_hex,
        }
        seal_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    (root / ROOT_NAME).write_text(json.dumps({
        "version": 1,
        "built": datetime.now().isoformat(timespec="seconds"),
        "root": root_hex,
        "leaf_count": len(leaves),
        "members": [{k: e[k] for k in ("name", "leaf", "adapter_digest",
                                       "data_digest")} for e in members],
        "proves": ("Each listed adapter was these exact bytes when the root "
                   "was built. Keep the root somewhere the adapters are not: "
                   "an attacker who can rewrite an adapter can rewrite the "
                   "seal beside it, but not a root you took away."),
    }, indent=2) + "\n", encoding="utf-8")

    if not quiet:
        print(f"  root {root_hex[:16]}  over {len(leaves)} sealed adapter(s)")
        for entry in members:
            print(f"    {entry['name']}")
        for folder in skipped:
            print(f"    (skipped, not sealed) {_rel(folder, root)}")
    return 0


def load_root_doc(root: Path) -> dict | None:
    """The whole adapters_root.json, or None when no root has been built."""
    try:
        return json.loads((root / ROOT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_root(root: Path) -> str | None:
    """The current root hex, or None when no root has been built."""
    return (load_root_doc(root) or {}).get("root")


def verify_root(root: Path) -> int:
    """Re-derive the root from the adapters on disk and compare."""
    path = root / ROOT_NAME
    if not path.exists():
        print(f"  no {ROOT_NAME} — run `root` first")
        return 1
    doc = json.loads(path.read_text(encoding="utf-8"))
    claimed = doc.get("root", "")
    by_name = {m["name"]: m for m in doc.get("members", [])}

    problems: list[str] = []
    leaves = []
    for folder in adapter_folders(root):
        name = _rel(folder, root)
        if name not in by_name:
            if (folder / SEAL_NAME).exists():
                problems.append(f"sealed adapter not in the root: {name}")
            continue
        try:
            m = _measure(folder)
        except FileNotFoundError:
            problems.append(f"weights gone: {name}")
            continue
        leaves.append((name, leaf_digest(name, m["adapter_digest"],
                                         m["data_digest"])))
    present = {name for name, _ in leaves}
    for name in by_name:
        if name not in present:
            problems.append(f"adapter in the root is missing: {name}")

    leaves.sort(key=lambda pair: pair[0])
    rebuilt = merkle_root([leaf for _, leaf in leaves])
    if not hmac.compare_digest(rebuilt, claimed):
        for name, leaf in leaves:
            if name in by_name and by_name[name]["leaf"] != leaf:
                problems.append(f"changed since the root was built: {name}")
        if not problems:
            problems.append("root does not match the adapters on disk")

    if problems:
        print(f"  ROOT BROKEN  (built {doc.get('built')})")
        for problem in problems:
            print(f"      {problem}")
        return 1
    print(f"  root {claimed[:16]} intact over {len(leaves)} adapter(s) "
          f"(built {doc.get('built')})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["seal", "verify", "seal-all", "verify-all",
                                       "challenge", "respond", "check",
                                       "root", "verify-root"])
    ap.add_argument("folder", nargs="?", help="an Adapter_skills/<NAME> folder")
    ap.add_argument("--nonce")
    ap.add_argument("--proof")
    ap.add_argument("--integrity-only", action="store_true",
                    help="seal weights with no training data beside them")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    if args.action == "root":
        return build_root(Path(args.root))
    if args.action == "verify-root":
        return verify_root(Path(args.root))

    if args.action in ("seal-all", "verify-all"):
        folders = _folders(Path(args.root))
        if not folders:
            print("Nothing in Adapter_skills/. Run archive_adapters.py first.")
            return 1
        root_doc = load_root_doc(Path(args.root))
        fn = (seal if args.action == "seal-all"
              else lambda f: verify(f, root_doc=root_doc,
                                    project_root=Path(args.root)))
        bad = sum(fn(f) != 0 for f in folders)
        print(f"\n  {len(folders) - bad}/{len(folders)} ok")
        return 1 if bad else 0

    if not args.folder:
        print(f"{args.action} needs a folder")
        return 1
    folder = Path(args.folder)
    if args.action == "seal":
        return seal(folder, integrity_only=args.integrity_only)
    if args.action == "verify":
        return verify(folder, root_doc=load_root_doc(Path(args.root)),
                      project_root=Path(args.root))
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
