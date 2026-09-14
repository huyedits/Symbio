#!/usr/bin/env python3
"""A deliberately rigid mock of a small AWS CLI surface.

This exists to be LEARNED, so everything about it is chosen to make learning
measurable rather than to be forgiving:

  * Exact syntax or nothing. `--role-name` is not `--rolename`, `s3://b/k` is
    not `s3:/b/k`, and a missing required flag is an error naming the flag. A
    forgiving environment teaches nothing, because every near-miss is
    reinforced as success.
  * Stateful. `mb` then `ls` shows the bucket; `rb` on a non-empty bucket
    fails the way the real one does. So a command can be graded on what it DID
    to the world, not on whether its text looked plausible.
  * Deterministic. Same state plus same command gives the same bytes, so a
    checkpoint that scores better really is better and not luckier.
  * Every error is a precise, actionable sentence. The point of a rigid
    environment is that being wrong is informative.

State lives in one JSON file so a run can be reset between graded attempts.
Nothing here talks to AWS, has credentials, or touches the network.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent / "state.json"

_EMPTY = {"buckets": {}, "instances": {}, "roles": {}}

_S3_URI = re.compile(r"^s3://(?P<bucket>[a-z0-9][a-z0-9.-]{1,61}[a-z0-9])(?:/(?P<key>.*))?$")
_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class CommandError(Exception):
    """A rigid, specific failure. The message is the teaching signal."""


def load_state() -> dict:
    if not STATE_FILE.exists():
        return json.loads(json.dumps(_EMPTY))
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(_EMPTY))
    for key, default in _EMPTY.items():
        data.setdefault(key, json.loads(json.dumps(default)))
    return data


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def reset_state(seed: dict | None = None) -> dict:
    state = json.loads(json.dumps(seed if seed is not None else _EMPTY))
    for key, default in _EMPTY.items():
        state.setdefault(key, json.loads(json.dumps(default)))
    save_state(state)
    return state


# ----------------------------------------------------------------- argument parsing

def _flags(args: list[str], *, allowed: set[str], required: set[str],
           boolean: set[str] = frozenset()) -> dict:
    """Parse `--flag value` pairs strictly.

    Strictly means: an unknown flag is an error that names it and lists what
    was allowed, a flag without its value is an error, and a positional
    argument where a flag belongs is an error. Real CLIs do this, and a mock
    that shrugs teaches a syntax that will not work anywhere else.
    """
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise CommandError(
                f"Unexpected argument '{token}'. Options must be written as "
                f"--flag value. Allowed here: {', '.join(sorted(allowed))}.")
        name = token[2:]
        if name not in allowed:
            raise CommandError(
                f"Unknown option '--{name}'. Allowed here: "
                f"{', '.join('--' + a for a in sorted(allowed))}.")
        if name in boolean:
            out[name] = "true"
            i += 1
            continue
        if i + 1 >= len(args) or args[i + 1].startswith("--"):
            raise CommandError(f"Option '--{name}' requires a value.")
        out[name] = args[i + 1]
        i += 2
    missing = sorted(required - set(out))
    if missing:
        raise CommandError(
            "Missing required option(s): "
            + ", ".join("--" + m for m in missing) + ".")
    return out


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI.match(uri)
    if not match:
        raise CommandError(
            f"'{uri}' is not a valid S3 URI. Expected s3://bucket/key, with a "
            f"lowercase bucket name of 3-63 characters.")
    return match.group("bucket"), match.group("key") or ""


# ------------------------------------------------------------------------- s3

def _s3(args: list[str], state: dict) -> str:
    if not args:
        raise CommandError("Usage: awsim s3 <ls|mb|rb|cp|rm> ...")
    op, rest = args[0], args[1:]

    if op == "ls":
        if not rest:
            return "\n".join(sorted(state["buckets"])) or "(no buckets)"
        bucket, prefix = _parse_s3_uri(rest[0])
        if bucket not in state["buckets"]:
            raise CommandError(f"NoSuchBucket: {bucket}")
        keys = sorted(k for k in state["buckets"][bucket] if k.startswith(prefix))
        return "\n".join(keys) or "(empty)"

    if op == "mb":
        if not rest:
            raise CommandError("Usage: awsim s3 mb s3://bucket")
        bucket, key = _parse_s3_uri(rest[0])
        if key:
            raise CommandError("mb takes a bucket, not a key: use s3://bucket")
        if not _BUCKET_NAME.match(bucket):
            raise CommandError(f"InvalidBucketName: {bucket}")
        if bucket in state["buckets"]:
            raise CommandError(f"BucketAlreadyExists: {bucket}")
        state["buckets"][bucket] = {}
        return f"make_bucket: {bucket}"

    if op == "rb":
        if not rest:
            raise CommandError("Usage: awsim s3 rb s3://bucket")
        bucket, _key = _parse_s3_uri(rest[0])
        if bucket not in state["buckets"]:
            raise CommandError(f"NoSuchBucket: {bucket}")
        if state["buckets"][bucket]:
            raise CommandError(
                f"BucketNotEmpty: {bucket} still holds "
                f"{len(state['buckets'][bucket])} object(s). Remove them first.")
        del state["buckets"][bucket]
        return f"remove_bucket: {bucket}"

    if op == "cp":
        if len(rest) < 2:
            raise CommandError("Usage: awsim s3 cp <source> <destination>")
        source, destination = rest[0], rest[1]
        if source.startswith("s3://") and destination.startswith("s3://"):
            sb, sk = _parse_s3_uri(source)
            db, dk = _parse_s3_uri(destination)
            if sb not in state["buckets"]:
                raise CommandError(f"NoSuchBucket: {sb}")
            if sk not in state["buckets"][sb]:
                raise CommandError(f"NoSuchKey: {sk}")
            if db not in state["buckets"]:
                raise CommandError(f"NoSuchBucket: {db}")
            if not dk:
                raise CommandError("A destination key is required: s3://bucket/key")
            state["buckets"][db][dk] = state["buckets"][sb][sk]
            return f"copy: {source} to {destination}"
        if destination.startswith("s3://"):
            db, dk = _parse_s3_uri(destination)
            if db not in state["buckets"]:
                raise CommandError(f"NoSuchBucket: {db}")
            if not dk:
                raise CommandError("A destination key is required: s3://bucket/key")
            state["buckets"][db][dk] = f"<contents of {source}>"
            return f"upload: {source} to {destination}"
        raise CommandError(
            "Downloads are not simulated. One side must be an s3:// URI.")

    if op == "rm":
        if not rest:
            raise CommandError("Usage: awsim s3 rm s3://bucket/key")
        bucket, key = _parse_s3_uri(rest[0])
        if bucket not in state["buckets"]:
            raise CommandError(f"NoSuchBucket: {bucket}")
        if not key:
            raise CommandError("rm needs a key: s3://bucket/key")
        if key not in state["buckets"][bucket]:
            raise CommandError(f"NoSuchKey: {key}")
        del state["buckets"][bucket][key]
        return f"delete: s3://{bucket}/{key}"

    raise CommandError(f"Unknown s3 operation '{op}'. Use ls, mb, rb, cp or rm.")


# ------------------------------------------------------------------------ ec2

def _ec2(args: list[str], state: dict) -> str:
    if not args:
        raise CommandError(
            "Usage: awsim ec2 <describe-instances|start-instances|"
            "stop-instances|create-tags> ...")
    op, rest = args[0], args[1:]

    if op == "describe-instances":
        flags = _flags(rest, allowed={"filters", "instance-ids"}, required=set())
        rows = list(state["instances"].items())
        if "instance-ids" in flags:
            wanted = set(flags["instance-ids"].split(","))
            rows = [(i, d) for i, d in rows if i in wanted]
        if "filters" in flags:
            spec = flags["filters"]
            m = re.fullmatch(r"Name=tag:(?P<tag>[\w:-]+),Values=(?P<values>.+)", spec)
            if not m:
                raise CommandError(
                    "--filters must be written exactly as "
                    "Name=tag:<TagName>,Values=<v1>[,<v2>...]")
            tag, values = m.group("tag"), set(m.group("values").split(","))
            rows = [(i, d) for i, d in rows if d.get("tags", {}).get(tag) in values]
        payload = {"Reservations": [
            {"Instances": [{"InstanceId": i, "State": {"Name": d["state"]},
                            "Tags": [{"Key": k, "Value": v}
                                     for k, v in sorted(d.get("tags", {}).items())]}]}
            for i, d in sorted(rows)]}
        return json.dumps(payload, indent=2)

    if op in ("start-instances", "stop-instances"):
        flags = _flags(rest, allowed={"instance-ids"}, required={"instance-ids"})
        target = "running" if op == "start-instances" else "stopped"
        changed = []
        for instance_id in flags["instance-ids"].split(","):
            if instance_id not in state["instances"]:
                raise CommandError(f"InvalidInstanceID.NotFound: {instance_id}")
            state["instances"][instance_id]["state"] = target
            changed.append(instance_id)
        return json.dumps({"Instances": [
            {"InstanceId": i, "CurrentState": {"Name": target}} for i in changed]},
            indent=2)

    if op == "create-tags":
        flags = _flags(rest, allowed={"resources", "tags"},
                       required={"resources", "tags"})
        m = re.fullmatch(r"Key=(?P<k>[\w:-]+),Value=(?P<v>.*)", flags["tags"])
        if not m:
            raise CommandError(
                "--tags must be written exactly as Key=<Name>,Value=<Value>")
        for instance_id in flags["resources"].split(","):
            if instance_id not in state["instances"]:
                raise CommandError(f"InvalidInstanceID.NotFound: {instance_id}")
            state["instances"][instance_id].setdefault("tags", {})[m.group("k")] = m.group("v")
        return ""

    raise CommandError(f"Unknown ec2 operation '{op}'.")


# ------------------------------------------------------------------------ iam

def _iam(args: list[str], state: dict) -> str:
    if not args:
        raise CommandError("Usage: awsim iam <create-role|list-roles|delete-role> ...")
    op, rest = args[0], args[1:]

    if op == "create-role":
        flags = _flags(rest, allowed={"role-name", "description"},
                       required={"role-name"})
        name = flags["role-name"]
        if not re.fullmatch(r"[\w+=,.@-]{1,64}", name):
            raise CommandError(f"ValidationError: invalid role name '{name}'")
        if name in state["roles"]:
            raise CommandError(f"EntityAlreadyExists: role {name} already exists")
        state["roles"][name] = {"description": flags.get("description", "")}
        return json.dumps({"Role": {"RoleName": name, "Arn":
                                    f"arn:aws:iam::000000000000:role/{name}"}},
                          indent=2)

    if op == "list-roles":
        _flags(rest, allowed=set(), required=set())
        return json.dumps({"Roles": [
            {"RoleName": n, "Arn": f"arn:aws:iam::000000000000:role/{n}"}
            for n in sorted(state["roles"])]}, indent=2)

    if op == "delete-role":
        flags = _flags(rest, allowed={"role-name"}, required={"role-name"})
        if flags["role-name"] not in state["roles"]:
            raise CommandError(f"NoSuchEntity: role {flags['role-name']} not found")
        del state["roles"][flags["role-name"]]
        return ""

    raise CommandError(f"Unknown iam operation '{op}'.")


SERVICES = {"s3": _s3, "ec2": _ec2, "iam": _iam}


def run(argv: list[str]) -> tuple[int, str]:
    """Execute one command. Returns (exit_code, output)."""
    if not argv:
        return 2, "Usage: awsim <s3|ec2|iam> <operation> [options]"
    service, rest = argv[0], argv[1:]
    if service not in SERVICES:
        return 2, (f"Unknown service '{service}'. "
                   f"Available: {', '.join(sorted(SERVICES))}.")
    state = load_state()
    try:
        output = SERVICES[service](rest, state)
    except CommandError as exc:
        return 1, f"error: {exc}"
    save_state(state)
    return 0, output


def main() -> int:
    code, output = run(sys.argv[1:])
    if output:
        print(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
