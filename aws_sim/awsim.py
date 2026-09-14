#!/usr/bin/env python3
"""A rigid cloud CLI that is deliberately NOT the AWS CLI.

The first version of this file mimicked AWS, and the experiment said so
immediately: the 14B scored 13/13 on the held-out battery before a single step
of training, and solved 60 of 60 self-teaching tasks on the first attempt. It
was not learning anything. It was reciting `aws s3 mb` from pretraining.

So every surface here is chosen to be internally consistent and externally
unfamiliar. Pretrained knowledge does not just fail to help, it actively
misleads:

    aws  s3  mb s3://bucket              awsim store new-container
                                             --name bucket --region ...
    aws  ec2 describe-instances              awsim compute list-nodes
         --filters Name=tag:Env,Values=prod      --where tag/Env:prod --region ...
    aws  iam create-role --role-name R        awsim access new-identity
                                             --identity R --region ...

Nothing is guessable. `--region` is required on every single command, there is
no URI scheme, tags are `Key:Value` and filters are `tag/Key:Value`.

The one way in is FAILURE. Every error names exactly what was allowed at that
point — the services, the verbs of a service, the options of a verb — so a
model that probes can reconstruct the whole API from its own mistakes in a
handful of attempts. That is the channel this environment teaches through, and
it is the only one: there is no help text in the system prompt.

State is one JSON file so a run can be reset between graded attempts. Nothing
here talks to a cloud, has credentials, or touches the network.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent / "state.json"

_EMPTY = {"containers": {}, "nodes": {}, "identities": {}}

REGIONS = ("ap-1", "eu-2", "us-3")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}[a-z0-9]$")


class CommandError(Exception):
    """A rigid, specific failure. The message is the entire teaching signal."""


# Every verb of every service, rendered for an error. Appended whenever a
# service or a verb is not recognised.
#
# The first version listed only the CURRENT service's verbs, and the traces
# showed exactly what that costs: asked to create a container, the model tried
# `awsim create-container`, was told the three services, guessed `compute`,
# and was then told compute's four verbs — none of which make containers. It
# never tried `store`. Every message was locally accurate and kept it inside
# the wrong service, and the run finished with 0 samples for new-container,
# duplicate and list-containers — the exact three cases the trained model then
# failed forever.
#
# Rigid is about refusing wrong input, not about rationing information. A
# model that is lost gets the whole map.
def verb_map() -> str:
    return " | ".join(
        f"{service}: {', '.join(sorted(verbs))}"
        for service, verbs in sorted(VERBS.items()))


VERBS = {
    "store": {"new-container", "drop-container", "list-containers", "put",
              "list", "drop", "duplicate"},
    "compute": {"list-nodes", "halt", "resume", "label"},
    "access": {"new-identity", "list-identities", "drop-identity"},
}


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


def _flags(args: list[str], verb: str, *, allowed: set[str],
           required: set[str]) -> dict:
    """Parse `--flag value` pairs strictly, and explain every refusal fully.

    The listing in each message is not decoration. It is how a model that has
    never seen this API discovers it: one wrong option returns the complete
    set of right ones.
    """
    allowed = allowed | {"region"}
    required = required | {"region"}
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise CommandError(
                f"'{verb}' takes no positional arguments; got '{token}'. "
                f"Every value is passed as --option value. Options for "
                f"'{verb}': {', '.join('--' + a for a in sorted(allowed))}.")
        name = token[2:]
        if name not in allowed:
            raise CommandError(
                f"'{verb}' has no option '--{name}'. Options for '{verb}': "
                f"{', '.join('--' + a for a in sorted(allowed))}.")
        if i + 1 >= len(args) or args[i + 1].startswith("--"):
            raise CommandError(f"Option '--{name}' requires a value.")
        out[name] = args[i + 1]
        i += 2
    missing = sorted(required - set(out))
    if missing:
        raise CommandError(
            f"'{verb}' is missing required option(s): "
            f"{', '.join('--' + m for m in missing)}. Every command needs "
            f"--region (one of {', '.join(REGIONS)}).")
    if out["region"] not in REGIONS:
        raise CommandError(
            f"Unknown region '{out['region']}'. Valid regions: "
            f"{', '.join(REGIONS)}.")
    return out


def _ckey(region: str, name: str) -> str:
    """A container's key. A plain separator rather than a stringified tuple:
    the first draft read those back with eval(), which is a habit that costs
    nothing until the day the data is not your own."""
    return f"{region}|{name}"


# ------------------------------------------------------------------- store

def _store(args, state):
    verbs = VERBS["store"]
    if not args or args[0] not in verbs:
        # Parenthesised, unlike the first draft: without them the ternary bound
        # so that the verb listing was appended to the "needs a verb" branch
        # ONLY, and a model that guessed a wrong verb — the overwhelmingly
        # common case — got "Unknown store verb 'mb'" with no hint of what the
        # right ones were. That is the discovery channel, and it was closed for
        # exactly the case it exists to serve.
        raise CommandError(
            (f"Unknown store verb {args[0]!r} " if args else "store needs a verb ")
            + f"— every verb, by service: {verb_map()}")
    verb, rest = args[0], args[1:]

    if verb == "new-container":
        f = _flags(rest, verb, allowed={"name"}, required={"name"})
        if not _NAME.match(f["name"]):
            raise CommandError(f"InvalidName: '{f['name']}' is not a valid container name.")
        key = _ckey(f["region"], f["name"])
        if key in state["containers"]:
            raise CommandError(f"ContainerExists: {f['name']} in {f['region']}")
        state["containers"][key] = {}
        return f"container created: {f['name']} ({f['region']})"

    if verb == "list-containers":
        f = _flags(rest, verb, allowed=set(), required=set())
        names = sorted(k.split("|", 1)[1] for k in state["containers"]
                       if k.split("|", 1)[0] == f["region"])
        return "\n".join(names) or "(none)"

    if verb == "drop-container":
        f = _flags(rest, verb, allowed={"name"}, required={"name"})
        key = _ckey(f["region"], f["name"])
        if key not in state["containers"]:
            raise CommandError(f"NoSuchContainer: {f['name']} in {f['region']}")
        if state["containers"][key]:
            raise CommandError(
                f"ContainerNotEmpty: {f['name']} holds "
                f"{len(state['containers'][key])} item(s); drop them first.")
        del state["containers"][key]
        return f"container dropped: {f['name']}"

    if verb == "put":
        f = _flags(rest, verb, allowed={"container", "path", "from"},
                   required={"container", "path", "from"})
        key = _ckey(f["region"], f["container"])
        if key not in state["containers"]:
            raise CommandError(f"NoSuchContainer: {f['container']} in {f['region']}")
        state["containers"][key][f["path"]] = f"<{f['from']}>"
        return f"stored: {f['path']} in {f['container']}"

    if verb == "list":
        f = _flags(rest, verb, allowed={"container"}, required={"container"})
        key = _ckey(f["region"], f["container"])
        if key not in state["containers"]:
            raise CommandError(f"NoSuchContainer: {f['container']} in {f['region']}")
        return "\n".join(sorted(state["containers"][key])) or "(empty)"

    if verb == "drop":
        f = _flags(rest, verb, allowed={"container", "path"},
                   required={"container", "path"})
        key = _ckey(f["region"], f["container"])
        if key not in state["containers"]:
            raise CommandError(f"NoSuchContainer: {f['container']} in {f['region']}")
        if f["path"] not in state["containers"][key]:
            raise CommandError(f"NoSuchPath: {f['path']}")
        del state["containers"][key][f["path"]]
        return f"dropped: {f['path']}"

    if verb == "duplicate":
        f = _flags(rest, verb,
                   allowed={"from-container", "from-path", "to-container", "to-path"},
                   required={"from-container", "from-path", "to-container", "to-path"})
        src = _ckey(f["region"], f["from-container"])
        dst = _ckey(f["region"], f["to-container"])
        if src not in state["containers"]:
            raise CommandError(f"NoSuchContainer: {f['from-container']}")
        if f["from-path"] not in state["containers"][src]:
            raise CommandError(f"NoSuchPath: {f['from-path']}")
        if dst not in state["containers"]:
            raise CommandError(f"NoSuchContainer: {f['to-container']}")
        state["containers"][dst][f["to-path"]] = state["containers"][src][f["from-path"]]
        return f"duplicated to {f['to-container']}/{f['to-path']}"
    raise CommandError(f"Unhandled store verb {verb!r}")


# ----------------------------------------------------------------- compute

def _compute(args, state):
    verbs = VERBS["compute"]
    if not args or args[0] not in verbs:
        raise CommandError(
            (f"Unknown compute verb {args[0]!r} " if args else "compute needs a verb ")
            + f"— every verb, by service: {verb_map()}")
    verb, rest = args[0], args[1:]

    if verb == "list-nodes":
        f = _flags(rest, verb, allowed={"where"}, required=set())
        rows = [(n, d) for n, d in state["nodes"].items()
                if d["region"] == f["region"]]
        if "where" in f:
            m = re.fullmatch(r"tag/(?P<k>[\w-]+):(?P<v>[\w-]+)", f["where"])
            if not m:
                raise CommandError(
                    "--where must be written as tag/<Key>:<Value>, "
                    "for example tag/Env:prod.")
            rows = [(n, d) for n, d in rows
                    if d.get("labels", {}).get(m.group("k")) == m.group("v")]
        return json.dumps({"nodes": [
            {"node": n, "status": d["status"],
             "labels": d.get("labels", {})} for n, d in sorted(rows)]}, indent=2)

    if verb in ("halt", "resume"):
        f = _flags(rest, verb, allowed={"node"}, required={"node"})
        if f["node"] not in state["nodes"]:
            raise CommandError(f"NoSuchNode: {f['node']}")
        state["nodes"][f["node"]]["status"] = "halted" if verb == "halt" else "up"
        return f"{f['node']} is now {state['nodes'][f['node']]['status']}"

    if verb == "label":
        f = _flags(rest, verb, allowed={"node", "label"},
                   required={"node", "label"})
        m = re.fullmatch(r"(?P<k>[\w-]+):(?P<v>[\w-]+)", f["label"])
        if not m:
            raise CommandError(
                "--label must be written as <Key>:<Value>, for example Owner:sre.")
        if f["node"] not in state["nodes"]:
            raise CommandError(f"NoSuchNode: {f['node']}")
        state["nodes"][f["node"]].setdefault("labels", {})[m.group("k")] = m.group("v")
        return f"labelled {f['node']} {m.group('k')}:{m.group('v')}"
    raise CommandError(f"Unhandled compute verb {verb!r}")


# ------------------------------------------------------------------ access

def _access(args, state):
    verbs = VERBS["access"]
    if not args or args[0] not in verbs:
        raise CommandError(
            (f"Unknown access verb {args[0]!r} " if args else "access needs a verb ")
            + f"— every verb, by service: {verb_map()}")
    verb, rest = args[0], args[1:]

    if verb == "new-identity":
        f = _flags(rest, verb, allowed={"identity"}, required={"identity"})
        if f["identity"] in state["identities"]:
            raise CommandError(f"IdentityExists: {f['identity']}")
        state["identities"][f["identity"]] = {"region": f["region"]}
        return f"identity created: {f['identity']}"

    if verb == "list-identities":
        _flags(rest, verb, allowed=set(), required=set())
        return "\n".join(sorted(state["identities"])) or "(none)"

    if verb == "drop-identity":
        f = _flags(rest, verb, allowed={"identity"}, required={"identity"})
        if f["identity"] not in state["identities"]:
            raise CommandError(f"NoSuchIdentity: {f['identity']}")
        del state["identities"][f["identity"]]
        return f"identity dropped: {f['identity']}"
    raise CommandError(f"Unhandled access verb {verb!r}")


SERVICES = {"store": _store, "compute": _compute, "access": _access}


def run(argv: list[str]) -> tuple[int, str]:
    if not argv:
        return 2, (f"awsim needs a service. Services: "
                   f"{', '.join(sorted(SERVICES))}. "
                   f"Usage: awsim <service> <verb> --option value ...")
    service, rest = argv[0], argv[1:]
    if service not in SERVICES:
        return 2, (f"error: Unknown service '{service}'. "
                   f"Every verb, by service: {verb_map()}")
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
