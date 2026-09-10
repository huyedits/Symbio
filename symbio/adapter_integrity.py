"""Check an adapter against its seal before the model is given it.

adapter_seal.py could always prove an adapter was the bytes it was sealed as.
Nothing ever asked it: it is a CLI you have to remember to run, and no code
path in symbio/ referenced it. A seal nobody checks catches nothing, and the
failures this system actually has are the kind a check at load would catch —
the half-written adapter an OOM kill leaves behind, an archive/restore that
copied a truncated file, weights swapped under a config that still claims to
match.

The implementation is not duplicated here. adapter_seal.py lives at the
project root and imports nothing from symbio, which is deliberate: it runs
under a bare `python3` on a machine with no MLX, and importing it as
`symbio.adapter_seal` would drag the whole inference stack in through
symbio/__init__.py. So it is loaded from its file, once, and its absence is a
missing feature rather than an error — an installed copy of the package that
ships no root scripts simply cannot verify, and says so if asked.

Policy lives in agent.verify_adapters:

    "warn"    (default) say so and load anyway
    "refuse"  do not hand tampered weights to the model
    "off"     do not look

The default is warn, not refuse, because the most common cause of a mismatch
is the innocent one: an adapter retrained since it was last sealed. Refusing
that by default would make every training run end in a broken session. Training
re-seals itself for exactly this reason, which is what makes "refuse" usable
for anyone who wants it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_seal_module: Any = None
_seal_tried = False


def _seal():
    """adapter_seal.py as a module, or None if it is not in this install."""
    global _seal_module, _seal_tried
    if _seal_tried:
        return _seal_module
    _seal_tried = True
    from symbio import constants

    path = constants.PROJECT_DIR / "adapter_seal.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("symbio_adapter_seal", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # Registered before exec so a module that imports itself by name (or a
        # traceback that wants to read its source) finds it.
        sys.modules["symbio_adapter_seal"] = module
        spec.loader.exec_module(module)
        _seal_module = module
    except Exception:
        _seal_module = None
    return _seal_module


def policy(config: dict | None) -> str:
    """The configured policy, falling back to the config file on disk.

    Most callers of load() do not pass a config — they never needed one — so a
    policy read only from the argument would leave "refuse" unreachable for
    every path but the two that happen to thread it through. config.json is
    read directly rather than through load_config(): this runs inside a model
    load, and the full loader merges defaults, applies env overrides and can
    write the file back, none of which belongs under a tripwire.
    """
    value = (config or {}).get("agent", {}).get("verify_adapters")
    if value is None:
        value = _policy_on_disk()
    value = str(value).strip().lower()
    return value if value in ("warn", "refuse", "off") else "warn"


_disk_policy: str | None = None


def _policy_on_disk() -> str:
    global _disk_policy
    if _disk_policy is not None:
        return _disk_policy
    from symbio import constants

    _disk_policy = "warn"
    try:
        import json

        raw = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
        _disk_policy = str(raw.get("agent", {}).get("verify_adapters", "warn"))
    except Exception:
        pass
    return _disk_policy


def check_adapter(adapter_path: str | Path | None) -> dict | None:
    """Report on the adapter directory, or None when there is nothing to say.

    None covers every "no claim was made" case — no path, no such directory,
    no seal, no adapter_seal.py to check with — because they are all the same
    answer: this adapter has never been vouched for, which is not the same as
    it having failed.
    """
    if not adapter_path:
        return None
    folder = Path(adapter_path)
    if not folder.is_dir():
        return None
    module = _seal()
    if module is None:
        return None
    from symbio import constants

    try:
        # The root is read through the module, not cached here: `root` can be
        # rebuilt between two loads in the same session.
        report = module.check(folder, module.load_root_doc(constants.PROJECT_DIR),
                              constants.PROJECT_DIR)
    except Exception:
        return None
    # "stale-format" is a seal this version cannot read, which is the same
    # kind of answer as "unsealed": no claim it can check, so no claim to
    # contradict. The CLI says re-seal; a load has nothing to report.
    return (None if report.get("state") in ("unsealed", "stale-format")
            else report)


def signature_problem(config: dict | None) -> str | None:
    """Why the root's signature is unacceptable, or None if it is fine.

    Silent until an identity is configured. A signature check with nothing to
    check against would warn on every install that has not set one up, and a
    check that fires when nothing is wrong is one people switch off — which
    would take the real check with it.

    This is the only check here an attacker with write access cannot satisfy.
    Everything else reads files that sit beside the adapters: re-sealing over
    tampered weights and rebuilding the root makes both of them pass, verified
    live. The private half of the signing key is not on that disk.
    """
    agent_cfg = (config or {}).get("agent", {})
    identity = str(agent_cfg.get("adapter_signer_identity", "") or "").strip()
    if not identity:
        return None
    module = _seal()
    if module is None:
        return None
    from symbio import constants

    signers = str(agent_cfg.get("adapter_signers", "~/.ssh/allowed_signers"))
    try:
        ok, detail = module.verify_root_signature(
            constants.PROJECT_DIR, signers, identity)
    except Exception as e:
        return f"the root's signature could not be checked ({e})"
    return None if ok else (detail or "the root's signature did not verify")


def describe(report: dict) -> str:
    """One block of text for a person, naming what moved."""
    lines = [f"  [Adapter] {report['name']}: SEAL BROKEN"]
    for problem in report.get("problems", []):
        lines.append(f"      {problem}")
    lines.append("      These are not the weights that were sealed. If you "
                 "retrained since, re-seal:")
    lines.append("        python3 adapter_seal.py seal <dir> && "
                 "python3 adapter_seal.py root")
    return "\n".join(lines)


def enforce(adapter_path: str | Path | None, config: dict | None,
            output_fn=print) -> None:
    """Check before a load, and act on the configured policy.

    Raises RuntimeError under "refuse". Everything else — unsealed, intact,
    unreadable, no checker available — passes through silently: this is a
    tripwire, and a tripwire that talks when nothing happened gets unplugged.
    """
    if policy(config) == "off":
        return

    # The signature first: it covers every adapter at once, and it is the one
    # thing a tamperer cannot forge from the disk.
    problem = signature_problem(config)
    if problem is not None:
        output_fn(
            f"  [Adapter] The signed root does not verify: {problem}\n"
            f"      Re-sign after a legitimate change:\n"
            f"        python3 adapter_seal.py root && "
            f"python3 adapter_seal.py sign-root --key <your key>")
        if policy(config) == "refuse":
            raise RuntimeError(f"adapters_root.json failed its signature: {problem}")

    report = check_adapter(adapter_path)
    if report is None or report.get("state") == "intact":
        return
    if report.get("state") == "unreadable":
        return
    output_fn(describe(report))
    if policy(config) == "refuse":
        raise RuntimeError(
            f"adapter {report['name']} failed its seal: "
            f"{'; '.join(report.get('problems', [])) or 'unknown'}")
