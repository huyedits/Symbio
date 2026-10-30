"""Weights that change for their person: folding what was learned into the base,
and taking it back out.

A LoRA adapter is a separate layer that has to be loaded on top of the base
every time. Each new training round replaces it, so whatever the last round
learned is only as durable as the next round's corpus. Binding makes it part of
the base instead: the adapter's low-rank update is added into the base weights
once and saved as a new version of the model, and the next round trains a
fresh adapter ON TOP of it. Unbinding is the reverse: rebuild the base without
one earlier update.

Nobody runs a command for this. At the start of a training round
(chat._guarded_train) the model is shown the candidates and decides for itself
(semif.choose, one forward pass, a close call is a no). What it cannot decide
is whether the change was good: every bind or unbind is a trial that has to
pass the golden battery before it is kept, and is rolled back if it does not.

Nothing learned is deleted. Before an adapter is folded in, a copy goes to
weights/archive/<base>/<id>/, and the ledger records the order in which
deltas were folded. The materialized model under weights/<base>/ is a cache:
it can always be rebuilt from the original base plus the archived deltas, so
an old version's directory is removed when a new one replaces it.

Only plain LoRA on dense linear layers can be folded. DoRA, full fine-tunes and
MoE switch layers are refused rather than approximated.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio import constants

DEFAULTS = {
    "self_reshape": True,       # the model may bind / unbind on its own
    "bind_min_age_days": 3,     # an adapter must have been in use this long
    "min_margin": 0.2,          # semif: a closer call than this is a no
    "fold_bits": 8,             # width a folded quantized layer is saved at
}


def settings(config: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULTS, **((config or {}).get("weights") or {})}


def _dir() -> Path:
    return constants.WEIGHTS_DIR


def _slug(base: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")


def _ledger_path() -> Path:
    return _dir() / "ledger.json"


def load_ledger() -> dict[str, Any]:
    try:
        return json.loads(_ledger_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_ledger(ledger: dict[str, Any]) -> None:
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def entry(base: str, ledger: dict[str, Any] | None = None) -> dict[str, Any]:
    ledger = load_ledger() if ledger is None else ledger
    e = ledger.get(base) or {}
    return {"bound": list(e.get("bound", [])), "path": e.get("path"),
            "events": list(e.get("events", []))}


# A trial version under test, by base. Set only while a gate is running, so
# the trial loads through the same path every real load takes.
_TRIAL: dict[str, str | None] = {}


def resolve(model_name: str) -> str:
    """What to actually load for `model_name`: its bound version if it has
    one, else the name unchanged. Every load and every trainer goes through
    this, so the name in config.json stays the user's chosen model while the
    bytes underneath carry what it has learned."""
    if not isinstance(model_name, str):
        return model_name
    if model_name in _TRIAL:
        return _TRIAL[model_name] or model_name
    path = (load_ledger().get(model_name) or {}).get("path")
    if path and (Path(path) / "config.json").is_file():
        return path
    return model_name


def adapter_base(adapter_dir: Path) -> str | None:
    """The model an adapter was trained on, as its adapter_config says."""
    try:
        cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return cfg.get("model")


def foldable(adapter_dir: Path) -> str | None:
    """None when the adapter can be folded into its base, else why not."""
    adapter_dir = Path(adapter_dir)
    try:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "no readable adapter_config.json"
    if not (adapter_dir / "adapters.safetensors").is_file():
        return "no adapters.safetensors"
    kind = cfg.get("fine_tune_type", "lora")
    if kind != "lora":
        return f"a {kind} adapter cannot be folded as a low-rank update"
    return None


def archive_adapter(adapter_dir: Path, base: str, label: str) -> tuple[str, Path]:
    """Copy an adapter into the archive; returns (delta id, archived path)."""
    did = f"{label}-{datetime.now():%Y%m%d-%H%M%S}"
    dest = _dir() / "archive" / _slug(base) / did
    dest.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_config.json", "adapters.safetensors"):
        shutil.copy2(Path(adapter_dir) / name, dest / name)
    return did, dest


# ---- folding (runs in a child process: it holds a whole base model) ----

def lora_deltas(adapter_dir: Path) -> dict[str, Any]:
    """{module path: dense update} for one adapter, as mlx arrays."""
    from symbio.mlx_gate import attr as _mlx
    mx = _mlx("mlx.core")
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = float((cfg.get("lora_parameters") or {}).get("scale", 20.0))
    tensors = mx.load(str(Path(adapter_dir) / "adapters.safetensors"))
    out = {}
    for key, a in tensors.items():
        if not key.endswith(".lora_a"):
            continue
        path = key[: -len(".lora_a")]
        b = tensors.get(path + ".lora_b")
        if b is None or a.ndim != 2:
            raise ValueError(f"{path}: not a dense LoRA pair (MoE switch layers are not foldable)")
        # Same arithmetic as mlx_lm's LoRALinear.fuse.
        out[path] = (scale * b.T) @ a.T
    return out


def fold(model: Any, adapter_dirs: list[Path], bits: int = 8) -> dict[str, Any]:
    """Add every adapter's update into the model's linear weights, in order.

    A quantized layer that changes is requantized ONCE, at `bits` (8 by
    default), not at the base's own width. Measured on Qwen3-4B-4bit: putting
    a folded layer back to 4-bit moved the logits by 1.19 against the 0.98 the
    update itself moved them — rounding to the 4-bit grid costs more than a
    LoRA update is worth, so the fold would mostly erase what it was keeping.
    Only the layers an adapter touched pay for the extra width.

    Returns {module path: quantization params, or None for a dense layer}
    for every module that changed; the params go into the saved config so the
    loader rebuilds each layer at the width it was written at.
    """
    from symbio.mlx_gate import attr as _mlx
    mx = _mlx("mlx.core")
    nn = _mlx("mlx.nn")
    tree_unflatten = _mlx("mlx.utils.tree_unflatten")
    total: dict[str, Any] = {}
    for d in adapter_dirs:
        for path, delta in lora_deltas(d).items():
            total[path] = delta if path not in total else total[path] + delta
    modules = dict(model.named_modules())
    new, layout = [], {}
    for path, delta in total.items():
        lin = modules.get(path)
        if lin is None:
            raise ValueError(f"{path}: no such module in the base model")
        if hasattr(lin, "lora_a"):   # loaded with a live adapter: fold under it
            lin = lin.linear
        quant = isinstance(lin, nn.QuantizedLinear)
        w = lin.weight
        if quant:
            w = mx.dequantize(w, lin.scales, getattr(lin, "biases", None),
                              group_size=lin.group_size, bits=lin.bits,
                              mode=getattr(lin, "mode", "affine"))
        if tuple(delta.shape) != tuple(w.shape):
            raise ValueError(f"{path}: update {tuple(delta.shape)} does not fit weight {tuple(w.shape)}")
        has_bias = "bias" in lin
        out_dims, in_dims = w.shape
        dense = nn.Linear(in_dims, out_dims, bias=has_bias)
        dense.weight = w + delta.astype(w.dtype)
        if has_bias:
            dense.bias = lin.bias
        params = None
        if quant:
            params = {"group_size": 64, "bits": max(int(bits), int(lin.bits)), "mode": "affine"}
            dense = nn.QuantizedLinear.from_linear(dense, params["group_size"], params["bits"],
                                                   mode="affine")
        new.append((path, dense))
        layout[path] = params
    if new:
        model.update_modules(tree_unflatten(new))
    return layout


def _materialize_here(base: str, adapter_dirs: list[str], out_dir: str, bits: int = 8) -> int:
    from huggingface_hub import snapshot_download
    from symbio.app.modelload import load
    from symbio.mlx_gate import attr as _mlx
    model, tokenizer = load(base, resolve=False)
    layout = fold(model, [Path(d) for d in adapter_dirs], bits=bits)
    src = Path(base) if Path(base).exists() else Path(snapshot_download(base))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Everything but the weights comes from the original: config, tokenizer,
    # chat template, processor files.
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") \
                and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    # Per-layer widths for the folded layers (mlx_lm reads a module path in
    # the quantization block as that module's own settings). A dense layer
    # folded into a quantized model is marked unquantized.
    cfg = json.loads((out / "config.json").read_text())
    for key in ("quantization", "quantization_config"):
        if isinstance(cfg.get(key), dict):
            for path, params in layout.items():
                cfg[key][path] = params if params else False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    _mlx("mlx_lm.utils.save_model")(out, model, donate_model=True)
    return len(layout)


def materialize(base: str, adapter_dirs: list[Path], out_dir: Path,
                bits: int = DEFAULTS["fold_bits"]) -> tuple[bool, str]:
    """Build base + adapters as a standalone model directory, in a child
    process so its copy of the weights is gone when this returns."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if not adapter_dirs:
        return True, "no deltas: the original base"
    proc = subprocess.run(
        [sys.executable, "-m", "symbio.app.weights", "materialize", base, str(out_dir),
         f"--bits={int(bits)}", *[str(d) for d in adapter_dirs]],
        capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return False, tail[0]
    return True, proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "ok"


def disk_room_for(base: str) -> str | None:
    """None when there is room for another copy of `base`, else why not."""
    src = Path(resolve(base))
    if not src.exists():
        try:
            from huggingface_hub import snapshot_download
            src = Path(snapshot_download(base, local_files_only=True))
        except Exception:
            return "the base model is not in the local cache"
    need = sum(f.stat().st_size for f in src.rglob("*.safetensors"))
    _dir().mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(_dir()).free
    if free < need * 1.2 + 1e9:
        return (f"needs about {need * 1.2 / 1e9 + 1:.1f} GB free for a new version "
                f"of the model, {free / 1e9:.1f} GB is free")
    return None


# ---- proposals: what the model is asked about ----

def _age_days(path: Path) -> float:
    try:
        return (datetime.now().timestamp() - path.stat().st_mtime) / 86400
    except OSError:
        return 0.0


def candidates(config: dict[str, Any], failing_now: set[str] | None = None) -> list[dict[str, Any]]:
    """The changes worth asking the model about, each with the evidence it
    will be shown. A bind for the headmaster's current adapter once it has
    been in use a while; an unbind for the most recent fold that came before
    checks which now fail."""
    base = config["model_name"]
    s = settings(config)
    out: list[dict[str, Any]] = []
    adapter = constants.ADAPTER_DIR
    weights = adapter / "adapters.safetensors"
    if weights.is_file() and foldable(adapter) is None:
        trained_on = adapter_base(adapter)
        age = _age_days(weights)
        if trained_on in (base, resolve(base)) and age >= float(s["bind_min_age_days"]):
            n = entry(base)["bound"]
            out.append({
                "kind": "bind", "base": base, "adapter": str(adapter),
                "question": (
                    f"Your learned adapter has been in use for {age:.0f} days and is still a "
                    f"separate layer on top of your base weights ({len(n)} earlier update(s) "
                    f"already folded in). Fold it permanently into your base weights, so what "
                    f"you learn next builds on it? A copy is archived and it can be taken out "
                    f"again."),
                # "no" first: any lean toward the first letter is a lean
                # toward leaving the weights alone.
                "options": [("no", "No, keep it as a separate adapter for now"),
                            ("yes", "Yes, fold it into my base weights")],
            })
    if failing_now:
        for rec in reversed(entry(base)["bound"]):
            lost = sorted(set(rec.get("passing_before", [])) & failing_now)
            if lost:
                out.append({
                    "kind": "unbind", "base": base, "id": rec["id"], "lost": lost,
                    "question": (
                        f"An update ({rec['id']}) was folded into your base weights on "
                        f"{rec.get('bound_at', '?')[:10]}. These checks passed before it and fail "
                        f"now: {', '.join(lost)}. Take that update back out of your base weights? "
                        f"It stays archived."),
                    "options": [("no", "No, keep it folded in"),
                                ("yes", "Yes, take it back out")],
                })
                break
    return out


def decide(model: Any, tokenizer: Any, cand: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The model's own call on one candidate. A close call is a no."""
    from symbio.app import semif
    res = semif.choose(model, tokenizer, cand["question"], cand["options"],
                       min_margin=float(settings(config)["min_margin"]))
    res["yes"] = res.get("choice") == "yes"
    return res


# ---- the trial and its verdict ----

def _version_dir(base: str, n: int) -> Path:
    return _dir() / _slug(base) / f"v{n}"


def stage(cand: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """Build the trial version for a candidate and point loads at it. Returns
    (ok, message, plan); the plan is what commit() or abandon() needs."""
    base = cand["base"]
    e = entry(base)
    bound = e["bound"]
    room = disk_room_for(base)
    if room:
        return False, room, {}
    if cand["kind"] == "bind":
        why = foldable(Path(cand["adapter"]))
        if why:
            return False, why, {}
        did, arch = archive_adapter(Path(cand["adapter"]), base, "adapter")
        new_bound = bound + [{"id": did, "archive": str(arch),
                              "bound_at": datetime.now().isoformat(timespec="seconds")}]
    else:
        did = cand["id"]
        new_bound = [r for r in bound if r["id"] != did]
    n = 1 + max([int(p.name[1:]) for p in (_dir() / _slug(base)).glob("v[0-9]*")
                 if p.name[1:].isdigit()] or [0])
    out = _version_dir(base, n)
    ok, msg = materialize(base, [Path(r["archive"]) for r in new_bound], out)
    plan = {"base": base, "kind": cand["kind"], "id": did, "bound": new_bound,
            "path": str(out) if new_bound else None, "prev_path": e["path"]}
    if not ok:
        return False, f"could not build the new version: {msg}", plan
    _TRIAL[base] = plan["path"]
    return True, msg, plan


def commit(plan: dict[str, Any], before: set[str], after: set[str]) -> None:
    """Keep the trial: it becomes the base. The version it replaced is a cache
    of archived deltas and is removed."""
    base = plan["base"]
    _TRIAL.pop(base, None)
    ledger = load_ledger()
    e = entry(base, ledger)
    bound = plan["bound"]
    if plan["kind"] == "bind":
        bound[-1]["passing_before"] = sorted(before)
        bound[-1]["passing_after"] = sorted(after)
    e["bound"] = bound
    e["path"] = plan["path"]
    e["events"].append({"at": datetime.now().isoformat(timespec="seconds"),
                        "kind": plan["kind"], "id": plan["id"],
                        "passing_before": len(before), "passing_after": len(after)})
    ledger[base] = e
    save_ledger(ledger)
    prev = plan.get("prev_path")
    if prev and prev != plan["path"]:
        shutil.rmtree(prev, ignore_errors=True)


def abandon(plan: dict[str, Any]) -> None:
    """Drop the trial; loads go back to the version before it. A bind's
    archived copy is kept: the adapter it came from is still live."""
    if not plan:
        return
    _TRIAL.pop(plan["base"], None)
    if plan.get("path"):
        shutil.rmtree(plan["path"], ignore_errors=True)


def _main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "materialize":
        rest = argv[3:]
        bits = DEFAULTS["fold_bits"]
        if rest and rest[0].startswith("--bits="):
            bits, rest = int(rest[0].split("=", 1)[1]), rest[1:]
        changed = _materialize_here(argv[1], rest, argv[2], bits=bits)
        print(f"folded {len(rest)} delta(s) into {changed} module(s)")
        return 0
    print("usage: python -m symbio.app.weights materialize BASE OUT_DIR ADAPTER...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
