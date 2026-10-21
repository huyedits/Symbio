"""weld_prune.py — weld adapter weights together, and prune the noise out.

Three verbs, one file:

  merge  several adapters (the root one next to the skill workers, or any
         dirs you name) become ONE adapter file. Per module the rank blocks
         are concatenated in adapter space — mathematically exact, since
         ΔW is a sum of rank-one blocks: scale·BᵀAᵀ of the concat equals
         the sum of the parts' ΔW. Adapters a module never belonged to are
         padded with zero blocks, and the file carries ONE uniform rank
         (mlx_lm builds every LoRA layer at the config's single rank; a
         mixed-rank file loads but silently keeps its zeros — tested).

  prune  drops the rank slices whose contribution is numerically nothing.
         Slice i of a pair contributes scale·B[i,:]ᵀA[:,i]ᵀ, whose exact
         Frobenius norm is scale·|A[:,i]|·|B[i,:]|. Slices are kept until
         coverage of Σe² crosses --keep (0.999 default); --min-rel drops
         any slice weaker than that share of the strongest one. A module
         whose every slice is zero — a fresh pair, lora_b never trained —
         is dropped whole. The written file pins ONE rank = strongest
         module kept, zero-padded below, so it still loads at inference.

  weld   prune, then fold what survives into the base model's weights,
         through symbio.app.weights' own path: the surviving adapter is
         archived, a new version of the model is materialized beside the
         originals, and the ledger points loads at it — every step the
         in-chat bind would take, minus asking the model's permission,
         plus the pruning. Nothing folded is lost: the archive can be
         taken back out by the existing unbind flow. After the weld the
         live headmaster adapter is removed (it now lives in the base;
         leaving it would apply the same ΔW twice) — its workers stay.

Usage:
  venv/bin/python scripts/weld_prune.py merge adapters/ adapters/workers/X \\
      --out adapters_merged/
  venv/bin/python scripts/weld_prune.py prune adapters/ --out adapters_pruned/
  venv/bin/python scripts/weld_prune.py prune adapters/ --in-place --keep 0.99
  venv/bin/python scripts/weld_prune.py weld adapters/           # dry run
  venv/bin/python scripts/weld_prune.py weld adapters/ --yes

merge and prune only write new directories. weld without --yes builds the
version, reports it, and takes it back; --yes keeps it and points the
ledger at it. Run /golden afterwards — the in-chat flow grades a bind
against the golden battery before keeping it, and a manual weld skips that
on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

KEEP_DEFAULT = 0.999


# --------------------------------------------------------------- reading

def read_config(dir_: Path) -> dict:
    cfg = json.loads((Path(dir_) / "adapter_config.json").read_text())
    if str(cfg.get("fine_tune_type", "lora")) != "lora":
        raise ValueError(f"{dir_}: a {cfg.get('fine_tune_type')} adapter cannot be welded")
    return cfg


def scale_of(cfg: dict) -> float:
    return float((cfg.get("lora_parameters") or {}).get("scale", 20.0))


def rank_of(cfg: dict) -> int:
    return int((cfg.get("lora_parameters") or {}).get("rank", 0))


def module_pairs(dir_: Path) -> dict[str, tuple[mx.array, mx.array]]:
    """{module: (A (in, r), B (r, out))} from one adapter file."""
    tensors = mx.load(str(Path(dir_) / "adapters.safetensors"))
    out: dict[str, tuple[mx.array, mx.array]] = {}
    for key, a in tensors.items():
        if not key.endswith(".lora_a"):
            continue
        name = key[: -len(".lora_a")]
        b = tensors.get(name + ".lora_b")
        if b is None or a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{name}: not an ordinary LoRA pair (weld leaves MoE alone)")
        out[name] = (a, b)
    return out


def delta(pair: tuple[mx.array, mx.array], scale: float) -> mx.array:
    """A module's dense update, the same arithmetic fold() uses."""
    a, b = pair
    return (scale * b.T) @ a.T


def suffix(keys: list[str] | None, names: list[str]) -> list[str]:
    """The keys set for a written config: the union of every source's own
    `keys` (suffix style, e.g. "self_attn.q_proj") and every module in the
    file, de-layered the same way."""
    import re
    out = set(keys or [])
    for n in names:
        out.add(re.sub(r"^model\.layers\.\d+\.", "", n))
    return sorted(out)


# --------------------------------------------------------------- pruning

def energies(pair: tuple[mx.array, mx.array], scale: float) -> list[float]:
    """Per rank slice: the exact Frobenius norm of that slice's contribution."""
    a, b = pair
    out = []
    for i in range(a.shape[1]):
        na = math.sqrt(float(mx.sum(a[:, i] * a[:, i]).item()))
        nb = math.sqrt(float(mx.sum(b[i, :] * b[i, :]).item()))
        out.append(float(scale) * na * nb)
    return out


def choose(ener: list[float], keep: float, min_rel: float) -> list[int]:
    """Indices kept, in original order; coverage counts squared energy."""
    if min_rel > 0.0 and ener:
        strongest = max(ener)
        ener = [e if e >= min_rel * strongest else 0.0 for e in ener]
    if keep >= 1.0:
        return [i for i, e in enumerate(ener) if e > 0.0]
    order = sorted(range(len(ener)), key=lambda i: ener[i], reverse=True)
    total = sum(e * e for e in ener)
    if total <= 0.0:
        return []
    strongest = ener[order[0]]
    kept: list[int] = []
    acc = 0.0
    for i in order:
        kept.append(i)
        acc += ener[i] * ener[i]
        if acc / total >= keep - 1e-12:
            break
    if min_rel > 0.0:
        kept = [i for i in kept if ener[i] >= min_rel * strongest]
    return sorted(kept)


def rebuild(pair: tuple[mx.array, mx.array], kept: list[int],
            rank_out: int) -> tuple[mx.array, mx.array]:
    """The pair cut to its kept slices, zero-padded to a uniform rank."""
    a, b = pair
    in_f, out_f = a.shape[0], b.shape[1]
    dtype = a.dtype
    cols = [a[:, i:i + 1] for i in kept] + [mx.zeros((in_f, rank_out - len(kept)), dtype)]
    rows = [b[i:i + 1, :] for i in kept] + [mx.zeros((rank_out - len(kept), out_f), dtype)]
    return mx.concatenate(cols, axis=1), mx.concatenate(rows, axis=0)


def write(dir_: Path, pairs: dict, rank_out: int, cfg: dict) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}
    for name, (a, b) in pairs.items():
        tensors[name + ".lora_a"] = a
        tensors[name + ".lora_b"] = b
    mx.save_safetensors(str(dir_ / "adapters.safetensors"), tensors)
    cfg = json.loads(json.dumps(cfg))
    params = cfg.setdefault("lora_parameters", {})
    params["rank"] = rank_out
    params["keys"] = suffix(params.get("keys"), list(pairs))
    cfg["adapter_path"] = str(dir_)
    (dir_ / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def file_bytes(dir_: Path) -> int:
    return (Path(dir_) / "adapters.safetensors").stat().st_size \
        + (Path(dir_) / "adapter_config.json").stat().st_size


# ----------------------------------------------------------------- merge

def merge(dirs: list[Path], out: Path) -> None:
    """One file out of many: exact block concat, one uniform rank."""
    cfgs = {str(d): read_config(d) for d in dirs}
    bases = {c.get("model") for c in cfgs.values()}
    if len(bases) != 1:
        which = ", ".join(sorted(b for b in bases if b) or ["<unrecorded>"])
        raise ValueError(f"these adapters were trained on different bases ({which}) "
                         f"— one file loads against one model; refusing to weld them")
    base, scale_out = next(iter(bases)), scale_of(next(iter(cfgs.values())))
    layers: dict[str, list[tuple[mx.array, mx.array, float]]] = {}
    for d in dirs:
        for name, pair in module_pairs(d).items():
            layers.setdefault(name, []).append((*pair, scale_of(cfgs[str(d)])))

    ranks = {n: sum(a.shape[1] for a, _, _ in v) for n, v in layers.items()}
    rank_out = max(ranks.values())

    # Absorb each part's own scale into its B block: scale_out·(k·B)ᵀAᵀ
    # equals scale_i·BᵀAᵀ when k = scale_i/scale_out.
    merged: dict[str, tuple[mx.array, mx.array]] = {}
    for name, parts in layers.items():
        a0, b0, _ = parts[0]
        for a, b, _ in parts:
            if a.shape[0] != a0.shape[0] or b.shape[1] != b0.shape[1] \
                    or a.shape[1] != b.shape[0] or a.dtype != a0.dtype:
                raise ValueError(f"{name}: adapters' blocks {a.shape}/{b.shape} do not fit")
        dtype = a0.dtype
        A = mx.concatenate([p[0] for p in parts]
                           + [mx.zeros((a0.shape[0], rank_out - ranks[name]), dtype)], axis=1)
        B = mx.concatenate([p[1] * (p[2] / scale_out) for p in parts]
                           + [mx.zeros((rank_out - ranks[name], b0.shape[1]), dtype)], axis=0)
        merged[name] = (A, B)

    cfg = json.loads(json.dumps(next(iter(cfgs.values()))))
    params = cfg["lora_parameters"]
    keys = {k for c in cfgs.values() for k in (c["lora_parameters"].get("keys") or [])}
    params["keys"] = sorted(keys | set(suffix(None, list(merged))))
    params["rank"] = rank_out
    cfg["num_layers"] = max(int(c.get("num_layers", 0)) for c in cfgs.values())
    cfg["adapter_path"] = str(out)
    write(out, merged, rank_out, cfg)

    before = sum(file_bytes(d) for d in dirs)
    print(f"merged {len(dirs)} adapters into {out}: {len(merged)} module(s), "
          f"rank {rank_out} (zero-padded: "
          f"{', '.join(f'{n}/{r}' for n, r in sorted(ranks.items()))})")
    print(f"  base {base}; scale {scale_out} (from the first source); "
          f"num_layers {cfg['num_layers']}")
    print(f"  bytes: {before / 1e6:.2f} MB in -> {file_bytes(out) / 1e6:.2f} MB out; "
          f"pruning will shrink it further: scripts/weld_prune.py prune {out}")


# ----------------------------------------------------------------- prune

def prune(dir_: Path, out: Path, keep: float, min_rel: float) -> dict:
    """Cut negligible rank slices; returns what happened, for tests and logs."""
    cfg = read_config(dir_)
    scale = scale_of(cfg)
    pairs = module_pairs(dir_)
    stats: dict = {"modules": {}, "dropped": [], "rank_in": rank_of(cfg), "out": str(out),
                   "bytes_in": file_bytes(dir_)}
    cuts: dict[str, list[int]] = {}
    rank_out = 0
    for name, pair in pairs.items():
        es = energies(pair, scale)
        kept = choose(es, keep, min_rel)
        if not kept:
            stats["dropped"].append(name)
            continue
        before = float(mx.sqrt(mx.sum(delta(pair, scale) ** 2)).item())
        keptes = [es[i] for i in kept]
        stats["modules"][name] = {
            "rank": len(kept), "slices": len(es),
            "energy_kept": sum(e * e for e in keptes) / max(sum(e * e for e in es), 1e-30),
            "delta_before": before, "delta_after": before,
        }
        cuts[name] = kept
        rank_out = max(rank_out, len(kept))
    if not cuts:
        raise ValueError(f"{dir_}: every module is untouched (all lora_b zero) — "
                         f"nothing to prune; this adapter is already empty")

    pruned = {n: rebuild(pairs[n], cuts[n], rank_out) for n in cuts}
    write(out, pruned, rank_out, cfg)
    for n, pair in pruned.items():
        stats["modules"][n]["delta_after"] = float(
            mx.sqrt(mx.sum(delta(pair, scale) ** 2)).item())
    stats["rank_out"] = rank_out
    stats["bytes_out"] = file_bytes(out)
    stats["pruned_any"] = any(s["rank"] < s["slices"] for s in stats["modules"].values()) \
        or bool(stats["dropped"])
    return stats


def report_prune(stats: dict) -> None:
    print(f"pruned {stats['out']}: rank {stats['rank_in']} -> {stats['rank_out']} "
          f"(weakest module, zero-padded below)")
    for n, s in sorted(stats["modules"].items()):
        print(f"  {n}: rank {s['slices']} -> {s['rank']}, "
              f"ΔW |{s['delta_before']:.4f} -> {s['delta_after']:.4f}|, "
              f"{s['energy_kept'] * 100:.3f}% of its energy kept")
    for n in stats["dropped"]:
        print(f"  {n}: every slice zero — dropped whole")
    print(f"  bytes: {stats['bytes_in'] / 1e6:.2f} -> {stats['bytes_out'] / 1e6:.2f} MB")


# ------------------------------------------------------------------ weld

def same_live_adapter(dir_: Path) -> bool:
    from symbio import constants
    return Path(dir_).resolve() == Path(constants.ADAPTER_DIR).resolve()


def model_name() -> str:
    from symbio.app.config import load_config
    name = load_config().get("model_name")
    if not name:
        raise ValueError("config.json sets no model_name — pass --base")
    return name


def weld(dir_: Path, base: str | None, keep: float, min_rel: float,
         no_prune: bool, yes: bool) -> int:
    """Prune, then bind what survives into the base, via the app's own path."""
    from symbio.app import training, weights
    from symbio import constants

    dir_ = Path(dir_ or constants.ADAPTER_DIR)
    base = base or model_name()
    trained_on = read_config(dir_).get("model")
    reach = {base, weights.resolve(base)}
    if trained_on not in reach:
        raise ValueError(f"{dir_} was trained on {trained_on}, which is neither "
                         f"{base} nor {weights.resolve(base)} — it will not weld")
    if not yes:
        print(f"plan: fold {dir_} (trained on {trained_on}) into {weights.resolve(base)}")

    if no_prune:
        prepared, stats = dir_, None
    else:
        stats = prune(dir_, _wip() / "pruned", keep, min_rel)
        report_prune(stats)
        prepared = dir_ if not stats["pruned_any"] else Path(stats["out"])

    cand = {"kind": "bind", "base": base, "adapter": str(prepared)}
    ok, msg, plan = weights.stage(cand)
    if not ok:
        print(f"not changed: {msg}")
        return 1
    if not yes:
        weights.abandon(plan)
        print(f"dry run: the version was built at {plan['path']} and taken back; "
              f"nothing points at it. Run the same command with --yes to keep it.")
        return 0

    weights.commit(plan, set(), set())
    print(f"welded {plan['id']} into {plan['path']}: {msg}")
    if same_live_adapter(dir_):
        training.remove_adapter()
        print("live adapter removed: its update now lives in the base, and leaving "
              "it would apply the same ΔW twice. workers/ untouched.")
    print(f"archived at {plan['bound'][-1]['archive']} — the unbind machinery can "
          f"take it back out. Run /golden to grade this the way the in-chat flow would.")
    return 0


def _wip() -> Path:
    """A scratch dir for the prepared (pruned) adapter. cwd's scratch/ when the
    repo layout is there, else the plain temp dir."""
    import tempfile
    scratch = Path.cwd() / "scratch"
    return Path(tempfile.mkdtemp(prefix="weld-",
                                 dir=scratch if scratch.is_dir() else None))


# ------------------------------------------------------------------- cli

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="weld adapter weights; prune the noise out")
    sub = p.add_subparsers(dest="verb", required=True)

    m = sub.add_parser("merge", help="several adapters -> one file")
    m.add_argument("dirs", nargs="+", type=Path)
    m.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("prune", help="drop negligible rank slices")
    r.add_argument("dir", type=Path)
    r.add_argument("--out", type=Path)
    r.add_argument("--in-place", action="store_true")
    r.add_argument("--keep", type=float, default=KEEP_DEFAULT)
    r.add_argument("--min-rel", type=float, default=0.0)

    w = sub.add_parser("weld", help="prune, then fold into the base model")
    w.add_argument("dir", type=Path, nargs="?", default=None)
    w.add_argument("--base", default=None)
    w.add_argument("--no-prune", action="store_true")
    w.add_argument("--keep", type=float, default=KEEP_DEFAULT)
    w.add_argument("--min-rel", type=float, default=0.0)
    w.add_argument("--yes", action="store_true")

    a = p.parse_args(argv)
    try:
        if a.verb == "merge":
            if len(a.dirs) < 2:
                p.error("merge takes at least two adapter directories")
            merge(a.dirs, a.out)
            return 0
        if a.verb == "prune":
            out = a.out if a.out else a.dir.with_name(a.dir.name + "-pruned")
            if a.in_place:
                out = a.dir
            report_prune(prune(Path(a.dir), Path(out), a.keep, a.min_rel))
            return 0
        return weld(a.dir, a.base, a.keep, a.min_rel, a.no_prune, a.yes)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, RuntimeError) as e:
        print(f"no: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))