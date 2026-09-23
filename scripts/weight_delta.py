"""weight_delta.py — how far has fine-tuning moved the model's weights?

A LoRA adapter never rewrites the base. Each targeted module gets

    W_effective = W + scale · (lora_a @ lora_b)

so the change a fine-tune made is exactly ΔW = scale · lora_a @ lora_b,
computable from the adapter file alone. This reports, per module:

  |ΔW|      Frobenius norm of the change;
  rel       |ΔW| / |W| against the base weight, when --base is given (the base
            is 3-bit; its module is dequantised on the fly, nothing else loads);
  moved     between two adapters (two checkpoints of one run, or before/after
            a retrain): |ΔW_new - ΔW_old|, i.e. how much the latest training
            actually changed the model, module by module.

Usage:
  venv/bin/python scripts/weight_delta.py adapters/
  venv/bin/python scripts/weight_delta.py NEW_DIR --since OLD_DIR
  venv/bin/python scripts/weight_delta.py adapters/ --base mlx-community/Qwen3-14B-3bit
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import mlx.core as mx


def adapter_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "adapters.safetensors" if path.is_dir() else path


def lora_scale(path: str | Path) -> float:
    cfg_path = (Path(path) if Path(path).is_dir() else Path(path).parent) / "adapter_config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
        return float(cfg.get("lora_parameters", {}).get("scale", 20.0))
    except (OSError, ValueError):
        return 20.0


def deltas(path: str | Path, scale: float | None = None) -> dict[str, mx.array]:
    """Module name -> ΔW (in_features x out_features), float32."""
    scale = lora_scale(path) if scale is None else scale
    weights = mx.load(str(adapter_file(path)))
    out = {}
    for key, a in weights.items():
        if not key.endswith(".lora_a"):
            continue
        module = key[: -len(".lora_a")]
        b = weights.get(module + ".lora_b")
        if b is None:
            continue
        out[module] = scale * (a.astype(mx.float32) @ b.astype(mx.float32))
    return out


def norm(x: mx.array) -> float:
    return float(mx.sqrt((x * x).sum()))


def base_norms(model: str, modules: list[str]) -> dict[str, float]:
    """|W| of just the named modules of a (possibly quantised) base model."""
    from huggingface_hub import snapshot_download

    root = Path(model) if Path(model).is_dir() else Path(
        snapshot_download(model, local_files_only=True))
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((root / "config.json").read_text())
    quant = cfg.get("quantization") or {}
    shards: dict[str, dict] = {}
    out = {}
    for module in modules:
        key = module + ".weight"
        shard = index.get(key)
        if shard is None:
            continue
        if shard not in shards:
            shards[shard] = mx.load(str(root / shard))
        tensors = shards[shard]
        w = tensors[key]
        if module + ".scales" in tensors:
            w = mx.dequantize(w, tensors[module + ".scales"], tensors.get(module + ".biases"),
                              group_size=quant.get("group_size", 64), bits=quant.get("bits", 4))
        out[module] = norm(w.astype(mx.float32))
    return out


def _layer(module: str) -> int:
    m = re.search(r"layers\.(\d+)\.", module)
    return int(m.group(1)) if m else -1


def report(path, since=None, base=None) -> list[dict]:
    now = deltas(path)
    old = deltas(since) if since else {}
    bases = base_norms(base, list(now)) if base else {}
    rows = []
    for module in sorted(now, key=lambda m: (_layer(m), m)):
        row = {"module": module, "delta": norm(now[module])}
        if module in bases and bases[module]:
            row["relative"] = row["delta"] / bases[module]
        if since:
            prev = old.get(module)
            row["moved"] = norm(now[module] - prev) if prev is not None else row["delta"]
        rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    since = argv[argv.index("--since") + 1] if "--since" in argv else None
    base = argv[argv.index("--base") + 1] if "--base" in argv else None
    rows = report(argv[0], since=since, base=base)
    if not rows:
        print(f"no LoRA modules in {adapter_file(argv[0])}")
        return 1
    top = max(r["delta"] for r in rows)
    header = "module".ljust(34) + "  |dW|    " + ("  rel     " if base else "") + ("  moved " if since else "")
    print(header)
    for r in rows:
        bar = "#" * max(1, int(24 * r["delta"] / top)) if top else ""
        line = f"{r['module'][-34:]:34}  {r['delta']:7.3f}"
        if base:
            line += f"  {r.get('relative', float('nan')) * 100:6.3f}%"
        if since:
            line += f"  {r['moved']:7.3f}"
        print(f"{line}  {bar}")
    total = sum(r["delta"] ** 2 for r in rows) ** 0.5
    print(f"\n{len(rows)} modules changed; total |dW| {total:.3f}"
          + (f"; moved since {since}: {sum(r['moved'] ** 2 for r in rows) ** 0.5:.3f}" if since else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
