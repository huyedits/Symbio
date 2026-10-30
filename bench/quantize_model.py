"""Quantize FableVibes-14B-A3B (or any MLX repo) down with a MoE-safe recipe.

Target: 6–7 GB from the 8.5 GB q4. The route is a MIXED quantization —
sensitive weights (attention, router gates, embeddings) get more bits,
expert MLPs get fewer — which is where MoE accuracy survives a cut. The
REAP-pruned model has half its specialists already removed; the router is
the part we must not round hard.

mlx_lm convert ships exactly these predicates:
    mixed_3_6  (sensitive→6, robust→3)   — our default, lands ~6.5 GB
    mixed_3_4  (sensitive→4, robust→3)   — lands ~5.5 GB, floor
    mixed_4_6  (sensitive→6, robust→4)   — lands ~9.5 GB, not needed (q4 exists)

Usage:
    SYMBIO_SRC_MODEL=khanh2023/Qwen3.6-14B-A3B-FableVibes-mlx-q4 \\
    ./venv/bin/python bench/quantize_model.py --predicate mixed_3_6 --tag q36

The output is a full MLX model directory with its own config; point
config.json at the local path and run the A/B like any other model.
The A/B script re-runs the same three probes, so the quality cost of the
quantization is MEASURED against the q4 original, not asserted.
"""
import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("SYMBIO_SRC_MODEL",
              "khanh2023/Qwen3.6-14B-A3B-FableVibes-mlx-q4"))
    ap.add_argument("--predicate", default="mixed_3_6",
                    choices=["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"])
    ap.add_argument("--tag", default=None, help="output dir suffix, e.g. q36")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = a.tag or a.predicate
    out = a.out or f"models/{os.path.basename(a.src)}-{tag}"
    cmd = [sys.executable, "-m", "mlx_lm", "convert",
           "--hf-path", a.src,
           "-o", out,
           "--quant-predicate", a.predicate,
           "--dtype", "bfloat16"]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode == 0:
        print(f"\nQuantized model at: {out}")
        print("Next: config.json model_name -> that path, then run "
              "bench/ab_two_models.py to measure the quality cost.")
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())