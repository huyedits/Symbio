#!/bin/sh
# One-time setup on an AMD MI300X (ROCm) or NVIDIA box, after server/push.sh.
#   sh symbio/server/setup_rocm.sh
# Every stage prints PASS/FAIL and stops on the first failure, so a broken
# port costs minutes, not credit-hours. Safe to rerun.
#
# Assumes the provider's "ROCm + PyTorch" image: torch with ROCm already
# installed system-wide. The venv inherits it (--system-site-packages) —
# pip-installing torch over a vendor build is the classic way to lose the GPU.
set -eu
cd "$HOME/symbio"
export HF_HOME=${HF_HOME:-/mnt/scratch/hf}      # the 5 TB scratch disk; boot disk is 720 GB
[ -d /mnt/scratch ] || export HF_HOME="$HOME/hf"
mkdir -p "$HF_HOME"
stage() { printf '\n== %s\n' "$1"; }

stage "1/6 GPU visible to torch"
python3 - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "none"
mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if ok else 0
print(f"torch {torch.__version__}  hip={getattr(torch.version, 'hip', None)}  device={name}  {mem:.0f} GiB")
sys.exit(0 if ok else "FAIL: no GPU visible to torch — wrong image or driver")
PY

stage "2/6 venv + deps"
[ -d venv ] || python3 -m venv --system-site-packages venv
. venv/bin/activate
pip install -q --upgrade pip
# pyproject's core deps carry no torch; requirements.txt does, and would
# replace the vendor ROCm build with a CUDA wheel. Never install from it here.
pip install -q -e .
pip install -q "transformers>=4.51" peft accelerate datasets pytest
python -c "import torch; assert torch.cuda.is_available(), 'pip replaced the ROCm torch'" \
  && echo PASS || { echo "FAIL: torch lost the GPU after pip — reinstall the vendor build"; exit 1; }

stage "3/6 the harness's own tests (no model)"
SYMBIO_BACKEND=cuda python -m pytest -q --no-header -p no:cacheprovider \
  tests/test_fix_loop.py tests/test_devtools.py tests/test_semif.py tests/test_backend.py tests/test_rocm.py

stage "4/6 load + generate on the GPU (Qwen3-0.6B, 16-bit)"
SYMBIO_BACKEND=cuda python - <<'PY'
from symbio import backend
cfg = {"backend": "cuda", "cuda": {"load_in_bits": 16}, "agent": {"temperature": 0}}
m, t = backend.load("Qwen/Qwen3-0.6B", config=cfg)
p = t.apply_chat_template([{"role": "user", "content": "Say OK."}], tokenize=False,
                          add_generation_prompt=True, enable_thinking=False)
out = backend.generate(m, t, p, max_tokens=8, config=cfg)
print("reply:", repr(out))
assert out.strip(), "FAIL: empty generation"
print("PASS")
PY

stage "5/6 LoRA trainer, 5 steps (symbio.cuda_lora, 16-bit)"
D=$(mktemp -d)
python - "$D" <<'PY'
import json, sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
rows = [json.dumps({"text": t.apply_chat_template(
    [{"role": "user", "content": f"What is {i} plus {i}?"},
     {"role": "assistant", "content": str(2 * i)}], tokenize=False)}) for i in range(8)]
open(f"{sys.argv[1]}/train.jsonl", "w").write("\n".join(rows) + "\n")
open(f"{sys.argv[1]}/valid.jsonl", "w").write("\n".join(rows[:2]) + "\n")
PY
SYMBIO_BACKEND=cuda python -m symbio.cuda_lora --model Qwen/Qwen3-0.6B --train --data "$D" \
  --adapter-path "$D/adapter" --iters 5 --batch-size 1 --num-layers 4 --max-seq-length 256 \
  --learning-rate 1e-4 --load-in-bits 16
ls "$D/adapter"/adapter*.safetensors >/dev/null && echo PASS || { echo "FAIL: no adapter written"; exit 1; }

stage "6/6 bench copy"
# The bench writes each bug into its copy's source: never the live tree.
rsync -a --delete --exclude venv/ "$HOME/symbio/" "$HOME/symbio_bench/"
echo PASS
printf '\nREADY. next: sh symbio/server/plan.sh <stage>  (see plan.sh header)\n'
