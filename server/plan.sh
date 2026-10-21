#!/bin/sh
# The GPU-server run plan. One stage per call; every stage logs under
# ~/runs/<stage>/ and is resumable (the bench and rsi skip finished steps).
# Run inside tmux so a dropped SSH session does not kill a paid run:
#   tmux new -s s   then   sh symbio/server/plan.sh ceiling
#
#   ceiling   the 20-bug bench + 40 held-out bugs on a ~235B model served by
#             vLLM: is the gap model size or harness?          (~3 h)
#   rsi       the self-improvement rounds (base x2, gen1, r1, ctrl, gen2, r2)
#             on a 32B with LoRA at 16-bit                     (~12-15 h)
#   status    every result so far
#
# BIG=<hf repo> overrides the served model, MID=<hf repo> the trained one.
# SERVE=llamacpp serves a GGUF instead (BIG=<repo>, GGUF=<quant dir/file
# pattern>), e.g. GLM-5.3-Flash at UD-Q3_K_XL (148 GB) on one 192 GB card:
#   SERVE=llamacpp BIG=unsloth/GLM-5.3-Flash-GGUF GGUF=UD-Q3_K_XL sh plan.sh ceiling
# Best single-card ceiling: Qwen3.8-Flash-Next (180B, 512 experts) at near-lossless Q6 (169 GB):
#   SERVE=llamacpp BIG=unsloth/Qwen3.8-Flash-Next-GGUF GGUF=UD-Q6_K_XL sh plan.sh ceiling
# needs llama.cpp built for ROCm (llama-server on PATH).
# DELETE THE DROPLET WHEN DONE: a powered-off droplet still bills.
set -eu
STAGE=${1:?usage: plan.sh ceiling|rsi|status}
cd "$HOME/symbio"
. venv/bin/activate
export HF_HOME=${HF_HOME:-/mnt/scratch/hf}
[ -d /mnt/scratch ] || export HF_HOME="$HOME/hf"
export SYMBIO_BACKEND=cuda
BENCH="$HOME/symbio_bench"
OUT="$HOME/runs/$STAGE"
mkdir -p "$OUT"
BIG=${BIG:-QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ}
MID=${MID:-Qwen/Qwen3-32B}

case "$STAGE" in
ceiling)
  # vLLM comes with the ROCm image (or: pip install vllm per AMD's ROCm docs).
  # --max-model-len 16384: repair prompts are ~5k tokens, room to spare.
  if ! curl -sf localhost:8000/v1/models >/dev/null; then
    if [ "${SERVE:-vllm}" = llamacpp ]; then
      huggingface-cli download "$BIG" --include "*${GGUF:?set GGUF}*" --local-dir "$HF_HOME/gguf/$BIG"
      FIRST=$(find "$HF_HOME/gguf/$BIG" -name "*${GGUF}*.gguf" | sort | head -1)
      nohup llama-server -m "$FIRST" -ngl 999 -c 16384 --jinja --port 8000 \
        > "$OUT/vllm.log" 2>&1 &
    else
      # one shard per GPU: the 8-card droplet holds DeepSeek-V4.1-Flash (763 GB) at full precision
      NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
      nohup vllm serve "$BIG" --tensor-parallel-size "$NGPU" --trust-remote-code \
        --max-model-len 16384 --gpu-memory-utilization 0.92 \
        > "$OUT/vllm.log" 2>&1 &
    fi
    echo "starting ${SERVE:-vllm} ($BIG); log: $OUT/vllm.log"
    i=0
    until curl -sf localhost:8000/v1/models >/dev/null; do
      i=$((i + 1)); [ $i -gt 360 ] && { echo "FAIL: vLLM not up after 60 min"; tail -30 "$OUT/vllm.log"; exit 1; }
      sleep 10
    done
  fi
  cd "$BENCH"
  for set in bugs heldout_bugs; do
    tag="$(echo "$BIG" | tr "/" "_")_${set}"   # per model: a second model must not skip as done
    [ -f "$OUT/$tag.done" ] && continue
    GEN=openai OPENAI_BASE_URL=http://localhost:8000/v1 SYMBIO_HOME="$BENCH" \
      python "$BENCH/bench/self_repair/drive.py" "$BIG" "$tag" \
      "$BENCH/bench/self_repair/$set.json" "$OUT" | tee "$OUT/$tag.log"
    touch "$OUT/$tag.done"
  done
  pkill -f "vllm serve" || pkill -f llama-server || true
  ;;
rsi)
  MODEL="$MID" GEN=hf LOAD_BITS=16 CLONE="$BENCH" OUT="$OUT" \
    python "$BENCH/bench/self_repair/rsi.py" 2>&1 | tee -a "$OUT/driver.out"
  ;;
status)
  grep -h "REPAIR\|RESULT\|data[12]:\|trained" "$HOME"/runs/*/*.log "$HOME"/runs/*/rsi.log 2>/dev/null || echo "nothing yet"
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
