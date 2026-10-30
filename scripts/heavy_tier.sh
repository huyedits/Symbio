#!/usr/bin/env bash
# Heavy-tier inference sidecar for Symbio: mmap-streamed Qwen3.5-35B-A3B.
#
# Runs llama-server with the "streaming weights" profile: macOS pages expert
# tensors from NVMe on demand, so ~13GB on disk resolves to ~4-6GB resident
# weights. Leaves ~9GB of unified memory for the MLX headmaster, workers,
# and the desktop. Verified recipe (base M4 / 16GB): --n-gpu-layers 0 +
# --mmap + NO mlock (mlock pins file pages and defeats streaming entirely).
#
# Harness wiring (symbio/mcp/config.py reads these from .env):
#   OLLAMA_BASE_URL=http://127.0.0.1:8081
#   LOCAL_MODEL=qwen3.5-35b-a3b
#
# Tuning notes:
#   - THREADS=8 suits the 4P+6E M4; sweep 4..10 if throughput differs.
#   - Cold-expert page-ins make first token take 2-5s after idle; raise
#     agent.first_chunk_timeout_ms in config.json if wiring the main loop.
#   - Keep SSD free space >20% — paging bandwidth degrades on a full disk.

set -u

MODEL="${HOME}/.local/share/llama-models/Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="${PROJECT_DIR}/cache/heavy_tier.pid"
LOG_FILE="${PROJECT_DIR}/cache/heavy_tier.log"
PORT=8081
ALIAS="qwen3.5-35b-a3b"
THREADS="${HEAVY_TIER_THREADS:-8}"
CTX="${HEAVY_TIER_CTX:-16384}"

is_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

warn_if_lmstudio_busy() {
  if pgrep -f "lmstudio.*llama-server" >/dev/null 2>&1; then
    echo "NOTE: LM Studio's llama-server is running (holding wired memory)." >&2
    echo "      Stop it first if the residency budget looks tight." >&2
  fi
}

start() {
  if is_running; then
    echo "already running (pid $(cat "$PID_FILE")) on :${PORT}"
    return 0
  fi
  if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    echo "download it with:" >&2
    echo "  .trial-venv/bin/huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \\" >&2
    echo "    Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf --local-dir \"\$HOME/.local/share/llama-models\"" >&2
    return 1
  fi
  mkdir -p "${PROJECT_DIR}/cache"
  warn_if_lmstudio_busy
  nohup llama-server \
    --model "$MODEL" \
    --alias "$ALIAS" \
    --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$CTX" \
    --n-gpu-layers 0 \
    --load-mode mmap \
    --flash-attn on \
    --reasoning off \
    --threads "$THREADS" \
    --jinja \
    --no-webui \
    >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "started (pid $!) -> http://127.0.0.1:${PORT}/v1  (log: $LOG_FILE)"
  echo "watch residency: tail -f $LOG_FILE  (look for 'weights resident')"
}

stop() {
  if ! is_running; then
    echo "not running"
    rm -f "$PID_FILE"
    return 0
  fi
  kill "$(cat "$PID_FILE")" 2>/dev/null
  rm -f "$PID_FILE"
  echo "stopped"
}

status() {
  if ! is_running; then
    echo "not running"
    return 1
  fi
  echo "running (pid $(cat "$PID_FILE")) -> http://127.0.0.1:${PORT}/v1  model: $ALIAS"
  curl -s "http://127.0.0.1:${PORT}/health" | head -c 200 || true
  echo
}

test_run() {
  echo "prompting ${ALIAS} on :${PORT} ..."
  curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$ALIAS"'","messages":[{"role":"user","content":"Reply with exactly: SIDE CAR OK"}],"max_tokens":20,"temperature":0}' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print("usage:", r.get("usage"))'
}

logs() {
  tail -40 "$LOG_FILE"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  test) test_run ;;
  logs) logs ;;
  *)
    echo "usage: $0 {start|stop|restart|status|test|logs}" >&2
    exit 2
    ;;
esac