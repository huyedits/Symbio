#!/usr/bin/env bash
# Symbio Desktop — launch the premium adapter ecosystem visualizer.
#
# Usage:
#   ./symbio_desktop/launch.sh              # start server + open browser
#   ./symbio_desktop/launch.sh --no-open    # start server only
#
# Or once installed:
#   symbio-desktop
#   symbio-desktop --port 9000 --no-open

set -euo pipefail

cd "$(dirname "$0")/.."

# Use venv if present
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Ensure deps
if ! python -c "import fastapi, websockets" 2>/dev/null; then
  echo "  → Installing fastapi + uvicorn + websockets..."
  pip install fastapi uvicorn websockets -q
fi

NO_OPEN=""
if [ "${1:-}" = "--no-open" ]; then
  NO_OPEN="--no-open"
fi

python -m symbio_desktop.cli --port 8742 $NO_OPEN
