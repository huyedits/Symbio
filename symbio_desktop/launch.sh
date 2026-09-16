#!/usr/bin/env bash
# Symbio Desktop — chat window over the resident model.
#
#   ./symbio_desktop/launch.sh              # serve + open your browser
#   ./symbio_desktop/launch.sh --no-open    # serve only
#   ./symbio_desktop/launch.sh --window     # native macOS window (see below)
#
# Or once installed: symbio-desktop [--port 9000] [--no-open] [--window]
#
# There is nothing to install. The server is the standard library, and the
# model is not in this process at all — it lives in `symb daemon`, which this
# talks to over a Unix socket. Measured 28 MB idle, 31 MB after serving every
# endpoint. `--window` hosts WebKit in-process and costs ~400 MB all told, so
# it is off by default.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
  source venv/bin/activate
fi

python -m symbio_desktop.cli --port 8742 "$@"
