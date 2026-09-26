#!/usr/bin/env bash
# Symbio Pet — a cat on the desktop that draws the fine-tune as it happens.
#
#   ./symbio_pet/launch.sh             # watch this install's model and training
#   ./symbio_pet/launch.sh --demo      # a scripted run; nothing is trained
#
# Or once installed: symbio-pet [--demo] [--port 8742]
#
# The model is not in this process. The pet reads what the daemon, the trainer
# and the golden gate leave on disk, so it costs a window and a timer, not a
# copy of the weights.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
  source venv/bin/activate
fi

python -m symbio_pet "$@"
