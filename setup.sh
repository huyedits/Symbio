#!/usr/bin/env bash
set -euo pipefail

# Symbio quick setup script (Hermes-style one-liner installer)
# Usage: curl -fsSL https://raw.githubusercontent.com/huyedits/Symbio/main/setup.sh | bash
#
# Adjustable via environment variables:
#   INSTALL_DIR            where to install (default: ~/Symbio)
#   PYTHON                 python interpreter to use (default: python3)
#   SYMBIO_USER_NAME       your name (default: prompted, else "User")
#   SYMBIO_ASSISTANT_NAME  the assistant's name (default: prompted, else "Symbio")

REPO_URL="https://github.com/huyedits/Symbio.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/Symbio}"
PYTHON="${PYTHON:-python3}"

echo "==> Installing Symbio into $INSTALL_DIR"

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR already exists. Updating..."
    cd "$INSTALL_DIR"
    git pull --ff-only || true
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo "==> Creating virtual environment"
$PYTHON -m venv venv
source venv/bin/activate

echo "==> Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Setting identity"
# Prompt on the terminal even when this script is piped from curl.
# Env vars win; empty answers keep an existing config value or the default.
if [ -z "${SYMBIO_USER_NAME:-}" ] && [ -r /dev/tty ]; then
    read -r -p "Your name [User]: " SYMBIO_USER_NAME < /dev/tty || SYMBIO_USER_NAME=""
fi
if [ -z "${SYMBIO_ASSISTANT_NAME:-}" ] && [ -r /dev/tty ]; then
    read -r -p "Assistant name [Symbio]: " SYMBIO_ASSISTANT_NAME < /dev/tty || SYMBIO_ASSISTANT_NAME=""
fi
export SYMBIO_USER_NAME="${SYMBIO_USER_NAME:-}"
export SYMBIO_ASSISTANT_NAME="${SYMBIO_ASSISTANT_NAME:-}"

# Write config and seed identity notes with the chosen names (venv python).
python - <<'PY'
import os
from symbio import load_config, save_config, ensure_seed_notes
from symbio.config import _write_identity_notes

config = load_config()
config["user_name"] = os.environ.get("SYMBIO_USER_NAME") or config.get("user_name") or "User"
config["assistant_name"] = os.environ.get("SYMBIO_ASSISTANT_NAME") or config.get("assistant_name") or "Symbio"
config["first_run"] = False
save_config(config)
_write_identity_notes(config["assistant_name"], config["user_name"])
ensure_seed_notes(config)
print(f"Assistant: {config['assistant_name']}")
print(f"User:      {config['user_name']}")
PY

echo ""
echo "==> Setup complete. Start Symbio with:"
echo "    cd $INSTALL_DIR"
echo "    source venv/bin/activate"
echo "    python main.py"
