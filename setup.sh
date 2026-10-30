#!/usr/bin/env bash
set -euo pipefail

# Symbio one-line installer:
#
#     curl -fsSL https://raw.githubusercontent.com/huyedits/Symbio/main/setup.sh | bash
#
# Clones (or updates) the repository, then hands over to ./install.sh, which
# does the real work: platform checks, its own virtualenv, the package with
# the MLX engine, and the browser engine.
#
# This used to install from requirements.txt by itself, which had drifted from
# pyproject.toml: no fastmcp, fastapi or anthropic, so the app failed on import,
# and no `symb` command either. One installer now, and this only fetches it.
#
#   INSTALL_DIR   where to put the checkout (default: ~/Symbio)
#   PYTHON        interpreter for the virtualenv (default: python3)
#   Any other arguments are passed to install.sh, e.g.  | bash -s -- --no-browser

REPO_URL="${SYMBIO_REPO:-https://github.com/huyedits/Symbio.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/Symbio}"

die() { printf ' fail %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "Symbio runs on macOS with Apple silicon only (found $(uname -s))."
command -v git >/dev/null 2>&1 || die "git is missing. Install the command line tools: xcode-select --install"

if [ -d "$INSTALL_DIR/.git" ]; then
    echo "==> Updating $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --ff-only \
        || echo "    (could not fast-forward; keeping your local checkout as it is)"
elif [ -e "$INSTALL_DIR" ]; then
    die "$INSTALL_DIR exists and is not a Symbio checkout. Set INSTALL_DIR=/another/path"
else
    echo "==> Cloning into $INSTALL_DIR"
    git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
# Piped from curl, this script's stdin is the script itself. Give install.sh
# the terminal back so it can drop the user into the ready environment.
if [ -r /dev/tty ] && [ -t 1 ]; then
    exec ./install.sh "$@" < /dev/tty
fi
exec ./install.sh "$@"
