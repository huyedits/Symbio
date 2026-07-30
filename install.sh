#!/usr/bin/env bash
set -euo pipefail

# Symbio install script.
# Keeps symbio_native/ by default. Remove it with SKIP_NATIVE=1 or by saying
# "no" at the prompt.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== Symbio installer ==="

if [ "${SKIP_NATIVE:-}" = "1" ]; then
    echo "SKIP_NATIVE=1 -> removing symbio_native/"
    rm -rf symbio_native
else
    # Ask unless already forced via environment.
    if [ -z "${KEEP_NATIVE:-}" ]; then
        read -rp "Keep the experimental symbio_native model? [Y/n] " answer </dev/tty || true
        case "${answer:-Y}" in
            [Nn]*)
                echo "Removing symbio_native/"
                rm -rf symbio_native
                ;;
            *)
                echo "Keeping symbio_native/"
                KEEP_NATIVE=1
                ;;
        esac
    else
        echo "KEEP_NATIVE set -> keeping symbio_native/"
    fi
fi

if [ -d symbio_native ]; then
    echo "Installing with native model dependencies..."
    pip install -e ".[native,dev]"
else
    echo "Installing without native model dependencies..."
    pip install -e ".[dev]"
fi

echo "=== Done ==="
