#!/usr/bin/env bash
set -euo pipefail

# Symbio install script.
# Keeps symbio_native/ by default. Remove it with --skip-native or SKIP_NATIVE=1.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

SKIP_NATIVE="${SKIP_NATIVE:-0}"
KEEP_NATIVE="${KEEP_NATIVE:-0}"
INTERACTIVE="${INTERACTIVE:-auto}"

usage() {
    echo "Usage: $0 [--skip-native] [--keep-native]"
    echo "  --skip-native   Remove symbio_native/ before installing"
    echo "  --keep-native   Keep symbio_native/ without prompting"
    echo "  SKIP_NATIVE=1   Same as --skip-native"
    echo "  KEEP_NATIVE=1   Same as --keep-native"
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --skip-native) SKIP_NATIVE=1 ;;
        --keep-native) KEEP_NATIVE=1 ;;
        -h|--help) usage ;;
    esac
done

echo "=== Symbio installer ==="

if [ "$SKIP_NATIVE" = "1" ]; then
    echo "SKIP_NATIVE set -> removing symbio_native/"
    rm -rf symbio_native
elif [ "$KEEP_NATIVE" = "1" ]; then
    echo "KEEP_NATIVE set -> keeping symbio_native/"
else
    # Try interactive prompt; fall back to keeping native if no TTY.
    if [ "$INTERACTIVE" = "auto" ] && [ -t 0 ] && [ -t 1 ]; then
        read -rp "Keep the experimental symbio_native model? [Y/n] " answer < /dev/tty || true
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
        echo "No TTY detected -> keeping symbio_native/ by default (use --skip-native to remove)"
        KEEP_NATIVE=1
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
