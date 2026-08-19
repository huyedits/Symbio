#!/usr/bin/env bash
set -euo pipefail

# Symbio installer.
#
# One command, from a clean machine:
#
#     ./install.sh
#
# It creates its own virtualenv, installs into it, fetches the browser engine,
# and tells you exactly what to run next. Safe to re-run.
#
# The previous version assumed you had already made and activated a venv and
# went straight to `pip install -e`. On a current macOS that either fails with
# PEP 668 ("externally-managed-environment") or quietly installs into the
# system Python, which is worse. Everything below exists because it was a way
# the install could fail without saying why.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

VENV="${SYMBIO_VENV:-$REPO_ROOT/venv}"
WITH_NATIVE="${WITH_NATIVE:-0}"
WITH_BROWSER="${WITH_BROWSER:-1}"
PREFETCH_MODEL="${PREFETCH_MODEL:-0}"
DEV="${DEV:-0}"
ENTER_SHELL="${ENTER_SHELL:-1}"

RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s  ok%s  %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s warn%s %s\n' "$YEL" "$OFF" "$*"; }
die()  { printf '%s fail%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

  --with-native      Also install the experimental symbio_native extras
  --no-browser       Skip the Playwright browser download (~150 MB)
  --prefetch-model   Download the 4.1 GB model now instead of on first run
  --dev              Include dev/test dependencies
  --no-shell         Do not drop into the environment when finished
  --venv PATH        Use a different virtualenv path (default: ./venv)
  -h, --help         This message

Environment equivalents: WITH_NATIVE=1 WITH_BROWSER=0 PREFETCH_MODEL=1 DEV=1
                         ENTER_SHELL=0
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-native) WITH_NATIVE=1 ;;
        --no-browser) WITH_BROWSER=0 ;;
        --prefetch-model) PREFETCH_MODEL=1 ;;
        --dev) DEV=1 ;;
        --no-shell) ENTER_SHELL=0 ;;
        --venv) shift; VENV="${1:?--venv needs a path}" ;;
        -h|--help) usage ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
    shift
done

say "=== Symbio installer ==="
say ""

# ---------------------------------------------------------------- preflight
# Checked up front, because every one of these otherwise surfaces as a
# confusing error several minutes into a large download.

case "$(uname -s)" in
    Darwin) ;;
    *) die "Symbio runs on macOS. MLX is Apple-silicon only, and there is no
       CPU fallback — this will not work on $(uname -s)." ;;
esac

if [ "$(uname -m)" != "arm64" ]; then
    die "Apple silicon required (found $(uname -m)). MLX needs an M-series chip;
       an Intel Mac cannot run this."
fi
ok "macOS on Apple silicon ($(uname -m))"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || die "no python3 on PATH. Install it with: brew install python@3.12"

# `|| echo 0` matters: under `set -e` a python that exists but cannot run
# (wrong arch, broken symlink, a shim that exits non-zero) would abort the
# script on this line with no message at all — the installer would simply
# stop, mid-preflight, looking like it had finished.
PY_OK="$("$PY" -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)' 2>/dev/null || echo 0)"
[ "$PY_OK" = "1" ] || die "need a working Python 3.10+. \`$PY\` reported: $("$PY" -V 2>&1 || echo 'it failed to run at all')
       Try: brew install python@3.12   (or set PYTHON=/path/to/python3)"
ok "$("$PY" -V 2>&1)"

# The model alone is 4.1 GB, plus ~1.5 GB of wheels. Fail now, not at 90%.
NEED_GB=8
FREE_GB="$(df -g . | awk 'NR==2 {print $4}')"
if [ "${FREE_GB:-0}" -lt "$NEED_GB" ]; then
    die "need ~${NEED_GB} GB free, found ${FREE_GB} GB.
       The model is 4.1 GB and the Python packages are ~1.5 GB."
fi
ok "${FREE_GB} GB free"

TOTAL_RAM_GB="$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
if [ "$TOTAL_RAM_GB" -lt 16 ]; then
    warn "${TOTAL_RAM_GB} GB of RAM. The default 8B model wants 16 GB;
       pick a smaller model in the setup wizard or it will swap badly."
else
    ok "${TOTAL_RAM_GB} GB RAM"
fi

say ""

# ------------------------------------------------------------------- venv
# Made here rather than assumed, so a clean machine needs exactly one command
# and nobody has to remember to activate anything first.
if [ -d "$VENV" ]; then
    ok "reusing virtualenv at $VENV"
else
    say "  creating virtualenv at $VENV"
    "$PY" -m venv "$VENV" || die "could not create a virtualenv at $VENV"
    ok "virtualenv created"
fi

VPY="$VENV/bin/python"
[ -x "$VPY" ] || die "virtualenv looks broken: no $VPY. Remove $VENV and re-run."

say "  upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip

# ---------------------------------------------------------------- install
EXTRAS=""
if [ "$WITH_NATIVE" = "1" ] && [ "$DEV" = "1" ]; then EXTRAS="[native,dev]"
elif [ "$WITH_NATIVE" = "1" ]; then EXTRAS="[native]"
elif [ "$DEV" = "1" ]; then EXTRAS="[dev]"
fi

say "  installing symbio${EXTRAS} ${DIM}(a few minutes; torch alone is ~535 MB)${OFF}"
"$VPY" -m pip install --quiet -e ".${EXTRAS}" || die "pip install failed. Full output:
       $VPY -m pip install -e \".${EXTRAS}\""
ok "symbio installed"

# ---------------------------------------------------------------- browser
# Playwright ships a Python package and a separate browser binary. Installing
# only the first leaves every browser tool failing at runtime with a message
# about a missing executable, long after the install "succeeded".
if [ "$WITH_BROWSER" = "1" ]; then
    if "$VPY" -c 'import playwright' 2>/dev/null; then
        say "  installing the Chromium engine for browser tools (~150 MB)"
        if "$VPY" -m playwright install chromium >/dev/null 2>&1; then
            ok "browser engine installed"
        else
            warn "could not install the Chromium engine. Browser tools will not work
       until you run: $VPY -m playwright install chromium"
        fi
    fi
else
    warn "skipping the browser engine (--no-browser). Browser tools will not work
       until you run: $VPY -m playwright install chromium"
fi

# ------------------------------------------------------------------ model
if [ "$PREFETCH_MODEL" = "1" ]; then
    say "  downloading the model now (4.1 GB)"
    "$VPY" - <<'PY' || warn "model prefetch failed; it will download on first run instead"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B-MLX-4bit")
PY
    ok "model cached"
fi

# ------------------------------------------------------------------- done
say ""
say "=== Done ==="
say ""

# Drop the user straight into the environment.
#
# A script cannot activate a venv in the shell that launched it — it runs in
# its own process, and its exports die with it. Telling people to remember
# `source venv/bin/activate` afterwards is the friction, so instead we exec a
# new shell that already has it, the way nix-shell does. The user's original
# shell is untouched: this replaces the installer's own process, so typing
# `exit` returns them exactly where they started.
#
# Only with a terminal on both ends. In CI or a pipe there is nobody to land
# in a shell, and exec'ing one there would hang the job.
if [ "$ENTER_SHELL" = "1" ] && [ -t 0 ] && [ -t 1 ]; then
    say "Entering the Symbio environment. ${DIM}Type 'exit' to leave.${OFF}"
    say ""
    say "    ${GRN}symbio${OFF}    start it ${DIM}(first run: setup wizard, then a 4.1 GB"
    say "              model download and ~25s of cache warming; later"
    say "              starts are about a second)${OFF}"
    say ""

    # What bin/activate does, minus the prompt rewriting, which differs per
    # shell and is not ours to mangle.
    VIRTUAL_ENV="$VENV"; export VIRTUAL_ENV
    PATH="$VENV/bin:$PATH"; export PATH
    unset PYTHONHOME 2>/dev/null || true
    export SYMBIO_ENV=1

    exec "${SHELL:-/bin/zsh}"
fi

say "Start it with:"
say ""
say "    $VENV/bin/symbio"
say ""
say "  ${DIM}First launch runs the setup wizard, then downloads the model (4.1 GB)"
say "  and spends ~25s warming its cache. Later starts are about a second.${OFF}"
say ""
say "  ${DIM}Or activate the environment yourself:"
say "      source $VENV/bin/activate${OFF}"
say ""
