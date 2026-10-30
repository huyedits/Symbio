#!/bin/sh
# Copy the CODE (not your data) from this Mac to a GPU server.
#   sh server/push.sh root@203.0.113.7
#
# An allowlist, not a denylist: exactly the files git tracks plus new files
# git does not ignore, uncommitted edits included, so the server runs what
# was tested here. Everything .gitignore keeps out of the public repo — notes,
# sessions, memory.db, adapters, training data, config.json, soul.md — stays
# off the server too; a cloud box is someone else's computer.
set -eu
HOST=${1:?usage: push.sh user@host}
cd "$(dirname "$0")/.."
LIST=$(mktemp)
git ls-files -co --exclude-standard | grep -v '__pycache__' > "$LIST"
echo "sending $(wc -l < "$LIST" | tr -d ' ') files"
rsync -az --files-from="$LIST" ./ "$HOST:symbio/"
rm -f "$LIST"
echo "next: ssh $HOST 'sh symbio/server/setup_rocm.sh'"
