#!/usr/bin/env bash
# Keep a viz server in sync with origin/main (run next to `bench/viz.py serve`, e.g. in tmux):
#   every INTERVAL seconds: git pull --ff-only, then regenerate the pages of every listed compiler
#   whose compiler.py changed since its last generation (sha1 stamp in viz/.stamp-<name>).
# Env: COMPILERS (space separated), INTERVAL (s, default 300), PYTHON (default .venv/bin/python),
#      JOBS (default 2), SET (metric|sanity|all, default metric).
set -uo pipefail
cd "$(dirname "$0")/.."
COMPILERS="${COMPILERS:-systolic systolic_po optimizer-optimizer optimizer-optimizer_po hubstream rectbatch gates naive}"
INTERVAL="${INTERVAL:-300}"
PYTHON="${PYTHON:-.venv/bin/python}"
JOBS="${JOBS:-2}"
SET="${SET:-metric}"
mkdir -p viz
while true; do
  git pull --ff-only -q || echo "viz_sync: git pull failed (local changes?), using the current checkout"
  for c in $COMPILERS; do
    f="compilers/$c/compiler.py"
    [ -f "$f" ] || continue
    h=$(sha1sum "$f" | cut -c1-12)
    stamp="viz/.stamp-$c"
    if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$h" ]; then
      echo "viz_sync: $(date +%H:%M:%S) regenerating $c ($h)"
      if "$PYTHON" bench/viz.py all "$f" --set "$SET" --jobs "$JOBS" > "viz/.log-$c" 2>&1; then
        echo "$h" > "$stamp"
      else
        echo "viz_sync: $c failed, see viz/.log-$c"
      fi
    fi
  done
  sleep "$INTERVAL"
done
