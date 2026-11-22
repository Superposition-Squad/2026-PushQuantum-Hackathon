#!/usr/bin/env bash
# Long, resumable MIP-LNS mining on the current best pipeline (relay_best_l3).
# Restart the same command to resume: cached plans are reused, finished circuits are skipped,
# unfinished ones continue from their checkpoint (l3/out/mine_relay_best_l3/<circuit>.ckpt).
# Live results: l3/out/mine_relay_best_l3/<circuit>.log (improvements), <circuit>.json (finished).
set -u
cd "$(dirname "$0")/.."
PY=/home/anton/Documents/code/PushQuantumHackathon2026/2026-PushQuantum-Hackathon/.venv/bin/python
OUT=l3/out/mine_relay_best_l3
mkdir -p "$OUT"
n=$(ls l3/cache/relay_best_l3/*.pkl 2>/dev/null | wc -l)
if [ "$n" -lt 20 ]; then
  echo "$(date -Is) caching relay_best_l3 ($n/20 present)"
  $PY l3/cache.py relay_best_l3 --jobs 2 --small-first 2>&1 | grep -v Warn
fi
echo "$(date -Is) mining"
files=$(ls -S -r l3/cache/relay_best_l3/*.pkl)
$PY l3/mine.py "$OUT.json" 3 5 $files 2>&1 | grep -v Warn
echo "$(date -Is) done"
