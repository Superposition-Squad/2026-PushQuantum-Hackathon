#!/usr/bin/env bash
# Fast dev iteration for compilers/optimizer: benches a scratch copy with DEV_FAST = True
# (one plan per circuit, no variant search). Extra args go to bench.py (e.g. --filter 50q).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
mkdir -p "$tmp/optimizer_dev"
sed 's/^DEV_FAST = False/DEV_FAST = True/' "$root/compilers/optimizer/compiler.py" > "$tmp/optimizer_dev/compiler.py"
grep -q '^DEV_FAST = True' "$tmp/optimizer_dev/compiler.py"
cd "$root"
.venv/bin/python bench/bench.py "$tmp/optimizer_dev/compiler.py" --name optimizer_dev --jobs "${JOBS:-6}" "$@"
