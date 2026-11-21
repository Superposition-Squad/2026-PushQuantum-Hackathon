#!/usr/bin/env bash
# Fast dev iteration for compilers/pulsebeam: benches a scratch copy with DEV_FAST = True
# (one plan per circuit, no variant search). Extra args go to bench.py (e.g. --filter 50q).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
mkdir -p "$tmp/pulsebeam_dev"
sed 's/^DEV_FAST = False/DEV_FAST = True/' "$root/compilers/pulsebeam/compiler.py" > "$tmp/pulsebeam_dev/compiler.py"
grep -q '^DEV_FAST = True' "$tmp/pulsebeam_dev/compiler.py"
cd "$root"
.venv/bin/python bench/bench.py "$tmp/pulsebeam_dev/compiler.py" --name pulsebeam_dev --jobs "${JOBS:-6}" "$@"
