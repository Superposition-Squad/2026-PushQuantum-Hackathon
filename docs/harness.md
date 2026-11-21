# Local harness

Everything runs from the repo root with the repo's uv env (`uv sync` once; Python 3.12).
Worktrees have no `.venv`, so use the main repo's interpreter from inside a worktree:
`/home/hackathon/challenge/.venv/bin/python bench/bench.py ...`.

## Commands

```sh
# score a compiler on the 36 metric circuits (official score), fast mode, 2 workers
uv run python bench/bench.py compilers/<idea>/compiler.py

# quick iteration on a subset (partial score over the selection)
uv run python bench/bench.py compilers/<idea>/compiler.py --filter 12q
uv run python bench/bench.py compilers/<idea>/compiler.py --filter qaoa

# sanity circuits (13 small/odd circuits; not scored, but they must pass)
uv run python bench/bench.py compilers/<idea>/compiler.py --set sanity --mode full

# merge gate: everything, every check
uv run python bench/bench.py compilers/<idea>/compiler.py --set all --mode strict --jobs 4

# rank ideas by their latest full metric run (results/<name>.json)
uv run python bench/compare.py [--per-circuit]

# build the submission notebook and score it exactly as the leaderboard imports it
uv run python bench/package.py compilers/<idea>/compiler.py --score
```

## Modes

| mode | compile + replay + metrics | transversal-CZ check | official equivalence | extra equivalence (> 19 q) | typical time (naive, 36 circuits) |
|---|---|---|---|---|---|
| fast | ✓ | ✓ | – | – | ~8 s with 6 workers |
| full | ✓ | ✓ | ✓ (QCEC ≤ 19 q, interaction pairs above) | – | about 1 min |
| strict | ✓ | ✓ | ✓ | statevector ×3 random inputs (≤ 22 q), QCEC with 120 s timeout (50 q) | several minutes |

Compile time counts too: circuits take ≤ 300 s each in the real scorer, so keep the 50-qubit p=3
circuits under about 60 s locally.

## Outputs

- `results/<name>.json`: the latest full-metric run of a compiler (name defaults to its folder).
  Commit it with the compiler change.
- `results/runs/*.json`: every run (gitignored).
- Each record has a label, file, family, metrics (`num_moves`, `move_steps`, `move_distance`,
  `move_time`, `depth`, `num_operations`, `spacing_warnings`), equivalence status, timings and any
  error.

## Correctness of the harness

`compilers/naive/compiler.py` reproduces `vendor/pushquantum_validator/naive_baseline.json` exactly
on all 49 circuits and all six metrics, and scores +0.00 %. If you change the harness, re-check this:

```sh
uv run python bench/bench.py compilers/naive/compiler.py --set all --jobs 6
```
