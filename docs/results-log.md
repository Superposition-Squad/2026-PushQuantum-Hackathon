# Results log (orchestrator)

Official score = 36 metric circuits, `bench/bench.py --mode full` or stricter. Naive = 0.00 %.

## Round 0: harness (2026-11-21 ~12:40 UTC)

| compiler | score | dist % | depth % | validity | notes |
|---|---|---|---|---|---|
| naive | +0.00 | +0.00 | +0.00 | full ✓ | reproduces `naive_baseline.json` exactly (49/49 circuits, all metrics) |

## Round 1 (started 12:55 UTC, time box 15:30 UTC)

Worktrees `/home/hackathon/challenge-wt/<idea>`, branches `idea/<idea>` from `0ea558b`.

| idea | literature # | lane |
|---|---|---|
| gates | 1, 2, 3, 11 | exact resynthesis, hub-as-control, commutation-aware CZ rounds, `schedule_rounds()` |
| hubstream | 5, 6, 2 | park the hub in the zone, stream partners from row 9 at cost ≈ 1.0 |
| rectbatch | 6, 7, 10 | generic rectangle batching of each round, placement optimisation |
| zonedwell | 4, 8, 9 | keep atoms in the zone, in-zone re-pairing, batch staging |

| idea | score | dist % | depth % | validity | commit | verdict |
|---|---|---|---|---|---|---|
| gates | +72.87 | +71.72 | +77.47 | full ✓ 36+13; strict 33/36 so far, no failures | c82ea11 | gate layer is the reference (commutation DAG, hub-as-control, ZXZ merge, in-zone retention); `gatesb` variant +66.39 (batching, HEA dist +75.7) |
| rectbatch | +77.54 | +77.99 | +75.74 | full ✓ 36+13; strict 20q ✓ | 953450e | validator-exact move checker, visitor-first lifting, placement search. **Wall-clock-budgeted search → ±0.2 noise; must be made deterministic before merging** |
| hubstream | +81.51 | +83.80 | +72.33 | full ✓ 36+13; strict running | 87210ad | hub parking (hub moves 2× per circuit), keep atoms in zone when in-between 1q = identity, exact θ=0 elision, reserved hub pair. 50q still ~1.3 atoms/move. **Round-1 leader** |
| zonedwell | +79.59 (fa42f55) | +80.25 | +76.95 | full ✓ 36+13; strict running | fa42f55 | conveyor with hub; final tweak pending |
