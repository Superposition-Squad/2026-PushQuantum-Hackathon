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
