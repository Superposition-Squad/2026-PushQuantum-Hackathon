# fanframe: experiments beyond solo (2026-11-22)

`compiler.py` is a fork of `compilers/solo` (snapshot ~02:30) with experimental options. Default
`MODE = "best"` runs the frame path and the solo path and keeps the cheaper; at 50q that exceeds
300 s, so use `MODE = "solo"` (= solo snapshot) or `"frame"` for measurements.

## Ideas tried, with measured results

| idea | where | result |
|---|---|---|
| **Fan-out Clifford frames.** U = V^dag (prod exp(-i th V P V^dag)) V with V = prod CNOT(hub->i): every hub term Z_a Y_i becomes the 1q Y_i, every non-hub term Z_u Y_v is unchanged; G = K^dag F K for hub-as-target terms | `fanframe_ops`, `OneQ`, `compile_ops` | exact (statevector + bench replay), valid. **Break-even**: two hub stars cost the same as one layer of hub blocks (688.7 vs 689.7 at 50q). Stagger p3 needs 6 stars = 6(n-1) hub CZ, same as 3 hub layers. Tree circuits: hub-Z and hub-Y terms always anticommute, so ~7 frame switches per layer (bad). Gains only if a fan-out is much cheaper than a serial star |
| relocation through the empty zone | `_relocate_zone`, `VIA_ZONE` | 3-4x worse than storage relocation (drops fragment) |
| static "zip" layouts (rounds as order-preserving merges of two row segments, no relocation) | scratch estimate | ~10k vs 1520 on 50q stagger p3 non-hub: the graph is too irregular |
| merge repeated rotations across layers by commutation | scratch | 0 merges on all p3 circuits |
| beam search over steps (round composition, hub batch vs round) and over band-plan rank | `beam_search`, `run_forced`, `PLAN_RANK` | <= 2.4 % (12q stagger p3), else 0: solo's greedy is at the optimum of its decision space |
| SA over the target layout of single transitions (with solo's and with a batching-aware planner) | scratch | no gain |
| batching-aware relocation planner for fixed targets | scratch `planner.py` | no gain: solo's targets have no batchable structure |
| 10x solo search budget | scratch | no gain |
| **snake mode**: route edges (the non-hub graph is a Hamiltonian path 0-1-...-n-2 plus random chords) are neighbours in the boustrophedon layout; a round of all ready route edges of one parity is one rigid lift with route-edge dummies | `SNAKE`, `_snake_round` | round 0 costs 0.8 per block (solo already gets this in layer 1). Later layers: route edges become ready piecemeal and the snake is scrambled by chord relocations; restoring it costs more than it saves; holding route edges back makes it worse (+5-12 %) |
| hub parked at the nearest free storage site after a hub batch (instead of its far home) | `HUB_PARK_NEAR`, extra candidate in `compile_ihva` | 50q stagger p3 2617.4/330 vs 2610.1/343 (-32.5 cost); full set +91.59 (fast) vs solo +91.60 (snapshot older than solo's latest); worth adding to solo as a candidate |
| "moves-first": from solo's round-start states, find collective row-segment shifts that make ready blocks adjacent | scratch | at most 1 new ready adjacency per move (often 0) with 22-39 ready blocks: no batching possible once the layout is scrambled |

## Measured cost anatomy (solo)

- Pulse count already equals the hub floor (every pulse layer has a hub CZ). At 12q each pulse is
  exactly one rigid band lift + one rigid drop (1.2 each).
- **60-72 % of distance is storage relocation between rounds**, mostly single-atom moves:
  relocation cost is ~87 % of the sum of individual displacements (no batching), while the
  perfect-batching bound is 3-10x lower. 50q: bands reach storage row 4 (4 rigid moves of ~6
  each per round) plus ~52 relocation per round.
- Hub batches (50q stagger p3, 294 hub pulses, 1106): partner lift+drop 652 (at the floor,
  1.11 each), hub hops 249, row rotations 144, hub to/from storage 62.
- Depth is 10-20 % above the hub floor: critical path of 50q tree p3 = 270 hub CZ + 26 non-hub
  CZ + 43 leaf 1q + 22 hub 1q (role switches).

## What would be needed for a big step

95 % needs about half of today's distance. Under the idle rule each pulse change costs >= 2 moves
of >= 1.0, pulses >= hub CZ count, so the remaining lever is making transitions batchable, i.e.
a layout that stays structured across rounds although half of the non-hub edges are random
chords. None of the structured choreographies above survived the chords.
