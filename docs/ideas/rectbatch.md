# Idea: rectbatch

- Branch: idea/rectbatch  ·  Worktree: /home/hackathon/challenge-wt/rectbatch
- Base: main 0ea558b, starting from `compilers/naive/compiler.py`
- Status: in progress

## Hypothesis

The naive compiler pays 2 moves per CZ atom (4 per CZ, ~8.3 distance per move). Under our metric a
move costs the *max* displacement of its atoms, so carrying many atoms in one rectangle is almost
free. If we (a) schedule the native circuit into **rounds** of disjoint CZ pairs, (b) place qubits so
that the partners of a round sit in the same storage row at adjacent columns, and (c) lift each
round with as few rectangle moves as possible (ideally one), the distance per round drops to
`2 · (10 − y_top)` where `y_top` is the topmost storage row involved — about 2.1–2.5 for the bottom
two rows, independent of how many pairs ride along.

Target: distance −85 % or better, depth −40 % (from `optimization_level=2` resynthesis plus
round scheduling). Score estimate 70–80.

## Geometry facts this design rests on

- storage: physical `(x, y) = (col, row)`. entangling: `y = row`, `x = 2·(col//2) + 0.3·(col%2)`.
- An interaction pair is `(2k, 2k+1)` in one entangling **row**; a `cz` needs both atoms there.
- A move picks a **contiguous** row range × **contiguous** col range; every *occupied* intersection
  must be listed (ghosts are illegal), empty rows/cols inside the range are silently dropped
  (`_remove_empty` squeeze), targets must be ascending (order preserving), and *every* intersection's
  target (including empty ones) must lie in a zone — so a move must not mix zones.
- Lifting storage rows `y..y+k` to entangling rows `10..10+k` gives every atom the same vertical
  displacement `10 − y`: **one move for 10·(k+1) atoms at cost ≈ 10 − y**.
- Our own transversal check counts a pair as occupied only if *both* sites are filled, so a lone atom
  on a pair site is free (bench/bench.py:115).

## Design

Five separable stages (so the orchestrator can lift any of them out):

1. `native_circuit(circuit)` — `transpile(basis_gates=["rx","rz","cz"], optimization_level=2)`,
   barriers dropped. No layout/routing pass runs (no coupling map), so qubit indices are preserved.
2. `schedule_rounds(native)` — ASAP layering of the per-qubit DAG. Repeatedly: emit every ready 1q
   op (storage phase), then take a greedy **maximal matching** of the ready `cz` gates as the next
   round. Output `[(ops_1q, [(u,v), ...]), ...]`. Any topological order of the original DAG is
   emitted, so equivalence holds by construction (ADR 0003). At most one `cz` per pair per round, so
   every occupied pair gets exactly one pulse per segment (ADR 0007).
3. `assign_targets(pairs, pos)` — give every round pair an interaction pair `(row, 2k/2k+1)`.
   Pairs whose atoms share a storage row keep their column band (small DP over the ≤5 slots,
   minimising the max displacement) and get entangling row `10 + rank(y)`; cross-row pairs get their
   own entangling row.
4. `plan_moves(round, placement)` — build moves **agglomeratively**: start with one move per pair
   (or one per atom for cross-row pairs), then repeatedly merge two moves when the merged rectangle
   is legal (`try_merge` re-implements the validator's rules: contiguous picked lines, monotone
   row/col maps, no ghosts, no occupied targets) and the merged cost (max displacement) is below the
   sum of the parts. Return moves are the exact inverses in reverse order. Cost of a plan =
   Σ max-displacement, which is exactly the scored metric, and is used as the placement objective.
5. `optimise_placement(rounds, n)` — greedy construction: weight `(u,v)` by how often they are
   partners, greedily match, lay matched partners into storage columns `(2k, 2k+1)` of the same row,
   fill rows from 9 upward with the busiest qubits lowest. Then local search (swap two sites, keep if
   `sum(move_cost)` drops) under a wall-clock budget.

## Legality / equivalence argument

- Only `rx`, `rz`, `cz`, `move`, `measure` are emitted; 1q ops only while all atoms are in storage,
  `cz` only on atoms the planner just put on one interaction pair (ADR 0002).
- Gate order is a topological order of the native DAG → truly equivalent at every size (ADR 0003).
- Each round emits exactly one `cz` per occupied pair between two moves; atoms that ride along land
  on lone pair slots → transversal rule holds (ADR 0007).
- `try_merge` is a reimplementation of `_validation.site_mapping`/`apply_move` checks, and the bench
  replay is the oracle.

Sources: 2505.22715 (grouping compatible moves), 2509.12083 (composite moves), 2405.15095 (edge
colouring / matchings), 2405.06183 (cross-minimisation → monotone AOD maps), 2411.11784 (reuse),
2611.19994 (incremental placement cost).

## Results

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|

## Verdict

TBD
