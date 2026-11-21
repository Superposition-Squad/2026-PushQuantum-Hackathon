# Idea: hubstream

- Branch: idea/hubstream · Worktree: /home/hackathon/challenge-wt/hubstream
- Base: main 0ea558b, starting from `compilers/naive/compiler.py`
- Status: parked (finished time box; candidate for merge as the move planner of a combined compiler)

## Hypothesis

In iHVA and QAOA the last qubit `q[n-1]` is the Max-Cut reference "hub" with degree n−1. Its
2(n−1) CZs per cost layer are strictly sequential, so they set the number of zone round trips.
Two facts make it cheap:

1. If every ZZ core is oriented so that the **mid rotation lands on the partner**, the hub's wire
   carries *only CZs* for a whole cost layer. In iHVA the `rx(±π/2)` that wrap each block are
   adjacent **on the hub's wire** (nothing else touches the hub in between) and cancel. So the hub
   never needs a single-qubit gate inside a layer and can be **parked in the entangling zone**.
2. Everything else only has to cross storage↔entangling row 9↔10, i.e. 1.0 per crossing, and many
   atoms can cross in **one** `Move` (cost = max displacement, not sum).

Naive spends ≈ 28 distance units per CZ (50q p3 iHVA: 882 CZ, D = 24 907). A round-based compiler
where each round costs ~2 moves of ~1–4 should land at 3–8 per CZ ⇒ 75–90 % less distance, and
depth ≈ #rounds × ~4 instead of ~2 300.

Sources: 2411.11784 (reuse-aware zoned compilation / keeping qubits in the entangling zone),
2503.02272 (reduce 1q/2q interleaving, CZ chains), 2312.03982 (zoned model, transversal pulse),
2505.22715 / 2509.12083 (group compatible moves into one stroke), 2608.01316 (shortest zone hop).

## Design

### 0. Geometry facts used

- storage site `(c, r)` has physical `(c, r)`; entangling site `(c, r)` has physical
  `(2·(c//2) + 0.3·(c%2), r)`. Interaction pair = same row, columns `(2k, 2k+1)`.
- storage row 9 → entangling row 10 at an even column is exactly 1.0; at an odd column 1.22.
- A `Move` picks a contiguous ascending row range × column range and maps lines monotonically.
  If `new_cols` is shorter than `cols`, the **empty** picked columns are dropped first
  (`_validation.py:_remove_empty`), so an arbitrary monotone matching of occupied columns is legal.
  Non-proportional spacing only produces a *spacing warning* (time only, not scored).
- Every atom on a picked intersection must be listed (no ghosts); no deposit on a non-moving atom.

### 1. Front end — `parse_items(circuit)`

Produces a flat list of items in circuit order:

- `("u1", q, m)` — a 2×2 unitary (any single-qubit gate, via `op.to_matrix()`).
- `("zz", u, v, theta)` — `exp(-i·theta/2·Z_u Z_v)`, from
  (a) the consecutive pattern `cx(u,v); rz(theta) v; cx(u,v)` and (b) `rzz(theta) u,v`.
- `("cz", u, v)`.
- `("meas", q, clbit)`.

`cx(c,t)` that is not part of a ZZ block desugars to `u1(H,t), cz(c,t), u1(H,t)`.
If the circuit contains any gate outside `{1q, cx, cz, rzz, measure, barrier}` (sanity 12/13), the
**whole** circuit is first `transpile(..., basis_gates=["rx","rz","cz"], optimization_level=1)`d and
then parsed (generic fallback; no ZZ structure, one task per `cz`).

### 2. Wires and CZ tasks — `build_wires`

A ZZ block `(u,v,theta)` is synthesised with a chosen **rotation side** `s`:

    H_s · CZ(u,v) · RX(theta)_s · CZ(u,v) · H_s        (exactly equal: CX(u,s)=H_s CZ H_s, H RZ H = RX)

`s` = the non-hub endpoint (hub never takes the rotation). This is where hub parking comes from.
Per qubit we get a wire of entries `("g", m)` / `("cz", task)` / `("meas", clbit)`. Tasks are
numbered in circuit order. No reordering of non-commuting operations ever happens: the only
freedom used is (i) the orientation of a ZZ core (exact identity), (ii) interleaving operations on
disjoint qubits, (iii) merging single-qubit runs on one wire. ⇒ equivalence is exact at every size
(ADR 0003).

### 3. Scheduler — `schedule_rounds`

ASAP list scheduling on the gate-order DAG. `nextcz[q]` = first `cz` entry at/after the pointer;
task `t=(u,v)` is ready iff `nextcz[u] == nextcz[v] == t`. Each round takes a maximal set of ready
tasks with disjoint qubits, at most `MAX_PAIRS = 25` (5 zone rows), priority:
hub tasks → tasks whose atoms are already parked in the zone → task index.

Because in QAOA/iHVA each hub edge is the *last* block of its vertex, the hub's 2(n−1) tasks are
naturally spread over the whole layer and non-hub work rides in the same rounds — without any
commutation reasoning.

### 4. Mover — one round = lower / gates / lift / pulse

State: `pos[q]`, `occ[site]`. No fixed homes after the start.

1. **Pair assignment.** A task whose atom is parked keeps that atom's interaction pair; otherwise
   take the free pair minimising `max(dist(pos[u]→slot), dist(pos[v]→slot))`, scanning rows 10 up.
   The hub always sits in the **right** slot `(10, 2k+1)`, its partner in the left slot `(10, 2k)`
   (exactly 1.0 from storage `(2k, 9)`).
2. **Lower phase.** Every zone atom that is not already at its assigned slot for this round moves to
   storage: target = the **nearest free storage site** to its slot (self-organising: atoms cluster
   just under the zone).
3. **Gate phase.** For every qubit in storage, emit its pending single-qubit run up to its next
   `cz`/`meas`, merged into one 2×2 matrix and decomposed into ≤3 `rz/rx/rz`. Identity runs emit
   nothing — this is what makes the hub's `rx(π/2)·rx(−π/2)` vanish.
4. **Lift phase.** Storage → assigned slots.
5. **Pulse.** One `cz` per pair of the round. Assert: every interaction pair with two atoms is a
   pair of this round ⇒ ADR 0007 holds by construction (parked atoms are alone on their pair).

**Park rule.** After the pulse an atom stays in the zone iff its pending 1q run is identity and it
still has a next task. At most one atom per pair may stay, unless both atoms' next task is the same
task (the `theta = 0` warm-start blocks, where the two CZs are consecutive). Prefer the hub, then
the atom with more remaining tasks.

### 5. Move planner — `plan_transfer`

Input: `{q: (src, dst)}` for one phase. Group by (source row → target row); inside a group sort by
source column and cut a new run when (a) a non-moving atom sits between two moving columns
(ghost), or (b) the target column is not strictly increasing. Emit
`Move(rows=(r,), cols=(cmin..cmax), new_rows=(tr,), new_cols=targets)`.
Then merge runs of consecutive source rows with identical column spans/targets and consecutive
target rows into one multi-row move (HEA lifts 25 pairs in one stroke this way).

### 6. Placement

Initial loading is free, so a compact block `col = i % 10`, rows `r0 .. r0+ceil(n/10)-1` with
`r0 = max(0, 9 - ceil(n/10) - 2)`, leaving 2–3 free rows just under the entangling zone as landing
space. Hub parked pair = `(10, 0)/(10, 1)`; hub's partners are placed, in the order the hub uses
them, on the free storage sites nearest to `(10, 0)` (iteration 2).

### Validity argument

- Only `rx`, `rz`, `cz`, `move`, `measure` are emitted; `rx/rz` only while the atom is in storage,
  `cz` only on an interaction pair (ADR 0002, zone rules).
- Equivalence: see §2. Exact rewrites only, no reordering of non-commuting ops (ADR 0003).
- Transversal CZ: §4 step 5 asserts it (ADR 0007).
- Move legality: §5 constructs ghost-free monotone rectangles; the bench replay is the oracle.

## Results

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|
| naive | – | 0.00 | 0.0 | 0.0 | D = 276 726.91, P = 23 676 |
| step 1 (round template) | `83101da` | 76.93 | 78.41 | 71.01 | D = 59 758.6; naive rows 2–6 loading, 25 pairs/round, per-atom nearest lowering |
| + loading by first use, one-row rounds, packed lowering | `ff4d3b3` | 81.06 | 83.24 | 72.33 | D = 46 367.6, P = 6 550; `MAX_PAIRS = 4` |
| + end pair of row 10 reserved for the hub, all n | – | 81.35 | 83.61 | 72.33 | D = 45 359.8; 50q p3 stagger 5 981 → 5 579, but 20q qaoa p3 1 262 → 1 387 |
| + end pair only for n > 20 (final) | `6262b1b` | 81.51 | 83.80 | 72.33 | D = 44 825.4; hub at entangling (9,10), home (9,9); other tasks never use pair (8,9) of row 10 |
| + round 2: gates' commutation-DAG front end, per-circuit pick | `52039db` | **81.94** | 83.61 | **75.28** | D = 45 368.3, P = 5 854; both front ends compiled, cheaper `D + 2.92·depth` kept |

### Round 2 — commutation DAG front end (step 3r2)

`parse_items`/`build_wires`/`schedule_rounds` (the ASAP wire scheduler) are replaced as the *only*
front end by gates' pipeline where it pays; the mover (`emit`, parking, end pair, deferred rz) is
shared by both. Copied verbatim from `compilers/gates/compiler.py`: the Pauli/`_rx_mat`/`_rz_mat`/
`_BASIS_CHANGE` algebra, `Item`, `_native_plan` (levels 1 then 0, with an operator check — never
raised), `_next_on`, `lower_to_items`, `two_qubit_degree`, `expand_rot2`, `merge_1q_runs`,
`build_commutation_dag`. New: `choose_hub_dag`, `schedule_rounds_dag` (gates' scheduler + a round cap
+ the hub/warm key), `_dag_frontend` (phases → our wires/tasks/rounds), `_move_distance`,
`_gates_depth`.

`compile_circuit` now compiles a circuit with every candidate front end and keeps the one with the
lowest `D + 2.92·depth` (the portfolio exchange rate), computed from its own moves — no file-name or
family special cases. Candidates: the wire front end plus the DAG front end with round caps
{2,3,4} ({2} for n > 40, to bound compile time).

| candidate | score | dist % | depth % |
|---|---|---|---|
| wire front end only (6262b1b) | 81.51 | 83.80 | 72.33 |
| DAG only, cap 2 / 3 / 4 / 5 / 6 / 25 | 79.65 / 80.88 / **81.08** / 80.41 / 80.31 / 80.68 | – | 80.0–81.9 |
| pick over {wire, dag2, dag3, dag4} | **81.94** | 83.61 | 75.28 |

The `warm`-first key variant (`(hub-first, warm-first, −crit, i)`) gives **bit-identical** output on all
36 circuits (the hub-first and criticality terms already decide every round), so only key 0 is kept.
`MAX_PAIRS` sweep is the DAG-only row above; the pick wins 11 / 9 / 12 / 4 circuits for
wire / dag3 / dag4 / dag2. DAG-only depth is +81–82 % (vs +72.3 % for the wire path) but it pays
~15–25 % more distance on the 50q p3 circuits, which is why the pick (not a wholesale replacement)
is what ships. Compile time: worst metric circuit 4.2 s, 40q sanity 10.2 s, 70q sanity 12.2 s.

A hard pin of the hub to the end pair (instead of a reservation) failed on 10 circuits: after the
hub's mixer descent a parked atom could hold the pair when the hub came back ("no free interaction
pair"). Reserving the pair for hub tasks fixes it. Re-tuning `MAX_PAIRS` on top of it: 3 → 81.01,
**4 → 81.51**, 5 → 81.21 (depth identical in all three: gate-order ready sets rarely exceed 4 tasks).

Measured on the 8-circuit probe subset (score / distance), used to pick knobs:

- round size `MAX_PAIRS`: 3 → 80.45, **4 → 80.80**, 5 → 80.38, 6 → 80.12 (full metric set, fast mode).
- hub loading of §6/STEP 2 (`_loading_hub`: hub at storage `(1,9)`, partners queued nearest to
  entangling slot `(0,10)`) with `MAX_PAIRS = 4`: **80.53** vs first-use loading **80.80** — the
  first-use loading wins, so the hub-specific loading was dropped (the hub itself is lifted once and
  then never moves, so where its partners start matters less than starting the soonest-used atoms
  shallow).
- parking a non-hub atom only when its next task is ≤ K rounds away: K = 0 → 79.10, **K = 1 → 80.80**,
  K = 2 → 80.80, K = 5 → 80.57.
- forcing the hub onto the lowest-leftmost pair (so the rest of row 10 stays contiguous): 80.19.
- assigning each round to a contiguous *block* of pairs: 80.24 (the block constraint costs more in
  lift distance than the merged moves save).
- deferring the trailing `rz` of every single-qubit run across the next `cz` (it commutes): exact,
  −1 gate per run, depth 71.01 % → 72.33 % at unchanged distance.

## Verdict

**Final (`6262b1b`, `--mode full`): +81.51 %** — distance +83.80 % (D 44 825.4), depth +72.33 %
(P 6 550). Families: hea +78.57 / +48.39, ihva +86.19 / +75.21, qaoa +80.52 / +69.01. 36/36 metric
circuits pass full mode with zero physics/architecture/equivalence failures. Sanity: 13/13 pass in full mode
(QCEC-exact on sample 1–11; sample_circuit_11's check alone takes ~476 s in the harness).

50q circuits (distance, vs naive): QAOA p1 2 021 (−75.7 %), QAOA p3 5 861 (−76.5 %), iHVA stagger
p3 5 579 (−77.6 %), iHVA tree p3 5 858 (−76.5 %), tree_warm p3 751 (−97.0 %), HEA p3 726 (−72.7 %).
The 50q circuits are 33 513 of the remaining 44 825 (75 %), so they are where any next gain is.

What carries over to a merged compiler:
1. **Hub parking via ZZ orientation**: rotation side = non-hub endpoint, so the hub's wire holds only
   CZs for a whole layer (iHVA's `rx(±π/2)` cancel on the hub wire). The hub moves twice per
   circuit instead of ~1 000 times. Exact, no commutation reasoning needed.
2. **The generic "park while the merged 1q run is identity" rule** (also keeps the non-rotation
   side of every block in the zone between its two CZs).
3. **θ≡0 block elision and deferred trailing `rz` across `cz`** (exact; the first is why the warm
   circuits are so cheap, the second gives +1.3 pt depth).
4. Reserving an end pair of row 10 for the hub so it does not split row-10 rectangles (n > 20).

What did not work: batching each round's lower/lift into one storage row (step 3, all variants
worse), because the ghost rule splits rectangles around idle atoms and 50 atoms leave no shallow
free row. Hub-specific partner loading / a `(0,9)` staging site: local wins on 50q p1, net losses.

Next step: the remaining cost is ~1.3 atoms per move at a mean length of ~4 on 50q. The lever is a
**position-aware choice of round composition** (pick, among ready tasks, those whose atoms are
column-contiguous in one storage row), combined with the commutation-aware rounds of the `gates`
branch, which give much larger ready sets to choose from. Gate-order ASAP leaves only ~3 ready
tasks per round, which is too little freedom for batching.

Earlier notes (step 2): works and is well behaved: 36/36 metric circuits and 13/13 sanity circuits pass, hub qubit is moved
twice per circuit (its first lift and its final lowering) instead of ~1 000 times, `D` is 6× smaller
than naive. Remaining cost is dominated by the storage depth of the ~50 atoms: each task needs one
lower + one lift, and 50 atoms occupy rows 9…5, so the mean move length is ≈1.4 instead of ≈1.
A round is still split into several `move`s when a parked atom sits between the moving columns of a
row (the validator's ghost rule forbids spanning it), which is where the next gain is.

Compile time: worst metric circuit 2.0 s (50q iHVA p3), sanity 13 (70q) 8.6 s. Note the *harness*
QCEC check of `sample_circuit_11` (18 qubits, 459 CZ) takes 340 s on an idle machine and ~15 min
under load; the compiled circuit has fewer gates (589) and a lower depth (79) than the naive
compilation (704 gates, depth 82), so this is the checker, not the compiler.

## STEP 3 (batching): measured, all worse — not kept

Goal was one lower + one lift move per round (≥3 atoms/move, ≤2.5 mean distance on 50q p1, which
starts at 1.28 atoms/move and 4.48). Every variant was implemented and measured on the full 36
metric circuits (fast mode, D sums):

| variant | score | dist % | D | p1 `qaoa` D | p1 atoms/move | p1 mean dist |
|---|---|---|---|---|---|---|
| step-2 baseline (kept) | **81.06** | 83.24 | 46 368 | 2 407 | 1.28 | 4.48 |
| 3a batched descent (one storage row per group) | 78.51 | 80.06 | 55 188 | 2 397 | 1.14 | 3.96 |
| 3a + 3b (loading leaves row 9 free) | 78.45 | 79.97 | 55 415 | 2 397 | 1.13 | 3.92 |
| 3a + 3b + 3c (pairs handed out left-to-right by storage column) | 77.56 | 78.86 | 58 492 | 2 769 | 1.13 | 4.55 |
| 3a + 3b + 3c with `MAX_PAIRS` 5 / 10 | 77.64 / 76.89 | – | 58 206 / 60 784 | – | – | – |
| 3d (hub pair forced to row-10 columns 0,1 + `(0,9)` staging for its partner) | 80.41 | 82.42 | 48 640 | 2 166 | 1.19 | 3.75 |
| 3d forcing only / staging only | 80.74 / 80.41 | – | 47 484 / 48 640 | – | – | – |

**Why batching does not pay.** A `Move` picks a *contiguous* row×column rectangle, so the descent
(or lift) can only span a contiguous run of occupied source columns: any atom sitting between two
moving atoms of the same row is a ghost and forces a cut. A round has ~7 descending/ascending atoms
drawn from a 10-site row, interleaved with idle atoms (and with the parked hub), so the group always
splits into 3-6 runs *whatever* row it is sent to. Measured directly: a 7-atom descent assigned to a
single storage row still came out as 3 moves (columns (0,), (2,3), (4,5,6,7)).
Second, 50 atoms fill storage rows 5-9 completely, so a 7-atom group never fits a row near the zone
(nor after 3b: row 9 holds 10, the atoms stay ~8 rounds); the fallback then sends the group deep, and
`D` is 2 × mean depth × moves, so the depth loss eats the move-count win. Making a round's atoms
contiguous as well as co-rowed would need the round composition to depend on atom columns, i.e. a
position-aware scheduler (ready-set ordering may be chosen freely without touching wire order) —
that is the next thing to try, together with keeping the hub at a row-10 end column so it does not
split the run.
Both 3d pieces are real local wins on `50q qaoa p1` (2 407 → 2 166) but lose ~2 % overall, so they
are reverted too; the step-2 compiler is unchanged.

### Round 2 — 15-circuit comparison, old (`6262b1b`) vs new, `D + 2.92·depth`

| circuit | old D | old depth | old cost | new D | new depth | new cost | Δ |
|---|---|---|---|---|---|---|---|
| 12q_ihva_stagger_p1 | 181.2 | 53 | 335.9 | 196.8 | 45 | 328.2 | −7.7 |
| 12q_ihva_stagger_p3 | 596.0 | 140 | 1004.8 | 596.0 | 140 | 1004.8 | 0 |
| 12q_ihva_tree_p1 | 181.4 | 72 | 391.6 | 196.0 | 64 | 382.9 | −8.8 |
| 12q_ihva_tree_p3 | 612.3 | 184 | 1149.6 | 612.3 | 184 | 1149.6 | 0 |
| 20q_ihva_stagger_p1 | 430.6 | 88 | 687.6 | 480.9 | 56 | 644.4 | −43.1 |
| 20q_ihva_stagger_p3 | 1263.3 | 236 | 1952.4 | 1263.3 | 236 | 1952.4 | 0 |
| 20q_ihva_tree_p1 | 441.4 | 100 | 733.4 | 449.9 | 73 | 663.0 | −70.4 |
| 20q_ihva_tree_p3 | 1302.2 | 270 | 2090.6 | 1477.4 | 197 | 2052.6 | −37.9 |
| 12q_qaoa_p1 | 234.9 | 71 | 442.2 | 228.6 | 62 | 409.7 | −32.5 |
| 12q_qaoa_p3 | 643.5 | 171 | 1142.8 | 668.9 | 133 | 1057.3 | −85.5 |
| 20q_qaoa_p1 | 523.7 | 105 | 830.3 | 479.2 | 79 | 709.9 | −120.5 |
| 20q_qaoa_p3 | 1261.8 | 269 | 2047.2 | 1467.5 | 189 | 2019.4 | −27.9 |
| 50q_qaoa_p3 | 5860.9 | 677 | 7837.7 | 5860.9 | 677 | 7837.7 | 0 |
| 50q_ihva_stagger_p3 | 5578.8 | 576 | 7260.7 | 5578.8 | 576 | 7260.7 | 0 |
| 50q_hea_linear_p3 | 725.8 | 22 | 790.1 | 725.8 | 22 | 790.1 | 0 |
| **total** | | | **28 696.9** | | | **28 262.7** | **−434.3 (−1.5 %)** |
