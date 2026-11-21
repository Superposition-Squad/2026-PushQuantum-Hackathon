# Idea: hubstream

- Branch: idea/hubstream · Worktree: /home/hackathon/challenge-wt/hubstream
- Base: main 0ea558b, starting from `compilers/naive/compiler.py`
- Status: in progress

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

Works and is well behaved: 36/36 metric circuits and 13/13 sanity circuits pass, hub qubit is moved
twice per circuit (its first lift and its final lowering) instead of ~1 000 times, `D` is 6× smaller
than naive. Remaining cost is dominated by the storage depth of the ~50 atoms: each task needs one
lower + one lift, and 50 atoms occupy rows 9…5, so the mean move length is ≈3 instead of ≈1.
A round is still split into several `move`s when a parked atom sits between the moving columns of a
row (the validator's ghost rule forbids spanning it), which is where the next gain is.
