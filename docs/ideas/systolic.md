# Idea: systolic

- Branch: idea/systolic · Worktree: /home/hackathon/challenge-wt/systolic
- Base: main f462cf5 (+ bounds d0ed179); compiler forked from `compilers/zonedwell/compiler.py`
  (the 50q winner, valid on all 49 circuits)
- Status: in progress

## Hypothesis

A `Move` applies one monotone row map and one monotone column map to every atom in a rectangle
and costs only the longest displacement. zonedwell already uses this for the hub conveyor
(1.0 per hub `cz`, at the bound) and for bulk dips, but realises every **non-hub** matching with
one `Move` per visitor. Measured on 50q QAOA p3 (2911 total):

| move kind | count | distance |
|---|---|---|
| 1-atom moves (non-hub visitors, place + unplace) | 858 | 1859 |
| ≥10-atom moves (bulk dips, zone switches, conveyor slides) | 318 | 998 |
| 4–9-atom moves | 54 | 54 |

So 64 % of the distance is single-atom re-pairing and ~24 % is dips. Two whole-array levers:

1. **Rectangle batching of re-pairing.** Plan the relocations of a matching (and the return
   trip) as a set of atom moves, then fuse them into few legal rectangle moves with monotone maps.
   Choose orientation (who visits whom) and target pairs to *maximise shared displacement
   vectors*, so fusing actually happens. Expected: ~858 moves → ~250, distance 1859 → ~700.
2. **Windows (super-rounds).** All blocks in one zonedwell group commute. Do the `cz #1` pulses of
   several matchings back to back (re-pairing in the zone, no 1q gate needed), then **one** dip in
   which every target receives its `rx(θ)`, then the `cz #2` pulses in reverse order. Legal iff in
   a window every target has exactly one block and no target is also a control of another block
   of that window (otherwise `X_t Z_c Z_x` 3-body terms). Expected: 7 dips/layer → ~3.
3. **Shallow dips by placement.** Dip depth = 1 + deepest row offset of an atom that needs a 1q
   gate; place target pairs in the lowest free zone rows.

Estimate: 50q QAOA p3 2911 → ~1500, 50q iHVA p3 4452/5428 → ~3000/3500. Together with 20q gains
≈ +1.5–2.5 points of portfolio score.

Further, bound-approaching lever (phase C, explore after A/B work): **ride-along on the conveyor.**
A conveyor slide of lane j can be issued as the rectangle `cols 0..2j` instead of `col 2j`: pair
columns left of the lane then shift down rigidly with it (pairs preserved, rows cross into storage
one per step). A full pair at zone offset u is pulsed u+1 times on the way down and u+1 times on
the way up, with its storage visit in between, so **pairs at even offsets get a complete ZZ block
for free** during the hub pass, and odd offsets get identity. That puts non-hub blocks on the
mandatory ~98 hub moves per layer, which is what the bound (294 on 50q p3) needs.

## Design (phase A: batching + windows; phase B: layout search on real cost)

Everything outside the non-hub matching path stays as in zonedwell (parsing, 2×2 1q buffering
flushed only in storage, ready-set iHVA scheduler, conveyor, generic fallback).

### A1. Move batcher (`_plan_moves(reloc, occ) -> list[dict q -> (src, dst)]`)
- Input: relocations `{q: (src, dst)}` and the current occupancy.
- Legality of a candidate group (validator-exact, cf. rectbatch `move_params`): consistent and
  strictly monotone row map and col map; the bounding rectangle of the sources contains no atom
  outside the group **unless** the maps send that atom to its own site (then it is added as a
  0-displacement member); no destination occupied by an atom outside the group; targets in
  0..9 × 0..19 (moves may cross the zone boundary: both zones span all columns).
- Agglomerative merge: start from singletons, repeatedly fuse the pair with the largest saving
  `cost(A)+cost(B)−cost(A∪B)` (≥ 0) that stays legal. Then order groups so every group's targets
  are free when it runs (a group may land on sites its own members vacate); re-check legality on the
  live state just before emitting, and split into single-atom moves (via a temporary free site
  for swap cycles) if it fails.

### A2. Batching-aware assignment
For a matching M, every block has candidate placements: visitor u → right/left free slot of
dweller v's pair (dweller lone), or the reverse, or both atoms to a free pair. Compute each
candidate's grid displacement vector; greedily pick the most common vector among unassigned blocks,
assign every block that can use it (conflict-free target sites), repeat. Run 2–3 strategies (this
one, zonedwell's min-distance, vector-greedy with target row penalty) and keep the cheapest by
batcher cost (simulated, no emission). The return trip reverses the emitted moves in reverse order
(the state between is restored by dips, so the reverse moves are legal).

### A3. Windows
Inside a group: build the matchings (as now), then pack them into windows greedily (a matching
joins the current window if its targets are new and disjoint from the window's controls and
targets-vs-controls stays disjoint). Window: place M1, pulse, re-pair to M2 (one relocation set:
M1 visitors home + M2 visitors out, batched), pulse, … , one dip with all targets' `rx(θ)`, then
pulses for Mk … M1 with the reverse re-pairing, then return home. Equivalence: all blocks of a
group commute (antichain), the per-target sequence is `H cz_e rx(θ_e) cz_e H` per window, and
controls see only `cz` in the zone (commuting). The pending-1q buffer and `_needs_flush` asserts
stay in place.

### B. Layout search on the real cost
zonedwell's local search uses a proxy (Σ grid distance). Add a deterministic second stage: a
fixed number of swap/relocate trials evaluated by running the real non-emitting planner on the
whole circuit, iteration budget scaled so 50q p3 compiles in ≤ 30 s.

### Validity
Batcher legality is re-checked on the live state before every emitted move, `_State.move` asserts
the validator rules, `_DEBUG` pulse check (occupied pairs == intended pairs) runs in tests. Any
assertion → fallback to the generic path (valid by construction). Strict mode + Clifford-tableau
check (angles → π/2) on 50q before claiming.

## Results

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|
| fork = zonedwell r1 | 43f5e62 | +80.79 | 81.68 | 77.25 | baseline |
| r1 + shallow dip before conveyor (scratch) | – | +81.45 | 82.46 | 77.43 | full 12q ✓, sanity 13/13 ✓, Clifford ✓ |
| phase A WIP on r1 (batcher only) | ae910ec | +81.27 | 82.27 | 77.25 | 50q QAOA p3 2911→2721; windows broke 50q iHVA (fallback) |
| **re-seed: zonedwell r2 (25f752c) + shallow dip** | 34f724c | **+87.52** | 89.27 | 80.51 | full 36/36 ✓, sanity 13/13 ✓, Clifford tableaux equal on 6 × 50q |
| + A1 rectangle move batcher | b91d742 | +87.90 | 89.74 | 80.51 | 50q QAOA p3 2612→2455, 20q QAOA p3 769→693 |
| + A2 batched-cost placement strategies | 552bcb4 | **+88.01** | 89.84 | 80.68 | 20q QAOA p3 693→678, depth 4614→4574 |

Per-circuit distance/depth, base `34f724c` → final (same depth where equal):

| circuit | base D/P | systolic D/P |
|---|---|---|
| 19 50q iHVA stagger p1 | 1404.3 / 133 | 1366.1 / 133 |
| 20 50q iHVA stagger p3 | 4061.2 / 382 | 3897.1 / 382 |
| 21 50q iHVA tree p1 | 1683.1 / 158 | 1653.6 / 158 |
| 22 50q iHVA tree p3 | 4931.1 / 451 | 4820.2 / 451 |
| 29 20q QAOA p1 | 256.3 / 73 | 226.1 / 70 |
| 30 20q QAOA p3 | 769.0 / 203 | 678.4 / 200 |
| 31 20q QAOA warm p1 | 256.3 / 73 | 226.1 / 70 |
| 32 20q QAOA warm p3 | 769.0 / 205 | 678.4 / 200 |
| 33 50q QAOA p1 | 870.7 / 138 | 818.3 / 138 |
| 34 50q QAOA p3 | 2612.2 / 390 | 2455.0 / 390 |
| 35 50q QAOA warm p1 | 870.7 / 138 | 818.3 / 138 |
| 36 50q QAOA warm p3 | 2612.2 / 390 | 2455.0 / 390 |

Max compile time 10.3 s per circuit (50q p3).

## What was implemented (phase A)

* **A1 rectangle move batcher** (`_group_params`, `_plan_moves`, `_emit_plan`, `_emit_singles`):
  validator-exact group legality (consistent + strictly increasing row/col maps, ghost-free
  rectangle, fixed-point atoms may join as zero-displacement members, destinations free or vacated
  by the group, squeeze emission, zone crossing allowed); agglomerative merging on the largest
  saving; a topological order on the "destination is another group's source" relation with a
  temporary free site for genuine swap cycles; legality re-checked on the live state before every
  emitted move, with a per-atom fallback. `_place` emits it; `_unplace` re-plans the return trip
  batched (the two trips are now ~16 moves instead of ~24 single-atom moves each).
* **A2 placement strategies scored by the real batcher** (`_assign` + `_assign_cost` +
  `_choose_assign`): the base min-distance assignment, a variant with a strong shallow-row
  penalty, and a "dweller" variant that restricts a block to the pair one of its atoms already
  sits on (so only one atom travels). Each candidate is scored by the *non-emitting* batcher plan
  (outward trip + return trip + 2·dip depth) and the cheapest wins; deterministic tie-breaks.

## What did not pay off (numbers)

* **Windows / super-rounds (A3)** — packing several matchings behind one storage dip. A target may
  not be touched by any other block of the same window (its `rx` would land between another pair's
  two `cz`), so a window's blocks need *fresh targets*: each partner can be a target only once per
  window. With 49 partners and 98 non-hub blocks per layer that forces shrinking matchings
  (24, 6, 2, …) and 3 windows per layer, i.e. ~1.5× more placement stages, each of which must visit
  the previous configuration twice (forward and reverse pulses). Measured on 50q QAOA p3: dips
  18→9 (−170) but placement 1395→2288 (+890) → 3408 vs 2455. Not worth it: the dips here cost
  ~17 each, the placement ~75 per matching.
* **Direct transitions between matchings (task 4)** — chaining configs without returning visitors
  home, with the dweller allowed to stay at its *current* site. Fewer atoms move but each trip is
  much longer (3.0–4.1 per atom vs 1.6): 50q QAOA p3 2684 vs 2455. A visitor that stays becomes a
  dweller on an odd site whose pair partner is far from the next visitor.
* Fewer/larger matchings: the greedy packing already gives 6 rounds for 98 blocks (49 qubits,
  ≤24 blocks per round ⇒ ≥5); four alternative orderings and 30 random shuffles all gave 6.

## Verdict

**+88.01 %** (distance +89.84 %, depth +80.68 %) at `552bcb4`, up +0.49 pts from the re-seeded base
(+87.52): 36/36 metric `--mode full`, 13/13 sanity `--mode full`, and the exact Clifford-tableau
check equal on 50q circuits 19–22 and 33–36 (two angle draws each). Every sanity circuit ≤ 18
qubits is `equivalent_up_to_global_phase` under QCEC. Remaining distance on 50q QAOA p3
(2455): placement 1395, conveyor 404, dips 353, zone moves 20 — placement is still the lever, but
it is now ~1.6 per atom-move, i.e. close to the ~1.04 geometric floor of a visitor hop.
