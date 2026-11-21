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
| fork = zonedwell | | +80.79 | 81.68 | 77.25 | baseline |

## Verdict
(pending)
