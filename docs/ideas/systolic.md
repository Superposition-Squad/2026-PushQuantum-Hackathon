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
| + A2 batched-cost placement strategies | 552bcb4 | +88.01 | 89.84 | 80.68 | 20q QAOA p3 693→678, depth 4614→4574 |
| + B1 deterministic multi-start (schedules × layouts) | — | +88.53 | 90.42 | 80.97 | iHVA 50q p1 1404→1010, stagger p3 4061→3326 |
| + B2 zig-zag path layouts, B3 run-merge, lookahead schedules | 60279cd | **+88.57** | 90.50 | 80.82 | iHVA tree p3 4931→4495, HEA +80.98 |

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

## Phase B: layout and scheduling candidates

* **B1 deterministic multi-start.** `compile_circuit` compiles a deterministic candidate list
  (schedule policy × layout) and keeps the lowest `distance + 2.922 * depth` (depth with moves and
  barriers stripped). Candidate count is fixed by qubit count (12 for n ≤ 20, 8 for n ≤ 34, 4
  otherwise), so 50q p3 stays at ~25 s; there is a hard 120 s stop that never triggers. The same
  layout is often produced by two seeds, so candidates are de-duplicated by their `home` map.
* **B2 zig-zag layouts.** Longest simple path of the non-hub block graph (Warnsdorff greedy DFS from
  every start, ties by index — covers all 49 partners on 50q), laid over lane pairs (0,1),(2,3) and
  the tail in lane 4, plus (1,2),(3,4)/(0,1),(3,4) variants and both path orientations. Then
  edge (p2k,p2k+1) is a same-row pair (visitor 1.7) and (p2k+1,p2k+2) an offset-+1 pair (1.97).
* **Schedule policies.** `lookahead(w)`: take the ready category whose group plus the blocks it
  frees (weighted by w) is largest, incremental counters keep it O(edges) per step; plus non-first,
  hub-first, largest, and ASAP antichain layering. Priority order look2, look0.5, non, hub, largest,
  asap.
* **B3 run-merge placement** (scored against the other strategies by the batched-cost model): whole
  contiguous lane runs whose atoms are all desired pairs with one destination lane at a fixed row
  offset (>= 2 pairs) become one rectangle move; remaining blocks are placed by the base strategy
  with the run sites claimed.
* **Role-aware target choice**: for `zz` blocks the middle `rx` goes to the endpoint that needs no
  extra flush.

Per-circuit distance/depth, phase A (`9e27b23`) → phase B (`60279cd`):

| circuit | phase A D/P | phase B D/P |
|---|---|---|
| 19 50q iHVA stagger p1 | 1404.3 / 133 | **1010.4 / 143** |
| 20 50q iHVA stagger p3 | 4061.2 / 382 | **3325.6 / 416** |
| 21 50q iHVA tree p1 | 1683.1 / 158 | 1533.3 / 175 |
| 22 50q iHVA tree p3 | 4931.1 / 451 | 4494.9 / 486 |
| 29 20q QAOA p1 | 256.3 / 73 | 226.1 / 70 |
| 30 20q QAOA p3 | 769.0 / 203 | 678.4 / 200 |
| 31 20q QAOA warm p1 | 256.3 / 73 | 226.1 / 70 |
| 32 20q QAOA warm p3 | 769.0 / 205 | 678.4 / 200 |
| 33 50q QAOA p1 | 870.7 / 138 | 818.3 / 138 |
| 34 50q QAOA p3 | 2612.2 / 390 | 2455.0 / 390 |
| 35 50q QAOA warm p1 | 870.7 / 138 | 818.3 / 138 |
| 36 50q QAOA warm p3 | 2612.2 / 390 | 2455.0 / 390 |

Winning candidates: 50q iHVA/QAOA → `seed12345/look2` or `…/non`; 20q/12q → `seed*/look2` or other
seeds; HEA 20q/50q → `zig0`. On 50q QAOA p3 the placement is 444 single-atom moves (was 858 in
zonedwell r1) plus 78 runs of 2–3 and 54 of 4–9; the run-merge fires on most matchings of 20q/12q
(54 opportunities) but rarely wins on 50q, where the lane columns are too dense for long runs.

## Verdict

**+88.57 %** (distance +90.50 %, depth +80.82 %) at `60279cd` (phase A: +88.01 % at `552bcb4`, up +0.49 pts from the re-seeded base
(+87.52): 36/36 metric `--mode full`, 13/13 sanity `--mode full`, and the exact Clifford-tableau
check equal on 50q circuits 19–22 and 33–36 (two angle draws each). Every sanity circuit ≤ 18
qubits is `equivalent_up_to_global_phase` under QCEC. Remaining distance on 50q QAOA p3
(2455): placement 1395, conveyor 404, dips 353, zone moves 20 — placement is still the lever, but
it is now ~1.6 per atom-move, i.e. close to the ~1.04 geometric floor of a visitor hop.

## ADR-0010 (zone-span) and the hub walk

ADR-0010 forbids a `move` that picks up (or drops on) more than one row spanning both zones, so the
lane conveyor and the old whole-zone dips became illegal. `c47a22e` replaced them:

* `dip(d)` moves only zone rows 10..9+d down to storage rows 10−d..9 and back (same cost, the rows
  above stay); `to_storage`/`to_zone` move only the `L` lane rows; the batcher refuses zone-spanning
  groups and `_State.move` asserts the rule.
* `conveyor()` now runs a **hub walk**: the hub steps onto each served partner's free right slot
  (a single-row move), one dip carries every partner's `rx`, then the hub walks back for the second
  `cz`. `conveyor_lane` is deleted. The blocks of a hub group commute, so the walk order is free.

`<this commit>` optimises the walk and the layout search that feeds it:

* forward order = nearest-neighbour + 2-opt over the partner slots from the hub's current site
  (`_tsp_path`, ≤ 49 points); the return order is the cheaper of reversing the forward path and a
  fresh 2-opt path from its end (measured on 50q the old per-lane snake was already optimal for the
  full 5×10 grid, which is why 50q QAOA is unchanged; iHVA hub groups are sparse and gain).
* the walk dip depth is `1 + deepest served row`, as before (all served partners are in rows
  10..9+d by construction).
* `_LayoutCost`'s conveyor term is now the walk cost, `entry + 2·path + 2·(1 + deepest row)`
  (the path is paid twice), so the layout search groups a hub's partners in adjacent lanes/rows
  and keeps them shallow.
* the layout local search is incremental: the cost decomposes per group, so a swap re-evaluates
  only the groups containing the two swapped qubits (verified identical to full re-evaluation).
  This is what pays for the more expensive walk term: 50q iHVA p3 59 s → ~16 s CPU, so the
  deterministic multi-start budget stays at 12/8/4 candidates.

Not done (not obviously good): overlapping the next non-hub matching with the hub's return walk.
Moves are serialised, so this only helps if the relocations can be fused into one rectangle; the
hub hops are single atoms on the partner columns and the matching placements use other lanes, and
merging them with the shared dip would break the "one dip, every rx" window.

Results, `c47a22e` → this commit (fast and full mode give the same D/P):

| circuit | c47 D/P | now D/P |
|---|---|---|
| 19 50q iHVA stagger p1 | 1070.8 / 152 | **1024.5 / 146** |
| 20 50q iHVA stagger p3 | 3480.6 / 390 | **3329.8 / 380** |
| 21 50q iHVA tree p1 | 1573.4 / 139 | **1444.9 / 140** |
| 22 50q iHVA tree p3 | 4394.7 / 422 | 4410.2 / **413** |
| 29 20q QAOA p1 | 230.6 / 66 | 232.5 / **61** |
| 30 20q QAOA p3 | 693.1 / 182 | **684.4 / 179** |
| 31 20q QAOA warm p1 | 230.6 / 66 | 232.5 / **61** |
| 32 20q QAOA warm p3 | 693.1 / 182 | **684.4 / 179** |
| 33 50q QAOA p1 | 805.4 / 137 | **783.3 / 131** |
| 34 50q QAOA p3 | 2416.2 / 389 | **2362.8 / 376** |
| 35 50q QAOA warm p1 | 805.4 / 137 | **783.3 / 131** |
| 36 50q QAOA warm p3 | 2416.2 / 389 | **2362.8 / 378** |

**+88.63 % → +88.95 %** (distance 26 398.5 → 25 733.3, depth 4426 → 4272; full `--mode full`).
36/36 metric full, 13/13 sanity full, exact Clifford tableaux equal on 19–22 and 33–36 (two angle
draws each), bench zone-span check clean on every circuit, deterministic on 50q p3, max compile
time 16.5 s CPU (50q QAOA warm p3, idle).

## Round 3: dip-aware placement (1d7aeeb +)

The matching rounds pay two dips at the deepest atom that needs them: the flush dip (pre-gates
before `cz #1`) and the mid dip (`rx`).  `_assign_cost` only counted the mid dip, and `flush_dip`
took the depth over qubits with *identity* pending gates too.  Now:

* `_assign_cost` simulates the pending pre-gate of every involved qubit and adds the flush dip
  (and takes the mid dip as `max(flush, target)`), so `_choose_assign` prefers placements that pull
  the flushing/target atoms into low rows; bare-cx groups (HEA) no longer pay a phantom mid dip.
* `flush_dip` ignores identity pending gates when sizing its dip.
* `park_hub` parks on the nearest free lone site of the lane rows instead of the fixed park.

`1d7aeeb` → this: **+88.95 % → +89.01 %** (distance 25 733.3 → 25 475.1, depth 4272 → 4289).
The dip depth is now explicit in the placement cost; every circuit except HEA moves (QAOA 50q p3
2362.8 → 2333.0, 20q p1 232.5 → 216.2 at +6 depth).

Validation of this round: 36/36 metric `--mode full` (+89.01 %, distance 25 475.1, depth 4289),
12q 12/12 full (+89.16 % partial), 13/13 sanity full, exact Clifford tableaux equal on 19–22 and
33–36 (two angle draws each), bench zone-span check clean on all 49 circuits plus
`/tmp/sy/zonespan.py` 0 on the eight 50q circuits, compile-twice bit-identical (50q QAOA p3,
50q iHVA tree p3, 20q QAOA warm p3), max compile 14.5 s CPU (20q warm p3; 50q p3 13.8 s).

Tried and rejected (measured, not kept): a return-walk terminal bias towards the park (+25 on
50q tree p3), two extra high-row-penalty placement strategies (never won, only cost time),
depth-sorted matching order (50q partial 88.52 → 88.34), layout weights 8 (matching dip) and 4
(walk dip): 88.31 / 88.53 vs 88.52.

## Validator v1.0.0 `idle` rule: storage-home redesign

v1.0.0 rejects lone atoms in the entangling zone during any two-qubit gate, so the zone-dwelling
machine (and the hub walk's parked hub) is illegal. The lane path is replaced by a storage-home
machine:

* every atom's home is in storage, rows 10−R..9 with R = ceil(n/10), all ten columns; the layout
  search is reused on those slots with a lift-distance cost proxy;
* between rounds every atom is at home, so pending 1q gates are flushed for free (no dips, no zone
  switches);
* a matching round lifts the matched atoms, pulses, brings them home, applies the mid `rx` in
  storage, lifts them again with the same placement, pulses, home again; bare cx/cz blocks do a
  single lift/pulse/return;
* a hub group lifts the hub once, then per partner moves the hub to the partner's pair slot at row
  10 and lifts the partner straight up, pulses, and brings it home; after all cz #1 the partner
  `rx` gates run in storage, then the same in reverse for cz #2, then the hub goes home;
* `_State.pulse` asserts the zone holds exactly the pulsed atoms (the idle rule), and
  `_State.move` keeps the ADR-0010 row rule.

Interim (`v100`): all 49 circuits valid under the v1.0.0 validator (idle/zone/pair/loading), bench
fast 36/36, score +76.27 (distance 67 879.7, depth 4858).

### Round-4 optimisations (storage-home)

* Control stays: for a mid (`rx`) block only the target visits storage between the two pulses; the
  control remains on its pair (no `cz` runs while it sits there, so `idle` does not apply).
* `_mid_spots`: the target's middle trips go to the shallowest storage site freed by the lift, not
  to its own home.
* `_pairing_layout`: a matching of real blocks laid out in adjacent slot pairs (one block per
  interaction pair), so the first matching round of each layer is column-preserving.
* `_assign_vertical(..., rowlift=True)`: home row y maps to zone row 19−y, so every home row lifts
  as ONE single-row move (cost 10−y for up to ten atoms) and returns the same way; `_diag_order`
  now orders blocks by home row so the greedy packing puts whole rows into one round.
* Lazy layout stream (only the layouts a budget reaches are computed) keeps 50q p3 at ~12–20 s.

`v107`: +82.54 (distance 46 987.0, depth 4593), all 36 bench-valid; the row-lift cuts the first
matching round of each layer from ~350 to ~126.

### Final (validator v1.0.0 vendored, `main` merged)

With the vendored v1.0.0 the bench itself checks `idle`; the merged tree passes 36/36 metric
`--mode full` at **+82.54 %** (distance 46 987.0 = +83.02 %, depth 4593 = +80.60 %) and 13/13
sanity `--mode full`. Also: `/tmp/sy/idlecheck.py` (the new validator directly) reports 0 idle /
zone / pair / loading violations on all 49 circuits; exact Clifford tableaux equal on 19–22 and
33–36 (two angle draws each); compile-twice bit-identical on 50q QAOA p3 and 50q iHVA tree p3;
max compile ~20–24 s CPU (50q QAOA p3, four candidates), 20q p3 6.6 s.

Per-circuit D/P (`metric_circuit_nn`, v1.0.0): 01 26.0/15, 02 83.7/35, 03 55.3/15, 04 171.5/35,
05 113.2/15, 06 339.5/35, 07 206.1/58, 08 598.0/173, 09 257.6/60, 10 750.4/172, 11 67.9/29,
12 67.9/29, 13 434.6/76, 14 1329.7/213, 15 389.3/76, 16 1175.6/219, 17/18 152.6/35,
19 1963.9/138, 20 6169.7/408, 21 2373.8/146, 22 7202.1/460, 23/24 665.7/73, 25 175.8/51,
26 527.5/147, 27 175.8/53, 28 527.5/149, 29/31 412.3/67, 30/32 1236.8/189–191,
33/35 2108.7/135, 34/36 6326.2/393.
