# Idea: zonedwell

- Branch: idea/zonedwell · Worktree: /home/hackathon/challenge-wt/zonedwell
- Base: main 0ea558b, compiler starts from `compilers/naive/compiler.py`
- Status: implemented (stage 0-2; see Results)

## Hypothesis

The naive compiler pays a storage round trip per CZ. The usual fix ("park the hub, stream its
partners": literature #5) still pays **two moves of ~1.0 per CZ**. zonedwell claims we can get to
**one move of exactly 1.0 per CZ** by never making a dedicated storage trip at all: the move that
re-pairs the hub *is* the partner's storage trip.

The mechanism is a **vertical conveyor lane**. Columns are the natural lane direction because a
grid column spans both zones (storage rows 0–9, entangling rows 10–19) and because an **even**
column has the same physical x in both zones, so a one-row shift across the zone boundary costs
exactly 1.0 (an odd column costs √(1+0.3²)=1.044 … actually √(1²+0.7²)=1.22 at the boundary, so
lanes must be even columns).

Secondary claim (literature #4/#8): atoms whose next operation is another CZ never go to storage;
they dwell in the zone as **lone atoms on a pair** (legal under ADR-0007, confirmed in
`bench/bench.py:115` — `occupied_pairs()` only counts pairs with *both* sites filled), and get
re-paired by a short in-zone move.

Expected effect, 50q QAOA/iHVA cost layer (hub degree 49 → 98 serial hub CZs):
naive 8 300 / layer, hub-streaming ≈ 220 / layer, conveyor ≈ 106 / layer.

## Design

### Geometry facts used

Site = `(x=col, y=row)`. Storage y 0–9, entangling y 10–19, x 0–9 in both.
Physical: storage `(x, y)`; entangling `(2·(x//2) + 0.3·(x%2), y)`.
So for **even** x the physical x is identical in both zones ⇒ a one-row hop across the zone
boundary costs 1.0. Interaction pairs are `(2k, 2k+1)` in one row; pair pitch is 2.0, intra-pair
0.3. Therefore the cheapest re-pairing of an atom is a **vertical** 1-row hop (1.0), not a
horizontal pair hop (2.0).

Move semantics that the design exploits (from `vendor/.../_validation.py`):
`site_mapping` + `_remove_empty` mean that if we hand fewer target lines than picked-up lines, the
**empty** picked-up lines are dropped and the *occupied* lines map in order to the targets. So one
move can send the occupied rows/cols to any ascending target lines — but it is always a product
(row-map × col-map), so it can never reorder atoms.

### Gate synthesis (exact, by construction)

* `cx(u,v) · rz(θ)_v · cx(u,v)` → `H_v · CZ(u,v) · rx(θ)_v · CZ(u,v) · H_v` (because
  `CZ·X_v·CZ = Z_u X_v`, so `CZ rx(θ)_v CZ = exp(-iθ/2 Z_u X_v)`, and `H_v` turns `X_v` into `Z_v`).
  **Only the target v sees a 1q gate between the two CZs — the control u sees nothing.** That is
  what lets the hub dwell.
  ZZ is symmetric, so we may choose either endpoint as the target: we always pick the **hub as the
  control**.
* bare `cx(u,v)` → `H_v · CZ(u,v) · H_v` (one CZ, no 1q gate in between) — HEA.
* `h` / `ry` / runs of 1q gates → merged per qubit into `rz·rx·rz` (ZXZ Euler), in storage.

### Layout

Hub `h` = the qubit of degree n−1 (iHVA, QAOA; `q[n-1]`). Partners = the rest.
Lanes: lane `j` is **column 2j** (even), j = 0…4, holding up to 10 partners. Partner `i` → lane
`i // L`, lane index `i % L`. A lane occupies either storage rows `10-m … 9` or entangling rows
`10 … 9+m` (m = lane size). The hub parks on the **right slot** `(2j+1, 10)` of the lane it serves.
Initial loading puts every partner in storage at its lane column and the hub at `(1, 9)`.

### The conveyor (hub blocks, one layer)

Precondition: lane j's m atoms at entangling rows 10…9+m, column 2j; hub at `(2j+1, 10)`.

```
D pass (m slides, m CZs):      repeat m times:  cz(hub, atom at row 10) ; slide lane down 1 row
   slide = Move(rows=(r0..r1), cols=(2j,), new_rows=(r0-1..r1-1), new_cols=(2j,))   cost 1.0
   after the m-th slide the whole lane is in storage (rows 10-m … 9)
storage phase:                 rx(θ_v) on every lane atom (its ZZ-block angle)
U pass (m slides, m CZs):      repeat m times:  slide lane up 1 row ; cz(hub, atom at row 10)
   after the m-th slide the lane is back at entangling rows 10 … 9+m
```

Each atom therefore gets CZ#1 on its way out of the zone, its `rx(θ)` in storage, and CZ#2 on its
way back in — a complete ZZ block for **2 slides = 2.0 of distance**. 49 partners ⇒ 98.0 per layer
plus 4 hub hops of 2.0 between lanes.

Legality of a slide: picked-up rows contiguous/ascending ✓, one column line ✓, targets ascending ✓,
no ghosts (lane column holds only lane atoms, the hub is in column 2j+1) ✓, the target site below
the lane is free ✓, all targets inside a zone ✓ (rows 0–19).
ADR-0007 in every segment: the only pair with two atoms is `(2j, 2j+1) × row 10` (hub + head of
lane); every other lane atom is **alone** on its pair ✓. So exactly one `cz` per segment.

### Non-hub blocks (and HEA)

All partners are parked at **left slots** (even columns) with their right slots free. For a set of
blocks forming a matching we pick, per block, a **dweller** (stays at its lane site) and a
**visitor** that is moved to the dweller's right slot `(2j+1, r)`:

1. move the visitors in (greedy batching: visitors sharing one translation and forming a
   ghost-free rectangle go in one `Move`),
2. pulse → one `cz` on every co-located pair; every other atom is alone ✓,
3. bulk-lower the visitors to storage, apply `rx(θ)` for each block, bulk-lift them back,
4. pulse again → CZ#2,
5. move the visitors home.

For HEA (bare CX, no `rx` in between) steps 3–4 are skipped.
HEA has no hub: the chain is laid along entangling rows (`q_i` at row `i//10`, col `i%10`), so the
even matching `(0,1),(2,3),…` is exactly the zone's pairs (one bulk lift, cost ≈ 5.05) and the odd
matching `(1,2),(3,4),…` is a lift of storage cols 1–9 onto zone cols 0–8 (cost ≈ 5.1).

### Fallback

Any circuit that does not match a recognised family falls back to a naive per-CZ path over
`transpile(circuit, basis_gates=["rx","rz","cz"], optimization_level=1)`, which is valid by
construction. Correctness first. This is not optional: the sanity set contains `rzz` in the *input*
(circuits 03/05/06/07/08/10), `ry`+`cz`-only circuits (09/11), and two kitchen sinks —
`sample_circuit_12` (40 qubits) and `sample_circuit_13` (**70 qubits**, ~10 000 gates) with `ccx`,
`cswap`, `swap`, `cu1`, `u2`, `rxx`, … So the family paths are gated on `n <= 50` (the lane scheme
has 5 × 10 = 50 partner slots) as well as on the family test, and nothing expensive (e.g. the
O(blocks²) commutation DAG) may run on the fallback path.

**`optimization_level >= 2` is forbidden for anything we read gate-by-gate**: it runs
`ElidePermutations`, which removes a `swap` from `.data` and records it as a layout permutation
instead, so reading `.data` is then silently wrong. The sanity kitchen sinks contain 34 and 108
`swap` (plus `cswap`), so this is a live trap (hit by the sibling `gates` branch).

### Why it stays equivalent (ADR-0003)

* Every rewrite above is an exact identity (ZZ sandwich, `cx = H·CZ·H`, ZXZ Euler up to global
  phase). No reliance on the >19q equivalence gap.
* Reordering: QAOA cost-layer blocks are all diagonal and commute freely. For iHVA the blocks are
  ZY-type and do **not** all commute, so blocks are only reordered when the commutation test
  (two Pauli terms commute iff they anticommute on an even number of shared qubits) allows it; the
  hub's blocks `Z_h ⊗ Y_v` pairwise commute (they share only h, on which both act with Z), so
  grouping all hub blocks of one layer is safe.
* ADR-0007: in every segment, either no `cz` at all, or exactly one `cz` on *every* pair that holds
  two atoms. All dwelling atoms are alone on their pairs.
* Unwanted co-location never happens: the only atoms ever placed on a right slot are the hub and
  the current visitors.

Sources: 2411.11784 (reuse-aware / ZAC), 2503.02272 (reduce 1q/2q interleaving), 2509.12083
(composite moves), 2608.01316 (short vertical hops), 2312.03982 (zoned model, transversal CZ).

## Prototype validation (planner, before the compiler existed)

Two throwaway prototypes checked against the real validator (`/tmp/zw/conv.py`, `/tmp/zw/full.py`):

1. **Conveyor, 12-qubit pure star of ZZ blocks.** `verify_movement` OK, `verify_equivalence` =
   `equivalent_up_to_global_phase`. Confirms the slide is legal across the zone boundary, costs
   exactly 1.0 in an even column, and that a lone atom on a pair raises no `PhysicsError`.
2. **Full QAOA path** (conveyor + non-hub visitor rounds, no 1q merging):

| circuit | prototype distance | naive | Δ | prototype depth | naive depth |
|---|---|---|---|---|---|
| 12q qaoa p1 | 356.7 | 1956.5 | −82 % | 50 | 206 |
| 50q qaoa p3 | 3687.8 | 24906.9 | −85 % | 382 | 2275 |

Cost split at 50q: the hub conveyor is only ~100 per cost layer (≈1.0 per `cz`, as predicted); the
non-hub visitor rounds are ~1100 per layer and dominate, because each visitor gets its own `Move`.

Two traps found and fixed in the design: (a) a hub parked on a right slot forms an unwanted
two-atom pair with the lane atom below it — park it at `(9, 19)` opposite a permanently empty slot;
(b) the bulk storage dip must cover **all** columns 0–9, otherwise a visitor left at column 9 gets
an `rx` in the entangling zone. Bulk moves over cols 0–9 produce "not evenly spaced" spacing
warnings; those only affect `move_time`, which is not scored.

### iHVA structure (measured, `/tmp/zw/ihva.py`)

iHVA has **no gates between its p layers**, so a 50q p=3 circuit is one stream of 441 Pauli
rotations `exp(-iθ/2 Z_u Y_v)` (147 hub + 294 non-hub). They anticommute iff exactly one of
`u == v'`, `v == u'` holds. A ready-set (antichain) scheduler preferring hub-Z, then hub-Y, then
non-hub gives 29 groups on `stagger` (hub groups of size 19/13/12/5 and up to 16) and 57 on `tree`.
Hub groups are inherently serial, so each needs a *partial* conveyor pass. Measured cost of the
whole 147-block hub part, by layout:

| partner → lane/slot layout | stagger | tree |
|---|---|---|
| order as-is, block-wise | 936 | 1382 |
| order as-is, round-robin | 1120 | 1224 |
| **sorted by first hub block, block-wise** | **852** | **916** |
| sorted by first hub block, round-robin | 1078 | 1228 |

### Where the remaining cost sits, and the planned fixes

Cost split of the verified 50q QAOA p=3 prototype (3 688 total, ~1 230 per cost layer):

| part | per layer | note |
|---|---|---|
| hub conveyor (98 `cz`) | ~100 | ≈1.0 per `cz` — at the predicted floor |
| hub hops between lanes | ~17 | 4 × 2.0 plus parking |
| mixer storage dip | 20 | 2 bulk moves |
| **non-hub visitor rounds (98 blocks)** | **~1 090** | 5 bulk moves × 10 per matching (350) + ~28 single-atom visitor moves × ~5 per matching (980 over 7 matchings) |

So the conveyor is done and the non-hub rounds are 90 % of what is left. Three fixes, in order of
value:

1. **Super-rounds.** Hoist the storage dips: do every matching's `cz #1` in one zone visit, then
   one bulk dip where every target gets its `rx(θ)`, then every matching's `cz #2`. Valid only if
   the matchings in one super-round have **pairwise disjoint target sets** — otherwise a target `w`
   would see `cz_i, cz_j, rx(θ_i), rx(θ_j), cz_i, cz_j`, which is *not* equal to the product of the
   two blocks (`rx(θ_i)` and `cz_j` both act on `w` and do not commute). Orienting blocks to keep
   the max target-degree at ~2 gives ~2–3 super-rounds, i.e. ~120 per layer instead of ~350.
2. **Target chaining (the idea's namesake).** With per-qubit pending-unitary bookkeeping the `H`
   after one block's `cz #2` cancels the `H` before the next block's `cz #1` on the same target
   (`H·H = I`), so a chained target needs **no storage visit** between two of its blocks — just an
   in-zone re-pair. Also lets a visitor move straight from one dweller's right slot to the next
   instead of going home first (~2 moves per block → ~1).
3. **Placement (literature #7).** The visitor move length (~5 on a round-robin layout) is the
   single biggest multiplier left. Minimise Σ over non-hub edges of the re-pair distance by local
   swaps of the partner → (lane, slot) assignment; no validity risk.

Together these should take 50q QAOA p=3 from ~3 700 to roughly ~1 300–1 800 (−93 to −95 % vs naive).

## What was implemented

All of the below lives in `compilers/zonedwell/compiler.py` (one file, stdlib + numpy + qiskit):

* `parse_blocks` / `_absorb_ihva_pre` — linear scan of the ORIGINAL circuit into 1q items, ZZ
  blocks (`cx; rz; cx`, with the immediately adjacent 1q gates on the target absorbed as pre/post,
  which is what makes an iHVA block `exp(-i th/2 Z_u Y_v)`), bare `cx`, bare `cz` and `measure`.
* `_block_generator` / `_commute` / `plan_groups` — the commutation DAG and ready-set scheduling
  (`hubZ`, then `hubY`, then `non`). Every ready set is an antichain, so all its members commute.
* `plan_layout` + `_LayoutCost` — partners packed block-wise into five even "lanes" of 10 slots,
  then improved by a seeded local search (swap two partners, keep the improvement) on the objective
  `sum over non-hub blocks of the two atoms' grid distance + conveyor slides`. This was worth ~4
  score points and is what makes the non-hub matching cheap.
* `_Machine` — one bulk move (`rows 10..19 <-> 0..9`) switches zones; `conveyor_lane` slides a
  column down/up one row at a time (a *partial* pass is fine: a segment whose row-10 atom is not in
  the group simply emits no `cz`, which ADR-0007 permits); `_conveyor_ops` folds the per-qubit 1q
  gates of a whole pass into one operator before and one after, using `post_j pre_{j+1} = I`
  (`H H = I`, `rx(-pi/2) rx(pi/2) = I`), and refuses the pass if a pair does not cancel.
  `matching_round` + `_assign` + `_place`/`_unplace` place a matching on free pairs (greedy
  min-max displacement), pulse once, take one bulk dip for the `rx` gates, pulse again, and move
  only the atoms that moved back.
* Every 1q gate is buffered as a 2x2 matrix per qubit and emitted as `rz rx rz` only while the atom
  is in storage (`_State.flush`); the buffering is what makes the `H`/`rx(+-pi/2)` cancellations
  automatic, so no gate is ever hand-placed.
* Falls back to `_compile_generic` (transpile to `rx/rz/cz`, then one `cz` at a time) for anything
  outside the family set or above 50 qubits — that is what the 40- and 70-qubit sanity circuits use.

Two things in the original spec turned out to be wrong:
* a "bulk dip" per non-hub matching is not needed for the conveyor, and the conveyor needs no
  dedicated park site if the hub is parked lazily (only before a matching round);
* `Architecture(st.pos)` must be the *initial* loading, not the final positions — most circuits hide
  this because everything returns home, but the hub does not when the last group is a conveyor.

## Results

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|
| stage 1 conveyor + matching | 1e2e764 | +74.73 | 74.14 | 77.09 | 36/36 metric `--mode full` |
| optimised lane packing, hub-less circuits on the lane machine | a8a75ab | +78.80 | 79.26 | 76.95 | HEA 439 -> 151 on 12q p1 |
| initial-loading fix, `cz` needs no `H`, lazy hub parking | e0d4431 | +79.59 | 80.25 | 76.95 | 36/36 metric + 13/13 sanity `--mode full` |

Per family at `e0d4431`: HEA distance +63.5 % / depth +16.1 %, iHVA +76.4 % / +74.9 %,
QAOA +87.0 % / +80.7 %. Max per-circuit compile time 3.3 s.

## Where the remaining distance is

At `e0d4431`, `D = 54 643`. iHVA is 38 315 of it (70 %), QAOA 14 117, HEA 2 216. On a 50-qubit
iHVA p3 circuit (naive 24 907): ~2 700 is bulk zone moves (~135 of them, i.e. about two per
scheduling group, which is the structural minimum), ~900 the conveyor slides, ~2 200 the non-hub
positioning. The next lever is therefore **fewer scheduling groups** (the iHVA tree ordering needs
57, the stagger 29) or a cheaper way to reach storage before a group's first `cz`; merging groups is
blocked because Z-side and Y-side hub blocks anticommute on the hub, so their order is fixed.

## Verdict

(pending)
