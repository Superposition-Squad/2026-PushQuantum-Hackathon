# Idea: zonedwell

- Branch: idea/zonedwell · Worktree: /home/hackathon/challenge-wt/zonedwell
- Base: main 0ea558b, compiler starts from `compilers/naive/compiler.py`
- Status: in progress

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

Any circuit that does not match a recognised family (the 13 sanity circuits, odd sizes) falls back
to the naive per-CZ path, which is valid by construction. Correctness first.

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

## Results

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|

## Verdict

(pending)
