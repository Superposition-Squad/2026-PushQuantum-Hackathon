# Idea: hubwalk

- Branch: idea/hubwalk · Worktree: /home/hackathon/challenge-wt/hubwalk
- Base: main f462cf5 (+ bounds d0ed179); compiler forks `compilers/zonedwell/compiler.py`
- Status: in progress

## Hypothesis

On 50q QAOA p1, zonedwell spends 970: ~100 on the hub conveyor (at the 1.0/cz floor), ~200 on 10.0
bulk zone switches/dips, and **~550 on single-atom visitor moves** for the 98 non-hub blocks
(both atoms of every block move to a free pair, one `Move` each, ~2.6 moves per block). The bound
(docs/bounds.md) is 98. The non-hub part is where the distance is.

Analysis (planner, see "Design notes" below) of what one move can buy:

- Any re-pairing costs ≥ 1.0 (vertical hop) or ≥ 1.7 (hop to the right slot of the lane on the
  left). The hub's 98 sequential cz per layer therefore cost ≥ 98, and the conveyor achieves that.
- A conveyor move can serve at most the 2 boundary columns of its rectangle, so the non-hub blocks
  cannot ride on the hub's slides (a hub leaf may meet nobody but the hub between its storage
  visits). Non-hub blocks need their own moves.
- Every ZZ block forces its *leaf* (the atom with the mid `rx`) through storage between its two cz.
  That trip is the expensive part (zone dips are as deep as the deepest leaf).

So hubwalk keeps the conveyor and rebuilds the non-hub part around **star rounds**:

1. **Only the leaf moves.** Centers stay at their lane home (left slot, even column); the leaf
   moves to the center's right slot. Halves the moves vs zonedwell's "both atoms to a free pair".
2. **Star rounds with one mid dip.** A round is a set of commuting blocks where every qubit is
   either a leaf of exactly one block or a center (no mid gate) of up to `CAP` blocks. Phase A =
   `T ≤ CAP` pulses (leaves of pulse t visit their centers), then ONE shallow dip for all leaves'
   `rx(θ)`, then phase B replays the pulses in reverse (the last pulse's leaves stay in place across
   the dip, saving two moves). zonedwell pays a dip per matching; matchings ≈ 7 per layer, rounds ≈ 3–4.
3. **Batched leaf moves.** Leaves of one pulse with the same displacement whose sources form a
   ghost-free rectangle move in one `Move` (cost = max, not sum).
4. **Layout optimised for rounds** (step 2): SA over the partner → (lane, slot) assignment with an
   objective that runs the round planner (number of move groups × length + dip depths), so leaves
   end up one lane to the right of their centers at the same row (vector (−1.7, 0), batchable).

Estimate, 50q QAOA per layer: conveyor 108 + zone switches/mixer 40 + rounds 4 × (dip ~15 +
moves ~25) ≈ 310 ⇒ p1 ≈ 320 (zonedwell 970), p3 ≈ 930 (2911). iHVA uses the same rounds for its
non-hub antichains.

## Design

See the implementer spec in the session log; summary:

- `plan_star_rounds(group, home)` (pure): orientation (iHVA: leaf = cx target `v`, which carries
  the `rx(±π/2)` pre/post; QAOA: either end), greedy round packing with `CAP`, pulse index per leaf.
- `_Machine.star_rounds(group)`: queue leaf pre ops, `flush_dip` for anyone with non-identity
  pending (centers included), phase A, mid dip (`flush(leaf)` + `rx(θ)`), phase B, queue post ops.
  Falls back to zonedwell's `matching(group)` if planning fails (before anything is emitted).
- Validity: every pulse's occupied pairs are exactly the (center, leaf) pairs of that pulse
  (centers and idle atoms lone at home, hub parked lone at (9,19)), so ADR-0007 holds with no
  parity tricks. Rewrites are zonedwell's (exact ZZ sandwich, 2×2 buffering flushed only in
  storage); blocks in a group commute (antichain), so reordering into rounds is exact. Evidence:
  QCEC ≤ 19 q, statevector 20 q, Clifford-tableau check (angles = π/2) at 50 q.

## Design notes (rejected alternatives)

- Hub walk over lone partners in the zone (brief fact 3): 49 lone partners need all 10 zone rows,
  so the mid storage trip costs 2 × 10; ~126 per hub round vs the conveyor's ~108.
- Full-density storage rows lifted into the zone (5 cz per 1.22 move) are 4× cheaper per cz, but
  only adjacent atoms of a storage row pair up; the non-hub graph is a path plus ~50 random chords,
  and re-sequencing atoms between matchings has no cheap primitive (monotone maps never reorder).
- Riding non-hub blocks on the conveyor (a second lane in the rectangle's other boundary column):
  works for one stationary center per pass (≤ 6 blocks), too little.
- Parity relaxation (idle–idle co-locations cancel if pulsed an even number of times) is exact and
  would allow whole-lane shears; kept for step 3 if batching is the bottleneck.

## Interim status (planner, 17:30 UTC)

- Standalone (fast, 36/36): **+87.06 %** at 02bad93 (zonedwell +80.79). Estimated portfolio with
  hubwalk added: **86.24 → 87.89** (hubwalk wins 20 circuits: all 12 QAOA, 20q iHVA
  stagger/tree, 50q iHVA stagger/tree). Full mode 12q passes (QCEC); 50q Clifford-tableau check
  (`/tmp/hwproto/cliff.py`, verified to catch a dropped cz / perturbed rx) equal on 50q QAOA p1,
  iHVA stagger p1, tree p1, tree_warm p3.
- In progress: ILP round packing (2b). Next: Clifford-frame block synthesis (half of all dip
  distance is flush dips caused by H/rx(±π/2) wrappers; iHVA ZY blocks are exactly
  CZ·ry(θ)·CZ with no wrappers), hub park on row 10, lazy return via the parity rule.

## Results

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|
| fork of zonedwell | da6a920 | +80.79 | 81.68 | 77.25 | baseline |
| 1: star rounds (leaf-only moves, one dip per round), STAR_CAP=1 | 0e3065b | +81.45 | 82.42 | 77.57 | 50q QAOA p1 970→920; leaf moves barely batch (181 moves for 98 blocks) |
| 2a: DAG-closed phases (iHVA) | 73b90b7 | +82.12 | | | stagger p1 11 groups → 4 phases |
| 2f: row-major lane slots | 3281ce1 | +83.81 | | | shallow lanes for 12/20q: 20q QAOA p3 1085→810 |
| 2e: exact θ≡0 block elision | f8e3a28 | +86.43 | | | tree_warm circuits |
| 2c: conveyor pre-flush by shallow dip | 4c8dcec | +86.69 | 88.41 | 79.83 | portfolio with hubwalk ≈ 87.58 (wins 20 circuits) |
| 2g: layout SA (200× steps + leaf-move cost) | 02bad93 | +87.06 | 88.90 | 79.72 | 50q QAOA p1 904→841; 50q iHVA stagger p3 3258→3251 |
| 2b: MILP round packing | (not kept) | +87.06 | | | HiGHS: cap-1 depth objective times out (gap never ≤0.05 in 10 s); cap-2 feasibility is instant but ~50 % more leaf moves → 50q QAOA p1 841→1218. Greedy kept. |
| 3H: hub parked on entangling row 10 | 6538907 | +87.25 | 89.11 | 79.77 | hub home (9,0)/(9,10); every dip reaches the hub, lane 4 needs no hop |
| 3F: Clifford frames in star rounds | b4b61b4 | +87.85 | 90.31 | 77.98 | no H/rx(±π/2) wrappers; flush dips mostly gone, +depth from `rz rx rz` mids |
| 3W: hub by degree dominance (deg≥4, ≥2×second) | 4a9ffaa | +88.36 | 90.64 | 79.25 | tree_warm hub deg 31/11 now rides the conveyor: 50q p3 1042→397 |
| (3) mover decoupled from leaf role | (this commit) | +88.47 | 90.77 | 79.25 | the deeper endpoint visits the shallower's right slot → shallower dips |
| 3F conveyor frames | (not kept) | +88.16 | 90.76 | 77.73 | distance −331 but +depth from `ry` mids; net below the old conveyor |
| 4D: 2-gate Y mids (S folded into the leaf frame) | a6b2e3a | +88.67 | 90.77 | 80.26 | depth 4913→4674 at equal distance; conveyor frames retried with it: +88.6705 vs +88.6708, not kept |
| 4L: lazy return + parity rule | f54d907 | +88.70 | 90.91 | 79.87 | movers stay on their anchors; idle pairs get cz·cz=I. Distance 25531→25156, depth 4674→4765 |
| 4O: deeper pairs first | (not kept) | +88.64 | 90.89 | 79.66 | with 4L it regressed (52q QAOA depth); reverted |
| 5Q: fixed bipartition for QAOA phases | 3b6a209 | +88.73 | 90.93 | 79.92 | ~1 frame-change dip per pass instead of one per round |
| 5V: per-circuit lazy on/off selection | 9ef4e3e | +88.79 | 90.92 | 80.26 | QAOA picks lazy, iHVA picks non-lazy; depth 4753→4673 |
| ADR-0010: zone-span compliance | ed8a876 | +88.85 | 90.92 | 80.57 | dips restricted to rows 10..9+depth; conveyor → hub walk; `_State.move` guard; all 36 + sanity pass under the new bench check |

Planner offline models (`/tmp/hwproto/sim*.py`): with random chords, neither star rounds with
SA layout (≈800 for the 50q QAOA p1 non-hub part) nor whole-lane shears (≈870) batch well; the
gains come from fewer phases/rounds/dips, not from batching.
| step 1: instrumentation + star rounds, `STAR_CAP=1` | (this commit) | +81.45 | 82.42 | 77.57 | 36/36 metric + 13/13 sanity `--mode full`; zonedwell base +80.79 |
| `STAR_CAP=2` | | +79.16 | 79.54 | 77.63 | too many phase-A/B moves |
| `STAR_CAP=3` | | +79.04 | 79.43 | 77.45 | |

Step 1 alone already beats zonedwell: cap 1 gives every round `T=1`, so a round is *two* batched
leaf moves (in, out) around one mid dip, while the hub-less group that used to need one matching per
block now packs into one round per "star matching". Per family: HEA unchanged, iHVA distance
+79.17 % / depth +75.89 % (zonedwell +78.43 / +75.30), QAOA +88.35 % / +80.70 % (zonedwell
+87.55 / +80.74). 50q QAOA p1 970.3 → 919.6, 20q QAOA p1 388.6 → 361.8, 50q iHVA stagger p1
1484.7 → 1434.3, 20q iHVA stagger p1 658.7 → 620.2. The remaining round cost is dominated by
unbatched leaf moves on the dense zonedwell layout — step 2's round-aware layout is what should
turn those into shared-displacement rectangles.

## Verdict

(pending)
