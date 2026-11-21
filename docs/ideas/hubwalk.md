# Idea: hubwalk

- Branch: idea/hubwalk · Worktree: /home/hackathon/challenge-wt/hubwalk
- Base: main f462cf5 (+ bounds d0ed179); compiler forks `compilers/zonedwell/compiler.py`
- Status: done — ready for merge review (final 775f761, code = 0438413; +88.91 % standalone, ADR-0010 compliant)

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

## Design (final)

`compilers/hubwalk/compiler.py` (single file, stdlib + numpy + qiskit; forked from zonedwell).

- **Phases** (`plan_phases`): instead of one antichain per group, a phase is a DAG-closed chain of
  same-type blocks (hubZ / hubY / non-hub); four priority orders are tried, fewest hub phases win.
  iHVA 50q stagger p1: 11 groups → 4 phases; tree p3 ~57 → 29.
- **Hub detection by dominance** (deg ≥ 4 and ≥ 2× the second degree), so the elided tree_warm
  circuits (hub degree 31 / 11) still get a hub path.
- **Exact θ ≡ 0 elision**: a `zz` block whose mid is the identity and whose pre·post is the identity
  is dropped.
- **Layout**: partners on even-column lanes, slots filled **row-major** (shallow lanes for 12/20 q),
  hub parked at (9,10) with (8,10) kept empty (every dip carries the hub into storage), SA over
  the assignment (200·n steps, cost = 2·physical leaf move + 0.5·pair row + hub-pass term).
- **Clifford-frame synthesis**: every block is a Pauli rotation exp(−iθ′/2 P⊗Q) (signs checked
  numerically from the block's own 4×4). Pending 1q gates are kept as a frame U; the block is
  emitted as CZ · exp(−iφ/2 R)_leaf · CZ in the conjugated frame (center must see ±Z, leaf ±X/±Y),
  so H / rx(±π/2) wrappers disappear and qubits only visit storage when their frame class must
  change (then with a chosen Clifford frame). iHVA ZY blocks are exactly CZ·ry(θ)·CZ. Y-axis mids are
  emitted as S†·rx with the trailing S folded into the frame (2 gates).
- **Star rounds** (non-hub phases): CAP = 1 rounds (matchings) from ready sets; the deeper endpoint
  moves onto the right slot of the shallower one (pair sits on the lower row); ONE mid dip per round
  (depth = deepest leaf); QAOA phases use a fixed max-cut bipartition (centers / leaves) so frames
  change ~once per phase. **Lazy return** (QAOA variant): movers stay parked; idle co-located pairs
  get cz in phase A and phase B (cz·cz = I, their pending ops stay buffered and are never flushed in
  between) — exact, and ADR-0007 holds because every occupied pair gets exactly one cz per pulse.
- **Hub walk** (ADR-0010): the conveyor (a lane sliding across rows 9/10) is illegal under the
  zone-span rule, so hub phases are star rounds with the hub as center AND mover: partners stay
  home as leaves, the hub walks over their right slots (snake by lane, 1.0 vertical / 2.0 lane
  change), one restricted dip, then the walk in reverse. 50q QAOA: ~105 per layer vs ~100 for the old
  conveyor.
- **Dips** move only the crossing rows (10..9+d → 10−d..9 and back), so every multi-row move stays
  inside one zone (ADR-0010); `_State.move` asserts the rule.
- **Per-circuit variant selection**: {row-major seed 12345, lane-order seed 777 (≥ 20 q), row-major
  seed 999 (≤ 20 q)} × {lazy on, off}, each compiled on a copy; the lowest official cost
  0.8·D/276726.91 + 0.2·P/23676 wins (deterministic, ties → first). Max compile 20 s (50q QAOA p3).
- Anything else (HEA, sanity kitchen sinks) goes through zonedwell's lane matching / generic path.

## Design notes (rejected alternatives)

- Hub walk over lone partners in the zone (brief fact 3): 49 lone partners need all 10 zone rows,
  so the mid storage trip costs 2 × 10; ~126 per hub round vs the conveyor's ~108.
- Full-density storage rows lifted into the zone (5 cz per 1.22 move) are 4× cheaper per cz, but
  only adjacent atoms of a storage row pair up; the non-hub graph is a path plus ~50 random chords,
  and re-sequencing atoms between matchings has no cheap primitive (monotone maps never reorder).
- Riding non-hub blocks on the conveyor (a second lane in the rectangle's other boundary column):
  works for one stationary center per pass (≤ 6 blocks), too little.
- Whole-lane shears with the parity relaxation: offline model ≈ 870 for the 50q QAOA p1 non-hub
  part (random chords leave few edges per shear); the parity rule is used for lazy return instead.
- MILP round packing (scipy HiGHS): the depth objective does not close in 10 s; CAP-2 rounds and
  CAP-2/3 center walks cost more moves than the dips they save (measured, not kept).

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
| 6S: layout seed/order variants | 0438413 | +88.91 | 90.99 | 80.57 | 1–3 layouts × lazy on/off, picked by official cost; max compile 22 s. **Final** (full 36/36 + sanity 13/13 under ADR-0010, planner-verified) |
| 6C: CAP≥2 center-walk rounds | (not kept) | +88.90 | | | walk=2 never beat walk=1 on the chosen layout and pushed 50q QAOA p3 to 53 s; reverted to 6S |

Planner offline models (throwaway, not committed): with random chords, neither star rounds with
SA layout (≈800 for the 50q QAOA p1 non-hub part) nor whole-lane shears (≈870) batch well; the
gains come from fewer phases/rounds/dips, not from batching.

STAR_CAP sweep at step 1 (greedy, fast): 1 → +81.45, 2 → +79.16, 3 → +79.04 (extra phase-A/B moves
outweigh the saved dips).

## Verdict

**Done, ready for merge review: +88.91 % standalone (distance +90.99 %, depth +80.57 %) at
775f761 (compiler identical to 0438413), valid under ADR-0010.** zonedwell was +80.79 %.

Per-circuit against the current post-ADR-0010 portfolio (main 4bcdc71, +82.23 %) — hubwalk wins
25 of 36 circuits (all 12 QAOA, all 12 20q/50q iHVA, 12q iHVA stagger p3); estimated portfolio
with hubwalk added: **82.23 → 89.41 %**.

| circuit | file | portfolio D/P | hubwalk D/P | pick |
|---|---|---|---|---|
| 01 | 12q_hea_linear_p1 | 15/12 | 82/15 | portfolio |
| 02 | 12q_hea_linear_p3 | 44/28 | 205/31 | portfolio |
| 03 | 20q_hea_linear_p1 | 12/12 | 101/15 | portfolio |
| 04 | 20q_hea_linear_p3 | 36/28 | 266/31 | portfolio |
| 05 | 50q_hea_linear_p1 | 45/12 | 240/15 | portfolio |
| 06 | 50q_hea_linear_p3 | 134/28 | 712/31 | portfolio |
| 07 | 12q_ihva_stagger_p1 | 197/45 | 204/56 | portfolio |
| 08 | 12q_ihva_stagger_p3 | 596/140 | 547/144 | **hubwalk** |
| 09 | 12q_ihva_tree_p1 | 196/64 | 250/72 | portfolio |
| 10 | 12q_ihva_tree_p3 | 612/184 | 707/213 | portfolio |
| 11 | 12q_ihva_tree_warm_p1 | 91/19 | 90/32 | portfolio |
| 12 | 12q_ihva_tree_warm_p3 | 91/19 | 90/32 | portfolio |
| 13 | 20q_ihva_stagger_p1 | 481/56 | 285/69 | **hubwalk** |
| 14 | 20q_ihva_stagger_p3 | 1263/236 | 763/193 | **hubwalk** |
| 15 | 20q_ihva_tree_p1 | 450/73 | 308/83 | **hubwalk** |
| 16 | 20q_ihva_tree_p3 | 1477/197 | 964/204 | **hubwalk** |
| 17 | 20q_ihva_tree_warm_p1 | 195/41 | 125/37 | **hubwalk** |
| 18 | 20q_ihva_tree_warm_p3 | 195/41 | 125/37 | **hubwalk** |
| 19 | 50q_ihva_stagger_p1 | 1971/207 | 877/147 | **hubwalk** |
| 20 | 50q_ihva_stagger_p3 | 5579/576 | 2625/412 | **hubwalk** |
| 21 | 50q_ihva_tree_p1 | 1924/223 | 1255/166 | **hubwalk** |
| 22 | 50q_ihva_tree_p3 | 5783/644 | 3625/450 | **hubwalk** |
| 23 | 50q_ihva_tree_warm_p1 | 758/83 | 295/77 | **hubwalk** |
| 24 | 50q_ihva_tree_warm_p3 | 758/83 | 295/77 | **hubwalk** |
| 25 | 12q_qaoa_p1 | 229/62 | 184/53 | **hubwalk** |
| 26 | 12q_qaoa_p3 | 669/133 | 490/148 | **hubwalk** |
| 27 | 12q_qaoa_warm_p1 | 229/62 | 184/53 | **hubwalk** |
| 28 | 12q_qaoa_warm_p3 | 665/131 | 490/148 | **hubwalk** |
| 29 | 20q_qaoa_p1 | 479/79 | 256/66 | **hubwalk** |
| 30 | 20q_qaoa_p3 | 1468/189 | 728/184 | **hubwalk** |
| 31 | 20q_qaoa_warm_p1 | 479/85 | 256/66 | **hubwalk** |
| 32 | 20q_qaoa_warm_p3 | 1412/183 | 728/184 | **hubwalk** |
| 33 | 50q_qaoa_p1 | 2021/274 | 832/141 | **hubwalk** |
| 34 | 50q_qaoa_p3 | 5861/677 | 2457/389 | **hubwalk** |
| 35 | 50q_qaoa_warm_p1 | 2021/273 | 832/141 | **hubwalk** |
| 36 | 50q_qaoa_warm_p3 | 5861/679 | 2457/389 | **hubwalk** |

portfolio 82.23  hubwalk standalone 88.91  portfolio+hubwalk 89.41


Validity (planner, independently of the implementer, on frozen snapshots):
`--mode full` 36/36 with the ZoneSpanError check active; `--set sanity --mode full` 13/13 (incl.
sample_circuit_11 QCEC-exact, the 40/70-qubit kitchen sinks via the generic path); `--filter 20q
--mode strict` 12/12 (statevector ×3); 50q Clifford-tableau check
(`docs/ideas/hubwalk_cliff_check.py`: every non-Clifford angle → pseudo-random kπ/2, two seeds,
stabilizer tableaux of compiled vs reference compared; verified to catch a dropped cz and a
perturbed rx) equal on 50q QAOA p3/p1, iHVA stagger/tree/tree_warm p3, HEA p3. Deterministic
(identical output across runs and PYTHONHASHSEED values). Max compile 20 s (50q QAOA p3, 6 variants);
35 s on the 70-qubit sanity circuit.

Where the 50q p3 distance still goes (final HUBWALK_STATS): QAOA 2457 = hub walk 315 + dips 790 +
round moves 1329 + other 23; iHVA stagger 2625 = 337 + 678 + 1568 + 43; iHVA tree 3625 = 589 + 1265 +
1648 + 123. The bound is 294 each.

What worked (in order of value): Clifford-frame synthesis (most flush dips gone), DAG-closed phases
and θ≡0 elision with dominance hub detection (iHVA), row-major shallow lanes (12/20 q), mover =
deeper endpoint, hub parked on row 10, layout SA tuning, per-circuit variant selection.
What did not: batching leaf moves (random chords give every leaf its own displacement — offline
models and step 1 agree), MILP rounds, CAP ≥ 2 rounds / center walks, deeper-pairs-first ordering;
lazy return and the QAOA bipartition helped only a little (+0.03 each).

Next steps: (1) the non-hub round moves are now 55 % of 50q cost — a layout that changes between
phases (cheap lane-level permutations) or a different primitive for random chords is needed to go
below ~1 move of ~3 per block; (2) iHVA tree still has 25 hub phases — a look-ahead phase scheduler
(type-change lower bound is ~22) and treating 1-block hub phases via the hub's visitor slot (8,10);
(3) bipartite edge colouring (König, exactly Δ rounds) for the QAOA cut blocks to cut ~2 dips per
layer.
