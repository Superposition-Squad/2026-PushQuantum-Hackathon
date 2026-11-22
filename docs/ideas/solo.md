# Idea: solo — an iHVA-specialised compiler with a dynamic storage band

- Branch: idea/solo · Worktree: /home/hackathon/challenge-wt/solo · File: `compilers/solo/compiler.py`
- Status: valid end-to-end, optimising. Validator v5.0.0 (`idle` rule) throughout.

## Design

**Frame (exact).** Default "S-frame": every qubit is conjugated by S (initial `S^dag h`, final `S`),
which turns every block Z_c Y_t into Z_c X_t = `cz . rx_t(th) . cz` with no per-block basis change
(verified numerically). When the hub is the block's target it must stay static, so the block uses a
swapped form found by exhaustive search over short Clifford words: rotor c gets `rx(pi/2) rz(pi/2)`
before / inverse after, the hub gets `rz(pi/2) rx(pi/2)` / inverse (one `rx` per hub role switch
in steady state). The old frame ("R", below) and an H-swapped variant stay as search options.

**Special angles (exact).** th = s*pi/2 (mod 2pi) blocks are Clifford: in the S-frame
exp(-i s pi/4 Z_c X_t) = (I x h) . cz . (S^s x S^s) . (I x h) (global phase), i.e. ONE pulse with c
getting S^s (diagonal, stays pending) and t getting h then S^s before / h after; no mid rotation.
Rounds split such blocks into a single-pulse sub-round (2 band moves, no dummies); hub batches of
them need a single pass (hub-as-target: batch-level h on the hub makes every per-block hub gate
diagonal). th = pi (mod 2pi) is Z_c X_t: local Paulis, no cz. Only the S-frame uses these forms;
the other frames fall back to the generic form. All 158 special blocks of the scored set are in
the six tree_warm circuits (every non-zero block there is +-pi/2): 12q 63/22 -> 45/15, 20q 128/32 ->
83/21, 50q 442/72 -> 262/41 (+0.39 points). The optional triangle parity network was skipped: in
the band machine the saved cz would need a pulse ordering with mid rotations on two different
qubits, for a small expected gain.

**Gates (frame "R").** Each iHVA block `rx(pi/2)_t cx(c,t) rz(th)_t cx(c,t) rx(-pi/2)_t` is
exp(-i th/2 Z_c M_t), realised as `t: rx(pi/2), h, [cz], rx(th), [cz], h, rx(-pi/2)` (normal form,
rotor = t) or `c: h, [cz], rx(th), [cz], h; t: rx(pi/2), [cz], -, [cz], rx(-pi/2)` (swapped form,
rotor = c). `th == 0` blocks are dropped (identity). 1q gates are buffered per qubit as a 2x2 matrix;
a diagonal buffer commutes with cz and stays pending, a non-diagonal one is flushed (ZXZ) while the
atom is in storage. The rotor's `rx(th)` sits between the two pulses. Per block the form is chosen
to keep the hub static and minimise 1q depth.

**Schedule.** Exact commutation DAG (Pauli strings anticommute). Steps are either a *round* (a
matching of ready blocks, may include one hub block) or a *hub batch* (all ready hub blocks with the
same hub role: cz#1 for every partner, one `rx` layer, cz#2 in reverse — valid because the blocks
commute). Online scheduling: policies (lookahead weight, hub-batch thresholds, rounds-first) and
position-aware matchings.

**Rounds = dynamic band (the main lever).** No fixed homes. For a round, choose a storage
rectangle (rows a..b near the zone, a pair-column range, column parity 0 or 1) and assign the pairs
to its interaction-pair slots with a Hungarian assignment (distance + eviction penalty). Then
relocate in storage (sequenced, temp sites for cycles, greedy batching into legal rectangle moves),
evict non-members from the rectangle, and lift the whole rectangle rigidly: lift, pulse, drop (rotor
gates), lift, pulse, drop = 4 moves of cost ~h for all pairs together. Atoms stay where they were
dropped (the configuration evolves). Two non-members that sit on one interaction pair inside the
rectangle may ride along as a *dummy pair* (cz·cz = I, nothing applied in between) if the pair is
an interaction pair of the input (validator pair-set check above 19 qubits).

**Hub batches.** The hub stays in zone row 10; partners are lifted one per pulse from storage row
9; rows are rotated through row 9 (`row 9 -> buffer row`, `rows -> +1`; the buffer row only needs to
be free at the columns of row 9; the second pass rotates backwards and replays the first pass in
reverse), or partners are first compacted into the bottom rows, or holes in the bottom rows are first
refilled from above ("gravity", +2-5 %), or partners are lifted directly; the cheapest strategy is
chosen on a cloned machine.

**HEA / generic cz programs.** `cx -> h cz h`, 1q gates buffered, cz's layered into rounds of
disjoint pairs, executed by the same band executor with a single pulse (no dummies), from a 2-row
snake layout; lazy flushing (trailing rz slides through the cz into the next flush) cuts depth.
12q HEA p1/p3: 10.1/10 and 33.2/22 (naive 244/19, 732/43). Bounded work (<= 240 cz, one variant
above 60 cz), otherwise naive.

**Plan simulation.** Per round, the best 6 band plans by the Hungarian estimate are executed on a
cloned machine and the cheapest real outcome (distance + 3.06 x depth delta) is kept (+1-3 % on
50q); the hub strategy is chosen the same way.

**Search.** Deterministic multi-start: 13 base candidates (hub-in-round threshold x lookahead x
matching key x parity set, chosen by a greedy set cover over the 12 non-warm scored iHVA circuits),
then for the best 3-5: simulation-guided round composition (EVAL), estimate-only plans, a first-use
initial layout derived from the candidate's own step order; then seeded cost perturbations of the
best (fixed budgets by size, no wall clock). Score = distance + 3.06 x depth.

**Hub rotations in band rounds.** Both endpoints already return to storage between the two CZs.
The search also tries letting the hub take the middle rotation, using the normal S-frame form
when that avoids basis changes. Dedicated hub sweeps still keep the hub stationary.

**Phase lowering.** After routing, combine Rz gates between consecutive Rx gates on the same atom.
Candidate rewrites also propagate local Pauli corrections through CZ, where an X adds a Z to
its partner, and flip rotation signs through anticommuting corrections. Combined phases are
placed during storage visits. The retiming pass uses the longest dependency paths before and after
each eligible insertion point to place phases. The compiler keeps a rewrite only when it reduces
depth.

**Initial loading and final moves.** Fold storage setup into the initial loading when replay
confirms that the revised positions allow each remaining move. Leave qubits at a valid resting
site after their last interaction when that saves movement. The solver reuses the search's stored
schedules for first-use layouts and compares four finalists using movement distance and gate depth
after both passes.

**Fallback.** Anything not iHVA-shaped (HEA, sanity circuits) goes to the naive scheme (valid; the
portfolio covers HEA).

## Results (v5 leaderboard set)

| version | commit | official | notes |
|---|---|---|---|
| v1 static homes, naive lifts | 294cc4c | +80.88 (fast) | |
| v2 dynamic band + hub rotation | | +86.66 (fast) | rounds 19.5 -> 7/block |
| v3 multi-start, hub blocks in rounds | | +88.22 (fast) | |
| + dummy pairs, rect choice, batching, parity, form choice, search | | +89.92 (fast) | dummies were not pair-restricted: failed full mode on 50q |
| dummies restricted to input pairs + guard | 3a… | +89.80 (full, 20/20) | portfolio_po + solo est. 90.37 |
| + layouts, partial rotation, simulated plans, pruned candidates | ed8ef89 | +90.11 (fast) | |
| + gravity hub strategy, S-frame | | +90.36 (fast) | |
| + HEA path | 8f5366d | +90.86 (full, 20/20) | portfolio_po + solo est. 90.91; solo wins 18/20 |
| + special-angle blocks (+-pi/2 one cz, pi local) | 682d68b | **+91.25 (full, 20/20)** | solo wins 20/20 vs portfolio_po |
| + hub rotations in band rounds and phase lowering | | **+91.48 (full, 20/20)** | 50q tree p3 depth 418 → 367 at the same movement distance |
| + storage-aware phase retiming and boundary moves | | **+91.55 (full, 20/20)** | 50q tree p3 movement 3278.4 → 3248.0 and depth 367 → 362 |

Per circuit vs portfolio_po (dist/depth) at 8f5366d: solo wins 18/20, e.g. 50q stagger p3 4603/471
-> 2664/341, tree p3 5742/554 -> 3278/418, 20q stagger p3 1164/189 -> 783/150, 12q HEA p3 44/28 ->
33/22. Portfolio keeps only 50q tree_warm p1/p3 (375/77 vs 442/72).

Validity evidence at 682d68b: `bench --mode full` 20/20 (QCEC equivalent on all 12q), sanity 13/13
full, strict 20q 6/6 (statevector x3), exact Clifford tableaux equal on all six 50q circuits twice:
once as compiled (Clifford angles exercise the +-pi/2 and pi paths) and once with the special-angle
classification forced off (generic path). Earlier evidence: `bench --mode full` 20/20 (QCEC equivalent on all 12q), `--set sanity --mode
full` 13/13, `--mode strict --filter 20q` 6/6 (statevector x3), exact Clifford tableaux equal on all
six 50q circuits (two angle draws each, `docs/ideas/hubwalk_cliff_check.py`), deterministic, max
compile ~14 s (50q tree p3).

## What did not pay off
- Beam search over per-round plan choices on a fixed step list: no better than the greedy.
- Pooling band plans from several cost-matrix variants, eviction-target policies (push up / keep
  compact), hub role-switch penalty (the DAG fixes the switches), lazy flushing for iHVA (the kept rz
  lands on the rotor's mid rx, on the critical path).
- Merging same-pair blocks: none are mergeable in the scored set.
- Lookahead term (next partner) in slot assignment: noise level (+-3 %).
- Distance-aware matching keys alone: +-2 %.
- Seeded perturbations: 0-3 %.
- Direct lifts into preselected bands, non-rigid columns, reverse routing, separately executed
  sub-bands, spectral starts, and physical line re-embedding gave small or inconsistent gains.
- Additional hub sweeps and sparse bands matched the previous 50q scores. Regrouping transfers
  between fixed pulse positions saved 8.4 total movement units across the six 50q cases.

## Cross-round residency

On 50q tree p3, 55 transitions between different multi-pair CZ layers accounted for 1902.6 distance.
The median transition shared nine atoms that needed no intervening one-qubit gate. At each new
pulse, every resident needs its scheduled partner. A local experiment that varied the next round's
zone slots and reconnected to the original storage layout found no savings over the fixed-endpoint
replan. To exploit the overlap, scheduling and routing would need to choose matchings and storage
layouts over several rounds together.
