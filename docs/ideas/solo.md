# Idea: solo — an iHVA-specialised compiler with a dynamic storage band

- Branch: idea/solo · Worktree: /home/hackathon/challenge-wt/solo · File: `compilers/solo/compiler.py`
- Status: valid end-to-end, optimising. Validator v5.0.0 (`idle` rule) throughout.

## Design

**Gates (exact).** Each iHVA block `rx(pi/2)_t cx(c,t) rz(th)_t cx(c,t) rx(-pi/2)_t` is
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
9; rows are rotated through row 9 (`row 9 -> free row`, `rows -> +1`, reversed for the second pass,
so no restore is needed), or partners are first compacted into the bottom rows, or lifted directly;
the cheapest strategy is chosen on a cloned machine.

**Plan simulation.** Per round, the best 6 band plans by the Hungarian estimate are executed on a
cloned machine and the cheapest real outcome (distance + 3.06 x depth delta) is kept (+1-3 % on
50q); the hub strategy is chosen the same way.

**Search.** Deterministic multi-start: 13 base candidates (hub-in-round threshold x lookahead x
matching key x parity set, chosen by a greedy set cover over the 12 non-warm scored iHVA circuits),
then for the best 3-5: simulation-guided round composition (EVAL), estimate-only plans, a first-use
initial layout derived from the candidate's own step order; then seeded cost perturbations of the
best (fixed budgets by size, no wall clock). Score = distance + 3.06 x depth.

**Fallback.** Anything not iHVA-shaped (HEA, sanity circuits) goes to the naive scheme (valid; the
portfolio covers HEA).

## Results (v5 leaderboard set)

| version | commit | official | notes |
|---|---|---|---|
| v1 static homes, naive lifts | 294cc4c | +80.88 (fast) | |
| v2 dynamic band + hub rotation | | +86.66 (fast) | rounds 19.5 -> 7/block |
| v3 multi-start, hub blocks in rounds | | +88.22 (fast) | |
| + dummy pairs, rect choice, batching, parity, form choice, search | | +89.92 (fast) | dummies were not pair-restricted: failed full mode on 50q |
| dummies restricted to input pairs + guard | see git log | **+89.80 (full, 20/20)** | portfolio_po + solo est. **90.37** |

Per circuit vs portfolio_po (dist/depth): solo wins 12q stagger/tree p3, all six 20q stagger/tree,
all four 50q stagger/tree (50q stagger p3 4603/471 -> 2991/364, tree p3 5742/554 -> 3474/424).
Portfolio keeps HEA and the tree_warm circuits.

Validity evidence: `bench --mode full` 20/20 (QCEC equivalent on all 12q), `--set sanity --mode
full` 13/13, `--mode strict --filter 20q` 6/6 (statevector x3), exact Clifford tableaux equal on all
six 50q circuits (two angle draws each, `docs/ideas/hubwalk_cliff_check.py`), deterministic, max
compile ~14 s (50q tree p3).

## What did not pay off
- Lookahead term (next partner) in slot assignment: noise level (+-3 %).
- Distance-aware matching keys alone: +-2 %.
- Seeded perturbations: 0-3 %.
