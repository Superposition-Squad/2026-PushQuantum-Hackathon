# Project walkthrough (state 2026-11-22 ~01:00 CET)

A guided tour of the planqc compiler track: what the task is, the core ideas, what worked and what
did not, which parts the compilers share, and a deep dive into every compiler in `compilers/`.
Numbers are on the **v5 leaderboard set** (20 circuits, validator v5.0.0) unless marked otherwise.

## 1. The task in one page

- **Machine:** 100 storage sites (rows 0–9) below a 100-site entangling zone (rows 10–19). In the
  zone, columns (2k, 2k+1) of a row form an *interaction pair* (0.3 apart; pairs 2.0 apart).
  Storage row 9 and zone row 10 are 1.0 apart.
- **Native gates:** `rx`, `rz` only on atoms in storage; `cz` only on the two atoms of a pair in the
  zone. Atoms are moved by AOD **moves**: pick a contiguous rectangle of rows × columns (every
  atom inside moves), map rows and columns monotonically (order is preserved), drop. One move
  costs the **longest single-atom path** in it, however many atoms it carries.
- **Score:** `0.8 · %less distance + 0.2 · %less depth` vs the naive compiler, summed over the
  scored circuits. One depth unit is worth 3.06 distance units on the v5 set.
- **Scored set (v5, final):** the 18 iHVA circuits (12, 20, 50 qubits; stagger / tree / tree_warm;
  p = 1, 3) and the two 12-qubit HEA circuits. iHVA ≈ 90 % of the naive distance, **50q alone
  ≈ 61 %**. QAOA and larger HEA no longer count.
- **Rules that shape the design** ([rules.md](rules.md)):
  - **idle rule:** in every CZ layer (the gates between two moves) every atom in the zone must
    take part in a CZ. Nobody may wait in the zone while a pulse fires.
  - **single-zone moves:** a move picks up in one zone and drops off in one zone.
  - no `rzz`, true equivalence at every size, same qubit count, 300 s per circuit.

## 2. The core ideas (vocabulary)

| idea | what it means |
|---|---|
| **block** | the circuits are built from 2-qubit Pauli rotations, e.g. `exp(-iθ/2 Z⊗Y)` (iHVA) = `cx · rz · cx` wrapped in basis changes. On this machine a block is `CZ · rx(θ) on one atom · CZ` |
| **control / target (rotator)** | only one atom of a block (the *target*) needs the `rx` between the two CZs, so only it has to visit storage in between |
| **hub** | in iHVA/QAOA one qubit (the Max-Cut reference) meets every other qubit: 2(n−1) serial CZs per layer. It sets the number of pulses and the critical path |
| **round / pulse / CZ layer** | a set of disjoint pairs pulsed together between two moves |
| **commutation** | many blocks commute (all QAOA blocks; iHVA blocks with compatible Paulis), so they may be reordered and grouped into bigger rounds |
| **batching** | moves cost their longest path, not the sum: carry many atoms per rectangle. Ghosts (atoms inside the rectangle that must stay) and order preservation are what break batches |
| **staging row** | storage row 9 is 1.0 from the zone: atoms lifted from there are cheap; atoms deeper cost one unit per row |

## 3. History: three rule regimes in one day

1. **v0 rules (lone atoms in the zone allowed):** the winning designs kept atoms *in the zone*:
   zonedwell's vertical **conveyor** slid partner columns through the boundary past a parked hub
   (1.0 per hub CZ), hubwalk/systolic refined the non-hub part. Portfolio up to +87.8, our old
   optimizer +89.5, strict portfolio +89.9 (all on the old 36-circuit set).
2. **ADR-0010 (no multi-row move spanning both zones):** killed the conveyor; hub walks replaced it
   at small cost.
3. **v1 → v5 (idle rule, ~20:10):** zone dwelling is dead. Every pulse now needs atoms to come up
   and go down. Everything valid had to be redesigned: atoms live in **storage**, the zone only holds
   the current pairs. Scores dropped by 4–10 points for everyone; the target set shrank to 20
   circuits.

## 4. Approaches: what worked, what did not

| approach | where | result under v5 | verdict |
|---|---|---|---|
| per-CZ trips (naive) | naive | 0 | baseline |
| exact gate resynthesis + commutation rounds | gates | +73.24 | the gate layer everybody reuses; movement too simple |
| generic rectangle batching of rounds | rectbatch | +79.83 | good on HEA, weak on hub circuits |
| hub parked in zone + streamed partners | hubstream | 12/20 valid (8 idle violations) | partly invalid under idle (parks atoms) |
| zone-resident lanes + conveyor | zonedwell, hubwalk | invalid under v5 | best under v0, dead now |
| **storage homes + hub stream + row rotation** | **systolic** | **+85.34** | best single compiler under v5 |
| beam search over pulses, atoms in storage, block drops | optimizer-optimizer | +81.79 | wins 6 of 8 12q iHVA circuits, weak at 50q |
| **post-optimizer on any output** | postopt, `*_po` | systolic 85.34 → 86.30 | generic, safe, +1 point |
| deeper postopt (beam / local search over layer configs) | experiment | no gain | myopic; fixed homes |
| free working row, narrower rows | systolic2 | ±0 / worse | homes must move, see §6.9 |
| **portfolio of all `*_po`** | **portfolio_po** | **+86.63** (33/33 full) | **current best, packaged** |

## 5. Shared building blocks

| part | where | used by |
|---|---|---|
| **front end:** parse into blocks + 1q runs, recognise `cx rz cx` (and iHVA `rx(±π/2)` wraps), elide θ = 0 blocks, exact Pauli commutation DAG | `gates` (reference), `zonedwell`/`hubwalk`/`systolic` (`parse_blocks`), `optimizer-optimizer` (`front_end`, `build_dag`) | all hub compilers |
| **1q synthesis:** merge every 1q run into ≤ 3 `rz/rx` (ZXZ/XZX), Clifford basis changes for the block frames | every compiler | depth |
| **rectangle batcher:** group atom relocations into legal moves (contiguous, order-preserving, ghost-free, single-zone) | `rectbatch` (`move_params`), `systolic` (`_plan_moves`), `optimizer-optimizer` / `postopt` (`group_transfers` / `_po_group`) | all |
| **validator-exact replay** before returning | portfolio builder (embeds the official validator), `postopt` (`_po_replay_ok`), `optimizer-optimizer` (`self_check`), systolic (`_State` asserts the idle rule) | safety |
| **portfolio builder** | `bench/build_portfolio.py` (ADR 0008) | runs components, keeps the cheapest *valid* result per circuit |
| **post-optimizer wrapper** | `bench/make_postopt.py` → `compilers/<name>_po/` | systolic, systolic2, optimizer-optimizer, hubstream, rectbatch, gates |
| **harness** | `bench/bench.py` (v5 set by default; fast / full / strict), `bench/devbench.sh`, `bench/viz.py` | all |

What differs between compilers is almost entirely the **movement layer**: where atoms wait, how a
round is lifted and returned, how the hub's CZ chain is served.

## 6. The compilers, one by one

### 6.1 `naive` (reference, 0.00)
Transpiles to `rx/rz/cz` at optimization level 0, then for **every** CZ carries the two atoms up,
pulses, carries them back. Reproduces `naive_baseline.json` exactly; defines the score's zero.

### 6.2 `gates` (+73.24)
The gate-level reference: exact resynthesis (`cx → H·CZ·H`, 1q-run merging), the **hub as control**
(its wire carries only CZs, so it never needs storage inside a layer), and **commutation-aware
rounds** (edge-colouring of commuting blocks). Movement is a simple per-round lift/return with a
compact loading near the zone. Valid under v5. `gatesb` is a batching variant (weaker overall).

### 6.3 `rectbatch`
Generic **rectangle batching**: schedule rounds, place partners side by side in a storage row, lift
each round with as few rectangles as possible (ideally one: cost ≈ 2·(10 − top row)). Validator-
exact move checker, deterministic placement search. Strong on HEA (12q HEA is part of the scored
set), weaker on hub circuits where one hub meets everyone.

### 6.4 `hubstream`
Parks the hub in the zone and **streams partners** from row 9 (1.0 per crossing), orients every block
so the mid rotation lands on the partner, cancels the iHVA `rx(±π/2)` on the hub. Designed under v0:
the parts that park atoms in the zone violate the idle rule, so the official portfolio uses it only
where its output is valid.

### 6.5 `zonedwell` (invalid under v5)
The v0 winner: partner columns as **vertical conveyors** through the zone boundary past a parked
hub (exactly 1.0 per hub CZ), lone atoms dwelling in the zone between CZs. Killed first by ADR-0010
(rectangles across the boundary), then by the idle rule.

### 6.6 `hubwalk` (invalid under v5)
Fork of zonedwell: keeps the conveyor, rebuilds the non-hub part around **star rounds** (only the
leaf moves to its center's right slot; one shallow dip for all leaves' `rx`), Clifford-frame block
synthesis, layout SA. +88.91 under v0 (36 circuits), stopped after the idle rule (+81.56, no wins).

### 6.7 `systolic` (+85.34, best single compiler)
Three redesigns in one afternoon (v0 lanes + conveyor → ADR-0010 hub walk → v5 storage homes).
The v5 machine (`_StorageMachine`, `_compile_storage`):
- **Homes** packed in storage rows `10−R..9` (R = ⌈n/10⌉, all ten columns), chosen by several
  layout generators (local search on a cost model, block-pairing layout, boustrophedon longest
  path) and scheduling policies; the cheapest of a deterministic candidate budget wins.
- **Hub stream:** the hub stays in the zone, each partner rises straight from **row 9** into the
  hub's pair, pulses, returns; partners of one storage row are done as a batch (all cz #1, all `rx`
  in one layer, all cz #2); **row rotation** (row 9 → bottom, others up one: cost R+1) brings the
  next row to row 9. This is why systolic wins at 50q (R = 5): without rotation the deep rows would
  rise from far away.
- **Matching rounds** (non-hub blocks): lift the matched atoms from their homes (several placement
  candidates, incl. whole-row lifts), cz #1, targets to the nearest freed storage sites for `rx`
  (controls wait in the zone; no pulse fires meanwhile), back, cz #2, everyone home.
- `_State.pulse` asserts that the zone holds exactly the pulsed atoms (idle rule).
- Phase breakdown (50q stagger p3, D = 5385): matching rounds 79 % (≈ 77 per round in ~22 mostly
  single-atom moves), hub groups 21 %. See [ideas/systolic2.md](ideas/systolic2.md).

### 6.8 `optimizer-optimizer` (+81.79; formerly "optimizer" / `pulsebeam`)
Our fresh build ([ideas](../compilers/optimizer-optimizer/compiler.py) docstring):
- Front end: Pauli-rotation blocks, exact commutation DAG, executor with per-qubit pending
  unitary and roles (target in storage between its CZs; open frames let a control start several
  blocks, as the hub does).
- **BeamPlanner** (v5): atoms live in storage; one step = one pulse: zone atoms not in the next pulse
  (or owing 1q gates) drop, rising atoms flush their gates, pairs form (anchors stay; others get
  order-preserving zone sites by DP), CZs fire. A beam search over pulse candidates (critical-path
  and geometry-driven: storage neighbours rise together) and drop policies (home, next to the next
  partner, **block drop with a column offset**) keeps the best plans by distance + 3.06·depth.
- Line layout (longest path of the interaction graph, row by row).
- Wins most 12q iHVA circuits (e.g. tree p1 174 vs systolic 237) with lower depth; at 20q/50q it
  runs with beam width 1 and loses (no row rotation, no hub stream).
- Older planners kept in the file (`CyclePlanner`, `Compiler`) are the v0 zone-resident designs.

### 6.9 `systolic2` (+85.41)
Fork of systolic for the matching rounds: an order-preserving placement candidate, a block mid-drop
where the working row is free, narrower home rows for n ≤ 30. Small gains. The plan for the real
step (a staging-row choreography with moving homes and lookahead) is
[ideas/systolic2.md](ideas/systolic2.md).

### 6.10 `postopt` and the `*_po` components
[`compilers/postopt.py`](../compilers/postopt.py): between two CZ layers only the start positions,
the end positions and the 1q gates matter. Each gap is re-planned (drop the zone atoms that must
leave as a block, or the whole zone; the gates; grouped moves to the end positions) and kept only
if cheaper. Gates and their order are untouched (depth and equivalence unchanged); the result is
replayed with the validator's rules. `bench/make_postopt.py <name>` builds `compilers/<name>_po/`.
Effect: systolic 85.34 → 86.30, systolic2 85.41 → 86.35.

### 6.11 `portfolio` and `portfolio_po`
Generated by `bench/build_portfolio.py`: every component runs, each output goes through the
official replay, the transversal-CZ check and the zone-span check, and the cheapest valid result
per circuit wins (150 s budget guard). `portfolio` (official before tonight): systolic, hubstream,
rectbatch, gates: +85.50. **`portfolio_po`**: systolic_po, optimizer-optimizer_po, hubstream_po,
rectbatch_po, gates_po: **+86.63, full 33/33** (20 scored + 13 sanity), packaged as
`submission/portfolio_po.py`.

## 7. Current numbers (v5 set, 20 circuits)

Fresh `bench/bench.py` runs (fast mode, 2026-11-22 ~00:55 CET; `results/<name>-v5check.json` for the
older compilers):

| compiler | valid / 20 | v5 score | note |
|---|---|---|---|
| naive | 20 | 0.00 | reference |
| gates | 20 | +73.24 | |
| gatesb | 5 | – | 15 circuits violate the v5 rules |
| rectbatch | 20 | +79.83 | |
| hubstream | 12 | – | 8 `idle` violations (parked atoms); usable per circuit in a portfolio |
| zonedwell | 0 | – | v0 design (conveyor, dwelling) |
| hubwalk | 0 | – | v0 design |
| optimizer-optimizer | 20 | +81.79 | wins most 12q iHVA circuits |
| systolic | 20 | +85.34 | best single compiler |
| systolic2 | 20 | +85.41 | |
| systolic_po | 20 | +86.30 | |
| systolic2_po | 20 | +86.35 | |
| portfolio (systolic, hubstream, rectbatch, gates) | 20 | +85.50 | previous official |
| **portfolio_po** (all `*_po`) | **20 (+13 sanity, full mode)** | **+86.63** | **current best, `submission/portfolio_po.py`** |

## 8. The movement layer's input

[l3-input.md](l3-input.md): the exact pulse schedule and storage obligations every movement planner
has to realise (`bench/l3_input.py`), with a full 12q example and the scale at 50q.

## 9. Where the remaining points are

- 50q circuits are 61 % of the scored distance; there the matching rounds of systolic cost ~77 per
  round where ~5 is possible in principle. The plan is [ideas/systolic2.md](ideas/systolic2.md)
  (M2 first: the free-sorting bound tells within an hour whether it is worth it).
- 12q: optimizer-optimizer already wins most circuits; more beam width is cheap there.
- Depth: ~2600 vs a hub-chain floor near ~2000 on the scored set; secondary.
