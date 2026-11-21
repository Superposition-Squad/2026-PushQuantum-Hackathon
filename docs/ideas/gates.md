# Idea: gates — gate-level resynthesis + commutation-aware round scheduling

- Branch: idea/gates  ·  Worktree: /home/hackathon/challenge-wt/gates
- Base: main 0ea558b, starting from `compilers/naive/compiler.py`
- Status: in progress

## Hypothesis

The naive compiler transpiles with `optimization_level=0` (every `h` becomes 3 gates, every `cx`
becomes `H·cz·H` with no cancellation) and then carries the two atoms of **every single `cz`** to the
entangling zone and back, one atom per `Move`, from a loading that puts the 50-qubit circuits in
storage rows 0–4 (up to 10.0 away from the zone).

Two independent levers:

1. **Depth (20 % of the score).** Depth is measured on the compiled circuit *with moves stripped*,
   so it is exactly the depth of the gate-level circuit. Exact resynthesis (`cx → H·cz·H`, H·H
   cancellation, merging every 1q run into ≤ 3 `rz/rx` gates) plus putting the inter-CZ rotation of
   every two-qubit Pauli rotation on the **low-degree** side (so the degree-(n−1) reference qubit
   sees only CZs) should take depth close to its floor of `2·deg(hub)·p` per circuit.
   Estimate: −60 to −75 %.
2. **Distance (80 %).** A *round* = a set of disjoint CZ pairs executed between two moves. The
   round count is a gate-level property: with commutation-aware scheduling it drops to ≈ 2·Δ·p
   (Δ = max degree = n−1 for the hub families), which is the lower bound for the 2-CZ ZZ
   decomposition. Fewer rounds ⇒ fewer trips. On top of that, a compact loading (storage rows
   nearest the zone) and a per-round site assignment that minimises the Euclidean cost of each trip
   cut the per-move distance from ~8.3 (naive) to ~2.5–3.5. Estimate: −50 to −65 %.

Target score: 0.8·55 + 0.2·70 ≈ 58 %.

## Design

### Stage 1 — lowering to items (exact)

Walk the input circuit once with per-qubit "next op" tables and produce a flat list of *items*:

- `Op1(q, U)` — a 1q gate as its exact 2×2 matrix.
- `OpCZ(u, v)`.
- `Rot2(u, v, θ, Pu, Pv)` — `exp(-i θ/2 · P_u ⊗ P_v)`, `P ∈ {X, Y, Z}`. Recognised from
  - the **ZZ block** `cx(c,t); rz(θ) t; cx(c,t)` (QAOA cost terms, iHVA cores — the iHVA
    `rx(±π/2)` wrappers are left as ordinary `Op1`s on the Y-side qubit and merge away later),
  - native `rzz/rxx/ryy` (sanity circuits).
- `OpMeas(q, c)`. Barriers are dropped (no semantics).
- Anything else (`ccx`, `cswap`, `cu3`, …) is lowered by `transpile(basis=[rx,rz,cz], level=2)` on a
  k-qubit scratch circuit, cached by `(name, params)`. Exact by construction.

### Stage 2 — Rot2 expansion with a choice of mid side (the "hub as control" move)

With `C_P Z C_P† = P`,

    exp(-iθ/2 P_u⊗P_v) = (C_Pu⊗C_Pv) · exp(-iθ/2 Z⊗Z) · (C_Pu⊗C_Pv)†
    exp(-iθ/2 Z⊗Z)     = H_m · CZ · rx(θ)_m · CZ · H_m     for either m ∈ {u, v}

so the inter-CZ `rx(θ)` can sit on **either** qubit. We put it on the qubit with the lower
two-qubit degree, i.e. never on the reference/hub qubit. Consequence for QAOA: with `P = Z` on the
hub, `C_Z = I`, so the hub's entire cost layer is a bare chain of 2·deg(hub) CZs with no 1q gate in
between — all of them mutually commuting. (Literature #2/#11, arXiv:2503.02272, 2411.11784.)
The 4×4 matrix of the expansion is checked against `expm(-iθ/2 P⊗P)` in a unit test.

### Stage 3 — per-qubit 1q run merge (pre-scheduling)

For each qubit, merge maximal runs of consecutive `Op1`s on that wire into one matrix and drop
identities. This kills the iHVA `rx(π/2)·rx(−π/2)` pairs between consecutive hub blocks and the
`H·H` of back-to-back `cx`s, which is what frees the hub's CZ chain to commute.

### Stage 4 — commutation DAG

Each item carries a per-qubit *basis* `b(q) ∈ {X, Y, Z, None}`: for `Op1` the Pauli it commutes with
(`None` if none), for `OpCZ` `Z` on both (diagonal), for a leftover `Rot2` its `P`. Soundness: every
item's generator is a sum of Pauli strings `S` with `S(q) ∈ {I, b(q)}`, so two items whose bases
agree on every shared qubit anticommute on zero qubits and therefore commute exactly.
Dependencies are built with the standard commutation-set construction (as in qiskit's
`CommutationAnalysis`): per qubit, maintain the current group of mutually commuting items; a new
item joins the group if its basis matches, otherwise it starts a new group and depends on every
item of the previous group.

### Stage 5 — `schedule_rounds(items) -> list[Phase]` (the separable artifact)

ASAP list scheduling over the commutation DAG, alternating:

- **`Phase1Q`**: repeatedly emit every ready 1q item (and measurements) until none is ready.
- **`PhaseCZ`**: the ready CZ items form a candidate set; pick a *maximal matching* greedily by
  descending criticality (longest remaining CZ-path to the end), which prioritises the hub and
  recovers edge-colouring behaviour on commuting layers (arXiv:2405.15095). Emit it as one round.

Rounds are therefore ≈ 2Δ per cost layer. Output is a clean list of phases that a movement planner
consumes; nothing in it knows about geometry.

### Stage 6 — emission and final 1q merge

Per `Phase1Q` and qubit, the gates of that phase are merged into ≤ 3 gates with the ZXZ Euler
decomposition (`rz·rx·rz`, zero rotations dropped), accumulating `global_phase`. Equivalence is up
to global phase, but we track it anyway.

### Stage 7 — movement (deliberately simple, batched where free)

- **Loading:** fill storage rows nearest the entangling zone first (row 9, then 8, …), 10 per row,
  so the vertical trip is 1–5 instead of the naive 6–10.
- **Per round:** assign each pair a free interaction site, preferring zone row 10 upwards and the
  column nearest the pair's storage columns, in storage-column order so that monotone column maps
  exist. Lift (single-atom `Move`s, which can never have ghosts), emit one `cz` per pair, lower.
  Every occupied site holds 2 atoms and gets exactly one `cz` ⇒ ADR-0007 holds trivially.
- **Greedy move merging:** after building a round's move list, repeatedly merge two moves into one
  rectangle `Move` when it is legal (same picked-up row set, contiguous picked-up columns with no
  non-participating atom inside, ascending targets, targets free). The cost of a merged move is the
  max displacement, not the sum — this is where distance is actually saved.

### Validity

- Gate set `rx/rz/cz` only (ADR 0002); no ancillas, qubit count unchanged.
- Every rewrite is exact algebra (Pauli conjugation, diagonal commutation, Euler decomposition), so
  equivalence holds at every size (ADR 0003). Verified with QCEC `--mode full` on the ≤ 19q
  circuits and `--mode strict` on the whole metric set.
- Transversal CZ (ADR 0007): within a round all occupied sites carry exactly one `cz`.
- Single file, stdlib + qiskit + numpy only (ADR 0006).

### Stage 8 — in-zone retention (added after the first measurement, ZAC-style idle reuse)

`schedule_rounds` knows exactly which atoms get a 1q gate between two rounds, so it also knows which
atoms *don't*: those may stay in the entangling zone instead of making a pointless round trip
(arXiv:2411.11784). Measured potential on the 36 metric circuits: 7 965 of 18 712 atom-trips
(35 % of the raw distance). Rules that keep it legal:

- an atom may only stay if it takes part in the **next** round and has no 1q gate in between;
- at most one atom per interaction pair stays, so a staying atom always has its partner slot free;
- a site may only be claimed if both its slots are empty or already hold one of that pair's atoms,
  which makes every move of a round collision-free and emittable in any order;
- a rectangle may not span a site that another move of the same batch deposits into (otherwise that
  atom is a ghost by the time the move executes);
- a staying atom alone on its pair receives no `cz`, which ADR 0007 allows (only *doubly* occupied
  pairs must get equal pulse counts).

Two refinements that mattered:

- **lookahead site choice:** a staying atom pins its pair's site for many rounds, so the site cost
  adds the mean distance from the homes of its next (up to 16) partners to the opposite slot. The
  hub therefore parks in a *central* column instead of wherever its first partner happened to be.
- **merge-aware cost model:** lifts out of storage can be merged into one rectangle (cost = max of
  the trips) while an atom already in the zone needs a move of its own (cost adds up). Scoring a
  site with `max(lifts) + sum(in-zone moves)` instead of `max(all)` was worth +1.6 points on its
  own, because the naive `max` made in-zone relocation look free.

Gap parking (staying through rounds the atom does not take part in) was measured and dropped: only
549 of 8 514 opportunities have a gap > 1.

## Results

All runs: `bench/bench.py compilers/gates/compiler.py --jobs 2`, 36 official metric circuits.

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|
| 1 gate layer + per-pair moves | 664bd4b | **+63.86** | +60.45 | +77.47 | resynthesis, hub-as-control, round schedule, compact loading |
| 2 placement | 3c9ba94 | **+64.66** | +61.46 | +77.47 | degree-weighted snake loading, both slot orientations |
| 3 synthesis fix | 54bc0dd | +64.66 | +61.46 | +77.47 | `optimization_level=2` had elided `swap` permutations (real equivalence bug) |
| 4 retention | 43791f1 | **+71.53** | +70.05 | +77.47 | atoms with no 1q gate in between stay in the zone |
| 5 robustness | e0f12d8 | +71.54 | +70.05 | +77.47 | per-orientation target check |
| 6 lookahead sites | 1af2060 | **+72.06** | +70.70 | +77.47 | central parking column for sticky atoms |
| 7 relocation + cost model | 3e464c8 | **+72.87** | +71.72 | +77.47 | in-zone re-pairing, merge-aware site cost |

Per family (iteration 7): hea distance +55.86 / depth +35.48, ihva +71.86 / +78.63,
qaoa +72.40 / +76.72.

Validity: 36/36 metric and 13/13 sanity circuits pass `--mode full`; `--mode strict` on the metric
set passes as well (statevector x3 for <= 22 qubits, QCEC for the 50-qubit circuits). Compile time
<= 1.6 s per metric circuit, 40 s for the 70-qubit sanity circuit.

### Where the remaining distance sits

- **Depth is essentially at its floor.** For a cost layer the reference qubit needs 2 CZs per
  incident edge and the partner needs its mid rotation between them, so the critical path is
  `3 * deg(hub) * p`: 441 for the 50q QAOA p=3 circuit, where we emit 459.
- **Rounds are at their floor too:** 294-302 rounds for the 50q p=3 circuits vs the bound
  `2 * deg(hub) * p = 294`.
- **Batching is structurally blocked on the big circuits.** 50 atoms fill storage rows 5-9
  completely, and a hub-dominated round has only ~3 pairs (6 of 50 atoms), so almost every
  rectangle that would carry several atoms contains a non-participant (a ghost). 2 129 of 2 761
  moves carry a single atom. Making rounds fatter cannot help: the hub serialises
  `2 * deg(hub)` rounds, so the average round *has* to be thin.
- What is left is therefore the per-trip distance of the partners (storage rows 5-9, mean 3.0 from
  the zone) and the HEA family, where a proper rectangle cover should lift a whole storage row of
  pairs in one move (HEA is only 2.2 % of `D_naive`, so ~1 point).

## Verdict

**Merge the gate layer.** Stages 1-6 (`lower_to_items`, `expand_rot2`, `merge_1q_runs`,
`build_commutation_dag`, `schedule_rounds`, `euler_zxz`) are a self-contained, verified front end
that any movement planner can consume; they deliver the depth score (+77.5 %, near the floor) and
the minimal number of CZ rounds, and they are what every other branch needs. The retention logic
(stage 8) overlaps with the `hubstream` branch by construction — whichever movement layer wins
should be put on top of this gate layer.

Carry over to other ideas:

- `schedule_rounds` returns alternating `("1q", items)` / `("cz", items)` phases; the pairs of a
  `"cz"` phase are always disjoint, so a round is executable between two moves.
- the derived tables `blocked_between[r]` (qubits with a 1q gate between round r and r+1) and
  `partner_in[r]` are what retention/parking planners need, and they are cheap to recompute.
- put the inter-CZ rotation of every two-qubit Pauli rotation on the **low-degree** side: it is
  exact, it shortens the critical path, and it is what makes hub parking possible at all.
- never use `transpile(..., optimization_level >= 2)` on a sub-circuit and read its `data`:
  `ElidePermutations` turns a `swap` into a layout change and the gate silently disappears.
