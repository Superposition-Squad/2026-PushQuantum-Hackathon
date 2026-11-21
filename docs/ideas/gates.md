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

## Results

| iteration | commit | score | dist % | depth % | notes |
|---|---|---|---|---|---|

## Verdict

(pending)
