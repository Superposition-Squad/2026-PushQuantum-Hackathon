# Literature review: techniques transferable to our metric

Scope: zoned neutral-atom compilation papers in the local `papers` corpus. **Caveat:** the corpus
has abstracts only (no full text, no method digests) for almost every compiler paper below. Claims
about a paper's technique come from its abstract. Everything about *our* grid, costs and
estimates was worked out by us from `docs/scoring.md`, the validator and the QASM files.

## 1. Framing: how our objective differs from the papers'

The papers (ZAC, ZAP, PowerMove, Enola, NEAT, routing-aware placement) minimise **execution time
and fidelity loss**: AOD stroke time, number of rearrangement steps, atom-transfer loss, idle
decoherence and Rydberg crosstalk on idle atoms. Their main levers, parallel moves and fewer
sequential steps, are worth **nothing** here. Our score counts:

- 80 %: **Σ over `move` instructions of the max distance any atom in that move travels**. A move
  that carries 30 atoms 1.0 costs the same as one that carries a single atom 1.0. Two parallel moves
  cost twice as much.
- 20 %: plain qiskit `depth()` without moves.

There is no atom loss, no time cost and no crosstalk. Spacing warnings are free. The physics
constraint is ADR-0007: every occupied interaction pair gets the same number of CZ between moves,
and a lone atom on a pair is fine. Two hard validity facts the papers don't share with us:

- **(a)** The compiled circuit must have **exactly the same number of qubits** as the input
  (`verify.py:187`, and the loading must cover every qubit). So **no ancillas**, and no "flying
  ancilla" (Q-Pilot) or fan-out-via-ancilla tricks.
- **(b)** For more than 19 qubits, every CZ must act on a pair that interacts in the reference. So
  parity networks may only use CNOTs on graph edges.

Geometry facts that drive everything:

- Storage row 9 → entangling row 10 costs exactly 1.0 for an atom in an even storage column
  (x = 2k stays x = 2k). An atom in an odd column costs √(1+0.49) = 1.22, because the right slot is
  at x = 2k+0.3.
- Lifting k adjacent storage rows together costs ≈ k, the same as the farthest row. Lifting them
  one by one costs 1 + 3 + 5 + …
- The degree-(n−1) reference qubit forces **≥ 2(n−1) sequential CZs per layer** (98 for 50q) with
  the 2-CZ ZZ decomposition. That fixes the number of zone round trips, so the winning lever is to
  make each round trip cost ≈ 1.0 per direction and carry everything else for free.

Naive cost for scale: 50q p3 iHVA/QAOA = 24 907 distance, 2 988 moves (≈ 8.3 per move), depth
1 933–2 293.

## 2. Papers

| arXiv id | title | key technique (from abstract) | transferable? |
|---|---|---|---|
| 2411.11784 | Reuse-Aware Compilation for Zoned Quantum Architectures (ZAC) | Keep qubits in the entangling zone if their next gate is a CZ. Placement plus scheduling, ZAIR IR | **yes**: reuse saves whole round trips. Must respect ADR-0007 (parked atoms alone on a pair) |
| 2411.14037 | ZAP: Zoned Architecture and Performant Compiler for FPAA | ASAP-separate scheduling, look-ahead placement, conflict-aware routing, single pass | **partly**: look-ahead placement yes. Conflict/crosstalk routing is irrelevant to us |
| 2411.12263 | PowerMove | Joint scheduling, qubit allocation, movement and inter-zone communication. Uses the storage zone actively | **partly**: co-design of scheduling and placement yes. Its time objective no |
| 2405.15095 | Enola (DPQA compilation, near-optimal scheduling) | **Edge-colouring** scheduling, provably ≤ optimal+1 two-qubit stages. Placement and routing split | **yes**: edge colouring of commuting ZZ layers (Vizing/Misra–Gries ≤ Δ+1) |
| 2306.03487 | Compiling for DPQA (OLSQ-DPQA) | Discretised state space, solver-based layout synthesis with 2D array moves | **partly**: exact solver is too slow for 50q in 60 s. Useful for 12q only |
| 2505.22715 | Routing-Aware Placement for Zoned NA (MQT) | A*-based placement that groups compatible moves into parallel steps | **yes**: "group compatible moves" equals our single-rectangle batching. Here batching saves distance, not time |
| 2512.13790 | Search Smarter, Not Harder (IDS, relaxed routing) | Goal-directed search for scalable placement. Relaxed routing cuts rearrangement overhead 28 % | **partly**: search idea for placement. Scale is far beyond ours |
| 2611.19994 | NEAT: Efficient AOD-Transport Compilation | Transport-informed placement with an incremental cost per decision. Partial-progress routing via waypoints | **partly**: incremental-cost greedy placement transfers if the cost is our max-distance-per-move |
| 2509.12083 | HiPARS | Highly parallel **composite moves**: many atoms picked up at once and carried far | **yes** in spirit: one big move beats many small ones under our metric |
| 2405.06183 | Arctic | Mapping as **max-cut plus layered cross-minimisation**, "stacking" of rows | **yes**: cross-minimisation equals ordering atoms so that monotone (order-preserving) AOD maps work |
| 2311.15123 | Atomique | MAX k-Cut on a gate-frequency graph to assign qubits to arrays, then parallel-gate router | **partly**: partition idea for SLM/AOD split and row assignment |
| 2409.04578 | Parallax | Zero-SWAP, movement-only scheduling under AOD constraints | **partly**: confirms movement-only is enough. Not zoned |
| 2409.03185 | DasAtom | Split the circuit into subcircuits, each executable under one mapping, then shuttle between mappings | **partly**: "segment, then transition" matches our layer → rearrange → layer loop |
| 2503.02272 | Mantra | Rewrite programs to **reduce interleaving of 1q and 2q gates**: fountain-shaped CZ chain, ZZ protocol without a 1q gate, preemptive scheduling. −68 % inter-zone moves | **yes** (the CZ-chain and rewriting idea). Their "ZZ without 1q gate" needs a native gate we can't use (we only have fixed CZ, ADR-0002). Mechanism unverified |
| 2607.01787 | Structure-Aware Compilation for NA | Graph-theoretic CZ transport scheduling. Rank-one decomposition of 1q gate families into addressing layers | **partly**: CZ transport scheduling model |
| 2411.05061 | Optimal Routing Protocols for Reconfigurable Atom Arrays | Θ(√N log N) parallel-step bounds for permutations on 2D arrays | **no** (step count, not distance). Gives intuition that row/column sweeps realise permutations |
| 2607.28149 | Atom rearrangements with a single AOD | Shear, rotate and reflect whole 2D arrays in O(log) strokes via binary decompositions | **partly**: whole-array shears with one rectangle are cheap per atom under our max metric |
| 2605.29964 | Hub-assisted shuttling compiler | Empty "hub traps" as transit waypoints, plus a per-gate SWAP-vs-shuttle rule | **no**: monolithic, and its "hub" means traps, not a high-degree qubit |
| 2608.01316 | Piqasso (3D NA compiler) | Short vertical hops between stacked zone layers. 2.1× less transport distance | **partly**: same principle as our row 9 ↔ row 10 hop (shortest crossing) |
| 2311.16190 | Q-Pilot (flying ancillas) | Data qubits fixed, mobile ancillas route 2q gates | **no**: ancillas break the qubit-count check |
| oa:W3092255708 | Efficient Circuit Compilation Flow for QAOA | Reorder commuting ZZ gates. Up to −53 % depth | **yes**: commutation reordering for depth |
| oa:W3102896149 | Circuit Compilation Methodologies for QAOA | Four reordering methodologies for commuting CPHASE | **yes**: same as above |
| 2104.00934 | Phase polynomial synthesis for NISQ | Parity-network synthesis with lower CNOT count and depth, constrained connectivity | **partly**: only with CNOTs restricted to graph edges (fact (b)). Small gains |
| 2307.14996 | Decompositions with limited local addressability | Synthesis around global xy rotations | **no**: we may address rx locally |
| 2312.03982 | Bluvstein logical processor | Zoned storage/entangling/readout. Row and column AOD moves of whole blocks, transversal CZ | **yes** as the physical model: transversal pulse (ADR-0007), whole-block moves |
| 2408.09083 | iHVA (Wang et al.) | Tree arrangement of ZY-type gates. Defines the iHVA benchmarks | **partly**: tells us gate order matters (ZY terms don't all commute) |

## 3. Candidate techniques

Notation: *round* = one CZ segment between two moves. *Phase transition* = storage ↔ zone crossing.
Costs use the facts in §1.

### 1. Native resynthesis and 1q merging (depth)
- **Idea:** Rewrite `cx(c,t)` as `H_t · cz · H_t`. Cancel adjacent H·H. Push `rz` through `cz`
  (both are diagonal) and merge every 1q run into ≤ 3 gates `rz·rx·rz` (qiskit
  `Optimize1qGatesDecomposition(basis=["rz","rx"])`, or `transpile(..., optimization_level=2/3)`
  followed by our own scheduler). The naive compiler uses `optimization_level=0`, so every `h`
  expands to 3 gates and every `cx` adds two serial H chains.
- **Impact:** about −40 to −60 % depth on its own. It also removes 1q gates, and every removed 1q
  gate between two CZs on the same atom saves a storage round trip (see #4).
- **Difficulty:** S.
- **Risks:** none for equivalence if done with exact qiskit passes (check with QCEC at ≤ 19q). Keep
  the qubit count identical. Don't let a transpiler layout or routing pass reorder qubits.
- **Sources:** oa:W3092255708, 2503.02272.

### 2. ZZ orientation: the hub is always the control
- **Idea:** `cx(u,v)·rz_v·cx(u,v)` equals `cx(v,u)·rz_u·cx(v,u)`. Pick the reference qubit `a` as
  control. In CZ form it then sees only CZs, with no 1q gate between its 2(n−1) CZs. All its blocks
  commute on `a` (all diagonal), so `a` can **park in the entangling zone for the whole cost layer**.
  For iHVA (`rx(π/2)_t · ZZ · rx(−π/2)_t`): when `a` is the Y-side, consecutive commuting hub
  blocks give `rx(−π/2)_a·rx(π/2)_a = I`. Group the hub blocks so that `a` gets one rx before the
  run and one after, and choose the ZZ core's CX orientation so that `a` is the control.
- **Impact:** removes about 2(n−1) hub round trips per layer. That is the main D term of the 18
  iHVA/QAOA 50q/20q circuits.
- **Difficulty:** S.
- **Risks:** for iHVA, grouping must respect commutation (#3). ADR-0007: the parked hub has to be
  alone on its pair whenever another pair is pulsed without its partner present.
- **Sources:** 2411.11784 (reuse), 2503.02272.

### 3. Commutation-aware layering and edge colouring
- **QAOA:** all ZZ terms in a cost layer commute. Edge-colour the graph (Misra–Gries, ≤ Δ+1
  colours, ≈ Enola) for non-hub edges. The hub imposes 2(n−1) ≥ 98 rounds anyway, and the
  non-hub graph has max degree ~6, so **every non-hub edge can ride along in hub rounds at zero
  extra phase transitions**. Pack them into the same rectangles.
- **iHVA:** build a DAG with the exact Pauli commutation rule. Terms `Z_uY_v` and `Z_wY_x` commute
  iff they anticommute on an even number of shared qubits. Then list-schedule maximal matchings
  (stagger/tree differ only in initial order).
- **HEA:** fixed 2 CZ layers per layer (even, then odd).
- **Impact:** depth down to roughly the hub bound, about 100–200 per layer (1q layers interleave)
  vs ~700 per layer naive at 50q. Rounds drop to about 2Δ.
- **Difficulty:** S/M.
- **Risks:** wrong commutation for iHVA silently breaks equivalence above 19q. Unit-test the
  commutation predicate and check 12q with QCEC.
- **Sources:** 2405.15095, oa:W3092255708, oa:W3102896149, 2408.09083.

### 4. Target chaining: back-to-back CZs without a storage trip
- **Idea:** for target `v` with consecutive ZZ blocks to `u1`, `u2`, the sequence
  `H CZ(u1) Rx CZ(u1) H · H CZ(u2) Rx CZ(u2) H` becomes `H CZ(u1) Rx CZ(u1)·CZ(u2) Rx CZ(u2) H`.
  `CZ(u1,v)` and `CZ(u2,v)` are adjacent, so `v` stays in the zone and is only re-paired by a short
  in-zone hop (0.3 to 2.0). Storage trips per target drop from 2·deg to deg+1. This is the
  "reduce 1q/2q interleaving" or "CZ chain" idea.
- **Impact:** ~30–45 % fewer phase transitions for non-hub atoms. The gain is smaller once #2
  makes hub rounds the bottleneck. It matters more for iHVA tree, where ordering limits batching.
- **Difficulty:** M.
- **Risks:** an in-zone re-pair move must not drop an atom onto an occupied site, and transversal
  counts must stay equal.
- **Sources:** 2503.02272, 2411.11784.

### 5. Hub parking plus partner streaming at cost 1.0
- **Idea:** park `a` at entangling (row 10, col 2k+1), physical x = 2k+0.3. Put its partners in
  storage **column 2k** (x = 2k), staged in row 9, so that lifting a partner to (10, 2k) is purely
  vertical, distance **1.0** each way. Each hub round has two moves: lift (the partner plus the
  ride-along pairs of #3 in the same rectangle), then lower. Estimate for a 50q layer:
  98 rounds × 2 × ~1.1 ≈ 220, plus restaging. For 50q p3 that is ≈ 700–1 300 vs 24 907, about
  95–97 % on the biggest circuits.
- **Difficulty:** M.
- **Risks:** restaging partners into row 9 / column 2k costs distance (#6, #7). ADR-0007: when the
  hub's partner is absent, the hub must be alone. At 50q the hub fits on one pair. 12q/20q use the
  same scheme.
- **Sources:** 2411.11784, 2608.01316 (short vertical hops).

### 6. Single-rectangle batching: one move per phase transition
- **Idea:** a move is (monotone row map) × (monotone col map) on a contiguous rectangle. Every
  atom inside moves, empty intersections are fine, and the cost is the max displacement. Per round,
  put all lifting atoms in **one** rectangle. Use row 9 as the staging row, and keep non-participants
  outside the rectangle's column span or in other rows. Lift 1 row: 1.0–1.22 for up to 5 pairs.
  Lift k rows: ≈ √(k² + 0.49). Never lift rows 8 and 9 in separate moves (1.22 + 3.08 vs 2.12
  jointly). Return moves mirror lift moves. Inside the zone, re-pairing a whole row with one
  horizontal shift costs 2.0 for up to 5 pairs.
- **Implementation:** a greedy cover of the round's atoms by the fewest legal rectangles (check
  monotonicity, ghosts and occupied targets), minimising Σ max-distance.
- **Impact:** without it, rounds × #rectangles multiplies D. This is the core of any compiler here.
- **Difficulty:** M.
- **Risks:** ghosts and deposits onto occupied sites. Use the validator's replay as the oracle in
  tests.
- **Sources:** 2505.22715, 2509.12083, 2405.15095, 2312.03982.

### 7. Initial placement as an optimisation problem
- **Idea:** loading is free, so place qubits to minimise future restaging:
  - The hub's partners go in even columns near row 9, in the order the schedule needs them.
  - Edges that ride along in the same round go in adjacent storage columns (2k, 2k+1) of the same
    row, so one row lift forms the pairs (cost 1.22).
  - Vertices of a cluster go in one row segment (the brief: four triangle clusters, "entangle
    within clusters first").
  - Order atoms within rows to minimise crossings, so that monotone AOD maps exist (Arctic's
    cross-minimisation).
- **Method:** start greedy or with spectral ordering, then use simulated annealing or local swaps
  scored by **actually running the move planner** (#6) and summing D. Compile time permits
  thousands of evaluations at 50q if the planner is fast.
- **Impact:** large secondary term (restaging is probably 30–60 % of the post-#5 distance).
- **Difficulty:** M/L.
- **Risks:** compile time limit (aim < 60 s). No validity risk.
- **Sources:** 2405.06183, 2311.15123, 2411.11784, 2611.19994, 2512.13790.

### 8. Batch staging and hub hopping (advanced, beats 1.0 per hub round)
- **Idea:** instead of lifting each partner separately, lift a whole batch of partners in one move,
  e.g. storage rows 8–9 even columns → entangling rows 10–11 left slots, cost 2.0 for 10 partners,
  each alone on its pair (ADR-0007 OK). Then **move only the hub** along the right slots: one move
  per round, 1.0 vertical or 2.0 horizontal. After each partner's first CZ, the batch returns in one
  move for its Rx and comes back in one move for the second CZs.
- **Impact:** per-round cost ≈ 1.0–1.5 plus amortised batch moves, vs ≈ 2.0–2.4 in #5. Roughly
  −20 to −35 % on the hub-dominated D. The schedule ordering gets more complex.
- **Difficulty:** L.
- **Risks:** ride-along non-hub pairs become harder to batch. Transversal counts with many parked
  atoms.
- **Sources:** 2509.12083, 2607.28149 (whole-array strokes), 2411.11784.

### 9. Fewer phase transitions per layer: merge the last and first rounds across layers
- **Idea:** mixers (`rx` on all qubits) and the next layer's first H/1q gates merge (#1) into one
  storage phase. The final CZ round of layer ℓ and the first of layer ℓ+1 need a storage phase in
  between anyway, so schedule them so that the same atoms sit in row 9. Also start layer ℓ+1's hub
  run in the reverse partner order (snake order). Then the restaging done at the end of layer ℓ is
  reused, and an atom just lowered is lifted again from the same spot.
- **Impact:** removes restaging between layers. Probably 5–15 % of the remaining D at p = 3.
- **Difficulty:** S once #5/#6 exist.
- **Risks:** none.
- **Sources:** 2409.03185 (segment-transition view), 2411.14037 (look-ahead).

### 10. HEA special case: two-row ladder
- **Idea:** linear chain. Put qubits 0..n−1 snake-wise into storage rows 9 (and 8, 7…) so that even
  pairs (2k, 2k+1) share a row at adjacent columns: one lift per row (1.22) gives all even CZs.
  For odd pairs (2k+1, 2k+2), shift the row's odd-indexed atoms by one pair pitch inside the zone
  with one move (≈ 2.0), or lift a second, offset copy of the layout. 1q layers in between happen in
  storage.
- **Impact:** HEA is ~2.2 % of D_naive in total, so it is worth little. Its depth share is also
  small. Do it cheaply, or let the generic planner handle it.
- **Difficulty:** S.
- **Risks:** row boundaries of the snake need a separate pair. Odd n.
- **Sources:** 2312.03982, 2405.06183.

### 11. Edge-restricted phase-polynomial resynthesis (optional, small)
- **Idea:** inside a triangle (u, v, w), all three edges exist, so the three ZZ terms can be made
  with 5 CNOTs instead of 6:
  `CX(u,v)` [rz on v = u⊕v], `CX(w,u)` [rz on u = u⊕w], `CX(u,v)` [rz on v = v⊕w], `CX(w,u)`,
  `CX(w,v)`.
  Each saved CZ is also a saved round slot.
- **Impact:** small (≤ 1 CZ per triangle per layer). Possibly a few % depth on 12q.
- **Difficulty:** M.
- **Risks:** must only use edge pairs (fact (b)). More 1q gates on CZ paths can *add* phase
  transitions. Verify with QCEC.
- **Sources:** 2104.00934.

### Ruled out (validity)
- Ancilla copies of the hub, log-depth fan-out via ancillas, flying ancillas (2311.16190): these
  break the equal-qubit-count check.
- Fan-out through non-neighbour data qubits creates pairs outside the reference set.
- Native `rzz` (ADR-0002). Parking two atoms on one pair without matching CZ counts (ADR-0007).

## 4. Suggested first parallel round

1. **Gate layer: #1, #2 and #3 as one shared module.** Every compiler needs exact resynthesis,
   hub-as-control orientation and a commutation-correct round schedule. It takes the depth score
   from ~0 to roughly −60 to −75 % and fixes the round count every movement planner works from.
2. **Hub-streaming planner: #5 plus a simple #6, with a naive placement.** This is the biggest
   single distance lever (≈ 95 % on the 50q iHVA/QAOA circuits, which carry most of D_naive) and is
   achievable in one sprint.
3. **Rectangle-cover move planner: #6 done properly, validated against the validator replay.**
   It is the reusable core (no ghosts, monotone maps, minimal Σ max-distance), and #5, #7, #8 and
   #9 all build on it.
4. **Placement optimiser: #7, annealing scored by the real planner.** It attacks the restaging
   distance left after #5. It plugs in independently once a planner exposes a cost function.
