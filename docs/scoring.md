# Scoring and validation (what actually counts)

> **Validator v5.0.0 (final, ADR 0011):** only 20 circuits are scored (18 iHVA + 12q HEA p1/p3;
> naive totals D = 163 364.3, depth = 13 364), the `idle` rule applies (every atom in the entangling
> zone must be in a two-qubit gate of each gate layer), moves pick up / drop off in one zone, and the
> entangling zone allows only `cz`. Sections below describe the original v0.0.0 scoring where they differ.

Source of truth: `vendor/pushquantum_validator/` (copied from the public workspace
`planqc/pushquantum-2026-leaderboard` v0.0.0) and its scoring notebook
`planqc-pushquantum-2026/leaderboard/readme.py`. Where the prose brief and the validator disagree,
the validator decides the score, but our own rules (ADRs) may be stricter.

## Score

36 metric circuits (`vendor/pushquantum_validator/metric_benchmarks/`): HEA ×6, iHVA ×18
(stagger / tree / tree_warm), QAOA ×12 (plain / warm), each at 12, 20 and 50 qubits, p = 1 and 3.

    score = 0.8 · (D_naive − D) / D_naive · 100  +  0.2 · (P_naive − P) / P_naive · 100

- `D` = Σ over all circuits, Σ over all `move` instructions, of the **longest physical distance any
  atom of that move travels**. Totals of the naive compiler: `D_naive = 276 726.91`.
- `P` = Σ over all circuits of `QuantumCircuit.depth()` of the compiled circuit **with moves and
  barriers removed** (measurements included). `P_naive = 23 676`.
- Higher is better; the naive compiler scores 0 %. Only the best submission counts.
- Distance dominates (80 %). Note the size skew: the 50-qubit p=3 circuits alone are ~9 % of
  `D_naive` each (24 907), the 12-qubit HEA p=1 is 0.09 % (244).
- Move parallelism does **not** reduce `D` (it reduces `move_time`, which is not scored). What
  reduces `D` is carrying many atoms in **one** `Move` and moving them a short way.
- Spacing warnings (non-proportional line spacing, or mixing first/second pair sites with 3+
  columns) double a move's *time* only. They don't affect the score.

## Validity (any failure on any circuit rejects the submission)

1. `compile_circuit(circuit, spec)` runs in a child process (`python -I`, no network, 300 s CPU and
   wall clock, 4 GiB address space) and must return `(QuantumCircuit, obj)` where
   `obj.initial_loading` is `{qubit: (x, y)}`.
2. Movement replay: the loading is within storage and has no duplicate sites. Every `move` is legal
   (see below). Every gate is in a zone that allows it. `cz` acts only on an interaction pair.
3. Equivalence (moves and barriers stripped, final measurements removed):
   - ≤ 19 qubits: QCEC (ZX + simulation, then the exact alternating checker), up to global phase.
   - > 19 qubits: only the *set of interacting qubit pairs* is compared. It must contain every pair
     of `transpile(reference, ["rx","rz","cz"], optimization_level=1)` and nothing outside the
     reference's pairs. **We do not rely on this gap** (ADR 0003). Our harness runs extra checks.

## Architecture (`architectures/default.json`)

| zone | x (cols) | y (rows) | ops | geometry |
|---|---|---|---|---|
| storage | 0–9 | 0–9 | `rx`, `rz` | physical (x, y) = (col, row), spacing 1 |
| entangling | 0–9 | 10–19 | `cz` (and `rzz`, which we don't use: ADR 0002) | physical y = row. Physical x = 2·(col//2) + 0.3·(col%2) |

- Interaction pairs: same row, columns (2k, 2k+1), physically 0.3 apart. Pairs repeat every 2.0.
- Storage row 9 and entangling row 10 are 1.0 apart: the cheapest zone crossing.
- `measure` and `barrier` are allowed anywhere. All single-qubit gates (`rx`, `rz`) need the atom in
  **storage**, even though the brief's Fig. 1 describes a global Rx on both zones.
- Capacity: 100 storage sites, 50 interaction pairs (100 sites). The 50-qubit circuits fill half of
  storage.

## Move rules (`Move(rows, cols, new_rows, new_cols, num_qubits)`, rows = y, cols = x)

- Picked-up rows and cols must be contiguous and ascending. Targets must be ascending (order is
  preserved), with no more target lines than picked-up lines.
- Every atom on a picked-up row×col intersection moves, and they must be exactly the listed qubits.
  Unlisted atoms ("ghosts") make the move illegal.
- If there are fewer target lines than picked-up lines, empty picked-up lines are dropped. The rest
  must map one-to-one.
- Illegal: depositing on a site occupied by an atom that is not moving, or a target outside every
  zone.
- Distance of a move = max over its atoms of the Euclidean distance between their physical positions
  before and after.
- Parallel steps (time only): consecutive moves with disjoint qubits whose row and column spans stay
  apart, and whose paths cross no stationary atom.

## Our extra rule: zone span (ADR 0010)

A move that selects two or more rows must keep its picked-up rows within one zone and its target
rows within one zone. Single-row moves may cross the boundary. Not checked by the validator;
`bench` enforces it (`ZoneSpanError`).

## Our extra rule: transversal CZ (ADR 0007)

The validator treats each `cz` as an independent gate. Physically a CZ pulse hits **every** atom
pair on an interaction pair. The harness therefore checks that, between two consecutive moves, every
occupied interaction pair receives the same number of `cz` gates.

## Benchmark structure (useful for ideas)

- Vertex `v` → `q[v-1]`. In iHVA and QAOA the last qubit is the Max-Cut reference qubit `a`, which
  couples to every vertex. That makes it a hub with degree n−1.
- HEA: `ry`/`rz` layers plus linear `cx` chains (even pairs, then odd pairs).
- QAOA: `h`, then per layer `cx–rz–cx` ZZ blocks on every edge plus reference edges, then `rx` mixers.
- iHVA: `h` layer, then per Max-Cut edge a block `rx(π/2) · cx · rz(θ) · cx · rx(−π/2)` (a
  ZY-type two-qubit rotation). The `stagger` and `tree` variants differ in edge ordering, and `warm`
  adds a warm-start preparation.
- The graph in the 12-qubit QAOA/iHVA circuits uses 11 vertices plus the reference `q[11]`, so it is
  not identical to the 12-stop example in the brief. Read the edges from the QASM; don't hard-code
  the brief's graph.
- Gate totals over all 36 circuits: 9 356 `cx`, 5 752 `rx`, 5 176 `rz`, 1 312 `ry`, 656 `h`.
