# planqc track — validator / scoring notes

Source: `leaderboard/pushquantum_validator/` (public workspace `planqc/pushquantum-2026-leaderboard` v0.0.0)
and `leaderboard/readme.py`.

## Submission flow

1. Clone `planqc/pushquantum-2026-template`, implement `compile_circuit` as a top-level `@app.function` in the notebook.
2. Publish a workspace version; the workspace must be readable by `planqc`.
3. In the leaderboard notebook, sign in, enter `<your-username>/<workspace>`, submit.
4. Scoring downloads the notebook as `planqc`, imports it in a sandboxed child process with **no network**,
   runs only the setup cell + top-level functions/classes (not ordinary cells), and calls
   `compile_circuit(reference, spec)` on every benchmark.
5. Only the best score is kept; ties go to the earliest submission.

## Validity checks (any failure rejects the whole submission)

- **Movement replay** (`verify_movement`): initial loading all in storage, no duplicate sites; every `move`
  must be legal; each gate must be in a zone allowing it; `cz` only on an interaction pair.
- **Equivalence** (`verify_equivalence`): QCEC on the compiled circuit with moves/barriers stripped.
  - Up to 19 qubits: ZX + simulation, then exact alternating check if inconclusive.
  - Above 19 qubits: **no full equivalence check**; only the set of entangled qubit pairs is compared against a
    depth-1 transpilation of the reference (missing/extra pairs fail).

## Metric

36 metric benchmarks: HEA (6), IHVA (18, stagger/tree/warm), QAOA (12, incl. warm starts); 12/20/50 qubits.
Naive baseline totals: **move distance 276,726.91**, **depth 23,676**, 31,000 moves.

    score = 0.8 * (naive_distance - distance) / naive_distance * 100
          + 0.2 * (naive_depth - depth) / naive_depth * 100

- `distance` = sum over moves of the longest distance any atom in that move travels (physical distances;
  entangling-zone interaction pairs are 0.3 apart, pairs repeat every 2.0).
- `depth` = depth of the circuit **with moves and barriers removed**.
- Parallelising moves does not reduce `distance`; it reduces `time` (not part of the score) and can group
  atoms so one move carries several (each move still counts its own max distance once).
- Spacing warnings double a move's `time` only (not distance/depth).

## Move semantics (validator)

- `Move(rows, cols, new_rows, new_cols, num_qubits)`; rows = y, cols = x.
- Pickup lines must be contiguous ascending; targets ascending (order preserved), targets <= pickup count.
- Proportional spacing is a warning, not an error (unless empty-line squeezing mismatches one-to-one).
- Ghost pickup, depositing onto an occupied site, or leaving all zones is an error.
- Parallel steps: consecutive moves share a step only if their row/col spans stay apart and paths do not cross
  stationary atoms.

## Architecture (default)

- storage: x 0–9, y 0–9, ops `rx`, `rz`, spacing 1.0.
- entangling: x 0–9, y 10–19, ops `cz`, `rzz`, pairs along x with distance 0.3.
- `measure` and `barrier` allowed anywhere; loading zone is storage.

## Practical implications for a solution

- Minimal-distance strategy: batch many atoms per AOD move, use long row/column moves that carry whole
  groups, and avoid one-move-per-CZ (the naive baseline).
- Depth: schedule disjoint CZ pairs in parallel layers; minimise swap/reroute gates; single-qubit `rx`/`rz`
  are free of movement while atoms sit in storage.
- The 20q/50q circuits skip full equivalence: only interaction pairs are checked, which leaves room for
  aggressive algebraic optimisation — but any pair mismatch rejects.
