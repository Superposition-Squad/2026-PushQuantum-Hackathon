# planqc compiler track: instructions for agents

We compile quantum circuits for planqc's simplified zoned neutral-atom machine (PushQuantum
Hackathon 2026). Goal: maximise the leaderboard score, computed locally by `bench/`.

Read before working: `docs/scoring.md` (what counts), `docs/harness.md` (how to measure),
`docs/adr/` (decisions you must respect), `docs/literature.md` (technique ideas).

## Scoring (validator v5.0.0, final — ADR 0011)

Only 20 circuits are scored: the 18 iHVA circuits + 12q HEA p1/p3. `bench` defaults to that set
(`--full-set` for all 36). The `idle` rule is mandatory: in every two-qubit gate layer, every atom in
the entangling zone must take part in a two-qubit gate (no atoms waiting in the zone).

## Hard rules

1. **No outward actions.** Never submit to the leaderboard, create or publish aqora workspaces, run
   aqora GraphQL mutations, or `git push` (ADR 0005). Local commits on your own branch are fine.
2. **Gate set:** only `rx`, `rz` (storage zone), `cz` (entangling zone), `move`, `measure` and
   `barrier`. No `rzz` (ADR 0002).
3. **Truly equivalent** output for every circuit size. Never rely on the validator skipping full
   equivalence above 19 qubits (ADR 0003).
4. **Transversal CZ:** between two moves, every occupied interaction pair gets the same number of
   `cz` (ADR 0007). `bench` enforces this.
5. **Zone span:** a move selecting ≥ 2 rows keeps its picked-up rows in one zone and its target
   rows in one zone; single-row moves may cross (ADR 0010). `bench` enforces this.
6. **One self-contained file per compiler:** `compilers/<idea>/compiler.py`, importing only the
   stdlib, qiskit, numpy and scipy. It defines top-level `compile_circuit(circuit, spec)`. Copy the
   `Move` and `Architecture` plumbing from `compilers/naive/compiler.py` (ADR 0006).
7. **Compile time:** at most about 60 s per circuit locally (the hard limit is 300 s).
8. Don't edit `vendor/` or `bench/` without the orchestrator's approval. If you think the harness is
   wrong, say so in your report.

## Workflow

- Measure: `<python> bench/bench.py compilers/<idea>/compiler.py [--filter 12q] [--mode full]`. Use
  `/home/hackathon/challenge/.venv/bin/python` as `<python>` inside worktrees.
- Before reporting a result: run the full metric set in `--mode full`, and `--set sanity --mode full`.
- Commit `compilers/<idea>/`, `results/<idea>.json` and `docs/ideas/<idea>.md` on your branch.
- Report: the score, distance and depth improvements, failures, and what to try next.
