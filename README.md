# planqc compiler track (PushQuantum Hackathon 2026)

Compilers for planqc's zoned neutral-atom machine: map a circuit to `rx`/`rz` (storage zone), `cz`
(entangling zone) and AOD `move`s, minimising total move distance (80 %) and depth (20 %).

## Current state (validator v5.0.0, final)

- **Scored set:** 20 circuits: the 18 iHVA circuits (12, 20 and 50 qubits) and the two 12-qubit HEA
  circuits. iHVA is ~90 % of the scored naive distance; 50q alone ~61 %. QAOA is not scored.
- **Rules that shape everything** (full list: [docs/rules.md](docs/rules.md)):
  - **idle rule:** at every CZ layer, every atom in the entangling zone must be in a CZ (no lone
    or parked atoms in the zone during a pulse);
  - **single-zone moves:** each move picks up in one zone and drops off in one zone;
  - no `rzz`; true equivalence at every size; same qubit count (no ancillas).

| compiler | v5 score (20 circuits) | what it is |
|---|---|---|
| `systolic_po` | **+86.30** | systolic + post-optimizer (below) |
| `systolic` | +85.34 | storage homes, hub stays in the zone, row rotation, longest-path layout |
| `optimizer-optimizer` | +81.79 | pulse-beam planner: beam search over pulses; atoms live in storage, block drops; wins most 12q iHVA |
| `portfolio` (official, `submission/portfolio.py`) | +85.50 | systolic, hubstream, rectbatch, gates |
| `portfolio_po` | +86.63 (full 20/20; sanity 12 times out, fix pending) | best of all `*_po` components per circuit |

History and per-round numbers: [docs/results-log.md](docs/results-log.md).

## Layout

- `compilers/<name>/compiler.py`: one self-contained compiler per folder (stdlib, qiskit, numpy,
  scipy; top-level `compile_circuit(circuit, spec)`).
- `compilers/postopt.py`: **post-optimizer** for any compiler's output. Between two CZ layers only
  the start positions, the end positions and the 1q gates in between matter; each gap is
  re-planned (block drop of the zone, the gates, grouped moves to the end positions) and kept only
  if cheaper. Gates and their order are untouched (same depth, same equivalence); the result is
  replayed with the validator's rules before it is returned.
- `bench/make_postopt.py <name>`: builds `compilers/<name>_po/` = component + postopt + wrapper.
- `bench/build_portfolio.py <components...>`: generates a portfolio compiler (runs all components,
  keeps the cheapest valid result per circuit; never edit the generated file by hand, ADR 0008).
- `vendor/pushquantum_validator/`: the official validator v5.0.0, byte-identical.
- `docs/`: scoring, harness, rules, ADRs, idea docs ([docs/README.md](docs/README.md)).

## Measure

```sh
uv sync                                                        # once
.venv/bin/python bench/bench.py compilers/<name>/compiler.py   # v5 set, fast mode
.venv/bin/python bench/bench.py compilers/<name>/compiler.py --mode full        # + official equivalence
.venv/bin/python bench/bench.py compilers/<name>/compiler.py --set all --mode strict   # merge gate
bench/devbench.sh --filter 12q                                 # fast dev loop for optimizer-optimizer
.venv/bin/python bench/package.py compilers/<name>/compiler.py -o submission/<name>.py --score
```

Submitting to the leaderboard is a manual step that needs the team's OK (ADR 0005).

## Visualiser

```sh
.venv/bin/python bench/viz.py page compilers/<name>/compiler.py --circuit metric_circuit_19
.venv/bin/python bench/viz.py all  compilers/<name>/compiler.py --set metric
.venv/bin/python bench/viz.py serve --host 127.0.0.1 --port 9000     # http://127.0.0.1:9000/
```

Pages are generated locally and are **not in git** (`viz/` is ignored), so every machine shows
what it generated itself. To keep a hosted viz current with `main`, run
`bench/viz_sync.sh` next to the server: it pulls every few minutes and regenerates the pages of
every compiler whose code changed (details in [docs/viz.md](docs/viz.md)).
