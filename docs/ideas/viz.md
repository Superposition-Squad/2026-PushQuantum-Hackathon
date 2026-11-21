# Tool: viz (step-by-step compiled-circuit visualiser)

- Branch: tool/viz · Worktree: /home/hackathon/challenge-wt/viz
- Base: f684e66
- Status: in progress

## Hypothesis

Planners and implementers currently read numbers (distance, depth) and replay logs. Seeing *where*
the atoms go, which AOD rows/columns are picked up, and which interaction pairs sit idle while a CZ
fires makes wasted distance, serialised moves and transversal-CZ violations obvious in seconds.
This is tooling: it does not change any score, it shortens the debug loop.

## Design

Two Python modules in `bench/` (no compiler, no vendor or bench.py edits):

- `bench/viz.py` — data extraction with the **official validator** + CLI (`page`, `all`, `index`,
  `serve`).
- `bench/viz_page.py` — pure rendering: `render_page(data) -> html`, `render_index(rows) -> html`.
  Vanilla JS + inline SVG + a canvas, no CDN, works from `file://`.

### Data source (must match the scorer exactly)

1. `compiled, loading = run_compiler(reference, "<path>:<fn>", network=False)`
2. `movement = verify_movement(compiled, loading, log_file=os.devnull)` → `trace` of `MoveRecord`
   (`index`, `qubits`, `positions`, `distance`, `step`, `spacing_warnings`).
3. `gates = strip_moves(compiled)`, `metrics = compute_metrics(compiled, movement, gates)`.
4. `physics_check(compiled, loading, trace)` from `bench/bench.py` (transversal CZ, ADR 0007).
5. Physical coordinates from `pushquantum_validator._validation.physical(zone, line, axis)` and
   `pushquantum_validator.spec.parse_spec(None)`.

If `verify_movement` raises `ArchitectureError`, a fallback replay (same `ValidationArchitecture`,
instruction by instruction) renders everything up to the first violation and the page shows the
error. If `run_compiler` raises, the page is an error card.

Invariants asserted at build time (the page must agree with `bench.py`):
`sum(move.d) == metrics.move_distance` (1e-6) and `depth_at[-1] == metrics.depth`.

### Schema (authoritative; `const DATA` embedded in each page, `"schema": 1`)

```jsonc
{
 "schema": 1,
 "meta": {
   "compiler": "zonedwell",            // display name == output folder
   "compiler_path": "/abs/compiler.py:compile_circuit",
   "label": "metric_circuit_36",
   "file": "metric_benchmarks/qaoa/bus_stop_50q_qaoa_warm_p3.qasm",
   "family": "qaoa",                   // hea|ihva|qaoa|sanity
   "qubits": 50,
   "status": "ok" | "invalid" | "compile_error",
   "error": null | "ArchitectureError[pair]: ...",
   "error_index": null | 1234,         // instruction index of the first violation
   "violations": ["Instruction 12 (cz on [3, 7]): ...", ...],   // <= 20
   "physics_ok": true,
   "physics": "transversal CZ ok (412 segments)" | "PhysicsError: ..." | "not checked",
   "metrics": {"num_moves":2988,"move_steps":2988,"move_distance":24906.91,"move_time":24906.91,
               "depth":2293,"num_operations":7865,"spacing_warnings":0},
   "naive": {"move_distance":24906.91,"depth":2293} | null,
   "compile_s": 1.47,
   "generated": "2026-11-21T13:55:00"
 },
 "spec": {"loading_zone":"storage",
          "zones":[{"name":"storage","x":[0,9],"y":[0,9],"spacing":1.0,
                    "pair_axis":null,"pair_distance":null,"operations":["rx","rz"],
                    "sites":[[gx,gy,px,py], ...]}, ...]},   // every trap, grid + physical
 "loading": [[x,y], ...],              // grid site per qubit index
 "hub": 49,                            // max-degree qubit of the reference (-1 if none)
 "degree": [11, 11, ..., 49],          // reference interaction degree per qubit
 "ops":   [[code,a,b], ...],           // one entry per instruction of the compiled circuit
 "angles":[0.0, 1.5708, ...],          // aligned with ops, 0.0 when not applicable
 "col":   [0,0,1, ...],                // ASAP column per instruction
 "ncols": 4321,
 "colmove":[0,1,0, ...],               // per column: 1 if it holds moves
 "depth_at":[0,1,1,2, ...],            // cumulative depth (moves+barriers stripped) after instr i
 "moves": [{"i":9,"q":[0,1],"b":[[0,0],[1,0]],"a":[[0,10],[1,10]],
            "rows":[0],"cols":[0,1],"nrows":[10],"ncols":[0,1],
            "d":10.02447,"s":0,"w":0}, ...],
 "warnings": [[moveIdx, "text"], ...]  // spacing warnings, <= 50
}
```

`ops` codes: `0 rz, 1 rx, 2 cz, 3 move, 4 measure, 5 barrier, 6 other`.
`0/1/4`: `a` = qubit, `b` = -1. `2`: `a`,`b` = qubits. `3`: `a` = index into `moves`, `b` = -1.
`5`: `a` = -1. Move `i` is the instruction index; `b`/`a` are sites **before**/**after** for `q`.

Columns: ASAP layering, `c = max(col[q] for q in qubits) + 1`, then advanced until the column is
free or of the same kind (moves never share a column with gates).

### Page

- Left: atom view, SVG in physical coordinates, **entangling zone on top** (screen y inverted) like
  Fig. 1: orange panel with trap doublets (pair distance 0.3, pitch 2.0), light-blue storage panel
  with a 10x10 trap grid. Atoms = filled blue dots labelled `q<i>`, hub in a distinct colour.
  rx/rz → ring on the atom (rz also draws the dashed targeting cross), cz → orange oval over the
  pair. Moves highlight the picked-up rows/cols as AOD lines and interpolate atoms from `b` to `a`
  with faint trails. Occupied interaction pairs are shaded; a pair that is occupied but received
  fewer CZ than the maximum in the current segment (between two moves) is marked red →
  transversal-CZ violations are visible.
- Right: wire diagram on a canvas, one wire per qubit, columns from `col`, virtualised (only the
  visible column range is drawn), horizontal scroll, cursor line with auto-scroll, click to jump.
- Controls: play/pause, speed, step back/forward, scrubber, granularity (instruction | frame, where
  a frame is one move or a maximal run of consecutive non-move instructions). Keys: space, ←, →.
- Stats bar: instruction i/N, move m/M, this move's distance, cumulative distance (+ naive total and
  % so far), depth so far, CZ count, spacing warnings, compiler, circuit, validity badge.

### Output tree (`viz/` at the repo root, gitignored)

`viz/<name>/<label>.html`, `viz/<name>/meta.json` (compiler + one lite record per circuit),
`viz/index.html` rebuilt from every `viz/*/meta.json`, `viz/status.json` for `serve --regen`.
`<name>` = compiler folder name inside this repo; for a compiler in another worktree
`/home/hackathon/challenge-wt/<wt>/compilers/<folder>/compiler.py` it is `<wt>` (or `<wt>-<folder>`
if they differ), so worktree pages never collide with main's.

## Results

| iteration | what | status |
|---|---|---|
| 1 | `bench/viz.py` (data + CLI) and `bench/viz_page.py` (renderer) implemented to the schema above | done |
| 2 | naive, all 49 circuits: every metric (`move_distance`, `depth`, `num_moves`, `move_steps`, `num_operations`, `spacing_warnings`) identical to `bench/bench.py` | verified |
| 3 | zonedwell, 36 metric circuits: viz reports +78.80 % (distance +79.26 %, depth +76.95 %), equal to its own `results/zonedwell.json` | verified |
| 4 | per-move distances were rounded to 6 dp, so the page's cumulative sum drifted from `metrics.move_distance` on 40 k-move circuits (the page's MISMATCH guard caught it) → store full precision, and assert on the *stored* values | fixed |
| 5 | invalid compilation (scratch broken compiler): `status=invalid`, `error_index=37`, partial metrics, violations listed, page renders up to the failure | verified |
| 6 | `serve`: `/` 200, pages 200, `/regen` 202 → 409 while running → `status.json` progress, `/etc/passwd` and traversal 400, `POST` 405 | verified |
| 7 | 99 generated pages pass the static JS check (balanced brackets, `DATA` parses, no dangling DOM ids) | verified |

Sizes: `metric_circuit_36` (50 q, 11 k instructions) 0.70 MB; biggest overall is the 70-qubit
sanity circuit `sample_circuit_13` (140 k instructions, 40 k moves) at 9.0 MB.

## Verdict

Delivered. Not a scoring change: this is debugging tooling. The guarantee that matters is that the
numbers on a page are the numbers `bench/bench.py` scores — asserted at build time, re-checked in
the page, and cross-checked against bench runs for two compilers (49 + 36 circuits).
No browser or JS engine exists in this image, so the pages were verified statically and through the
renderer's `--demo` modes, never clicked.
