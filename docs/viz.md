# Visualiser (`bench/viz.py`)

Step-by-step view of what a compiler does: the compiled circuit (wire diagram) and the atom
positions on the zoned machine (2D, physical coordinates, styled like Fig. 1/2 of the brief), in
sync, with a cursor you can play, step, scrub or click.

Everything shown comes from the **official validator**: the circuit is compiled with
`run_compiler(reference, compiler, network=False)` (sandboxed child process, like `bench.py`), and
replayed with `verify_movement(compiled, loading)`. Atom positions, per-move distances, parallel
steps and spacing warnings are the validator's `MoveRecord` trace; the final totals are
`compute_metrics(...)`, so they equal `bench.py`'s numbers. The transversal-CZ check is
`bench.physics_check`. An invalid compilation is still rendered: the validator's replay is re-run
without raising (it skips illegal instructions), every violation is marked, and the first error is
shown in a red banner with a button that jumps to it.

## Usage

Run from the repo root (inside a worktree use `/home/hackathon/challenge/.venv/bin/python`):

```sh
# one page (bench label or any .qasm file); also refreshes viz/index.html
python bench/viz.py page compilers/<idea>/compiler.py --circuit metric_circuit_07 [--mode full]

# every circuit of a set (default: all 49), 2 workers; prints bench-style lines and the score
python bench/viz.py all compilers/<idea>/compiler.py [--set metric|sanity|all] [--filter 50q] [--jobs 2]

python bench/viz.py index        # rebuild viz/index.html from the sidecars
python bench/viz.py rewrap       # re-embed existing pages into the current template (no recompile)
python bench/viz.py serve        # http://<host>:9000/ serving viz/ (default host 0.0.0.0)
```

- Output: `viz/<name>/<label>.html` (one self-contained page, no external assets) plus
  `viz/<name>/<label>.json` (sidecar with metrics, used by the index). `viz/` is gitignored.
  `--out DIR` changes the output directory for every subcommand.
- `<name>`: the idea folder for compilers under `compilers/` of this checkout or of the main repo
  (`naive`, `gates`), `<worktree>-<idea>` for `/home/hackathon/challenge-wt/<wt>/compilers/<idea>/`
  (`rectbatch-rectbatch`), `scratch-<folder>` otherwise; `--name` overrides.
- `--mode fast` (default) = compile + replay + metrics + transversal-CZ check, like `bench.py`;
  `--mode full` adds the official equivalence check.
- The pages work from `file://` too; only the ↻ Regenerate button needs the server.
- `serve` exposes `POST /api/regenerate` `{"compiler": <abs path>, "circuit": <bench label>,
  "mode": "fast"|"full"}`, used by the page's ↻ button: it recompiles that circuit and rebuilds
  the page. Only `*.py` files whose resolved path (symlinks followed) lies under
  `/home/hackathon/challenge/compilers/` or `/home/hackathon/challenge-wt/<wt>/compilers/` are
  accepted, and only known bench labels; one regeneration at a time (409 when busy).
  `GET /api/ping` tells the page that the API exists.
- Smoke checks (dev-only packages `playwright`, `esprima`; see `pyproject.toml`):
  `python bench/viz_check.py viz/naive/metric_circuit_01.html ...` (static + headless Chromium) and
  `python bench/viz_check.py --compare viz/<name> results/runs/<run>.json|naive` (sidecar metrics
  vs a bench run). Chromium needs `python -m playwright install chromium`; on this machine its
  missing shared libraries are in `~/.cache/viz-chromium-libs/lib` (conda-forge, via micromamba),
  so prefix `LD_LIBRARY_PATH=/home/hackathon/.cache/viz-chromium-libs/lib`.

## The page

Left: **atom view** in physical coordinates (storage at the bottom, entangling zone on top, as in
the brief's Fig. 1/2), with row/column grid labels. Right: **circuit view** (one wire per qubit,
ASAP columns, horizontally scrollable), an **overview strip** of the whole instruction stream,
the controls and the live stats.

| control | key |
|---|---|
| play / pause | `Space` |
| step back / forward (current granularity) | `←` / `→` |
| step one frame back / forward | `Shift`+`←` / `→` |
| previous / next move | `↑` / `↓` (or `,` / `.`) |
| start (initial loading) / end | `Home` / `End` |
| granularity: instruction ↔ frame | `G` |
| speed ×1.5 / ÷1.5 (0.25–60 steps/s) | `+` / `−` |
| circuit zoom | `]` / `[` (or Ctrl+wheel) |
| atom labels on/off | `L` |
| jump to the first error | `E` |
| jump anywhere | click the circuit, the overview strip, or drag the scrubber |

- Cursor = "instruction *i* has just executed"; position 0 is the initial loading. A **frame** is
  one move, or a maximal block of consecutive non-move instructions.
- Stepping forward onto a move animates it (ease-in-out, from the old to the new physical
  positions) with the picked-up AOD rows/columns drawn as green dashed bands that travel with
  the atoms. Resting on a move shows the bands at their target, dashed ghosts at the old
  positions and arrows. Jumps and backward steps are instant.
- Stats: position and frame, the current instruction (angles in rad and multiples of π), moves
  m/M, this (or the last) move's distance, Σ distance so far vs this circuit's total and the
  naive compiler's (black tick on the bar; Δ % = total/naive − 1, negative is better), depth so
  far (moves and barriers stripped, as scored) vs total and naive, CZ count, spacing warnings,
  parallel step, validity (validator replay, equivalence status, transversal-CZ check), and the
  final totals from `compute_metrics()`.
- Hovering a circuit glyph shows its details; for moves, the qubits, `rows→new_rows`,
  `cols→new_cols`, distance, parallel step and spacing warnings.

Legend (also on the page, "legend" in the footer):

| atom view | circuit view |
|---|---|
| hollow gray circle: trap site (entangling traps are doublets 0.3 apart) | blue box "X": `rx` |
| filled blue dot: atom (label = qubit index) | green box "Z": `rz` |
| blue ring: `rx` on that atom | two orange dots + line: `cz` |
| green ring + dashed crosshair: `rz` (targeted) | purple bar + diamonds: `move` (diamonds on moved qubits) |
| orange overlapping discs: `cz` pair | gray "M": `measure`; dashed gray line: `barrier` |
| green dashed bands: AOD rows/cols of the current move | yellow background: current instruction (pale: rest of its frame) |
| red ✕: instruction that violates a rule | faded glyphs: not executed yet; red outline/fill: violation/error |
| red dashed ring: under-pulsed pair of a failing CZ segment | red dashed line: cursor |

In instruction mode, the gates of the current frame other than the current one are drawn faintly
in the atom view, so you can see the whole parallel block around the cursor.

## Invalid compilations

- `compile` (the compiler raised, timed out or returned bad output): banner with the message;
  no circuit or atoms.
- `loading` (invalid initial loading): the instructions are shown, without a replay.
- `movement` (illegal move, gate outside its zone, CZ off a pair): the validator's replay is
  re-run without raising (`ValidationArchitecture._replay`, which skips illegal instructions as
  the validator does) so the page renders the whole circuit; every violation is outlined red,
  the first one is the banner's jump target, and totals are marked *partial*. Illegal moves are
  not applied, exactly like the validator.
- `physics` (transversal CZ, ADR 0007): official metrics are kept; the banner jumps to the last
  instruction of the faulty segment and the under-pulsed pairs are ringed in red.
- `equivalence` (`--mode full` only): banner with the QCEC / interaction-pair message.

## Limitations

- Sizes: 50-qubit p=3 pages are ~0.6–0.9 MB (gates/naive); the 70-qubit sanity circuit
  `sample_circuit_13` (naive: 140 k instructions, 40 k moves) is ~11 MB and takes ~25 s to build,
  but loads in < 1 s and stays responsive. Numbers are compacted (physical coordinates rounded
  to 4 decimals, angles to 5); move distances are stored unrounded.
- Positions are drawn only at sites (start/end of each move); the straight-line interpolation
  during the animation is illustrative, not the validator's path model (it doesn't model AOD
  ordering constraints or the parallel-step grouping beyond showing the step number).
- The circuit view's ASAP columns put a move or barrier across the whole span of its qubits'
  wires, so it shows the instruction order faithfully but is not the scored depth (which strips
  moves). The scored depth is the "depth so far" stat.
- The regenerate endpoint runs the compiler on the server machine; keep the server on a trusted
  network.
- `rewrap` reuses the embedded data; after changing `bench/viz.py`'s data extraction, rebuild
  with `page`/`all` instead.

## Page data format (contract between `bench/viz.py` and `bench/viz_page.html`)

`bench/viz.py` reads the template `bench/viz_page.html`, replaces the single placeholder line
`<script id="viz-data" type="application/json">__VIZ_DATA__</script>` content `__VIZ_DATA__` with
`json.dumps(data, separators=(",", ":"))` in which every `</` is escaped as `<\/`, and writes one
self-contained HTML file. The page's script reads it with
`JSON.parse(document.getElementById("viz-data").textContent)`.

All coordinates called *physical* are the validator's physical coordinates (`physical()` in
`vendor/pushquantum_validator/_validation.py`): storage `(x, y)` = `(col, row)`; entangling
`x = 2*(col//2) + 0.3*(col%2)`, `y = row`. y grows **upwards** (storage at the bottom, entangling on
top, as in the brief). Physical numbers are rounded to 4 decimals. *Sites* are integer grid
`(x, y)` = (col, row) as used by moves.

```jsonc
{
  "version": 1,
  "meta": {
    "name": "naive",                    // page/compiler name (folder in viz/)
    "compiler": "/abs/path/compilers/naive/compiler.py",
    "circuit": "metric_circuit_07",     // bench label, or the qasm file stem for other files
    "qasm": "metric_benchmarks/hea/....qasm",  // path relative to vendor/pushquantum_validator, or absolute
    "family": "hea",                    // hea | ihva | qaoa | sanity | other
    "num_qubits": 12, "num_clbits": 12,
    "generated": "2026-11-21T15:00:00Z",
    "compile_s": 1.23,
    "mode": "fast"                      // fast | full (full = official equivalence check was run)
  },
  "geometry": {
    "zones": [                          // spec order; physical bounding box of the trap sites
      {"name": "storage", "ops": ["rx","rz"], "x0": 0, "x1": 9, "y0": 0, "y1": 9,
       "px0": 0.0, "px1": 9.0, "py0": 0.0, "py1": 9.0, "paired": false},
      {"name": "entangling", "ops": ["cz","rzz"], "x0": 0, "x1": 9, "y0": 10, "y1": 19,
       "px0": 0.0, "px1": 8.3, "py0": 10.0, "py1": 19.0, "paired": true}
    ],
    "traps": [[0.0, 0.0, 0, 0, 0], ...],  // every site of every zone: [px, py, zone_index, col, row]
    "bounds": [xmin, ymin, xmax, ymax]   // physical bounding box over all traps
  },
  "loading": [[px, py, col, row], ...],  // per qubit (index = qubit); null if no compiled output
  "ops": ["rx","rz","cz","move","measure","barrier"],   // name table; other names appended
  "inst": {                             // one entry per instruction of compiled.data, in order
    "op":  [0, 1, 2, 3, ...],           // index into "ops"
    "qs":  [[0], [3], [2,5], [2,5], ...],  // qubit indices of the instruction
    "par": [1.5708, -0.7854, null, ...],   // first parameter (rad, 5 decimals) for rx/rz, else null
    "col": [0, 0, 1, 2, ...],           // ASAP column in the wire diagram (see below)
    "depth": [1, 1, 2, 2, ...],         // depth of compiled.data[:i+1] with moves and barriers removed
    "mv":  [-1, -1, -1, 0, ...]         // index into "moves" for a replayed move instruction, else -1
  },
  "moves": [                            // the validator's trace, one per *replayed* move, in order
    {"i": 3,                            // instruction index
     "q": [2, 5],                       // moved qubits (MoveRecord.qubits)
     "f": [px, py, px, py],             // physical positions before the move, aligned with q
     "t": [px, py, px, py],             // physical positions after the move, aligned with q
     "ts": [x, y, x, y],                // target sites after the move, aligned with q
     "d": 10.0123,                      // MoveRecord.distance (full precision float)
     "s": 3,                            // MoveRecord.step (parallel step, 0-based)
     "w": [0],                          // indices into "warnings" (spacing warnings of this move)
     "rows": [5], "cols": [2, 3], "nrows": [10], "ncols": [2, 3],  // raw move params
     "br": [[5.0, 10.0]],               // AOD row bands: [physical y before, after] per used row line
     "bc": [[2.0, 2.0], [3.0, 2.3]]     // AOD col bands: [physical x before, after] per used col line
    }
  ],
  "warnings": ["Target cols ... not evenly spaced ...", ...],  // unique spacing-warning texts
  "violations": [{"i": 17, "kind": "pair", "msg": "Instruction 17 (cz on [1, 4]): ..."}],
  "error": null,                        // or {"stage": "...", "msg": "...", "i": 17 | null}
                                        // stage: compile | loading | movement | physics | equivalence | harness
  "totals": {                           // compute_metrics() when the replay passed; else partial, from the trace
    "num_moves": 20, "move_steps": 18, "move_distance": 123.456789, "move_time": 130.1,
    "depth": 45, "num_operations": 80, "spacing_warnings": 0, "cz": 30, "partial": false
  },
  "naive": {"move_distance": 244.1, "depth": 60, "num_moves": 40, ...} ,  // naive_baseline.json metrics or null
  "equivalence": "skipped (fast)",      // or the bench-style status string, or "FAILED: ..."
  "physics": {"ok": true, "segments": 10, "msg": null},
  "regen": {"compiler": "/abs/path/compiler.py", "circuit": "metric_circuit_07"}
}
```

ASAP column: per instruction, `span` = all wires from min to max of its qubits (a single-qubit
gate spans one wire; a move or barrier spans min..max of its qubits). `col = max(next_free[w] for w
in span)`, then `next_free[w] = col + 1` for every w in span. Instructions with no qubits get
`max(next_free)`.

Depth: qiskit's `depth()` of the stripped circuit, computed incrementally over qubits **and clbits**:
for every instruction that is not `move`/`barrier`, `lvl = 1 + max(level[b] for b in its qubits and
clbits)` (0 if none), all its bits get `lvl`, running depth = max so far. Moves and barriers keep the
running depth. The last value equals `strip_moves(compiled).depth()`.
