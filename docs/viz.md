# viz: step-by-step visualiser for compiled circuits

`bench/viz.py` replays a compiled circuit with the **official validator** and writes a
self-contained HTML page per circuit: the 2D atom view (left) and the wire diagram (right),
animated instruction by instruction. It is debugging tooling — it never changes a score, and it
reads the same numbers `bench/bench.py` scores.

Rendering lives in `bench/viz_page.py` (`render_page`, `render_index`); `bench/viz.py` does the
data extraction and the CLI. Pages are vanilla JS + inline SVG/canvas, no CDN, and work from
`file://`.

## Commands

```sh
P=/home/hackathon/challenge/.venv/bin/python     # inside a worktree; else: uv run python

# one circuit (label from the bench sets, or a path to a .qasm)
$P bench/viz.py page compilers/naive/compiler.py --circuit metric_circuit_36
$P bench/viz.py page compilers/naive/compiler.py --circuit sample_circuit_03 -o /tmp/s3.html
$P bench/viz.py page compilers/naive/compiler.py --circuit metric_circuit_07 --json /tmp/d.json

# every circuit of a set -> viz/<name>/<label>.html + viz/<name>/meta.json + viz/index.html
$P bench/viz.py all compilers/naive/compiler.py --set all --jobs 2
$P bench/viz.py all /home/hackathon/challenge-wt/zonedwell/compilers/zonedwell/compiler.py --set metric

# rebuild only the overview table from every viz/*/meta.json
$P bench/viz.py index

# browse (read-only static server over viz/)
$P bench/viz.py serve --port 9000            # http://127.0.0.1:9000/
```

A `page` written to its default location (and every `all` run) also updates `viz/<name>/meta.json`
and rebuilds `viz/index.html`, so one-off pages show up in the overview too; records of circuits you
did not rebuild are kept.

Output goes to `viz/` at the repo root, which is gitignored. `<name>` is the compiler's folder
name; a compiler living in another worktree
(`/home/hackathon/challenge-wt/<wt>/compilers/<folder>/compiler.py`) is named `<wt>` (or
`<wt>-<folder>`), so pages from different branches never collide. `--name` overrides it.

`serve` also answers `GET /regen?compiler=<path>&set=<metric|sanity|all>`: it regenerates that
compiler's pages in a background thread and reports progress in `viz/status.json`, which the index
page polls. Only compiler paths inside this repo's `compilers/` or inside
`/home/hackathon/challenge-wt/*/compilers/` are accepted; everything else is rejected with 400.
The server is read-only otherwise and binds `127.0.0.1` by default.

`bench/viz_page.py` can also be exercised on its own, which is how the renderer is tested without a
browser (there is no node/deno in this image):

```sh
$P bench/viz_page.py --demo       -o /tmp/demo.html        # synthetic 12q circuit
$P bench/viz_page.py --demo-big   -o /tmp/demo_big.html    # 50 qubits, ~11k instructions
$P bench/viz_page.py --demo-index -o /tmp/demo_index.html
$P bench/viz_page.py --data /tmp/page.json -o /tmp/page.html   # from `viz.py page --json`
```
Each run prints a *static* JS check (balanced brackets outside strings/comments, and the embedded
`DATA`/`ROWS` JSON parses).

## Reading a page

**Atom view** (left) is drawn in *physical* coordinates, with the entangling zone on top, like
Fig. 1 of the challenge brief:

| element | meaning |
|---|---|
| light-blue panel, 10x10 open circles | storage zone (grid x 0-9, y 0-9), `rx`/`rz` allowed |
| orange panel, circles in doublets | entangling zone (y 10-19); physical x = `2*(gx//2) + 0.3*(gx%2)`, so interaction pairs sit 0.3 apart and repeat every 2.0 |
| filled blue dot `q<i>` | an atom; the purple one is the **hub** (max-degree qubit of the reference circuit) |
| blue ring / green ring + dashed cross | `rx` / targeted `rz` on that atom |
| orange oval over two atoms | a `cz` on that interaction pair |
| grey shading behind a pair | an occupied interaction pair |
| red outline on a pair | that pair is occupied but got fewer `cz` than another pair in the same segment (between two moves) — a transversal-CZ violation (ADR 0007) |
| dashed magenta lines | the grid rows/columns the AOD picked up for the current move |
| faint trail | where the moved atoms came from |

**Wire diagram** (right): one wire per qubit, instructions laid out in ASAP columns (moves get their
own columns). `rz` green box, `rx` blue box, `cz` dot-line-dot in orange, `measure` grey box,
`move` a thin magenta bar spanning the moved qubits. The cursor line marks the current position and
auto-scrolls; click a column to jump there; hover for the instruction details.

**Controls**: play/pause, step back/forward, speed, scrubber, and a granularity toggle —
*instruction* steps one instruction at a time, *frame* steps one move or one maximal block of
consecutive non-move instructions. Keys: `Space`, `←`, `→` (shift = x10), `Home`/`End`.

**Stats bar**: instruction i/N, move m/M, the current move's distance, cumulative distance with the
naive total for that circuit and the % of it used so far, depth so far, `cz` count, spacing
warnings, and a validity badge.

## Correctness

The page is built from `run_compiler(..., network=False)` + `verify_movement(...)` +
`compute_metrics(...)` — the same calls `bench/bench.py` makes — and the transversal-CZ check is
`physics_check` imported from `bench/bench.py`. The builder asserts that the sum of the per-move
distances equals `metrics.move_distance`, that the final `depth_at` equals `metrics.depth`, and
that the move and spacing-warning counts match, so the numbers on the page are the scored ones.
The page itself re-checks this at the end of the timeline and shows a red MISMATCH chip otherwise.

## Limitations

- No equivalence check: `viz` never runs QCEC. A page can be perfectly valid here and still fail
  `bench.py --mode full`. Use `bench.py` for verdicts, `viz` for understanding.
- Page size grows with the number of instructions: the biggest scored circuit
  (`metric_circuit_36`, 50 qubits, ~11k instructions) is ~0.7 MB, but the 70-qubit sanity circuit
  `sample_circuit_13` (~140k instructions, ~40k moves with naive) is ~9 MB and takes a few seconds
  to open.
- The pages were reviewed statically and exercised through `--demo`; there is no browser or JS
  engine in this image, so nothing was clicked by a machine.
- An invalid compilation is replayed by a fallback replay that stops at the **first** violation, so
  its metrics are partial (the page says so) and later violations are only listed, not drawn.
- `move_time` and parallel steps are shown but not animated in parallel: every move is drawn as its
  own step even when the validator merges it into one parallel step (the step number is displayed).
- Angles are shown for `rx`/`rz` with numeric parameters only.
- Pages are static snapshots; after changing a compiler, regenerate (CLI or the index's regenerate
  link).
