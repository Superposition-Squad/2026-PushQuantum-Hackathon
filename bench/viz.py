#!/usr/bin/env python3
"""Visualiser data builder for planqc compilers (companion to bench/bench.py).

Every number comes from the official validator, using exactly the same calls as `bench.py`, so the
pages match the benchmark. The page and index templates are `bench/viz_page.html` and
`bench/viz_index.html` (self-contained HTML/JS; the data is embedded as JSON, see docs/viz.md); this
file extracts the data, writes pages and sidecars, runs the local preview server and prints the
bench-style score.

Commands (run from the repo root):
    python bench/viz.py page  <compiler.py> --circuit <label|path.qasm> [--name NAME] ...
    python bench/viz.py all   <compiler.py> [--set metric|sanity|all] [--filter SUBSTR] ...
    python bench/viz.py index [--out DIR]
    python bench/viz.py rewrap [--out DIR]     (re-embed existing pages into the current template)
    python bench/viz.py serve [--host 0.0.0.0] [--port 9000] [--out DIR]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import functools
import html
import json
import logging
import math
import numbers
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "vendor"))

import bench  # noqa: E402  (importing bench also puts vendor/ on sys.path)

from qiskit import QuantumCircuit  # noqa: E402

from pushquantum_validator._validation import ValidationArchitecture, physical  # noqa: E402
from pushquantum_validator.errors import (  # noqa: E402
    ArchitectureError,
    CompilerOutputError,
    NotEquivalentError,
)
from pushquantum_validator.spec import parse_spec  # noqa: E402
from pushquantum_validator.verify import (  # noqa: E402
    benchmark_label,
    compute_metrics,
    metric_benchmarks,
    naive_baseline,
    run_compiler,
    sanity_benchmarks,
    strip_moves,
    verify_equivalence,
    verify_movement,
)

logging.getLogger("pushquantum_validator").setLevel(logging.ERROR)  # spacing warnings are counted instead

BASE_OPS = ["rx", "rz", "cz", "move", "measure", "barrier"]
"""Name table; `ops` is this plus any other operation encountered, in order of appearance."""

EXCLUDED = frozenset({"portfolio"})
"""Compilers left out of the visualisation (portfolio = per-circuit best of the others): `page`, `all`
and the regenerate endpoint refuse them (unless `--force`), and the index, `manifest.json` and the
comparison sidebar skip them even if their pages exist. Matched against the idea folder / page name."""


def is_excluded(name_or_path) -> bool:
    """Whether a page name (`portfolio`, `<worktree>-portfolio`) or a compiler file is excluded."""
    text = str(name_or_path)
    if text.endswith(".py"):
        return Path(text).resolve().parent.name in EXCLUDED
    return text in EXCLUDED or text.rsplit("-", 1)[-1] in EXCLUDED


CHALLENGE_COMPILERS = Path(__file__).resolve().parents[1] / "compilers"
"""Compiler root of the canonical checkout."""

WT_ROOT = Path(__file__).resolve().parents[2] / "challenge-wt"
"""Root of the worktree checkouts; `<wt>/compilers/<idea>/...` is allowed too."""

VIZ_VERSION = 1

# Minimal stand-in used only if bench/viz_page.html is missing. It is never written to disk.
FALLBACK_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__VIZ_TITLE__</title></head>
<body>
<script id="viz-data" type="application/json">__VIZ_DATA__</script>
<p>The visualiser page template <code>bench/viz_page.html</code> is missing; this is a stand-in.</p>
</body></html>
"""


# ---------------------------------------------------------------------- paths and names


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", str(name)).strip("-.")
    return cleaned or "compiler"


def compiler_name(path: str | Path) -> str:
    """Page folder name for a compiler file (see the contract)."""
    p = Path(path).resolve()
    for root in (ROOT / "compilers", CHALLENGE_COMPILERS):
        if p.is_relative_to(root):
            return sanitize_name(p.parent.name)
    try:
        rel = p.relative_to(WT_ROOT)
    except ValueError:
        pass
    else:
        if len(rel.parts) >= 4 and rel.parts[1] == "compilers":
            return sanitize_name(f"{rel.parts[0]}-{rel.parts[2]}")
    return sanitize_name("scratch-" + p.parent.name)


@functools.lru_cache(maxsize=1)
def known_circuits() -> dict[str, Path]:
    """Bench label -> qasm path for both bundled sets."""
    found: dict[str, Path] = {}
    for path in sanity_benchmarks() + metric_benchmarks():
        label = benchmark_label(path)
        if label:
            found[label] = path.resolve()
    return found


def resolve_circuit(value: str) -> tuple[Path, str | None]:
    """`--circuit` value (bench label or qasm path) -> (resolved path, bench label or None)."""
    known = known_circuits()
    if value in known:
        return known[value], value
    path = Path(value)
    if path.is_file():
        p = path.resolve()
        return p, benchmark_label(p)
    raise SystemExit(f"unknown circuit label or qasm file: {value}")


def output_dir(value: str | None) -> Path:
    out = Path(value) if value else ROOT / "viz"
    return out.resolve()


def _qasm_relative(path: Path) -> str:
    base = (ROOT / "vendor" / "pushquantum_validator").resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


@functools.lru_cache(maxsize=1)
def _naive_baseline() -> dict:
    return naive_baseline()


def _naive_for(label: str | None) -> dict | None:
    if not label:
        return None
    record = _naive_baseline().get(label)
    if not record:
        return None
    metrics = dict(record["metrics"])
    metrics["spacing_warnings"] = record.get("spacing_warnings")
    return metrics


# ---------------------------------------------------------------------- geometry


def _zone_map() -> tuple[list, dict]:
    """Zones in spec order plus a site -> zone map (the first zone containing a site wins)."""
    zones, _loading_zone = parse_spec()
    zone_list = list(zones.values())
    zone_at: dict = {}
    for zone in reversed(zone_list):  # later zones overwritten by earlier ones, as in the validator
        for x in range(zone.x[0], zone.x[1] + 1):
            for y in range(zone.y[0], zone.y[1] + 1):
                zone_at[(x, y)] = zone
    return zone_list, zone_at


def _geometry() -> dict:
    zone_list, zone_at = _zone_map()
    zones = []
    for index, zone in enumerate(zone_list):
        zones.append(
            {
                "name": zone.name,
                "ops": sorted(zone.operations),
                "x0": zone.x[0],
                "x1": zone.x[1],
                "y0": zone.y[0],
                "y1": zone.y[1],
                "px0": round(physical(zone, zone.x[0], "x"), 4),
                "px1": round(physical(zone, zone.x[1], "x"), 4),
                "py0": round(physical(zone, zone.y[0], "y"), 4),
                "py1": round(physical(zone, zone.y[1], "y"), 4),
                "paired": zone.pair_axis is not None,
            }
        )
    traps = []
    claimed: set = set()
    for index, zone in enumerate(zone_list):
        for x in range(zone.x[0], zone.x[1] + 1):
            for y in range(zone.y[0], zone.y[1] + 1):
                site = (x, y)
                if site in claimed:
                    continue
                claimed.add(site)
                traps.append([round(physical(zone, x, "x"), 4), round(physical(zone, y, "y"), 4), index, x, y])
    if traps:
        xs = [t[0] for t in traps]
        ys = [t[1] for t in traps]
        bounds = [min(xs), min(ys), max(xs), max(ys)]
    else:
        bounds = [0.0, 0.0, 0.0, 0.0]
    return {"zones": zones, "traps": traps, "bounds": bounds}


def _phys(site, arch, zone_at) -> tuple[float, float]:
    """Physical position of a grid site, as the validator computes it."""
    if arch is not None:
        try:
            return arch.physical_position(site)
        except ArchitectureError:
            pass
    zone = zone_at.get(tuple(site))
    if zone is None:
        return float(site[0]), float(site[1])
    return physical(zone, site[0], "x"), physical(zone, site[1], "y")


def _loading_positions(loading, num_qubits: int, arch, zone_at) -> list[list]:
    positions = []
    for qubit in range(num_qubits):
        site = loading.get(qubit, (0, 0))
        px, py = _phys(site, arch, zone_at)
        positions.append([round(px, 4), round(py, 4), int(site[0]), int(site[1])])
    return positions


# ---------------------------------------------------------------------- instruction arrays


def _instruction_arrays(compiled: QuantumCircuit, move_of_index: dict[int, int]) -> tuple[list, dict]:
    index = {bit: i for i, bit in enumerate(compiled.qubits)}
    cl_index = {bit: i for i, bit in enumerate(compiled.clbits)}
    num_qubits = compiled.num_qubits
    next_free = [0] * num_qubits
    level: dict = {}
    running = 0
    ops = list(BASE_OPS)
    op_map = {name: i for i, name in enumerate(ops)}
    arrays = {"op": [], "qs": [], "par": [], "col": [], "depth": [], "mv": []}
    for i, instruction in enumerate(compiled.data):
        operation = instruction.operation
        name = operation.name
        if name not in op_map:
            op_map[name] = len(ops)
            ops.append(name)
        qubits = [index[q] for q in instruction.qubits]
        clbits = [cl_index[c] for c in instruction.clbits]

        if qubits:
            span = range(min(qubits), max(qubits) + 1)
            column = max(next_free[w] for w in span)
            for w in span:
                next_free[w] = column + 1
        else:
            column = max(next_free) if next_free else 0

        if name not in ("move", "barrier"):
            keys = [("q", q) for q in qubits] + [("c", c) for c in clbits]
            lvl = 1 + max((level.get(key, 0) for key in keys), default=0)
            for key in keys:
                level[key] = lvl
            running = max(running, lvl)

        parameter = None
        params = getattr(operation, "params", ())
        if name != "move" and params and isinstance(params[0], numbers.Real) and not isinstance(params[0], bool):
            value = float(params[0])
            if math.isfinite(value):
                parameter = round(value, 5)

        arrays["op"].append(op_map[name])
        arrays["qs"].append(qubits)
        arrays["par"].append(parameter)
        arrays["col"].append(column)
        arrays["depth"].append(running)
        arrays["mv"].append(move_of_index.get(i, -1))
    return ops, arrays


# ---------------------------------------------------------------------- move arrays


def _move_arrays(trace, loading, arch, zone_at) -> tuple[list, list]:
    pos = dict(loading)
    warnings: list[str] = []
    warning_index: dict[str, int] = {}
    moves = []
    for record in trace:
        qubits = list(record.qubits)
        before = {q: pos.get(q, (0, 0)) for q in qubits}
        after = {q: record.positions[q] for q in qubits}
        before_sites = [before[q] for q in qubits]
        after_sites = [after[q] for q in qubits]
        f: list[float] = []
        t: list[float] = []
        ts: list[int] = []
        for q in qubits:
            px, py = _phys(before[q], arch, zone_at)
            f.extend([round(px, 4), round(py, 4)])
            px, py = _phys(after[q], arch, zone_at)
            t.extend([round(px, 4), round(py, 4)])
            ts.extend([int(after[q][0]), int(after[q][1])])
        pos.update(record.positions)

        w: list[int] = []
        for message in record.spacing_warnings:
            key = str(message)
            if key not in warning_index:
                warning_index[key] = len(warnings)
                warnings.append(key)
            w.append(warning_index[key])

        rows = [int(v) for v in record.move.rows]
        cols = [int(v) for v in record.move.cols]
        nrows = [int(v) for v in record.move.new_rows]
        ncols = [int(v) for v in record.move.new_cols]

        if len(nrows) == len(rows):
            row_pairs = list(zip(rows, nrows))
        else:
            row_pairs = sorted({(before[q][1], after[q][1]) for q in qubits})
        if len(ncols) == len(cols):
            col_pairs = list(zip(cols, ncols))
        else:
            col_pairs = sorted({(before[q][0], after[q][0]) for q in qubits})

        def line_phys(sites, fallback, line: int, axis: str) -> float:
            for site in sites:
                if (site[1] if axis == "y" else site[0]) == line:
                    zone = zone_at.get(tuple(site))
                    if zone is not None:
                        return physical(zone, line, axis)
            zone = zone_at.get(tuple(fallback))
            if zone is not None:
                return physical(zone, line, axis)
            return float(line)

        br = [
            [
                round(line_phys(before_sites, before_sites[0], b, "y"), 4),
                round(line_phys(after_sites, after_sites[0], a, "y"), 4),
            ]
            for b, a in row_pairs
        ]
        bc = [
            [
                round(line_phys(before_sites, before_sites[0], b, "x"), 4),
                round(line_phys(after_sites, after_sites[0], a, "x"), 4),
            ]
            for b, a in col_pairs
        ]

        moves.append(
            {
                "i": record.index,
                "q": qubits,
                "f": f,
                "t": t,
                "ts": ts,
                "d": record.distance,
                "s": record.step,
                "w": w,
                "rows": rows,
                "cols": cols,
                "nrows": nrows,
                "ncols": ncols,
                "br": br,
                "bc": bc,
                "rm": [[int(b), int(a)] for b, a in row_pairs],
                "cm": [[int(b), int(a)] for b, a in col_pairs],
            }
        )
    return moves, warnings


# ---------------------------------------------------------------------- totals


def _partial_totals(trace, depth: int, num_operations: int, cz: int) -> dict:
    step_times: dict[int, float] = {}
    spacing = 0
    for record in trace:
        step_times[record.step] = max(step_times.get(record.step, 0.0), record.time)
        spacing += len(record.spacing_warnings)
    return {
        "num_moves": len(trace),
        "move_steps": len(step_times),
        "move_distance": round(sum(r.distance for r in trace), 6),
        "move_time": round(sum(step_times.values()), 6),
        "depth": depth,
        "num_operations": num_operations,
        "spacing_warnings": spacing,
        "cz": cz,
        "partial": True,
    }


def _official_totals(metrics, cz: int, spacing_warnings: int) -> dict:
    return {
        "num_moves": metrics.num_moves,
        "move_steps": metrics.move_steps,
        "move_distance": round(metrics.move_distance, 6),
        "move_time": round(metrics.move_time, 6),
        "depth": metrics.depth,
        "num_operations": metrics.num_operations,
        "spacing_warnings": spacing_warnings,
        "cz": cz,
        "partial": False,
    }


# ---------------------------------------------------------------------- build


def _regex_index(pattern: str, message: str) -> int | None:
    match = re.search(pattern, message)
    return int(match.group(1)) if match else None


def _harness_message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}\n{traceback.format_exc()[-1500:]}"


def _violation_list(errors) -> list[dict]:
    out = []
    for error in list(errors)[:500]:
        out.append(
            {
                "i": _regex_index(r"Instruction (\d+)", str(error)),
                "kind": getattr(error, "kind", "move"),
                "msg": str(error)[:1000],
            }
        )
    return out


def build(compiler_path: str | Path, qasm_path: str | Path, name: str, mode: str, timeout: float = 300):
    """Compile `qasm_path`, replay it and return `(data, summary)` (see docs/viz.md)."""
    compiler_path = Path(compiler_path).resolve()
    qasm_path = Path(qasm_path).resolve()
    bench_label = benchmark_label(qasm_path)
    label = bench_label or qasm_path.stem
    family = bench._family(qasm_path) if bench_label else "other"
    zone_list, zone_at = _zone_map()
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    data = {
        "version": VIZ_VERSION,
        "meta": {
            "name": name,
            "compiler": str(compiler_path),
            "circuit": label,
            "qasm": _qasm_relative(qasm_path),
            "family": family,
            "num_qubits": 0,
            "num_clbits": 0,
            "generated": generated,
            "compile_s": 0.0,
            "mode": mode,
            "warnings": [],
        },
        "geometry": _geometry(),
        "loading": None,
        "ops": list(BASE_OPS),
        "inst": {"op": [], "qs": [], "par": [], "col": [], "depth": [], "mv": []},
        "moves": [],
        "warnings": [],
        "violations": [],
        "error": None,
        "totals": {
            "num_moves": 0,
            "move_steps": 0,
            "move_distance": 0.0,
            "move_time": 0.0,
            "depth": 0,
            "num_operations": 0,
            "spacing_warnings": 0,
            "cz": 0,
            "partial": True,
        },
        "naive": _naive_for(bench_label),
        "equivalence": "skipped (fast)" if mode == "fast" else "not checked",
        "physics": {"ok": None, "segments": None, "msg": None},  # None = not checked
        "regen": {"compiler": str(compiler_path), "circuit": bench_label},
    }

    try:
        reference = QuantumCircuit.from_qasm_file(str(qasm_path))
    except Exception as error:  # noqa: BLE001 - the page must still be produced
        data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}
        return data, _summary(data)

    data["meta"]["num_qubits"] = reference.num_qubits
    data["meta"]["num_clbits"] = reference.num_clbits

    t0 = time.perf_counter()
    try:
        from pushquantum_validator._sandbox import run_compiler_process
        from pushquantum_validator.spec import load_trusted_spec

        compiled, loading = run_compiler_process(
            reference, load_trusted_spec(), f"{compiler_path}:compile_circuit", timeout, 4096, network=False
        )
    except CompilerOutputError as error:
        data["meta"]["compile_s"] = round(time.perf_counter() - t0, 3)
        data["error"] = {"stage": "compile", "msg": str(error)[:2000], "i": None}
        data["equivalence"] = "skipped (compile failed)"
        return data, _summary(data)
    except Exception as error:  # noqa: BLE001
        data["meta"]["compile_s"] = round(time.perf_counter() - t0, 3)
        data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}
        return data, _summary(data)
    data["meta"]["compile_s"] = round(time.perf_counter() - t0, 3)

    try:
        gates = strip_moves(compiled)
        depth = gates.depth()
        num_operations = gates.size()
    except Exception as error:  # noqa: BLE001
        data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}
        return data, _summary(data)
    cz_count = sum(1 for instruction in compiled.data if instruction.operation.name == "cz")

    try:
        arch = ValidationArchitecture.from_loading(loading)
    except ArchitectureError:
        arch = None

    movement = None
    trace: list = []
    loading_failed = False
    try:
        movement = verify_movement(compiled, loading, log_file=os.devnull)
        trace = movement.trace
    except ArchitectureError as error:
        partial_errors = [error]
        if arch is not None:
            try:
                trace, partial_errors = arch._replay(compiled, [])
            except ArchitectureError as replay_error:
                trace = []
                partial_errors = [replay_error]
                loading_failed = True
            else:
                loading_failed = not partial_errors
        else:
            loading_failed = True
        if not partial_errors:
            partial_errors = [error]
        first = partial_errors[0]
        data["violations"] = _violation_list(partial_errors)
        if loading_failed:
            if error.kind != "loading" and partial_errors and getattr(partial_errors[0], "kind", "") == "loading":
                first = partial_errors[0]
            data["error"] = {"stage": "loading", "msg": str(first)[:2000], "i": None}
        else:
            data["error"] = {
                "stage": "movement",
                "msg": str(first)[:2000],
                "i": _regex_index(r"Instruction (\d+)", str(first)),
            }

    if movement is not None:
        try:
            metrics = compute_metrics(compiled, movement, gates)
            totals = _official_totals(metrics, cz_count, movement.spacing_warnings)
        except Exception as error:  # noqa: BLE001
            totals = _partial_totals(trace, depth, num_operations, cz_count)
            if data["error"] is None:
                data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}
    else:
        totals = _partial_totals(trace, depth, num_operations, cz_count)
    data["totals"] = totals

    if movement is not None:
        try:
            segments = bench.physics_check(compiled, loading, movement.trace)
            data["physics"] = {"ok": True, "segments": segments, "msg": None}
        except bench.PhysicsError as error:
            message = str(error)
            data["physics"] = {"ok": False, "segments": None, "msg": message[:2000]}
            if data["error"] is None:
                data["error"] = {
                    "stage": "physics",
                    "msg": message[:2000],
                    "i": _regex_index(r"instruction (\d+)", message),
                }
        except Exception as error:  # noqa: BLE001
            if data["error"] is None:
                data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}

    if mode == "full":
        try:
            result = verify_equivalence(compiled, reference, gates)
            if result is None:
                data["equivalence"] = "interaction pairs only (> 19 qubits)"
            else:
                data["equivalence"] = str(result.equivalence).split(".")[-1]
        except NotEquivalentError as error:
            data["equivalence"] = "FAILED: " + str(error)
            if data["error"] is None:
                data["error"] = {"stage": "equivalence", "msg": str(error)[:2000], "i": None}
        except Exception as error:  # noqa: BLE001
            data["equivalence"] = "error: " + type(error).__name__
            if data["error"] is None:
                data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}

    move_of_index = {record.index: k for k, record in enumerate(trace)}

    try:
        ops, arrays = _instruction_arrays(compiled, move_of_index)
        data["ops"] = ops
        data["inst"] = arrays
    except Exception as error:  # noqa: BLE001
        if data["error"] is None:
            data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}

    try:
        moves, warnings = _move_arrays(trace, loading, arch, zone_at)
        data["moves"] = moves
        data["warnings"] = warnings
    except Exception as error:  # noqa: BLE001
        if data["error"] is None:
            data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}

    try:
        data["loading"] = _loading_positions(loading, reference.num_qubits, arch, zone_at)
    except Exception as error:  # noqa: BLE001
        if data["error"] is None:
            data["error"] = {"stage": "harness", "msg": _harness_message(error)[:2000], "i": None}

    # Internal consistency checks: never fatal, only reported.
    if movement is not None and data["inst"]["depth"]:
        depths = data["inst"]["depth"]
        if depths[-1] != totals["depth"]:
            message = f"depth mismatch: inst {depths[-1]} vs metrics {totals['depth']}"
            data["meta"]["warnings"].append(message)
            print("WARNING " + message, file=sys.stderr, flush=True)
        distance = sum(record.distance for record in trace)
        if abs(distance - totals["move_distance"]) > 1e-6:
            message = f"distance mismatch: trace {distance:.6f} vs metrics {totals['move_distance']:.6f}"
            data["meta"]["warnings"].append(message)
            print("WARNING " + message, file=sys.stderr, flush=True)

    return data, _summary(data)


def _summary(data: dict) -> dict:
    error = data["error"]
    totals = data["totals"]
    official = not totals.get("partial", True)
    metrics = None
    if official:
        metrics = {
            "num_moves": totals["num_moves"],
            "move_steps": totals["move_steps"],
            "move_distance": totals["move_distance"],
            "move_time": totals["move_time"],
            "depth": totals["depth"],
            "num_operations": totals["num_operations"],
            "spacing_warnings": totals["spacing_warnings"],
        }
    return {
        "name": data["meta"]["name"],
        "label": data["meta"]["circuit"],
        "family": data["meta"]["family"],
        "ok": error is None,
        "stage": error["stage"] if error else None,
        "error": error["msg"] if error else None,
        "error_i": error["i"] if error else None,
        "metrics": metrics,
        "totals": totals,
        "cz": totals["cz"],
        "compile_s": data["meta"]["compile_s"],
        "equivalence": data["equivalence"],
        "physics": data["physics"],
        "page_bytes": None,
    }


# ---------------------------------------------------------------------- writing pages


def _template_text() -> str:
    path = ROOT / "bench" / "viz_page.html"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return FALLBACK_TEMPLATE


def write_outputs(data: dict, summary: dict, out: Path) -> tuple[dict, Path]:
    """Write `viz/<name>/<label>.html` and the sidecar JSON; return the updated summary and path."""
    name = data["meta"]["name"]
    label = data["meta"]["circuit"]
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    page_path = directory / f"{label}.html"
    sidecar_path = directory / f"{label}.json"

    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = _template_text()
    if template.count("__VIZ_DATA__") != 1:
        print(
            f"WARNING: template {ROOT / 'bench' / 'viz_page.html'} has "
            f"{template.count('__VIZ_DATA__')} __VIZ_DATA__ placeholders (expected 1)",
            file=sys.stderr,
            flush=True,
        )
    if "__VIZ_MANIFEST__" in template:
        try:
            snapshot = manifest_snapshot(build_manifest(out), name, label)
        except Exception as error:  # noqa: BLE001 - the page itself matters more than the sidebar
            print(f"WARNING: manifest snapshot failed: {error}", file=sys.stderr, flush=True)
            snapshot = {"version": 1, "snapshot": True, "pages": [], "circuits": {}, "compilers": []}
        template = template.replace("__VIZ_MANIFEST__", _manifest_json(snapshot), 1)
    page_text = template.replace("__VIZ_DATA__", payload, 1)
    if "__VIZ_TITLE__" in page_text:
        page_text = page_text.replace("__VIZ_TITLE__", html.escape(f"{name} · {label}"))
    page_path.write_text(page_text, encoding="utf-8")
    page_bytes = len(page_text.encode("utf-8"))

    error = data["error"]
    official = not data["totals"].get("partial", True)
    sidecar = {
        "name": name,
        "label": label,
        "family": data["meta"]["family"],
        "qubits": data["meta"]["num_qubits"],
        "compiler": data["meta"]["compiler"],
        "ok": summary["ok"],
        "stage": error["stage"] if error else None,
        "error": error["msg"][:300] if error else None,
        "metrics": summary["metrics"] if official else None,
        "cz": data["totals"]["cz"],
        "naive": data["naive"],
        "equivalence": data["equivalence"],
        "compile_s": data["meta"]["compile_s"],
        "page": f"{label}.html",
        "page_bytes": page_bytes,
        "generated": data["meta"]["generated"],
    }
    if not official:
        sidecar["partial"] = data["totals"]
    sidecar_path.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")

    updated = dict(summary)
    updated["page_bytes"] = page_bytes
    updated["page_path"] = str(page_path)
    return updated, page_path


# ---------------------------------------------------------------------- index data


def collect_index(out: Path) -> list[dict]:
    """Load every sidecar under `out` and group it per compiler, with bench scores when possible."""
    out = Path(out)
    sidecars = []
    if out.is_dir():
        for path in sorted(out.glob("*/*.json")):
            try:
                sidecar = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(sidecar, dict):
                sidecars.append(sidecar)

    groups: dict[str, list[dict]] = {}
    for sidecar in sidecars:
        name = sidecar.get("name")
        if not name:
            continue
        if is_excluded(name):
            continue
        groups.setdefault(str(name), []).append(sidecar)

    metric_labels = {label for label in _naive_baseline() if label.startswith("metric_")}
    result = []
    for name in sorted(groups, key=lambda value: (value != "naive", value)):
        rows = groups[name]
        rows.sort(key=lambda row: (0 if str(row.get("label", "")).startswith("metric_") else 1,
                                   str(row.get("label", ""))))
        present = {row.get("label") for row in rows}
        summary = None
        if metric_labels and metric_labels.issubset(present):
            records = [
                {
                    "label": row.get("label"),
                    "family": row.get("family"),
                    "ok": bool(row.get("ok")),
                    "metrics": row.get("metrics"),
                }
                for row in rows
                if row.get("label") in metric_labels
            ]
            summary = bench.score(records)
        compilers = sorted({str(row.get("compiler")) for row in rows if row.get("compiler")})
        result.append(
            {
                "name": name,
                "compiler": compilers[0] if compilers else None,
                "summary": summary,
                "rows": rows,
            }
        )
    return result


def render_index(groups: list[dict]) -> str:
    """The index page: `bench/viz_index.html` with the groups embedded as JSON (rendered client-side)."""
    payload = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "groups": groups}
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = (ROOT / "bench" / "viz_index.html").read_text(encoding="utf-8")
    return template.replace("__VIZ_INDEX__", text, 1)


def _circuit_info(label: str) -> dict:
    """Family, size and variant of a bench label, from its qasm file name."""
    path = known_circuits().get(label)
    stem = path.stem if path else label
    variant = stem[len("bus_stop_"):] if stem.startswith("bus_stop_") else stem
    layers = re.search(r"_p(\d+)$", stem)
    size = re.search(r"(\d+)q", variant)
    return {
        "family": bench._family(path) if path else "other",
        "variant": variant.replace("_", " "),
        "layers": int(layers.group(1)) if layers else None,
        "qubits": int(size.group(1)) if size else None,
        "set": "metric" if label.startswith("metric_") else "sanity" if label.startswith("sample_") else "other",
    }


def build_manifest(out: Path, groups: list[dict] | None = None) -> dict:
    """Every compiler x circuit page under `out` with its metrics (see docs/viz.md, "manifest")."""
    groups = collect_index(out) if groups is None else groups
    naive = _naive_baseline()
    metric = [k for k in naive if k.startswith("metric_")]
    circuits: dict[str, dict] = {}
    pages = []
    compilers = []
    for group in groups:
        summary = group.get("summary") or {}
        compilers.append({
            "name": group["name"],
            "compiler": group.get("compiler"),
            "pages": len(group["rows"]),
            "valid": sum(1 for r in group["rows"] if r.get("ok")),
            "score": summary.get("score") if summary.get("valid") else None,
        })
        for row in group["rows"]:
            label = str(row.get("label"))
            if label not in circuits:
                info = _circuit_info(label)
                if row.get("qubits"):
                    info["qubits"] = row["qubits"]
                base = naive.get(label, {}).get("metrics")
                info["naive_d"] = base["move_distance"] if base else None
                info["naive_p"] = base["depth"] if base else None
                circuits[label] = info
            metrics = row.get("metrics") or row.get("partial") or {}
            pages.append({
                "name": group["name"],
                "label": label,
                "ok": bool(row.get("ok")),
                "stage": row.get("stage"),
                "partial": row.get("metrics") is None,
                "d": metrics.get("move_distance"),
                "p": metrics.get("depth"),
                "moves": metrics.get("num_moves"),
                "warn": metrics.get("spacing_warnings"),
                "cz": row.get("cz"),
                "equivalence": row.get("equivalence"),
                "page": f"{group['name']}/{row.get('page') or label + '.html'}",
                "bytes": row.get("page_bytes"),
                "generated": row.get("generated"),
            })
    return {
        "version": 1,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weights": {
            "distance": bench.SCORE_DISTANCE_WEIGHT,
            "depth": bench.SCORE_DEPTH_WEIGHT,
            "naive_distance_total": round(sum(naive[k]["metrics"]["move_distance"] for k in metric), 6),
            "naive_depth_total": sum(naive[k]["metrics"]["depth"] for k in metric),
        },
        "compilers": compilers,
        "circuits": circuits,
        "pages": pages,
    }


def manifest_snapshot(manifest: dict, name: str, label: str) -> dict:
    """The part of the manifest one page needs: its circuit's compilations and its compiler's circuits."""
    pages = [p for p in manifest["pages"] if p["name"] == name or p["label"] == label]
    labels = {p["label"] for p in pages} | {label}
    return {**manifest, "snapshot": True, "pages": pages,
            "circuits": {k: v for k, v in manifest["circuits"].items() if k in labels}}


_MANIFEST_RE = re.compile(r'(<script id="viz-manifest" type="application/json">)(.*?)(</script>)', re.S)


def _manifest_json(snapshot: dict) -> str:
    return json.dumps(snapshot, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def refresh_snapshots(out: Path, manifest: dict, name: str | None = None, label: str | None = None) -> int:
    """Re-embed the manifest snapshot in existing pages (all, or those of `name` or of `label`)."""
    count = 0
    for page in sorted(Path(out).glob("*/*.html")):
        page_name, page_label = page.parent.name, page.stem
        if (name or label) and page_name != name and page_label != label:
            continue
        text = page.read_text(encoding="utf-8")
        match = _MANIFEST_RE.search(text)
        if not match:
            continue
        snap = _manifest_json(manifest_snapshot(manifest, page_name, page_label))
        page.write_text(text[:match.start(2)] + snap + text[match.end(2):], encoding="utf-8")
        count += 1
    return count


def regenerate_index(out: Path, refresh: str | tuple | None = None) -> Path:
    """Rewrite `index.html` and `manifest.json`; `refresh="all"` or `(name, label)` also updates the
    manifest snapshot embedded in existing pages (used by file:// viewers)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = collect_index(out)
    manifest = build_manifest(out, groups)
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(out / "manifest.json")
    if refresh == "all":
        refresh_snapshots(out, manifest)
    elif refresh:
        refresh_snapshots(out, manifest, *refresh)
    text = render_index(groups)
    path = out / "index.html"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------- CLI helpers


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _summary_line(summary: dict) -> str:
    size = _format_size(summary["page_bytes"] or 0)
    if summary["ok"]:
        metrics = summary["metrics"] or {}
        baseline = _naive_baseline().get(summary["label"], {}).get("metrics")
        relative = ""
        if baseline:
            distance = metrics.get("move_distance", 0.0)
            depth = metrics.get("depth", 0)
            relative = (
                f" dist {distance:.1f} ({(distance / baseline['move_distance'] - 1) * 100:+.1f}%)"
                f" depth {depth} ({(depth / baseline['depth'] - 1) * 100:+.1f}%)"
            )
        return (
            f"PASS {summary['label']:<18}{relative} "
            f"moves {metrics.get('num_moves')} cz {summary['cz']} [{summary['compile_s']}s] {size}"
        )
    return (
        f"FAIL {summary['label']:<18} {summary['stage']}: "
        f"{(summary['error'] or '')[:200]} {size}"
    )


def _worker(args: tuple) -> dict:
    compiler_path, qasm_path, name, mode, timeout, out = args
    data, summary = build(compiler_path, qasm_path, name, mode, timeout)
    summary, _ = write_outputs(data, summary, Path(out))
    return summary


# ---------------------------------------------------------------------- server


class VizRequestHandler(SimpleHTTPRequestHandler):
    """Static server with a no-store cache policy and the regenerate API."""

    output_dir: Path = ROOT / "viz"

    def end_headers(self) -> None:  # noqa: D102
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: D102
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: D102
        if self.path.split("?", 1)[0] == "/api/ping":
            self._send_json(200, {"ok": True})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: D102
        if self.path.split("?", 1)[0] != "/api/regenerate":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"ok": False, "error": "bad Content-Length"})
            return
        if length <= 0 or length > 10000:
            self._send_json(400, {"ok": False, "error": "bad body size"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"ok": False, "error": "invalid JSON"})
            return
        if not isinstance(body, dict):
            self._send_json(400, {"ok": False, "error": "expected a JSON object"})
            return

        compiler = body.get("compiler")
        label = body.get("circuit")
        mode = body.get("mode", "fast")

        if not isinstance(compiler, str) or not compiler:
            self._send_json(403, {"ok": False, "error": "compiler path is required"})
            return
        try:
            resolved = Path(compiler).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            self._send_json(403, {"ok": False, "error": "compiler path does not exist"})
            return
        if is_excluded(resolved) or is_excluded(compiler_name(resolved)):
            self._send_json(403, {"ok": False, "error": f"compiler is excluded from the visualisation ({', '.join(sorted(EXCLUDED))})"})
            return
        if not _compiler_allowed(resolved):
            self._send_json(403, {"ok": False, "error": "compiler path is not allowed"})
            return
        if not isinstance(label, str) or label not in known_circuits():
            self._send_json(400, {"ok": False, "error": "unknown circuit label"})
            return
        if mode not in ("fast", "full"):
            self._send_json(400, {"ok": False, "error": "mode must be fast or full"})
            return
        if not REGEN_LOCK.acquire(blocking=False):
            self._send_json(409, {"ok": False, "error": "a regeneration is already running"})
            return
        try:
            url = f"/{compiler_name(resolved)}/{label}.html"
            command = [
                sys.executable,
                str(ROOT / "bench" / "viz.py"),
                "page",
                str(resolved),
                "--circuit",
                label,
                "--mode",
                mode,
                "--out",
                str(self.output_dir),
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=900, cwd=str(ROOT)
                )
                log = (completed.stdout or "") + (completed.stderr or "")
                ok = completed.returncode == 0
            except subprocess.TimeoutExpired as error:
                log = f"regeneration timed out after 900 s: {error}"
                ok = False
            except OSError as error:
                log = f"could not start regeneration: {error}"
                ok = False
            self._send_json(200, {"ok": ok, "url": url, "log": log[-2000:]})
        finally:
            REGEN_LOCK.release()


REGEN_LOCK = threading.Lock()


def _compiler_allowed(path: Path) -> bool:
    if not (path.suffix == ".py" and path.is_file()):
        return False
    if path.is_relative_to(CHALLENGE_COMPILERS):
        return True
    try:
        relative = path.relative_to(WT_ROOT)
    except ValueError:
        return False
    return len(relative.parts) >= 4 and relative.parts[1] == "compilers"


# ---------------------------------------------------------------------- commands


def cmd_page(args) -> int:
    compiler = Path(args.compiler).resolve()
    if not compiler.is_file():
        raise SystemExit(f"no such compiler file: {compiler}")
    qasm, bench_label = resolve_circuit(args.circuit)
    name = sanitize_name(args.name) if args.name else compiler_name(compiler)
    if (is_excluded(compiler) or is_excluded(name)) and not args.force:
        raise SystemExit(f"{compiler} is excluded from the visualisation (EXCLUDED = {sorted(EXCLUDED)}); use --force")
    out = output_dir(args.out)
    data, summary = build(compiler, qasm, name, args.mode, args.timeout)
    summary, page_path = write_outputs(data, summary, out)
    regenerate_index(out, refresh=(name, data["meta"]["circuit"]))
    print(f"{page_path} ({_format_size(summary['page_bytes'])})")
    print(_summary_line(summary))
    for warning in data["meta"]["warnings"]:
        print("WARNING " + warning, file=sys.stderr)
    return 0 if summary["ok"] else 1


def cmd_all(args) -> int:
    compiler = Path(args.compiler).resolve()
    if not compiler.is_file():
        raise SystemExit(f"no such compiler file: {compiler}")
    name = sanitize_name(args.name) if args.name else compiler_name(compiler)
    if (is_excluded(compiler) or is_excluded(name)) and not args.force:
        raise SystemExit(f"{compiler} is excluded from the visualisation (EXCLUDED = {sorted(EXCLUDED)}); use --force")
    out = output_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    if args.bset in ("sanity", "all"):
        paths += sanity_benchmarks()
    if args.bset in ("metric", "all"):
        paths += metric_benchmarks()
    if args.filter:
        paths = [p for p in paths if args.filter in (benchmark_label(p) or "") or args.filter in str(p)]
    if not paths:
        raise SystemExit("no benchmarks selected")

    naive = _naive_baseline()
    t0 = time.perf_counter()
    summaries: list[dict] = []
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_worker, (str(compiler), str(path), name, args.mode, args.timeout, str(out))): path
            for path in paths
        }
        for future in cf.as_completed(futures):
            path = futures[future]
            try:
                summary = future.result()
            except Exception as error:  # noqa: BLE001 - one circuit must not stop the run
                label = benchmark_label(path) or path.stem
                summary = {
                    "name": name,
                    "label": label,
                    "family": bench._family(path) if benchmark_label(path) else "other",
                    "ok": False,
                    "stage": "harness",
                    "error": _harness_message(error),
                    "error_i": None,
                    "metrics": None,
                    "totals": {},
                    "cz": 0,
                    "compile_s": 0.0,
                    "equivalence": None,
                    "physics": None,
                    "page_bytes": 0,
                }
            summaries.append(summary)
            print(_summary_line(summary), flush=True)

    summaries.sort(key=lambda item: str(item["label"]))
    records = [
        {
            "label": item["label"],
            "family": item["family"],
            "ok": bool(item["ok"]),
            "metrics": item["metrics"],
        }
        for item in summaries
    ]
    result = bench.score(records)
    elapsed = round(time.perf_counter() - t0, 1)
    failed = [item["label"] for item in summaries if not item["ok"]]

    index_path = regenerate_index(out, refresh="all")
    print("-" * 80)
    print(
        f"{name}: {len(summaries) - len(failed)}/{len(summaries)} passed in {elapsed}s ({args.mode})"
    )
    if failed:
        print("FAILED:", ", ".join(failed))
    if result.get("valid"):
        tag = "OFFICIAL SCORE" if result["official"] else f"partial score ({result['n']} metric circuits)"
        print(
            f"{tag}: {result['score']:+.2f}%  (distance {result['distance']:.1f} vs "
            f"{result['naive_distance']:.1f}: {result['distance_improvement']:+.2f}%; "
            f"depth {result['depth']} vs {result['naive_depth']}: {result['depth_improvement']:+.2f}%)"
        )
        for family, values in result["families"].items():
            print(
                f"  {family:<5} distance {values['distance_improvement']:+.2f}%  "
                f"depth {values['depth_improvement']:+.2f}%"
            )
    elif result.get("n"):
        print("SCORE: not computed (a metric benchmark failed)")
    print(f"index: {index_path}")
    return 1 if failed else 0


def cmd_index(args) -> int:
    out = output_dir(args.out)
    path = regenerate_index(out, refresh="all")
    groups = collect_index(out)
    print(f"{path} ({len(groups)} compiler(s))")
    return 0


_DATA_RE = re.compile(r'<script id="viz-data" type="application/json">(.*?)</script>', re.S)


def cmd_rewrap(args) -> int:
    """Re-embed the data of every existing page into the current `bench/viz_page.html` (no recompile)."""
    out = output_dir(args.out)
    template = _template_text()
    count = 0
    for page in sorted(out.glob("*/*.html")):
        text = page.read_text(encoding="utf-8")
        match = _DATA_RE.search(text)
        if not match:
            continue
        data = json.loads(match.group(1).replace("<\\/", "</"))
        payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
        title = html.escape(f"{data['meta']['name']} · {data['meta']['circuit']}")
        text = template.replace("__VIZ_MANIFEST__", "{}", 1)
        page.write_text(text.replace("__VIZ_DATA__", payload, 1).replace("__VIZ_TITLE__", title), encoding="utf-8")
        count += 1
    regenerate_index(out, refresh="all")
    print(f"rewrapped {count} page(s) in {out}")
    return 0


def cmd_serve(args) -> int:
    out = output_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "index.html").is_file() or not (out / "manifest.json").is_file():
        regenerate_index(out)
    VizRequestHandler.output_dir = out
    handler = functools.partial(VizRequestHandler, directory=str(out))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    print(f"serving {out} on http://{args.host}:{args.port}/", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", file=sys.stderr, flush=True)
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    page = sub.add_parser("page", help="build one page")
    page.add_argument("compiler")
    page.add_argument("--circuit", required=True, help="bench label or path to a .qasm file")
    page.add_argument("--name", default=None)
    page.add_argument("--mode", choices=["fast", "full"], default="fast")
    page.add_argument("--timeout", type=float, default=300)
    page.add_argument("--out", default=None)
    page.add_argument("--force", action="store_true", help="build pages even for an excluded compiler")
    page.set_defaults(func=cmd_page)

    every = sub.add_parser("all", help="build every page")
    every.add_argument("compiler")
    every.add_argument("--set", dest="bset", choices=["metric", "sanity", "all"], default="all")
    every.add_argument("--filter", default=None)
    every.add_argument("--jobs", type=int, default=2)
    every.add_argument("--name", default=None)
    every.add_argument("--mode", choices=["fast", "full"], default="fast")
    every.add_argument("--timeout", type=float, default=300)
    every.add_argument("--out", default=None)
    every.add_argument("--force", action="store_true", help="build pages even for an excluded compiler")
    every.set_defaults(func=cmd_all)

    index = sub.add_parser("index", help="regenerate the index")
    index.add_argument("--out", default=None)
    index.set_defaults(func=cmd_index)

    rewrap = sub.add_parser("rewrap", help="re-embed existing pages into the current page template")
    rewrap.add_argument("--out", default=None)
    rewrap.set_defaults(func=cmd_rewrap)

    serve = sub.add_parser("serve", help="serve the output directory")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=9000)
    serve.add_argument("--out", default=None)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
