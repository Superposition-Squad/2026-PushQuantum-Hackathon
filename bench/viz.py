"""Data extraction + CLI for the step-by-step compiled-circuit visualiser.

Extracts, with the *official* validator (so the page shows exactly what `bench/bench.py` scores),
everything needed to animate a compilation, and writes self-contained HTML pages.

Usage (from the repo root):
    python bench/viz.py page <compiler.py[:fn]> --circuit <label|path> [-o out.html] [--json out.json]
    python bench/viz.py all  <compiler.py[:fn]> [--set metric|sanity|all] [--jobs N] [--name NAME]
    python bench/viz.py index
    python bench/viz.py serve [--port 9000] [--host 127.0.0.1]

Output tree (gitignored): viz/<name>/<label>.html, viz/<name>/meta.json, viz/index.html,
viz/status.json (written while `serve /regen` runs).  Rendering lives in `bench/viz_page.py`
(`render_page`, `render_index`); minimal fallbacks keep this tool usable without it.

The JSON embedded in every page follows the "Schema" section of docs/ideas/viz.md and is
authoritative: field names there must be reproduced exactly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import functools
import html
import http.server
import json
import logging
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from numbers import Real
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import bench` works

from bench import PhysicsError, _family, physics_check  # noqa: E402
from pushquantum_validator._validation import (  # noqa: E402
    ValidatedMove,
    ValidationArchitecture,
    physical,
)
from pushquantum_validator.errors import ArchitectureError, CompilerOutputError  # noqa: E402
from pushquantum_validator.spec import parse_spec  # noqa: E402
from pushquantum_validator.verify import (  # noqa: E402
    benchmark_label,
    compute_metrics,
    metric_benchmarks,
    naive_baseline,
    run_compiler,
    sanity_benchmarks,
    strip_moves,
    verify_movement,
)

VIZ = ROOT / "viz"
CHALLENGE_WT = Path("/home/hackathon/challenge-wt")
SCHEMA = 1
CODES = {"rz": 0, "rx": 1, "cz": 2, "move": 3, "measure": 4, "barrier": 5}
IDENTITIES = frozenset({"move", "barrier"})
SLUG = re.compile(r"^[A-Za-z0-9._-]+$")
FUNCTION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------- helpers


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _r6(value: float) -> float:
    return round(float(value), 6)


def _code(name: str) -> int:
    return CODES.get(name, 6)


@functools.lru_cache(maxsize=1)
def _naive() -> dict[str, dict[str, Any]]:
    return naive_baseline()


def _spec_geometry() -> dict[str, Any]:
    """Every trap of every zone, grid site + physical coordinates (docs/ideas/viz.md schema)."""
    zones, loading_zone = parse_spec(None)
    out: dict[str, Any] = {"loading_zone": loading_zone, "zones": []}
    for zone in zones.values():
        sites = [
            [gx, gy, _r6(physical(zone, gx, "x")), _r6(physical(zone, gy, "y"))]
            for gx in range(zone.x[0], zone.x[1] + 1)
            for gy in range(zone.y[0], zone.y[1] + 1)
        ]
        out["zones"].append(
            {
                "name": zone.name,
                "x": list(zone.x),
                "y": list(zone.y),
                "spacing": zone.spacing,
                "pair_axis": zone.pair_axis,
                "pair_distance": zone.pair_distance,
                "operations": sorted(zone.operations),
                "sites": sites,
            }
        )
    return out


def _hub_degree(circuit) -> tuple[int, list[int]]:
    """Interaction graph of the reference: degree per qubit and the hub (lowest index on a tie)."""
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    neighbours: list[set[int]] = [set() for _ in range(circuit.num_qubits)]
    found = False
    for instruction in circuit.data:
        if instruction.operation.name in IDENTITIES:
            continue
        qubits = [index[q] for q in instruction.qubits]
        if len(qubits) < 2:
            continue
        found = True
        for a in range(len(qubits)):
            for b in range(a + 1, len(qubits)):
                neighbours[qubits[a]].add(qubits[b])
                neighbours[qubits[b]].add(qubits[a])
    degree = [len(n) for n in neighbours]
    return (degree.index(max(degree)) if found else -1), degree


def _moves_from_trace(trace, loading: dict) -> list[dict[str, Any]]:
    positions = dict(loading)
    moves: list[dict[str, Any]] = []
    for record in trace:
        before = [list(positions[q]) for q in record.qubits]
        after = [list(record.positions[q]) for q in record.qubits]
        moves.append(
            {
                "i": record.index,
                "q": list(record.qubits),
                "a": after,
                "b": before,
                "rows": list(record.move.rows),
                "cols": list(record.move.cols),
                "nrows": list(record.move.new_rows),
                "ncols": list(record.move.new_cols),
                "d": float(record.distance),  # full precision: the page sums these and compares to metrics
                "s": record.step,
                "w": len(record.spacing_warnings),
            }
        )
        positions.update(record.positions)
    return moves


def _warnings(trace) -> list[list[Any]]:
    pairs = [[i, text] for i, record in enumerate(trace) for text in record.spacing_warnings]
    return pairs[:50]


def _partial_replay(compiled, loading: dict) -> tuple[list[dict], list[list[Any]], int | None, float]:
    """Mirror `_ValidationArchitecture._replay`, stopping at the first violation (error card path)."""
    arch = ValidationArchitecture.from_loading(loading, None)
    index_of = {bit: i for i, bit in enumerate(compiled.qubits)}
    moves: list[dict[str, Any]] = []
    warnings: list[list[Any]] = []
    error_index: int | None = None
    total = 0.0
    for index, instruction in enumerate(compiled.data):
        qubits = tuple(index_of[q] for q in instruction.qubits)
        try:
            if instruction.operation.name != "move":
                arch.check_gate(instruction.operation.name, qubits)
                continue
            move = ValidatedMove.from_instruction(instruction.operation)
            before = {q: arch.positions[q] for q in qubits}
            _mapping, spacing = arch.apply_move(move, qubits)
            after = {q: arch.positions[q] for q in qubits}
            distance = max(
                (math.dist(arch.physical_position(before[q]), arch.physical_position(after[q])) for q in qubits),
                default=0.0,
            )
        except ArchitectureError:
            error_index = index
            break
        for text in spacing:
            if len(warnings) < 50:
                warnings.append([len(moves), text])
        moves.append(
            {
                "i": index,
                "q": list(qubits),
                "a": [list(after[q]) for q in qubits],
                "b": [list(before[q]) for q in qubits],
                "rows": list(move.rows),
                "cols": list(move.cols),
                "nrows": list(move.new_rows),
                "ncols": list(move.new_cols),
                "d": float(distance),  # full precision (see _moves_from_trace)
                "s": len(moves),
                "w": len(spacing),
            }
        )
        total += distance
    return moves, warnings, error_index, total


def _layers(compiled, instructions, moves: list[dict]) -> dict[str, Any]:
    """ops, angles, col/ncols/colmove and depth_at for `instructions` (a prefix of `compiled.data`)."""
    qindex = {bit: i for i, bit in enumerate(compiled.qubits)}
    offset = len(compiled.qubits)
    cindex = {bit: offset + i for i, bit in enumerate(compiled.clbits)}
    nwire = offset + len(compiled.clbits)
    col = [-1] * nwire
    colkind: dict[int, str] = {}
    depth = [0] * nwire
    move_at = {m["i"]: i for i, m in enumerate(moves)}
    ops: list[list[int]] = []
    angles: list[float] = []
    cols: list[int] = []
    depth_at: list[int] = []
    for index, instruction in enumerate(instructions):
        name = instruction.operation.name
        code = _code(name)
        qubits = tuple(qindex[q] for q in instruction.qubits)
        wires = qubits + tuple(cindex[c] for c in instruction.clbits)
        angle = 0.0
        if code in (0, 1) and instruction.operation.params and isinstance(instruction.operation.params[0], Real):
            angle = round(float(instruction.operation.params[0]), 5)
        if code == 2 and len(qubits) >= 2:
            ops.append([2, qubits[0], qubits[1]])
        elif code == 3:
            ops.append([3, move_at.get(index, -1), -1])
        else:
            ops.append([code, qubits[0] if qubits else -1, -1])
        angles.append(angle)

        column = max((col[w] for w in wires), default=-1) + 1
        kind = "move" if code == 3 else "gate"
        while column in colkind and colkind[column] != kind:
            column += 1
        colkind[column] = kind
        for w in wires:
            col[w] = column
        cols.append(column)

        if name not in IDENTITIES:
            level = max((depth[w] for w in wires), default=0) + 1
            for w in wires:
                depth[w] = level
        depth_at.append(max(depth, default=0))

    ncols = max(cols) + 1 if cols else 0
    colmove = [0] * ncols
    for column, op in zip(cols, ops):
        if op[0] == 3:
            colmove[column] = 1
    return {
        "ops": ops,
        "angles": angles,
        "col": cols,
        "ncols": ncols,
        "colmove": colmove,
        "depth_at": depth_at,
    }


def _metrics_dict(metrics, warnings: int) -> dict[str, Any]:
    return {
        "num_moves": metrics.num_moves,
        "move_steps": metrics.move_steps,
        "move_distance": _r6(metrics.move_distance),
        "move_time": _r6(metrics.move_time),
        "depth": metrics.depth,
        "num_operations": metrics.num_operations,
        "spacing_warnings": warnings,
    }


def _meta_skeleton(compiler: str, label: str, path: Path, name: str) -> dict[str, Any]:
    return {
        "compiler": name,
        "compiler_path": compiler,
        "label": label,
        "file": str(path.relative_to(ROOT / "vendor" / "pushquantum_validator"))
        if path.is_relative_to(ROOT / "vendor" / "pushquantum_validator")
        else str(path),
        "family": _family(path),
        "qubits": None,
        "status": "compile_error",
        "error": None,
        "error_index": None,
        "violations": [],
        "physics_ok": None,
        "physics": "not checked (compile error)",
        "metrics": None,
        "naive": None,
        "compile_s": None,
        "generated": _now(),
    }


def build_data(compiler: str, qasm_path: Path, name: str, timeout: float = 300) -> dict[str, Any]:
    """Everything one page needs (schema 1 of docs/ideas/viz.md). Never raises for compiler output."""
    from qiskit import QuantumCircuit

    logging.getLogger("pushquantum_validator").setLevel(logging.ERROR)
    path = Path(qasm_path).resolve()
    label = benchmark_label(path) or path.stem
    data: dict[str, Any] = {
        "schema": SCHEMA,
        "meta": _meta_skeleton(compiler, label, path, name),
        "spec": _spec_geometry(),
        "loading": [],
        "hub": -1,
        "degree": [],
        "ops": [],
        "angles": [],
        "col": [],
        "ncols": 0,
        "colmove": [],
        "depth_at": [],
        "moves": [],
        "warnings": [],
    }
    meta = data["meta"]
    try:
        reference = QuantumCircuit.from_qasm_file(str(path))
        meta["qubits"] = reference.num_qubits
        data["hub"], data["degree"] = _hub_degree(reference)
    except Exception as error:  # unreadable file: an error card is the best we can do
        meta["error"] = f"{type(error).__name__}: {error}"[:2000]
        return data

    baseline = _naive().get(label)
    meta["naive"] = (
        {"move_distance": baseline["metrics"]["move_distance"], "depth": baseline["metrics"]["depth"]}
        if baseline
        else None
    )

    started = time.perf_counter()
    try:
        compiled, loading = run_compiler(reference, compiler, timeout=timeout, network=False)
    except CompilerOutputError as error:
        meta["error"] = f"CompilerOutputError: {error}"[:2000]
        return data
    except Exception as error:
        meta["error"] = f"{type(error).__name__}: {error}"[:2000]
        return data
    meta["compile_s"] = round(time.perf_counter() - started, 2)
    data["loading"] = [list(loading[q]) for q in range(len(loading))]

    movement = None
    try:
        movement = verify_movement(compiled, loading, log_file=os.devnull)
        gates = strip_moves(compiled)
        metrics = compute_metrics(compiled, movement, gates)
    except ArchitectureError as error:
        _build_invalid(data, compiled, loading, error)
        return data

    meta["status"] = "ok"
    meta["physics"] = "not checked"
    data["moves"] = _moves_from_trace(movement.trace, loading)
    data["warnings"] = _warnings(movement.trace)
    data.update(_layers(compiled, list(compiled.data), data["moves"]))
    meta["metrics"] = _metrics_dict(metrics, movement.spacing_warnings)

    # Sum the *stored* values, in the order the page sums them: this is exactly the page's own check.
    total = sum(move["d"] for move in data["moves"])
    assert abs(total - metrics.move_distance) < 1e-6, "move distances disagree with compute_metrics"
    assert (data["depth_at"][-1] if data["depth_at"] else 0) == metrics.depth, "depth disagrees with compute_metrics"
    assert len(data["moves"]) == metrics.num_moves, "move count disagrees with compute_metrics"
    assert sum(m["w"] for m in data["moves"]) == movement.spacing_warnings, "warning count disagrees"

    try:
        segments = physics_check(compiled, loading, movement.trace)
        meta["physics"], meta["physics_ok"] = f"transversal CZ ok ({segments} segments)", True
    except PhysicsError as error:
        meta["physics"], meta["physics_ok"] = f"PhysicsError: {error}"[:2000], False
    return data


def _build_invalid(data: dict[str, Any], compiled, loading: dict, error: ArchitectureError) -> None:
    """Partial trace up to the first violation; metrics of the truncated replay are marked partial."""
    meta = data["meta"]
    kinds = sorted({v.kind for v in (error.violations or (error,))})
    meta["status"] = "invalid"
    meta["error"] = f"ArchitectureError[{','.join(kinds)}]: {error}"[:2000]
    meta["violations"] = [str(v)[:2000] for v in (error.violations or (error,))][:20]
    meta["physics"], meta["physics_ok"] = "not checked (invalid replay)", None

    moves, warnings, error_index, total = _partial_replay(compiled, loading)
    meta["error_index"] = error_index
    limit = len(compiled.data) if error_index is None else error_index + 1
    data["moves"], data["warnings"] = moves, warnings
    data.update(_layers(compiled, list(compiled.data)[:limit], moves))

    depth_at = data["depth_at"]
    operations = sum(1 for inst in list(compiled.data)[:limit] if inst.operation.name not in IDENTITIES)
    meta["metrics"] = {
        "num_moves": len(moves),
        "move_steps": (max((m["s"] for m in moves), default=-1) + 1) if moves else 0,
        "move_distance": _r6(total),
        "move_time": _r6(total),
        "depth": depth_at[-1] if depth_at else 0,
        "num_operations": operations,
        "spacing_warnings": sum(m["w"] for m in moves),
    }
    meta["metrics_partial"] = True


# ---------------------------------------------------------------------- rendering seam


def _renderers() -> tuple[Callable[[dict], str], Callable[[dict], str]]:
    try:
        from viz_page import render_index, render_page  # written by another agent

        return render_page, render_index
    except Exception:
        return _fallback_page, _fallback_index


def _fallback_page(data: dict) -> str:
    meta = data.get("meta", {})
    blob = json.dumps(data, separators=(",", ":"))
    return (
        "<!doctype html><meta charset=utf-8><title>"
        f"{html.escape(str(meta.get('label')))}</title>"
        "<h1>viz fallback (bench/viz_page.py missing or broken)</h1>"
        f"<pre>{html.escape(json.dumps(meta, indent=1))}</pre>"
        f"<script>const DATA = {blob};</script>"
    )


def _fallback_index(rows: dict) -> str:
    items = []
    for meta in rows.get("compilers", []):
        items.append(f"<h2>{html.escape(str(meta.get('compiler')))}</h2><ul>")
        for circuit in meta.get("circuits", []):
            link = html.escape(str(circuit.get("page") or ""))
            items.append(
                f'<li><a href="{link}">{html.escape(str(circuit.get("label")))}</a> '
                f'{html.escape(str(circuit.get("status")))}</li>'
            )
        items.append("</ul>")
    return (
        "<!doctype html><meta charset=utf-8><title>viz</title><h1>compiled circuits</h1>"
        f"<p>generated {html.escape(str(rows.get('generated')))}</p>" + "".join(items)
    )


def _render_page(data: dict) -> str:
    render_page, _ = _renderers()
    try:
        return render_page(data)
    except Exception:
        return _fallback_page(data)


def _render_index(rows: dict) -> str:
    _, render_index = _renderers()
    try:
        return render_index(rows)
    except Exception:
        return _fallback_index(rows)


# ---------------------------------------------------------------------- output naming


def _output_name(compiler_path: Path, explicit: str | None) -> str:
    if explicit is not None:
        name = explicit
    elif compiler_path.is_relative_to(ROOT):
        name = compiler_path.parent.name
    else:
        name = compiler_path.parent.name
        try:
            parts = compiler_path.relative_to(CHALLENGE_WT).parts
        except ValueError:
            parts = ()
        if len(parts) >= 3 and parts[1] == "compilers":
            folder = parts[2] if len(parts) > 3 else compiler_path.parent.name
            name = parts[0] if parts[0] == folder else f"{parts[0]}-{folder}"
    if not SLUG.match(name):
        raise SystemExit(f"unsafe output name {name!r}: expected ^[A-Za-z0-9._-]+$")
    return name


def _resolve_compiler(arg: str) -> tuple[str, Path]:
    raw = arg if ":" in arg else f"{arg}:compile_circuit"
    path_part, _, function = raw.rpartition(":")
    if not FUNCTION.match(function):
        raise SystemExit(f"invalid compiler function {function!r}")
    path = Path(path_part).resolve()
    if not path.is_file():
        raise SystemExit(f"no such compiler file: {path}")
    return f"{path}:{function}", path


def _resolve_circuit(which: str) -> Path:
    path = Path(which)
    if path.is_file():
        return path.resolve()
    for candidate in sanity_benchmarks() + metric_benchmarks():
        if benchmark_label(candidate) == which or candidate.stem == which:
            return candidate
    raise SystemExit(f"unknown circuit {which!r}: not a .qasm file and no such benchmark label")


def _select_paths(which: str) -> list[Path]:
    paths: list[Path] = []
    if which in ("sanity", "all"):
        paths += sanity_benchmarks()
    if which in ("metric", "all"):
        paths += metric_benchmarks()
    return paths


# ---------------------------------------------------------------------- meta.json / index


def _meta_set(circuits: list[dict]) -> str:
    metric = any(c["label"].startswith("metric_") for c in circuits)
    sanity = any(c["label"].startswith("sample_") for c in circuits)
    return "all" if metric and sanity else ("metric" if metric else "sanity")


def summarize(circuits: list[dict]) -> dict[str, Any]:
    """`bench/bench.py`'s score formula over the ok metric circuits present."""
    rows = [c for c in circuits if c["label"].startswith("metric_")]
    valid = bool(rows) and all(c["status"] == "ok" for c in rows)
    out: dict[str, Any] = {
        "n_metric": len(rows),
        "distance": None,
        "naive_distance": None,
        "distance_improvement": None,
        "depth": None,
        "naive_depth": None,
        "depth_improvement": None,
        "score": None,
        "valid": valid,
    }
    baseline = _naive()
    done = [c for c in rows if c["status"] == "ok" and c.get("metrics") and c["label"] in baseline]
    if not done:
        return out
    distance = sum(c["metrics"]["move_distance"] for c in done)
    depth = sum(c["metrics"]["depth"] for c in done)
    naive_distance = sum(baseline[c["label"]]["metrics"]["move_distance"] for c in done)
    naive_depth = sum(baseline[c["label"]]["metrics"]["depth"] for c in done)
    di = (naive_distance - distance) / naive_distance * 100 if naive_distance else 0.0
    pi = (naive_depth - depth) / naive_depth * 100 if naive_depth else 0.0
    out.update(
        distance=round(distance, 3),
        naive_distance=round(naive_distance, 3),
        distance_improvement=round(di, 3),
        depth=depth,
        naive_depth=naive_depth,
        depth_improvement=round(pi, 3),
        score=round(0.8 * di + 0.2 * pi, 4),
    )
    return out


def _write_meta(name: str, compiler: str, records: list[dict]) -> dict[str, Any]:
    """Merge `records` into viz/<name>/meta.json (circuits not rebuilt keep their record)."""
    VIZ.mkdir(parents=True, exist_ok=True)
    folder = VIZ / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "meta.json"
    merged: dict[str, dict] = {}
    if path.is_file():
        try:
            for circuit in json.loads(path.read_text(encoding="utf-8")).get("circuits", []):
                merged[circuit["label"]] = circuit
        except Exception:
            merged = {}
    for record in records:
        merged[record["label"]] = record
    circuits = sorted(merged.values(), key=lambda c: c["label"])
    meta = {
        "compiler": name,
        "compiler_path": compiler,
        "set": _meta_set(circuits),
        "generated": _now(),
        "circuits": circuits,
        "summary": summarize(circuits),
    }
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _collect_metas() -> list[dict]:
    out = []
    for path in sorted(VIZ.glob("*/meta.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta["circuits"] = sorted(meta.get("circuits", []), key=lambda c: c.get("label", ""))
        out.append(meta)
    return sorted(out, key=lambda m: str(m.get("compiler") or ""))


def rebuild_index() -> dict[str, Any]:
    rows = {"generated": _now(), "compilers": _collect_metas()}
    VIZ.mkdir(parents=True, exist_ok=True)
    (VIZ / "index.html").write_text(_render_index(rows), encoding="utf-8")
    return rows


# ---------------------------------------------------------------------- parallel builds


def _lite(data: dict, name: str, page: Path) -> dict[str, Any]:
    meta = data["meta"]
    return {
        "label": meta["label"],
        "file": meta["file"],
        "family": meta["family"],
        "qubits": meta["qubits"],
        "status": meta["status"],
        "page": f"{name}/{page.name}",
        "metrics": meta["metrics"],
        "naive": meta["naive"],
        "physics_ok": meta["physics_ok"],
        "error": meta["error"],
        "compile_s": meta["compile_s"],
        "bytes": page.stat().st_size,
    }


def _task(compiler: str, qasm: str, name: str, timeout: float) -> dict[str, Any]:
    """One circuit: extract -> render -> write its page. Never raises."""
    path = Path(qasm)
    try:
        data = build_data(compiler, path, name, timeout)
        folder = VIZ / name
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{data['meta']['label']}.html"
        target.write_text(_render_page(data), encoding="utf-8")
        return _lite(data, name, target)
    except Exception as error:  # a broken page must not kill the run
        label = benchmark_label(path) or path.stem
        return {
            "label": label,
            "file": str(path),
            "family": _family(path),
            "qubits": None,
            "status": "error",
            "page": None,
            "metrics": None,
            "naive": None,
            "physics_ok": None,
            "error": f"{type(error).__name__}: {error}"[:2000],
            "compile_s": None,
            "bytes": 0,
        }


def _print_record(record: dict) -> None:
    if record["status"] == "ok" and record.get("metrics"):
        m = record["metrics"]
        print(
            f"PASS {record['label']:<18} {record['file']:<44} "
            f"dist {m['move_distance']:.1f} depth {m['depth']} [{record['compile_s']}s]",
            flush=True,
        )
    else:
        print(f"FAIL {record['label']:<18} {record['file']:<44} {str(record['error'])[:300]}", flush=True)


def run_builds(
    compiler: str,
    paths: list[Path],
    bset: str,
    jobs: int,
    name: str,
    timeout: float = 300,
    on_done: Callable[[int, int, dict], None] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build + render every path in a process pool, then write meta.json and rebuild the index."""
    records: list[dict] = []
    total = len(paths)
    VIZ.mkdir(parents=True, exist_ok=True)
    try:
        with cf.ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {pool.submit(_task, compiler, str(p), name, timeout): p for p in paths}
            for future in cf.as_completed(futures):
                try:
                    record = future.result()
                except Exception as error:
                    record = _task(compiler, str(futures[future]), name, timeout)
                    record["status"] = "error"
                    record["error"] = f"{type(error).__name__}: {error}"[:2000]
                records.append(record)
                if verbose:
                    _print_record(record)
                if on_done:
                    on_done(len(records), total, record)
    finally:
        meta = _write_meta(name, compiler, records) if records else _empty_meta(name, compiler, bset)
        rebuild_index()
    return meta


def _empty_meta(name: str, compiler: str, bset: str) -> dict[str, Any]:
    VIZ.mkdir(parents=True, exist_ok=True)
    folder = VIZ / name
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"compiler": name, "compiler_path": compiler, "set": bset, "generated": _now(), "circuits": [],
            "summary": summarize([])}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _print_summary(meta: dict[str, Any]) -> None:
    summary = meta.get("summary", {})
    circuits = meta.get("circuits", [])
    ok = sum(1 for c in circuits if c["status"] == "ok")
    print("-" * 80)
    print(f"{meta['compiler']}: {ok}/{len(circuits)} passed ({meta['set']})")
    if summary.get("score") is not None:
        print(
            f"SCORE (metric circuits): {summary['score']:+.2f}%  "
            f"(distance {summary['distance']:.1f} vs {summary['naive_distance']:.1f}: "
            f"{summary['distance_improvement']:+.2f}%; depth {summary['depth']} vs "
            f"{summary['naive_depth']}: {summary['depth_improvement']:+.2f}%)"
        )
    elif summary.get("n_metric"):
        print("SCORE: not computed (a metric circuit failed)")


# ---------------------------------------------------------------------- serve


def _is_allowed_compiler(path: Path) -> bool:
    if path.is_relative_to(ROOT / "compilers"):
        return True
    try:
        parts = path.relative_to(CHALLENGE_WT).parts
    except ValueError:
        return False
    return len(parts) >= 3 and parts[1] == "compilers" and (CHALLENGE_WT / parts[0]).is_dir()


def _served_compiler(raw: str) -> tuple[str, Path, str]:
    """Validate a `/regen?compiler=` argument; raises ValueError with a short reason."""
    if not raw:
        raise ValueError("missing compiler")
    path_part, sep, function = raw.rpartition(":")
    if not sep:
        path_part, function = raw, "compile_circuit"
    if not FUNCTION.match(function):
        raise ValueError("invalid function name")
    path = Path(path_part).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("compiler must be an existing .py file")
    if not _is_allowed_compiler(path):
        raise ValueError("compiler must live under compilers/ in this repo or a challenge-wt worktree")
    name = _output_name(path, None)
    return f"{path}:{function}", path, name


_REGEN_LOCK = threading.Lock()
_REGEN: dict[str, Any] = {"thread": None}


def _write_status(payload: dict[str, Any]) -> None:
    VIZ.mkdir(parents=True, exist_ok=True)
    tmp = VIZ / "status.json.tmp"
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, VIZ / "status.json")


def _regen_running() -> bool:
    thread = _REGEN.get("thread")
    return bool(thread is not None and thread.is_alive())


def _regen_job(compiler: str, paths: list[Path], bset: str, jobs: int, name: str) -> None:
    status: dict[str, Any] = {
        "running": True,
        "compiler": name,
        "message": "starting",
        "done": 0,
        "total": len(paths),
        "started": _now(),
        "finished": None,
        "ok": False,
    }
    _write_status(status)

    def on_done(done: int, total: int, record: dict) -> None:
        status.update(done=done, total=total, message=f"{record['label']}: {record['status']}")
        _write_status(status)

    try:
        status["message"] = f"building {len(paths)} circuits"
        _write_status(status)
        meta = run_builds(compiler, paths, bset, jobs, name, on_done=on_done, verbose=False)
        status["ok"] = all(c["status"] == "ok" for c in meta["circuits"]) if meta["circuits"] else False
        status["message"] = f"done ({status['done']}/{status['total']})"
    except Exception as error:
        status["message"] = f"error: {type(error).__name__}: {error}"[:500]
        status["ok"] = False
    finally:
        status.update(running=False, finished=_now())
        _write_status(status)


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(VIZ), **kwargs)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _regen(self, url: urllib.parse.ParseResult) -> None:
        params = urllib.parse.parse_qs(url.query)
        one = lambda key, default: (params.get(key) or [default])[0]  # noqa: E731
        try:
            compiler, path, name = _served_compiler(one("compiler", ""))
            bset = one("set", "metric")
            if bset not in ("metric", "sanity", "all"):
                raise ValueError("set must be metric, sanity or all")
            jobs = max(1, min(8, int(one("jobs", "2"))))
            name = _output_name(path, one("name", "") or None)
        except SystemExit as error:  # unsafe derived folder name
            return self._json(400, {"error": str(error)})
        except ValueError as error:
            return self._json(400, {"error": str(error)})
        paths = _select_paths(bset)
        with _REGEN_LOCK:
            if _regen_running():
                return self._json(409, {"error": "a regeneration is already running"})
            thread = threading.Thread(
                target=_regen_job, args=(compiler, paths, bset, jobs, name), daemon=True, name="viz-regen"
            )
            _REGEN["thread"] = thread
            thread.start()
        self._json(202, {"started": True, "compiler": name, "set": bset, "jobs": jobs, "total": len(paths)})

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        if url.path == "/regen":
            return self._regen(url)
        if url.path in ("/", "/index.html") and not (VIZ / "index.html").exists():
            rebuild_index()
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path == "/regen":
            return self._json(405, {"error": "method not allowed"})
        super().do_HEAD()

    def _reject(self) -> None:
        self._json(405, {"error": "method not allowed"})

    do_POST = do_PUT = do_DELETE = do_PATCH = _reject  # noqa: N815


def serve(host: str, port: int) -> None:
    VIZ.mkdir(parents=True, exist_ok=True)
    if not (VIZ / "index.html").exists():
        rebuild_index()
    server = http.server.ThreadingHTTPServer((host, port), _Handler)
    print(f"viz: http://{host}:{port}/  (serving {VIZ})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nviz: shutting down", flush=True)
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------- CLI


def _cmd_page(args: argparse.Namespace) -> int:
    compiler, path = _resolve_compiler(args.compiler)
    name = _output_name(path, args.name)
    circuit = _resolve_circuit(args.circuit)
    data = build_data(compiler, circuit, name, timeout=args.timeout)
    meta = data["meta"]
    out = Path(args.output) if args.output else VIZ / name / f"{meta['label']}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_page(data), encoding="utf-8")
    if out.parent == VIZ / name:  # a page inside viz/ belongs in meta.json and the index
        _write_meta(name, compiler, [_lite(data, name, out)])
        rebuild_index()
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"json: {target}")
    print(f"page: {out} ({out.stat().st_size} bytes)")
    print(f"{meta['label']} ({meta['family']}, {meta['qubits']} qubits): {meta['status']}  {meta['physics']}")
    if meta["error"]:
        print(f"error: {meta['error'][:500]}")
    if meta["metrics"]:
        m = meta["metrics"]
        tag = " (partial)" if meta.get("metrics_partial") else ""
        print(
            f"metrics{tag}: {m['num_moves']} moves in {m['move_steps']} steps, "
            f"distance {m['move_distance']:.3f}, depth {m['depth']}, "
            f"{m['num_operations']} ops, {m['spacing_warnings']} warnings"
        )
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    compiler, path = _resolve_compiler(args.compiler)
    name = _output_name(path, args.name)
    paths = _select_paths(args.bset)
    if not paths:
        raise SystemExit("no benchmarks selected")
    meta = run_builds(compiler, paths, args.bset, args.jobs, name, timeout=args.timeout)
    _print_summary(meta)
    return 0 if all(c["status"] == "ok" for c in meta["circuits"]) else 1


def _cmd_index(args: argparse.Namespace) -> int:
    rows = rebuild_index()
    count = sum(len(m.get("circuits", [])) for m in rows["compilers"])
    print(f"index: {VIZ / 'index.html'} ({len(rows['compilers'])} compilers, {count} circuits)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    page = sub.add_parser("page", help="build one circuit's page")
    page.add_argument("compiler", help="path/to/compiler.py[:function] (default: compile_circuit)")
    page.add_argument("--circuit", required=True, help="benchmark label or .qasm path")
    page.add_argument("-o", "--output", default=None)
    page.add_argument("--json", default=None, help="also dump the raw data dict here")
    page.add_argument("--name", default=None, help="output folder (default: derived from the compiler)")
    page.add_argument("--timeout", type=float, default=300)
    page.set_defaults(func=_cmd_page)

    every = sub.add_parser("all", help="build a whole benchmark set")
    every.add_argument("compiler")
    every.add_argument("--set", dest="bset", choices=["metric", "sanity", "all"], default="metric")
    every.add_argument("--jobs", type=int, default=2)
    every.add_argument("--name", default=None)
    every.add_argument("--timeout", type=float, default=300)
    every.set_defaults(func=_cmd_all)

    index = sub.add_parser("index", help="rebuild viz/index.html from every viz/*/meta.json")
    index.set_defaults(func=_cmd_index)

    server = sub.add_parser("serve", help="serve viz/ with a /regen endpoint")
    server.add_argument("--port", type=int, default=9000)
    server.add_argument("--host", default="127.0.0.1")
    server.set_defaults(func=lambda a: serve(a.host, a.port) or 0)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
