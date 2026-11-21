"""Local benchmark harness for planqc compilers, built on the official `pushquantum_validator`.

Usage (from the repo root):
    uv run python bench/bench.py compilers/<idea>/compiler.py [--mode fast|full|strict]
        [--set metric|sanity|all] [--jobs N] [--filter SUBSTR] [--name LABEL] [--timeout S]

Every mode also runs the transversal-CZ physics check (docs/adr/0007-transversal-cz.md) unless
--no-physics is given.

Modes:
    fast   compile + movement replay + metrics (no equivalence check)
    full   fast + the official equivalence check (QCEC <= 19 qubits, interaction pairs above)
    strict full + our own extra checks above 19 qubits: random-input statevector comparison for
           <= 22 qubits, QCEC with a timeout (best effort) for 50 qubits. Use before merging anything.

The score printed for `--set metric` without `--filter` is exactly the official leaderboard score.
With `--filter`, a partial score over the selected circuits is shown (same formula).
Results go to results/<name>.json (latest, meant to be committed) and results/runs/ (history).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from pushquantum_validator.verify import (  # noqa: E402
    MAX_EQUIVALENCE_QUBITS,
    SCORE_DEPTH_WEIGHT,
    SCORE_DISTANCE_WEIGHT,
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
from pushquantum_validator.errors import ArchitectureError, CompilerOutputError, NotEquivalentError  # noqa: E402

STRICT_QCEC_TIMEOUT = 120
NO_PHYSICS = os.environ.get("PQ_NO_PHYSICS") == "1"
STATEVECTOR_MAX_QUBITS = 22


def _family(path: Path) -> str:
    return path.parent.name if path.parent.name in ("hea", "ihva", "qaoa") else "sanity"


def _strict_check(gates, reference) -> str:
    """Extra equivalence evidence above the official QCEC limit. Returns a status string.

    <= STATEVECTOR_MAX_QUBITS: both circuits are applied to 3 random product input states (a random
    single-qubit layer prepended to both); every output overlap must be 1. Above: QCEC (ZX +
    simulation + alternating) with a timeout; only `not_equivalent` fails, inconclusive is reported."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    a = gates.remove_final_measurements(inplace=False)
    b = reference.remove_final_measurements(inplace=False)
    if a.num_qubits <= STATEVECTOR_MAX_QUBITS:
        try:
            from qiskit_aer import AerSimulator  # dev-only dependency, ~1000x faster than Statevector
            sim = AerSimulator(method="statevector", fusion_enable=True)
        except ImportError:
            sim = None
        rng = np.random.default_rng(1234)
        for trial in range(3):
            prep = QuantumCircuit(a.num_qubits)
            for q in range(a.num_qubits):
                prep.u(*rng.uniform(0, 2 * np.pi, 3), q)
            states = []
            for circ in (a, b):
                full = prep.compose(circ)
                if sim is None:
                    states.append(Statevector(full))
                else:
                    full = full.decompose(reps=0)
                    full.save_statevector()
                    states.append(Statevector(sim.run(full).result().get_statevector()))
            fid = abs(states[0].inner(states[1])) ** 2
            if fid < 1 - 1e-6:
                raise NotEquivalentError(f"statevector check {trial}: fidelity {fid:.6f} < 1")
        return "statevector x3 ok"
    from mqt import qcec

    try:
        result = qcec.verify(a, b, run_alternating_checker=True, run_construction_checker=False,
                             run_zx_checker=True, run_simulation_checker=True, timeout=STRICT_QCEC_TIMEOUT)
    except Exception as error:  # memory, internal errors
        return f"qcec inconclusive ({type(error).__name__})"
    eq = str(result.equivalence).split(".")[-1]
    if eq == "not_equivalent":
        raise NotEquivalentError("QCEC (strict): not_equivalent")
    return f"qcec {eq}"


class PhysicsError(Exception):
    """The compiled circuit relies on something the validator does not model (see docs/adr/0007)."""


def physics_check(compiled, loading, trace) -> int:
    """Transversal CZ: a CZ pulse acts on *every* atom pair sitting on an interaction pair of the
    entangling zone. Between two consecutive moves the atom positions are fixed, so within each such
    segment every occupied interaction pair must receive the same number of `cz` (= number of pulses).
    Returns the number of segments checked; raises `PhysicsError` on the first violation."""
    from pushquantum_validator.spec import load_trusted_spec

    ent = load_trusted_spec()["zones"]["entangling"]
    (x0, x1), (y0, y1) = ent["x"], ent["y"]
    move_at = {rec.index: rec.positions for rec in trace}
    index = {bit: i for i, bit in enumerate(compiled.qubits)}
    pos = dict(loading)

    def occupied_pairs():
        sites = {s: q for q, s in pos.items() if x0 <= s[0] <= x1 and y0 <= s[1] <= y1}
        pairs = set()
        for (x, y), q in sites.items():
            if (x - x0) % 2 == 0 and (x + 1, y) in sites:
                pairs.add(frozenset((q, sites[(x + 1, y)])))
        return pairs

    segments = 0
    counts: dict = {}

    def close(at):
        nonlocal segments
        if counts:
            segments += 1
            occ = occupied_pairs()
            ks = {counts.get(p, 0) for p in occ}
            if len(ks) > 1 or set(counts) - occ:
                idle = sorted(sorted(p) for p in occ if counts.get(p, 0) != max(ks))
                raise PhysicsError(
                    f"segment ending at instruction {at}: CZ pulses hit pairs unevenly "
                    f"(counts {sorted((sorted(p), k) for p, k in counts.items())[:6]}, under-pulsed pairs {idle[:6]})")
        counts.clear()

    for i, inst in enumerate(compiled.data):
        name = inst.operation.name
        if name == "move":
            close(i)
            pos.update(move_at.get(i, {}))
        elif name == "cz":
            p = frozenset(index[q] for q in inst.qubits)
            counts[p] = counts.get(p, 0) + 1
    close(len(compiled.data))
    return segments


def run_one(path_str: str, compiler: str, mode: str, timeout: float) -> dict:
    from qiskit import QuantumCircuit

    import logging

    logging.getLogger("pushquantum_validator").setLevel(logging.ERROR)  # spacing warnings are counted instead
    path = Path(path_str)
    label = benchmark_label(path) or path.stem
    record = {"label": label, "file": str(path.relative_to(ROOT / "vendor" / "pushquantum_validator")),
              "family": _family(path), "ok": False}
    t0 = time.perf_counter()
    try:
        reference = QuantumCircuit.from_qasm_file(str(path))
        record["qubits"] = reference.num_qubits
        compiled, loading = run_compiler(reference, compiler, timeout=timeout, network=False)
        record["compile_s"] = round(time.perf_counter() - t0, 2)
        movement = verify_movement(compiled, loading, log_file=os.devnull)
        gates = strip_moves(compiled)
        metrics = compute_metrics(compiled, movement, gates)
        record["metrics"] = {
            "num_moves": metrics.num_moves,
            "move_steps": metrics.move_steps,
            "move_distance": round(metrics.move_distance, 6),
            "move_time": round(metrics.move_time, 6),
            "depth": metrics.depth,
            "num_operations": metrics.num_operations,
            "spacing_warnings": movement.spacing_warnings,
        }
        if not NO_PHYSICS:
            physics_check(compiled, loading, movement.trace)
        record["equivalence"] = "skipped (fast)"
        if mode in ("full", "strict"):
            result = verify_equivalence(compiled, reference, gates)
            record["equivalence"] = (
                f"interaction pairs only (> {MAX_EQUIVALENCE_QUBITS} qubits)" if result is None
                else str(result.equivalence).split(".")[-1]
            )
            if mode == "strict" and result is None:
                record["equivalence"] += "; strict: " + _strict_check(gates, reference)
        record["ok"] = True
    except ArchitectureError as error:
        kinds = sorted({v.kind for v in (error.violations or (error,))})
        record["error"] = f"ArchitectureError[{','.join(kinds)}]: {str(error)[:500]}"
    except NotEquivalentError as error:
        record["error"] = f"NotEquivalentError: {str(error)[:500]}"
    except PhysicsError as error:
        record["error"] = f"PhysicsError: {str(error)[:500]}"
    except CompilerOutputError as error:
        record["error"] = f"CompilerOutputError: {str(error)[:1500]}"
    except Exception as error:  # harness bug; keep going
        record["error"] = f"HarnessError {type(error).__name__}: {error}\n{traceback.format_exc()[-1500:]}"
    record["wall_s"] = round(time.perf_counter() - t0, 2)
    return record


def score(records: list[dict]) -> dict:
    naive = naive_baseline()
    rows = [r for r in records if r["label"].startswith("metric_")]
    out = {"n": len(rows), "valid": all(r["ok"] for r in rows) and bool(rows)}
    if not out["valid"]:
        return out
    d = sum(r["metrics"]["move_distance"] for r in rows)
    p = sum(r["metrics"]["depth"] for r in rows)
    nd = sum(naive[r["label"]]["metrics"]["move_distance"] for r in rows)
    npth = sum(naive[r["label"]]["metrics"]["depth"] for r in rows)
    di, pi = (nd - d) / nd * 100, (npth - p) / npth * 100
    out.update(distance=round(d, 3), naive_distance=round(nd, 3), distance_improvement=round(di, 3),
               depth=p, naive_depth=npth, depth_improvement=round(pi, 3),
               score=round(SCORE_DISTANCE_WEIGHT * di + SCORE_DEPTH_WEIGHT * pi, 4),
               official=len(rows) == 36)
    fam = {}
    for f in ("hea", "ihva", "qaoa"):
        fr = [r for r in rows if r["family"] == f]
        if fr:
            fd = sum(r["metrics"]["move_distance"] for r in fr)
            fnd = sum(naive[r["label"]]["metrics"]["move_distance"] for r in fr)
            fp = sum(r["metrics"]["depth"] for r in fr)
            fnp = sum(naive[r["label"]]["metrics"]["depth"] for r in fr)
            fam[f] = {"distance_improvement": round((fnd - fd) / fnd * 100, 2),
                      "depth_improvement": round((fnp - fp) / fnp * 100, 2)}
    out["families"] = fam
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compiler", help="path/to/compiler.py[:function] (default function: compile_circuit)")
    ap.add_argument("--mode", choices=["fast", "full", "strict"], default="fast")
    ap.add_argument("--set", dest="bset", choices=["metric", "sanity", "all"], default="metric")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--filter", default=None, help="only benchmarks whose label or file contains this")
    ap.add_argument("--name", default=None, help="result label (default: the compiler's folder name)")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-physics", action="store_true", help="skip the transversal-CZ check (docs/adr/0007)")
    args = ap.parse_args(argv)
    if args.no_physics:
        os.environ["PQ_NO_PHYSICS"] = "1"
        global NO_PHYSICS
        NO_PHYSICS = True

    compiler = args.compiler if ":" in args.compiler else f"{args.compiler}:compile_circuit"
    cpath = Path(compiler.rpartition(":")[0]).resolve()
    if not cpath.is_file():
        ap.error(f"no such compiler file: {cpath}")
    compiler = f"{cpath}:{compiler.rpartition(':')[2]}"
    name = args.name or cpath.parent.name

    paths = []
    if args.bset in ("sanity", "all"):
        paths += sanity_benchmarks()
    if args.bset in ("metric", "all"):
        paths += metric_benchmarks()
    if args.filter:
        paths = [p for p in paths if args.filter in (benchmark_label(p) or "") or args.filter in str(p)]
    if not paths:
        ap.error("no benchmarks selected")

    naive = naive_baseline()
    t0 = time.perf_counter()
    records = []
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, str(p), compiler, args.mode, args.timeout): p for p in paths}
        for fut in cf.as_completed(futs):
            r = fut.result()
            records.append(r)
            if not args.quiet:
                if r["ok"]:
                    m, b = r["metrics"], naive.get(r["label"], {}).get("metrics")
                    rel = ""
                    if b:
                        rel = (f" dist {m['move_distance']:.1f} ({(m['move_distance'] / b['move_distance'] - 1) * 100:+.1f}%)"
                               f" depth {m['depth']} ({(m['depth'] / b['depth'] - 1) * 100:+.1f}%)")
                    print(f"PASS {r['label']:<18} {r['file']:<44}{rel} [{r['wall_s']}s] {r['equivalence']}", flush=True)
                else:
                    print(f"FAIL {r['label']:<18} {r['file']:<44} {r['error'][:300]}", flush=True)
    records.sort(key=lambda r: r["label"])
    summary = score(records)
    elapsed = round(time.perf_counter() - t0, 1)

    out = {"name": name, "compiler": str(cpath.relative_to(ROOT)) if cpath.is_relative_to(ROOT) else str(cpath),
           "mode": args.mode, "physics_check": not args.no_physics, "set": args.bset, "filter": args.filter,
           "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "elapsed_s": elapsed,
           "summary": summary, "failed": [r["label"] for r in records if not r["ok"]], "records": records}
    (ROOT / "results" / "runs").mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (ROOT / "results" / "runs" / f"{name}-{args.mode}-{stamp}.json").write_text(json.dumps(out, indent=1))
    if args.bset in ("metric", "all") and not args.filter:
        (ROOT / "results" / f"{name}.json").write_text(json.dumps(out, indent=1))

    print("-" * 80)
    print(f"{name}: {len(records) - len(out['failed'])}/{len(records)} passed in {elapsed}s ({args.mode})")
    if out["failed"]:
        print("FAILED:", ", ".join(out["failed"]))
    if summary.get("valid"):
        tag = "OFFICIAL SCORE" if summary["official"] else f"partial score ({summary['n']} metric circuits)"
        print(f"{tag}: {summary['score']:+.2f}%  (distance {summary['distance']:.1f} vs {summary['naive_distance']:.1f}:"
              f" {summary['distance_improvement']:+.2f}%; depth {summary['depth']} vs {summary['naive_depth']}:"
              f" {summary['depth_improvement']:+.2f}%)")
        for f, v in summary["families"].items():
            print(f"  {f:<5} distance {v['distance_improvement']:+.2f}%  depth {v['depth_improvement']:+.2f}%")
    elif summary["n"]:
        print("SCORE: not computed (a metric benchmark failed)")
    return 1 if out["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
