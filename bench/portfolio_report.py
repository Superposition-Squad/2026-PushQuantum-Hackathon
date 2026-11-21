"""Which component the portfolio picks per circuit, with every component's cost and timing.

Usage (from the repo root):
    <python> bench/portfolio_report.py [compilers/portfolio/compiler.py] [--set metric|sanity|all]
        [--filter SUBSTR] [--jobs 2] [--json out.json]

Runs the generated compiler's `portfolio_compile` in-process (not in the scorer's sandbox, so timings
are indicative only) and prints a per-circuit table, pick counts and the slowest total compile time.
Scores and validity still have to come from bench/bench.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from pushquantum_validator.verify import benchmark_label, metric_benchmarks, sanity_benchmarks  # noqa: E402

_MODULE = None


def _load(path: str):
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location("portfolio_compiler", path)
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


def run_one(compiler: str, qasm: str) -> dict:
    from qiskit import QuantumCircuit

    module = _load(compiler)
    circuit = QuantumCircuit.from_qasm_file(qasm)
    _, report = module.portfolio_compile(circuit, None)
    p = Path(qasm)
    return {"label": benchmark_label(p) or p.stem, "file": f"{p.parent.name}/{p.name}", **report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compiler", nargs="?", default=str(ROOT / "compilers" / "portfolio" / "compiler.py"))
    ap.add_argument("--set", dest="bset", choices=["metric", "sanity", "all"], default="metric")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    compiler = str(Path(args.compiler).resolve())
    paths = ([] if args.bset == "metric" else sanity_benchmarks()) + ([] if args.bset == "sanity" else metric_benchmarks())
    if args.filter:
        paths = [p for p in paths if args.filter in str(p) or args.filter in (benchmark_label(p) or "")]
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run_one, [compiler] * len(paths), [str(p) for p in paths]))
    rows.sort(key=lambda r: r["label"])
    names = list(rows[0]["components"]) if rows else []
    print(f"{'label':<18} {'file':<40} {'chosen':<10} {'total_s':>7}  " + "  ".join(f"{n:>16}" for n in names))
    for r in rows:
        cells = []
        for n in names:
            e = r["components"].get(n, {})
            if e.get("status") == "valid":
                cells.append(f"{e['distance']:>9.1f}/{e['depth']:<5}{'*' if n == r['chosen'] else ' '}")
            else:
                cells.append(f"{(e.get('status') or '-')[:16]:>16}")
        print(f"{r['label']:<18} {r['file']:<40} {r['chosen']:<10} {r['total_s']:>7.2f}  " + "  ".join(cells))
    print("picks:", dict(Counter(r["chosen"] for r in rows)))
    rejected = [(r["label"], n, e["status"]) for r in rows for n, e in r["components"].items()
                if e.get("status") not in ("valid", None)]
    for lab, n, st in rejected:
        print(f"  not used: {lab} {n}: {st}")
    if rows:
        slow = max(rows, key=lambda r: r["total_s"])
        print(f"slowest total: {slow['total_s']:.2f}s wall, {slow['cpu_s']:.2f}s CPU ({slow['label']} {slow['file']})")
        if args.bset == "metric" and not args.filter:
            cost = sum(r["components"][r["chosen"]]["cost"] for r in rows)
            print(f"predicted score: {(1 - cost) * 100:+.4f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
