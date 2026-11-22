#!/usr/bin/env python3
"""Re-plan the movement of a cached compiler output and compare with the original.

    python l3/run.py <cache dir> [--filter s] [--variant name] [--eq]
"""
import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
from check import check  # noqa: E402

VB = HERE.parent / "vendor/pushquantum_validator/metric_benchmarks"


def plan(sch, loading, variant):
    mod, _, kw = variant.partition(":")
    P = importlib.import_module(mod)
    kwargs = eval("dict(" + kw + ")") if kw else {}
    return P.plan(sch, loading, **kwargs)


def main():
    cdir = Path(sys.argv[1])
    flt = sys.argv[sys.argv.index("--filter") + 1] if "--filter" in sys.argv else ""
    variants = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--variant"] or ["planner"]
    eq = "--eq" in sys.argv
    tot = {}
    for f in sorted(cdir.glob("*.pkl")):
        if flt not in f.name:
            continue
        circ, loading = M.load_cached(f)
        n = circ.num_qubits
        ref = None
        if eq and n <= 19:
            from qiskit import QuantumCircuit
            fam = "hea" if "hea" in f.name else "ihva"
            ref = QuantumCircuit.from_qasm_file(str(VB / fam / (f.stem + ".qasm")))
        base = check(circ, loading)
        line = f"{f.stem[9:]:24s} src D={base['distance']:7.1f} d={base['depth']:4d}"
        tot.setdefault("src", 0.0)
        tot["src"] += base["distance"]
        for v in variants:
            sch = M.extract(circ)
            M.cancel_dummies(sch)
            t = time.time()
            try:
                out, load = plan(sch, loading, v)
                assert M.signature(out) == M.schedule_signature(sch), "signature mismatch"
                r = check(out, load, ref)
                line += f" | {v}: D={r['distance']:7.1f} d={r['depth']:4d} {r['equivalence'][:5]} {time.time()-t:4.1f}s"
                tot[v] = tot.get(v, 0.0) + min(r["distance"], base["distance"])
                tot[v + " raw"] = tot.get(v + " raw", 0.0) + r["distance"]
            except Exception as e:  # noqa: BLE001
                line += f" | {v}: FAIL {type(e).__name__}: {str(e)[:120]}"
                tot[v] = tot.get(v, 0.0) + base["distance"]
        print(line, flush=True)
    print("TOTAL", {k: round(v, 1) for k, v in tot.items()})


if __name__ == "__main__":
    main()
