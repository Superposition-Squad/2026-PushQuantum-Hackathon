#!/usr/bin/env python3
"""Compile the 20 scored circuits with a compiler and pickle (compiled circuit, initial loading).

    .venv/bin/python l3/cache.py <compiler> [--filter s] [--jobs 6]
Output: l3/cache/<compiler>/<circuit stem>.pkl
"""
import concurrent.futures as cf
import importlib.util
import logging
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
VB = ROOT / "vendor/pushquantum_validator/metric_benchmarks"


def scored_files():
    fs = sorted(VB.glob("ihva/*.qasm")) + [VB / "hea/bus_stop_12q_hea_linear_p1.qasm", VB / "hea/bus_stop_12q_hea_linear_p3.qasm"]
    return fs


def one(comp, path):
    logging.disable(logging.WARNING)
    from qiskit import QuantumCircuit
    from pushquantum_validator.spec import load_trusted_spec
    spec = importlib.util.spec_from_file_location("c", ROOT / "compilers" / comp / "compiler.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    qc = QuantumCircuit.from_qasm_file(str(path))
    t = time.time()
    out, arch = m.compile_circuit(qc, load_trusted_spec())
    dst = ROOT / "l3/cache" / comp / (path.stem + ".pkl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    qi = {b: i for i, b in enumerate(out.qubits)}
    ci = {b: i for i, b in enumerate(out.clbits)}
    ops = [(ins.operation.name, [qi[q] for q in ins.qubits], [ci[c] for c in ins.clbits],
            [tuple(p) if isinstance(p, (tuple, list)) else float(p) for p in ins.operation.params]) for ins in out.data]
    with open(dst, "wb") as f:
        pickle.dump({"nq": out.num_qubits, "nc": out.num_clbits, "ops": ops,
                     "loading": dict(arch.initial_loading), "secs": time.time() - t}, f)
    return path.stem, round(time.time() - t, 1)


if __name__ == "__main__":
    comp = sys.argv[1]
    flt = sys.argv[sys.argv.index("--filter") + 1] if "--filter" in sys.argv else ""
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    files = [f for f in scored_files() if flt in f.name]
    files.sort(key=lambda f: -int(f.name.split("_")[2][:-1]))  # big first
    with cf.ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(one, [comp] * len(files), files):
            print(*r, flush=True)
