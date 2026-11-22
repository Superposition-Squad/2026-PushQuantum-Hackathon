#!/usr/bin/env python3
"""Run MIP-LNS on every cached plan of a compiler; save improved plans + a JSON line per circuit.
    python l3/lns_all.py <cache dir> <out dir> [W] [tl] [jobs]"""
import concurrent.futures as cf, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def one(args):
    path, outdir, W, tl = args
    import logging; logging.disable(logging.WARNING)
    import milp_lns as L
    logf = open(Path(outdir) / (Path(path).stem + ".log"), "w")
    res, out, load = L.lns(path, W, tl, 1, log=lambda s: (logf.write(s + "\n"), logf.flush()))
    qi = {b: i for i, b in enumerate(out.qubits)}; ci = {b: i for i, b in enumerate(out.clbits)}
    ops = [(ins.operation.name, [qi[q] for q in ins.qubits], [ci[c] for c in ins.clbits],
            [tuple(p) if isinstance(p, (tuple, list)) else float(p) for p in ins.operation.params]) for ins in out.data]
    with open(Path(outdir) / (Path(path).stem + ".pkl"), "wb") as f:
        pickle.dump({"nq": out.num_qubits, "nc": out.num_clbits, "ops": ops, "loading": load}, f)
    return res


if __name__ == "__main__":
    cdir, outdir = Path(sys.argv[1]), Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else 5
    jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    files = sorted(cdir.glob("*.pkl"), key=lambda f: f.stat().st_size)   # small first
    with cf.ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(one, [(str(f), str(outdir), W, tl) for f in files]):
            print(json.dumps(r), flush=True)
