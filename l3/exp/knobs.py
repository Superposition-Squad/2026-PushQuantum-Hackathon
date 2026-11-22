"""Sweep global overrides of solo_wide Executor class attributes over the 20 scored circuits.
    python l3/exp/knobs.py '<json list of {attr: value} dicts>' [jobs]"""
import concurrent.futures as cf, sys, json
from pathlib import Path
VB = Path(__file__).resolve().parents[2] / 'vendor/pushquantum_validator/metric_benchmarks'
FILES = sorted(VB.glob('ihva/*.qasm')) + [VB / 'hea/bus_stop_12q_hea_linear_p1.qasm', VB / 'hea/bus_stop_12q_hea_linear_p3.qasm']
def one(a):
    i, ov, f = a
    import importlib.util, logging; logging.disable(logging.WARNING)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vendor'))
    from check import check
    from qiskit import QuantumCircuit
    from pushquantum_validator.spec import load_trusted_spec
    spec = importlib.util.spec_from_file_location('sw', str(Path(__file__).resolve().parents[2] / 'compilers/solo_wide/compiler.py'))
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    for k, v in ov.items():
        if k.startswith('MOD_'):
            setattr(S, k[4:], v)
        else:
            setattr(S.Executor, k, tuple(v) if isinstance(v, list) else v)
    out, arch = S.compile_circuit(QuantumCircuit.from_qasm_file(str(f)), load_trusted_spec())
    r = check(out, arch.initial_loading)
    return i, f.stem, r['distance'], r['depth']
if __name__ == '__main__':
    ovs = json.loads(sys.argv[1]); jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    tasks = [(i, ov, f) for f in sorted(FILES, key=lambda f: -int(f.name.split('_')[2][:-1])) for i, ov in enumerate(ovs)]
    res = {}
    with cf.ProcessPoolExecutor(jobs) as ex:
        for i, stem, d, dep in ex.map(one, tasks):
            res.setdefault(i, {})[stem] = (d, dep)
    out = Path(__file__).parent / 'knobs.json'
    old = json.loads(out.read_text()) if out.exists() else []
    for i, ov in enumerate(ovs):
        v = res[i]; D = sum(x[0] for x in v.values()); P = sum(x[1] for x in v.values())
        sc = 80 * (163364.298 - D) / 163364.298 + 20 * (13364 - P) / 13364
        print(json.dumps(ov), 'D', round(D, 1), 'depth', P, 'score', round(sc, 4), flush=True)
        old.append({'ov': ov, 'res': v, 'score': sc})
    out.write_text(json.dumps(old))
