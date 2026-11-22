#!/usr/bin/env python3
"""Generate compilers/<name>_l3/compiler.py = compilers/<name>/compiler.py + the l3 peephole pass
(l3/model.py, l3/group.py, l3/chain.py, l3/peep.py embedded as private modules). Single file,
stdlib + qiskit only (ADR 0006).    python l3/make_l3.py <name> [<name> ...]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L3 = ROOT / "l3"


def body(name, subs):
    src = (L3 / f"{name}.py").read_text()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("from __future__")
                    and not l.startswith("sys.path.insert"))
    for a, b in subs:
        src = src.replace(a, b)
    if "\nif __name__ == \"__main__\":" in src:
        src = src[:src.index("\nif __name__ == \"__main__\":")]
    return src


MODS = [
    ("model", []),
    ("group", [("import model as M", "M = _sys.modules['_l3_model']")]),
    ("chain", [("import model as M", "M = _sys.modules['_l3_model']"),
               ("from group import execute", "execute = _sys.modules['_l3_group'].execute")]),
    ("peep", [("import chain as C  # noqa: E402", "C = _sys.modules['_l3_chain']"),
              ("import model as M  # noqa: E402", "M = _sys.modules['_l3_model']")]),
]

HEAD = '''

# ================================================================ l3 peephole pass (l3/make_l3.py)
import sys as _l3sys
import types as _l3types

_L3_SOURCES = {}
'''
TAIL = '''

def _l3_load():
    for _name in ("model", "group", "chain", "peep"):
        _key = "_l3_" + _name
        if _key in _l3sys.modules:
            continue
        _mod = _l3types.ModuleType(_key)
        _mod.__dict__["_sys"] = _l3sys
        _l3sys.modules[_key] = _mod
        exec(compile(_L3_SOURCES[_name], _key, "exec"), _mod.__dict__)
    return _l3sys.modules["_l3_model"], _l3sys.modules["_l3_chain"], _l3sys.modules["_l3_peep"]


_L3_TOTAL_EVALS = 250000   # deterministic work cap (no wall clock)


def _l3_optimise(out, loading):
    M, C, P = _l3_load()
    sch = M.extract(out)
    X, gaps = C.incumbent(out, loading, sch)
    B = C.binding(sch)
    base = new = 0.0
    left = _L3_TOTAL_EVALS
    for k in range(sch.K + 1):
        need = P.gap_need(sch, k, X[k], X[k + 1], B)
        old = C.gap_cost(X[k], gaps[k])
        if left <= 0:
            base += old
            new += old
            continue
        mv, cst = P.optimise_gap(X[k], gaps[k], need, X[k + 1], budget=min(4000, left))
        left -= P.LAST_EVALS[0]
        base += old
        if cst is not None and cst < old - 1e-9:
            gaps[k] = mv
            new += cst
        else:
            new += old
    if new >= base - 1e-9:
        return None
    res, load, dist = C.assemble(sch, loading, gaps)
    if M.signature(res) != M.signature(out) or dist >= base - 1e-9:
        return None
    return res


_l3_base_compile_circuit = compile_circuit


_L3_WALL = 230.0  # seconds since the start of compile_circuit; skip the pass on slow machines


def compile_circuit(circuit, architecture_spec):
    import time as _l3time
    _t0 = _l3time.time()
    out, arch = _l3_base_compile_circuit(circuit, architecture_spec)
    if _l3time.time() - _t0 > _L3_WALL:
        return out, arch
    try:
        res = _l3_optimise(out, dict(arch.initial_loading))
        if res is not None:
            out = res
    except Exception:  # noqa: BLE001 - keep the component's own (valid) result
        pass
    return out, arch
'''

for name in sys.argv[1:]:
    src = (ROOT / "compilers" / name / "compiler.py").read_text().rstrip()
    parts = [src, HEAD]
    for m, subs in MODS:
        parts.append(f"_L3_SOURCES[{m!r}] = {body(m, subs)!r}\n")
    parts.append(TAIL)
    dst = ROOT / "compilers" / f"{name}_l3" / "compiler.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(parts))
    print("wrote", dst)
