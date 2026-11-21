"""Tests for the systolic compiler: move-batcher legality, and end-to-end debug asserts.

Run with `/home/hackathon/challenge/.venv/bin/python -m pytest compilers/systolic -q` or plainly
`/home/hackathon/challenge/.venv/bin/python compilers/systolic/test_systolic.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))

_spec = importlib.util.spec_from_file_location("systolic", Path(__file__).with_name("compiler.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _occ(atoms):
    return {site: q for q, site in atoms.items()}


# ---------------------------------------------------------------- batcher legality


def test_group_params_accepts_rigid_shift():
    # two atoms in one column, both shift one row down: one legal rectangle move
    atoms = {0: ((2, 10), (2, 11)), 1: ((2, 12), (2, 13))}
    params = C._group_params(atoms, _occ({0: (2, 10), 1: (2, 12)}))
    assert params is not None
    rows, cols, new_rows, new_cols, members = params
    assert rows == (10, 11, 12)
    assert cols == (2,)
    assert new_rows == (11, 13)  # empty row 11 is dropped by the squeeze rule
    assert set(members) == {0, 1}


def test_group_params_rejects_inconsistent_rows():
    # same source row, two different target rows: no single row map
    atoms = {0: ((0, 12), (0, 13)), 1: ((1, 12), (1, 15))}
    assert C._group_params(atoms, _occ({0: (0, 12), 1: (1, 12)})) is None


def test_group_params_rejects_non_monotone_map():
    # source rows 12, 13 would have to map to 14, 13 (not increasing)
    atoms = {0: ((0, 12), (0, 14)), 1: ((0, 13), (0, 13))}
    atoms[1] = ((1, 13), (1, 13))
    assert C._group_params(atoms, _occ({0: (0, 12), 1: (1, 13)})) is None


def test_group_params_rejects_ghost_in_rectangle():
    # atom 2 sits inside the rectangle and is neither a member nor a fixed point
    atoms = {0: ((0, 10), (0, 11)), 1: ((1, 10), (1, 11))}
    occ = {(0, 10): 0, (1, 10): 1, (0, 11): 2}
    assert C._group_params(atoms, occ) is None


def test_group_params_pulls_in_fixed_point():
    # atom 2 is a fixed point of both maps -> legal, and joins as a zero-displacement member
    atoms = {0: ((0, 10), (0, 11)), 1: ((1, 10), (1, 11)), 2: ((0, 13), (0, 13))}
    occ = {(0, 10): 0, (1, 10): 1, (0, 13): 2}
    params = C._group_params(atoms, occ)
    assert params is not None
    assert set(params[4]) == {0, 1, 2}


def test_group_params_rejects_occupied_destination():
    atoms = {0: ((0, 10), (0, 11))}
    occ = {(0, 10): 0, (0, 11): 3}
    assert C._group_params(atoms, occ) is None


def test_group_params_allows_zone_crossing():
    atoms = {0: ((2, 9), (2, 10)), 1: ((4, 9), (4, 10))}
    assert C._group_params(atoms, _occ({0: (2, 9), 1: (4, 9)})) is not None


def test_plan_moves_is_ordered_and_legal():
    # a swap cycle: two atoms exchange sites -> the plan must stay legal (via a temporary site)
    reloc = {0: ((0, 10), (1, 10)), 1: ((1, 10), (0, 10))}
    occ = {(0, 10): 0, (1, 10): 1}
    plan = C._plan_moves(reloc, occ)
    live = dict(occ)
    for group in plan:
        params = C._group_params(group, live)
        assert params is not None, group
        C._apply_atoms(live, group)
    assert live[(1, 10)] == 0 and live[(0, 10)] == 1


def test_plan_moves_fuses_identical_vectors():
    reloc = {q: ((2, 10 + 2 * q), (2, 11 + 2 * q)) for q in range(4)}
    occ = {(2, 10 + 2 * q): q for q in range(4)}
    plan = C._plan_moves(reloc, occ)
    assert len(plan) == 1
    assert abs(C._move_cost(plan[0]) - 1.0) < 1e-9


# ---------------------------------------------------------------- end to end


BENCH = ROOT / "vendor" / "pushquantum_validator" / "metric_benchmarks"


def _compile(path, debug=True):
    from qiskit import QuantumCircuit

    old = C._DEBUG
    C._DEBUG = debug  # asserts that every pulse hits exactly the intended pairs
    try:
        return C.compile_circuit(QuantumCircuit.from_qasm_file(str(path)), None)
    finally:
        C._DEBUG = old


def test_12q_qaoa_debug_pulses():
    out, arch = _compile(BENCH / "qaoa/bus_stop_12q_qaoa_p1.qasm")
    assert out.num_qubits == 12
    assert len(arch.initial_loading) == 12


def test_12q_ihva_debug_pulses():
    out, arch = _compile(BENCH / "ihva/bus_stop_12q_ihva_stagger_p1.qasm")
    assert out.num_qubits == 12


def test_20q_hea_debug_pulses():
    out, arch = _compile(BENCH / "hea/bus_stop_20q_hea_linear_p1.qasm")
    assert out.num_qubits == 20


def test_gate_set_is_restricted():
    out, _arch = _compile(BENCH / "qaoa/bus_stop_12q_qaoa_p1.qasm")
    allowed = {"rx", "rz", "cz", "move", "measure", "barrier"}
    assert {inst.operation.name for inst in out.data} <= allowed


def test_no_fallback_on_family_circuits():
    # the lane path (not the generic fallback) must handle the metric families
    C._STATS.update(lanes=0, generic=0, fell_back=0)
    _compile(BENCH / "qaoa/bus_stop_12q_qaoa_p3.qasm")
    assert C._STATS["lanes"] == 1 and C._STATS["generic"] == 0
    assert C._STATS["fell_back"] == 0


if __name__ == "__main__":  # pragma: no cover - plain script mode
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as error:  # noqa: BLE001
                failures += 1
                print("FAIL", name, error)
    raise SystemExit(1 if failures else 0)
