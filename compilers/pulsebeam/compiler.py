"""pulsebeam: idle-rule pulse planner (atoms live in storage; each pulse = block drop + lifts) with beam search over pulses and drop/lift choices.

(Historical notes below describe the earlier zone-resident planners kept in this file.)

Pipeline (see docs/ideas/optimizer.md):

1. Front end. The input is parsed into a commutation DAG of
   - Pauli-rotation blocks exp(-i theta/2 P_a P_b) (recognised from `cx rz cx`, optionally
     conjugated by a 1q gate and its inverse on either wire, and from `rzz`),
   - single-CZ blocks (bare `cx` / `cz`),
   - single-qubit unitaries (2x2 matrices) and final measurements.
   Two ops sharing a qubit get a DAG edge unless they provably commute.
2. Executor. A block exp(-i theta/2 P_o P_r) is synthesised as
   W_o^+ W_r^+ . CZ . rx_r(theta) . CZ . W_r W_o   (W_o P_o W_o^+ = Z, W_r P_r W_r^+ = X).
   Only the *rotator* r needs a 1q gate between its two CZs; the other qubit o is "open" and may
   start further blocks with the same Pauli (the hub). Every qubit carries the 2x2 unitary that
   is still pending; it is emitted (ZXZ/XZX, fewest gates) when the atom leaves storage.
3. Scheduler. One CZ pulse at a time: a set of disjoint CZ events picked from the ready ones
   (critical-path priority, a few candidates compared by real move cost).
4. Router. Destinations are pair constraints (partner already in the zone -> exactly its sibling
   site; both moving -> a free pair chosen jointly). Transfers are grouped into moves by pairwise
   order compatibility (a move is a monotone row map x a monotone column map) with a ghost-free
   rectangle; cost of a move = its longest displacement. Strict rule: every pickup rectangle and
   every drop-off lies inside a single zone.
5. Self-check: positions are replayed with the validator's rules and the transversal-CZ rule; any
   failure or unsupported input falls back to the naive compiler.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer

Site = tuple[int, int]

# ---------------------------------------------------------------- template plumbing (keep as is)


def _as_lines(values, name):
    items = [values] if isinstance(values, Integral) else list(values)
    if not items:
        raise ValueError(f"`{name}` must not be empty.")
    result = tuple(int(v) for v in items)
    if len(set(result)) != len(result):
        raise ValueError(f"`{name}` must not contain duplicates, got {result!r}.")
    return result


def _check_lines(start, end, name):
    if len(end) > len(start):
        raise ValueError(f"`new_{name}` cannot have more lines than `{name}`.")
    if start != tuple(range(start[0], start[0] + len(start))):
        raise ValueError(f"Picked-up {name} {start} must be contiguous and ascending.")
    if any(b <= a for a, b in itertools.pairwise(end)):
        raise ValueError(f"Target {name} {end} must be ascending.")


class Move(Instruction):
    """Params are `(rows, cols, new_rows, new_cols)`, tuples of ints; rows are y, columns are x."""

    def __init__(self, rows, cols, new_rows, new_cols, num_qubits):
        names = ("rows", "cols", "new_rows", "new_cols")
        lines = [_as_lines(v, n) for v, n in zip((rows, cols, new_rows, new_cols), names, strict=True)]
        _check_lines(lines[0], lines[2], "rows")
        _check_lines(lines[1], lines[3], "cols")
        super().__init__("move", num_qubits, 0, lines)

    def validate_parameter(self, parameter):
        return parameter

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


class Architecture:
    """Only `initial_loading` is read by the validator (qubit -> (x, y) site in storage)."""

    def __init__(self, initial_loading: dict[int, Site], spec: Any = None) -> None:
        self.initial_loading = dict(initial_loading)
        self.spec = spec


class Unsupported(Exception):
    """Input this compiler does not handle; the naive path takes over."""


# ---------------------------------------------------------------- geometry (from the spec)

DEFAULT_SPEC = {
    "loading_zone": "storage",
    "zones": {
        "storage": {"x": [0, 9], "y": [0, 9], "spacing": 1.0, "operations": ["rx", "rz"]},
        "entangling": {
            "x": [0, 9],
            "y": [10, 19],
            "spacing": 1.0,
            "pairs": {"axis": "x", "distance": 0.3},
            "operations": ["cz", "rzz"],
        },
    },
}


class Geo:
    """Sites, zones and physical coordinates exactly as the validator computes them."""

    def __init__(self, spec: Any) -> None:
        if not isinstance(spec, dict) or "zones" not in spec:
            spec = DEFAULT_SPEC
        zones = spec["zones"]
        self.zone_of: dict[Site, str] = {}
        self.phys: dict[Site, tuple[float, float]] = {}
        self.storage = self.ent = None
        for name, z in reversed(list(zones.items())):  # first zone containing a site wins
            x0, x1 = z["x"]
            y0, y1 = z["y"]
            sp = float(z.get("spacing", 1.0))
            pairs = z.get("pairs")
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    self.zone_of[(x, y)] = name
                    self.phys[(x, y)] = (self._coord(x, x0, sp, pairs, "x"), self._coord(y, y0, sp, pairs, "y"))
        for name, z in zones.items():
            ops = set(z.get("operations", []))
            if {"rx", "rz"} <= ops and self.storage is None:
                self.storage = name
            if "cz" in ops and z.get("pairs") and self.ent is None:
                self.ent = name
        if self.storage is None or self.ent is None or spec.get("loading_zone", "storage") != self.storage:
            raise Unsupported("unexpected architecture")
        ez = zones[self.ent]
        if ez["pairs"]["axis"] != "x":
            raise Unsupported("pairs along y")
        self.ex0, self.ex1 = ez["x"]
        self.ey0, self.ey1 = ez["y"]
        sz = zones[self.storage]
        self.sx0, self.sx1 = sz["x"]
        self.sy0, self.sy1 = sz["y"]
        self.storage_sites = [s for s, n in self.zone_of.items() if n == self.storage]
        self.ent_sites = [s for s, n in self.zone_of.items() if n == self.ent]
        self.sibling: dict[Site, Site] = {}
        self.pairs: list[tuple[Site, Site]] = []
        for y in range(self.ey0, self.ey1 + 1):
            for x in range(self.ex0, self.ex1, 2):
                a, b = (x, y), (x + 1, y)
                if self.zone_of.get(a) == self.ent and self.zone_of.get(b) == self.ent:
                    self.sibling[a], self.sibling[b] = b, a
                    self.pairs.append((a, b))
        # storage below the entangling zone? (used only for heuristics, never for legality)
        self.ent_above = self.ey0 > self.sy1

    @staticmethod
    def _coord(line, start, spacing, pairs, axis):
        if pairs is None or pairs.get("axis") != axis:
            return line * spacing
        pair, slot = divmod(line - start, 2)
        return start * spacing + pair * 2 * spacing + slot * float(pairs["distance"])

    def dist(self, a: Site, b: Site) -> float:
        pa, pb = self.phys[a], self.phys[b]
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def in_ent(self, s: Site) -> bool:
        return self.zone_of.get(s) == self.ent

    def in_storage(self, s: Site) -> bool:
        return self.zone_of.get(s) == self.storage


# ---------------------------------------------------------------- 1q algebra

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"X": PX, "Y": PY, "Z": PZ}
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
SGATE = np.array([[1, 0], [0, 1j]], dtype=complex)


def _is_identity(u: np.ndarray, tol: float = 1e-9) -> bool:
    return abs(abs(np.trace(u)) - 2.0) < tol


def _pauli_of(m: np.ndarray) -> tuple[str, int] | None:
    """(P, sign) if m = sign * P for a Pauli P and sign = +-1."""
    for name, p in PAULI.items():
        c = np.trace(p @ m) / 2
        if abs(abs(c) - 1) < 1e-9 and abs(c.imag) < 1e-9 and np.allclose(m, c * p, atol=1e-9):
            return name, (1 if c.real > 0 else -1)
    return None


def _cliffords() -> list[np.ndarray]:
    found: list[np.ndarray] = [I2]
    frontier = [I2]
    while frontier:
        nxt = []
        for u in frontier:
            for g in (HAD, SGATE):
                v = g @ u
                if not any(_is_identity(v.conj().T @ w) for w in found):
                    found.append(v)
                    nxt.append(v)
        frontier = nxt
    return found


_CLIFF = _cliffords()
_BASIS: dict[tuple[str, str], np.ndarray] = {}
for _p in "XYZ":
    for _t in "XZ":
        for _w in _CLIFF:
            if np.allclose(_w @ PAULI[_p] @ _w.conj().T, PAULI[_t], atol=1e-9):
                _BASIS[(_p, _t)] = _w
                break


def basis_change(p: str, target: str) -> np.ndarray:
    """A Clifford W with W P W^+ = +target."""
    return _BASIS[(p, target)]


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


_ZXZ = OneQubitEulerDecomposer("ZXZ")
_XZX = OneQubitEulerDecomposer("XZX")


def synth_1q(u: np.ndarray) -> list[tuple[str, float]]:
    """Fewest rx/rz gates (time order) equal to u up to global phase."""
    if _is_identity(u):
        return []
    best = None
    for dec in (_ZXZ, _XZX):
        qc = dec(u, simplify=True, atol=1e-10)
        gates = [(ins.operation.name, float(ins.operation.params[0])) for ins in qc.data]
        gates = [(n, a) for n, a in gates if abs(math.remainder(a, 4 * math.pi)) > 1e-10]
        if best is None or len(gates) < len(best):
            best = gates
    return best


# ---------------------------------------------------------------- front end

U1, ROT, CZB, MEAS = "u1", "rot", "czb", "meas"
_SKIP = {"barrier", "delay"}
_KNOWN_2Q = {"cx", "cz", "rzz"}


class Op:
    __slots__ = ("kind", "qubits", "mat", "paulis", "theta", "basis", "clbit", "idx", "comm")

    def __init__(self, kind, qubits, mat=None, paulis=None, theta=0.0, basis=None, clbit=None):
        self.kind, self.qubits, self.mat, self.paulis = kind, tuple(qubits), mat, paulis
        self.theta, self.basis, self.clbit, self.idx = theta, basis, clbit, -1
        self.comm = None
        if kind == U1:
            self.comm = frozenset(p for p, m in PAULI.items() if np.allclose(mat @ m, m @ mat, atol=1e-9))


def _raw_gates(circuit: QuantumCircuit):
    """(name, qubits, clbits, op) with qubit/clbit indices; unknown multi-qubit gates are decomposed."""
    names = {ins.operation.name for ins in circuit.data}
    ok = all(
        n in _SKIP or n == "measure" or n in _KNOWN_2Q or (ins.operation.num_qubits == 1 and not ins.operation.num_clbits)
        for ins in circuit.data
        for n in [ins.operation.name]
    )
    if not ok:
        circuit = transpile(circuit, basis_gates=["u", "cx", "cz", "rzz", "measure"], optimization_level=0)
    del names
    qi = {b: i for i, b in enumerate(circuit.qubits)}
    ci = {b: i for i, b in enumerate(circuit.clbits)}
    out = []
    for ins in circuit.data:
        op = ins.operation
        if op.name in _SKIP:
            continue
        qs = [qi[q] for q in ins.qubits]
        cs = [ci[c] for c in ins.clbits]
        if op.name == "measure":
            out.append(("measure", qs, cs, op))
        elif op.name in _KNOWN_2Q:
            out.append((op.name, qs, cs, op))
        elif op.num_qubits == 1 and not cs:
            out.append(("u1", qs, cs, op))
        else:
            raise Unsupported(f"gate {op.name}")
    return out, circuit


def front_end(circuit: QuantumCircuit) -> tuple[list[Op], int]:
    raw, circuit = _raw_gates(circuit)
    n = circuit.num_qubits
    mats: dict[int, np.ndarray] = {}
    for i, (name, qs, cs, op) in enumerate(raw):
        if name == "u1":
            mats[i] = Operator(op).data
    # per-wire sequences of raw indices
    wire: list[list[int]] = [[] for _ in range(n)]
    for i, (_, qs, _, _) in enumerate(raw):
        for q in qs:
            wire[q].append(i)
    posw = {}
    for q in range(n):
        for k, i in enumerate(wire[q]):
            posw[(q, i)] = k

    def nxt(q, i, d=1):
        k = posw[(q, i)] + d
        return wire[q][k] if 0 <= k < len(wire[q]) else None

    consumed: set[int] = set()
    replaced: dict[int, Op] = {}
    for i, (name, qs, _, op) in enumerate(raw):
        if name != "cx" or i in consumed:
            continue
        c, t = qs
        j = nxt(t, i)
        k = nxt(t, j) if j is not None else None
        if j is None or k is None or j in consumed or k in consumed:
            continue
        if raw[j][0] != "u1" or raw[k][0] != "cx" or raw[k][1] != [c, t] or nxt(c, i) != k:
            continue
        d = mats[j]
        if abs(d[0, 1]) > 1e-12 or abs(d[1, 0]) > 1e-12:
            continue
        theta = float(np.angle(d[1, 1] / d[0, 0]))
        paulis = {c: "Z", t: "Z"}
        sign = 1
        extra: list[int] = []
        for q in (c, t):
            p, s = nxt(q, i, -1), nxt(q, k, 1)
            if p is None or s is None or p in consumed or s in consumed or p == s:
                continue
            if raw[p][0] != "u1" or raw[s][0] != "u1":
                continue
            mp, ms = mats[p], mats[s]
            if not _is_identity(ms @ mp):
                continue
            pq = _pauli_of(mp.conj().T @ PZ @ mp)
            if pq is None:
                continue
            paulis[q] = pq[0]
            sign *= pq[1]
            extra += [p, s]
        consumed.update([i, j, k, *extra])
        replaced[i] = Op(ROT, (c, t), paulis=(paulis[c], paulis[t]), theta=sign * theta)
    ops: list[Op] = []
    measured: set[int] = set()
    for i, (name, qs, cs, op) in enumerate(raw):
        if i in replaced:
            ops.append(replaced[i])
            continue
        if i in consumed:
            continue
        if any(q in measured for q in qs):
            raise Unsupported("gate after measurement")
        if name == "u1":
            ops.append(Op(U1, qs, mat=mats[i]))
        elif name == "measure":
            ops.append(Op(MEAS, qs, clbit=cs[0]))
            measured.add(qs[0])
        elif name == "cx":
            ops.append(Op(CZB, qs, paulis=("Z", "X"), basis=(I2, HAD)))
        elif name == "cz":
            ops.append(Op(CZB, qs, paulis=("Z", "Z"), basis=(I2, I2)))
        elif name == "rzz":
            ops.append(Op(ROT, qs, paulis=("Z", "Z"), theta=float(op.params[0])))
    # elide identity rotations (theta = 0 mod 2pi gives +-identity)
    ops = [o for o in ops if not (o.kind == ROT and abs(math.remainder(o.theta, 2 * math.pi)) < 1e-12)]
    for k, o in enumerate(ops):
        o.idx = k
    return ops, n


def _pauli_on(o: Op, q: int) -> str:
    return o.paulis[o.qubits.index(q)]


def _commute(a: Op, b: Op) -> bool:
    shared = set(a.qubits) & set(b.qubits)
    if not shared:
        return True
    if a.kind == MEAS or b.kind == MEAS:
        return False
    if a.kind == U1 and b.kind == U1:
        return False
    if a.kind == U1 or b.kind == U1:
        u, o = (a, b) if a.kind == U1 else (b, a)
        return _pauli_on(o, u.qubits[0]) in u.comm
    if a.kind == ROT and b.kind == ROT:
        anti = sum(_pauli_on(a, q) != _pauli_on(b, q) for q in shared)
        return anti % 2 == 0
    # at least one single-CZ block: functions of its per-qubit Paulis -> need qubitwise agreement
    return all(_pauli_on(a, q) == _pauli_on(b, q) for q in shared)


def build_dag(ops: list[Op], n: int):
    succ: list[set[int]] = [set() for _ in ops]
    indeg = [0] * len(ops)
    per_q: list[list[int]] = [[] for _ in range(n)]
    for o in ops:
        for q in o.qubits:
            for j in per_q[q]:
                if o.idx not in succ[j] and not _commute(ops[j], o):
                    succ[j].add(o.idx)
                    indeg[o.idx] += 1
            per_q[q].append(o.idx)
    return succ, indeg


# ---------------------------------------------------------------- machine state + move grouping


class State:
    """Atom positions with the validator's move semantics (strict: one zone per side)."""

    def __init__(self, geo: Geo, loading: dict[int, Site]):
        self.geo = geo
        self.pos = dict(loading)
        self.occ = {s: q for q, s in self.pos.items()}

    def copy(self) -> "State":
        st = State.__new__(State)
        st.geo, st.pos, st.occ = self.geo, dict(self.pos), dict(self.occ)
        return st

    def apply(self, group: list[tuple[int, Site]]) -> tuple[tuple, float]:
        """group = [(qubit, target)]; returns (move params, distance). Raises on illegal moves."""
        g = self.geo
        src = {q: self.pos[q] for q, _ in group}
        tgt = dict(group)
        xs = [s[0] for s in src.values()]
        ys = [s[1] for s in src.values()]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        members = set(src)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                o = self.occ.get((x, y))
                if o is not None and o not in members:
                    raise RuntimeError("ghost")
        rmap: dict[int, int] = {}
        cmap: dict[int, int] = {}
        for q in members:
            (sx, sy), (tx, ty) = src[q], tgt[q]
            if rmap.setdefault(sy, ty) != ty or cmap.setdefault(sx, tx) != tx:
                raise RuntimeError("not a product map")
        for m in (rmap, cmap):
            ks = sorted(m)
            if any(m[a] >= m[b] for a, b in itertools.pairwise(ks)):
                raise RuntimeError("order not preserved")
        zs = {g.zone_of[s] for s in src.values()}
        zt = {g.zone_of.get(t) for t in tgt.values()}
        if len(zs) != 1 or len(zt) != 1 or None in zt:
            raise RuntimeError("rectangle spans zones")
        for q in members:
            o = self.occ.get(tgt[q])
            if o is not None and o not in members:
                raise RuntimeError("target occupied")
        dist = max(g.dist(src[q], tgt[q]) for q in members)
        for q in members:
            del self.occ[src[q]]
        for q in members:
            self.pos[q] = tgt[q]
            self.occ[tgt[q]] = q
        rows = tuple(range(y0, y1 + 1))
        cols = tuple(range(x0, x1 + 1))
        new_rows = tuple(rmap[y] for y in sorted(rmap))
        new_cols = tuple(cmap[x] for x in sorted(cmap))
        return (rows, cols, new_rows, new_cols, sorted(members)), dist


def _sgn(v: int) -> int:
    return (v > 0) - (v < 0)


def _compatible(a, b) -> bool:
    """Transfers a=(q, src, dst), b: can one monotone row map x column map carry both?"""
    (_, (ax, ay), (atx, aty)), (_, (bx, by), (btx, bty)) = a, b
    return _sgn(ax - bx) == _sgn(atx - btx) and _sgn(ay - by) == _sgn(aty - bty)


def group_transfers(st: State, transfers: list[tuple[int, Site, Site]]):
    """Split transfers into legal moves (largest displacement seeds a group, everything compatible
    joins). Returns [(params, dist)] and applies them to st."""
    g = st.geo
    todo = {q: (q, s, d) for q, s, d in transfers if s != d}
    moves = []
    guard = 0
    while todo:
        guard += 1
        if guard > 10 * len(transfers) + 50:
            raise RuntimeError("grouping did not converge")
        movers = set(todo)
        # a transfer can go now if its target is free, or held by another mover (who may join)
        cands = sorted(todo.values(), key=lambda t: -g.dist(t[1], t[2]))
        seed = None
        for t in cands:
            o = st.occ.get(t[2])
            if o is None:
                seed = t
                break
        if seed is None:
            # cycle: park one blocker on a free site of its own zone
            t = cands[0]
            q = st.occ[t[2]]
            if q not in todo:
                raise RuntimeError("target held by a stationary atom")
            free = [s for s in (g.ent_sites if g.in_ent(st.pos[q]) else g.storage_sites) if s not in st.occ]
            free.sort(key=lambda s: g.dist(st.pos[q], s))
            params, dist = st.apply([(q, free[0])])
            moves.append((params, dist))
            todo[q] = (q, free[0], todo[q][2])
            continue
        zs, zt = g.zone_of[seed[1]], g.zone_of[seed[2]]

        def fits(t, grp):
            return g.zone_of[t[1]] == zs and g.zone_of[t[2]] == zt and all(_compatible(t, u) for u in grp)

        group = [seed]
        excluded: set[int] = set()
        for t in cands:
            if t is not seed and fits(t, group):
                group.append(t)
        sx, sy = seed[1]
        while True:
            mem = {u[0] for u in group}
            xs = [u[1][0] for u in group]
            ys = [u[1][1] for u in group]
            ghosts = [
                st.occ[(x, y)]
                for x in range(min(xs), max(xs) + 1)
                for y in range(min(ys), max(ys) + 1)
                if (x, y) in st.occ and st.occ[(x, y)] not in mem
            ]
            held = [u for u in group if st.occ.get(u[2]) is not None and st.occ.get(u[2]) not in mem]
            if not ghosts and not held:
                break
            pulled = False
            for q in ghosts:
                if q in todo and q not in excluded and fits(todo[q], group):
                    group.append(todo[q])
                    pulled = True
            if pulled:
                continue
            if held:
                excluded.update(u[0] for u in held)
                group = [u for u in group if u not in held]  # the seed's target is free
                continue
            if len(group) == 1:
                raise RuntimeError("ghost on a single-atom move")
            far = max(group[1:], key=lambda u: abs(u[1][0] - sx) + abs(u[1][1] - sy))
            excluded.add(far[0])
            group.remove(far)
        del movers
        params, dist = st.apply([(u[0], u[2]) for u in group])
        moves.append((params, dist))
        for u in group:
            del todo[u[0]]
    return moves


# ---------------------------------------------------------------- the compiler


class Compiler:
    def __init__(self, circuit: QuantumCircuit, spec: Any, options: dict | None = None):
        self.opt = {"cands": 4, "proactive_down": True}
        if options:
            self.opt.update(options)
        self.circuit = circuit
        self.geo = Geo(spec)
        self.ops, self.n = front_end(circuit)
        if self.n > len(self.geo.storage_sites):
            raise Unsupported("too many qubits")
        self.succ, self.indeg = build_dag(self.ops, self.n)
        if len(self.ops) > 150 and not (options and "cands" in options):
            self.opt["cands"] = 1  # keep large irregular circuits fast

    # ------------------------------------------------------------ setup
    def initial_loading(self) -> dict[int, Site]:
        g = self.geo
        # rows nearest the entangling zone first
        rows = sorted(range(g.sy0, g.sy1 + 1), key=lambda y: -y if g.ent_above else y)
        sites = [(x, y) for y in rows for x in range(g.sx0, g.sx1 + 1)]
        return {q: sites[q] for q in range(self.n)}

    # ------------------------------------------------------------ main loop
    def run(self):
        g = self.geo
        n = self.n
        ops = self.ops
        self.loading = self.initial_loading()
        st = State(g, self.loading)
        self.st = st
        self.out: list[tuple] = []  # ("u", q, name, angle) | ("cz", a, b) | ("move", params) | ("meas", q, c)
        self.dist_total = 0.0
        U = [I2.copy() for _ in range(n)]
        mode = ["idle"] * n  # idle | rot | open
        frame: list[str | None] = [None] * n
        open_cnt = [0] * n
        indeg = list(self.indeg)
        bstate: dict[int, str] = {}  # block idx -> "cz1"
        brot: dict[int, int] = {}
        done = [False] * len(ops)
        remaining_cz = [0] * n
        for o in ops:
            if o.kind in (ROT, CZB):
                for q in o.qubits:
                    remaining_cz[q] += 2 if o.kind == ROT else 1
        ready_blocks: set[int] = set()
        pend_meas: list[Op] = []
        level = [0] * n  # ASAP depth level per qubit (gates emitted so far)
        self.U, self.mode, self.level = U, mode, level

        def finish(i):
            done[i] = True
            for j in self.succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)

        ready = [o.idx for o in ops if indeg[o.idx] == 0]

        def absorb():
            # eagerly absorb ready 1q ops / measurements, collect ready blocks
            progress = True
            while progress:
                progress = False
                nonlocal ready
                rest = []
                for i in ready:
                    o = ops[i]
                    if o.kind == U1:
                        q = o.qubits[0]
                        if mode[q] == "idle":
                            U[q] = o.mat @ U[q]
                            finish(i)
                            progress = True
                        else:
                            rest.append(i)
                    elif o.kind == MEAS:
                        pend_meas.append(o)
                        finish(i)
                        progress = True
                    else:
                        ready_blocks.add(i)
                ready = [i for i in rest if not done[i]]

        absorb()
        guard = 0
        while ready_blocks or bstate:
            guard += 1
            if guard > 20 * len(ops) + 100:
                raise RuntimeError("scheduler stuck")
            events = self._events(ready_blocks, bstate, brot, mode, frame, remaining_cz)
            if not events:
                raise RuntimeError("no events")
            S = self._choose(events, remaining_cz)
            # commit: basis changes before CZ1 (they are part of the pending unitary)
            for ev in S:
                kind, i, r, o_ = ev
                op = ops[i]
                if kind == "start":
                    if op.kind == ROT:
                        pr, po = _pauli_on(op, r), _pauli_on(op, o_)
                        U[r] = basis_change(pr, "X") @ U[r]
                        if mode[o_] != "open":
                            U[o_] = basis_change(po, "Z") @ U[o_]
                            mode[o_], frame[o_] = "open", po
                        open_cnt[o_] += 1
                        mode[r] = "rot"
                    else:
                        for q, w in zip(op.qubits, op.basis):
                            U[q] = w @ U[q]
            self._route_and_pulse(S, U)
            for ev in S:
                kind, i, r, o_ = ev
                op = ops[i]
                for q in op.qubits:
                    remaining_cz[q] -= 1
                if kind == "start" and op.kind == ROT:
                    bstate[i] = "cz1"
                    brot[i] = r
                    U[r] = _rx(op.theta) @ U[r]
                    ready_blocks.discard(i)
                elif kind == "finish":
                    del bstate[i]
                    r = brot.pop(i)
                    o_ = [q for q in op.qubits if q != r][0]
                    U[r] = basis_change(_pauli_on(op, r), "X").conj().T @ U[r]
                    mode[r] = "idle"
                    open_cnt[o_] -= 1
                    if open_cnt[o_] == 0:
                        U[o_] = basis_change(frame[o_], "Z").conj().T @ U[o_]
                        mode[o_], frame[o_] = "idle", None
                    finish(i)
                else:  # single-CZ block
                    for q, w in zip(op.qubits, op.basis):
                        U[q] = w.conj().T @ U[q]
                    ready_blocks.discard(i)
                    finish(i)
            absorb()
        # final: every atom with a pending unitary goes to storage, then measurements
        down = [q for q in range(n) if not _is_identity(U[q]) and g.in_ent(st.pos[q])]
        self._send_down(down)
        for q in range(n):
            if not _is_identity(U[q]):
                self._flush(q)
        for o in sorted(pend_meas, key=lambda o: o.idx):
            self.out.append(("meas", o.qubits[0], o.clbit))
        return self.out

    # ------------------------------------------------------------ scheduling
    def _events(self, ready_blocks, bstate, brot, mode, frame, rem):
        ops = self.ops
        evs = []
        for i in bstate:
            r = brot[i]
            o_ = [q for q in ops[i].qubits if q != r][0]
            evs.append(("finish", i, r, o_))
        for i in ready_blocks:
            op = ops[i]
            a, b = op.qubits
            if op.kind == CZB:
                if mode[a] == "idle" and mode[b] == "idle":
                    evs.append(("start", i, None, None))
                continue
            for r, o_ in ((a, b), (b, a)):
                if mode[r] != "idle":
                    continue
                po = _pauli_on(op, o_)
                if mode[o_] == "idle" or (mode[o_] == "open" and frame[o_] == po):
                    evs.append(("start", i, r, o_))
        # rotator choice: keep only the better orientation per block (rotate the less busy qubit,
        # never one that is open)
        best: dict[int, tuple] = {}
        for ev in evs:
            if ev[0] != "start" or ev[2] is None:
                continue
            k = (self.mode[ev[3]] == "open", rem[ev[3]] - rem[ev[2]])
            if ev[1] not in best or k > best[ev[1]][0]:
                best[ev[1]] = (k, ev)
        return [ev for ev in evs if ev[0] != "start" or ev[2] is None or best[ev[1]][1] is ev]

    def _ev_qubits(self, ev):
        return self.ops[ev[1]].qubits

    def _choose(self, events, rem):
        """Greedy matching by priority; a few candidate sets are compared by real routing cost."""
        st = self.st
        g = self.geo

        def est(ev):
            a, b = self._ev_qubits(ev)
            return g.dist(st.pos[a], st.pos[b])

        def prio(ev):
            a, b = self._ev_qubits(ev)
            return max(rem[a], rem[b])

        evs = sorted(events, key=lambda e: (-prio(e), e[0] != "finish", est(e), e[1]))
        cands = []
        top = evs[: self.opt["cands"]]
        for first in top:
            used = set(self._ev_qubits(first))
            S = [first]
            for e in evs:
                if e is first:
                    continue
                qs = self._ev_qubits(e)
                if used & set(qs):
                    continue
                S.append(e)
                used.update(qs)
            cands.append(S)
        if len(cands) == 1:
            return cands[0]
        best, best_cost = None, None
        for S in cands:
            cost = self._dry_cost(S)
            score = cost / (len(S) + 2.0 * sum(prio(e) for e in S[:1]) / max(1, max(rem)))
            if best_cost is None or score < best_cost:
                best, best_cost = S, score
        return best

    def _dry_cost(self, S):
        saved = (self.st, self.out, self.dist_total, self.U)
        self.st = self.st.copy()
        self.out = []
        self.dist_total = 0.0
        try:
            U = [u.copy() for u in self.U]
            for kind, i, r, o_ in S:
                op = self.ops[i]
                if kind == "start" and op.kind == ROT:
                    U[r] = basis_change(_pauli_on(op, r), "X") @ U[r]
                    if self.mode[o_] != "open":
                        U[o_] = basis_change(_pauli_on(op, o_), "Z") @ U[o_]
                elif kind == "start":
                    for q, w in zip(op.qubits, op.basis):
                        U[q] = w @ U[q]
            self.U = U
            self._route_and_pulse(S, U, dry=True)
            return self.dist_total
        finally:
            self.st, self.out, self.dist_total, self.U = saved[0], saved[1], saved[2], saved[3]

    # ------------------------------------------------------------ routing
    def _emit_move(self, params, dist):
        self.out.append(("move", params))
        self.dist_total += dist

    def _flush(self, q):
        assert self.geo.in_storage(self.st.pos[q])
        for name, ang in synth_1q(self.U[q]):
            self.out.append(("u", q, name, ang))
        self.U[q] = I2.copy()

    def _storage_targets(self, qs: list[int], st: State) -> dict[int, Site]:
        """Drop zone atoms straight down into the free storage sites nearest the zone, keeping
        the order of rows within every column."""
        g = self.geo
        res: dict[int, Site] = {}
        bycol: dict[int, list[int]] = {}
        for q in qs:
            bycol.setdefault(st.pos[q][0], []).append(q)
        taken: set[Site] = set()
        for x, group in bycol.items():
            group.sort(key=lambda q: st.pos[q][1])
            xs = min(max(x, g.sx0), g.sx1)
            free = [(xs, y) for y in range(g.sy0, g.sy1 + 1) if (xs, y) not in st.occ and (xs, y) not in taken]
            free.sort(key=lambda s: -s[1] if g.ent_above else s[1])
            if len(free) >= len(group):
                chosen = sorted(free[: len(group)], key=lambda s: s[1])
                for q, s in zip(group, chosen):
                    res[q] = s
                    taken.add(s)
            else:
                for q in group:
                    cand = [s for s in g.storage_sites if s not in st.occ and s not in taken]
                    cand.sort(key=lambda s: g.dist(st.pos[q], s))
                    res[q] = cand[0]
                    taken.add(cand[0])
        return res

    def _send_down(self, qs):
        if not qs:
            return
        st = self.st
        tg = self._storage_targets(qs, st)
        for params, dist in group_transfers(st, [(q, st.pos[q], tg[q]) for q in qs]):
            self._emit_move(params, dist)

    def _route_and_pulse(self, S, U, dry=False):
        g = self.geo
        st = self.st
        ops = self.ops
        pairs = [tuple(ops[i].qubits) for _, i, _, _ in S]
        in_S = {q for p in pairs for q in p}
        # phase A: dirty atoms in the zone go to storage (those in S must; others proactively)
        down = []
        for q in range(self.n):
            if not g.in_ent(st.pos[q]):
                continue
            if not _is_identity(U[q]) and (q in in_S or self.opt["proactive_down"]):
                down.append(q)
            elif IDLE_RULE and q not in in_S:
                down.append(q)  # validator v1.0.0: no atom may sit in the zone during a pulse it is not in
        self._send_down(down)
        # phase B: destinations as pair constraints
        reserved: set[Site] = set()
        tg: dict[int, Site] = {}

        def free(s):
            return s not in st.occ and s not in reserved

        order = sorted(pairs, key=lambda p: -min(self._anchor_cost(p, st)))
        for a, b in order:
            if g.in_ent(st.pos[a]) and g.sibling.get(st.pos[a]) == st.pos[b]:
                continue  # already together
            best = None
            for anc, mov in ((a, b), (b, a)):
                s = st.pos[anc]
                if g.in_ent(s) and free(g.sibling[s]):
                    c = g.dist(st.pos[mov], g.sibling[s])
                    if best is None or c < best[0]:
                        best = (c, {mov: g.sibling[s]})
            # both move to a free pair
            for left, right in g.pairs:
                if not (free(left) or st.occ.get(left) in (a, b)) or not (free(right) or st.occ.get(right) in (a, b)):
                    continue
                if st.occ.get(left) is not None and st.occ.get(left) not in (a, b):
                    continue
                pa, pb = st.pos[a], st.pos[b]
                for qa, qb in ((a, b), (b, a)):
                    c = max(g.dist(st.pos[qa], left), g.dist(st.pos[qb], right))
                    if best is None or c < best[0] - 1e-9:
                        best = (c, {qa: left, qb: right})
                del pa, pb
            if best is None:
                raise RuntimeError("no room in the entangling zone")
            for q, s in best[1].items():
                tg[q] = s
                reserved.add(s)
        transfers = [(q, st.pos[q], s) for q, s in tg.items() if st.pos[q] != s]
        # atoms leaving storage get their pending 1q gates first
        for q, src, _ in transfers:
            if g.in_storage(src) and not _is_identity(U[q]):
                if not dry:
                    self._flush(q)
                else:
                    U[q] = I2.copy()
        for q in in_S:
            if not _is_identity(U[q]):
                if g.in_storage(st.pos[q]):
                    if not dry:
                        self._flush(q)
                else:
                    raise RuntimeError("dirty atom in the zone")
        for params, dist in group_transfers(st, transfers):
            self._emit_move(params, dist)
        # sanity: every pulse pair together, no other pair doubly occupied
        if not dry:
            for a, b in pairs:
                if g.sibling.get(st.pos[a]) != st.pos[b]:
                    raise RuntimeError("pair not formed")
            for left, right in g.pairs:
                qa, qb = st.occ.get(left), st.occ.get(right)
                if qa is not None and qb is not None and {qa, qb} not in [set(p) for p in pairs]:
                    raise RuntimeError("unwanted pair")
            for a, b in pairs:
                self.out.append(("cz", a, b))

    def _anchor_cost(self, p, st):
        g = self.geo
        a, b = p
        return [g.dist(st.pos[a], st.pos[b])]


# ---------------------------------------------------------------- executor (gate-level state machine)


class Exec:
    """DAG progress plus, per qubit, the pending 1q unitary and the role (idle / rot / open frame).

    Events: ("start", block, rotator) and ("finish", block, rotator); rotator is None for single-CZ
    blocks. `pre_ws(ev)` = the basis changes an event adds before its CZ (per qubit)."""

    def __init__(self, ops, succ, indeg, n):
        self.ops, self.succ, self.n = ops, succ, n
        self.indeg = list(indeg)
        self.U = [I2.copy() for _ in range(n)]
        self.mode = ["idle"] * n
        self.frame: list[str | None] = [None] * n
        self.open_cnt = [0] * n
        self.bstate: dict[int, int] = {}
        self.done = [False] * len(ops)
        self.ready = [o.idx for o in ops if indeg[o.idx] == 0]
        self.ready_blocks: set[int] = set()
        self.pend_meas: list[Op] = []
        self.absorb()

    def clone(self) -> "Exec":
        e = Exec.__new__(Exec)
        e.ops, e.succ, e.n = self.ops, self.succ, self.n
        e.indeg = list(self.indeg)
        e.U = [u.copy() for u in self.U]
        e.mode, e.frame, e.open_cnt = list(self.mode), list(self.frame), list(self.open_cnt)
        e.bstate, e.done, e.ready = dict(self.bstate), list(self.done), list(self.ready)
        e.ready_blocks, e.pend_meas = set(self.ready_blocks), list(self.pend_meas)
        return e

    def finished(self) -> bool:
        return not self.ready_blocks and not self.bstate

    def _finish(self, i):
        self.done[i] = True
        for j in self.succ[i]:
            self.indeg[j] -= 1
            if self.indeg[j] == 0:
                self.ready.append(j)

    def absorb(self):
        progress = True
        while progress:
            progress = False
            rest = []
            for i in self.ready:
                o = self.ops[i]
                if o.kind == U1:
                    q = o.qubits[0]
                    if self.mode[q] == "idle":
                        self.U[q] = o.mat @ self.U[q]
                        self._finish(i)
                        progress = True
                    else:
                        rest.append(i)
                elif o.kind == MEAS:
                    self.pend_meas.append(o)
                    self._finish(i)
                    progress = True
                else:
                    self.ready_blocks.add(i)
            self.ready = [i for i in rest if not self.done[i]]

    def other(self, i, r):
        return [q for q in self.ops[i].qubits if q != r][0]

    def start_roles(self, i) -> list[int | None]:
        """Feasible rotators for starting ready block i (None for a single-CZ block)."""
        op = self.ops[i]
        a, b = op.qubits
        if op.kind == CZB:
            return [None] if self.mode[a] == "idle" and self.mode[b] == "idle" else []
        res = []
        for r, o in ((a, b), (b, a)):
            if self.mode[r] != "idle":
                continue
            po = _pauli_on(op, o)
            if self.mode[o] == "idle" or (self.mode[o] == "open" and self.frame[o] == po):
                res.append(r)
        return res

    def pre_ws(self, ev) -> dict[int, np.ndarray]:
        kind, i, r = ev
        op = self.ops[i]
        if kind == "finish":
            return {}
        if op.kind == CZB:
            return {q: w for q, w in zip(op.qubits, op.basis)}
        o = self.other(i, r)
        res = {r: basis_change(_pauli_on(op, r), "X")}
        if self.mode[o] != "open":
            res[o] = basis_change(_pauli_on(op, o), "Z")
        return res

    def apply_pre(self, ev, skip=()):
        kind, i, r = ev
        op = self.ops[i]
        for q, w in self.pre_ws(ev).items():
            if q not in skip:
                self.U[q] = w @ self.U[q]
        if kind == "start" and op.kind == ROT:
            o = self.other(i, r)
            if self.mode[o] != "open":
                self.mode[o], self.frame[o] = "open", _pauli_on(op, o)
            self.open_cnt[o] += 1
            self.mode[r] = "rot"

    def apply_post(self, ev):
        kind, i, r = ev
        op = self.ops[i]
        if kind == "start" and op.kind == ROT:
            self.bstate[i] = r
            self.U[r] = _rx(op.theta) @ self.U[r]
            self.ready_blocks.discard(i)
        elif kind == "finish":
            del self.bstate[i]
            o = self.other(i, r)
            self.U[r] = basis_change(_pauli_on(op, r), "X").conj().T @ self.U[r]
            self.mode[r] = "idle"
            self.open_cnt[o] -= 1
            if self.open_cnt[o] == 0:
                self.U[o] = basis_change(self.frame[o], "Z").conj().T @ self.U[o]
                self.mode[o], self.frame[o] = "idle", None
            self._finish(i)
        else:
            for q, w in zip(op.qubits, op.basis):
                self.U[q] = w.conj().T @ self.U[q]
            self.ready_blocks.discard(i)
            self._finish(i)


def find_hub(ops: list[Op], n: int) -> int | None:
    deg = [set() for _ in range(n)]
    for o in ops:
        if o.kind in (ROT, CZB):
            a, b = o.qubits
            deg[a].add(b)
            deg[b].add(a)
    if n < 4:
        return None
    h = max(range(n), key=lambda q: (len(deg[q]), q))
    return h if len(deg[h]) >= 0.5 * (n - 1) else None


class CyclePlanner:
    """Hub circuits: the whole register is lifted into the zone, the hub walks over its partners
    (one CZ per step), non-hub pairs are formed before the last pulse of a cycle, the register is
    dropped back (rotations happen in storage), and the next lift re-forms those pairs for CZ2."""

    def __init__(self, geo: Geo, ops, succ, indeg, n, hub, order=None, match="index", hub_min=0.0,
                 slack=None, hub_max=1.0, trips=False):
        self.geo, self.ops, self.succ, self.indeg, self.n, self.hub = geo, ops, succ, indeg, n, hub
        self.match, self.hub_min, self.slack, self.hub_max = match, hub_min, slack, hub_max
        self.trips = trips
        self.work = 0  # deterministic effort counter (routing evaluations)
        # block level in the DAG (1q ops pass levels through): QAOA layer l -> level l
        lvl = [0] * len(ops)
        for i, o in enumerate(ops):  # edges always point forward in op order
            if o.kind in (ROT, CZB):
                lvl[i] += 1
            for j in succ[i]:
                lvl[j] = max(lvl[j], lvl[i])
        self.lvl = lvl
        g = geo
        slots = order if order is not None else self.default_order()
        partners = [p for p in slots if p is not None]
        m = len(partners)
        ncols = (g.ex1 - g.ex0 + 1) // 2
        R = max(1, math.ceil(m / ncols))
        if len(slots) > R * ncols:
            raise Unsupported("order longer than the grid")
        if R > g.ey1 - g.ey0 + 1 or R > g.sy1 - g.sy0 + 1 or not g.ent_above:
            raise Unsupported("layout does not fit")
        self.R = R
        self.shift = g.ey0 - (g.sy1 - R + 1)
        self.home: dict[int, Site] = {}
        for k, p in enumerate(slots):
            if p is None:
                continue
            c, row = divmod(k, R)
            self.home[p] = (g.ex0 + 2 * c, g.ey0 + row)
        walk = []
        for c in range(ncols):
            col = [p for p in partners if self.home[p][0] == g.ex0 + 2 * c]
            col.sort(key=lambda p: self.home[p][1], reverse=bool(c % 2))
            walk += col
        self.walk = walk
        self.widx = {p: k for k, p in enumerate(walk)}
        first = walk[0]
        hx, hy = self.home[first]
        self.loading = {p: (s[0], s[1] - self.shift) for p, s in self.home.items()}
        self.loading[hub] = (hx + 1, hy - self.shift)
        self.direction = 1

    def default_order(self) -> list[int]:
        seen = []
        for o in self.ops:
            if o.kind in (ROT, CZB) and self.hub in o.qubits:
                p = [q for q in o.qubits if q != self.hub][0]
                if p not in seen:
                    seen.append(p)
        seen += [q for q in range(self.n) if q != self.hub and q not in seen]
        return seen

    # ---------------------------------------------------------- output helpers
    def emit_moves(self, transfers):
        for params, dist in group_transfers(self.st, transfers):
            self.out.append(("move", params))
            self.dist_total += dist

    def flush(self, q):
        for name, ang in synth_1q(self.ex.U[q]):
            self.out.append(("u", q, name, ang))
        self.ex.U[q] = I2.copy()

    # ---------------------------------------------------------- planning one cycle
    def direction_of(self, cands) -> bool:
        hpos = self.st.pos[self.hub]
        a = min(cands, key=lambda p: self.widx[p])
        b = max(cands, key=lambda p: self.widx[p])
        return self.geo.dist(hpos, self.home[a]) <= self.geo.dist(hpos, self.home[b])

    def _allowed(self, ex: Exec, i) -> bool:
        if self.slack is None:
            return True
        low = min((self.lvl[j] for j, o in enumerate(self.ops) if o.kind in (ROT, CZB) and not ex.done[j]
                   and j not in ex.bstate), default=0)
        return self.lvl[i] <= low + self.slack

    def _hub_event(self, ex: Exec, p):
        h = self.hub
        for i, r in ex.bstate.items():
            if r == p and h in self.ops[i].qubits:
                return ("finish", i, r)
        for i in sorted(ex.ready_blocks):
            op = self.ops[i]
            if h in op.qubits and p in op.qubits and self._allowed(ex, i):
                roles = ex.start_roles(i)
                if op.kind == CZB and roles:
                    return ("start", i, None)
                if p in roles:
                    return ("start", i, p)
        return None

    def _qs(self, ev):
        return self.ops[ev[1]].qubits

    def _clean_ok(self, ex: Exec, ev, touched) -> bool:
        ws = ex.pre_ws(ev)
        return all(q not in touched or _is_identity(ws.get(q, I2) @ ex.U[q]) for q in self._qs(ev))

    def _commit(self, ex: Exec, pulse, touched):
        for ev in pulse:
            ex.apply_pre(ev)
            for q in self._qs(ev):
                if q not in touched:
                    ex.U[q] = I2.copy()  # flushed before the lift
                    touched.add(q)
        for ev in pulse:
            ex.apply_post(ev)
        ex.absorb()

    def plan_cycle(self):
        """Pulses of the next cycle, simulated on a copy of the executor: first the CZ2s of the
        pairs re-formed by the lift, then one hub event per pulse along the walk, and on the last
        pulse a matching of non-hub CZ1s (chosen on the state before that pulse)."""
        h = self.hub
        self.cycle_trips = None
        ex = self.ex.clone()
        touched: set[int] = set()
        pulses: list[list] = []
        p1 = [("finish", i, r) for i, r in sorted(ex.bstate.items()) if h not in self.ops[i].qubits]
        if p1:
            if not all(self._clean_ok(ex, ev, touched) for ev in p1):
                raise RuntimeError("lift pair not clean")
            self._commit(ex, p1, touched)
            pulses.append(p1)
        cands = [p for p in self.walk if self._hub_event(ex, p) is not None]
        if cands and self.hub_min > 0 and len(cands) < self.hub_min * len(self.walk):
            # few hub events ready: spend this cycle on non-hub work if there is any
            if any(h not in self.ops[i].qubits for i in ex.ready_blocks) and not any(
                h in self.ops[i].qubits for i in ex.bstate
            ):
                cands = []
        if cands:
            hpos = self.st.pos[h]
            if self.geo.dist(hpos, self.home[cands[-1]]) < self.geo.dist(hpos, self.home[cands[0]]):
                cands.reverse()
        snap = None
        n_hub = 0
        cap = max(1, math.ceil(self.hub_max * len(self.walk)))
        # finishes first (they free their partner), then starts, both in walk order
        if self.hub_max < 1.0 and cands:
            fin = [p for p in cands if self._hub_event(ex, p)[0] == "finish"]
            cands = fin + [p for p in cands if p not in fin]
            cands = cands[:cap]
            sign = 1 if self.direction_of(cands) else -1
            cands.sort(key=lambda p: self.widx[p] * sign)
        trips: list[set] = [set() for _ in pulses]
        if not self.trips:
            for p in cands:
                ev = self._hub_event(ex, p)
                if ev is None or not self._clean_ok(ex, ev, touched):
                    continue
                snap = (ex.clone(), set(touched), len(pulses))
                self._commit(ex, [ev], touched)
                pulses.append([ev])
                trips.append(set())
                n_hub += 1
        else:
            # dynamic walk: nearest feasible hub event next; a dirty hub may take a storage trip
            g = self.geo
            hpos = self.st.pos[h]
            done_p: set[int] = set()
            while n_hub < cap:
                best = None
                for p in self.walk:
                    if p in done_p:
                        continue
                    ev = self._hub_event(ex, p)
                    if ev is None:
                        continue
                    trip = False
                    if not self._clean_ok(ex, ev, touched):
                        # only the hub may be dirty, and only if it holds no open block
                        ws = ex.pre_ws(ev)
                        if not _is_identity(ws.get(p, I2) @ ex.U[p]) and p in touched:
                            continue
                        if ex.mode[h] == "open" and ev[0] == "start":
                            continue
                        trip = True
                    d = g.dist(hpos, g.sibling[self.home[p]]) + (2.5 if trip else 0.0)
                    if best is None or d < best[0]:
                        best = (d, p, ev, trip)
                if best is None:
                    break
                _, p, ev, trip = best
                snap = (ex.clone(), set(touched), len(pulses))
                if trip:
                    touched.discard(h)  # flushed during the trip
                self._commit(ex, [ev], touched)
                pulses.append([ev])
                trips.append({h} if trip else set())
                done_p.add(p)
                hpos = g.sibling[self.home[p]]
                n_hub += 1
        # matching on the state before the last pulse
        if snap is not None:
            ex, touched, k = snap[0], snap[1], snap[2]
            last = pulses[k]
            last_trip = trips[k]
            del pulses[k:]
            del trips[k:]
            if last_trip:
                touched.discard(h)
        else:
            last, last_trip = [], set()
        used = {q for ev in last for q in self._qs(ev)}
        M = []
        # remaining non-hub work per atom: busiest atoms first (greedy edge colouring)
        work = [0] * self.n
        for i, o in enumerate(self.ops):
            if not ex.done[i] and o.kind in (ROT, CZB) and h not in o.qubits:
                for q in o.qubits:
                    work[q] += 1
        if self.match == "work":
            mord = sorted(ex.ready_blocks, key=lambda i: (-max(work[q] for q in self.ops[i].qubits),
                                                            -min(work[q] for q in self.ops[i].qubits), i))
        else:
            mord = sorted(ex.ready_blocks)
        for i in mord:
            op = self.ops[i]
            if h in op.qubits or used & set(op.qubits) or not self._allowed(ex, i):
                continue
            for r in ex.start_roles(i):
                ev = ("start", i, r)
                if self._clean_ok(ex, ev, touched):
                    M.append(ev)
                    used.update(op.qubits)
                    break
        if last or M:
            self._commit(ex, last + M, touched)
            pulses.append(last + M)
            trips.append(last_trip)
        self.cycle_trips = trips
        return pulses

    # ---------------------------------------------------------- execution
    def run(self):
        g = self.geo
        h = self.hub
        self.ex = Exec(self.ops, self.succ, self.indeg, self.n)
        self.st = State(g, self.loading)
        self.out: list[tuple] = []
        self.dist_total = 0.0
        guard = 0
        while not self.ex.finished():
            guard += 1
            if guard > 4 * len(self.ops) + 20:
                raise RuntimeError("cycle planner stuck")
            pulses = self.plan_cycle()
            if not pulses and self.slack is not None:
                keep, self.slack = self.slack, None  # the barrier blocked everything: relax once
                pulses = self.plan_cycle()
                self.slack = keep
            if not pulses:
                raise RuntimeError("empty cycle")
            # which event touches each qubit first in this cycle
            first: dict[int, tuple] = {}
            for pl in pulses:
                for ev in pl:
                    for q in self.ops[ev[1]].qubits:
                        first.setdefault(q, ev)
            # flush before the lift: pending unitary + basis change of the first event
            ex = self.ex
            pre_done: dict[int, tuple] = {}
            sim = ex.clone()
            for pl in pulses:
                for ev in pl:
                    ws = sim.pre_ws(ev)
                    for q in self.ops[ev[1]].qubits:
                        if first[q] is ev and q not in pre_done:
                            pre_done[q] = (ev, ws.get(q))
                    sim.apply_pre(ev)
                for ev in pl:
                    sim.apply_post(ev)
                sim.absorb()
            for q, (ev, w) in pre_done.items():
                if w is not None:
                    ex.U[q] = w @ ex.U[q]
                if not g.in_storage(self.st.pos[q]):
                    if not _is_identity(ex.U[q]):
                        raise RuntimeError("dirty atom outside storage")
                    ex.U[q] = I2.copy()
                else:
                    self.flush(q)
            # lift everything that is in storage
            self.emit_moves([(q, s, (s[0], s[1] + self.shift)) for q, s in self.st.pos.items() if g.in_storage(s)])
            trips = getattr(self, "cycle_trips", None) or [set() for _ in pulses]
            for k, pl in enumerate(pulses):
                tripped = set()
                for q in trips[k] if k < len(trips) else ():
                    ev = next(e for e in pl if q in self.ops[e[1]].qubits)
                    x, _ = self.st.pos[q]
                    xs = min(max(x, g.sx0), g.sx1)
                    free = [(xs, y) for y in range(g.sy1, g.sy0 - 1, -1) if (xs, y) not in self.st.occ]
                    self.emit_moves([(q, self.st.pos[q], free[0])])
                    w = ex.pre_ws(ev).get(q)
                    if w is not None:
                        ex.U[q] = w @ ex.U[q]
                    self.flush(q)
                    tripped.add(q)
                self.route(pl)
                for ev in pl:
                    skip = [q for q in self.ops[ev[1]].qubits if first[q] is ev or q in tripped]
                    ex.apply_pre(ev, skip=skip)
                    for q in self.ops[ev[1]].qubits:
                        if not _is_identity(ex.U[q]):
                            raise RuntimeError("dirty atom at a pulse")
                for ev in pl:
                    a, b = self.ops[ev[1]].qubits
                    self.out.append(("cz", a, b))
                for ev in pl:
                    ex.apply_post(ev)
                ex.absorb()
            # drop everything
            self.emit_moves([(q, s, (s[0], s[1] - self.shift)) for q, s in self.st.pos.items() if g.in_ent(s)])
        for q in range(self.n):
            if not _is_identity(self.ex.U[q]):
                if not g.in_storage(self.st.pos[q]):
                    raise RuntimeError("final atom outside storage")
                self.flush(q)
        for o in sorted(self.ex.pend_meas, key=lambda o: o.idx):
            self.out.append(("meas", o.qubits[0], o.clbit))
        return self.out

    def route(self, pulse):
        """Moves before a pulse. Each pulse pair is a constraint "a and b on one pair"; for a
        non-hub pair either atom may come to the other's sibling site. Atoms that would otherwise
        sit in an unwanted pair leave (home if free, else the nearest site that is alone). The
        choice per pair is a local search over the real grouped move cost."""
        g, st, h = self.geo, self.st, self.hub
        pairs = [tuple(self.ops[ev[1]].qubits) for ev in pulse]
        decisions = []
        for a, b in pairs:
            if g.sibling.get(st.pos[a]) == st.pos[b]:
                continue
            if h in (a, b):
                p = b if a == h else a
                opt = {h: g.sibling[self.home[p]]}
                if st.pos[p] != self.home[p]:
                    opt[p] = self.home[p]
                decisions.append([opt])
                continue
            ev = pulse[pairs.index((a, b))]
            rot = ev[2] if ev[2] is not None else b
            non = a if rot == b else b
            hn = self.home[non]
            canon = {rot: g.sibling[hn]}
            if st.pos[non] != hn:
                canon[non] = hn
            opts = [canon]
            for anchor, mover in ((a, b), (b, a)):
                sa = st.pos[anchor]
                if g.in_ent(sa):
                    o = {mover: g.sibling[sa]}
                    if o not in opts:
                        opts.append(o)
                ha = self.home[anchor]
                o = {mover: g.sibling[ha]}
                if ha != sa:
                    o[anchor] = ha
                if o not in opts:
                    opts.append(o)
            decisions.append(opts)
        in_pulse = {q for pr in pairs for q in pr}
        best = None
        choice = [0] * len(decisions)
        cache: dict = {}

        def evaluate(ch):
            key = tuple(ch)
            if key in cache:
                return cache[key]
            res = self._realize(decisions, ch, pairs, in_pulse)
            cache[key] = res
            return res

        best = evaluate(choice)
        for _ in range(2):
            improved = False
            for k, opts in enumerate(decisions):
                for j in range(len(opts)):
                    if j == choice[k]:
                        continue
                    trial = list(choice)
                    trial[k] = j
                    r = evaluate(trial)
                    if r[0] < best[0] - 1e-9:
                        best, choice, improved = r, trial, True
            if not improved:
                break
        if best[1] is None:
            raise RuntimeError("no legal routing for a pulse")
        self.emit_moves(best[1])
        for left, right in g.pairs:
            qa, qb = st.occ.get(left), st.occ.get(right)
            if qa is not None and qb is not None and {qa, qb} not in [set(p) for p in pairs]:
                raise RuntimeError("unwanted pair before pulse")
        for a, b in pairs:
            if g.sibling.get(st.pos[a]) != st.pos[b]:
                raise RuntimeError("pair not formed")

    def _realize(self, decisions, choice, pairs, in_pulse):
        """(cost, transfers) for one choice per decision; cost inf if illegal."""
        self.work += 1
        g, st, h = self.geo, self.st, self.hub
        tg: dict[int, Site] = {}
        for opts, c in zip(decisions, choice):
            for q, s in opts[c].items():
                if q in tg and tg[q] != s:
                    return (math.inf, None)
                tg[q] = s
        final = dict(st.pos)
        final.update(tg)
        occ: dict[Site, int] = {}
        for q in tg:  # movers first: their targets must be distinct
            if final[q] in occ:
                return (math.inf, None)
            occ[final[q]] = q
        for q, s in final.items():
            if q in tg:
                continue
            if s in occ:  # a stationary atom sits on a target: it has to leave
                if q in in_pulse:
                    return (math.inf, None)
                final[q] = None
                continue
            occ[s] = q
        pairset = [set(p) for p in pairs]
        # atoms displaced by a target, or sitting in an unwanted pair, leave
        for _ in range(3 * self.n):
            bad = [q for q, s in final.items() if s is None]
            for left, right in g.pairs:
                qa, qb = occ.get(left), occ.get(right)
                if qa is not None and qb is not None and {qa, qb} not in pairset:
                    cand = [q for q in (qa, qb) if q not in in_pulse and q not in tg]
                    if not cand:
                        return (math.inf, None)
                    # prefer moving the one away from home (or the hub)
                    cand.sort(key=lambda q: (q != h and final[q] == self.home.get(q), q))
                    bad.append(cand[0])
                    del occ[final[cand[0]]]
                    final[cand[0]] = None
            if not bad:
                break
            for q in bad:
                if final.get(q) is not None:
                    continue
                spot = None
                hm = self.home.get(q)
                if hm is not None and hm not in occ and g.sibling[hm] not in occ:
                    spot = hm
                else:
                    cur = st.pos[q]
                    spots = [s for s in g.ent_sites if s not in occ and g.sibling[s] not in occ
                             and s[1] < g.ey0 + self.R]
                    if not spots:
                        return (math.inf, None)
                    spot = min(spots, key=lambda s: g.dist(cur, s))
                final[q] = spot
                occ[spot] = q
                tg[q] = spot
        transfers = [(q, st.pos[q], s) for q, s in tg.items() if st.pos[q] != s]
        try:
            moves = group_transfers(st.copy(), transfers)
        except RuntimeError:
            return (math.inf, None)
        return (sum(d for _, d in moves), transfers)

    def _target_home(self, q):
        return self.home[q] if q != self.hub else self.st.pos[q]



def sa_order(geo: Geo, ops: list[Op], n: int, hub: int, seed: int, iters: int = 40000) -> list[int]:
    """Partner order (zone homes column-major) that keeps non-hub neighbours close: simulated
    annealing on the summed visitor distance, fixed iteration budget and seed (deterministic)."""
    import random

    edges = [tuple(o.qubits) for o in ops if o.kind in (ROT, CZB) and hub not in o.qubits]
    parts = [q for q in range(n) if q != hub]
    ncols = (geo.ex1 - geo.ex0 + 1) // 2
    R = max(1, math.ceil(len(parts) / ncols))
    cells = [(c, r) for c in range(ncols) for r in range(R)]

    def site(cell):
        return (geo.ex0 + 2 * cell[0], geo.ey0 + cell[1])

    D = {}
    for a in cells:
        for b in cells:
            sa, sb = site(a), site(b)
            D[a, b] = min(geo.dist(sa, geo.sibling[sb]), geo.dist(sb, geo.sibling[sa]))
    rng = random.Random(seed)
    pos = {p: cells[i] for i, p in enumerate(parts)}
    occ = {c: p for p, c in pos.items()}
    nb: dict[int, list[int]] = {p: [] for p in parts}
    for a, b in edges:
        nb[a].append(b)
        nb[b].append(a)

    def local(p):
        return sum(D[pos[p], pos[q]] for q in nb[p])

    avg_deg = max(1.0, 2 * len(edges) / max(1, len(parts)))
    iters = int(min(iters, 160000 / avg_deg, 1000 * len(parts)))
    T = 3.0
    for _ in range(iters):
        a = rng.choice(parts)
        cell = rng.choice(cells)
        b = occ.get(cell)
        if b == a:
            continue
        old = local(a) + (local(b) if b is not None else 0)
        pa = pos[a]
        pos[a] = cell
        if b is not None:
            pos[b] = pa
        delta = local(a) + (local(b) if b is not None else 0) - old
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            occ[cell] = a
            if b is not None:
                occ[pa] = b
            else:
                del occ[pa]
        else:
            pos[a] = pa
            if b is not None:
                pos[b] = cell
        T = max(0.01, T * 0.9998)
    order = [None] * len(cells)
    for p, c in pos.items():
        order[c[0] * R + c[1]] = p
    return order  # None marks an empty home cell


# ---------------------------------------------------------------- beam planner (validator v1.0.0)


class Plan:
    """One partial plan of the beam search: gate state, positions, homes, depth levels, cost.
    The output is a chain of per-step chunks (`parent`, `chunk`) so copies stay cheap."""

    __slots__ = ("ex", "st", "home", "owner", "lvl", "dist", "parent", "chunk", "steps")

    def copy(self) -> "Plan":
        p = Plan.__new__(Plan)
        p.ex, p.st = self.ex.clone(), self.st.copy()
        p.home, p.owner, p.lvl = dict(self.home), dict(self.owner), list(self.lvl)
        p.dist, p.parent, p.chunk, p.steps = self.dist, self, [], self.steps
        return p

    def output(self) -> list[tuple]:
        chunks = []
        node = self
        while node is not None:
            chunks.append(node.chunk)
            node = node.parent
        return [item for ch in reversed(chunks) for item in ch]

    def depth(self) -> int:
        return max(self.lvl) if self.lvl else 0


class BeamPlanner:
    """Idle rule (validator v1.0.0): at every pulse the entangling zone holds exactly the pulsed pairs.

    Atoms live in storage next to the zone. One step = one pulse: zone atoms that are not in it (or
    still owe 1q gates) drop, rising atoms flush their pending 1q gates, the pairs form in the zone
    (an atom already there is an anchor; the others get zone pair sites by an order-preserving
    assignment, so lifts group into few rectangles), the CZs fire. A beam search over the choice of
    pulse and of drop targets keeps the best partial plans by distance + 2.92 * depth + an estimate
    of the remaining work. Every candidate move is checked by `State.apply` (validator semantics)."""

    def __init__(self, geo: Geo, ops, succ, indeg, n, opts: dict | None = None):
        self.geo, self.ops, self.succ, self.indeg, self.n = geo, ops, succ, indeg, n
        self.opt = {"beam": 8, "lam": 2.0, "order": None, "zone_rows": 10, "layout": "line", "block": True}
        if opts:
            self.opt.update(opts)
        self.qblocks: list[list[int]] = [[] for _ in range(n)]
        for o in ops:
            if o.kind in (ROT, CZB):
                for q in o.qubits:
                    self.qblocks[q].append(o.idx)
        height = [0] * len(ops)
        for i in range(len(ops) - 1, -1, -1):
            o = ops[i]
            height[i] = max((height[j] for j in succ[i]), default=0) + (2 if o.kind == ROT else 1 if o.kind == CZB else 0)
        self.height = height
        self.total_cz = sum(2 if o.kind == ROT else 1 for o in ops if o.kind in (ROT, CZB))
        self.hub = find_hub(ops, n)
        g = geo
        if not g.ent_above:
            raise Unsupported("expects the entangling zone above storage")
        self.srows = sorted(range(g.sy0, g.sy1 + 1), reverse=True)
        self.zrows = list(range(g.ey0, min(g.ey1, g.ey0 + self.opt["zone_rows"] - 1) + 1))
        self.loading = self.initial_loading()

    # ---------------------------------------------------------- layout
    def initial_loading(self) -> dict[int, Site]:
        g = self.geo
        order = self.opt["order"] or (self.line_order() if self.opt.get("layout") == "line" else self.pair_order())
        sites = [(x, y) for y in self.srows for x in range(g.sx0, g.sx1 + 1)]
        return {q: sites[k] for k, q in enumerate(order) if q is not None}

    def line_order(self) -> list[int | None]:
        """Atoms along a long path of the interaction graph (hub excluded), row by row, so that
        neighbours on the path are neighbours in storage; the hub goes last."""
        import random

        h = self.hub
        adj: dict[int, set[int]] = {q: set() for q in range(self.n)}
        for o in self.ops:
            if o.kind in (ROT, CZB):
                a, b = o.qubits
                if h not in (a, b):
                    adj[a].add(b)
                    adj[b].add(a)
        nodes = [q for q in range(self.n) if q != h]
        rng = random.Random(self.opt.get("seed", 0))
        best: list[int] = []
        for trial in range(200):
            start = nodes[trial % len(nodes)] if trial < len(nodes) else rng.choice(nodes)
            path, seen = [start], {start}
            while True:
                cand = [v for v in adj[path[-1]] if v not in seen]
                if not cand:
                    break
                # Warnsdorff: fewest onward options first
                cand.sort(key=lambda v: (len([w for w in adj[v] if w not in seen]), rng.random()))
                path.append(cand[0])
                seen.add(cand[0])
            if len(path) > len(best):
                best = path
            if len(best) == len(nodes):
                break
        rest = [q for q in nodes if q not in best]
        line = best + rest
        if h is not None:
            line.append(h)
        return line

    def pair_order(self) -> list[int | None]:
        slots: list[int | None] = []
        placed: set[int] = set()
        for o in self.ops:
            if o.kind not in (ROT, CZB):
                continue
            a, b = o.qubits
            if a in placed or b in placed:
                continue
            if len(slots) % 2:
                slots.append(None)
            slots += [a, b]
            placed.update((a, b))
        for q in range(self.n):
            if q not in placed:
                slots.append(q)
        while len(slots) > len(self.geo.storage_sites):
            slots.remove(None)
        return slots

    # ---------------------------------------------------------- helpers on a plan
    def emit_moves(self, P: Plan, transfers):
        for params, dist in group_transfers(P.st, transfers):
            P.chunk.append(("move", params))
            P.dist += dist

    def flush(self, P: Plan, q):
        for name, ang in synth_1q(P.ex.U[q]):
            P.chunk.append(("u", q, name, ang))
            P.lvl[q] += 1
        P.ex.U[q] = I2.copy()

    def set_home(self, P: Plan, q, s):
        old = P.home.get(q)
        if old is not None and P.owner.get(old) == q:
            del P.owner[old]
        P.home[q] = s
        P.owner[s] = q

    def next_partner(self, P: Plan, q) -> int | None:
        ex = P.ex
        for i in self.qblocks[q]:
            if not ex.done[i] and i not in ex.bstate:
                return [x for x in self.ops[i].qubits if x != q][0]
        return None

    def zone_dist(self, P: Plan, q) -> float:
        g = self.geo
        x, y = P.st.pos[q]
        return 0.0 if g.in_ent((x, y)) else float(g.ey0 - y)

    def below(self, P: Plan, q, taken) -> Site:
        g, st = self.geo, P.st
        hm = P.home.get(q)
        if hm is not None and hm not in st.occ and hm not in taken:
            return hm
        x = min(max(st.pos[q][0], g.sx0), g.sx1)
        for y in self.srows:
            s = (x, y)
            if s not in st.occ and s not in taken and P.owner.get(s, q) == q:
                return s
        free = [s for s in g.storage_sites if s not in st.occ and s not in taken and P.owner.get(s, q) == q]
        if not free:
            free = [s for s in g.storage_sites if s not in st.occ and s not in taken]
        return min(free, key=lambda s: g.dist(st.pos[q], s))

    def beside(self, P: Plan, q, taken) -> Site | None:
        p = self.next_partner(P, q)
        if p is None:
            return None
        g, st = self.geo, P.st
        s = st.pos[p] if g.in_storage(st.pos[p]) else P.home.get(p)
        if s is None:
            return None
        x, y = s
        t = (x + 1, y) if (x - g.ex0) % 2 == 0 else (x - 1, y)
        if g.in_storage(t) and t not in st.occ and t not in taken and P.owner.get(t, q) == q:
            return t
        return None

    # ---------------------------------------------------------- pulse candidates
    def pulses(self, P: Plan) -> list[list]:
        """A few candidate pulses: all open blocks finish; new blocks are added greedily in two
        orders (critical path first / cheapest lift first) and with different caps."""
        ex, ops, h, g, st = P.ex, self.ops, self.hub, self.geo, P.st
        base, used = [], set()
        for i, r in sorted(ex.bstate.items(), key=lambda kv: (-self.height[kv[0]], kv[0])):
            qs = ops[i].qubits
            if used & set(qs):
                continue
            base.append(("finish", i, r))
            used.update(qs)

        def lift_cost(i):
            qs = ops[i].qubits
            if h is not None and h in qs:
                p = qs[0] if qs[1] == h else qs[1]
                hs = st.pos[h]
                return g.dist(st.pos[p], g.sibling[hs]) if g.in_ent(hs) else self.zone_dist(P, p) + 0.5
            return max(self.zone_dist(P, q) for q in qs)

        def role(i):
            roles = ex.start_roles(i)
            if not roles or ops[i].kind != ROT:
                return roles[0] if roles else None
            roles.sort(key=lambda r: (r == h, self.zone_dist(P, r)))
            return roles[0]

        if self.opt.get("rounds") and base:
            return [base]  # round mode: an open round only finishes (same rectangles back up)
        ready = [i for i in ex.ready_blocks if ex.start_roles(i)]
        orders = {
            "crit": sorted(ready, key=lambda i: (-self.height[i], lift_cost(i), i)),
            "cheap": sorted(ready, key=lambda i: (lift_cost(i), -self.height[i], i)),
        }
        # geometry: storage neighbours whose block is ready rise together (both alignments)
        geo_sets = []
        rowsmap: dict[int, list[tuple[int, int]]] = {}
        for q in range(self.n):
            x, y = st.pos[q]
            if g.in_storage((x, y)):
                rowsmap.setdefault(y, []).append((x, q))
        pairblock = {}
        for i in ready:
            a, b = ops[i].qubits
            pairblock[frozenset((a, b))] = i
        for par in (0, 1):
            S, u = list(base), set(used)
            for y, items in rowsmap.items():
                items.sort()
                for k in range(len(items) - 1):
                    (x1, a), (x2, b) = items[k], items[k + 1]
                    if x2 != x1 + 1 or (x1 - g.sx0) % 2 != par:
                        continue
                    i = pairblock.get(frozenset((a, b)))
                    if i is None or u & {a, b}:
                        continue
                    r = role(i)
                    if r is None and ops[i].kind == ROT:
                        continue
                    S.append(("start", i, r))
                    u.update((a, b))
            # the hub joins with its cheapest ready partner
            if h is not None and h not in u:
                hb = [i for i in ready if h in ops[i].qubits and not (u & set(ops[i].qubits))]
                if hb:
                    i = min(hb, key=lift_cost)
                    r = role(i)
                    if r is not None or ops[i].kind == CZB:
                        S.append(("start", i, r))
                        u.update(ops[i].qubits)
            geo_sets.append(S)
        out, seen = [], set()
        for S in geo_sets:
            key = tuple(sorted((e[1], e[2] if e[2] is not None else -1) for e in S))
            if S and key not in seen:
                seen.add(key)
                out.append(S)
        for name, order in orders.items():
            for cap in (None, 3, 1):
                S, u = list(base), set(used)
                added = 0
                for i in order:
                    if cap is not None and added >= cap:
                        break
                    qs = ops[i].qubits
                    if u & set(qs):
                        continue
                    r = role(i)
                    if r is None and ops[i].kind == ROT:
                        continue
                    S.append(("start", i, r))
                    u.update(qs)
                    added += 1
                key = tuple(sorted((e[1], e[2] if e[2] is not None else -1) for e in S))
                if S and key not in seen:
                    seen.add(key)
                    out.append(S)
        return out

    # ---------------------------------------------------------- zone sites (order preserving)
    def assign_sites(self, P: Plan, todo, reserved):
        """Pairs whose atoms both come from storage get zone pairs by a monotone assignment: pairs
        sorted by the x of their atoms map onto free pair sites sorted by (row, x), minimising the
        summed lift distance (DP). Returns {qubit: site}."""
        g, st = self.geo, P.st
        if not todo:
            return {}
        slots = [(l, r) for (l, r) in g.pairs if l[1] in self.zrows and l not in st.occ and r not in st.occ
                 and l not in reserved and r not in reserved]
        slots.sort(key=lambda lr: (lr[0][1], lr[0][0]))
        items = []
        for a, b in todo:
            pa, pb = st.pos[a], st.pos[b]
            qa, qb = (a, b) if (pa[0], -pa[1]) <= (pb[0], -pb[1]) else (b, a)
            items.append((min(pa[0], pb[0]), qa, qb))
        items.sort()
        k, m = len(items), len(slots)
        if m < k:
            raise RuntimeError("entangling zone full")
        INF = float("inf")
        cost = [[0.0] * m for _ in range(k)]
        for i, (_, qa, qb) in enumerate(items):
            for j, (l, r) in enumerate(slots):
                cost[i][j] = max(g.dist(st.pos[qa], l), g.dist(st.pos[qb], r))
        # dp over rows of slots is not monotone across rows; restrict to one row band at a time
        best = None
        rows = sorted({l[1] for l, _ in slots})
        for start in range(len(rows)):
            band = [j for j, (l, _) in enumerate(slots) if l[1] in rows[start:]]
            band.sort(key=lambda j: (slots[j][0][0], slots[j][0][1]))
            mb = len(band)
            if mb < k:
                continue
            D = [[INF] * (mb + 1) for _ in range(k + 1)]
            for j in range(mb + 1):
                D[0][j] = 0.0
            for i in range(1, k + 1):
                for j in range(1, mb + 1):
                    D[i][j] = min(D[i][j - 1], D[i - 1][j - 1] + cost[i - 1][band[j - 1]])
            if best is None or D[k][mb] < best[0]:
                # backtrack
                pick, i, j = [], k, mb
                while i > 0:
                    if D[i][j] == D[i][j - 1]:
                        j -= 1
                    else:
                        pick.append(band[j - 1])
                        i, j = i - 1, j - 1
                best = (D[k][mb], list(reversed(pick)))
            break  # the first band (nearest rows) is enough in practice
        res = {}
        for (_, qa, qb), j in zip(items, best[1]):
            l, r = slots[j]
            res[qa], res[qb] = l, r
        return res

    # ---------------------------------------------------------- one pulse on a plan
    def step(self, P: Plan, S, pol):
        g, ops = self.geo, self.ops
        ex, st = P.ex, P.st
        drop_pol, keep_pol = pol
        for ev in S:
            ex.apply_pre(ev)
        pairs = [tuple(ops[ev[1]].qubits) for ev in S]
        Sq = {q for pr in pairs for q in pr}

        def dirty(q):
            return not _is_identity(ex.U[q])

        zone = [q for q in range(self.n) if g.in_ent(st.pos[q])]
        # only the hub (and pairs that stay intact) may stay up; everything else drops, so that
        # drops and lifts move whole rows instead of fragments
        keep = set()
        for a, b in pairs:
            if g.in_ent(st.pos[a]) and g.sibling.get(st.pos[a]) == st.pos[b] and not dirty(a) and not dirty(b):
                keep.update((a, b))
        if keep_pol == "none":
            keep = set()
        elif keep_pol == "clean":
            keep.update(q for q in zone if q in Sq and not dirty(q))
        elif self.hub is not None and self.hub in Sq and not dirty(self.hub) and g.in_ent(st.pos[self.hub]):
            keep.add(self.hub)
        dropping = [q for q in zone if q not in keep]
        taken: set[Site] = set()
        drops = []
        if isinstance(drop_pol, tuple):  # ("shift", dx, row): the group drops as one block
            _, dx, row0 = drop_pol
            for q in dropping:
                x, y = st.pos[q]
                t = (x + dx, row0 - (y - g.ey0))
                if not g.in_storage(t) or t in taken:
                    raise RuntimeError("block drop does not fit")
                o = st.occ.get(t)
                if o is not None:
                    raise RuntimeError("block drop target occupied")
                if P.owner.get(t, q) != q:
                    o2 = P.owner[t]
                    if not g.in_ent(st.pos[o2]):
                        raise RuntimeError("block drop onto a home")
                taken.add(t)
                drops.append((q, st.pos[q], t))
                self.set_home(P, q, t)
            dropping = []
        for q in sorted(dropping, key=lambda q: (q not in Sq, st.pos[q])):
            t = None
            if q not in Sq and drop_pol == "beside":
                t = self.beside(P, q, taken)
            if t is None:
                t = self.below(P, q, taken)
            taken.add(t)
            drops.append((q, st.pos[q], t))
            self.set_home(P, q, t)
        self.emit_moves(P, drops)
        for q in Sq:
            if dirty(q):
                if not g.in_storage(st.pos[q]):
                    raise RuntimeError("dirty atom left in the zone")
                self.flush(P, q)
        reserved: set[Site] = set()
        tg: dict[int, Site] = {}
        todo = []
        for a, b in pairs:
            sa, sb = st.pos[a], st.pos[b]
            if g.in_ent(sa) and g.sibling.get(sa) == sb:
                reserved.update((sa, sb))
                continue
            todo.append((a, b))
        rest = []
        for a, b in todo:
            ok = False
            for anc, mov in ((a, b), (b, a)):
                sa = st.pos[anc]
                if g.in_ent(sa) and sa not in reserved:
                    sib = g.sibling[sa]
                    o = st.occ.get(sib)
                    if (o is None or o == mov) and sib not in reserved:
                        tg[mov] = sib
                        reserved.update((sa, sib))
                        ok = True
                        break
            if not ok:
                rest.append((a, b))
        # atoms of `rest` still in the zone (clean, re-pairing) are treated like storage atoms
        tg.update(self.assign_sites(P, rest, reserved))
        self.emit_moves(P, [(q, st.pos[q], s) for q, s in tg.items() if st.pos[q] != s])
        for a, b in pairs:
            if g.sibling.get(st.pos[a]) != st.pos[b]:
                raise RuntimeError("pair not formed")
        for q in range(self.n):
            if g.in_ent(st.pos[q]) and q not in Sq:
                raise RuntimeError("idle atom in the zone")
        for a, b in pairs:
            P.chunk.append(("cz", a, b))
            m = max(P.lvl[a], P.lvl[b]) + 1
            P.lvl[a] = P.lvl[b] = m
        for ev in S:
            ex.apply_post(ev)
        ex.absorb()
        P.steps += 1

    def finish(self, P: Plan):
        g, st = self.geo, P.st
        taken: set[Site] = set()
        drops = []
        for q in range(self.n):
            if g.in_ent(st.pos[q]):
                t = self.below(P, q, taken)
                taken.add(t)
                drops.append((q, st.pos[q], t))
        self.emit_moves(P, drops)
        for q in range(self.n):
            if not _is_identity(P.ex.U[q]):
                self.flush(P, q)
        for o in sorted(P.ex.pend_meas, key=lambda o: o.idx):
            P.chunk.append(("meas", o.qubits[0], o.clbit))

    # ---------------------------------------------------------- search
    def remaining_cz(self, P: Plan) -> int:
        ex = P.ex
        left = 0
        for i, o in enumerate(self.ops):
            if o.kind in (ROT, CZB) and not ex.done[i]:
                left += 1 if (o.kind == CZB or i in ex.bstate) else 2
        return left

    def score(self, P: Plan) -> float:
        return P.dist + DEPTH_WEIGHT * P.depth() + self.opt["lam"] * self.remaining_cz(P)

    def run(self):
        g = self.geo
        root = Plan.__new__(Plan)
        root.ex = Exec(self.ops, self.succ, self.indeg, self.n)
        root.st = State(g, self.loading)
        root.home, root.owner = {}, {}
        for q, s in self.loading.items():
            self.set_home(root, q, s)
        root.lvl = [0] * self.n
        root.dist, root.parent, root.chunk, root.steps = 0.0, None, [], 0
        beam = [root]
        B = max(1, int(self.opt["beam"]))
        guard = 0
        self.work = 0
        while any(not P.ex.finished() for P in beam):
            guard += 1
            if guard > 4 * len(self.ops) + 50:
                raise RuntimeError("beam planner stuck")
            cands = []
            for P in beam:
                if P.ex.finished():
                    cands.append(P)
                    continue
                if self.opt.get("rounds"):
                    pols = [("below", "none")]
                else:
                    pols = [(d, k) for d in ("beside", "below") for k in ("clean", "hub")] if B > 1 else [("beside", "clean")]
                if self.opt.get("block"):
                    pols = pols + [(("shift", dx, row), k) for dx in (-2, -1, 0, 1, 2) for row in self.srows[:2]
                                   for k in ("hub", "none")]
                for S in self.pulses(P):
                    for pol in pols:
                        Q = P.copy()
                        self.work += 1
                        try:
                            self.step(Q, S, pol)
                        except RuntimeError:
                            continue
                        cands.append(Q)
            if not cands:
                raise RuntimeError("no feasible pulse")
            cands.sort(key=self.score)
            nxt, seen = [], set()
            for Q in cands:
                key = (tuple(Q.st.pos[q] for q in range(self.n)), self.remaining_cz(Q), tuple(sorted(Q.ex.bstate)))
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(Q)
                if len(nxt) >= B:
                    break
            beam = nxt
        finals = []
        for P in beam:
            Q = P.copy()
            self.finish(Q)
            finals.append(Q)
        best = min(finals, key=lambda Q: Q.dist + DEPTH_WEIGHT * Q.depth())
        self.dist_total = best.dist
        self.out = best.output()
        return self.out


# ---------------------------------------------------------------- emit + self-check


def _emit(circuit: QuantumCircuit, prog: list[tuple]) -> QuantumCircuit:
    out = circuit.copy_empty_like()
    out.data.clear() if hasattr(out.data, "clear") else None
    for item in prog:
        if item[0] == "u":
            _, q, name, ang = item
            (out.rx if name == "rx" else out.rz)(ang, q)
        elif item[0] == "cz":
            out.cz(item[1], item[2])
        elif item[0] == "move":
            rows, cols, nr, nc, qs = item[1]
            out.append(Move(rows, cols, nr, nc, len(qs)), list(qs))
        elif item[0] == "meas":
            out.measure(item[1], item[2])
    return out


def self_check(geo: Geo, loading: dict[int, Site], prog: list[tuple]) -> None:
    """Replays positions with the validator's rules plus transversal CZ."""
    st = State(geo, loading)
    seg_cz: dict[frozenset, int] = {}
    seg_pairs: set[frozenset] = set()

    def close_segment():
        if seg_cz:
            counts = set(seg_cz.values())
            if len(counts) != 1:
                raise RuntimeError("non-uniform pulses")
            if IDLE_RULE:
                active = {q for k in seg_cz for q in k}
                if any(geo.in_ent(s) and q not in active for q, s in st.pos.items()):
                    raise RuntimeError("idle atom in the entangling zone during a pulse")
            for left, right in geo.pairs:
                qa, qb = st.occ.get(left), st.occ.get(right)
                if qa is not None and qb is not None and frozenset((qa, qb)) not in seg_cz:
                    raise RuntimeError("occupied pair without cz")
        seg_cz.clear()

    for item in prog:
        if item[0] == "move":
            close_segment()
            rows, cols, nr, nc, qs = item[1]
            rmap = dict(zip([y for y in rows if any(st.pos[q][1] == y for q in qs)], nr, strict=True))
            cmap = dict(zip([x for x in cols if any(st.pos[q][0] == x for q in qs)], nc, strict=True))
            st.apply([(q, (cmap[st.pos[q][0]], rmap[st.pos[q][1]])) for q in qs])
        elif item[0] == "u":
            if not geo.in_storage(st.pos[item[1]]):
                raise RuntimeError("1q gate outside storage")
        elif item[0] == "cz":
            a, b = item[1], item[2]
            if geo.sibling.get(st.pos[a]) != st.pos[b]:
                raise RuntimeError("cz not on a pair")
            k = frozenset((a, b))
            seg_cz[k] = seg_cz.get(k, 0) + 1
    close_segment()
    del seg_pairs


# ---------------------------------------------------------------- naive fallback (from compilers/naive)


def _naive(circuit: QuantumCircuit, spec: Any):
    n = circuit.num_qubits
    native = transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=0)
    loading = {q: (q % 10, q // 10) for q in range(n)}
    pos = dict(loading)
    out = native.copy_empty_like()
    index = {bit: i for i, bit in enumerate(native.qubits)}

    def mv(qs, rows, cols, nr, nc):
        out.append(Move(rows, cols, nr, nc, len(qs)), list(qs))

    for instruction in native.data:
        name = instruction.operation.name
        qubits = [index[q] for q in instruction.qubits]
        if name != "cz":
            out.append(instruction)
            continue
        a, b = sorted(qubits, key=lambda q: pos[q])
        (xa, ya), (xb, yb) = pos[a], pos[b]
        if ya == yb and xb == xa + 1:
            c = xa - xa % 2
            mv([a, b], (ya,), (xa, xb), (10,), (c, c + 1))
            out.append(instruction)
            mv([a, b], (10,), (c, c + 1), (ya,), (xa, xb))
        else:
            k = xa - xa % 2
            mv([a], (ya,), (xa,), (10,), (k,))
            mv([b], (yb,), (xb,), (10,), (k + 1,))
            out.append(instruction)
            mv([a], (10,), (k,), (ya,), (xa,))
            mv([b], (10,), (k + 1,), (yb,), (xb,))
    return out, Architecture(loading, spec=spec)


# ---------------------------------------------------------------- entry point


BEAM_BUDGET = 15000  # beam width = BEAM_BUDGET / (atoms x ops): ~10 at 12q p1, 1 at 20q p3 and 50q
IDLE_RULE = True  # validator v1.0.0: every atom in the entangling zone must be in a CZ of each pulse
DEV_FAST = False  # dev iteration only (bench/devbench.sh flips it in a scratch copy): one plan, no search
DEPTH_WEIGHT = 276726.912853 / (4 * 23676)  # one depth unit = 2.92 distance units in the score


def _depth(out: QuantumCircuit) -> int:
    stripped = out.copy_empty_like()
    for ins in out.data:
        if ins.operation.name not in ("move", "barrier"):
            stripped.append(ins)
    return stripped.depth()


def compile_with(circuit: QuantumCircuit, spec: Any, options: dict | None = None):
    options = options or {}
    comp = Compiler(circuit, spec, options)
    hub = find_hub(comp.ops, comp.n)
    if IDLE_RULE and options.get("planner", "pulse") == "pulse":
        # deterministic effort: beam width from the problem size (n atoms x ops), both layouts
        size = max(1, comp.n * len(comp.ops))
        beam = 1 if (DEV_FAST or comp.n > 12) else max(1, min(24, int(BEAM_BUDGET / size)))
        if options.get("pulse_opts"):
            variants = [options["pulse_opts"]]
        else:
            variants = [{"beam": beam, "layout": "line"}]
            if beam >= 2:
                variants.append({"beam": beam, "layout": "pair"})
        best = None
        for v in variants:
            try:
                pp = BeamPlanner(comp.geo, comp.ops, comp.succ, comp.indeg, comp.n, v)
                prog = pp.run()
                self_check(comp.geo, pp.loading, prog)
            except (RuntimeError, Unsupported):
                continue
            out = _emit(circuit, prog)
            cost = pp.dist_total + DEPTH_WEIGHT * _depth(out)
            if best is None or cost < best[0]:
                best = (cost, out, pp)
        if best is None:
            raise RuntimeError("no plan")
        return best[1], Architecture(best[2].loading, spec=spec), best[2]
    if hub is not None and not IDLE_RULE and options.get("planner", "cycle") == "cycle":
        dev = DEV_FAST
        if dev:
            options = {**options, "budget": -1}
        orders = [None] + ([] if dev else [sa_order(comp.geo, comp.ops, comp.n, hub, seed) for seed in (1, 2, 3)])
        space = {
            "order": list(range(len(orders))),
            "match": ["index", "work"],
            "hub_max": [1.0, 0.75, 0.5, 0.25],
            "slack": [None, 0, 1],
            "hub_min": [0.0, 0.6],
            "trips": [False, True],
        }
        cur = {k: v[0] for k, v in space.items()}
        cur["order"] = 2 if len(orders) > 2 else 0
        space["order"] = list(range(len(orders)))
        fixed = {k: options[k] for k in space if k in options}
        cur.update(fixed)
        seen: dict = {}

        def evaluate(cfg):
            key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
            if key in seen:
                return seen[key]
            res = None
            try:
                cp = CyclePlanner(comp.geo, comp.ops, comp.succ, comp.indeg, comp.n, hub, orders[cfg["order"]],
                                  cfg["match"], cfg["hub_min"], cfg["slack"], cfg["hub_max"], cfg["trips"])
                try:
                    prog = cp.run()
                finally:
                    spent[0] += cp.work
                self_check(comp.geo, cp.loading, prog)
                out = _emit(circuit, prog)
                res = (cp.dist_total + DEPTH_WEIGHT * _depth(out), out, cp)
            except (RuntimeError, Unsupported):
                res = None
            seen[key] = res
            return res

        spent = [0]
        budget = options.get("budget", 20000)
        best = evaluate(cur)
        for _ in range(options.get("sweeps", 1)):
            for k, vals in space.items():
                if k in fixed:
                    continue
                for v in vals:
                    if v == cur[k] or spent[0] > budget:
                        continue
                    trial = dict(cur)
                    trial[k] = v
                    r = evaluate(trial)
                    if r is not None and (best is None or r[0] < best[0] - 1e-9):
                        best, cur = r, trial
        if best is not None:
            return best[1], Architecture(best[2].loading, spec=spec), best[2]
    prog = comp.run()
    self_check(comp.geo, comp.loading, prog)
    return _emit(circuit, prog), Architecture(comp.loading, spec=spec), comp


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any):
    if circuit.num_qubits > 50 or len(circuit.data) > 5000:
        return _naive(circuit, architecture_spec)
    try:
        out, arch, _ = compile_with(circuit, architecture_spec)
        return out, arch
    except (Unsupported, RuntimeError, KeyError, IndexError, ValueError, AssertionError):
        return _naive(circuit, architecture_spec)
