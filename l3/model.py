"""Abstract L3 model: a compiled circuit reduced to its pulse schedule, and back.

Schedule (the L3 input):
    pulses[k]  = list of pairs (a, b) that must sit on one interaction pair at pulse k; nobody else
                 may be in the entangling zone then (idle rule).
    seg[q][i]  = 1q instructions of atom q between its (i-1)-th and i-th CZ (i = 0: before the
                 first CZ; i = ncz[q]: after the last). They need the atom in storage somewhere in
                 that window.
    owe[k]     = atoms of pulse k whose segment right before pulse k is non-empty (the `s` marks).
    meas       = final measurement instructions.

Gates and their per-qubit order are kept exactly, so any plan that realises the schedule is
equivalent to the source and has the same gate depth (or less, after cancelling cz.cz pairs).
"""
from __future__ import annotations

import itertools
import math
import pickle
from dataclasses import dataclass, field

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction

# ---------------------------------------------------------------- geometry (default spec)

SX, SY0, SY1 = 10, 0, 9      # storage: x 0..9, y 0..9
EY0, EY1 = 10, 19            # entangling: x 0..9, y 10..19, pairs (2k, 2k+1), 0.3 apart
PAIR_D = 0.3


def phys(s):
    x, y = s
    if y >= EY0:
        p, r = divmod(x, 2)
        return (2.0 * p + r * PAIR_D, float(y))
    return (float(x), float(y))


def dist(a, b):
    (ax, ay), (bx, by) = phys(a), phys(b)
    return math.hypot(ax - bx, ay - by)


def in_zone(s):
    return s[1] >= EY0


def zone_of(s):
    return "e" if s[1] >= EY0 else "s"


class Move(Instruction):
    def __init__(self, rows, cols, new_rows, new_cols, num_qubits):
        super().__init__("move", num_qubits, 0, [tuple(rows), tuple(cols), tuple(new_rows), tuple(new_cols)])

    def validate_parameter(self, parameter):
        return parameter

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


# ---------------------------------------------------------------- state + legal moves


class MoveError(ValueError):
    pass


class State:
    def __init__(self, pos):
        self.pos = dict(pos)
        self.occ = {s: q for q, s in self.pos.items()}

    def copy(self):
        st = State.__new__(State)
        st.pos, st.occ = dict(self.pos), dict(self.occ)
        return st

    def check(self, group):
        """group = {q: target}. Returns (move params, cost) or raises MoveError. Validator rules +
        single-zone source/target rectangles."""
        src = {q: self.pos[q] for q in group}
        xs = [s[0] for s in src.values()]
        ys = [s[1] for s in src.values()]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                o = self.occ.get((x, y))
                if o is not None and o not in group:
                    raise MoveError("ghost")
        rmap, cmap = {}, {}
        for q, (sx, sy) in src.items():
            tx, ty = group[q]
            if rmap.setdefault(sy, ty) != ty or cmap.setdefault(sx, tx) != tx:
                raise MoveError("not a product map")
        for mp in (rmap, cmap):
            ks = sorted(mp)
            if any(mp[a] >= mp[b] for a, b in itertools.pairwise(ks)):
                raise MoveError("order")
        if len({zone_of(s) for s in src.values()}) != 1 or len({zone_of(t) for t in group.values()}) != 1:
            raise MoveError("zones")
        for t in group.values():
            if not (0 <= t[0] <= 9 and 0 <= t[1] <= 19):
                raise MoveError("outside")
            o = self.occ.get(t)
            if o is not None and o not in group:
                raise MoveError("occupied")
        cost = max(dist(src[q], group[q]) for q in group)
        params = (tuple(range(y0, y1 + 1)), tuple(range(x0, x1 + 1)),
                  tuple(rmap[y] for y in sorted(rmap)), tuple(cmap[x] for x in sorted(cmap)))
        return params, cost

    def ok(self, group):
        try:
            return self.check(group)
        except MoveError:
            return None

    def apply(self, group):
        params, cost = self.check(group)
        for q in group:
            del self.occ[self.pos[q]]
        for q, t in group.items():
            self.pos[q] = t
            self.occ[t] = q
        return params, cost


# ---------------------------------------------------------------- schedule


@dataclass
class Schedule:
    n: int
    pulses: list            # [[(a, b), ...], ...]
    seg: list               # seg[q] = [[instr, ...], ...]
    meas: list              # [(instruction, qubits, clbits)]
    template: QuantumCircuit = field(repr=False, default=None)

    @property
    def K(self):
        return len(self.pulses)

    def cz_index(self):
        """ev[q] = list of (pulse k, partner) in order."""
        ev = [[] for _ in range(self.n)]
        for k, P in enumerate(self.pulses):
            for a, b in P:
                ev[a].append((k, b))
                ev[b].append((k, a))
        return ev

    def owe(self):
        """owe[k] = atoms of pulse k with a non-empty segment right before pulse k."""
        ev = self.cz_index()
        o = [set() for _ in range(self.K)]
        for q in range(self.n):
            for i, (k, _) in enumerate(ev[q]):
                if self.seg[q][i]:
                    o[k].add(q)
        return o

    def final_owe(self):
        ev = self.cz_index()
        return {q for q in range(self.n) if self.seg[q][len(ev[q])]}

    def members(self):
        return [{q for p in P for q in p} for P in self.pulses]


def extract(circuit: QuantumCircuit) -> Schedule:
    qi = {b: i for i, b in enumerate(circuit.qubits)}
    n = circuit.num_qubits
    pulses, seg, meas = [], [[[]] for _ in range(n)], []
    cur = None  # list of pulses of the current CZ layer

    def close():
        nonlocal cur
        if cur:
            pulses.extend(cur)
        cur = None

    for ins in circuit.data:
        name = ins.operation.name
        qs = [qi[q] for q in ins.qubits]
        if name == "cz":
            if cur is None:
                cur = [[]]
            a, b = qs
            # a pair seen again in this layer starts the next pulse of the layer
            k = 0
            while k < len(cur) and any(a in p or b in p for p in cur[k]):
                k += 1
            if k == len(cur):
                cur.append([])
            cur[k].append((a, b))
            continue
        if name == "move":
            close()
            continue
        if name in ("rx", "rz"):
            if cur is not None and any(qs[0] in p for P in cur for p in P):
                raise ValueError("1q gate on a pulsed atom inside a CZ layer")
        elif name == "measure":
            meas.append((ins.operation, qs, [circuit.find_bit(c).index for c in ins.clbits]))
        elif name == "barrier":
            continue
        else:
            raise ValueError("unexpected op " + name)
    close()
    # segments in CZ order: a new segment per CZ of the atom
    seg = [[[]] for _ in range(n)]
    for ins in circuit.data:
        name = ins.operation.name
        qs = [qi[q] for q in ins.qubits]
        if name == "cz":
            for q in qs:
                seg[q].append([])
        elif name in ("rx", "rz"):
            seg[qs[0]][-1].append(ins.operation)
    sch = Schedule(n, pulses, seg, meas, circuit.copy_empty_like())
    return sch


def cancel_dummies(sch: Schedule) -> int:
    """Remove cz.cz pairs with nothing in between on either atom (identity). Returns #removed CZ."""
    removed = 0
    while True:
        ev = sch.cz_index()
        kill = None
        for q in range(sch.n):
            for i in range(len(ev[q]) - 1):
                (k1, p1), (k2, p2) = ev[q][i], ev[q][i + 1]
                if p1 != p2 or sch.seg[q][i + 1]:
                    continue
                j = ev[p1].index((k1, q))
                if j + 1 < len(ev[p1]) and ev[p1][j + 1] == (k2, q) and not sch.seg[p1][j + 1]:
                    kill = (q, i, p1, j, k1, k2)
                    break
            if kill:
                break
        if not kill:
            break
        q, i, p, j, k1, k2 = kill
        for a, ia in ((q, i), (p, j)):
            s = sch.seg[a]
            s[ia:ia + 3] = [s[ia] + s[ia + 2]]   # merge segment before cz#1 and after cz#2
        for k in (k1, k2):
            sch.pulses[k] = [pr for pr in sch.pulses[k] if set(pr) != {q, p}]
        removed += 2
    sch.pulses = [P for P in sch.pulses if P]
    return removed


def load_cached(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    qc = QuantumCircuit(d["nq"], d["nc"])
    for name, qs, cs, params in d["ops"]:
        if name == "move":
            qc.append(Move(*params, len(qs)), qs)
        elif name == "rx":
            qc.rx(params[0], qs[0])
        elif name == "rz":
            qc.rz(params[0], qs[0])
        elif name == "cz":
            qc.cz(*qs)
        elif name == "measure":
            qc.measure(qs[0], cs[0])
        elif name == "barrier":
            qc.barrier(*qs)
        else:
            raise ValueError(name)
    return qc, {int(q): tuple(s) for q, s in d["loading"].items()}


# ---------------------------------------------------------------- plan -> circuit


class Emitter:
    """Realise a schedule step by step. The planner calls move()/pulse(); 1q segments are emitted as
    soon as their atom is in storage and its previous CZ is done. Checks every rule as it goes."""

    def __init__(self, sch: Schedule, loading):
        self.sch = sch
        self.st = State(loading)
        self.loading = dict(loading)
        self.out = sch.template.copy_empty_like()
        self.ev = sch.cz_index()
        self.done = [0] * sch.n          # CZs done per atom
        self.emitted = [-1] * sch.n      # index of the last emitted segment
        self.k = 0                       # next pulse
        self.dist = 0.0
        self.nmoves = 0
        self._flush(range(sch.n))

    def _flush(self, qs):
        for q in qs:
            i = self.done[q]
            if self.emitted[q] < i and not in_zone(self.st.pos[q]):
                for op in self.sch.seg[q][i]:
                    self.out.append(op, [self.out.qubits[q]])
                self.emitted[q] = i

    def pending(self, q):
        """True if q still owes its current segment (must visit storage before its next CZ)."""
        i = self.done[q]
        return self.emitted[q] < i and bool(self.sch.seg[q][i])

    def move(self, group):
        if not group:
            return 0.0
        params, cost = self.st.apply(group)
        qs = sorted(group)
        self.out.append(Move(*params, len(qs)), [self.out.qubits[q] for q in qs])
        self.dist += cost
        self.nmoves += 1
        self._flush(qs)
        return cost

    def pulse(self):
        k = self.k
        P = self.sch.pulses[k]
        mem = {q for p in P for q in p}
        zone = {q for q, s in self.st.pos.items() if in_zone(s)}
        if zone != mem:
            raise MoveError(f"pulse {k}: zone {sorted(zone)} != members {sorted(mem)}")
        for a, b in P:
            (xa, ya), (xb, yb) = self.st.pos[a], self.st.pos[b]
            if ya != yb or min(xa, xb) % 2 or abs(xa - xb) != 1:
                raise MoveError(f"pulse {k}: {a},{b} not on a pair")
            for q in (a, b):
                i = self.done[q]
                if self.emitted[q] < i and self.sch.seg[q][i]:
                    raise MoveError(f"pulse {k}: atom {q} still owes 1q gates")
                self.emitted[q] = max(self.emitted[q], i)
        for a, b in P:
            self.out.cz(a, b)
        for a, b in P:
            self.done[a] += 1
            self.done[b] += 1
        self.k += 1
        self._flush(mem)

    def finish(self):
        if self.k != self.sch.K:
            raise MoveError("not all pulses done")
        for q in range(self.sch.n):
            if self.pending(q):
                raise MoveError(f"atom {q} ends with unemitted gates")
        for op, qs, cs in self.sch.meas:
            self.out.append(op, [self.out.qubits[q] for q in qs], [self.out.clbits[c] for c in cs])
        return self.out, self.loading


def schedule_signature(sch: Schedule):
    ev = sch.cz_index()
    sig = []
    for q in range(sch.n):
        s = []
        for i in range(len(ev[q]) + 1):
            s += [(op.name, tuple(round(float(p), 12) for p in op.params)) for op in sch.seg[q][i]]
            if i < len(ev[q]):
                s.append(("cz", ev[q][i][1]))
        sig.append(s)
    return sig


def signature(circuit: QuantumCircuit):
    """Per-qubit sequence of (gate, params, partner) with moves/barriers removed: equal signatures
    (with consistent CZ pairing) mean an identical gate DAG, hence equivalence and equal depth."""
    qi = {b: i for i, b in enumerate(circuit.qubits)}
    sig = [[] for _ in range(circuit.num_qubits)]
    for ins in circuit.data:
        name = ins.operation.name
        if name in ("move", "barrier", "measure"):
            continue
        qs = [qi[q] for q in ins.qubits]
        if name == "cz":
            a, b = qs
            sig[a].append(("cz", b))
            sig[b].append(("cz", a))
        else:
            sig[qs[0]].append((name, tuple(round(float(p), 12) for p in ins.operation.params)))
    return sig
