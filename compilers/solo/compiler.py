"""solo: a compiler specialised on the scored iHVA circuits (validator v5, `idle` rule).

Model
-----
* An iHVA circuit is an `h` layer, then blocks `rx(pi/2)_t cx(c,t) rz(th)_t cx(c,t) rx(-pi/2)_t`,
  then measurements.  Each block is the Pauli rotation exp(-i th/2 Z_c (x) M_t) with
  M = Rx(-pi/2) Z Rx(pi/2).  It is realised exactly as
      t: rx(pi/2), h, [cz], rx(th), [cz], h, rx(-pi/2)        (c: nothing)
  or, with the control as the rotating atom ("swapped"),
      c: h, [cz], rx(th), [cz], h       t: rx(pi/2), [cz], -, [cz], rx(-pi/2).
  th == 0 exactly makes the block the identity (elided, exact).
* Two blocks commute iff their Pauli strings commute; the exact commutation DAG gives the
  freedom to reorder.
* Single-qubit gates are buffered per qubit as a 2x2 unitary.  A diagonal buffer commutes with cz
  and stays pending; a non-diagonal buffer is flushed (ZXZ Euler) while the atom is in storage.
* Physical: every atom has a home in storage.  Between cz layers the zone holds exactly the pulsed
  pairs (`idle` rule).  Every move is checked against the validator's rules before emission.

Anything that is not iHVA-shaped goes to a simple valid fallback (the naive scheme).
"""

import itertools
import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction

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


def _move(circuit: QuantumCircuit, qubits: Sequence[int], rows, cols, new_rows, new_cols) -> None:
    circuit.append(Move(rows, cols, new_rows, new_cols, len(qubits)), list(qubits))


# ---------------------------------------------------------------- fallback (naive, always valid)


def _naive(circuit: QuantumCircuit, architecture_spec: Any):
    n = circuit.num_qubits
    native = transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=0)
    loading = {q: (q % 10, q // 10) for q in range(n)}
    pos = dict(loading)
    out = native.copy_empty_like()
    index = {bit: i for i, bit in enumerate(native.qubits)}
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
            _move(out, [a, b], (ya,), (xa, xb), (10,), (c, c + 1))
            out.append(instruction)
            _move(out, [a, b], (10,), (c, c + 1), (ya,), (xa, xb))
        else:
            k = xa - xa % 2
            _move(out, [a], (ya,), (xa,), (10,), (k,))
            _move(out, [b], (yb,), (xb,), (10,), (k + 1,))
            out.append(instruction)
            _move(out, [a], (10,), (k,), (ya,), (xa,))
            _move(out, [b], (10,), (k + 1,), (yb,), (xb,))
    return out, Architecture(loading, spec=architecture_spec)


# ---------------------------------------------------------------- 1q algebra

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


_PRE_T = _H @ _rx(math.pi / 2)  # target, normal form: rx(pi/2) then h
_POST_T = _rx(-math.pi / 2) @ _H
_PRE_TS = _rx(math.pi / 2)  # target, swapped form
_POST_TS = _rx(-math.pi / 2)


def _is_diag(u, eps=1e-9):
    return abs(u[0, 1]) < eps and abs(u[1, 0]) < eps


def _is_id(u, eps=1e-9):
    return _is_diag(u, eps) and abs(u[0, 0] / u[1, 1] - 1) < eps


def _norm_angle(a):
    a = math.fmod(a, 4 * math.pi)
    if a > 2 * math.pi:
        a -= 4 * math.pi
    if a <= -2 * math.pi:
        a += 4 * math.pi
    return a


def _zxz(u):
    """[(name, angle)] in time order with u == phase * product, angles of ~0 dropped."""
    det = np.linalg.det(u)
    v = u / np.sqrt(det)  # SU(2)
    a, b = v[0, 0], v[0, 1]
    if abs(b) < 1e-10:
        lam = -2 * np.angle(a)
        return [("rz", lam)] if abs(_wrap(lam)) > 1e-9 else []
    if abs(a) < 1e-10:
        # v = Rz(phi) Rx(pi) Rz(lam): only phi+... fix lam = 0
        th = math.pi
        # Rz(phi) Rx(pi) = [[0, -i e^{-i phi/2}], [-i e^{i phi/2}, 0]]
        phi = 2 * np.angle(1j * b) * -1
        seq = [("rx", th), ("rz", phi)]
        return [(g, t) for g, t in seq if abs(_wrap(t)) > 1e-9]
    # v = Rz(phi) Rx(th) Rz(lam); v00 = cos(th/2) e^{-i(phi+lam)/2}, v01 = -i sin(th/2) e^{-i(phi-lam)/2}
    th = 2 * math.atan2(abs(b), abs(a))
    s = -np.angle(a) * 2  # phi + lam
    d = -np.angle(1j * b) * 2  # phi - lam
    phi, lam = (s + d) / 2, (s - d) / 2
    seq = [("rz", lam), ("rx", th), ("rz", phi)]
    return [(g, t) for g, t in seq if abs(_wrap(t)) > 1e-9]


def _wrap(t):
    """Angle modulo 2*pi in (-pi, pi] (a 2*pi rotation is -I, a global phase on one qubit)."""
    t = math.fmod(t, 2 * math.pi)
    if t > math.pi:
        t -= 2 * math.pi
    if t <= -math.pi:
        t += 2 * math.pi
    return t


def _gate_matrix(seq):
    m = _I2.copy()
    for g, t in seq:
        m = (_rz(t) if g == "rz" else _rx(t)) @ m
    return m


# ---------------------------------------------------------------- parsing


class Block:
    __slots__ = ("c", "t", "th", "idx")

    def __init__(self, c, t, th, idx):
        self.c, self.t, self.th, self.idx = c, t, th, idx

    def qubits(self):
        return (self.c, self.t)


def _close(a, b):
    return abs(a - b) < 1e-12


def parse_ihva(circuit: QuantumCircuit):
    """(n, blocks) for an iHVA-shaped circuit, None otherwise."""
    n = circuit.num_qubits
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    ops = []
    for inst in circuit.data:
        name = inst.operation.name
        qs = [index[q] for q in inst.qubits]
        if name == "barrier":
            continue
        params = []
        for p in inst.operation.params:
            try:
                params.append(float(p))
            except (TypeError, ValueError):
                return None
        ops.append((name, qs, params, [circuit.find_bit(c).index for c in inst.clbits]))
    i = 0
    if circuit.num_clbits < n:
        return None
    seen_h = set()
    while i < len(ops) and ops[i][0] == "h":
        seen_h.add(ops[i][1][0])
        i += 1
    if seen_h != set(range(n)):
        return None
    blocks = []
    while i + 4 < len(ops) and ops[i][0] == "rx":
        a, b, c, d, e = ops[i : i + 5]
        if not (a[0] == "rx" and b[0] == "cx" and c[0] == "rz" and d[0] == "cx" and e[0] == "rx"):
            return None
        t = a[1][0]
        ctl, tgt = b[1]
        if tgt != t or d[1] != [ctl, tgt] or c[1] != [t] or e[1] != [t] or ctl == t:
            return None
        if not (_close(a[2][0], math.pi / 2) and _close(e[2][0], -math.pi / 2)):
            return None
        blocks.append(Block(ctl, t, c[2][0], len(blocks)))
        i += 5
    meas = ops[i:]
    if len(meas) != n or any(m[0] != "measure" or m[1] != m[3] for m in meas):
        return None
    if {m[1][0] for m in meas} != set(range(n)):
        return None
    return n, blocks


def _anticommute(a, b):
    pa = {a.c: "Z", a.t: "Y"}
    k = 0
    for q, l in ((b.c, "Z"), (b.t, "Y")):
        if q in pa and pa[q] != l:
            k += 1
    return k % 2 == 1


def build_dag(blocks):
    m = len(blocks)
    preds = [set() for _ in range(m)]
    by_q = {}
    for j, b in enumerate(blocks):
        cand = set()
        for q in b.qubits():
            cand.update(by_q.get(q, ()))
        for i in cand:
            if _anticommute(blocks[i], b):
                preds[j].add(i)
        for q in b.qubits():
            by_q.setdefault(q, []).append(j)
    succs = [set() for _ in range(m)]
    for j in range(m):
        for i in preds[j]:
            succs[i].add(j)
    return preds, succs


# ---------------------------------------------------------------- machine


class Illegal(Exception):
    pass


def _phys(site):
    x, y = site
    if y >= 10:
        return (2 * (x // 2) + 0.3 * (x % 2), float(y))
    return (float(x), float(y))


def _dist(a, b):
    pa, pb = _phys(a), _phys(b)
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _zone(y):
    return 1 if y >= 10 else 0


class Machine:
    def __init__(self, n, loading, emit=True):
        self.n = n
        self.pos = dict(loading)
        self.occ = {s: q for q, s in loading.items()}
        self.emit = emit
        self.qc = QuantumCircuit(n, n) if emit else None
        self.dist = 0.0
        self.nmoves = 0
        self.pend = [_H.copy() for _ in range(n)]
        self.depth = [0] * n  # rough per-qubit depth estimate (for cost models)

    # -- moves
    def move(self, mapping):
        """mapping: qubit -> target site. Emits one legal rectangle move (validator rules)."""
        mapping = dict(mapping)
        if not mapping:
            return
        src = {q: self.pos[q] for q in mapping}
        ys = sorted({s[1] for s in src.values()})
        xs = sorted({s[0] for s in src.values()})
        rmap, cmap = {}, {}
        for q, (x, y) in src.items():
            tx, ty = mapping[q]
            if rmap.setdefault(y, ty) != ty or cmap.setdefault(x, tx) != tx:
                raise Illegal("inconsistent line map")
        if any(rmap[a] >= rmap[b] for a, b in itertools.pairwise(ys)):
            raise Illegal("row order")
        if any(cmap[a] >= cmap[b] for a, b in itertools.pairwise(xs)):
            raise Illegal("col order")
        rows = list(range(ys[0], ys[-1] + 1))
        cols = list(range(xs[0], xs[-1] + 1))
        # ghosts: every atom on the rectangle must be listed
        for (x, y), q in self.occ.items():
            if ys[0] <= y <= ys[-1] and xs[0] <= x <= xs[-1] and q not in mapping:
                if y in rmap and x in cmap:
                    raise Illegal(f"ghost {q}")
                if len(ys) == len(rows) and len(xs) == len(cols):
                    raise Illegal(f"ghost {q}")
                # on an empty line that is squeezed out: picked up anyway (validator selects it)
                raise Illegal(f"ghost {q} (squeeze)")
        for q, d in mapping.items():
            if not (0 <= d[0] <= 9 and 0 <= d[1] <= 19):
                raise Illegal("target outside")
            o = self.occ.get(d)
            if o is not None and o not in mapping:
                raise Illegal("deposit on occupied")
        if len({_zone(s[1]) for s in src.values()}) > 1 or len({_zone(d[1]) for d in mapping.values()}) > 1:
            raise Illegal("zone")
        new_rows = [rmap[y] for y in ys]
        new_cols = [cmap[x] for x in xs]
        if len(set(new_rows)) != len(new_rows) or len(set(new_cols)) != len(new_cols):
            raise Illegal("dup")
        d = max(_dist(src[q], mapping[q]) for q in mapping)
        if d == 0:
            return
        qs = sorted(mapping)
        if self.emit:
            _move(self.qc, qs, rows, cols, new_rows, new_cols)
        for q in qs:
            del self.occ[src[q]]
        for q in qs:
            self.pos[q] = mapping[q]
            self.occ[mapping[q]] = q
        self.dist += d
        self.nmoves += 1

    def can_move(self, mapping):
        try:
            self._check(mapping)
            return True
        except Illegal:
            return False

    def _check(self, mapping):
        saved = self.emit, self.qc, self.dist, self.nmoves, dict(self.pos), dict(self.occ)
        self.emit = False
        try:
            self.move(mapping)
        finally:
            self.emit, self.qc, self.dist, self.nmoves, self.pos, self.occ = saved

    # -- gates
    def apply1(self, q, u):
        self.pend[q] = u @ self.pend[q]

    def flush(self, q, force=False):
        u = self.pend[q]
        if _is_id(u):
            self.pend[q] = _I2.copy()
            return
        if not force and _is_diag(u):
            return
        if self.pos[q][1] >= 10:
            raise Illegal(f"1q gate on {q} in zone")
        seq = _zxz(u)
        if self.emit:
            for g, t in seq:
                if g == "rz":
                    self.qc.rz(t, q)
                else:
                    self.qc.rx(t, q)
        self.depth[q] += len(seq)
        self.pend[q] = _I2.copy()

    def pulse(self, pairs):
        inzone = {q for q, s in self.pos.items() if s[1] >= 10}
        used = set()
        for a, b in pairs:
            (xa, ya), (xb, yb) = self.pos[a], self.pos[b]
            if ya < 10 or ya != yb or min(xa, xb) % 2 or abs(xa - xb) != 1:
                raise Illegal("not a pair")
            for q in (a, b):
                if not _is_diag(self.pend[q]):
                    raise Illegal(f"non-diagonal pending on {q} at cz")
            used |= {a, b}
        if used != inzone:
            raise Illegal("idle atoms in zone")
        for a, b in pairs:
            if self.emit:
                self.qc.cz(a, b)
            dd = max(self.depth[a], self.depth[b]) + 1
            self.depth[a] = self.depth[b] = dd

    def finish(self):
        for q in range(self.n):
            if self.pos[q][1] >= 10:
                raise Illegal("atom left in zone")
            self.flush(q, force=True)
        if self.emit:
            for q in range(self.n):
                self.qc.measure(q, q)


# ---------------------------------------------------------------- scheduling


def schedule(blocks, preds, succs, hub, policy="look"):
    """Steps: ("hub", [block...], role) or ("round", [block...])."""
    m = len(blocks)
    npred = [len(p) for p in preds]
    ready = {i for i in range(m) if npred[i] == 0}
    steps = []
    done = 0
    while ready:
        hubZ = sorted(i for i in ready if blocks[i].c == hub)
        hubY = sorted(i for i in ready if blocks[i].t == hub)
        non = sorted(i for i in ready if hub not in blocks[i].qubits())
        # greedy maximal matching of non-hub ready blocks (by unlock count, then index)
        match = []
        if non:
            used = set()
            for i in sorted(non, key=lambda i: (-len(succs[i]), i)):
                b = blocks[i]
                if b.c in used or b.t in used:
                    continue
                used |= {b.c, b.t}
                match.append(i)
        cands = []
        if match:
            cands.append(("round", match))
        if hubZ:
            cands.append(("hub", hubZ))
        if hubY:
            cands.append(("hub", hubY))

        def gain(sel):
            k = 0
            for i in sel:
                for j in succs[i]:
                    if npred[j] == sum(1 for p in preds[j] if p in sel) and j not in ready:
                        k += 1
            return len(sel) + 2 * k

        if policy == "look":
            kind, sel = max(cands, key=lambda c: (gain(c[1]), c[0] == "round"))
        else:
            kind, sel = cands[0]
        if kind == "round" and len(sel) == 0:
            continue
        steps.append((kind, [blocks[i] for i in sel]))
        for i in sel:
            ready.discard(i)
            done += 1
            for j in succs[i]:
                npred[j] -= 1
                if npred[j] == 0:
                    ready.add(j)
    assert done == m
    return steps


# ---------------------------------------------------------------- executor


class Executor:
    def __init__(self, n, hub, home, emit=True):
        self.n, self.hub, self.home = n, hub, dict(home)
        self.m = Machine(n, home, emit)

    # helpers
    def free(self, s):
        return s not in self.m.occ

    def go(self, q, dst):
        self.m.move({q: dst})

    def prep(self, q, u):
        """Queue `u` before q's next cz; flush if non-diagonal (q must be in storage)."""
        self.m.apply1(q, u)
        self.m.flush(q)

    def zone_slot_for(self, blocked, near):
        """Free zone pair (row, j) closest to `near` (storage site)."""
        best = None
        for z in range(10, 20):
            for j in range(5):
                if (z, j) in blocked:
                    continue
                if not (self.free((2 * j, z)) and self.free((2 * j + 1, z))):
                    continue
                d = _dist(near, (2 * j, z))
                if best is None or d < best[0]:
                    best = (d, (z, j))
        return best[1]

    def run_round(self, blks):
        m = self.m
        roles = []  # (k, r, theta, pre_k, pre_r, post_k, post_r)
        for b in blks:
            roles.append((b.c, b.t, b.th, None, _PRE_T, None, _POST_T))
        # placement: pair (k, r) on zone pair (z, j), k left
        blocked = set()
        place = {}
        for k, r, *_ in sorted(roles, key=lambda x: (-max(self.m.pos[x[0]][1], self.m.pos[x[1]][1]), x[0])):
            hk, hr = m.pos[k], m.pos[r]
            mid = ((hk[0] + hr[0]) // 2, max(hk[1], hr[1]))
            z, j = self.zone_slot_for(blocked, mid)
            blocked.add((z, j))
            if hk[0] <= hr[0]:
                place[k], place[r] = (2 * j, z), (2 * j + 1, z)
            else:
                place[r], place[k] = (2 * j, z), (2 * j + 1, z)
        for k, r, th, pk, pr, _, _ in roles:
            if pk is not None:
                self.prep(k, pk)
            else:
                m.flush(k)
            self.prep(r, pr)
        self.lift(place)
        m.pulse([(min(k, r, key=lambda q: m.pos[q][0]), max(k, r, key=lambda q: m.pos[q][0])) for k, r, *_ in roles])
        # rotors down
        down = {r: self.home[r] for k, r, *_ in roles}
        self.lift(down)
        for k, r, th, *_ in roles:
            m.apply1(r, _rx(th))
            m.flush(r)
        self.lift({r: place[r] for k, r, *_ in roles})
        m.pulse([(min(k, r, key=lambda q: m.pos[q][0]), max(k, r, key=lambda q: m.pos[q][0])) for k, r, *_ in roles])
        self.lift({q: self.home[q] for k, r, *_ in roles for q in (k, r)})
        for k, r, th, pk, pr, qk, qr in roles:
            if qk is not None:
                m.apply1(k, qk)
            m.apply1(r, qr)

    def lift(self, mapping):
        """Realise a set of independent atom relocations with few legal rectangle moves."""
        todo = {q: d for q, d in mapping.items() if self.m.pos[q] != d}
        # group by (source row, target row, zone pair) greedily: try whole rows first
        while todo:
            groups = {}
            for q, d in todo.items():
                s = self.m.pos[q]
                groups.setdefault((s[1], d[1]), []).append(q)
            progressed = False
            for key in sorted(groups, key=lambda k: -len(groups[k])):
                qs = sorted(groups[key], key=lambda q: self.m.pos[q][0])
                # split into maximal sub-runs that are legal as one move
                i = 0
                while i < len(qs):
                    j = len(qs)
                    while j > i:
                        sub = {q: todo[q] for q in qs[i:j]}
                        if all(q in todo for q in sub) and self.m.can_move(sub):
                            self.m.move(sub)
                            for q in sub:
                                del todo[q]
                            progressed = True
                            break
                        j -= 1
                    i = j if j > i else i + 1
            if not progressed:
                raise Illegal(f"cannot realise relocations {todo}")

    def run_hub(self, blks):
        m, hub = self.m, self.hub
        role = "Z" if blks[0].c == hub else "Y"
        partners = []
        for b in blks:
            p = b.t if role == "Z" else b.c
            partners.append((p, b.th))
        # hub frame (swapped form when the hub is the target)
        if role == "Y":
            self.prep(hub, _PRE_TS)
        else:
            m.flush(hub)
        pre_p = _PRE_T if role == "Z" else _H
        post_p = _POST_T if role == "Z" else _H
        order = sorted(partners, key=lambda pt: (-self.home[pt[0]][1], self.home[pt[0]][0]))
        for p, _ in order:
            self.prep(p, pre_p)
        seq = order + order[::-1]
        first = True
        for idx, (p, th) in enumerate(seq):
            if idx == len(order):
                for q, t in order:
                    m.apply1(q, _rx(t))
                    m.flush(q)
            hx, hy = self.home[p]
            j = hx // 2
            hsite, psite = (2 * j, 10), (2 * j + 1, 10)
            if m.pos[hub] != hsite:
                self.go(hub, hsite)
            self.go(p, psite)
            m.pulse([(hub, p)])
            self.go(p, self.home[p])
        self.go(hub, self.home[hub])
        for p, _ in order:
            m.apply1(p, post_p)
        if role == "Y":
            m.apply1(hub, _POST_TS)


def default_layout(n, hub):
    R = max(1, math.ceil(n / 10))
    home = {}
    qs = [q for q in range(n) if q != hub] + [hub]
    for i, q in enumerate(qs):
        r = 9 - i // 10
        c = i % 10 if (i // 10) % 2 == 0 else 9 - i % 10
        home[q] = (c, r)
    return home


def compile_ihva(n, blocks, policy="look"):
    blocks = [b for b in blocks if b.th != 0.0]
    for i, b in enumerate(blocks):
        b.idx = i
    hub = n - 1
    preds, succs = build_dag(blocks)
    steps = schedule(blocks, preds, succs, hub, policy)
    home = default_layout(n, hub)
    ex = Executor(n, hub, home)
    for kind, blks in steps:
        if kind == "round":
            ex.run_round(blks)
        else:
            ex.run_hub(blks)
    ex.m.finish()
    return ex.m.qc, home


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any):
    parsed = None
    try:
        parsed = parse_ihva(circuit)
    except Exception:  # noqa: BLE001
        parsed = None
    if parsed is not None:
        n, blocks = parsed
        if n <= 100:
            try:
                qc, home = compile_ihva(n, blocks)
                qc.global_phase = 0
                return qc, Architecture(home, spec=architecture_spec)
            except Illegal:
                pass
    return _naive(circuit, architecture_spec)
