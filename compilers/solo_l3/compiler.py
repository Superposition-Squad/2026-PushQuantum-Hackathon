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
from functools import lru_cache
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
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SD = _S.conj().T
# Frame tables. "R": blocks as written (rotor t: rx(pi/2) h | cz rx cz | h rx(-pi/2)).
# "S": every qubit conjugated by S (initial S^dag h, final S): a block Z_c Y_t becomes Z_c X_t =
# cz . rx_t(th) . cz exactly, with no per-block basis change; swapped form: h on both, rx on c.
FRAMES = {
    "R": {"init": _H, "final": _I2, "Nt": (_PRE_T, _POST_T), "Sc": (_H, _H), "St": (_PRE_TS, _POST_TS)},
    "S": {"init": _SD @ _H, "final": _S, "Nt": (_I2, _I2),
          "Sc": (_rz(math.pi / 2) @ _rx(math.pi / 2), _rx(-math.pi / 2) @ _rz(-math.pi / 2)),
          "St": (_rx(math.pi / 2) @ _rz(math.pi / 2), _rz(-math.pi / 2) @ _rx(-math.pi / 2))},
    "SH": {"init": _SD @ _H, "final": _S, "Nt": (_I2, _I2), "Sc": (_H, _H), "St": (_H, _H)},
}


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
    """kind: "G" generic (cz rx cz), "C" th = s*pi/2 mod 2pi (one cz + local Cliffords),
    "P" th = pi mod 2pi (local Paulis, no cz), "Z" a bare cz (generic cz-program path)."""

    __slots__ = ("c", "t", "th", "idx", "kind", "s")

    def __init__(self, c, t, th, idx, kind="G", s=0):
        self.c, self.t, self.th, self.idx = c, t, th, idx
        self.kind, self.s = kind, s

    def qubits(self):
        return (self.c, self.t)


def classify(th):
    k = th / (math.pi / 2)
    if abs(k - round(k)) < 1e-12:
        r = int(round(k)) % 4
        if r == 1:
            return "C", 1
        if r == 3:
            return "C", -1
        if r == 2:
            return "P", 0
    return "G", 0


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


@lru_cache(maxsize=40000)
def _dist(a, b):
    pa, pb = _phys(a), _phys(b)
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _zone(y):
    return 1 if y >= 10 else 0


class Machine:
    LAZY = False

    def __init__(self, n, loading, emit=True, frame="S"):
        self.n = n
        self.pos = dict(loading)
        self.occ = {s: q for q, s in loading.items()}
        self.emit = emit
        self.qc = QuantumCircuit(n, n) if emit else None
        self.dist = 0.0
        self.nmoves = 0
        self.frame = FRAMES[frame]
        self.pend = [self.frame["init"].copy() for _ in range(n)]
        self.depth = [0] * n  # rough per-qubit depth estimate (for cost models)

    # -- moves
    def _plan(self, mapping):
        """Validator-exact legality of one rectangle move; returns (rows, cols, new_rows, new_cols, d)."""
        pos, occ = self.pos, self.occ
        if len(mapping) == 1:
            q, (tx, ty) = next(iter(mapping.items()))
            x, y = pos[q]
            if not 0 <= tx <= 9 or not 0 <= ty <= 19:
                raise Illegal("target outside")
            if occ.get((tx, ty), q) != q:
                raise Illegal("deposit on occupied")
            return [y], [x], [ty], [tx], _dist((x, y), (tx, ty))
        src = {q: pos[q] for q in mapping}
        rmap, cmap = {}, {}
        for q, (x, y) in src.items():
            tx, ty = mapping[q]
            if rmap.setdefault(y, ty) != ty or cmap.setdefault(x, tx) != tx:
                raise Illegal("inconsistent line map")
        ys = sorted(rmap)
        xs = sorted(cmap)
        new_rows = [rmap[y] for y in ys]
        new_cols = [cmap[x] for x in xs]
        if any(a >= b for a, b in itertools.pairwise(new_rows)):
            raise Illegal("row order")
        if any(a >= b for a, b in itertools.pairwise(new_cols)):
            raise Illegal("col order")
        y0, y1, x0, x1 = ys[0], ys[-1], xs[0], xs[-1]
        for (x, y), q in occ.items():
            if y0 <= y <= y1 and x0 <= x <= x1 and q not in mapping:
                raise Illegal(f"ghost {q}")
        z0 = None
        for q, d in mapping.items():
            if not (0 <= d[0] <= 9 and 0 <= d[1] <= 19):
                raise Illegal("target outside")
            o = occ.get(d)
            if o is not None and o not in mapping:
                raise Illegal("deposit on occupied")
        if len({s[1] >= 10 for s in src.values()}) > 1 or len({d[1] >= 10 for d in mapping.values()}) > 1:
            raise Illegal("zone")
        d = max(_dist(src[q], mapping[q]) for q in mapping)
        return list(range(y0, y1 + 1)), list(range(x0, x1 + 1)), new_rows, new_cols, d

    def move(self, mapping):
        """mapping: qubit -> target site. Emits one legal rectangle move (validator rules)."""
        if not mapping:
            return
        rows, cols, new_rows, new_cols, d = self._plan(mapping)
        if d == 0:
            return
        qs = sorted(mapping)
        if self.emit:
            _move(self.qc, qs, rows, cols, new_rows, new_cols)
        src = {q: self.pos[q] for q in qs}
        for q in qs:
            del self.occ[src[q]]
        for q in qs:
            self.pos[q] = mapping[q]
            self.occ[mapping[q]] = q
        self.dist += d
        self.nmoves += 1

    def can_move(self, mapping):
        try:
            self._plan(mapping)
            return True
        except Illegal:
            return False

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
        rest = _I2.copy()
        if self.LAZY and not force and len(seq) >= 2 and seq[-1][0] == "rz":
            # keep the trailing z rotation pending: it commutes with cz
            seq = seq[:-1]
            rest = u @ np.linalg.inv(_gate_matrix(seq))
            if not _is_diag(rest, 1e-7):
                seq = _zxz(u)
                rest = _I2.copy()
        if self.emit:
            for g, t in seq:
                if g == "rz":
                    self.qc.rz(t, q)
                else:
                    self.qc.rx(t, q)
        self.depth[q] += len(seq)
        self.pend[q] = rest

    def pulse(self, pairs, dummy=()):
        inzone = {q for q, s in self.pos.items() if s[1] >= 10}
        used = set()
        dummy = set(dummy)
        for a, b in pairs:
            (xa, ya), (xb, yb) = self.pos[a], self.pos[b]
            if ya < 10 or ya != yb or min(xa, xb) % 2 or abs(xa - xb) != 1:
                raise Illegal("not a pair")
            for q in (a, b):
                if q not in dummy and not _is_diag(self.pend[q]):
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
            self.apply1(q, self.frame["final"])
            self.flush(q, force=True)
        if self.emit:
            for q in range(self.n):
                self.qc.measure(q, q)


# ---------------------------------------------------------------- scheduling


def schedule(blocks, preds, succs, hub, policy=("look", 2.0)):
    return list(schedule_gen(blocks, preds, succs, hub, policy))


def schedule_gen(blocks, preds, succs, hub, policy=("look", 2.0), posfn=None, mu=None, evaluator=None,
                 depthfn=None):
    """Yields steps ("hub", [block...]) or ("round", [block...]); the caller may execute each step
    before the next one is chosen (`posfn()` gives the current positions for `mu`)."""
    m = len(blocks)
    npred = [len(p) for p in preds]
    ready = {i for i in range(m) if npred[i] == 0}
    steps = []
    done = 0
    hubround = len(policy) > 2 and policy[2]
    while ready:
        hubZ = sorted(i for i in ready if blocks[i].c == hub)
        hubY = sorted(i for i in ready if blocks[i].t == hub)
        non = sorted(i for i in ready if hub not in blocks[i].qubits())
        # greedy maximal matching of non-hub ready blocks (by unlock count, then index)
        match = []
        pool = non + (hubZ + hubY if hubround else [])
        if pool:
            used = set()
            if mu is None or posfn is None:
                key = lambda i: (-len(succs[i]), i)
            else:
                pos = posfn()
                key = lambda i: (_dist(pos[blocks[i].c], pos[blocks[i].t]) - mu * len(succs[i]), i)
            order = sorted(pool, key=key)
            if depthfn is not None and hubround:
                dep = depthfn()
                hubs = [i for i in order if hub in blocks[i].qubits()]
                if hubs:
                    other = lambda i: blocks[i].t if blocks[i].c == hub else blocks[i].c
                    hb = min(hubs, key=lambda i: (max(0, dep[other(i)] - dep[hub]), order.index(i)))
                    order = [hb] + [i for i in order if i != hb]
            for i in order:
                b = blocks[i]
                if b.c in used or b.t in used:
                    continue
                used |= {b.c, b.t}
                match.append(i)
            if evaluator is not None and posfn is not None and len(match) > 1:
                pos = posfn()
                variants = [match]
                for mu2 in (0.5, 2.0):
                    key2 = lambda i: (_dist(pos[blocks[i].c], pos[blocks[i].t]) - mu2 * len(succs[i]), i)
                    used2, m2 = set(), []
                    for i in sorted(pool, key=key2):
                        b = blocks[i]
                        if b.c in used2 or b.t in used2:
                            continue
                        used2 |= {b.c, b.t}
                        m2.append(i)
                    if sorted(m2) != sorted(match):
                        variants.append(m2)
                far = [i for i in match if _dist(pos[blocks[i].c], pos[blocks[i].t]) > evaluator.far]
                if far and len(far) < len(match):
                    variants.append([i for i in match if i not in far])
                if len(variants) > 1:
                    k = evaluator([[blocks[i] for i in v] for v in variants])
                    match = variants[k]
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

        kindp = policy[0]
        if kindp == "look":
            w = policy[1]
            kind, sel = max(cands, key=lambda c: (len(c[1]) + w * (gain(c[1]) - len(c[1])) / 2, c[0] == "round"))
            if hubround and kind == "hub" and len(sel) < policy[3]:
                kind, sel = "round", match
        elif kindp == "thresh":
            hubs = [c for c in cands if c[0] == "hub"]
            big = [c for c in hubs if len(c[1]) >= policy[1]]
            if big:
                kind, sel = max(big, key=lambda c: len(c[1]))
            elif match:
                kind, sel = "round", match
            elif hubround:
                kind, sel = "round", match
            else:
                kind, sel = max(hubs, key=lambda c: len(c[1]))
        else:
            kind, sel = cands[0]
        if kind == "round" and len(sel) == 0:
            continue
        yield (kind, [blocks[i] for i in sel])
        for i in sel:
            ready.discard(i)
            done += 1
            for j in succs[i]:
                npred[j] -= 1
                if npred[j] == 0:
                    ready.add(j)
    assert done == m


# ---------------------------------------------------------------- executor


class Executor:
    def __init__(self, n, hub, home, emit=True, frame="S"):
        self.n, self.hub, self.home = n, hub, dict(home)
        self.m = Machine(n, home, emit, frame)
        self.nxt = None

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

    # ---------------------------------------------------------- band rounds
    def relocate(self, targets):
        """Storage-only relocations q -> dst (dst free or vacated by another relocating atom)."""
        m = self.m
        pending = {q: d for q, d in targets.items() if m.pos[q] != d}
        reserved = set(targets.values())
        while pending:
            progress = False
            for q, d in pending.items():
                o = m.occ.get(d)
                if o is not None and o not in pending:
                    raise Illegal("relocation target held by a fixed atom")
            # atoms whose target is free now; greedy agglomeration into legal rectangle moves
            free_now = [q for q in pending if pending[q] not in m.occ]
            free_now.sort(key=lambda q: (-_dist(m.pos[q], pending[q]), q))
            while free_now:
                seed = free_now[0]
                dseed = _dist(m.pos[seed], pending[seed])
                sel = {seed: pending[seed]}
                for q in free_now[1:]:
                    if _dist(m.pos[q], pending[q]) > dseed + 1e-9:
                        continue
                    sel[q] = pending[q]
                    if not m.can_move(sel):
                        del sel[q]
                m.move(sel)
                for q in sel:
                    del pending[q]
                free_now = [q for q in free_now if q not in sel]
                progress = True
            if not progress:
                q = min(pending)
                tmp = self.nearest_free(m.pos[q], avoid=reserved)
                m.move({q: tmp})
        return

    def nearest_free(self, s, avoid=(), rows=range(10), kappa=0.0, y0=6, floor=0, kfloor=0.0):
        best = None
        for y in rows:
            for x in range(10):
                t = (x, y)
                if t in self.m.occ or t in avoid:
                    continue
                d = _dist(s, t) + kappa * max(0, y - y0) + kfloor * max(0, floor - y)
                if best is None or d < best[0]:
                    best = (d, t)
        if best is None:
            raise Illegal("no free site")
        return best[1]

    LOOK = 0.35
    EVICT = 1.5
    SHIFTS = (0, 1, 2)
    PARS = (0, 1)
    SEED = None
    NOISE = 0.6
    KAPPA = 0.0
    HUBDEP = True
    RECT_SIM = 6
    NPULSE = 2
    KFLOOR = 0.0

    def floor_row(self):
        return 10 - math.ceil(self.n / 10) - 1
    ROT_PARTIAL = True
    Y0 = 6
    allowed = frozenset()
    EVAL = False
    FAR = 4.0
    DUMMY_OFF = False

    def _lookpos(self, q):
        nx = self.nxt.get(q) if self.nxt else None
        return self.m.pos[nx] if nx is not None else None

    def run_round_band(self, blks):
        m = self.m
        for b in blks:
            if b.kind == "P":  # exp(-i pi/2 Z_c X_t) = Z_c X_t (S-frame)
                m.apply1(b.c, np.diag([1, -1]).astype(complex))
                m.apply1(b.t, np.array([[0, 1], [1, 0]], dtype=complex))
        cs = [b for b in blks if b.kind == "C"]
        gs = [b for b in blks if b.kind in ("G", "Z")]
        if cs:
            saved = self.NPULSE, self.DUMMY_OFF
            self.NPULSE, self.DUMMY_OFF = 1, True
            try:
                self._round(cs)
            finally:
                self.NPULSE, self.DUMMY_OFF = saved
        if gs:
            self._round(gs)

    @staticmethod
    @lru_cache(maxsize=100)
    def _band_layouts(count, pars, shifts):
        layouts = []
        for par in pars:
            for j1 in range(par, 5):
                for j2 in range(j1, 5):
                    hmin = math.ceil(count / (j2 - j1 + 1))
                    for h in (hmin, hmin + 1):
                        for sh in shifts:
                            if 1 <= h and h + sh <= 9:
                                a, bb = 10 - h - sh, 9 - sh
                                slots = [(2 * j - par, y) for y in range(a, bb + 1) for j in range(j1, j2 + 1)]
                                idx = [y * 9 + xl for xl, y in slots]
                                layouts.append((a, bb, j1, j2, par, slots, idx))
        return layouts

    def _round(self, blks):
        from scipy.optimize import linear_sum_assignment

        m = self.m
        lam = self.LOOK
        members = {q for b in blks for q in b.qubits()}
        look = {q: self._lookpos(q) for q in members}
        rects = self._band_layouts(len(blks), self.PARS, self.SHIFTS)
        plans = []
        occ = m.occ
        EV = self.EVICT
        nonmem = lambda site: site in occ and occ[site] not in members
        # slot universe: left storage column xl in 0..8, row y: index y*9 + xl
        Lx = np.array([xl for y in range(10) for xl in range(9)], dtype=float)
        Ly = np.array([y for y in range(10) for xl in range(9)], dtype=float)
        Rx = Lx + 1
        PA = np.array([m.pos[b.c] for b in blks], dtype=float)
        PB = np.array([m.pos[b.t] for b in blks], dtype=float)

        def dd(P2, X, Y):
            return np.hypot(P2[:, :1] - X[None, :], P2[:, 1:] - Y[None, :])

        C1 = dd(PA, Lx, Ly) + dd(PB, Rx, Ly)
        C2 = dd(PA, Rx, Ly) + dd(PB, Lx, Ly)
        if lam:
            LA = np.array([look[b.c] if look[b.c] is not None else (np.nan, np.nan) for b in blks], dtype=float)
            LB = np.array([look[b.t] if look[b.t] is not None else (np.nan, np.nan) for b in blks], dtype=float)
            C1 = C1 + lam * (np.nan_to_num(dd(LA, Lx, Ly)) + np.nan_to_num(dd(LB, Rx, Ly)))
            C2 = C2 + lam * (np.nan_to_num(dd(LA, Rx, Ly)) + np.nan_to_num(dd(LB, Lx, Ly)))
        PEN = np.array([EV * (nonmem((xl, y)) + nonmem((xl + 1, y))) for y in range(10) for xl in range(9)])
        CM = np.minimum(C1, C2) + PEN[None, :]
        if self.SEED is not None:
            CM = CM + self.NOISE * self.rng.random(CM.shape)
        OR = C2 < C1
        for a, bb, j1, j2, par, slots, idx in rects:
            cost = CM[:, idx]
            orient = OR[:, idx]
            ri, ci = linear_sum_assignment(cost)
            tot = float(cost[ri, ci].sum())
            used = set(ci.tolist())
            for k, (xl, y) in enumerate(slots):
                if k in used:
                    continue
                nl, nr = nonmem((xl, y)), nonmem((xl + 1, y))
                if nl and nr:
                    if self.DUMMY_OFF or frozenset((occ[(xl, y)], occ[(xl + 1, y)])) not in self.allowed:
                        tot += 2 * EV
                elif nl or nr:
                    tot += EV
            h = 10 - a
            tot += 2 * self.NPULSE * math.hypot(h, 1.0 if par else 0.7)
            plans.append((tot, len(plans), (a, bb, j1, j2, par, slots, ri, ci, orient)))
        plans.sort(key=lambda t: (t[0], t[1]))
        if self.RECT_SIM > 1 and len(plans) > 1:
            best = None
            for _, _, plan in plans[: self.RECT_SIM]:
                e = self.clone()
                e.RECT_SIM = 0
                try:
                    e._exec_round(blks, members, plan)
                except Illegal:
                    continue
                c = e.m.dist - m.dist + DEPTH_W * (max(e.m.depth) - max(m.depth))
                if best is None or c < best[0] - 1e-9:
                    best = (c, plan)
            plan = best[1] if best is not None else plans[0][2]
        else:
            plan = plans[0][2]
        self._exec_round(blks, members, plan)

    def _exec_round(self, blks, members, plan):
        m = self.m
        occ = m.occ
        a, bb, j1, j2, par, slots, ri, ci, orient = plan
        h = 10 - a
        targets = {}
        for i, k in zip(ri, ci):
            b = blks[i]
            xl, y = slots[k]
            L, R = (xl, y), (xl + 1, y)
            if orient[i, k]:
                targets[b.c], targets[b.t] = R, L
            else:
                targets[b.c], targets[b.t] = L, R
        reserved = set(targets.values())
        x1, x2 = 2 * j1 - par, 2 * j2 + 1 - par
        inside = lambda s: a <= s[1] <= bb and x1 <= s[0] <= x2
        dummies = []
        used_slots = {slots[k] for k in ci.tolist()}
        for xl, y in slots:
            if (xl, y) in used_slots or self.DUMMY_OFF:
                continue
            ql, qr = occ.get((xl, y)), occ.get((xl + 1, y))
            if ql is not None and qr is not None and ql not in members and qr not in members \
                    and frozenset((ql, qr)) in self.allowed:
                dummies.append((ql, qr))
        keep = {q for pr in dummies for q in pr}
        for q in sorted(q for q, s in m.pos.items() if inside(s) and q not in members and q not in keep):
            t = self.nearest_free(m.pos[q], avoid=reserved | {(x, y) for x in range(x1, x2 + 1) for y in range(a, bb + 1)},
                                  kappa=self.KAPPA, y0=self.Y0, floor=self.floor_row(), kfloor=self.KFLOOR)
            reserved.add(t)
            targets[q] = t
        self.relocate(targets)
        if self.NPULSE == 1:
            riders = members | keep
            for b in blks:
                if b.kind == "C":
                    ss = _S if b.s > 0 else _SD
                    m.apply1(b.c, ss)
                    m.apply1(b.t, ss @ _H)
            for q in riders:
                m.flush(q)
            up = {q: (m.pos[q][0] + par, m.pos[q][1] + h) for q in riders}
            down = {q: m.pos[q] for q in riders}
            pairs = [(b.c, b.t) if m.pos[b.c][0] < m.pos[b.t][0] else (b.t, b.c) for b in blks]
            m.move(up)
            m.pulse(pairs, ())
            m.move(down)
            for b in blks:
                if b.kind == "C":
                    m.apply1(b.t, _H)
            return
        forms = {}
        for b in blks:
            forms[b.idx] = f = self.choose_form(b)
            F = m.frame
            if f == "N":
                m.flush(b.c)
                self.prep(b.t, F["Nt"][0])
            else:
                self.prep(b.c, F["Sc"][0])
                self.prep(b.t, F["St"][0])
        riders = members | keep
        up = {q: (m.pos[q][0] + par, m.pos[q][1] + h) for q in riders}
        down = {q: m.pos[q] for q in riders}
        pairs = [(b.c, b.t) if m.pos[b.c][0] < m.pos[b.t][0] else (b.t, b.c) for b in blks]
        pairs += dummies
        m.move(up)
        m.pulse(pairs, keep)
        m.move(down)
        for b in blks:
            r = b.t if forms[b.idx] == "N" else b.c
            m.apply1(r, _rx(b.th))
            m.flush(r)
        m.move(up)
        m.pulse(pairs, keep)
        m.move(down)
        for b in blks:
            if forms[b.idx] == "N":
                m.apply1(b.t, m.frame["Nt"][1])
            else:
                m.apply1(b.c, m.frame["Sc"][1])
                m.apply1(b.t, m.frame["St"][1])

    FORMS = "auto"
    KEEP_HUB_STATIC = True
    LAZY_FRAME = False

    def _ng(self, q, u):
        v = u @ self.m.pend[q]
        if _is_diag(v):
            return 0, v
        return len(_zxz(v)), _I2

    def choose_form(self, b):
        hub = self.hub
        if self.KEEP_HUB_STATIC:
            if b.c == hub:
                return "N"
            if b.t == hub:
                return "S"
        if self.FORMS != "auto":
            return self.FORMS
        dep = self.m.depth
        best = None
        for f in ("N", "S"):
            if f == "N":
                gc, pc = self._ng(b.c, _I2)
                gt, pt = self._ng(b.t, self.m.frame["Nt"][0])
                rot, pr = b.t, pt
            else:
                gc, pc = self._ng(b.c, self.m.frame["Sc"][0])
                gt, pt = self._ng(b.t, self.m.frame["St"][0])
                rot, pr = b.c, pc
            mid = 0 if _is_id(pr) else (1 if _is_diag(pr) else 0)
            dc, dt = dep[b.c] + gc, dep[b.t] + gt
            key = (max(dc, dt) + 1 + mid, gc + gt + mid, f)
            if best is None or key < best[0]:
                best = (key, f)
        return best[1]

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

    def clone(self):
        e = Executor.__new__(Executor)
        e.__dict__.update({k: v for k, v in self.__dict__.items() if k != "m"})
        m = Machine.__new__(Machine)
        src = self.m
        m.n, m.emit, m.qc = src.n, False, None
        m.frame = src.frame
        m.LAZY = src.LAZY
        m.pos, m.occ = dict(src.pos), dict(src.occ)
        m.dist, m.nmoves = src.dist, src.nmoves
        m.pend = src.pend.copy()  # operations replace matrix entries rather than mutating them
        m.depth = list(src.depth)
        e.m = m
        return e

    def run_hub(self, blks):
        m = self.m
        for b in blks:
            if b.kind == "P":
                m.apply1(b.c, np.diag([1, -1]).astype(complex))
                m.apply1(b.t, np.array([[0, 1], [1, 0]], dtype=complex))
        for part in ([b for b in blks if b.kind == "C"], [b for b in blks if b.kind == "G"]):
            if not part:
                continue
            best = None
            for strat in self.HUB_STRATS:
                e = self.clone()
                try:
                    e._hub(part, strat)
                except Illegal:
                    continue
                c = e.m.dist - self.m.dist
                if best is None or c < best[0] - 1e-9:
                    best = (c, strat)
            if best is None:
                raise Illegal("no hub strategy")
            self._hub(part, best[1])

    HUB_STRATS = ("rotate", "direct", "compact", "gravity")

    def _gravity(self):
        """Refill holes in the bottom rows with the atoms above them (nearest first)."""
        from scipy.optimize import linear_sum_assignment

        m = self.m
        R = max(1, math.ceil((self.n - 1) / 10))
        lo = 10 - R
        holes = [(x, y) for y in range(lo, 10) for x in range(10) if (x, y) not in m.occ]
        above = sorted(q for q, st in m.pos.items() if st[1] < lo)
        if not holes or not above:
            return
        cost = np.array([[_dist(m.pos[q], hs) for hs in holes] for q in above])
        ri, ci = linear_sum_assignment(cost)
        self.relocate({above[i]: holes[k] for i, k in zip(ri, ci)})

    def _compact(self, pset):
        """Relocate the batch partners densely into the bottom storage rows (row 9-r stays free)."""
        from scipy.optimize import linear_sum_assignment

        m = self.m
        r = max(1, math.ceil(len(pset) / 10))
        sites = [(x, y) for y in range(10 - r, 10) for x in range(10)]
        ps = sorted(pset)
        cost = np.array([[_dist(m.pos[p], st) for st in sites] for p in ps])
        ri, ci = linear_sum_assignment(cost)
        targets = {ps[i]: sites[k] for i, k in zip(ri, ci)}
        reserved = set(targets.values())
        band = set(sites) | {(x, 9 - r) for x in range(10)}
        for q in sorted(q for q, st in m.pos.items() if st[1] >= 9 - r and st[1] <= 9 and q not in pset):
            if q == self.hub and False:
                continue
            t = self.nearest_free(m.pos[q], avoid=reserved | band, rows=range(0, 9 - r))
            reserved.add(t)
            targets[q] = t
        self.relocate(targets)

    def _hub_site(self, px):
        """(hub site, partner site) in zone row 10 for a partner lifted from storage column px."""
        if px % 2 == 0:
            return (px + 1, 10), (px, 10)
        return (px - 1, 10), (px, 10)

    def _serve(self, p):
        m, hub = self.m, self.hub
        px, py = m.pos[p]
        hs, ps = self._hub_site(px)
        if m.pos[hub] != hs:
            if hs in m.occ:
                raise Illegal("hub site busy")
            self.go(hub, hs)
        back = m.pos[p]
        self.go(p, ps)
        m.pulse([(hub, p)] if hs[0] < ps[0] else [(p, hub)])
        self.go(p, back)

    def _rotate(self, rt, forward):
        m = self.m
        K = 9 - rt
        if forward:
            row9 = {q: (s[0], rt) for q, s in m.pos.items() if s[1] == 9}
            m.move(row9)
            m.move({q: (s[0], s[1] + 1) for q, s in m.pos.items() if rt <= s[1] <= 8})
        else:
            m.move({q: (s[0], s[1] - 1) for q, s in m.pos.items() if rt + 1 <= s[1] <= 9})
            m.move({q: (s[0], 9) for q, s in m.pos.items() if s[1] == rt})

    def _hub(self, blks, strat):
        m, hub = self.m, self.hub
        role = "Z" if blks[0].c == hub else "Y"
        partners = []
        for b in blks:
            p = b.t if role == "Z" else b.c
            partners.append((p, b.th))
        single = all(b.kind == "C" for b in blks)
        if single:
            # one cz per block: exp(-i s pi/4 Z_c X_t) = (I x h) cz (S^s x S^s) (I x h)
            hs = _I2.copy()
            for b in blks:
                hs = (_S if b.s > 0 else _SD) @ hs
            if role == "Z":
                self.prep(hub, hs)
                for b in blks:
                    self.prep(b.t, (_S if b.s > 0 else _SD) @ _H)
                post_p, post_h = _H, None
            else:
                self.prep(hub, hs @ _H)
                for b in blks:
                    self.prep(b.c, _S if b.s > 0 else _SD)
                post_p, post_h = None, _H
        else:
            if role == "Y":
                self.prep(hub, m.frame["St"][0])
            else:
                m.flush(hub)
            pre_p = m.frame["Nt"][0] if role == "Z" else m.frame["Sc"][0]
            post_p = m.frame["Nt"][1] if role == "Z" else m.frame["Sc"][1]
            post_h = m.frame["St"][1] if role == "Y" else None
            for p, _ in partners:
                self.prep(p, pre_p)
        if strat == "compact":
            self._compact({p for p, _ in partners})
            strat = "rotate"
        elif strat == "gravity":
            self._gravity()
            strat = "rotate"
        hub_home = m.pos[hub]
        if strat == "direct":
            base = {q: m.pos[q] for q in range(self.n)}
            order = sorted(partners, key=lambda pt: (-base[pt[0]][1], base[pt[0]][0]))
            seq = [p for p, _ in order]
            for p in seq:
                self._serve(p)
            if not single:
                for p, th in partners:
                    m.apply1(p, _rx(th))
                    m.flush(p)
                for p in reversed(seq):
                    self._serve(p)
        else:
            pset = {p for p, _ in partners}
            rlo = min(m.pos[p][1] for p in pset)
            others = {q for q in range(self.n) if q != hub}
            rt = None
            if self.ROT_PARTIAL:
                cols9 = {m.pos[q][0] for q in others if m.pos[q][1] == 9}
                for r in range(rlo - 1, -1, -1):
                    if not any(m.pos[q][1] == r and m.pos[q][0] in cols9 for q in others):
                        rt = r
                        break
            else:
                for r in range(rlo - 1, -1, -1):
                    if not any(m.pos[q][1] == r for q in others):
                        rt = r
                        break
            if rt is None:
                raise Illegal("no free row for rotation")
            cap = 2 * (10 - rt) + 2

            visits = []

            def sweep(todo, forward):
                step = 0
                while True:
                    row = [p for p in todo if m.pos[p][1] == 9]
                    row.sort(key=lambda p: (m.pos[p][0] // 2, -(m.pos[p][0] % 2)))
                    if step % 2:
                        row.reverse()
                    if not forward:
                        k = len(visits) - 1 - step
                        if 0 <= k < len(visits) and set(visits[k]) == set(row):
                            row = visits[k][::-1]
                    else:
                        visits.append(row)
                    for p in row:
                        self._serve(p)
                        todo.discard(p)
                    if not todo:
                        return
                    step += 1
                    if step > cap:
                        raise Illegal("rotation did not reach every partner")
                    self._rotate(rt, forward)

            sweep(set(pset), True)
            if not single:
                for p, th in partners:
                    m.apply1(p, _rx(th))
                    m.flush(p)
                sweep(set(pset), False)
        if hub_home in m.occ:
            hub_home = self.nearest_free(m.pos[hub])
        self.go(hub, hub_home)
        if post_p is not None:
            for p, _ in partners:
                m.apply1(p, post_p)
        if post_h is not None:
            m.apply1(hub, post_h)


def default_layout(n, hub):
    R = max(1, math.ceil(n / 10))
    home = {}
    qs = [q for q in range(n) if q != hub] + [hub]
    for i, q in enumerate(qs):
        r = 9 - i // 10
        c = i % 10 if (i // 10) % 2 == 0 else 9 - i % 10
        home[q] = (c, r)
    return home


def layout_from_steps(n, hub, steps):
    """Atoms in order of first use, filled from storage row 9 upwards; first-round pairs adjacent."""
    order = []
    seen = set()
    for kind, blks in steps:
        for b in blks:
            for q in (b.c, b.t) if kind == "round" else ((b.t if b.c == hub else b.c),):
                if q not in seen:
                    seen.add(q)
                    order.append(q)
    order += [q for q in range(n) if q not in seen and q != hub]
    if hub not in seen:
        order.append(hub)
    # hub goes last (it is lifted alone for batches; in rounds it is relocated anyway)
    order = [q for q in order if q != hub] + [hub]
    home = {}
    for i, q in enumerate(order):
        home[q] = (i % 10, 9 - i // 10)
    return home


def next_partners(steps):
    """nx[k][q] = q's partner in the first step after k that involves q."""
    nx = [None] * len(steps)
    cur = {}
    for k in range(len(steps) - 1, -1, -1):
        nx[k] = dict(cur)
        for b in steps[k][1]:
            cur[b.c] = b.t
            cur[b.t] = b.c
    return nx


POLICIES = [("look", 2.0), ("look", 0.0), ("look", 1.0), ("look", 4.0), ("thresh", 8), ("thresh", 16), ("thresh", 30), ("thresh", 45), ("rounds",), ("look", 2.0, True, 8), ("look", 2.0, True, 20), ("look", 4.0, True, 12), ("thresh", 20, True), ("thresh", 30, True), ("thresh", 45, True), ("rounds", 0, True)]
DEPTH_W = 0.2 / 13364 / (0.8 / 163364.298)  # distance units per depth unit (~3.06)


def run_candidate(n, blocks, preds, succs, home, pol, lam, mu, opts=(), emit=False, allowed=None):
    hub = n - 1
    frame = dict(opts).get("FRAME", "S")
    if frame != "S" and any(b.kind != "G" for b in blocks):
        # the special-angle forms are written for the S-frame; elsewhere use the generic form
        blocks = [Block(b.c, b.t, b.th, b.idx) for b in blocks]
    ex = Executor(n, hub, home, emit=emit, frame=frame)
    ex.allowed = allowed if allowed is not None else frozenset(frozenset(b.qubits()) for b in blocks)
    ex.LOOK = lam
    for k, v in opts:
        setattr(ex, k, v)
    ex.m.LAZY = ex.LAZY_FRAME
    if ex.SEED is not None:
        ex.rng = np.random.default_rng(ex.SEED)
    byq = {}
    for b in blocks:
        for q in b.qubits():
            byq.setdefault(q, []).append(b)
    ptr = {q: 0 for q in byq}
    done = set()
    ex.steps = []
    evaluator = None
    if ex.EVAL:
        def evaluator(variants):
            best = None
            for k, v in enumerate(variants):
                e = ex.clone()
                try:
                    e.run_round_band(v)
                except Illegal:
                    continue
                c = (e.m.dist - ex.m.dist + DEPTH_W * (max(e.m.depth) - max(ex.m.depth))) / len(v)
                if best is None or c < best[0] - 1e-9:
                    best = (c, k)
            return 0 if best is None else best[1]
        evaluator.far = ex.FAR
    for kind, blks in schedule_gen(blocks, preds, succs, hub, pol, posfn=lambda: ex.m.pos, mu=mu, evaluator=evaluator,
                                   depthfn=(lambda: ex.m.depth) if ex.HUBDEP else None):
        cur = {b.idx for b in blks}
        nxt = {}
        for b in blks:
            for q in b.qubits():
                lst = byq[q]
                k = ptr[q]
                while k < len(lst) and lst[k].idx in done:
                    k += 1
                ptr[q] = k
                while k < len(lst) and (lst[k].idx in done or lst[k].idx in cur):
                    k += 1
                if k < len(lst):
                    o = lst[k]
                    nxt[q] = o.t if o.c == q else o.c
        ex.nxt = nxt
        ex.steps.append((kind, blks))
        (ex.run_round_band if kind == "round" else ex.run_hub)(blks)
        done |= cur
    ex.m.finish()
    return ex


BASE_POLS = (("look", 2.0, True, 8), ("look", 2.0, True, 20), ("look", 2.0, True, 30), ("look", 2.0, True, 40),
             ("look", 2.0, True, 48), ("thresh", 20, True), ("rounds",), ("thresh", 45))
_L8, _L20, _L40 = ("look", 2.0, True, 8), ("look", 2.0, True, 20), ("look", 2.0, True, 40)
# chosen by a greedy set cover over the 12 non-warm scored iHVA circuits (+ their individual winners)
CANDS = [(_L8, 0.0, 1.0, (("PARS", (0, 1)),)), (_L40, 0.0, 1.0, (("PARS", (0,)),)), (_L8, 0.35, 1.0, (("PARS", (0,)),)),
         (_L40, 0.35, 1.0, (("PARS", (0, 1)),)), (_L8, 0.0, None, (("PARS", (0, 1)),)), (_L8, 0.0, 1.0, (("PARS", (0,)),)),
         (_L40, 0.0, 1.0, (("PARS", (0, 1)),)), (_L8, 0.35, 1.0, (("PARS", (0, 1)),)),
         (_L20, 0.35, 1.0, (("PARS", (0, 1)),)), (_L20, 0.0, 1.0, (("PARS", (0,)),)), (_L20, 0.35, 1.0, (("PARS", (0,)),)),
         (("rounds",), 0.0, None, (("PARS", (0, 1)),)), (("thresh", 45), 0.0, None, (("PARS", (0, 1)),))]


def _budget(n, nblocks):
    """(top configs re-run with EVAL / RECT_SIM=1 / first-use layout, seeds for the best config)."""
    if nblocks <= 120:
        return 5, 16
    if nblocks <= 400:
        return 4, 10
    return 3, 6


def compile_ihva(n, blocks, cands=None, emit=True, seeds=None):
    allowed = frozenset(frozenset(b.qubits()) for b in blocks)  # every pair of the input (incl. th == 0)
    blocks = [b for b in blocks if b.th != 0.0]
    for b in blocks:
        b.kind, b.s = classify(b.th)
    for i, b in enumerate(blocks):
        b.idx = i
    hub = n - 1
    preds, succs = build_dag(blocks)
    home = default_layout(n, hub)
    res = []

    layouts = {None: home}

    def run(cand, lay=None):
        try:
            ex = run_candidate(n, blocks, preds, succs, layouts[lay], *cand, allowed=allowed)
        except Illegal:
            return None
        c = ex.m.dist + DEPTH_W * (max(ex.m.depth) + 1)
        res.append((c, len(res), cand, lay))
        return ex

    for cand in cands or CANDS:
        run(cand)
    if not res:
        raise Illegal("no candidate")
    top, nseed = _budget(n, len(blocks)) if seeds is None else seeds
    for _, _, cand, lay in sorted(res)[:top]:
        pol, lam, mu, opts = cand
        run((pol, lam, mu, tuple(opts) + (("EVAL", True),)), lay)
        run((pol, lam, mu, tuple(opts) + (("RECT_SIM", 1),)), lay)
        run((pol, lam, mu, tuple(opts) + (("FRAME", "R"),)), lay)
        run((pol, lam, mu, tuple(opts) + (("FRAME", "SH"),)), lay)
    # first-use layouts derived from the best configurations' own step sequences
    for _, k, cand, lay in sorted(res)[:top]:
        ex = run_candidate(n, blocks, preds, succs, layouts[lay], *cand, allowed=allowed)
        key = ("first", k)
        layouts[key] = layout_from_steps(n, hub, ex.steps)
        run(cand, key)
    if nseed:
        _, _, cand, lay = min(res)
        pol, lam, mu, opts = cand
        for sd in range(1, nseed + 1):
            run((pol, lam, mu, tuple(opts) + (("SEED", sd),)), lay)
    if n >= 12 and any(b.kind == "G" for b in blocks):
        # Both endpoints already return to storage in a band round. Let the
        # hub take the middle rotation if that avoids basis changes.
        for _, _, (pol, lam, mu, opts), lay in sorted(res)[:2]:
            for forms, lazy in (("N", False), ("auto", False), ("N", True), ("auto", True)):
                run((pol, lam, mu, (*opts, ("KEEP_HUB_STATIC", False), ("FORMS", forms),
                                       ("LAZY_FRAME", lazy))), lay)
    best = min(res)
    if not emit:
        return best[0], best[2]
    home = layouts[best[3]]
    ex = run_candidate(n, blocks, preds, succs, home, *best[2], emit=True, allowed=allowed)
    qc = ex.m.qc
    # guard: the interaction set must stay inside the input's pairs (validator check above 19 qubits)
    idx = {bit: i for i, bit in enumerate(qc.qubits)}
    for inst in qc.data:
        if inst.operation.name == "cz":
            if frozenset(idx[q] for q in inst.qubits) not in allowed:
                raise Illegal("cz outside the input's interaction pairs")
    return qc, home


# ---------------------------------------------------------------- generic cz-layer path (HEA)


def parse_cz_program(circuit: QuantumCircuit):
    """(n, ops) with ops = ("1q", q, 2x2) / ("cz", a, b), final measurements q -> q; None otherwise."""
    from qiskit.quantum_info import Operator

    n = circuit.num_qubits
    if circuit.num_clbits < n:
        return None
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    ops = []
    measured = set()
    for inst in circuit.data:
        op = inst.operation
        qs = [index[q] for q in inst.qubits]
        if op.name == "barrier":
            continue
        if op.name == "measure":
            if qs[0] != circuit.find_bit(inst.clbits[0]).index or qs[0] in measured:
                return None
            measured.add(qs[0])
            continue
        if measured & set(qs):
            return None
        if len(qs) == 1:
            try:
                u = Operator(op).data
            except Exception:  # noqa: BLE001
                return None
            ops.append(("1q", qs[0], u))
        elif op.name in ("cx", "cz") and len(qs) == 2:
            a, b = qs
            if op.name == "cx":
                ops.append(("1q", b, _H))
                ops.append(("cz", a, b))
                ops.append(("1q", b, _H))
            else:
                ops.append(("cz", a, b))
        else:
            return None
    if measured != set(range(n)):
        return None
    return n, ops


def snake_layout(n):
    """Two-row snake (row 9 left to right, row 8 right to left) for chains; plain rows otherwise."""
    W = (n + 1) // 2
    home = {}
    if W <= 10:
        for q in range(n):
            home[q] = (q, 9) if q < W else (W - 1 - (q - W), 8)
        return home
    for q in range(n):
        home[q] = (q % 10, 9 - q // 10)
    return home


def compile_cz_program(n, ops, home, emit=True, rect_sim=6, lazy=True):
    ex = Executor(n, -1, home, emit=emit, frame="R")
    m = ex.m
    m.LAZY = lazy
    m.pend = [_I2.copy() for _ in range(n)]
    ex.NPULSE = 1
    ex.DUMMY_OFF = True
    ex.RECT_SIM = rect_sim
    ex.allowed = frozenset()
    rnd, used, after = [], set(), {}

    def close():
        nonlocal rnd, used, after
        if rnd:
            ex.nxt = {}
            ex.run_round_band([Block(a, b, 0.0, k) for k, (a, b) in enumerate(rnd)])
            for q, us in after.items():
                for u in us:
                    m.apply1(q, u)
        rnd, used, after = [], set(), {}

    for op in ops:
        if op[0] == "1q":
            _, q, u = op
            if q in used:
                after.setdefault(q, []).append(u)
            else:
                m.apply1(q, u)
        else:
            _, a, b = op
            if a in used or b in used:
                close()
            rnd.append((a, b))
            used |= {a, b}
    close()
    m.finish()
    return ex


def _lower_z_phases(circuit):
    """Combine Z phases across CZs and keep Pauli corrections in a frame."""
    # Rz commutes with CZ. An Rx or measurement gives a storage-safe
    # endpoint for any combined phase. Movement and CZ pulses stay in place.
    indices = {q: i for i, q in enumerate(circuit.qubits)}
    n = circuit.num_qubits
    data = circuit.data

    def lower(propagate_paulis, late):
        phase = [0.0] * n
        pending_x = [False] * n
        pending_z = [False] * n
        previous_x = [None] * n
        extra = {}
        replacements = {}
        for i, ins in enumerate(data):
            name = ins.operation.name
            if name not in ("cz", "rx", "rz", "measure"):
                continue
            qs = [indices[q] for q in ins.qubits]
            q = qs[0]
            if name == "cz":
                # CZ conjugates X on either qubit to X on that qubit
                # and Z on its partner.
                a, b = qs
                pending_z[a] ^= pending_x[b]
                pending_z[b] ^= pending_x[a]
            elif name == "rz":
                theta = _wrap(float(ins.operation.params[0]))
                if propagate_paulis and abs(abs(theta) - math.pi) < 1e-10:
                    pending_z[q] ^= True
                else:
                    phase[q] += -theta if pending_x[q] else theta
                replacements[i] = ()
            else:
                theta = None if name == "measure" else _wrap(float(ins.operation.params[0]))
                if propagate_paulis and theta is not None and abs(abs(theta) - math.pi) < 1e-10:
                    pending_x[q] ^= True
                    replacements[i] = ()
                    continue
                angle = _wrap(phase[q] + (math.pi if name == "measure" and pending_z[q] else 0.0))
                if abs(angle) > 1e-10:
                    previous = previous_x[q]
                    slot = i if late or previous is None else previous + 1
                    extra.setdefault(slot, []).append(("rz", angle, q))
                phase[q] = 0.0
                if name == "measure":
                    if pending_x[q]:
                        extra.setdefault(i, []).append(("rx", math.pi, q))
                else:
                    rotation = -theta if pending_z[q] else theta
                    replacements[i] = (("rx", rotation, q),) if abs(rotation) > 1e-10 else ()
                    if replacements[i]:
                        previous_x[q] = i
        out = circuit.copy_empty_like()
        for i, ins in enumerate(data):
            for name, angle, q in extra.get(i, ()):
                getattr(out, name)(angle, q)
            if i in replacements:
                for name, angle, q in replacements[i]:
                    getattr(out, name)(angle, q)
            else:
                out.append(ins)
        return out

    def depth(qc):
        levels = [0] * n
        for ins in qc.data:
            if ins.operation.name in ("move", "barrier"):
                continue
            qs = [indices[q] for q in ins.qubits]
            level = max(levels[q] for q in qs) + 1
            for q in qs:
                levels[q] = level
        return max(levels, default=0)

    candidates = (lower(paulis, late) for paulis, late in itertools.product((False, True), repeat=2))
    return min(itertools.chain((circuit,), candidates), key=depth)


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
                qc = _lower_z_phases(qc)
                qc.global_phase = 0
                return qc, Architecture(home, spec=architecture_spec)
            except Illegal:
                pass
    try:
        prog = parse_cz_program(circuit)
        ncz = sum(1 for op in prog[1] if op[0] == "cz") if prog is not None else 0
        if prog is not None and prog[0] <= 20 and ncz <= 240:
            n, ops = prog
            best = None
            big = ncz > 60  # fixed work budget: one variant for larger programs
            for home in (snake_layout(n),) + (() if big else (default_layout(n, n - 1),)):
                for rs in ((1,) if big else (6, 1)):
                    for lz in ((True,) if big else (True, False)):
                        try:
                            ex = compile_cz_program(n, ops, home, emit=False, rect_sim=rs, lazy=lz)
                        except Illegal:
                            continue
                        c = ex.m.dist + DEPTH_W * (max(ex.m.depth) + 1)
                        if best is None or c < best[0] - 1e-9:
                            best = (c, home, rs, lz)
            if best is not None:
                ex = compile_cz_program(n, ops, best[1], emit=True, rect_sim=best[2], lazy=best[3])
                return ex.m.qc, Architecture(best[1], spec=architecture_spec)
    except Illegal:
        pass
    return _naive(circuit, architecture_spec)


# ================================================================ l3 peephole pass (l3/make_l3.py)
import sys as _l3sys
import types as _l3types

_L3_SOURCES = {}

_L3_SOURCES['model'] = '"""Abstract L3 model: a compiled circuit reduced to its pulse schedule, and back.\n\nSchedule (the L3 input):\n    pulses[k]  = list of pairs (a, b) that must sit on one interaction pair at pulse k; nobody else\n                 may be in the entangling zone then (idle rule).\n    seg[q][i]  = 1q instructions of atom q between its (i-1)-th and i-th CZ (i = 0: before the\n                 first CZ; i = ncz[q]: after the last). They need the atom in storage somewhere in\n                 that window.\n    owe[k]     = atoms of pulse k whose segment right before pulse k is non-empty (the `s` marks).\n    meas       = final measurement instructions.\n\nGates and their per-qubit order are kept exactly, so any plan that realises the schedule is\nequivalent to the source and has the same gate depth (or less, after cancelling cz.cz pairs).\n"""\n\nimport itertools\nimport math\nimport pickle\nfrom dataclasses import dataclass, field\n\nfrom qiskit import QuantumCircuit\nfrom qiskit.circuit import Instruction\n\n# ---------------------------------------------------------------- geometry (default spec)\n\nSX, SY0, SY1 = 10, 0, 9      # storage: x 0..9, y 0..9\nEY0, EY1 = 10, 19            # entangling: x 0..9, y 10..19, pairs (2k, 2k+1), 0.3 apart\nPAIR_D = 0.3\n\n\ndef phys(s):\n    x, y = s\n    if y >= EY0:\n        p, r = divmod(x, 2)\n        return (2.0 * p + r * PAIR_D, float(y))\n    return (float(x), float(y))\n\n\ndef dist(a, b):\n    (ax, ay), (bx, by) = phys(a), phys(b)\n    return math.hypot(ax - bx, ay - by)\n\n\ndef in_zone(s):\n    return s[1] >= EY0\n\n\ndef zone_of(s):\n    return "e" if s[1] >= EY0 else "s"\n\n\nclass Move(Instruction):\n    def __init__(self, rows, cols, new_rows, new_cols, num_qubits):\n        super().__init__("move", num_qubits, 0, [tuple(rows), tuple(cols), tuple(new_rows), tuple(new_cols)])\n\n    def validate_parameter(self, parameter):\n        return parameter\n\n    def _define(self):\n        self.definition = QuantumCircuit(self.num_qubits, name=self.name)\n\n\n# ---------------------------------------------------------------- state + legal moves\n\n\nclass MoveError(ValueError):\n    pass\n\n\nclass State:\n    def __init__(self, pos):\n        self.pos = dict(pos)\n        self.occ = {s: q for q, s in self.pos.items()}\n\n    def copy(self):\n        st = State.__new__(State)\n        st.pos, st.occ = dict(self.pos), dict(self.occ)\n        return st\n\n    def check(self, group):\n        """group = {q: target}. Returns (move params, cost) or raises MoveError. Validator rules +\n        single-zone source/target rectangles."""\n        src = {q: self.pos[q] for q in group}\n        xs = [s[0] for s in src.values()]\n        ys = [s[1] for s in src.values()]\n        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)\n        for x in range(x0, x1 + 1):\n            for y in range(y0, y1 + 1):\n                o = self.occ.get((x, y))\n                if o is not None and o not in group:\n                    raise MoveError("ghost")\n        rmap, cmap = {}, {}\n        for q, (sx, sy) in src.items():\n            tx, ty = group[q]\n            if rmap.setdefault(sy, ty) != ty or cmap.setdefault(sx, tx) != tx:\n                raise MoveError("not a product map")\n        for mp in (rmap, cmap):\n            ks = sorted(mp)\n            if any(mp[a] >= mp[b] for a, b in itertools.pairwise(ks)):\n                raise MoveError("order")\n        if len({zone_of(s) for s in src.values()}) != 1 or len({zone_of(t) for t in group.values()}) != 1:\n            raise MoveError("zones")\n        for t in group.values():\n            if not (0 <= t[0] <= 9 and 0 <= t[1] <= 19):\n                raise MoveError("outside")\n            o = self.occ.get(t)\n            if o is not None and o not in group:\n                raise MoveError("occupied")\n        cost = max(dist(src[q], group[q]) for q in group)\n        params = (tuple(range(y0, y1 + 1)), tuple(range(x0, x1 + 1)),\n                  tuple(rmap[y] for y in sorted(rmap)), tuple(cmap[x] for x in sorted(cmap)))\n        return params, cost\n\n    def ok(self, group):\n        try:\n            return self.check(group)\n        except MoveError:\n            return None\n\n    def apply(self, group):\n        params, cost = self.check(group)\n        for q in group:\n            del self.occ[self.pos[q]]\n        for q, t in group.items():\n            self.pos[q] = t\n            self.occ[t] = q\n        return params, cost\n\n\n# ---------------------------------------------------------------- schedule\n\n\n@dataclass\nclass Schedule:\n    n: int\n    pulses: list            # [[(a, b), ...], ...]\n    seg: list               # seg[q] = [[instr, ...], ...]\n    meas: list              # [(instruction, qubits, clbits)]\n    template: QuantumCircuit = field(repr=False, default=None)\n\n    @property\n    def K(self):\n        return len(self.pulses)\n\n    def cz_index(self):\n        """ev[q] = list of (pulse k, partner) in order."""\n        ev = [[] for _ in range(self.n)]\n        for k, P in enumerate(self.pulses):\n            for a, b in P:\n                ev[a].append((k, b))\n                ev[b].append((k, a))\n        return ev\n\n    def owe(self):\n        """owe[k] = atoms of pulse k with a non-empty segment right before pulse k."""\n        ev = self.cz_index()\n        o = [set() for _ in range(self.K)]\n        for q in range(self.n):\n            for i, (k, _) in enumerate(ev[q]):\n                if self.seg[q][i]:\n                    o[k].add(q)\n        return o\n\n    def final_owe(self):\n        ev = self.cz_index()\n        return {q for q in range(self.n) if self.seg[q][len(ev[q])]}\n\n    def members(self):\n        return [{q for p in P for q in p} for P in self.pulses]\n\n\ndef extract(circuit: QuantumCircuit) -> Schedule:\n    qi = {b: i for i, b in enumerate(circuit.qubits)}\n    n = circuit.num_qubits\n    pulses, seg, meas = [], [[[]] for _ in range(n)], []\n    cur = None  # list of pulses of the current CZ layer\n\n    def close():\n        nonlocal cur\n        if cur:\n            pulses.extend(cur)\n        cur = None\n\n    for ins in circuit.data:\n        name = ins.operation.name\n        qs = [qi[q] for q in ins.qubits]\n        if name == "cz":\n            if cur is None:\n                cur = [[]]\n            a, b = qs\n            # a pair seen again in this layer starts the next pulse of the layer\n            k = 0\n            while k < len(cur) and any(a in p or b in p for p in cur[k]):\n                k += 1\n            if k == len(cur):\n                cur.append([])\n            cur[k].append((a, b))\n            continue\n        if name == "move":\n            close()\n            continue\n        if name in ("rx", "rz"):\n            if cur is not None and any(qs[0] in p for P in cur for p in P):\n                raise ValueError("1q gate on a pulsed atom inside a CZ layer")\n        elif name == "measure":\n            meas.append((ins.operation, qs, [circuit.find_bit(c).index for c in ins.clbits]))\n        elif name == "barrier":\n            continue\n        else:\n            raise ValueError("unexpected op " + name)\n    close()\n    # segments in CZ order: a new segment per CZ of the atom\n    seg = [[[]] for _ in range(n)]\n    for ins in circuit.data:\n        name = ins.operation.name\n        qs = [qi[q] for q in ins.qubits]\n        if name == "cz":\n            for q in qs:\n                seg[q].append([])\n        elif name in ("rx", "rz"):\n            seg[qs[0]][-1].append(ins.operation)\n    sch = Schedule(n, pulses, seg, meas, circuit.copy_empty_like())\n    return sch\n\n\ndef cancel_dummies(sch: Schedule) -> int:\n    """Remove cz.cz pairs with nothing in between on either atom (identity). Returns #removed CZ."""\n    removed = 0\n    while True:\n        ev = sch.cz_index()\n        kill = None\n        for q in range(sch.n):\n            for i in range(len(ev[q]) - 1):\n                (k1, p1), (k2, p2) = ev[q][i], ev[q][i + 1]\n                if p1 != p2 or sch.seg[q][i + 1]:\n                    continue\n                j = ev[p1].index((k1, q))\n                if j + 1 < len(ev[p1]) and ev[p1][j + 1] == (k2, q) and not sch.seg[p1][j + 1]:\n                    kill = (q, i, p1, j, k1, k2)\n                    break\n            if kill:\n                break\n        if not kill:\n            break\n        q, i, p, j, k1, k2 = kill\n        for a, ia in ((q, i), (p, j)):\n            s = sch.seg[a]\n            s[ia:ia + 3] = [s[ia] + s[ia + 2]]   # merge segment before cz#1 and after cz#2\n        for k in (k1, k2):\n            sch.pulses[k] = [pr for pr in sch.pulses[k] if set(pr) != {q, p}]\n        removed += 2\n    sch.pulses = [P for P in sch.pulses if P]\n    return removed\n\n\ndef load_cached(path):\n    with open(path, "rb") as f:\n        d = pickle.load(f)\n    qc = QuantumCircuit(d["nq"], d["nc"])\n    for name, qs, cs, params in d["ops"]:\n        if name == "move":\n            qc.append(Move(*params, len(qs)), qs)\n        elif name == "rx":\n            qc.rx(params[0], qs[0])\n        elif name == "rz":\n            qc.rz(params[0], qs[0])\n        elif name == "cz":\n            qc.cz(*qs)\n        elif name == "measure":\n            qc.measure(qs[0], cs[0])\n        elif name == "barrier":\n            qc.barrier(*qs)\n        else:\n            raise ValueError(name)\n    return qc, {int(q): tuple(s) for q, s in d["loading"].items()}\n\n\n# ---------------------------------------------------------------- plan -> circuit\n\n\nclass Emitter:\n    """Realise a schedule step by step. The planner calls move()/pulse(); 1q segments are emitted as\n    soon as their atom is in storage and its previous CZ is done. Checks every rule as it goes."""\n\n    def __init__(self, sch: Schedule, loading):\n        self.sch = sch\n        self.st = State(loading)\n        self.loading = dict(loading)\n        self.out = sch.template.copy_empty_like()\n        self.ev = sch.cz_index()\n        self.done = [0] * sch.n          # CZs done per atom\n        self.emitted = [-1] * sch.n      # index of the last emitted segment\n        self.k = 0                       # next pulse\n        self.dist = 0.0\n        self.nmoves = 0\n        self._flush(range(sch.n))\n\n    def _flush(self, qs):\n        for q in qs:\n            i = self.done[q]\n            if self.emitted[q] < i and not in_zone(self.st.pos[q]):\n                for op in self.sch.seg[q][i]:\n                    self.out.append(op, [self.out.qubits[q]])\n                self.emitted[q] = i\n\n    def pending(self, q):\n        """True if q still owes its current segment (must visit storage before its next CZ)."""\n        i = self.done[q]\n        return self.emitted[q] < i and bool(self.sch.seg[q][i])\n\n    def move(self, group):\n        if not group:\n            return 0.0\n        params, cost = self.st.apply(group)\n        qs = sorted(group)\n        self.out.append(Move(*params, len(qs)), [self.out.qubits[q] for q in qs])\n        self.dist += cost\n        self.nmoves += 1\n        self._flush(qs)\n        return cost\n\n    def pulse(self):\n        k = self.k\n        P = self.sch.pulses[k]\n        mem = {q for p in P for q in p}\n        zone = {q for q, s in self.st.pos.items() if in_zone(s)}\n        if zone != mem:\n            raise MoveError(f"pulse {k}: zone {sorted(zone)} != members {sorted(mem)}")\n        for a, b in P:\n            (xa, ya), (xb, yb) = self.st.pos[a], self.st.pos[b]\n            if ya != yb or min(xa, xb) % 2 or abs(xa - xb) != 1:\n                raise MoveError(f"pulse {k}: {a},{b} not on a pair")\n            for q in (a, b):\n                i = self.done[q]\n                if self.emitted[q] < i and self.sch.seg[q][i]:\n                    raise MoveError(f"pulse {k}: atom {q} still owes 1q gates")\n                self.emitted[q] = max(self.emitted[q], i)\n        for a, b in P:\n            self.out.cz(a, b)\n        for a, b in P:\n            self.done[a] += 1\n            self.done[b] += 1\n        self.k += 1\n        self._flush(mem)\n\n    def finish(self):\n        if self.k != self.sch.K:\n            raise MoveError("not all pulses done")\n        for q in range(self.sch.n):\n            if self.pending(q):\n                raise MoveError(f"atom {q} ends with unemitted gates")\n        for op, qs, cs in self.sch.meas:\n            self.out.append(op, [self.out.qubits[q] for q in qs], [self.out.clbits[c] for c in cs])\n        return self.out, self.loading\n\n\ndef schedule_signature(sch: Schedule):\n    ev = sch.cz_index()\n    sig = []\n    for q in range(sch.n):\n        s = []\n        for i in range(len(ev[q]) + 1):\n            s += [(op.name, tuple(round(float(p), 12) for p in op.params)) for op in sch.seg[q][i]]\n            if i < len(ev[q]):\n                s.append(("cz", ev[q][i][1]))\n        sig.append(s)\n    return sig\n\n\ndef signature(circuit: QuantumCircuit):\n    """Per-qubit sequence of (gate, params, partner) with moves/barriers removed: equal signatures\n    (with consistent CZ pairing) mean an identical gate DAG, hence equivalence and equal depth."""\n    qi = {b: i for i, b in enumerate(circuit.qubits)}\n    sig = [[] for _ in range(circuit.num_qubits)]\n    for ins in circuit.data:\n        name = ins.operation.name\n        if name in ("move", "barrier", "measure"):\n            continue\n        qs = [qi[q] for q in ins.qubits]\n        if name == "cz":\n            a, b = qs\n            sig[a].append(("cz", b))\n            sig[b].append(("cz", a))\n        else:\n            sig[qs[0]].append((name, tuple(round(float(p), 12) for p in ins.operation.params)))\n    return sig'

_L3_SOURCES['group'] = '"""Rectangle grouping: execute a set of transfers {atom: target} as few, cheap legal moves.\n\nGreedy: repeatedly take the executable transfers (target free now), grow maximal legal groups\nfrom a few seeds, apply the group with the best saving. Cycles (every target occupied by another\nmover) are broken by parking one mover on a free temporary site.\n"""\n\nM = _sys.modules[\'_l3_model\']\n\n\ndef _free_sites(st, zone, near, limit=40):\n    """Free sites in `zone` (\'s\' or \'e\') sorted by distance to `near`."""\n    ys = range(0, 10) if zone == "s" else range(10, 20)\n    out = []\n    for y in ys:\n        for x in range(10):\n            if (x, y) not in st.occ:\n                out.append((M.dist((x, y), near), (x, y)))\n    out.sort()\n    return [s for _, s in out[:limit]]\n\n\ndef execute(em, transfers, seeds=6, order=None):\n    """Apply transfers through the emitter `em` (which owns the state). Returns total cost."""\n    st = em.st\n    rem = {q: t for q, t in transfers.items() if st.pos[q] != t}\n    total = 0.0\n    guard = 0\n    while rem:\n        guard += 1\n        if guard > 4 * len(transfers) + 50:\n            raise M.MoveError("grouping did not converge")\n        exe = [q for q, t in rem.items() if t not in st.occ]\n        if not exe:\n            # cycle: park one mover whose target is held by another mover\n            q = next(iter(rem))\n            blocker = st.occ[rem[q]]\n            tmp = next(s for s in _free_sites(st, M.zone_of(rem[blocker]), rem[blocker]))\n            if M.zone_of(tmp) != M.zone_of(st.pos[blocker]) and M.zone_of(st.pos[blocker]) != M.zone_of(rem[blocker]):\n                pass\n            total += em.move({blocker: tmp})\n            continue\n        d = {q: M.dist(st.pos[q], rem[q]) for q in exe}\n        cand = sorted(exe, key=lambda q: -d[q])\n        best = None\n        for seed in cand[:seeds]:\n            g = {seed: rem[seed]}\n            sx, sy = st.pos[seed]\n            others = sorted((q for q in exe if q != seed),\n                            key=lambda q: abs(st.pos[q][0] - sx) + abs(st.pos[q][1] - sy))\n            for q in others:\n                g[q] = rem[q]\n                if st.ok(g) is None:\n                    del g[q]\n            res = st.ok(g)\n            cost = res[1]\n            saving = sum(d[q] for q in g) - cost\n            key = (saving, len(g), -cost)\n            if best is None or key > best[0]:\n                best = (key, g)\n        g = best[1]\n        total += em.move(g)\n        for q in g:\n            del rem[q]\n    return total'

_L3_SOURCES['chain'] = '"""L3 as a chain-structured optimisation problem.\n\n    minimise   sum_k g_k(X_{k-1}, X_k)\n    subject to X_k feasible for pulse k  (members on pairs next to their partner, others in storage)\n\nX_k   = configuration (atom -> site) at pulse k; X_{-1} = initial loading, X_K = final configuration.\ng_k   = cost of the cheapest legal move sequence X_{k-1} -> X_k in which every atom of\n        B_k (in pulses k-1 and k, owing 1q gates) touches storage (inner problem).\n\nThis module: extract X and the per-gap moves from any compiled output (incumbent), an inner\nsolver `transition` (greedy set partitioning into rectangle moves), and assembly back into a\ncircuit through the checked Emitter.\n"""\n\nM = _sys.modules[\'_l3_model\']\nexecute = _sys.modules[\'_l3_group\'].execute\n\n\ndef parse_moves(circ):\n    """Yield (\'move\', {q: target}) / (\'cz\', (a, b)) / (\'g\', op, q) / (\'m\', ...) in order, using\n    validator semantics for the move targets."""\n    qi = {b: i for i, b in enumerate(circ.qubits)}\n    return qi\n\n\ndef incumbent(circ, loading, sch):\n    """From a compiled circuit whose schedule is `sch` (not dummy-cancelled): configurations X\n    (len K+2: loading, pulse 0..K-1, final) and per-gap move lists (len K+1) of {q: target}."""\n    qi = {b: i for i, b in enumerate(circ.qubits)}\n    st = M.State(loading)\n    X = [dict(st.pos)]\n    gaps = [[]]\n    in_layer = False\n    for ins in circ.data:\n        name = ins.operation.name\n        if name == "cz":\n            if not in_layer:\n                X.append(dict(st.pos))\n                in_layer = True\n            continue\n        if name == "move":\n            if in_layer:\n                gaps.append([])\n                in_layer = False\n            qs = [qi[q] for q in ins.qubits]\n            rows, cols, nr, nc = ins.operation.params\n            occ_r = [y for y in rows if any(st.pos[q][1] == y for q in qs)]\n            occ_c = [x for x in cols if any(st.pos[q][0] == x for q in qs)]\n            if len(nr) == len(rows):\n                occ_r = list(rows)\n            if len(nc) == len(cols):\n                occ_c = list(cols)\n            rm, cm = dict(zip(occ_r, nr)), dict(zip(occ_c, nc))\n            g = {q: (cm[st.pos[q][0]], rm[st.pos[q][1]]) for q in qs}\n            st.apply(g)\n            gaps[-1].append(g)\n    if in_layer:\n        gaps.append([])\n    X.append(dict(st.pos))\n    assert len(X) == sch.K + 2 and len(gaps) == sch.K + 1, (len(X), len(gaps), sch.K)\n    return X, gaps\n\n\ndef binding(sch):\n    """B[k] = atoms in pulses k-1 and k that owe 1q gates before pulse k; B[K] = final owers in\n    the zone at the end are handled by the final config being in storage."""\n    mem = sch.members()\n    owe = sch.owe()\n    B = [set() for _ in range(sch.K + 1)]\n    for k in range(1, sch.K):\n        B[k] = mem[k - 1] & mem[k] & owe[k]\n    return B\n\n\ndef gap_cost(start, moves):\n    st = M.State(start)\n    c = 0.0\n    for g in moves:\n        c += st.apply(g)[1]\n    return c\n\n\nclass _Rec:\n    """Minimal emitter stand-in for the grouping engine: records moves on a State."""\n\n    def __init__(self, pos):\n        self.st = M.State(pos)\n        self.moves = []\n        self.cost = 0.0\n\n    def move(self, g):\n        if not g:\n            return 0.0\n        _, c = self.st.apply(g)\n        self.moves.append(dict(g))\n        self.cost += c\n        return c\n\n\ndef _temp_storage(st, q, avoid):\n    """Free storage site closest to q (not in avoid)."""\n    best = None\n    for y in range(9, -1, -1):\n        for x in range(10):\n            s = (x, y)\n            if s in st.occ or s in avoid:\n                continue\n            c = M.dist(st.pos[q], s)\n            if best is None or c < best[0]:\n                best = (c, s)\n    return best[1]\n\n\ndef transition(start, end, dip, seeds=6):\n    """Inner solver: legal moves start -> end with every atom of `dip` touching storage.\n    Phases: (1) zone atoms going to storage + dippers (rigid row dip if possible), (2) storage\n    rearrangement, (3) zone->zone, (4) storage->zone. Returns (moves, cost)."""\n    rec = _Rec(start)\n    st = rec.st\n    endset = set(end.values())\n    dippers = [q for q in dip if M.in_zone(st.pos[q])]\n    if dippers and not _rigid_dip(rec, dippers, end):\n        tmp = {}\n        for q in sorted(dippers, key=lambda q: st.pos[q]):\n            tmp[q] = _temp_storage(st, q, endset | set(tmp.values()))\n        execute(rec, tmp, seeds=seeds)\n    execute(rec, {q: end[q] for q in end if st.pos[q] != end[q]}, seeds=seeds)\n    return rec.moves, rec.cost\n\n\ndef _rigid_dip(rec, dippers, end):\n    """Shift the zone rows holding the dippers down into free storage rows and back up (only\n    when every atom in those rows ends where it is now). Returns True if done."""\n    st = rec.st\n    rows = sorted({st.pos[q][1] for q in dippers})\n    block = [q for q, s in st.pos.items() if M.in_zone(s) and s[1] in rows]\n    h = len(rows)\n    for top in range(9, h - 2, -1):\n        trows = list(range(top - h + 1, top + 1))\n        if trows[0] < 0:\n            break\n        rmap = dict(zip(rows, trows))\n        down = {q: (st.pos[q][0], rmap[st.pos[q][1]]) for q in block}\n        if st.ok(down) is None:\n            continue\n        back = {q: st.pos[q] for q in block}\n        t = st.copy()\n        t.apply(down)\n        if t.ok(back) is None:\n            continue\n        rec.move(down)\n        rec.move(back)\n        return True\n    return False\n\n\ndef assemble(sch, loading, gaps):\n    """Replay per-gap move lists through the checked Emitter; returns (circuit, loading, dist)."""\n    em = M.Emitter(sch, loading)\n    for k in range(sch.K):\n        for g in gaps[k]:\n            em.move(g)\n        em.pulse()\n    for g in gaps[sch.K]:\n        em.move(g)\n    out, load = em.finish()\n    return out, load, em.dist'

_L3_SOURCES['peep'] = '#!/usr/bin/env python3\n"""Peephole optimiser on a move plan (fast, MILP-free). Operators found by the MIP-LNS:\n\n  P1 merge     : move j joins move i (i < j) or move i joins move j\n  P2 trim      : an atom that rides a down+up round trip it does not need leaves both moves\n  P3 waypoint  : an intermediate site of an atom (target of one move, source of a later one) is\n                 re-chosen among nearby free sites\n  P4 drop-atom : an atom is removed from a move and given its own move (enables other merges)\n\nEvery candidate rewrite of a gap is replayed exactly: legal moves, same end configuration, and the\nrequired storage visits (B_k). Accept if the gap gets cheaper. Repeat until no operator helps.\n\n    python l3/peep.py <cached pkl> [more pkls]\n"""\n\nimport sys\nimport time\nfrom pathlib import Path\n\nC = _sys.modules[\'_l3_chain\']\nM = _sys.modules[\'_l3_model\']\n\n\ndef replay(start, moves, need, end):\n    """Exact cost of a gap, or None if illegal / wrong end / missing storage visit."""\n    st = M.State(start)\n    seen = {q for q in need if not M.in_zone(st.pos[q])}\n    c = 0.0\n    try:\n        for g in moves:\n            if not g:\n                continue\n            c += st.apply(g)[1]\n            for q in need:\n                if q not in seen and not M.in_zone(st.pos[q]):\n                    seen.add(q)\n    except M.MoveError:\n        return None\n    if st.pos != end or len(seen) != len(need):\n        return None\n    return c\n\n\ndef gap_need(sch, k, start, end, B):\n    if k < sch.K:\n        return set(B[k])\n    fo = sch.final_owe()\n    return {q for q in fo if M.in_zone(start[q]) and M.in_zone(end[q])}\n\n\nLAST_EVALS = [0]\n\n\ndef optimise_gap(start, moves, need, end, max_span=8, budget=4000):\n    """Greedy first-improvement over the operators until no change. Returns (moves, cost);\n    the number of replays used is left in LAST_EVALS[0]."""\n    moves = [dict(g) for g in moves if g]\n    best = replay(start, moves, need, end)\n    LAST_EVALS[0] = 0\n    if best is None:\n        return moves, None\n    evals = 0\n    improved = True\n    while improved and evals < budget:\n        improved = False\n        n = len(moves)\n        # P1 merge\n        for i in range(n):\n            for j in range(i + 1, min(n, i + 1 + max_span)):\n                if set(moves[i]) & set(moves[j]):\n                    continue\n                for into in (i, j):\n                    cand = [dict(g) for g in moves]\n                    cand[into] = {**moves[i], **moves[j]}\n                    del cand[j if into == i else i]\n                    evals += 1\n                    c = replay(start, cand, need, end)\n                    if c is not None and c < best - 1e-9:\n                        moves, best, improved = cand, c, True\n                        break\n                if improved:\n                    break\n            if improved:\n                break\n        if improved:\n            continue\n        # P2 trim: atom q down in move i and back to the same site in move j, not needed\n        states = [M.State(start)]\n        for g in moves:\n            s2 = states[-1].copy()\n            s2.apply(g)\n            states.append(s2)\n        for i in range(n):\n            for q, t in moves[i].items():\n                if q in need or not (M.in_zone(states[i].pos[q]) != M.in_zone(t)):\n                    continue\n                for j in range(i + 1, n):\n                    if q in moves[j]:\n                        if moves[j][q] == states[i].pos[q]:\n                            cand = [dict(g) for g in moves]\n                            del cand[i][q]\n                            del cand[j][q]\n                            cand = [g for g in cand if g]\n                            evals += 1\n                            c = replay(start, cand, need, end)\n                            if c is not None and c < best - 1e-9:\n                                moves, best, improved = cand, c, True\n                        break\n                if improved:\n                    break\n            if improved:\n                break\n        if improved:\n            continue\n        # P3 waypoint: atom q targeted by move i and moved again by a later move j\n        for i in range(n):\n            for q in list(moves[i]):\n                j = next((j for j in range(i + 1, n) if q in moves[j]), None)\n                if j is None:\n                    continue\n                cur = moves[i][q]\n                occ_all = set()\n                for s in states[i:j + 1]:\n                    occ_all |= set(s.occ)\n                cands = []\n                for x in range(10):\n                    for y in range(20):\n                        s = (x, y)\n                        if s == cur or s in occ_all or M.zone_of(s) != M.zone_of(cur):\n                            continue\n                        dd = M.dist(states[i].pos[q], s) + M.dist(s, moves[j][q])\n                        cands.append((dd, s))\n                cands.sort()\n                for _, s in cands[:12]:\n                    cand = [dict(g) for g in moves]\n                    cand[i][q] = s\n                    evals += 1\n                    c = replay(start, cand, need, end)\n                    if c is not None and c < best - 1e-9:\n                        moves, best, improved = cand, c, True\n                        break\n                if improved:\n                    break\n            if improved:\n                break\n        if improved:\n            continue\n        # P2b collective trim: move j reverses move i on a subset; drop all unneeded atoms of it\n        # at once and send the needed ones to the nearest free storage site\n        for i in range(n):\n            for j in range(i + 1, min(n, i + 1 + max_span)):\n                rt = [q for q, t in moves[i].items() if q in moves[j] and moves[j][q] == states[i].pos[q]\n                      and M.in_zone(states[i].pos[q]) and not M.in_zone(t)]\n                if len(rt) < 2:\n                    continue\n                cand = [dict(g) for g in moves]\n                keep = [q for q in rt if q in need]\n                for q in rt:\n                    if q not in need:\n                        del cand[i][q]\n                        del cand[j][q]\n                if keep:\n                    occ_all = set()\n                    for s in states[i:j + 2]:\n                        occ_all |= {site for site, a in s.occ.items() if a not in rt}\n                    occ_all |= {states[i].pos[a] for a in rt}\n                    for q in keep:\n                        best_s = min(((M.dist(states[i].pos[q], (x, y)), (x, y)) for x in range(10) for y in range(10)\n                                      if (x, y) not in occ_all), default=None)\n                        if best_s:\n                            cand[i][q] = best_s[1]\n                            occ_all.add(best_s[1])\n                cand = [g for g in cand if g]\n                evals += 1\n                c = replay(start, cand, need, end)\n                if c is not None and c < best - 1e-9:\n                    moves, best, improved = cand, c, True\n                    break\n            if improved:\n                break\n        if improved:\n            continue\n        # P5 ride-along: an atom with a single-atom move at i rides with a translation move j\n        # (j < i: rides first, then fixes up; j > i: goes to t - v first, then rides to t)\n        for i in range(n):\n            if len(moves[i]) != 1:\n                continue\n            (q, t), = moves[i].items()\n            p = states[i].pos[q]\n            for j in range(max(0, i - max_span), min(n, i + 1 + max_span)):\n                if j == i or q in moves[j]:\n                    continue\n                vs = {(tt[0] - states[j].pos[a][0], tt[1] - states[j].pos[a][1]) for a, tt in moves[j].items()}\n                if len(vs) != 1:\n                    continue\n                (vx, vy), = vs\n                if any(q in moves[h] for h in range(min(i, j) + 1, max(i, j))):\n                    continue\n                cand = [dict(g) for g in moves]\n                if j < i:\n                    w = (p[0] + vx, p[1] + vy)\n                    if not (0 <= w[0] <= 9 and 0 <= w[1] <= 19):\n                        continue\n                    cand[j][q] = w\n                    cand[i] = {q: t}\n                else:\n                    w = (t[0] - vx, t[1] - vy)\n                    if not (0 <= w[0] <= 9 and 0 <= w[1] <= 19):\n                        continue\n                    cand[i] = {q: w}\n                    cand[j][q] = t\n                cand = [g for g in cand if g]\n                evals += 1\n                c = replay(start, cand, need, end)\n                if c is not None and c < best - 1e-9:\n                    moves, best, improved = cand, c, True\n                    break\n            if improved:\n                break\n        if improved:\n            continue\n        # P5b staging absorbed: single-atom move i of q, later translation move j also moving q:\n        # skip the staging move, ride with j from q\'s original position, fix up after j\n        for i in range(n):\n            if len(moves[i]) != 1:\n                continue\n            (q, t), = moves[i].items()\n            p = states[i].pos[q]\n            j = next((j for j in range(i + 1, n) if q in moves[j]), None)\n            if j is None or len(moves[j]) < 2:\n                continue\n            vs = {(tt[0] - states[j].pos[a][0], tt[1] - states[j].pos[a][1]) for a, tt in moves[j].items() if a != q}\n            if len(vs) != 1:\n                continue\n            (vx, vy), = vs\n            w = (p[0] + vx, p[1] + vy)\n            if not (0 <= w[0] <= 9 and 0 <= w[1] <= 19):\n                continue\n            final = moves[j][q]\n            cand = [dict(g) for g in moves]\n            cand[j][q] = w\n            cand.insert(j + 1, {q: final})\n            del cand[i]\n            evals += 1\n            c = replay(start, cand, need, end)\n            if c is not None and c < best - 1e-9:\n                moves, best, improved = cand, c, True\n                break\n        if improved:\n            continue\n        # P6 general ride-along: single-atom move {q: t} at i becomes a short pre-move {q: w} at i\n        # plus q joining a later move j with target t (non-uniform maps allowed; replay decides)\n        for i in range(n):\n            if len(moves[i]) != 1:\n                continue\n            (q, t), = moves[i].items()\n            p = states[i].pos[q]\n            old_i = M.dist(p, t)\n            for j in range(i + 1, min(n, i + 1 + max_span)):\n                if q in moves[j] or any(q in moves[h] for h in range(i + 1, j)):\n                    continue\n                tried_here = 0\n                for dx in (-2, -1, 0, 1, 2):\n                    for dy in (-2, -1, 0, 1, 2):\n                        w = (p[0] + dx, p[1] + dy)\n                        if w == p or not (0 <= w[0] <= 9 and 0 <= w[1] <= 19) or M.zone_of(w) != M.zone_of(p):\n                            continue\n                        if M.dist(p, w) >= old_i - 1e-9:\n                            continue\n                        cand = [dict(g) for g in moves]\n                        cand[i] = {q: w}\n                        cand[j][q] = t\n                        evals += 1\n                        tried_here += 1\n                        c = replay(start, cand, need, end)\n                        if c is not None and c < best - 1e-9:\n                            moves, best, improved = cand, c, True\n                            break\n                    if improved:\n                        break\n                if improved:\n                    break\n            if improved:\n                break\n        if improved:\n            continue\n        # P4 split an atom off a multi-atom move into its own move (right after), then hope P1 merges\n        for i in range(n):\n            if len(moves[i]) < 2:\n                continue\n            for q in list(moves[i]):\n                for pos in (i, i + 1):\n                    cand = [dict(g) for g in moves]\n                    t = cand[i].pop(q)\n                    cand.insert(pos if pos == i else i + 1, {q: t})\n                    evals += 1\n                    c = replay(start, cand, need, end)\n                    if c is not None and c < best - 1e-9:\n                        moves, best, improved = cand, c, True\n                        break\n                if improved:\n                    break\n            if improved:\n                break\n    LAST_EVALS[0] = evals\n    return moves, best\n\n\ndef peep(path, log=None):\n    c, l = M.load_cached(path)\n    s = M.extract(c)\n    X, gaps = C.incumbent(c, l, s)\n    B = C.binding(s)\n    t0 = time.time()\n    base = new = 0.0\n    for k in range(s.K + 1):\n        need = gap_need(s, k, X[k], X[k + 1], B)\n        old = C.gap_cost(X[k], gaps[k])\n        mv, cst = optimise_gap(X[k], gaps[k], need, X[k + 1])\n        base += old\n        if cst is not None and cst < old - 1e-9:\n            gaps[k] = mv\n            new += cst\n        else:\n            new += old\n    out, load, d = C.assemble(s, l, gaps)\n    return {"file": Path(path).stem, "base": round(base, 3), "peep": round(d, 3), "secs": round(time.time() - t0, 1)}, out, load, s, gaps\n\n'



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


def compile_circuit(circuit, architecture_spec):
    out, arch = _l3_base_compile_circuit(circuit, architecture_spec)
    try:
        res = _l3_optimise(out, dict(arch.initial_loading))
        if res is not None:
            out = res
    except Exception:  # noqa: BLE001 - keep the component's own (valid) result
        pass
    return out, arch
