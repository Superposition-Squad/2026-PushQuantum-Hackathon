"""relay_plus (l3abstract): relay + solo improvements made after the solo2 fork (b9a2a15 hub rotations
and phase lowering, d85a571 phase retiming and boundary moves, uncommitted hub-wave scheduling and
local resynthesis), merged 3-way on base solo@682d68b.

relay (fork of solo2: relayed hub fan-out, shorter Euler sequences): a compiler specialised on the scored iHVA circuits (validator v5, `idle` rule).

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

* Relayed fan-out (new here).  A hub batch is F . prod rx_p(th_p) . F with F = prod_p cz(hub, p).  F only
  depends on the parity of the partners: with a controlled-Y  P = rx_r(-pi/2) cz(l, r) rx_r(pi/2)  on a
  partner pair (l, r),  cz(hub, l) cz(hub, r) = P cz(hub, r) P  exactly.  So partners are grouped into
  pairs (one level) or paths of four (two levels); all groups are pulsed together in a storage band and
  the hub only meets one root per group: half or a quarter of the sequential hub pulses.  Above 19
  qubits the validator wants every input pair to interact, so relay pairs are input pairs and every
  partner is a root at least once in the circuit (roots alternate between the two passes; tracked).

Anything that is not iHVA-shaped goes to a simple valid fallback (the naive scheme).
"""

import itertools
import math
from bisect import bisect_left
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


@lru_cache(maxsize=4096)
def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


@lru_cache(maxsize=4096)
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
    """Native Euler angles, shared across simulations of repeated local gates."""
    return _cached_zxz(np.asarray(u, dtype=complex).tobytes())


@lru_cache(maxsize=8192)
def _cached_zxz(data):
    u = np.frombuffer(data, dtype=complex).reshape(2, 2)
    det = np.linalg.det(u)
    v = u / np.sqrt(det)  # SU(2)
    a, b = v[0, 0], v[0, 1]
    if abs(b) < 1e-10:
        lam = -2 * np.angle(a)
        return (("rz", lam),) if abs(_wrap(lam)) > 1e-9 else ()
    if abs(a) < 1e-10:
        # v = Rz(phi) Rx(pi) Rz(lam): only phi+... fix lam = 0
        th = math.pi
        # Rz(phi) Rx(pi) = [[0, -i e^{-i phi/2}], [-i e^{i phi/2}, 0]]
        phi = 2 * np.angle(1j * b) * -1
        seq = [("rx", th), ("rz", phi)]
        return tuple((g, t) for g, t in seq if abs(_wrap(t)) > 1e-9)
    # v = Rz(phi) Rx(th) Rz(lam); v00 = cos(th/2) e^{-i(phi+lam)/2}, v01 = -i sin(th/2) e^{-i(phi-lam)/2}
    th = 2 * math.atan2(abs(b), abs(a))
    s = -np.angle(a) * 2  # phi + lam
    d = -np.angle(1j * b) * 2  # phi - lam
    phi, lam = (s + d) / 2, (s - d) / 2
    seq = [("rz", lam), ("rx", th), ("rz", phi)]
    seq = [(g, t) for g, t in seq if abs(_wrap(t)) > 1e-9]
    # Rz(phi) Rx(th) Rz(lam) = Rz(phi - pi) Rx(-th) Rz(lam + pi): take the form with fewer gates
    alt = [("rz", lam + math.pi), ("rx", -th), ("rz", phi - math.pi)]
    alt = [(g, t) for g, t in alt if abs(_wrap(t)) > 1e-9]
    return tuple(alt if len(alt) < len(seq) else seq)


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
        self.hubq = -1
        self.met = set()  # qubits that had a cz with the hub (validator: every input pair must interact)

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

    def flush(self, q, force=False, lazy=False):
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
        if (self.LAZY or lazy) and not force and len(seq) >= 2 and seq[-1][0] == "rz":
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
            if a == self.hubq:
                self.met.add(b)
            elif b == self.hubq:
                self.met.add(a)

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


def hub_epoch_schedule(blocks, preds, succs, hub, policy, positions, depths):
    """Prepare the dependency frontier for one collinear hub wave at a time."""
    threshold, urgency, coverage = policy[1:]
    waves = []
    last = None
    for i, block in enumerate(blocks):
        if hub not in block.qubits():
            continue
        role = block.c == hub
        if role != last:
            waves.append([])
            last = role
        waves[-1].append(i)
    ancestors = []
    for i, dependencies in enumerate(preds):
        before = set(dependencies)
        for j in dependencies:
            before.update(ancestors[j])
        ancestors.append(before)
    needed = [len(p) for p in preds]
    ready = {i for i, count in enumerate(needed) if count == 0}
    done = set()
    while ready:
        wave = next((set(w) - done for w in waves if not set(w) <= done), set())
        available = wave & ready
        blocked = wave - available
        pressure = {}
        for i in blocked:
            for j in ancestors[i] & ready:
                pressure[j] = pressure.get(j, 0) + 1
        nonhub = ready - available
        local = [i for i in nonhub if hub not in blocks[i].qubits()]
        witnesses = {i for i in local if i in pressure}
        if available and (not local or len(available) >= threshold and
                          (not witnesses or len(available) >= coverage * len(wave))):
            selected = sorted(available)
            kind = "hub" if len(selected) > 1 else "round"
        else:
            pos = positions()
            depth = depths()
            mix = not witnesses or len(available) < threshold
            pool = local + (sorted(available) if mix else [])
            def priority(i):
                b = blocks[i]
                travel = _dist(pos[b.c], pos[b.t])
                crit = max(depth[b.c], depth[b.t]) - min(depth[b.c], depth[b.t])
                return travel + 0.15 * crit - urgency * pressure.get(i, 0) - 0.15 * len(succs[i]), i
            selected = []
            used = set()
            for i in sorted(pool, key=priority):
                qs = blocks[i].qubits()
                if used.isdisjoint(qs):
                    used.update(qs)
                    selected.append(i)
            kind = "round"
        if not selected:
            raise Illegal("empty hub wave")
        yield kind, [blocks[i] for i in selected]
        for i in selected:
            ready.remove(i)
            done.add(i)
            for j in succs[i]:
                needed[j] -= 1
                if needed[j] == 0:
                    ready.add(j)


def schedule_gen(blocks, preds, succs, hub, policy=("look", 2.0), posfn=None, mu=None, evaluator=None,
                 depthfn=None, hw=0.0):
    """Yields steps ("hub", [block...]) or ("round", [block...]); the caller may execute each step
    before the next one is chosen (`posfn()` gives the current positions for `mu`)."""
    m = len(blocks)
    npred = [len(p) for p in preds]
    ready = {i for i in range(m) if npred[i] == 0}
    steps = []
    done = 0
    hubround = len(policy) > 2 and policy[2]
    HW = hw  # weight of the remaining critical path (blocks to the last successor) in the matching key
    height = [0] * m
    if HW:
        for i in range(m - 1, -1, -1):
            height[i] = 1 + max((height[j] for j in succs[i]), default=0)
    unl = lambda i: len(succs[i]) + HW * height[i]
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
                key = lambda i: (-unl(i), i)
            else:
                pos = posfn()
                key = lambda i: (_dist(pos[blocks[i].c], pos[blocks[i].t]) - mu * unl(i), i)
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
                    key2 = lambda i: (_dist(pos[blocks[i].c], pos[blocks[i].t]) - mu2 * unl(i), i)
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
        self.m.hubq = hub
        self.nxt = None

    # helpers
    def free(self, s):
        return s not in self.m.occ

    def go(self, q, dst):
        self.m.move({q: dst})

    def prep(self, q, u, lazy=False):
        """Queue `u` before q's next cz; flush if non-diagonal (q must be in storage)."""
        self.m.apply1(q, u)
        self.m.flush(q, lazy=lazy)

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

    def nearest_free(self, s, avoid=(), rows=range(10), kappa=0.0, y0=6, floor=0, kfloor=0.0, pull=None, wpull=0.0):
        best = None
        for y in rows:
            for x in range(10):
                t = (x, y)
                if t in self.m.occ or t in avoid:
                    continue
                d = _dist(s, t) + kappa * max(0, y - y0) + kfloor * max(0, floor - y)
                if pull is not None:
                    d += wpull * _dist(t, pull)
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

    EVPULL = 0.0
    HW = 0.0
    nxt_all = None

    def _evpull(self, q, targets):
        """Where an evicted atom is needed next: the (planned) site of its next partner."""
        if not self.EVPULL or not self.nxt_all:
            return None
        o = self.nxt_all.get(q)
        if o is None or o == self.hub:
            return None
        return targets.get(o, self.m.pos[o])

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

    def _round(self, blks, place_only=False):
        from scipy.optimize import linear_sum_assignment

        m = self.m
        P = len(blks)
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
                    e._exec_round(blks, members, plan, place_only)
                except Illegal:
                    continue
                c = e.m.dist - m.dist + DEPTH_W * (max(e.m.depth) - max(m.depth))
                if place_only:  # the band lifts are not simulated: add their estimate
                    c += 2 * self.NPULSE * math.hypot(10 - plan[0], 1.0 if plan[4] else 0.7)
                if best is None or c < best[0] - 1e-9:
                    best = (c, plan)
            plan = best[1] if best is not None else plans[0][2]
            rc = best[0] if best is not None else None
        else:
            plan = plans[0][2]
            rc = None
        if self.GATHER and P <= self.GATHER_MAX and not place_only:
            g = self.clone()
            g.RECT_SIM = 0
            try:
                g._exec_gather(blks, members)
                gc = g.m.dist - m.dist + DEPTH_W * (max(g.m.depth) - max(m.depth))
            except Illegal:
                gc = None
            if gc is not None:
                if rc is None:
                    e = self.clone()
                    e.RECT_SIM = 0
                    try:
                        e._exec_round(blks, members, plan)
                        rc = e.m.dist - m.dist + DEPTH_W * (max(e.m.depth) - max(m.depth))
                    except Illegal:
                        rc = float("inf")
                if gc < rc - 1e-9:
                    self._exec_gather(blks, members)
                    return
        self._exec_round(blks, members, plan, place_only)

    GATHER = True
    GATHER_MAX = 5

    def _gather_plan(self, blks, members):
        """Pairs straight from their storage rows into zone row 10 with one move per ghost-free row run
        (monotone column maps), then a rigid drop onto a free storage band.  Returns
        (cost, runs [(mapping)], zone site per atom, drop site per atom) or raises Illegal."""
        m = self.m
        pos, occ = m.pos, m.occ
        P = len(blks)
        pair_of = {}
        for i, b in enumerate(blks):
            pair_of[b.c] = pair_of[b.t] = i
        rows = {}
        for q in members:
            if pos[q][1] >= 10:
                raise Illegal("member in zone")
            rows.setdefault(pos[q][1], []).append(q)
        before = [set() for _ in range(P)]  # before[j]: pairs that must sit left of pair j
        inner = {}
        runs = []
        for y, qs in rows.items():
            qs.sort(key=lambda q: pos[q][0])
            run = [qs[0]]
            for u, v in itertools.pairwise(qs):
                pu, pv = pair_of[u], pair_of[v]
                if pu == pv:
                    inner[pu] = (u, v)
                else:
                    before[pv].add(pu)
                if any((x, y) in occ for x in range(pos[u][0] + 1, pos[v][0])):
                    runs.append(run)
                    run = []
                run.append(v)
            runs.append(run)
        # transitive order of the pairs (Kahn, ties by mean column)
        meanx = [(pos[b.c][0] + pos[b.t][0]) / 2 for b in blks]
        order, placed = [], set()
        while len(order) < P:
            free = [j for j in range(P) if j not in placed and before[j] <= placed]
            if not free:
                raise Illegal("gather: cyclic pair order")
            j = min(free, key=lambda j: (meanx[j], j))
            order.append(j)
            placed.add(j)
        # a pair that is not adjacent in a shared row sequence breaks monotonicity
        for y, qs in rows.items():
            seq = [order.index(pair_of[q]) for q in qs]
            if any(a > b for a, b in itertools.pairwise(seq)):
                raise Illegal("gather: order")
            for k in range(len(seq) - 2):
                if seq[k] == seq[k + 2]:
                    raise Illegal("gather: order")
        lr = []
        for j in order:
            b = blks[j]
            if j in inner:
                lr.append(inner[j])
            else:
                u, v = (b.c, b.t) if (pos[b.c][0], -pos[b.c][1]) <= (pos[b.t][0], -pos[b.t][1]) else (b.t, b.c)
                lr.append((u, v))
        best = None
        memsites = {pos[q] for q in members}
        for sub in itertools.combinations(range(5), P):
            zs = {}
            for k, j in enumerate(sub):
                zs[lr[k][0]] = (2 * j, 10)
                zs[lr[k][1]] = (2 * j + 1, 10)
            gcost = sum(max(_dist(pos[q], zs[q]) for q in run) for run in runs)
            if best is not None and gcost >= best[0]:
                continue
            for par in (0, 1):
                if par and sub[0] == 0:
                    continue
                for sh in (0, 1, 2):
                    ok = True
                    for q, (zx, _) in zs.items():
                        t = (zx - par, 9 - sh)
                        if t in occ and t not in memsites:
                            ok = False
                            break
                    if not ok:
                        continue
                    c = gcost + (2 * self.NPULSE - 1) * math.hypot(1 + sh, 1.0 if par else 0.7)
                    if best is None or c < best[0]:
                        best = (c, zs, par, sh)
        if best is None:
            raise Illegal("gather: no free drop band")
        c, zs, par, sh = best
        down = {q: (zx - par, 9 - sh) for q, (zx, _) in zs.items()}
        return c, [{q: zs[q] for q in run} for run in runs], zs, down

    def _exec_gather(self, blks, members):
        m = self.m
        _, runs, zs, down = self._gather_plan(blks, members)
        pairs = [(b.c, b.t) if zs[b.c][0] < zs[b.t][0] else (b.t, b.c) for b in blks]
        up = dict(zs)
        if self.NPULSE == 1:
            for b in blks:
                if b.kind == "C":
                    ss = _S if b.s > 0 else _SD
                    m.apply1(b.c, ss)
                    m.apply1(b.t, ss @ _H)
            for q in members:
                m.flush(q)
            for run in sorted(runs, key=lambda r: -m.pos[next(iter(r))][1]):
                m.move(run)
            m.pulse(pairs, ())
            m.move(down)
            for b in blks:
                if b.kind == "C":
                    m.apply1(b.t, _H)
            return
        forms = {}
        F = m.frame
        for b in blks:
            forms[b.idx] = f = self.choose_form(b)
            if f == "N":
                m.flush(b.c)
                self.prep(b.t, F["Nt"][0])
            else:
                self.prep(b.c, F["Sc"][0])
                self.prep(b.t, F["St"][0])
        for run in sorted(runs, key=lambda r: -m.pos[next(iter(r))][1]):
            m.move(run)
        m.pulse(pairs, ())
        m.move(down)
        for b in blks:
            r = b.t if forms[b.idx] == "N" else b.c
            m.apply1(r, _rx(b.th))
            m.flush(r)
        m.move(up)
        m.pulse(pairs, ())
        m.move(down)
        for b in blks:
            if forms[b.idx] == "N":
                m.apply1(b.t, F["Nt"][1])
            else:
                m.apply1(b.c, F["Sc"][1])
                m.apply1(b.t, F["St"][1])

    def _exec_round(self, blks, members, plan, place_only=False):
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
                                  kappa=self.KAPPA, y0=self.Y0, floor=self.floor_row(), kfloor=self.KFLOOR,
                                  pull=self._evpull(q, targets), wpull=self.EVPULL)
            reserved.add(t)
            targets[q] = t
        self.relocate(targets)
        if place_only:
            return
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
    HUBN = True  # hub-as-target blocks in rounds may use the hub as the rotor (no basis change on the hub)
    HUBLAZY = True  # the hub's trailing rz of a batch basis change stays pending through the batch
    KEEP_HUB_STATIC = True  # solo (b9a2a15): False lets every hub block choose its form freely
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
            if b.t == hub and not self.HUBN:
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
        m.hubq, m.met = src.hubq, set(src.met)
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
            hub = self.hub
            opts = [(strat, 0) for strat in self.HUB_STRATS]
            if self.RELAY and len(part) >= self.RELAY_MIN:
                opts += [(sv, lv) for lv in self.RELAY_OPTS for sv in self.RELAY_SERVE]
            for strat, lv in opts:
                e = self.clone()
                try:
                    e._hub(part, strat, lv)
                except Illegal:
                    continue
                dd = max(e.m.depth[hub] - m.depth[hub], max(e.m.depth) - max(m.depth))
                c = e.m.dist - m.dist + self.HUB_DW * DEPTH_W * dd
                if best is None or c < best[0] - 1e-9:
                    best = (c, strat, lv)
            if best is None:
                raise Illegal("no hub strategy")
            self._hub(part, best[1], best[2])

    RELAY = True
    RELAY_OPTS = (1, 2, -1)  # 1, 2: levels on the compact template; -1: level 1 placed like a band round
    RELAY_SERVE = ("rotate", "direct")
    RELAY_MIN = 4
    RELAY_TRIES = 60
    RELAY_ALTS = 1
    HUB_DW = 1.0

    def _relay_ok(self, u, v):
        return self.n <= 19 or frozenset((u, v)) in self.allowed

    # -- relay groups: quads (l0, r0, R, l1) = a path of allowed pairs, pairs (l, R), singles
    hub_left = None  # qubit -> hub blocks still to come after the current step

    def _covered(self, v):
        """v may be a relay leaf in every pass of this batch: its pair with the hub interacts elsewhere."""
        return self.n <= 19 or v in self.m.met or (self.hub_left or {}).get(v, 0) > 0

    def _relay_groups(self, pset, levels, passes=2):
        m = self.m
        ps = sorted(pset)
        cov = {u: self._covered(u) for u in ps}
        adj = {u: [v for v in ps if v != u and self._relay_ok(u, v)] for u in ps}
        rng = np.random.default_rng(12345 + len(ps))
        best = None
        for it in range(self.RELAY_TRIES):
            used = set()
            quads, pairs = [], []
            order = list(ps)
            if it:
                rng.shuffle(order)
            else:
                order.sort(key=lambda u: (len(adj[u]), u))
            if levels >= 2:
                for u in order:
                    if u in used:
                        continue
                    # grow a path of 4 through unused vertices, low-degree vertices first
                    path = [u]
                    used.add(u)
                    while len(path) < 4:
                        ext = None
                        for end in (path[-1], path[0]):
                            cand = [v for v in adj[end] if v not in used]
                            if cand:
                                v = min(cand, key=lambda v: (sum(1 for w in adj[v] if w not in used), _dist(m.pos[v], m.pos[end]), v))
                                ext = (end, v)
                                break
                        if ext is None:
                            break
                        end, v = ext
                        used.add(v)
                        if end == path[-1]:
                            path.append(v)
                        else:
                            path.insert(0, v)
                    if len(path) == 4 and (cov[path[0]] and cov[path[3]] and (passes == 2 or cov[path[1]] or cov[path[2]])):
                        if passes == 1 and not cov[path[1]]:
                            path.reverse()  # the uncovered inner atom becomes the root (index 2)
                        quads.append(tuple(path))
                    else:
                        for v in path:
                            used.discard(v)
                        used.add(u)  # do not retry from u in this pass
                used = {v for qd in quads for v in qd}
            rest = [u for u in order if u not in used]
            cands = sorted(((_dist(m.pos[u], m.pos[v]), u, v) for i, u in enumerate(rest) for v in rest[i + 1:]
                            if self._relay_ok(u, v) and (passes == 2 or cov[u] or cov[v])),
                           key=lambda t: (t[0], min(t[1], t[2]), max(t[1], t[2])))
            if it:
                cands.sort(key=lambda t: (len([w for w in adj[t[1]] if w not in used]) + len([w for w in adj[t[2]] if w not in used]), t[0]))
            for _, u, v in cands:
                if u in used or v in used:
                    continue
                used |= {u, v}
                pairs.append((u, v) if cov[u] else (v, u))  # (leaf, root): an uncovered atom is the root
            singles = [u for u in ps if u not in used]
            score = (len(quads) + len(pairs) + len(singles), len(singles))  # atoms the hub meets per pass
            if best is None or score < best[0]:
                best = (score, quads, pairs, singles)
                alts = []
            if score == best[0] and len(alts) < self.RELAY_ALTS:
                key = (tuple(sorted(tuple(sorted(qd)) for qd in quads)), tuple(sorted(tuple(sorted(pr)) for pr in pairs)))
                if all(k != key for k, _ in alts):
                    alts.append((key, (quads, pairs, singles)))
        return [g for _, g in alts]

    def _relay_template(self, nq, npair, ns, extra=0):
        """Row templates from row 9 upwards: [(kind, row, start col)], kinds "Q" (4 sites), "P" (2), "S" (1).
        Quads and pairs start at even columns; singles fill the right end of the rows."""
        rows = []
        y = 9
        items = ["Q"] * nq + ["P"] * npair
        k = 0
        out = []
        free_sites = []
        while k < len(items):
            x = 0
            while k < len(items) and x + (4 if items[k] == "Q" else 2) <= 10:
                w = 4 if items[k] == "Q" else 2
                out.append((items[k], y, x))
                x += w
                k += 1
            free_sites += [(c, y) for c in range(x, 10)]
            y -= 1
            if y < 1:
                raise Illegal("relay: template does not fit")
        need = ns + extra
        while len(free_sites) < need:
            free_sites += [(c, y) for c in range(10)]
            y -= 1
            if y < 1:
                raise Illegal("relay: template does not fit")
        free_sites.sort(key=lambda t: (-t[1], -t[0]))
        return out, free_sites, y + 1  # y + 1: top band row

    def _relay_place_round(self, pset, passes=2):
        """Level 1 only: the relay pairs are brought together like the pairs of a band round (best
        rectangle by simulation); unpaired partners stay where they are."""
        if self.n <= 19:
            m = self.m
            ps = sorted(pset)
            cands = sorted((_dist(m.pos[u], m.pos[v]), u, v) for i, u in enumerate(ps) for v in ps[i + 1:])
            used, pairs = set(), []
            for _, u, v in cands:
                if u not in used and v not in used:
                    used |= {u, v}
                    pairs.append((u, v))
            singles = [u for u in ps if u not in used]
        else:
            _, pairs, singles = self._relay_groups(pset, 1, passes)[0]
        if len(pairs) < self.RELAY_MINP:
            raise Illegal("relay: too few pairs")
        blks = [Block(l, r, 0.0, -1 - k, "Z") for k, (l, r) in enumerate(pairs)]
        saved = self.NPULSE, self.DUMMY_OFF, self.nxt, self.SEED
        self.NPULSE, self.DUMMY_OFF, self.nxt, self.SEED = 4 * passes // 2, True, {}, None
        try:
            self._round(blks, place_only=True)
        finally:
            self.NPULSE, self.DUMMY_OFF, self.nxt, self.SEED = saved
        return [], pairs, singles, None

    def _relay_place(self, pset, levels, passes=2):
        """Choose groups and relocate the batch partners into the relay band.  Returns
        (quads [(l0, r0, R, l1)], pairs [(l, R)], singles, top row) with every group on adjacent sites."""
        from scipy.optimize import linear_sum_assignment

        m, hub = self.m, self.hub
        pos = m.pos
        K = len(pset)
        if self.n <= 19:
            nq = K // 4 if levels >= 2 else 0
            rem = K - 4 * nq
            # a row holds two quads and one pair
            if levels >= 2:
                nq = min(nq, 2 * math.ceil(K / 10))
                rem = K - 4 * nq
            npair = rem // 2
            ns = rem - 2 * npair
            tmpl, free_sites, top = self._relay_template(nq, npair, ns)
            sites = []
            for kind, y, x in tmpl:
                sites += [(x + i, y) for i in range(4 if kind == "Q" else 2)]
            sites += free_sites[:ns]
            free_used = set(free_sites[:ns])
            ps = sorted(pset)
            cost = np.array([[_dist(pos[q], st) for st in sites] for q in ps])
            ri, ci = linear_sum_assignment(cost)
            at = {sites[k]: ps[i] for i, k in zip(ri, ci)}
            targets = {q: st for st, q in at.items()}
            quads, pairs, singles = [], [], [at[st] for st in free_sites[:ns]]
            for kind, y, x in tmpl:
                if kind == "Q":
                    quads.append((at[(x, y)], at[(x + 1, y)], at[(x + 2, y)], at[(x + 3, y)]))
                else:
                    pairs.append((at[(x + 1, y)], at[(x, y)]))
        else:
            best = None
            for quads, pairs, singles in self._relay_groups(pset, levels, passes):
                for mirror in (False, True):
                    tmpl, free_sites, top = self._relay_template(len(quads), len(pairs), len(singles))
                    if mirror:
                        tmpl = [(kind, y, (6 if kind == "Q" else 8) - x) for kind, y, x in tmpl]
                        free_sites = [(9 - x, y) for x, y in free_sites]
                    free_used = set(free_sites[:len(singles)])
                    targets = {}
                    est = 0.0
                    qs = [t for t in tmpl if t[0] == "Q"]
                    if quads:
                        # either orientation of a quad fits a slot
                        cost = np.zeros((len(quads), len(qs)))
                        flip = np.zeros((len(quads), len(qs)), dtype=bool)
                        for i, qd in enumerate(quads):
                            for k, (_, y, x) in enumerate(qs):
                                f = sum(_dist(pos[qd[j]], (x + j, y)) for j in range(4))
                                r = sum(_dist(pos[qd[3 - j]], (x + j, y)) for j in range(4))
                                cost[i, k], flip[i, k] = min(f, r), r < f
                        ri, ci = linear_sum_assignment(cost)
                        est += float(cost[ri, ci].sum())
                        for i, k in zip(ri, ci):
                            _, y, x = qs[k]
                            for j in range(4):
                                targets[quads[i][j]] = (x + (3 - j if flip[i, k] else j), y)
                    psl = [t for t in tmpl if t[0] == "P"]
                    if pairs:
                        cost = np.zeros((len(pairs), len(psl)))
                        flip = np.zeros((len(pairs), len(psl)), dtype=bool)
                        for i, (u, v) in enumerate(pairs):
                            for k, (_, y, x) in enumerate(psl):
                                f = _dist(pos[u], (x, y)) + _dist(pos[v], (x + 1, y))
                                r = _dist(pos[v], (x, y)) + _dist(pos[u], (x + 1, y))
                                cost[i, k], flip[i, k] = min(f, r), r < f
                        ri, ci = linear_sum_assignment(cost)
                        est += float(cost[ri, ci].sum())
                        for i, k in zip(ri, ci):
                            u, v = pairs[i][::-1] if flip[i, k] else pairs[i]
                            _, y, x = psl[k]
                            targets[u], targets[v] = (x, y), (x + 1, y)
                    if singles:
                        fs = free_sites[:len(singles)]
                        cost = np.array([[_dist(pos[q], st) for st in fs] for q in singles])
                        ri, ci = linear_sum_assignment(cost)
                        est += float(cost[ri, ci].sum())
                        for i, k in zip(ri, ci):
                            targets[singles[i]] = fs[k]
                    # atoms that have to make room
                    tset = set(targets.values())
                    est += self.EVICT * sum(1 for st in tset if st in m.occ and m.occ[st] not in pset)
                    if best is None or est < best[0] - 1e-9:
                        best = (est, quads, pairs, singles, targets, free_used, top)
            _, quads, pairs, singles, targets, free_used, top = best
        # other atoms only have to leave the target sites and the rotation buffer row above the band;
        # they may wait on free band sites right of the groups (outside every lifted run)
        reserved = set(targets.values())
        band = {(x, y) for y in range(top, 10) for x in range(10)}
        buf = {(x, top - 1) for x in range(10)}
        movers = set(targets)
        for q in sorted((q for q, st in pos.items() if q not in pset and (st in reserved or st in buf)),
                        key=lambda q: (q != hub, q)):
            cand = [st for st in band if st not in reserved and (st not in m.occ or m.occ[st] in movers)]
            if cand:
                t = min(cand, key=lambda st: (_dist(pos[q], st) + (0.5 * (9 - st[1]) if q == hub else 0.0), st))
            else:
                t = self.nearest_free(pos[q], avoid=reserved | band | buf, rows=range(0, max(1, top - 1)))
            reserved.add(t)
            targets[q] = t
            movers.add(q)
        for q, st in pos.items():
            if q not in targets and q not in pset and st[1] >= top:
                # stays inside the band rows: it must sit right of every group of its row
                gx = max((t[0] for t in targets.values() if t[1] == st[1] and t not in free_used), default=-1)
                if st[0] < gx:
                    cand = [t for t in band if t not in reserved and (t not in m.occ or m.occ[t] in movers)
                            and t[0] > max((u[0] for u in targets.values() if u[1] == t[1] and u not in free_used), default=-1)]
                    if cand:
                        t = min(cand, key=lambda t: (_dist(st, t), t))
                    else:
                        t = self.nearest_free(st, avoid=reserved | band | buf, rows=range(0, max(1, top - 1)))
                    reserved.add(t)
                    targets[q] = t
                    movers.add(q)
        self.relocate(targets)
        return quads, pairs, singles, top

    def _rigid(self, mapping):
        """Lift/drop a set of atoms with few rectangle moves: per row maximal ghost-free runs, equal
        runs of consecutive rows merged into one rectangle."""
        m = self.m
        todo = {q: d for q, d in mapping.items() if m.pos[q] != d}
        if not todo:
            return
        if m.can_move(todo):
            m.move(todo)
            return
        rows = {}
        for q in todo:
            rows.setdefault(m.pos[q][1], []).append(q)
        runs = []  # (y, x0, x1, [atoms])
        for y, qs in rows.items():
            qs.sort(key=lambda q: m.pos[q][0])
            run = [qs[0]]
            for u, v in itertools.pairwise(qs):
                if any((x, y) in m.occ and m.occ[(x, y)] not in todo for x in range(m.pos[u][0] + 1, m.pos[v][0])) \
                        or todo[u][1] != todo[v][1]:
                    runs.append((y, m.pos[run[0]][0], m.pos[run[-1]][0], run))
                    run = []
                run.append(v)
            runs.append((y, m.pos[run[0]][0], m.pos[run[-1]][0], run))
        groups = {}
        for y, x0, x1, run in runs:
            groups.setdefault((x0, x1), []).append((y, run))
        for key in sorted(groups):
            lst = sorted(groups[key])
            i = 0
            while i < len(lst):
                j = i
                sel = {q: todo[q] for q in lst[i][1]}
                while j + 1 < len(lst) and lst[j + 1][0] == lst[j][0] + 1:
                    trial = dict(sel)
                    trial.update({q: todo[q] for q in lst[j + 1][1]})
                    if not m.can_move(trial):
                        break
                    sel = trial
                    j += 1
                if m.can_move(sel):
                    m.move(sel)
                else:
                    self.lift(sel)
                i = j + 1

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

    def _serve_at(self, p, hs):
        m, hub = self.m, self.hub
        ps = (hs[0] ^ 1, 10)
        if m.pos[hub] != hs:
            if hs in m.occ:
                raise Illegal("hub site busy")
            self.go(hub, hs)
        back = m.pos[p]
        self.go(p, ps)
        m.pulse([(hub, p)] if hs[0] < ps[0] else [(p, hub)])
        self.go(p, back)

    def _serve_plan(self, row):
        """Hub sites for the partners of storage row 9 (dynamic programme over the ten hub sites and
        both sweep directions): [(partner, hub site)]."""
        m, hub = self.m, self.hub
        best = None
        hp = m.pos[hub]
        for seq in (sorted(row, key=lambda p: m.pos[p][0]), sorted(row, key=lambda p: -m.pos[p][0])):
            cost = [0.0 if hp[1] < 10 else _dist(hp, (h, 10)) for h in range(10)]
            back = []
            for i, p in enumerate(seq):
                sv = [2 * _dist(m.pos[p], (h ^ 1, 10)) for h in range(10)]
                if i == 0:
                    cost = [cost[h] + sv[h] for h in range(10)]
                    back.append(list(range(10)))
                    continue
                new, bk = [], []
                for h in range(10):
                    g = min(range(10), key=lambda g: cost[g] + _dist((g, 10), (h, 10)))
                    new.append(cost[g] + _dist((g, 10), (h, 10)) + sv[h])
                    bk.append(g)
                cost = new
                back.append(bk)
            h = min(range(10), key=lambda h: cost[h])
            tot = cost[h]
            sites = [h]
            for i in range(len(seq) - 1, 0, -1):
                h = back[i][h]
                sites.append(h)
            sites.reverse()
            if best is None or tot < best[0] - 1e-9:
                best = (tot, [(p, (hh, 10)) for p, hh in zip(seq, sites)])
        return best[1]

    SERVE_DP = True

    def _hub(self, blks, strat, relay=0):
        """Hub batch.  Core: F . prod rx_p(th_p) . F with the fan-out F = prod_p cz(hub, p) (one F for
        single-pulse blocks).

        Relay (levels 1, 2): F only depends on the parity of the partners, so with
        P = prod over relay edges (child -> parent) of the controlled-Y  rx_p(-pi/2) cz(c, p) rx_p(pi/2),
        children first, F = P^dag . prod_roots cz(hub, root) . P exactly.  The relay edges of one level
        are disjoint pairs pulsed together in a band; the hub only meets the roots."""
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
            # with relays the partners' buffers are flushed lazily where they are needed
            pp = m.apply1 if relay else self.prep
            if role == "Z":
                self.prep(hub, hs)
                for b in blks:
                    pp(b.t, (_S if b.s > 0 else _SD) @ _H)
                post_p, post_h = _H, None
            else:
                self.prep(hub, hs @ _H, lazy=self.HUBLAZY)
                for b in blks:
                    pp(b.c, _S if b.s > 0 else _SD)
                post_p, post_h = None, _H
        else:
            if role == "Y":
                self.prep(hub, m.frame["St"][0], lazy=self.HUBLAZY)
            else:
                m.flush(hub)
            pre_p = m.frame["Nt"][0] if role == "Z" else m.frame["Sc"][0]
            post_p = m.frame["Nt"][1] if role == "Z" else m.frame["Sc"][1]
            post_h = m.frame["St"][1] if role == "Y" else None
            for p, _ in partners:
                (m.apply1 if relay else self.prep)(p, pre_p)
        pset = {p for p, _ in partners}
        top = None
        plan = {0: ([], set(pset)), 1: ([], set(pset))}  # pass -> (levels [[(child, parent)]], atoms the hub meets)
        if relay:
            if relay == -1:
                quads, pairs, singles, top = self._relay_place_round(pset, 1 if single else 2)
            else:
                quads, pairs, singles, top = self._relay_place(pset, relay, 1 if single else 2)
            if 2 * len(quads) + len(pairs) < self.RELAY_MINP:
                raise Illegal("relay: too few pairs")
            for alt in (0, 1):
                # second pass: the other atom of each pair / the other inner atom of each quad is the root
                l1 = [(a, b) for a, b, c, d in quads] + [(d, c) for a, b, c, d in quads]
                l1 += [(r, l) if alt else (l, r) for l, r in pairs]
                l2 = [(c, b) if alt else (b, c) for a, b, c, d in quads]
                roots = {(b if alt else c) for a, b, c, d in quads} | {(l if alt else r) for l, r in pairs}
                plan[alt] = ([l1] + ([l2] if l2 else []), roots | set(singles))
        else:
            if strat == "compact":
                self._compact(pset)
                strat = "rotate"
            elif strat == "gravity":
                self._gravity()
                strat = "rotate"
        hub_home = m.pos[hub]
        VR, VRD = _rx(math.pi / 2), _rx(-math.pi / 2)

        vmap = {}

        def band(levels, lv, compute):
            edges = levels[lv]
            if m.pos[hub][1] >= 10:
                moving = {q for e in edges for q in e}
                rowsb = {m.pos[q][1] for q in moving}
                cand = []
                for y in range(10):
                    for x in range(10):
                        st = (x, y)
                        if st in m.occ:
                            continue
                        if y in rowsb:
                            xs = [m.pos[q][0] for q in moving if m.pos[q][1] == y]
                            if min(xs) < x < max(xs):
                                continue
                        cand.append((_dist(m.pos[hub], st), st))
                if not cand:
                    raise Illegal("relay: no site for the hub")
                self.go(hub, min(cand)[1])
            for c, p in edges:
                m.flush(c, lazy=True)
                if compute:
                    # V = rx(pi/2) D^-1 with D the trailing diagonal of p's buffer: V^dag Z V stays in
                    # the XY plane (a controlled-W, W anticommuting with Z), and D never has to be emitted
                    sq = _zxz(m.pend[p])
                    dg = _rz(sq[-1][1]) if sq and sq[-1][0] == "rz" else _I2
                    vmap[(c, p)] = (VR @ dg.conj().T, dg @ VRD)
                self.prep(p, vmap[(c, p)][0])
            # zone sites: storage (x, y) -> (x + par, y + h); level 1 pairs on (even, odd), level 2 on (odd, even)
            par = lv % 2
            ys = [m.pos[q][1] for e in edges for q in e]
            h = 10 - min(ys)
            up = {q: (m.pos[q][0] + par, m.pos[q][1] + h) for e in edges for q in e}
            prs = [(c, p) if m.pos[c][0] < m.pos[p][0] else (p, c) for c, p in edges]
            dummy = []
            if par:
                # two leaves between neighbouring quads may ride along as a dummy pair (cz . cz = I): the
                # row then lifts as one run (DUMMY_ALL: only if every such gap of the band can be closed)
                inband = set(up)
                gaps, extra = 0, {}
                for (cq, pq) in edges:
                    x, y = max(m.pos[cq][0], m.pos[pq][0]), m.pos[cq][1]
                    if m.occ.get((x + 3, y)) not in inband:
                        continue
                    gaps += 1
                    a, b = m.occ.get((x + 1, y)), m.occ.get((x + 2, y))
                    if a is None or b is None or a in inband or b in inband:
                        continue
                    if not self._relay_ok(a, b) or not (_is_diag(m.pend[a]) and _is_diag(m.pend[b])):
                        continue
                    extra[a] = (x + 1 + par, y + h)
                    extra[b] = (x + 2 + par, y + h)
                    dummy.append((a, b))
                if dummy and (len(dummy) == gaps or not self.DUMMY_ALL):
                    up.update(extra)
                else:
                    dummy = []
            down = {q: m.pos[q] for q in up}
            self._rigid(up)
            m.pulse(prs + dummy, {q for d in dummy for q in d})
            self._rigid(down)
            for c, p in edges:
                m.apply1(p, vmap[(c, p)][1])

        if strat == "direct":
            memo = []

            def sweep(todo, forward):
                if forward or {p for p, _ in memo} != set(todo):
                    del memo[:]
                    rows = sorted({m.pos[p][1] for p in todo}, reverse=True)
                    for y in rows:
                        row = [p for p in todo if m.pos[p][1] == y]
                        if y == 9 and self.SERVE_DP:
                            plan = self._serve_plan(row)
                        else:
                            row.sort(key=lambda p: m.pos[p][0])
                            plan = [(p, self._hub_site(m.pos[p][0])[0]) for p in row]
                        for p, hs in plan:
                            self._serve_at(p, hs)
                            memo.append((p, hs))
                else:
                    for p, hs in memo[::-1]:
                        self._serve_at(p, hs)
        else:
            if top is not None:
                rt = top - 1
                if any(st[1] == rt for q, st in m.pos.items() if q != hub):
                    raise Illegal("relay: rotation row not free")
            else:
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
                    plan = None
                    if not forward:
                        k = len(visits) - 1 - step
                        if 0 <= k < len(visits) and {p for p, _ in visits[k]} == set(row):
                            plan = visits[k][::-1]
                    if plan is None:
                        if self.SERVE_DP and row:
                            plan = self._serve_plan(row)
                        else:
                            row.sort(key=lambda p: (m.pos[p][0] // 2, -(m.pos[p][0] % 2)))
                            if step % 2:
                                row.reverse()
                            plan = [(p, self._hub_site(m.pos[p][0])[0]) for p in row]
                    if forward:
                        visits.append(plan)
                    for p, hs in plan:
                        self._serve_at(p, hs)
                        todo.discard(p)
                    if not todo:
                        return
                    step += 1
                    if step > cap:
                        raise Illegal("rotation did not reach every partner")
                    self._rotate(rt, forward)

        def fan(k):
            levels, serve = plan[k]
            vmap.clear()
            for lv in range(len(levels)):
                band(levels, lv, True)
            for q in serve:
                m.flush(q, lazy=bool(levels))
            sweep(set(serve), k == 0)
            for lv in range(len(levels) - 1, -1, -1):
                band(levels, lv, False)

        fan(0)
        if not single:
            nxt_par = {p for _, p in plan[1][0][0]} if plan[1][0] else set()
            for p, th in partners:
                m.apply1(p, _rx(th))
                if p not in nxt_par:
                    m.flush(p)
            fan(1)
        if m.pos[hub][1] >= 10:
            if hub_home in m.occ:
                hub_home = self.nearest_free(m.pos[hub])
            self.go(hub, hub_home)
        if post_p is not None:
            for p, _ in partners:
                m.apply1(p, post_p)
        if post_h is not None:
            m.apply1(hub, post_h)

    RELAY_MINP = 2
    DUMMY_ALL = False


def default_layout(n, hub):
    R = max(1, math.ceil(n / 10))
    home = {}
    qs = [q for q in range(n) if q != hub] + [hub]
    for i, q in enumerate(qs):
        r = 9 - i // 10
        c = i % 10 if (i // 10) % 2 == 0 else 9 - i % 10
        home[q] = (c, r)
    return home


def sa_layout(n, hub, blocks, seed=0, iters=None, wy=1.0, rows=None):
    """Locality layout: simulated annealing on the summed partner distance of all non-hub blocks."""
    rng = np.random.default_rng(seed)
    R = rows or max(1, math.ceil(n / 10))
    sites = [(x, 9 - r) for r in range(R) for x in range(10)]
    w = {}
    for b in blocks:
        if hub in (b.c, b.t):
            continue
        k = (min(b.c, b.t), max(b.c, b.t))
        w[k] = w.get(k, 0) + 1
    nbr = {q: [] for q in range(n)}
    for (a, c), v in w.items():
        nbr[a].append((c, v))
        nbr[c].append((a, v))
    base = default_layout(n, hub)
    at = {s: None for s in sites}
    pos = {}
    for q, st in base.items():
        if st not in at:
            st = next(t for t in sites if at[t] is None)
        at[st] = q
        pos[q] = st

    def cost_q(q, st):
        tot = 0.0
        for o, v in nbr[q]:
            ox, oy = pos[o]
            tot += v * (abs(ox - st[0]) + wy * abs(oy - st[1]))
        if q == hub:
            tot += 0.0
        return tot

    iters = iters or 4000 * n
    T0, T1 = 2.0, 0.05
    ns = len(sites)
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        s1, s2 = sites[int(rng.integers(ns))], sites[int(rng.integers(ns))]
        q1, q2 = at[s1], at[s2]
        if q1 == q2:
            continue
        before = (cost_q(q1, s1) if q1 is not None else 0) + (cost_q(q2, s2) if q2 is not None else 0)
        if q1 is not None:
            pos[q1] = s2
        if q2 is not None:
            pos[q2] = s1
        after = (cost_q(q1, s2) if q1 is not None else 0) + (cost_q(q2, s1) if q2 is not None else 0)
        d = after - before
        if d <= 0 or rng.random() < math.exp(-d / T):
            at[s1], at[s2] = q2, q1
        else:
            if q1 is not None:
                pos[q1] = s1
            if q2 is not None:
                pos[q2] = s2
    return dict(pos)


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


LAST = {}


def _true_depth(qc):
    """Depth with moves and barriers removed (the scored quantity)."""
    idx = {b: i for i, b in enumerate(qc.qubits)}
    d = [0] * qc.num_qubits
    for inst in qc.data:
        if inst.operation.name in ("move", "barrier"):
            continue
        qs = [idx[q] for q in inst.qubits]
        k = max(d[q] for q in qs) + 1
        for q in qs:
            d[q] = k
    return max(d)


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
    hub_left = {}
    for b in blocks:
        if hub in (b.c, b.t) and b.kind != "P":
            o = b.t if b.c == hub else b.c
            hub_left[o] = hub_left.get(o, 0) + 1
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
    if pol[0] == "wave":
        sequence = hub_epoch_schedule(blocks, preds, succs, hub, pol, lambda: ex.m.pos, lambda: ex.m.depth)
    else:
        sequence = schedule_gen(blocks, preds, succs, hub, pol, posfn=lambda: ex.m.pos, mu=mu, evaluator=evaluator,
                                depthfn=(lambda: ex.m.depth) if ex.HUBDEP else None, hw=ex.HW)
    for kind, blks in sequence:
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
        if ex.EVPULL:
            na = {}
            for q, lst in byq.items():
                k = ptr[q]
                while k < len(lst) and (lst[k].idx in done or lst[k].idx in cur):
                    k += 1
                ptr[q] = k if not any(lst[j].idx in cur for j in range(ptr[q], k)) else ptr[q]
                if k < len(lst):
                    o = lst[k]
                    na[q] = o.t if o.c == q else o.c
            ex.nxt_all = na
        ex.steps.append((kind, blks))
        for b in blks:
            if hub in (b.c, b.t) and b.kind != "P":
                hub_left[b.t if b.c == hub else b.c] -= 1
        ex.hub_left = hub_left
        (ex.run_round_band if kind == "round" else ex.run_hub)(blks)
        done |= cur
    ex.m.finish()
    return ex


BASE_POLS = (("look", 2.0, True, 8), ("look", 2.0, True, 20), ("look", 2.0, True, 30), ("look", 2.0, True, 40),
             ("look", 2.0, True, 48), ("thresh", 20, True), ("rounds",), ("thresh", 45))
_L8, _L20, _L40 = ("look", 2.0, True, 8), ("look", 2.0, True, 20), ("look", 2.0, True, 40)
_L7, _L2 = ("look", 2.0, True, 7), ("look", 2.0, True, 2)
# chosen by a greedy set cover over the 12 non-warm scored iHVA circuits (+ their individual winners)
CANDS = [(_L8, 0.0, 1.0, (("PARS", (0, 1)),)), (_L40, 0.0, 1.0, (("PARS", (0,)),)), (_L8, 0.35, 1.0, (("PARS", (0,)),)),
         (_L40, 0.35, 1.0, (("PARS", (0, 1)),)), (_L8, 0.0, None, (("PARS", (0, 1)),)), (_L8, 0.0, 1.0, (("PARS", (0,)),)),
         (_L40, 0.0, 1.0, (("PARS", (0, 1)),)), (_L8, 0.35, 1.0, (("PARS", (0, 1)),)),
         (_L20, 0.35, 1.0, (("PARS", (0, 1)),)), (_L20, 0.0, 1.0, (("PARS", (0,)),)), (_L20, 0.35, 1.0, (("PARS", (0,)),)),
         (("rounds",), 0.0, None, (("PARS", (0, 1)),)), (("thresh", 45), 0.0, None, (("PARS", (0, 1)),)),
         # small hub groups as (relayed) batches
         (_L7, 0.0, 1.0, (("PARS", (0, 1)),)), (_L7, 0.35, 1.0, (("PARS", (0,)),)), (_L2, 0.0, 1.0, (("PARS", (0, 1)),))]


VARIANTS = ((("HUBN", False),), (("HW", 2.0),), (("HW", 10.0),), (("EVPULL", 0.6),), (("HW", 2.0), ("EVPULL", 0.6)), (("EVPULL", 1.0),))
RELAY_VARIANTS = ((("RELAY", False),), (("RELAY_OPTS", (1, 2)),), (("RELAY_OPTS", (-1,)),), (("HUB_DW", 0.5),), (("HUB_DW", 2.0),),
                  (("RELAY_ALTS", 6),), (("DUMMY_ALL", True),))


def _budget(n, nblocks):
    """(top configs re-run with EVAL / RECT_SIM=1 / first-use layout, seeds for the best config)."""
    if nblocks <= 120:
        return 5, 16
    if nblocks <= 400:
        return 4, 10
    return 3, 6


SEARCH_WALL = 180.0  # seconds; a safety net for slow machines, not reached with the fixed budgets


def compile_ihva(n, blocks, cands=None, emit=True, seeds=None):
    import time

    t0 = time.time()
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
    hubreq = {b.t if b.c == hub else b.c for b in blocks if hub in (b.c, b.t) and b.kind != "P"}
    schedules = []

    layouts = {None: home}
    seen = {}

    def run(cand, lay=None):
        key = (cand[0], cand[1], cand[2], tuple(sorted(cand[3], key=lambda kv: kv[0])), lay)
        if key in seen:
            return seen[key]
        if res and time.time() - t0 > SEARCH_WALL:
            return None  # safety net only (slow machine): keep what we have
        seen[key] = None
        try:
            ex = run_candidate(n, blocks, preds, succs, layouts[lay], *cand, allowed=allowed)
        except Illegal:
            return None
        if n > 19 and not hubreq <= ex.m.met:
            return None  # a hub pair of the input would not interact (validator pair check)
        c = ex.m.dist + DEPTH_W * (max(ex.m.depth) + 1)
        res.append((c, len(res), cand, lay))
        seen[key] = ex
        schedules.append(ex.steps)
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
    # diversity: critical-path matching keys, eviction towards the next partner, locality layout
    big = len(blocks) > 400  # fixed work budget: the 50q p=3 circuits get fewer variants
    if not big:
        layouts["sa"] = sa_layout(n, hub, blocks)
    for _, _, cand, lay in sorted(res)[:top]:
        pol, lam, mu, opts = cand
        if any(k in ("HW", "EVPULL", "HUBN") for k, _ in opts) or lay is not None:
            continue
        for extra in (VARIANTS[:4] if big else VARIANTS) + RELAY_VARIANTS:
            run((pol, lam, mu, tuple(opts) + extra), lay)
        if not big:
            run(cand, "sa")
    # first-use layouts derived from the best configurations' own step sequences
    for _, k, cand, lay in sorted(res)[:top]:
        ex = run(cand, lay)
        if ex is None:
            continue
        key = ("first", k)
        layouts[key] = layout_from_steps(n, hub, schedules[k])
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
    ordinary = sorted(res)[:4]
    experimental = len(res)
    if n >= 12 and any(b.kind == "G" for b in blocks):
        for threshold, urgency, coverage in ((3, 1.6, 1.01), (8, 0.6, 0.8), (16, 1.6, 0.8)):
            policy = "wave", threshold, urgency, coverage
            opts = (("KEEP_HUB_STATIC", False), ("FORMS", "N"), ("RECT_SIM", 2))
            run((policy, 0.2, None, opts))
    finalists = ordinary + sorted(res[experimental:])[:2]
    if not emit:
        return finalists[0][0], finalists[0][2]

    best = None
    for _, _, cand, lay in finalists:
        home = layouts[lay]
        ex = run_candidate(n, blocks, preds, succs, home, *cand, emit=True, allowed=allowed)
        qc = _finalize_phases(ex.m.qc)
        qc, home, distance = _fold_boundary_moves(qc, home, ex.m.dist)
        cost = distance + DEPTH_W * _scored_depth(qc)
        if best is None or cost < best[0] - 1e-9:
            best = cost, qc, home
    _, qc, home = best
    LAST["dist"], LAST["depth"] = best[0] - DEPTH_W * _scored_depth(qc), _true_depth(qc)
    # guard: the interaction set must stay inside the input's pairs (validator check above 19 qubits)
    idx = {bit: i for i, bit in enumerate(qc.qubits)}
    for inst in qc.data:
        if inst.operation.name == "cz":
            if n > 19 and frozenset(idx[q] for q in inst.qubits) not in allowed:
                raise Illegal("cz outside the input's interaction pairs")
    if n > 19 and not hubreq <= ex.m.met:
        raise Illegal("a hub pair of the input does not interact")
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

    candidates = (lower(paulis, late) for paulis, late in itertools.product((False, True), repeat=2))
    return min(itertools.chain((circuit,), candidates), key=_scored_depth)


def _scored_depth(circuit):
    indices = {q: i for i, q in enumerate(circuit.qubits)}
    levels = [0] * circuit.num_qubits
    for ins in circuit.data:
        if ins.operation.name in ("move", "barrier"):
            continue
        qs = [indices[q] for q in ins.qubits]
        level = max(levels[q] for q in qs) + 1
        for q in qs:
            levels[q] = level
    return max(levels, default=0)


def _place_z_phases(circuit):
    """Place each combined Z phase in a storage visit with maximal circuit slack."""
    n = circuit.num_qubits
    indices = {q: i for i, q in enumerate(circuit.qubits)}
    data = list(circuit.data)
    gates = [(ins.operation.name, tuple(indices[q] for q in ins.qubits)) for ins in data]
    timeline = [[] for _ in range(n)]
    storage = [[0] for _ in range(n)]
    prior = [-1] * n
    z = [0.0] * n
    phases = []
    for i, (name, qs) in enumerate(gates):
        if name == "move":
            if max(data[i].operation.params[2]) < 10:
                for q in qs:
                    storage[q].append(i + 1)
        elif name == "rz":
            q, = qs
            z[q] += float(data[i].operation.params[0])
        elif name == "cz":
            for q in qs:
                timeline[q].append(i)
        elif name in ("rx","measure"):
            q, = qs
            phase = _wrap(z[q])
            if abs(phase) > 1e-10:
                phases.append((q, phase, prior[q] + 1, i))
            z[q] = 0.0
            prior[q] = i
            timeline[q].append(i)

    def path_lengths(extra, reverse=False):
        lengths = {}
        levels = [0] * n
        order = reversed(range(len(gates))) if reverse else range(len(gates))
        for i in order:
            name, qs = gates[i]
            if not reverse:
                for q, _ in extra.get(i, ()):
                    levels[q] += 1
            if name in ("rx", "cz", "measure"):
                length = max(levels[q] for q in qs) + 1
                lengths[i] = length
                for q in qs:
                    levels[q] = length
            if reverse:
                for q, _ in extra.get(i, ()):
                    levels[q] += 1
        return max(levels, default=0), lengths

    _, prefix = path_lengths({})
    _, suffix = path_lengths({}, reverse=True)
    candidates = []
    for q, angle, start, end in phases:
        options = {}
        for slot in sorted({start, end} | {i for i in storage[q] if start <= i <= end}):
            j = bisect_left(timeline[q], slot)
            before = timeline[q][j - 1] if j else None
            after = timeline[q][j]
            options.setdefault((before, after), slot)
        candidates.append((q, angle, options))

    placements = []
    for direction in (1, -1):
        extra = {}
        chosen = []
        for q, angle, options in candidates:
            edge = min(options, key=lambda edge: (prefix.get(edge[0], 0) + suffix[edge[1]],
                                                   direction * options[edge]))
            slot = options[edge]
            extra.setdefault(slot, []).append((q, angle))
            chosen.append(slot)
        depth, _ = path_lengths(extra)
        placements.append((depth, extra, chosen))
    _, extra, chosen = min(placements, key=lambda trial: trial[0])

    # Refine the placement of flexible phases against the lengths of the
    # current circuit. Removing one phase makes its best insertion exact.
    flexible = [i for i, (_, _, options) in enumerate(candidates) if len(options) > 1]
    flexible.sort(key=lambda i: (-len(candidates[i][2]), i))
    for i in flexible[:96]:
        q, angle, options = candidates[i]
        slot = chosen[i]
        extra[slot].remove((q, angle))
        _, prefix = path_lengths(extra)
        _, suffix = path_lengths(extra, reverse=True)
        edge = min(options, key=lambda edge: (prefix.get(edge[0], 0) + suffix[edge[1]],
                                               abs(options[edge] - slot)))
        extra.setdefault(options[edge], []).append((q, angle))

    depth, _ = path_lengths(extra)
    if depth >= _scored_depth(circuit):
        return circuit
    out = circuit.copy_empty_like()
    for i, ins in enumerate(data):
        for q, theta in extra.get(i, ()):
            out.rz(theta, q)
        if ins.operation.name != "rz":
            out.append(ins)
    return out


def _fold_boundary_moves(circuit, home, original_distance):
    """Absorb storage setup into loading and stop moving atoms after their last use."""
    n = circuit.num_qubits
    indices = {q: i for i, q in enumerate(circuit.qubits)}
    reference = dict(home)
    moves = []
    first_lift = {}
    last_cz = {}
    last_rotation = {}
    for ins in circuit.data:
        name = ins.operation.name
        qs = [indices[q] for q in ins.qubits]
        step = len(moves)
        if name == "cz":
            last_cz.update(dict.fromkeys(qs, step))
        elif name in ("rx", "rz"):
            last_rotation[qs[0]] = step
        elif name == "move":
            _, _, new_rows, new_cols = ins.operation.params
            rows = dict(zip(sorted({reference[q][1] for q in qs}), new_rows, strict=True))
            cols = dict(zip(sorted({reference[q][0] for q in qs}), new_cols, strict=True))
            target = {q: (cols[reference[q][0]], rows[reference[q][1]]) for q in qs}
            for q, (_, y) in target.items():
                if y >= 10:
                    first_lift.setdefault(q, step)
            moves.append(target)
            reference.update(target)
    if not first_lift or not last_cz:
        return circuit, home, original_distance

    leading = min(first_lift.values())
    loading = dict(home)
    for target in moves[:leading]:
        loading.update(target)

    starts = {}
    stops = {}
    for step, target in enumerate(moves[leading:], leading):
        for q, site in target.items():
            if step < first_lift.get(q, 0):
                starts.setdefault(q, []).append((step + 1, site))
            if q in last_cz and step >= last_cz[q] and site[1] < 10:
                stops.setdefault(q, []).append(step + 1)
    last_pulse = max(last_cz.values())
    for q, options in stops.items():
        # After the final pulse an atom with only a measurement left may stay in the zone.
        if last_cz[q] == last_pulse and last_rotation.get(q, -1) < last_pulse:
            options.insert(0, last_pulse)
        options.pop()

    enter = {}
    leave = {}

    def selected(step):
        return {q: site for q, site in moves[step].items()
                if enter.get(q, leading) <= step < leave.get(q, len(moves))}

    def replay(end):
        m = Machine(n, loading, emit=False)
        for step in range(leading, end):
            m.move(selected(step))
        return m.dist

    if starts:
        end = max(first_lift.values()) + 1
        best = replay(end)
        for _ in range(2):
            for q, options in sorted(starts.items(), key=lambda item: (-len(item[1]), item[0])):
                prev = loading[q]
                cut = enter.get(q, leading)
                accepted = False
                for step, site in reversed(options):
                    if step <= cut:
                        break
                    if site != prev and site in loading.values():
                        continue
                    loading[q], enter[q] = site, step
                    try:
                        cost = replay(end)
                    except Illegal:
                        cost = math.inf
                    if cost < best - 1e-8:
                        best = cost
                        accepted = True
                        break
                    loading[q] = prev
                if not accepted:
                    loading[q] = prev
                    if cut == leading:
                        enter.pop(q, None)
                    else:
                        enter[q] = cut

    if any(stops.values()):
        best = replay(len(moves))
        for q, options in sorted(stops.items(), key=lambda item: (-len(item[1]), item[0])):
            for step in options:
                leave[q] = step
                try:
                    cost = replay(len(moves))
                except Illegal:
                    continue
                if cost < best - 1e-8:
                    best = cost
                    break
            else:
                leave.pop(q, None)

    if not leading and not enter and not leave:
        return circuit, home, original_distance

    out = circuit.copy_empty_like()
    m = Machine(n, loading, emit=False)
    step = 0
    for ins in circuit.data:
        if ins.operation.name != "move":
            out.append(ins)
            continue
        if step >= leading:
            target = selected(step)
            if target:
                rows, cols, new_rows, new_cols, distance = m._plan(target)
                if distance:
                    _move(out, sorted(target), rows, cols, new_rows, new_cols)
                    m.move(target)
        step += 1
    return out, loading, m.dist


def _fuse_local_gates(circuit):
    """Resynthesize independent single-qubit runs between CZs."""
    n = circuit.num_qubits
    indices = {q: i for i, q in enumerate(circuit.qubits)}
    runs = [[] for _ in range(n)]
    replacements = {}
    skip = set()

    def flush(q):
        run = runs[q]
        if len(run) > 1:
            matrix = _gate_matrix([(name, theta) for _, name, theta in run])
            zxz = _zxz(matrix)
            xzx = tuple(("rz" if name == "rx" else "rx", theta)
                        for name, theta in _zxz(_H @ matrix @ _H))
            seq = min((zxz, xzx), key=len)
            if len(seq) < len(run):
                replacements[run[0][0]] = (q, seq)
                skip.update(i for i, _, _ in run)
        run.clear()

    for i, ins in enumerate(circuit.data):
        name = ins.operation.name
        if name in ("rx", "rz"):
            q = indices[ins.qubits[0]]
            runs[q].append((i, name, float(ins.operation.params[0])))
        elif name in ("cz", "measure"):
            for q in ins.qubits:
                flush(indices[q])
    for q in range(n):
        flush(q)

    if not replacements:
        return circuit

    out = circuit.copy_empty_like()
    for i, ins in enumerate(circuit.data):
        if i in replacements:
            q, seq = replacements[i]
            for name, theta in seq:
                getattr(out, name)(theta, q)
        elif i not in skip:
            out.append(ins)
    return out


def _finalize_phases(circuit):
    original = _place_z_phases(_lower_z_phases(circuit))
    fused = _fuse_local_gates(original)
    if fused is original:
        return original
    optimized = _place_z_phases(_lower_z_phases(fused))
    return min((original, fused, optimized), key=_scored_depth)


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
