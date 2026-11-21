"""zonedwell: zoned neutral-atom compiler with a hub *conveyor* lane.

Self-contained (stdlib + numpy + qiskit). Only `rx`, `rz` (storage), `cz` (entangling zone),
`move`, `measure` and `barrier` are ever emitted.

Gate algebra (all identities exact, up to global phase):

  * `cx(u,v); rz(t) v; cx(u,v)` == `H_v cz(u,v) rx(t)_v cz(u,v) H_v`  == exp(-i t/2 Z_u Z_v),
    so only the *target* needs a 1q gate between the two `cz`; the control sits still.
  * `cx(u,v)` == `H_v cz(u,v) H_v`.
  * iHVA block `rx(pi/2) v; cx(u,v); rz(t) v; cx(u,v); rx(-pi/2) v` == exp(-i t/2 Z_u Y_v)
    == `rx(pi/2)_v [ H_t cz rx(t)_t cz H_t ] rx(-pi/2)_v` for *either* choice of t in {u,v}
    (the ZZ core is symmetric); we pick t = the non-hub endpoint.
  * every maximal run of 1q gates on a qubit is merged into one `rz.rx.rz` (ZXZ Euler) that is
    emitted only while the atom is in storage.

Movement:

  * the hub's partners live on the even columns ("lanes", 10 slots each). A lane is parked at
    storage rows 0..9 or entangling rows 10..19, so one bulk `move` (cost 10.0) switches zones.
  * hub blocks are served by the conveyor: the hub parks on the lane's right slot at row 10 and
    the lane slides down one row at a time; each partner collects `cz`, `rx` in storage, `cz`.
  * non-hub blocks are served as matchings: both atoms are placed on an interaction pair, one
    `cz` pulse fires, the whole zone takes one bulk dip for the `rx` gates, and a second pulse
    fires. Only the atoms that moved are moved back.
"""

from __future__ import annotations

import itertools
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction

Site = tuple[int, int]

_DEBUG = False

_STATS = {"lanes": 0, "generic": 0, "fell_back": 0}


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


# ---------------------------------------------------------------- geometry


def _phys(site):
    x, y = site
    if y <= 9:
        return (float(x), float(y))
    return (2.0 * (x // 2) + 0.3 * (x % 2), float(y))


def _dist(a, b):
    (x1, y1), (x2, y2) = _phys(a), _phys(b)
    return float(np.hypot(x2 - x1, y2 - y1))


def _in_storage(s):
    return 0 <= s[0] <= 9 and 0 <= s[1] <= 9


def _in_zone(s):
    return 0 <= s[0] <= 9 and 10 <= s[1] <= 19


def _partner_site(site):
    """The other site of the interaction pair that `site` belongs to."""
    x, y = site
    return (x + 1, y) if x % 2 == 0 else (x - 1, y)


# ---------------------------------------------------------------- move batcher
#
# A `Move` picks a contiguous row range and a contiguous column range and applies one strictly
# increasing row map and one strictly increasing column map to every atom in the rectangle, at a
# cost of only the longest displacement.  So the cheapest way to realise a set of relocations is
# to fuse as many as possible into one move.  `_group_params` re-implements the validator rules
# (cf. `compilers/rectbatch/compiler.py:move_params`, but zone crossing is allowed here).


def _move_cost(atoms):
    """Scored cost of a (legal or candidate) move: the longest physical displacement."""
    return max((_dist(s, d) for s, d in atoms.values()), default=0.0)


def _group_params(atoms, occ):
    """Validator-exact parameters of one batched move, or None.

    `atoms` maps qubit -> (source site, target site); `occ` maps site -> qubit.  Returns
    `(rows, cols, new_rows, new_cols, members)` where `members` additionally contains atoms that
    are picked up because they sit inside the source rectangle and are fixed points of both maps.
    """
    atoms = {q: (tuple(s), tuple(d)) for q, (s, d) in atoms.items()}
    if not atoms:
        return None
    seen = dict(atoms)
    while True:
        row_map: dict[int, int] = {}
        col_map: dict[int, int] = {}
        for src, dst in seen.values():
            sx, sy = src
            dx, dy = dst
            if not (0 <= sx <= 9 and 0 <= dx <= 9 and 0 <= sy <= 19 and 0 <= dy <= 19):
                return None
            if row_map.setdefault(sy, dy) != dy:
                return None
            if col_map.setdefault(sx, dx) != dx:
                return None
        src_rows = sorted(row_map)
        src_cols = sorted(col_map)
        if any(row_map[b] <= row_map[a] for a, b in itertools.pairwise(src_rows)):
            return None
        if any(col_map[b] <= col_map[a] for a, b in itertools.pairwise(src_cols)):
            return None
        rows = tuple(range(src_rows[0], src_rows[-1] + 1))
        cols = tuple(range(src_cols[0], src_cols[-1] + 1))
        added = False
        for (x, y), q in occ.items():
            if x in cols and y in rows and q not in seen:
                if row_map.get(y) == y and col_map.get(x) == x:
                    seen[q] = ((x, y), (x, y))
                    added = True
                else:
                    return None
        if not added:
            break
    members = set(seen)
    for dst in (d for _, d in seen.values()):
        other = occ.get(dst)
        if other is not None and other not in members:
            return None
    new_rows = tuple(row_map[y] for y in src_rows)
    new_cols = tuple(col_map[x] for x in src_cols)
    return rows, cols, new_rows, new_cols, seen


def _apply_atoms(occ, atoms):
    for q, (src, dst) in atoms.items():
        if occ.get(src) == q:
            del occ[src]
        occ[dst] = q


def _free_site(occ, zone_only=False):
    """A deterministic free site that does not form an interaction pair with a lone atom."""
    for row in range(19, 9, -1):
        for col in range(10):
            s = (col, row)
            if s not in occ and _partner_site(s) not in occ:
                return s
    for row in range(9, -1, -1):
        for col in range(10):
            s = (col, row)
            if s not in occ and _partner_site(s) not in occ:
                return s
    if zone_only:
        return None
    for row in range(9, -1, -1):
        for col in range(10):
            s = (col, row)
            if s not in occ:
                return s
    raise AssertionError("no free site")


def _plan_moves(reloc, occ):
    """Fuse relocations `{q: (src, dst)}` into as few legal rectangle moves as possible.

    Returns an ordered list of groups `{q: (src, dst)}` that can be emitted in order on `occ`
    (destinations free, or vacated by the group itself).  No emission happens here.
    """
    reloc = {q: (tuple(s), tuple(d)) for q, (s, d) in reloc.items() if tuple(s) != tuple(d)}
    if not reloc:
        return []
    groups = [{q: reloc[q]} for q in sorted(reloc)]
    while True:
        best = None
        n = len(groups)
        for i in range(n):
            ai = groups[i]
            for j in range(i + 1, n):
                aj = groups[j]
                if not ai.keys().isdisjoint(aj.keys()):
                    continue
                merged = {**ai, **aj}
                if _group_params(merged, occ) is None:
                    continue
                saving = _move_cost(ai) + _move_cost(aj) - _move_cost(merged)
                if saving < -1e-12:
                    continue
                key = (saving, -i, -j)
                if best is None or key > best[0]:
                    best = (key, i, j)
        if best is None:
            break
        _, i, j = best
        groups[i] = {**groups[i], **groups[j]}
        groups.pop(j)

    # Order the groups so that every destination is free (or vacated by the group itself) when it
    # runs.  A group is ready when none of its destinations is still occupied by another pending
    # group; that is a topological sort of the "destinations are other groups' sources" relation.
    # A cycle is broken by parking one atom on a free site for the duration.
    order = []
    busy = set(occ)
    for g in groups:
        busy.update(d for _, d in g.values())
    remaining = list(range(len(groups)))
    guard = 2 * len(reloc) + 8
    while remaining and guard:
        guard -= 1
        owner = {}
        for i in remaining:
            for src, _ in groups[i].values():
                owner[src] = i
        ready = []
        for i in remaining:
            if all(d not in owner or owner[d] == i for _, d in groups[i].values()):
                ready.append(i)
        if ready:
            for i in ready:
                order.append(groups[i])
                remaining.remove(i)
            continue
        # cycle: park one atom of the first pending group on a site that stays free
        i = remaining[0]
        g = groups[i]
        for q in sorted(g):
            src, dst = g[q]
            if dst in owner and owner[dst] != i:
                tmp = _free_site_away(busy)
                order.append({q: (src, tmp)})
                busy.add(tmp)
                g[q] = (tmp, dst)
                break
        else:  # pragma: no cover - defensive: nothing to break, release as single atoms
            remaining.remove(i)
            for q in sorted(g):
                order.append({q: g[q]})
    for i in remaining:  # pragma: no cover - defensive
        for q in sorted(groups[i]):
            order.append({q: groups[i][q]})
    return order


def _free_site_away(busy):
    """A free site (and partner) that is not in `busy`, so it stays free for a whole plan."""
    for row in range(19, 9, -1):
        for col in range(10):
            s = (col, row)
            if s not in busy and _partner_site(s) not in busy:
                return s
    for row in range(9, -1, -1):
        for col in range(10):
            s = (col, row)
            if s not in busy and _partner_site(s) not in busy:
                return s
    for row in range(19, -1, -1):
        for col in range(10):
            s = (col, row)
            if s not in busy:
                return s
    raise AssertionError("no free site for a cycle break")


# ---------------------------------------------------------------- 1q algebra

_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def _rx(t):
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0.0], [0.0, np.exp(0.5j * t)]], dtype=complex)


def _eq_phase(a, b, tol=1e-7):
    """`a == exp(i phi) * b` for 2x2 unitaries."""
    return abs(abs(np.trace(np.conj(a).T @ b)) / 2.0 - 1.0) < tol


def _zxz(u):
    """Angles `(a, b, c)` with `u == exp(i phase) * RZ(a) RX(b) RZ(c)`."""
    u = np.asarray(u, dtype=complex)
    un = u / np.sqrt(np.linalg.det(u))
    c00, c10 = un[0, 0], un[1, 0]
    beta = 2.0 * np.arctan2(abs(c10), abs(c00))
    ac2 = -np.angle(c00) if abs(c00) > 1e-12 else 0.0
    am2 = np.angle(c10) + np.pi / 2.0 if abs(c10) > 1e-12 else 0.0
    a, c = ac2 + am2, ac2 - am2
    if _DEBUG:
        assert _eq_phase(_rz(a) @ _rx(beta) @ _rz(c), un, 1e-6), "ZXZ decomposition failed"
    return a, beta, c


def _tight(t):
    """Angle reduced to (-pi, pi]; 0.0 when negligible."""
    t = float(t) % (2.0 * np.pi)
    if t > np.pi:
        t -= 2.0 * np.pi
    return 0.0 if abs(t) < 1e-9 else t


def _mid_theta(mid):
    """`theta` with `mid == exp(i phase) rz(theta)`, or None if `mid` is not a z rotation."""
    if mid is None:
        return None
    m = np.asarray(mid, dtype=complex)
    if abs(m[0, 1]) > 1e-9 or abs(m[1, 0]) > 1e-9:
        return None
    return -2.0 * np.angle(m[0, 0])


# ---------------------------------------------------------------- state


class _State:
    def __init__(self, n, nclbits):
        self.n = n
        self.out = QuantumCircuit(n, nclbits)
        self.pos: dict[int, Site] = {}
        self.occ: dict[Site, int] = {}
        self.pending: dict[int, np.ndarray] = {}
        self.loading: dict[int, Site] = {}
        self.in_zone = False
        self.distance = 0.0
        self.nmoves = 0
        self.measurements = []

    def load(self, home):
        self.loading = dict(home)
        self.pos = dict(home)
        self.occ = {s: q for q, s in self.pos.items()}
        assert len(self.occ) == len(self.pos), "duplicate loading site"

    # -- buffered 1q gates
    def queue_1q(self, q, u):
        self.pending[q] = u if q not in self.pending else u @ self.pending[q]

    def flush(self, q):
        u = self.pending.pop(q, None)
        if u is None:
            return
        assert _in_storage(self.pos[q]), f"1q flush of q{q} outside storage at {self.pos[q]}"
        a, b, c = _zxz(u)
        # `u == RZ(a) RX(b) RZ(c)` as a matrix product, so c is applied first
        for ang, gate in ((c, "rz"), (b, "rx"), (a, "rz")):
            ang = _tight(ang)
            if ang:
                getattr(self.out, gate)(ang, q)

    def flush_all(self):
        for q in list(self.pending):
            self.flush(q)

    # -- moves
    def move(self, rows, cols, new_rows, new_cols):
        rows = tuple(int(v) for v in rows)
        cols = tuple(int(v) for v in cols)
        new_rows = tuple(int(v) for v in new_rows)
        new_cols = tuple(int(v) for v in new_cols)
        assert rows == tuple(range(rows[0], rows[0] + len(rows))), rows
        assert cols == tuple(range(cols[0], cols[0] + len(cols))), cols
        assert all(b > a for a, b in itertools.pairwise(new_rows)), new_rows
        assert all(b > a for a, b in itertools.pairwise(new_cols)), new_cols
        picked = [(x, y) for y in rows for x in cols if (x, y) in self.occ]
        if not picked:
            return
        qubits = [self.occ[s] for s in picked]
        orows = sorted({y for _, y in picked})
        ocols = sorted({x for x, _ in picked})
        rsrc = orows if len(new_rows) < len(rows) else list(rows)
        csrc = ocols if len(new_cols) < len(cols) else list(cols)
        assert len(rsrc) == len(new_rows) and len(csrc) == len(new_cols), (rsrc, new_rows, csrc, new_cols)
        rmap = dict(zip(rsrc, new_rows, strict=True))
        cmap = dict(zip(csrc, new_cols, strict=True))
        tgt = {s: (cmap[s[0]], rmap[s[1]]) for s in picked}
        assert len(set(tgt.values())) == len(tgt), "two atoms to one site"
        qset = set(qubits)
        for s, t in tgt.items():
            assert 0 <= t[0] <= 9 and 0 <= t[1] <= 19, f"target {t} outside every zone"
            o = self.occ.get(t)
            assert o is None or o in qset, f"deposit {s}->{t} onto qubit {o}"
        before = {q: self.pos[q] for q in qubits}
        self.out.append(Move(rows, cols, new_rows, new_cols, len(qubits)), qubits)
        for s in picked:
            del self.occ[s]
        for s in picked:
            q = qubits[picked.index(s)]
            self.pos[q] = tgt[s]
            self.occ[tgt[s]] = q
        self.distance += max(_dist(before[q], self.pos[q]) for q in qubits)
        self.nmoves += 1
        assert len(self.occ) == self.n == len(self.pos)

    # -- cz
    def occupied_pairs(self):
        sites = {s: q for q, s in self.pos.items() if _in_zone(s)}
        out = []
        for (x, y), q in sites.items():
            if x % 2 == 0 and (x + 1, y) in sites:
                out.append((q, sites[(x + 1, y)]))
        return out

    def pulse(self, pairs):
        """Emit one `cz` on each pair; in debug mode assert that nothing else is hit."""
        expect = {frozenset(p) for p in pairs}
        if _DEBUG:
            got = {frozenset(p) for p in self.occupied_pairs()}
            assert got == expect, (
                f"pulse mismatch: got {sorted(map(sorted, got))[:6]} expect {sorted(map(sorted, expect))[:6]}"
            )
        for a, b in pairs:
            assert _in_zone(self.pos[a]) and _in_zone(self.pos[b]), (a, b, self.pos[a], self.pos[b])
            assert _partner_site(self.pos[a]) == self.pos[b], (self.pos[a], self.pos[b])
            self.out.cz(a, b)


def _emit_plan(st, plan):
    """Emit a plan produced by `_plan_moves`, rechecking legality on the live state."""
    for g in plan:
        g = {q: (tuple(s), tuple(d)) for q, (s, d) in g.items()}
        g = {q: sv for q, sv in g.items() if st.pos[q] != sv[1]}
        if not g:
            continue
        if len(g) > 1:
            params = _group_params(g, st.occ)
            if params is not None:
                rows, cols, new_rows, new_cols, _ = params
                st.move(rows, cols, new_rows, new_cols)
                continue
        _emit_singles(st, g)


def _emit_singles(st, atoms):
    """One atom at a time: destination-free first, and a temporary site only for a swap cycle."""
    todo = {q: d for q, (s, d) in atoms.items() if st.pos[q] != d}
    guard = 2 * len(todo) + 4
    while todo and guard:
        guard -= 1
        for q in sorted(todo):
            d = todo[q]
            if d in st.occ and st.occ[d] != q:
                continue
            cur = st.pos[q]
            st.move((cur[1],), (cur[0],), (d[1],), (d[0],))
            del todo[q]
            break
        else:
            if not todo:
                break
            q = min(todo)  # pragma: no cover - only reachable for a swap cycle
            cur = st.pos[q]
            tmp = _free_site(st.occ, zone_only=_in_zone(cur))
            st.move((cur[1],), (cur[0],), (tmp[1],), (tmp[0],))
    assert not todo, f"could not place {sorted(todo)}"


# ---------------------------------------------------------------- parsing

_ONE_Q = frozenset(
    {"h", "rx", "rz", "ry", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "u1", "u2", "u3", "u", "p", "id"}
)


class _Block:
    __slots__ = ("kind", "u", "v", "mid", "pre", "post")

    def __init__(self, kind, u, v, mid=None, pre=None, post=None):
        self.kind = kind  # 'zz' | 'cx' | 'cz'
        self.u = u
        self.v = v
        self.mid = mid  # 2x2 matrix of the gate between the two cz (None for a single cz)
        self.pre = pre  # list[(q, 2x2)] applied before the first cz, or None
        self.post = post  # list[(q, 2x2)] applied after the second cz, or None

    def qubits(self):
        return (self.u, self.v)

    def __repr__(self):
        return f"Block({self.kind},{self.u},{self.v})"


def _matrix(op):
    from qiskit.quantum_info import Operator

    return np.asarray(Operator(op).data, dtype=complex)


def parse_blocks(circuit):
    """Linear scan -> ordered items, or `None` if an unsupported gate occurs.

    Items:
        ('1q', q, 2x2)      single-qubit gate (merged per qubit later)
        ('blk', _Block)     one two-qubit block
        ('measure', q, c)   measurement
    """
    data = list(circuit.data)
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    cindex = {bit: i for i, bit in enumerate(circuit.clbits)}
    n = len(data)
    consumed = [False] * n
    items = []

    def bits(inst):
        return [index[b] for b in inst.qubits]

    def touches(inst, qs):
        return not set(bits(inst)).isdisjoint(qs)

    for i in range(n):
        if consumed[i]:
            continue
        inst = data[i]
        name = inst.operation.name
        if name == "barrier":
            consumed[i] = True
            continue
        if name == "measure":
            items.append(("measure", index[inst.qubits[0]], cindex[inst.clbits[0]]))
            consumed[i] = True
            continue
        qubits = bits(inst)
        if name in _ONE_Q and len(qubits) == 1:
            try:
                mat = _matrix(inst.operation)
            except Exception:
                return None
            items.append(("1q", qubits[0], mat))
            consumed[i] = True
            continue
        if name == "cz" and len(qubits) == 2:
            items.append(("blk", _Block("cz", qubits[0], qubits[1])))
            consumed[i] = True
            continue
        if name != "cx" or len(qubits) != 2:
            return None

        u, v = qubits
        pair = {u, v}
        j = i + 1
        while j < n and not touches(data[j], pair):
            j += 1
        k = -1
        if j < n and data[j].operation.name in ("rz", "u1", "p") and bits(data[j]) == [v]:
            k = j + 1
            while k < n and not touches(data[k], pair):
                k += 1
        if 0 < k < n and data[k].operation.name == "cx" and bits(data[k]) == [u, v]:
            mid = _matrix(data[j].operation)
            consumed[j] = True
            pre = post = None
            if i > 0 and not consumed[i - 1] and data[i - 1].operation.name in _ONE_Q:
                if bits(data[i - 1]) == [v]:
                    pre = [(v, _matrix(data[i - 1].operation))]
                    consumed[i - 1] = True
            if k + 1 < n and not consumed[k + 1] and data[k + 1].operation.name in _ONE_Q:
                if bits(data[k + 1]) == [v]:
                    post = [(v, _matrix(data[k + 1].operation))]
                    consumed[k + 1] = True
            items.append(("blk", _Block("zz", u, v, mid, pre, post)))
            consumed[i] = consumed[k] = True
            continue
        items.append(("blk", _Block("cx", u, v)))
        consumed[i] = True
    return _elide_identity_blocks(_absorb_ihva_pre(items))


def _elide_identity_blocks(items):
    """Drop `cx; rz(0); cx` blocks whose outer 1q gates also multiply to the identity.

    A `zz` block is `pre_v . (cz mid cz) . post_v`, so an identity `mid` reduces it to `post . pre`
    on `v`. When that is the identity too (the `rx(+pi/2)`, `rx(-pi/2)` of an absorbed iHVA block)
    the whole item disappears — dropping it, rather than leaving two 1q items behind, keeps the
    surrounding blocks in one run (1q items delimit runs).
    """
    eye = np.eye(2)
    out = []
    for item in items:
        if item[0] != "blk":
            out.append(item)
            continue
        blk = item[1]
        if blk.kind != "zz" or blk.mid is None or not _eq_phase(blk.mid, eye):
            out.append(item)
            continue
        pre = blk.pre or []
        post = blk.post or []
        if any(q != blk.v for q, _ in pre) or any(q != blk.v for q, _ in post):
            out.append(item)
            continue
        u = _fold([m for _, m in post]) @ _fold([m for _, m in pre])
        if not _eq_phase(u, eye):
            out.append(("1q", blk.v, u))
    return out


def _absorb_ihva_pre(items):
    """Attach the `rx(+pi/2)` that precedes an iHVA block (already scanned as a separate 1q item)."""
    for idx, item in enumerate(items):
        if item[0] != "blk":
            continue
        blk = item[1]
        if blk.kind != "zz" or blk.pre is not None or blk.post is None:
            continue
        if len(blk.post) != 1 or blk.post[0][0] != blk.v:
            continue
        if not _eq_phase(blk.post[0][1], _rx(-np.pi / 2)):
            continue
        if idx == 0 or items[idx - 1][0] != "1q":
            continue
        prev = items[idx - 1]
        if prev[1] != blk.v or not _eq_phase(prev[2], _rx(np.pi / 2)):
            continue
        blk.pre = [(blk.v, prev[2])]
        items[idx - 1] = ("dropped",)
    return [it for it in items if it[0] != "dropped"]


def _block_generator(blk):
    """Pauli string {qubit: letter} whose rotation this block is (up to sign), or None."""
    if blk.kind != "zz":
        return None
    u, v = blk.u, blk.v
    if blk.pre is None and blk.post is None:
        return {u: "Z", v: "Z"}
    if (
        blk.pre is not None
        and blk.post is not None
        and len(blk.pre) == 1
        and len(blk.post) == 1
        and blk.pre[0][0] == v
        and blk.post[0][0] == v
        and _eq_phase(blk.pre[0][1], _rx(np.pi / 2))
        and _eq_phase(blk.post[0][1], _rx(-np.pi / 2))
    ):
        return {u: "Z", v: "Y"}
    return None


def _commute(b1, g1, b2, g2):
    if b1.u not in (b2.u, b2.v) and b1.v not in (b2.u, b2.v):
        return True
    if g1 is None or g2 is None:
        return False
    shared = set(g1) & set(g2)
    return sum(1 for q in shared if g1[q] != g2[q]) % 2 == 0


def _category(blk, hub):
    if hub not in (blk.u, blk.v):
        return "non"
    g = _block_generator(blk)
    if g is not None and g.get(hub) == "Z":
        return "hubZ"
    return "hubY"


def plan_groups(blocks, hub):
    """Ready-set scheduling: every ready set is an antichain of the commutation DAG, so all its
    members pairwise commute and may be reordered freely."""
    n = len(blocks)
    gens = [_block_generator(b) for b in blocks]
    preds = [{i for i in range(j) if not _commute(blocks[i], gens[i], blocks[j], gens[j])} for j in range(n)]
    done: set[int] = set()
    groups = []
    while len(done) < n:
        ready = [j for j in range(n) if j not in done and not (preds[j] - done)]
        if not ready:  # pragma: no cover
            ready = [j for j in range(n) if j not in done]
        chosen = None
        for cat in ("hubZ", "hubY", "non"):
            sel = [j for j in ready if _category(blocks[j], hub) == cat]
            if sel:
                chosen = sel
                break
        groups.append([blocks[j] for j in chosen])
        done |= set(chosen)
    return groups


# ---------------------------------------------------------------- layout


def plan_layout(n, hub, blocks, groups, use_conveyor=True):
    """Pack the partners into the five even lanes (10 slots each) and improve the packing."""
    first = {}
    for idx, blk in enumerate(blocks):
        if blk.u == hub:
            first.setdefault(blk.v, idx)
        elif blk.v == hub:
            first.setdefault(blk.u, idx)
    partners = [q for q in range(n) if q != hub]
    if len(partners) > 49:
        return None
    partners.sort(key=lambda q: (first.get(q, 10 ** 9), q))
    L = max(1, min(10, -(-(len(partners) + 1) // 5)))
    slots = [(2 * j, 10 - L + s) for j in range(5) for s in range(L)][: len(partners)]
    assign = {slots[i]: partners[i] for i in range(len(partners))}
    costs = _LayoutCost(blocks, groups, hub, use_conveyor)
    best = costs.eval(assign)
    rng = np.random.default_rng(12345)
    keys = list(assign)
    for _ in range(200 * len(keys)):
        i, j = rng.integers(0, len(keys), 2)
        if i == j:
            continue
        a, b = keys[i], keys[j]
        assign[a], assign[b] = assign[b], assign[a]
        new = costs.eval(assign)
        if new < best - 1e-12:
            best = new
        else:
            assign[a], assign[b] = assign[b], assign[a]
    home = {q: site for site, q in assign.items()}
    home[hub] = (9, 9)
    if len(set(home.values())) != n:
        return None
    zone_home = {q: (s[0], s[1] + L) for q, s in home.items()}
    if (8, 9) in assign or (8, 9 + L) in zone_home.values():
        return None
    occupied = {v for q, v in zone_home.items() if q != hub}
    park = None
    for y in ([9 + L] if 9 + L <= 19 else []) + list(range(10, 10 + L)):
        if (9, y) not in occupied and (8, y) not in occupied:
            park = (9, y)
            break
    if park is None or park[1] > 9 + L:
        return None
    return {"home": home, "zone_home": zone_home, "lanes": _lanes(partners, home), "hub": hub,
            "partners": partners, "conveyor": use_conveyor, "L": L, "park": park}


def _lanes(partners, home):
    lanes = {}
    for q in partners:
        lanes.setdefault(home[q][0] // 2, []).append(q)
    for lane in lanes.values():
        lane.sort(key=lambda q: home[q][1])
    return lanes


class _LayoutCost:
    """Cost model of a layout: one pair of moves per block plus the conveyor slides."""

    def __init__(self, blocks, groups, hub, use_conveyor):
        self.hub = hub
        self.use_conveyor = use_conveyor
        self.groups = groups

    def eval(self, assign):
        base = min(site[1] for site in assign)
        pos = {q: (site[0], site[1] - base) for site, q in assign.items()}
        cost = 0.0
        for group in self.groups:
            if self.use_conveyor and all(self.hub in (b.u, b.v) for b in group):
                per = {}
                for b in group:
                    p = b.v if b.u == self.hub else b.u
                    if p in pos:
                        per.setdefault(pos[p][0] // 2, []).append(pos[p][1])
                for _lane, ss in per.items():
                    cost += 2.0 * (max(ss) + 1) + 2.0
            else:
                for b in group:
                    if b.u in pos and b.v in pos:
                        cost += float(np.hypot(pos[b.u][0] - pos[b.v][0], pos[b.u][1] - pos[b.v][1]))
                if self.use_conveyor:
                    # matching rounds dip down to their deepest dweller: prefer a shallow endpoint
                    dd = [min(pos[b.u][1], pos[b.v][1]) for b in group if b.u in pos and b.v in pos]
                    cost += 4.0 * (1 + max(dd)) * (1 + len(dd) // 8) if dd else 0.0
        return cost


# ---------------------------------------------------------------- scheduling machinery


class _Machine:
    """Movement on a lane layout: one bulk move switches zones, lanes slide, matchings pulse."""

    def __init__(self, st, lay):
        self.st = st
        self.lay = lay
        self.home = lay["home"]
        self.zone_home = lay["zone_home"]
        self.hub = lay["hub"]

    # -- zone helpers
    def to_storage(self):
        if self.st.in_zone:
            L = self.lay["L"]
            self.st.move(range(10, 20), range(0, 10), range(10 - L, 20 - L), range(0, 10))
            self.st.in_zone = False

    def to_zone(self):
        if not self.st.in_zone:
            self.st.flush_all()
            L = self.lay["L"]
            self.st.move(range(10 - L, 20 - L), range(0, 10), range(10, 20), range(0, 10))
            self.st.in_zone = True

    def storage_phase(self):
        self.to_storage()
        self.st.flush_all()

    def dip(self, depth, action):
        """Shift the whole zone down by `depth` rows, run `action`, shift it back exactly.

        A dip only has to be deep enough for the deepest atom that needs a 1q gate to reach
        storage, so `depth = 1 + max(y - 10)` over those atoms; the shift is uniform, so every
        pair keeps its occupancy and no `cz` is emitted while it is open.
        """
        st = self.st
        assert st.in_zone
        depth = max(1, min(int(depth), 10))
        before = dict(st.pos)
        st.move(range(10, 20), range(0, 10), range(10 - depth, 20 - depth), range(0, 10))
        action()
        st.move(range(10 - depth, 20 - depth), range(0, 10), range(10, 20), range(0, 10))
        assert st.pos == before, "dip did not restore the positions"

    def flush_dip(self, qubits):
        """Flush the pending 1q gates of `qubits` with the shallowest dip that reaches storage."""
        st = self.st
        need = [q for q in qubits if q in st.pending]
        if not need:
            return
        if not st.in_zone:
            for q in need:
                st.flush(q)
            return
        depth = 1 + max(st.pos[q][1] - 10 for q in need)

        def action():
            for q in need:
                st.flush(q)

        self.dip(depth, action)

    def _flush_in_storage(self):
        for q in list(self.st.pending):
            if _in_storage(self.st.pos[q]):
                self.st.flush(q)

    def park_hub(self):
        site = self.lay["park"]
        p = self.st.pos[self.hub]
        if p != site:
            self.st.move((p[1],), (p[0],), (site[1],), (site[0],))

    # -- conveyor ------------------------------------------------------
    def conveyor_lane(self, col, served):
        """`served` maps partner -> theta.  The lane's atoms must sit on entangling rows 10+slot."""
        st = self.st
        top = max(self.zone_home[q][1] - 10 for q in served)
        g = top + 1
        hp = st.pos[self.hub]
        target = (col + 1, 10)
        if hp != target:
            st.move((hp[1],), (hp[0],), (target[1],), (target[0],))
        for _ in range(g):  # D pass
            head = st.occ.get((col, 10))
            if head is not None and head in served:
                st.pulse([(self.hub, head)])
            rows = [y for (x, y) in st.occ if x == col]
            r0, r1 = min(rows), max(rows)
            st.move(range(r0, r1 + 1), (col,), range(r0 - 1, r1), (col,))
        for q, theta in served.items():
            assert _in_storage(st.pos[q]), f"conveyor mid gate for q{q} at {st.pos[q]}"
            ang = _tight(theta)
            if ang:
                st.out.rx(ang, q)
        for _ in range(g):  # U pass
            rows = [y for (x, y) in st.occ if x == col]
            r0, r1 = min(rows), max(rows)
            st.move(range(r0, r1 + 1), (col,), range(r0 + 1, r1 + 2), (col,))
            head = st.occ.get((col, 10))
            if head is not None and head in served:
                st.pulse([(self.hub, head)])

    def _conveyor_ops(self, group, targ):
        """Per-qubit `(before, after)` 1q operators for one conveyor pass, or None.

        For a qubit touched by several blocks of the pass the program order is
        `pre_1 [cz..] post_1 pre_2 [cz..] post_2 ...`. Adjacent `post_j pre_{j+1}` pairs cancel
        (`H H = I` for QAOA, `rx(-pi/2) rx(pi/2) = I` for iHVA), so the pass reduces to one
        operator before and one after; if a pair does not cancel the pass is refused.
        """
        order: dict[int, list] = {}
        for blk in group:
            for q, u in _block_pre(blk, targ[blk]):
                order.setdefault(q, []).append(("pre", u))
            for q, u in _block_post(blk, targ[blk]):
                order.setdefault(q, []).append(("post", u))
        before, after = {}, {}
        for q, ops in order.items():
            stack = []
            for kind, u in ops:
                if kind == "pre" and stack and stack[-1][0] == "post" and _eq_phase(u @ stack[-1][1], np.eye(2)):
                    stack.pop()
                    continue
                stack.append((kind, u))
            if any(kind == "post" for kind, _ in stack) and any(kind == "pre" for kind, _ in stack):
                first_post = next(i for i, (kind, _) in enumerate(stack) if kind == "post")
                if any(kind == "pre" for kind, _ in stack[first_post:]):
                    return None
            pre = [u for kind, u in stack if kind == "pre"]
            post = [u for kind, u in stack if kind == "post"]
            b = _fold(pre)
            a = _fold(post)
            if not _eq_phase(b, np.eye(2), 1e-9):
                before[q] = b
            if not _eq_phase(a, np.eye(2), 1e-9):
                after[q] = a
        return before, after

    def conveyor(self, group):
        st = self.st
        targ = {blk: (blk.v if blk.u == self.hub else blk.u) for blk in group}
        theta = {}
        for blk in group:
            t = _mid_theta(blk.mid)
            if t is None:
                return False
            theta[blk] = t
        ops = self._conveyor_ops(group, targ)
        if ops is None:
            return False
        before, after = ops
        for q, u in before.items():
            st.queue_1q(q, u)
        # every qubit the pass pulses must have no pending 1q gate: checking only `before` misses
        # the hub (or a partner whose queued pre cancelled its pending post) -> wrong gate order
        involved = {q for blk in group for q in blk.qubits()} | {self.hub}
        per_lane = {}
        for blk in group:
            p = targ[blk]
            per_lane.setdefault(self.home[p][0] // 2, {})[p] = theta[blk]
        if not st.in_zone:
            self.storage_phase()
            self.to_zone()
        elif self._needs_flush(involved):
            # shallow dip instead of a full zone round trip; bring the hub to row 10 first
            need = sorted(q for q in involved if q in st.pending and not _eq_phase(st.pending[q], np.eye(2), 1e-9))
            if self.hub in need:
                hp = st.pos[self.hub]
                tgt = (2 * min(per_lane) + 1, 10)
                if hp != tgt:
                    st.move((hp[1],), (hp[0],), (tgt[1],), (tgt[0],))
            self.flush_dip(need)
            assert not self._needs_flush(involved)
        for lane_idx in sorted(per_lane):
            self.conveyor_lane(2 * lane_idx, per_lane[lane_idx])
        for q, u in after.items():
            st.queue_1q(q, u)
        return True

    # -- matchings -----------------------------------------------------
    def _ensure_hub_lone(self):
        hub = self.hub
        if hub not in self.st.pos:
            return
        if any(hub in pair for pair in self.st.occupied_pairs()):
            self.park_hub()

    def matching(self, group):
        self._ensure_hub_lone()
        for sub in ([b for b in group if b.mid is not None], [b for b in group if b.mid is None]):
            if not sub:
                continue
            for M in _greedy_matchings(sub):
                self.matching_round(M)

    def _needs_flush(self, qubits):
        for q in qubits:
            u = self.st.pending.get(q)
            if u is not None and not _eq_phase(u, np.eye(2), 1e-9):
                return True
        return False

    def matching_round(self, M):
        st = self.st
        self.to_zone()
        assign = self._choose_assign(M)
        if assign is None:
            for blk in M:
                self.single_block(blk)
            return
        involved = set()
        for blk, qa, qb, sa, sb, targ in assign:
            involved.update((qa, qb, targ))
        for blk, qa, qb, sa, sb, targ in assign:
            for q, u in _block_pre(blk, targ):
                st.queue_1q(q, u)
        moved = self._place(assign)
        if self._needs_flush(involved):
            self.flush_dip(involved)
        st.pulse([(qa, qb) for _, qa, qb, _, _, _ in assign])
        if M[0].mid is not None:
            targs = [targ for _, _, _, _, _, targ in assign]
            depth = 1 + max(0, max(st.pos[q][1] - 10 for q in targs))
            if self._needs_flush(involved):
                depth = max(depth, 1 + max(st.pos[q][1] - 10 for q in involved))

            def mid():
                for blk, qa, qb, sa, sb, targ in assign:
                    st.flush(targ)
                    theta = _mid_theta(blk.mid)
                    if theta is not None and _tight(theta):
                        st.out.rx(_tight(theta), targ)
                self._flush_in_storage()

            if st.in_zone and depth > 0:
                self.dip(depth, mid)
            else:
                mid()
            st.pulse([(qa, qb) for _, qa, qb, _, _, _ in assign])
        for blk, qa, qb, sa, sb, targ in assign:
            for q, u in _block_post(blk, targ):
                st.queue_1q(q, u)
        self._unplace(moved)

    def _assign(self, M, row_penalty=0.5, dweller_only=False):
        """Place both atoms of every block on a free interaction pair -> [(blk, qa, qb, sa, sb, targ)].

        `dweller_only` restricts a block to the pair one of its atoms already sits on, so only
        one atom travels per block (a fused rectangle is then much likelier)."""
        st = self.st
        claimed: set = set()
        out = []
        for blk in M:
            best = None
            options = []
            if dweller_only:
                pairs = []
                for q in (blk.u, blk.v):
                    a_site = st.pos[q]
                    pairs.append((min(a_site, _partner_site(a_site)), max(a_site, _partner_site(a_site))))
                options = list(dict.fromkeys(pairs))
            else:
                options = [((2 * cipher, row), (2 * cipher + 1, row))
                           for cipher in range(5) for row in range(10, 10 + self.lay["L"])]
            for a_site, b_site in options:
                if a_site in claimed or b_site in claimed:
                    continue
                if any(s in st.occ and st.occ[s] not in (blk.u, blk.v) for s in (a_site, b_site)):
                    continue
                for (qu, su), (qv, sv) in (
                    ((blk.u, a_site), (blk.v, b_site)),
                    ((blk.u, b_site), (blk.v, a_site)),
                ):
                    cost = (_dist(st.pos[qu], su) + _dist(st.pos[qv], sv)
                            + row_penalty * (a_site[1] - 10))
                    if best is None or cost < best[0]:
                        best = (cost, qu, qv, su, sv)
            if best is None:
                return None
            _, qu, qv, su, sv = best
            claimed.update((su, sv))
            out.append((blk, qu, qv, su, sv, _targ_of(blk, qu, qv, su)))
        return out

    def _assign_cost(self, assign):
        """Simulated cost of a placement: batched outward trip + return trip + the storage dip."""
        st = self.st
        sites = {q: site for _, qa, qb, sa, sb, _ in assign for q, site in ((qa, sa), (qb, sb))}
        reloc = {q: (st.pos[q], site) for q, site in sites.items() if st.pos[q] != site}
        occ, pos = dict(st.occ), dict(st.pos)
        cost = 0.0
        for g in _plan_moves(reloc, occ):
            cost += _move_cost(g)
            _apply_atoms(occ, {q: (pos[q], d) for q, (s, d) in g.items()})
            for q, (s, d) in g.items():
                pos[q] = d
        back = {q: (pos[q], st.pos[q]) for q in reloc if pos[q] != st.pos[q]}
        cost += sum(_move_cost(g) for g in _plan_moves(back, occ))
        targs = [t for *_, t in assign]
        cost += 2.0 * (1 + max(0, max(pos[t][1] - 10 for t in targs)))
        return cost

    def _choose_assign(self, M):
        """Pick the placement strategy with the cheapest simulated batched cost (deterministic)."""
        best, best_cost = None, None
        for penalty, dweller_only in ((0.5, False), (2.0, False), (0.5, True), (4.0, False), (1.5, True)):
            assign = self._assign(M, penalty, dweller_only)
            if assign is None:
                continue
            cost = self._assign_cost(assign)
            if best_cost is None or cost < best_cost - 1e-12:
                best, best_cost = assign, cost
        return best

    def _place(self, assign):
        """Move every atom to its assigned site with batched rectangle moves."""
        st = self.st
        reloc = {}
        homes = {}
        for blk, qa, qb, sa, sb, targ in assign:
            for q, site in ((qa, sa), (qb, sb)):
                if st.pos[q] != site:
                    homes[q] = st.pos[q]
                    reloc[q] = (st.pos[q], site)
        _emit_plan(st, _plan_moves(reloc, st.occ))
        return homes

    def _free_zone_site(self):
        st = self.st
        for row in range(10 + self.lay["L"] - 1, 9, -1):
            for cipher in range(5):
                for site in ((2 * cipher, row), (2 * cipher + 1, row)):
                    if site not in st.occ and _partner_site(site) not in st.occ:
                        return site
        raise AssertionError("no free entropy site in the entangling zone")

    def _try_bbox(self, grp, dx, dy):
        st = self.st
        ys = sorted({s[1] for _, s, _ in grp})
        xs = sorted({s[0] for _, s, _ in grp})
        rows = tuple(range(ys[0], ys[-1] + 1))
        cols = tuple(range(xs[0], xs[-1] + 1))
        picked = [(x, y) for y in rows for x in cols if (x, y) in st.occ]
        if len(picked) != len(grp) or set(picked) != {s for _, s, _ in grp}:
            return False
        new_rows = tuple(y + dy for y in rows)
        new_cols = tuple(x + dx for x in cols)
        if new_rows[0] < 0 or new_rows[-1] > 19 or new_cols[0] < 0 or new_cols[-1] > 9:
            return False
        for _, _, t in grp:
            o = st.occ.get(t)
            if o is not None and o not in {q for q, _, _ in grp}:
                return False
        st.move(rows, cols, new_rows, new_cols)
        return True

    def _unplace(self, homes):
        """Send every atom that `_place` moved back where it was, again batched."""
        st = self.st
        reloc = {q: (st.pos[q], h) for q, h in homes.items() if st.pos[q] != h}
        _emit_plan(st, _plan_moves(reloc, st.occ))

    def single_block(self, blk):
        """Fallback: park the block on a free pair, pulse, (dip), pulse, go home."""
        st = self.st
        self.to_zone()
        assign = self._assign([blk])
        if assign is None:
            raise AssertionError(f"no free pair for {blk}")
        _, qa, qb, sa, sb, targ = assign[0]
        for q, u in _block_pre(blk, targ):
            st.queue_1q(q, u)
        if self._needs_flush({qa, qb, targ}):
            self.flush_dip({qa, qb, targ})
        moved = self._place(assign)
        st.pulse([(qa, qb)])
        if blk.mid is not None:
            def mid():
                st.flush(targ)
                theta = _mid_theta(blk.mid)
                if theta is not None and _tight(theta):
                    st.out.rx(_tight(theta), targ)
                self._flush_in_storage()

            self.dip(1 + max(0, st.pos[targ][1] - 10), mid)
            st.pulse([(qa, qb)])
        for q, u in _block_post(blk, targ):
            st.queue_1q(q, u)
        self._unplace(moved)


def _fold(mats):
    m = np.eye(2, dtype=complex)
    for u in mats:
        m = u @ m
    return m


def _targ_of(blk, qu, qv, su):
    """Which atom carries the middle 1q gate: for a bare cx it is the cx target, else either one."""
    if blk.kind == "cx":
        return blk.v
    return qv if su[0] % 2 == 0 else qu


def _greedy_matchings(group):
    rounds = []
    for blk in group:
        for M, used in rounds:
            if blk.u not in used and blk.v not in used:
                M.append(blk)
                used.update(blk.qubits())
                break
        else:
            rounds.append(([blk], set(blk.qubits())))
    return [M for M, _ in rounds]


def _block_pre(blk, t):
    if blk.kind == "cz":
        return []
    if blk.kind == "cx":
        return [(t, _H)]
    pre = list(blk.pre) if blk.pre else []
    pre.append((t, _H))
    return pre


def _block_post(blk, t):
    if blk.kind == "cz":
        return []
    if blk.kind == "cx":
        return [(t, _H)]
    post = [(t, _H)]
    if blk.post:
        post.extend(blk.post)
    return post


# ---------------------------------------------------------------- generic (fallback) path


def _compile_generic(circuit, spec):
    """Any circuit: transpile to rx/rz/cz and carry each cz to a pair, one at a time."""
    n = circuit.num_qubits
    native = transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=0)
    items = parse_blocks(native)
    if items is None:  # pragma: no cover - rx/rz/cz always parse
        raise RuntimeError("transpiled circuit did not parse")
    if n > 50:
        home = {q: (q % 10, (q // 10) % 5) for q in range(n)}
        if len(set(home.values())) != n:
            home = {q: (q % 10, (q // 10) % 10) for q in range(n)}
    else:
        home = {q: (q % 10, q // 10) for q in range(n)}
    if len(set(home.values())) != n:
        home = {q: (q % 10, (q // 10) % 10) for q in range(n)}
    assert len(set(home.values())) == n, "cannot place every qubit in storage"
    st = _State(n, circuit.num_clbits)
    st.load(home)
    measures = []
    for item in items:
        if item[0] == "1q":
            st.queue_1q(item[1], item[2])
        elif item[0] == "measure":
            measures.append((item[1], item[2]))
        else:
            blk = item[1]
            _generic_block(st, blk, home)
    st.flush_all()
    for q, c in measures:
        st.out.measure(q, c)
    return st


def _generic_block(st: _State, blk: _Block, home):
    a, b = blk.u, blk.v
    st.flush(a)
    st.flush(b)
    pa, pb = _free_pair(st, (a, b), home)
    cur = st.pos[a]
    st.move((cur[1],), (cur[0],), (pa[1],), (pa[0],))
    cur = st.pos[b]
    st.move((cur[1],), (cur[0],), (pb[1],), (pb[0],))
    st.pulse([(a, b)])
    if blk.mid is not None:
        theta = _mid_theta(blk.mid)
        assert theta is not None, "generic path expects a z rotation in the middle"
        dip = _free_storage(st)
        cur = st.pos[b]
        st.move((cur[1],), (cur[0],), (dip[1],), (dip[0],))
        st.flush(b)
        st.out.rx(_tight(theta), b)
        cur = st.pos[b]
        st.move((cur[1],), (cur[0],), (pb[1],), (pb[0],))
        st.pulse([(a, b)])
    for q, target in ((a, home[a]), (b, home[b])):
        cur = st.pos[q]
        if cur != target:
            st.move((cur[1],), (cur[0],), (target[1],), (target[0],))


def _free_pair(st, own, home):
    """A2 free interaction pair closest to the two atoms' homes."""
    best = None
    for cipher in range(5):
        for row in range(10, 20):
            sa, sb = (2 * cipher, row), (2 * cipher + 1, row)
            if sa in st.occ or sb in st.occ:
                continue
            cost = _dist(home[own[0]], sa) + _dist(home[own[1]], sb)
            if best is None or cost < best[0]:
                best = (cost, sa, sb)
    assert best is not None, "no free interaction pair"
    return best[1], best[2]


def _free_storage(st: _State):
    for row in (9, 8, 7, 6, 5, 4, 3, 2, 1):
        for col in range(10):
            if (col, row) not in st.occ:
                return (col, row)
    raise AssertionError("no free storage site")


# ---------------------------------------------------------------- hub path


def _runs(items):
    """Maximal runs of consecutive blocks (1q gates delimit them)."""
    runs, run = [], []
    for item in items:
        if item[0] == "blk":
            run.append(item[1])
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def _compile_lanes(circuit, items, lay, groups):
    n = circuit.num_qubits
    st = _State(n, circuit.num_clbits)
    st.load(lay["home"])
    mach = _Machine(st, lay)
    hub = lay["hub"]
    use_conveyor = lay.get("conveyor", True)
    measures = [(it[1], it[2]) for it in items if it[0] == "measure"]

    def schedule(group_list):
        for group in group_list:
            if use_conveyor and all(hub in (b.u, b.v) for b in group):
                if all(b.kind == "zz" and _mid_theta(b.mid) is not None for b in group):
                    if mach.conveyor(group):
                        continue
            mach.matching(group)

    todo = list(groups)
    idx = 0
    while idx < len(items):
        item = items[idx]
        if item[0] == "blk":
            schedule(todo.pop(0) if todo else [])
            while idx < len(items) and items[idx][0] == "blk":
                idx += 1
            continue
        if item[0] == "1q":
            st.queue_1q(item[1], item[2])
        idx += 1
    mach.storage_phase()
    st.flush_all()
    for q, c in measures:
        st.out.measure(q, c)
    return st


def _looks_like_hub(items, n):
    blocks = [it[1] for it in items if it[0] == "blk"]
    if not blocks:
        return None
    neigh: dict[int, set] = {}
    used = set()
    for blk in blocks:
        neigh.setdefault(blk.u, set()).add(blk.v)
        neigh.setdefault(blk.v, set()).add(blk.u)
        used.update(blk.qubits())
    if len(used) != n:
        return None
    hub = max(neigh, key=lambda q: (len(neigh[q]), q))
    if len(neigh[hub]) * 4 < n - 1:
        return None
    return hub


def _parked_qubit(items, n):
    """The least connected qubit, used as a stand-in hub for circuits with no hub."""
    blocks = [it[1] for it in items if it[0] == "blk"]
    deg: dict[int, int] = {}
    for blk in blocks:
        deg[blk.u] = deg.get(blk.u, 0) + 1
        deg[blk.v] = deg.get(blk.v, 0) + 1
    return min(range(n), key=lambda q: (deg.get(q, 0), -q))


# ---------------------------------------------------------------- entry point


def compile_circuit(circuit: QuantumCircuit, architecture_spec: Any = None):
    n = circuit.num_qubits
    items = parse_blocks(circuit)
    st = None
    if items is not None and n <= 50:
        hub = _looks_like_hub(items, n)
        use_conveyor = hub is not None
        if hub is None:
            hub = _parked_qubit(items, n)
        blocks = [it[1] for it in items if it[0] == "blk"]
        runs = _runs(items)
        try:
            groups = [plan_groups(run, hub) for run in runs]
        except Exception:
            groups = None
        if groups is not None:
            lay = plan_layout(n, hub, blocks, [g for gs in groups for g in gs], use_conveyor)
            if lay is not None:
                try:
                    st = _compile_lanes(circuit, items, lay, groups)
                except AssertionError:
                    _STATS["fell_back"] += 1
                    if _DEBUG:
                        raise
                    st = None
    if st is None:
        _STATS["generic"] += 1
        st = _compile_generic(circuit, architecture_spec)
    else:
        _STATS["lanes"] += 1
    return st.out, Architecture(st.loading, spec=architecture_spec)


# ---------------------------------------------------------------- manual check


def _main(argv=None):  # pragma: no cover - developer helper
    """`python compiler.py circuit.qasm` prints the movement metrics and the equivalence verdict."""
    import sys
    from pathlib import Path

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python compiler.py <circuit.qasm> [...]")
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
    from qiskit import QuantumCircuit

    from pushquantum_validator.verify import compute_metrics, verify_equivalence, verify_movement

    for path in argv:
        ref = QuantumCircuit.from_qasm_file(path)
        out, arch = compile_circuit(ref, None)
        movement = verify_movement(out, arch.initial_loading, log_file="/dev/null")
        print(f"{Path(path).name}: {compute_metrics(out, movement)}")
        try:
            result = verify_equivalence(out, ref)
            print("   equivalence:", "skipped" if result is None else result.equivalence)
        except Exception as error:  # noqa: BLE001 - the helper only reports
            print("   equivalence: FAIL", type(error).__name__, error)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
