"""postopt: generic move post-optimizer for any compiler's output (validator v5 rules).

Between two CZ layers ("gaps") only three things matter: where every atom is at the first layer,
where it is at the next layer, and the 1q gates applied in between (they need the atom in
storage). Each gap is re-planned from scratch and the new plan is kept only if it is cheaper:

  1. drop: atoms in the zone that owe 1q gates (or must leave the zone) go to storage, as one block
     (uniform row shift, smallest that fits) where possible;
  2. the gap's 1q gates (per-qubit order kept);
  3. every atom moves to its end position, grouped into legal rectangles.

Gates, their per-qubit order and the CZ layers are untouched, so depth and equivalence are
unchanged. The result is replayed with the validator's rules (single-zone moves, idle rule, CZ on
pairs, 1q gates in storage) before it is returned; on any doubt the input is returned unchanged.

Self-contained (stdlib + qiskit), appended to a component by `bench/make_postopt.py`.
"""

from __future__ import annotations

import itertools
import math

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction


class _PoMove(Instruction):
    def __init__(self, rows, cols, new_rows, new_cols, num_qubits):
        super().__init__("move", num_qubits, 0, [tuple(rows), tuple(cols), tuple(new_rows), tuple(new_cols)])

    def validate_parameter(self, parameter):
        return parameter

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


class _PoGeo:
    def __init__(self, spec):
        zones = (spec or {}).get("zones") if isinstance(spec, dict) else None
        if not zones:
            zones = {
                "storage": {"x": [0, 9], "y": [0, 9], "spacing": 1.0, "operations": ["rx", "rz"]},
                "entangling": {"x": [0, 9], "y": [10, 19], "spacing": 1.0,
                               "pairs": {"axis": "x", "distance": 0.3}, "operations": ["cz"]},
            }
        self.zone_of, self.phys = {}, {}
        self.ent = self.sto = None
        for name, z in reversed(list(zones.items())):
            (x0, x1), (y0, y1) = z["x"], z["y"]
            sp = float(z.get("spacing", 1.0))
            pairs = z.get("pairs")
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    self.zone_of[(x, y)] = name
                    px = self._c(x, x0, sp, pairs, "x")
                    py = self._c(y, y0, sp, pairs, "y")
                    self.phys[(x, y)] = (px, py)
        for name, z in zones.items():
            if z.get("pairs"):
                self.ent = name
                self.ex0, self.ey0, self.ey1 = z["x"][0], z["y"][0], z["y"][1]
            elif "rx" in z.get("operations", []):
                self.sto = name
                self.sy0, self.sy1 = z["y"]
        self.storage_sites = [s for s, n in self.zone_of.items() if n == self.sto]

    @staticmethod
    def _c(line, start, sp, pairs, axis):
        if not pairs or pairs.get("axis") != axis:
            return line * sp
        p, s = divmod(line - start, 2)
        return start * sp + p * 2 * sp + s * float(pairs["distance"])

    def dist(self, a, b):
        (ax, ay), (bx, by) = self.phys[a], self.phys[b]
        return math.hypot(ax - bx, ay - by)

    def in_ent(self, s):
        return self.zone_of.get(s) == self.ent

    def in_sto(self, s):
        return self.zone_of.get(s) == self.sto

    def sibling(self, s):
        x, y = s
        return (x + 1, y) if (x - self.ex0) % 2 == 0 else (x - 1, y)


class _PoSt:
    def __init__(self, geo, pos):
        self.geo, self.pos = geo, dict(pos)
        self.occ = {s: q for q, s in self.pos.items()}

    def copy(self):
        st = _PoSt.__new__(_PoSt)
        st.geo, st.pos, st.occ = self.geo, dict(self.pos), dict(self.occ)
        return st

    def apply(self, group):
        """group = [(q, target)]; validator semantics; returns ((rows, cols, nr, nc, qs), dist)."""
        g = self.geo
        src = {q: self.pos[q] for q, _ in group}
        tgt = dict(group)
        xs = [s[0] for s in src.values()]
        ys = [s[1] for s in src.values()]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mem = set(src)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                o = self.occ.get((x, y))
                if o is not None and o not in mem:
                    raise ValueError("ghost")
        rmap, cmap = {}, {}
        for q in mem:
            (sx, sy), (tx, ty) = src[q], tgt[q]
            if rmap.setdefault(sy, ty) != ty or cmap.setdefault(sx, tx) != tx:
                raise ValueError("not a product map")
        for mp in (rmap, cmap):
            ks = sorted(mp)
            if any(mp[a] >= mp[b] for a, b in itertools.pairwise(ks)):
                raise ValueError("order")
        if len({g.zone_of[s] for s in src.values()}) != 1 or len({g.zone_of.get(t) for t in tgt.values()}) != 1:
            raise ValueError("zones")
        if None in {g.zone_of.get(t) for t in tgt.values()}:
            raise ValueError("outside")
        for q in mem:
            o = self.occ.get(tgt[q])
            if o is not None and o not in mem:
                raise ValueError("occupied")
        d = max(g.dist(src[q], tgt[q]) for q in mem)
        for q in mem:
            del self.occ[src[q]]
        for q in mem:
            self.pos[q] = tgt[q]
            self.occ[tgt[q]] = q
        return (tuple(range(y0, y1 + 1)), tuple(range(x0, x1 + 1)),
                tuple(rmap[y] for y in sorted(rmap)), tuple(cmap[x] for x in sorted(cmap)), sorted(mem)), d


def _po_sgn(v):
    return (v > 0) - (v < 0)


def _po_compat(a, b):
    (_, (ax, ay), (atx, aty)), (_, (bx, by), (btx, bty)) = a, b
    return _po_sgn(ax - bx) == _po_sgn(atx - btx) and _po_sgn(ay - by) == _po_sgn(aty - bty)


def _po_group(st, transfers):
    """Grouped legal moves for transfers [(q, src, dst)] (largest displacement seeds a group)."""
    g = st.geo
    todo = {q: (q, s, d) for q, s, d in transfers if s != d}
    out = []
    guard = 0
    while todo:
        guard += 1
        if guard > 6 * len(transfers) + 30:
            raise ValueError("no progress")
        cands = sorted(todo.values(), key=lambda t: -g.dist(t[1], t[2]))
        seed = next((t for t in cands if st.occ.get(t[2]) is None), None)
        if seed is None:
            t = cands[0]
            q = st.occ[t[2]]
            if q not in todo:
                raise ValueError("target held by a stationary atom")
            pool = [s for s in g.zone_of if g.zone_of[s] == g.zone_of[st.pos[q]] and s not in st.occ]
            spot = min(pool, key=lambda s: g.dist(st.pos[q], s))
            out.append(st.apply([(q, spot)]))
            todo[q] = (q, spot, todo[q][2])
            continue
        zs, zt = g.zone_of[seed[1]], g.zone_of[seed[2]]

        def fits(t, grp):
            return g.zone_of[t[1]] == zs and g.zone_of[t[2]] == zt and all(_po_compat(t, u) for u in grp)

        grp = [seed] + [t for t in cands if t is not seed]
        grp = [seed] + [t for t in grp[1:] if fits(t, [seed])]
        keep = [seed]
        for t in grp[1:]:
            if fits(t, keep):
                keep.append(t)
        grp, excl = keep, set()
        sx, sy = seed[1]
        while True:
            mem = {u[0] for u in grp}
            xs = [u[1][0] for u in grp]
            ys = [u[1][1] for u in grp]
            ghosts = [st.occ[(x, y)] for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)
                      if (x, y) in st.occ and st.occ[(x, y)] not in mem]
            held = [u for u in grp if st.occ.get(u[2]) is not None and st.occ.get(u[2]) not in mem]
            if not ghosts and not held:
                break
            pulled = False
            for q in ghosts:
                if q in todo and q not in excl and fits(todo[q], grp):
                    grp.append(todo[q])
                    pulled = True
            if pulled:
                continue
            if held:
                excl.update(u[0] for u in held)
                grp = [u for u in grp if u not in held]
                continue
            far = max(grp[1:], key=lambda u: abs(u[1][0] - sx) + abs(u[1][1] - sy))
            excl.add(far[0])
            grp.remove(far)
        out.append(st.apply([(u[0], u[2]) for u in grp]))
        for u in grp:
            del todo[u[0]]
    return out


def _po_block_drop(st, qs):
    """Targets for dropping zone atoms `qs` as one block: zone row r -> storage row r - shift (same
    column), smallest shift with all targets free. None if no shift fits."""
    g = st.geo
    if not qs:
        return {}
    for shift in range(1, g.ey1 - g.sy0 + 1):
        tg = {}
        ok = True
        for q in qs:
            x, y = st.pos[q]
            t = (x, y - shift)
            if not g.in_sto(t) or (t in st.occ and st.occ[t] not in qs):
                ok = False
                break
            tg[q] = t
        if ok and len(set(tg.values())) == len(tg):
            return tg
    return None


def _po_replan_gap(geo, start, end, gates, n):
    """Cheapest of: drop only the atoms that must leave the zone, or drop the whole zone as a block."""
    best = None
    for whole in (False, True):
        try:
            r = _po_replan_gap1(geo, start, end, gates, n, whole)
        except (ValueError, KeyError):
            r = None
        if r is not None and (best is None or r[1] < best[1]):
            best = r
    return best


def _po_replan_gap1(geo, start, end, gates, n, whole):
    """New item list for one gap, or None. gates: list of (instruction, qubits) 1q gates in order."""
    st = _PoSt(geo, start)
    owe = {q for _, qs in gates for q in qs}
    items, cost = [], 0.0
    drop = [q for q in range(n) if geo.in_ent(st.pos[q]) and (whole or q in owe or st.pos[q] != end[q])]
    tg = _po_block_drop(st, drop)
    if tg is None:
        tg = {}
        taken = set()
        for q in drop:
            x, _ = st.pos[q]
            col = [(x, y) for y in range(geo.sy1, geo.sy0 - 1, -1) if (x, y) not in st.occ and (x, y) not in taken]
            if not col:
                return None
            tg[q] = col[0]
            taken.add(col[0])
    # atoms whose end is in storage go there directly if that site is free now
    for q in drop:
        e = end[q]
        if geo.in_sto(e) and q not in owe and e not in st.occ and e not in tg.values():
            tg[q] = e
    for params, d in _po_group(st, [(q, st.pos[q], t) for q, t in tg.items()]):
        items.append(("move", params))
        cost += d
    for ins, qs in gates:
        if not all(geo.in_sto(st.pos[q]) for q in qs):
            return None
        items.append(("gate", ins))
    for params, d in _po_group(st, [(q, st.pos[q], end[q]) for q in range(n) if st.pos[q] != end[q]]):
        items.append(("move", params))
        cost += d
    if any(st.pos[q] != end[q] for q in range(n)):
        return None
    return items, cost


def _po_replay_ok(geo, loading, circuit) -> bool:
    """Validator rules: legal moves, 1q in storage, cz on pairs, idle rule, equal pulses per layer."""
    st = _PoSt(geo, loading)
    qi = {b: i for i, b in enumerate(circuit.qubits)}
    layer: dict = {}

    def close():
        if not layer:
            return True
        if len(set(layer.values())) != 1:
            return False
        active = {q for k in layer for q in k}
        for q, s in st.pos.items():
            if geo.in_ent(s) and q not in active:
                return False
        return True

    for ins in circuit.data:
        op = ins.operation
        qs = [qi[q] for q in ins.qubits]
        if op.name == "move":
            if not close():
                return False
            layer.clear()
            rows, cols, nr, nc = op.params
            occ_rows = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
            occ_cols = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
            if len(nr) == len(rows):
                occ_rows = list(rows)
            if len(nc) == len(cols):
                occ_cols = list(cols)
            rm, cm = dict(zip(occ_rows, nr)), dict(zip(occ_cols, nc))
            try:
                st.apply([(q, (cm[st.pos[q][0]], rm[st.pos[q][1]])) for q in qs])
            except (ValueError, KeyError):
                return False
        elif op.name in ("rx", "rz"):
            if not geo.in_sto(st.pos[qs[0]]):
                return False
        elif op.name == "cz":
            a, b = (st.pos[q] for q in qs)
            if not (geo.in_ent(a) and geo.sibling(a) == b):
                return False
            k = frozenset(qs)
            layer[k] = layer.get(k, 0) + 1
    return close()


def postoptimize(circuit: QuantumCircuit, loading: dict, spec=None) -> QuantumCircuit:
    try:
        geo = _PoGeo(spec)
        n = circuit.num_qubits
        qi = {b: i for i, b in enumerate(circuit.qubits)}
        data = list(circuit.data)
        # positions before every instruction (replay)
        st = _PoSt(geo, loading)
        layers = []  # (start_index, end_index) of each CZ layer (consecutive czs between moves)
        pos_at = []
        cur = None
        for idx, ins in enumerate(data):
            op = ins.operation
            qs = [qi[q] for q in ins.qubits]
            if op.name == "cz":
                if cur is None:
                    cur = [idx, idx]
                    pos_at.append(dict(st.pos))
                else:
                    cur[1] = idx
                continue
            if op.name == "move":
                if cur is not None:
                    layers.append(tuple(cur))
                    cur = None
                rows, cols, nr, nc = op.params
                occ_rows = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
                occ_cols = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
                if len(nr) == len(rows):
                    occ_rows = list(rows)
                if len(nc) == len(cols):
                    occ_cols = list(cols)
                rm, cm = dict(zip(occ_rows, nr)), dict(zip(occ_cols, nc))
                st.apply([(q, (cm[st.pos[q][0]], rm[st.pos[q][1]])) for q in qs])
        if cur is not None:
            layers.append(tuple(cur))
        if len(layers) < 2:
            return circuit
        out = circuit.copy_empty_like()
        prev_end = -1
        # gap before the first layer and after the last layer are kept as they are
        keep_prefix = data[: layers[0][0]]
        for ins in keep_prefix:
            out.append(ins)
        for k in range(len(layers)):
            s, e = layers[k]
            for ins in data[s: e + 1]:
                out.append(ins)
            if k + 1 == len(layers):
                for ins in data[e + 1:]:
                    out.append(ins)
                break
            s2 = layers[k + 1][0]
            gap = data[e + 1: s2]
            moves = [ins for ins in gap if ins.operation.name == "move"]
            others = [ins for ins in gap if ins.operation.name != "move"]
            if any(ins.operation.name not in ("rx", "rz", "barrier") for ins in others):
                for ins in gap:
                    out.append(ins)
                continue
            # original cost of the gap
            st0 = _PoSt(geo, pos_at[k])
            old = 0.0
            ok = True
            for ins in moves:
                qs = [qi[q] for q in ins.qubits]
                rows, cols, nr, nc = ins.operation.params
                occ_rows = [y for y in rows if any(st0.pos[q][1] == y for q in qs)]
                occ_cols = [x for x in cols if any(st0.pos[q][0] == x for q in qs)]
                if len(nr) == len(rows):
                    occ_rows = list(rows)
                if len(nc) == len(cols):
                    occ_cols = list(cols)
                rm, cm = dict(zip(occ_rows, nr)), dict(zip(occ_cols, nc))
                try:
                    _, d = st0.apply([(q, (cm[st0.pos[q][0]], rm[st0.pos[q][1]])) for q in qs])
                except (ValueError, KeyError):
                    ok = False
                    break
                old += d
            gates = [(ins, [qi[q] for q in ins.qubits]) for ins in others if ins.operation.name in ("rx", "rz")]
            new = _po_replan_gap(geo, pos_at[k], pos_at[k + 1], gates, n) if ok else None
            if new is None or new[1] >= old - 1e-9:
                for ins in gap:
                    out.append(ins)
                continue
            for kind, val in new[0]:
                if kind == "gate":
                    out.append(val)
                else:
                    rows, cols, nr, nc, qs = val
                    out.append(_PoMove(rows, cols, nr, nc, len(qs)), [out.qubits[q] for q in qs])
        if not _po_replay_ok(geo, loading, out):
            return circuit
        return out
    except Exception:  # noqa: BLE001 - never break a valid compilation
        return circuit
