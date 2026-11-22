"""L3 as a chain-structured optimisation problem.

    minimise   sum_k g_k(X_{k-1}, X_k)
    subject to X_k feasible for pulse k  (members on pairs next to their partner, others in storage)

X_k   = configuration (atom -> site) at pulse k; X_{-1} = initial loading, X_K = final configuration.
g_k   = cost of the cheapest legal move sequence X_{k-1} -> X_k in which every atom of
        B_k (in pulses k-1 and k, owing 1q gates) touches storage (inner problem).

This module: extract X and the per-gap moves from any compiled output (incumbent), an inner
solver `transition` (greedy set partitioning into rectangle moves), and assembly back into a
circuit through the checked Emitter.
"""
from __future__ import annotations

import model as M
from group import execute


def parse_moves(circ):
    """Yield ('move', {q: target}) / ('cz', (a, b)) / ('g', op, q) / ('m', ...) in order, using
    validator semantics for the move targets."""
    qi = {b: i for i, b in enumerate(circ.qubits)}
    return qi


def incumbent(circ, loading, sch):
    """From a compiled circuit whose schedule is `sch` (not dummy-cancelled): configurations X
    (len K+2: loading, pulse 0..K-1, final) and per-gap move lists (len K+1) of {q: target}."""
    qi = {b: i for i, b in enumerate(circ.qubits)}
    st = M.State(loading)
    X = [dict(st.pos)]
    gaps = [[]]
    in_layer = False
    for ins in circ.data:
        name = ins.operation.name
        if name == "cz":
            if not in_layer:
                X.append(dict(st.pos))
                in_layer = True
            continue
        if name == "move":
            if in_layer:
                gaps.append([])
                in_layer = False
            qs = [qi[q] for q in ins.qubits]
            rows, cols, nr, nc = ins.operation.params
            occ_r = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
            occ_c = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
            if len(nr) == len(rows):
                occ_r = list(rows)
            if len(nc) == len(cols):
                occ_c = list(cols)
            rm, cm = dict(zip(occ_r, nr)), dict(zip(occ_c, nc))
            g = {q: (cm[st.pos[q][0]], rm[st.pos[q][1]]) for q in qs}
            st.apply(g)
            gaps[-1].append(g)
    if in_layer:
        gaps.append([])
    X.append(dict(st.pos))
    assert len(X) == sch.K + 2 and len(gaps) == sch.K + 1, (len(X), len(gaps), sch.K)
    return X, gaps


def binding(sch):
    """B[k] = atoms in pulses k-1 and k that owe 1q gates before pulse k; B[K] = final owers in
    the zone at the end are handled by the final config being in storage."""
    mem = sch.members()
    owe = sch.owe()
    B = [set() for _ in range(sch.K + 1)]
    for k in range(1, sch.K):
        B[k] = mem[k - 1] & mem[k] & owe[k]
    return B


def gap_cost(start, moves):
    st = M.State(start)
    c = 0.0
    for g in moves:
        c += st.apply(g)[1]
    return c


class _Rec:
    """Minimal emitter stand-in for the grouping engine: records moves on a State."""

    def __init__(self, pos):
        self.st = M.State(pos)
        self.moves = []
        self.cost = 0.0

    def move(self, g):
        if not g:
            return 0.0
        _, c = self.st.apply(g)
        self.moves.append(dict(g))
        self.cost += c
        return c


def _temp_storage(st, q, avoid):
    """Free storage site closest to q (not in avoid)."""
    best = None
    for y in range(9, -1, -1):
        for x in range(10):
            s = (x, y)
            if s in st.occ or s in avoid:
                continue
            c = M.dist(st.pos[q], s)
            if best is None or c < best[0]:
                best = (c, s)
    return best[1]


def transition(start, end, dip, seeds=6):
    """Inner solver: legal moves start -> end with every atom of `dip` touching storage.
    Phases: (1) zone atoms going to storage + dippers (rigid row dip if possible), (2) storage
    rearrangement, (3) zone->zone, (4) storage->zone. Returns (moves, cost)."""
    rec = _Rec(start)
    st = rec.st
    endset = set(end.values())
    dippers = [q for q in dip if M.in_zone(st.pos[q])]
    if dippers and not _rigid_dip(rec, dippers, end):
        tmp = {}
        for q in sorted(dippers, key=lambda q: st.pos[q]):
            tmp[q] = _temp_storage(st, q, endset | set(tmp.values()))
        execute(rec, tmp, seeds=seeds)
    execute(rec, {q: end[q] for q in end if st.pos[q] != end[q]}, seeds=seeds)
    return rec.moves, rec.cost


def _rigid_dip(rec, dippers, end):
    """Shift the zone rows holding the dippers down into free storage rows and back up (only
    when every atom in those rows ends where it is now). Returns True if done."""
    st = rec.st
    rows = sorted({st.pos[q][1] for q in dippers})
    block = [q for q, s in st.pos.items() if M.in_zone(s) and s[1] in rows]
    h = len(rows)
    for top in range(9, h - 2, -1):
        trows = list(range(top - h + 1, top + 1))
        if trows[0] < 0:
            break
        rmap = dict(zip(rows, trows))
        down = {q: (st.pos[q][0], rmap[st.pos[q][1]]) for q in block}
        if st.ok(down) is None:
            continue
        back = {q: st.pos[q] for q in block}
        t = st.copy()
        t.apply(down)
        if t.ok(back) is None:
            continue
        rec.move(down)
        rec.move(back)
        return True
    return False


def assemble(sch, loading, gaps):
    """Replay per-gap move lists through the checked Emitter; returns (circuit, loading, dist)."""
    em = M.Emitter(sch, loading)
    for k in range(sch.K):
        for g in gaps[k]:
            em.move(g)
        em.pulse()
    for g in gaps[sch.K]:
        em.move(g)
    out, load = em.finish()
    return out, load, em.dist
