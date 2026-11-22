"""Exact MILP for a window of the L3 problem (scipy.optimize.milp / HiGHS).

Window: start configuration S (all atoms), T move slots, pulses at given state indices, end
configuration E (all atoms) or free. Active atoms move; fixed atoms stay (rectangles must avoid
them). Every legal move of the validator rules is representable, except squeezing empty picked
lines (all picked lines get strictly increasing targets).

State s = 0..T, slot t moves state t -> t+1.
Variables (per active atom q, state s): one-hot row rz[q,s,y] (y in rows), one-hot col cz[q,s,x].
Per slot t: act, picked rows pr[y], picked cols pc[x] (contiguous), row map R[y], col map C[x]
(integers, strictly increasing over picked lines), source/target zone bits, atom-picked bits,
slot cost d[t] >= |displacement| of every picked atom (polygonal norm, 16 directions; exact cost
recomputed afterwards).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import model as M


def zx(x):
    return 2 * (x // 2) + 0.3 * (x % 2)


class LP:
    def __init__(self):
        self.lb, self.ub, self.integ, self.c = [], [], [], []
        self.rows, self.cols, self.vals, self.clo, self.chi = [], [], [], [], []
        self.nc = 0

    def var(self, lb=0.0, ub=1.0, integ=1, cost=0.0):
        self.lb.append(lb)
        self.ub.append(ub)
        self.integ.append(integ)
        self.c.append(cost)
        return len(self.lb) - 1

    def con(self, terms, lo=-np.inf, hi=np.inf):
        for v, a in terms:
            self.rows.append(self.nc)
            self.cols.append(v)
            self.vals.append(a)
        self.clo.append(lo)
        self.chi.append(hi)
        self.nc += 1

    def solve(self, time_limit=60, gap=1e-4, verbose=False):
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=(self.nc, len(self.lb))).tocsr()
        res = milp(c=np.array(self.c), constraints=LinearConstraint(A, self.clo, self.chi),
                   integrality=np.array(self.integ), bounds=Bounds(self.lb, self.ub),
                   options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": verbose})
        return res


def solve_window(start, pulses_at, members_at, pairs_at, need_storage, end=None, T=6, active=None,
                 rows=range(4, 16), time_limit=60, ub=None, verbose=False, gap=0.005):
    """start: {q: site}; pulses_at: list of state indices with a pulse; members_at/pairs_at: per
    pulse index; need_storage: list of (q, s_from, s_to): q must be in storage at some state in
    [s_from, s_to]; end: {q: site} or None. Returns (moves list of {q: target}, cost) or None."""
    t0 = time.time()
    atoms = sorted(start)
    act = sorted(active) if active is not None else atoms
    fixed = {q: start[q] for q in atoms if q not in act}
    rows = sorted(set(rows) | {start[q][1] for q in act} | ({end[q][1] for q in act} if end else set())
                  | {y for q, (_, y) in fixed.items() if False})
    Y, X = rows, list(range(10))
    S = T + 1
    lp = LP()
    BIG_R, BIG_C = 20, 10
    rz = {(q, s, y): lp.var() for q in act for s in range(S) for y in Y}
    cz = {(q, s, x): lp.var() for q in act for s in range(S) for x in X}
    for q in act:
        for s in range(S):
            lp.con([(rz[q, s, y], 1) for y in Y], 1, 1)
            lp.con([(cz[q, s, x], 1) for x in X], 1, 1)
    # start / end
    for q in act:
        x0, y0 = start[q]
        lp.lb[rz[q, 0, y0]] = 1
        lp.lb[cz[q, 0, x0]] = 1
        if end is not None:
            x1, y1 = end[q]
            lp.lb[rz[q, T, y1]] = 1
            lp.lb[cz[q, T, x1]] = 1
    # occupancy (continuous product vars per site) incl. fixed atoms
    fixed_sites = set(fixed.values())
    for s in range(S):
        for y in Y:
            for x in X:
                terms = []
                for q in act:
                    w = lp.var(0, 1, 0)
                    lp.con([(w, 1), (rz[q, s, y], -1), (cz[q, s, x], -1)], -1, np.inf)
                    terms.append((w, 1))
                hi = 0 if (x, y) in fixed_sites else 1
                lp.con(terms, -np.inf, hi)
    # in-zone indicator and physical x
    inz = {}
    px = {}
    for q in act:
        for s in range(S):
            inz[q, s] = [(rz[q, s, y], 1) for y in Y if y >= 10]
            # px = sum_x x*cz - sum_x (x - zx(x)) * (cz AND inzone)
            terms = [(cz[q, s, x], float(x)) for x in X]
            iz = lp.var(0, 1, 0)
            lp.con([(iz, 1)] + [(v, -a) for v, a in inz[q, s]], 0, 0)
            for x in X:
                if x - zx(x) > 1e-9:
                    v = lp.var(0, 1, 0)
                    lp.con([(v, 1), (cz[q, s, x], -1), (iz, -1)], -1, np.inf)
                    lp.con([(v, 1), (cz[q, s, x], -1)], -np.inf, 0)
                    lp.con([(v, 1), (iz, -1)], -np.inf, 0)
                    terms.append((v, -(x - zx(x))))
            px[q, s] = terms
    dirs = [(math.cos(2 * math.pi * j / 16), math.sin(2 * math.pi * j / 16)) for j in range(16)]
    dvars = []
    pick = {}
    acts = []
    for t in range(T):
        a = lp.var()
        acts.append(a)
        if t:
            lp.con([(acts[t - 1], 1), (a, -1)], 0, np.inf)        # idle slots last
        pr = {y: lp.var(0, 1 if y in Y else 0) for y in range(20)}
        pc = {x: lp.var() for x in X}
        sr = {y: lp.var() for y in range(20)}
        sc = {x: lp.var() for x in X}
        for y in range(20):
            lp.con([(pr[y], 1), (a, -1)], -np.inf, 0)
            lp.con([(sr[y], 1), (pr[y], -1)] + ([(pr[y - 1], 1)] if y else []), 0, np.inf)
        lp.con([(sr[y], 1) for y in range(20)], -np.inf, 1)
        lp.con([(pr[y], 1) for y in range(20)] + [(a, -1)], 0, np.inf)
        for x in X:
            lp.con([(pc[x], 1), (a, -1)], -np.inf, 0)
            lp.con([(sc[x], 1), (pc[x], -1)] + ([(pc[x - 1], 1)] if x else []), 0, np.inf)
        lp.con([(sc[x], 1) for x in X], -np.inf, 1)
        lp.con([(pc[x], 1) for x in X] + [(a, -1)], 0, np.inf)
        zs, zt = lp.var(), lp.var()
        for y in range(20):
            if y >= 10:
                lp.con([(pr[y], 1), (zs, -1)], -np.inf, 0)
            else:
                lp.con([(pr[y], 1), (zs, 1)], -np.inf, 1)
        R = {y: lp.var(0, 19, 1) for y in range(20)}
        C = {x: lp.var(0, 9, 1) for x in X}
        for y in range(20):
            lp.con([(R[y], 1), (pr[y], -19)], -np.inf, 0)        # unused map entries pinned
        for x in X:
            lp.con([(C[x], 1), (pc[x], -9)], -np.inf, 0)
        slot_picks = []
        for y in range(19):
            # R[y+1] >= R[y] + 1 if both picked
            lp.con([(R[y + 1], 1), (R[y], -1), (pr[y], -BIG_R), (pr[y + 1], -BIG_R)], 1 - 2 * BIG_R, np.inf)
        for x in range(9):
            lp.con([(C[x + 1], 1), (C[x], -1), (pc[x], -BIG_C), (pc[x + 1], -BIG_C)], 1 - 2 * BIG_C, np.inf)
        for y in range(20):
            # picked row target in target zone: zt=1 -> R>=10 ; zt=0 -> R<=9
            lp.con([(R[y], 1), (zt, -BIG_R), (pr[y], -BIG_R)], 10 - 2 * BIG_R, np.inf)
            lp.con([(R[y], 1), (zt, -BIG_R), (pr[y], BIG_R)], -np.inf, 9 + BIG_R)
        for q, (xf, yf) in fixed.items():
            lp.con([(pr[yf], 1), (pc[xf], 1)], -np.inf, 1)
        d = lp.var(0, 30, 0, cost=1.0)
        dvars.append(d)
        for q in act:
            ra, cb, p = lp.var(), lp.var(), lp.var()
            pick[q, t] = p
            slot_picks.append(p)
            for y in Y:
                lp.con([(ra, 1), (rz[q, t, y], -1), (pr[y], -1)], -1, np.inf)
                lp.con([(ra, 1), (rz[q, t, y], 1), (pr[y], -1)], -np.inf, 1)
            for y in range(20):
                if y not in Y:
                    continue
            for x in X:
                lp.con([(cb, 1), (cz[q, t, x], -1), (pc[x], -1)], -1, np.inf)
                lp.con([(cb, 1), (cz[q, t, x], 1), (pc[x], -1)], -np.inf, 1)
            lp.con([(p, 1), (ra, -1), (cb, -1)], -1, np.inf)
            lp.con([(p, 1), (ra, -1)], -np.inf, 0)
            lp.con([(p, 1), (cb, -1)], -np.inf, 0)
            # not picked -> same one-hot
            for y in Y:
                lp.con([(rz[q, t + 1, y], 1), (rz[q, t, y], -1), (p, -1)], -np.inf, 0)
                lp.con([(rz[q, t + 1, y], 1), (rz[q, t, y], -1), (p, 1)], 0, np.inf)
            for x in X:
                lp.con([(cz[q, t + 1, x], 1), (cz[q, t, x], -1), (p, -1)], -np.inf, 0)
                lp.con([(cz[q, t + 1, x], 1), (cz[q, t, x], -1), (p, 1)], 0, np.inf)
            # picked -> new row = R[old row], new col = C[old col]
            ynew = [(rz[q, t + 1, y], float(y)) for y in Y]
            xnew = [(cz[q, t + 1, x], float(x)) for x in X]
            for y in Y:
                lp.con(ynew + [(R[y], -1), (p, BIG_R), (rz[q, t, y], BIG_R)], -np.inf, 2 * BIG_R)
                lp.con(ynew + [(R[y], -1), (p, -BIG_R), (rz[q, t, y], -BIG_R)], -2 * BIG_R, np.inf)
            for x in X:
                lp.con(xnew + [(C[x], -1), (p, BIG_C), (cz[q, t, x], BIG_C)], -np.inf, 2 * BIG_C)
                lp.con(xnew + [(C[x], -1), (p, -BIG_C), (cz[q, t, x], -BIG_C)], -2 * BIG_C, np.inf)
            # atom zone == source zone if picked; new zone == target zone if picked
            lp.con([(v, 1) for v, _ in inz[q, t]] + [(zs, -1), (p, 1)], -np.inf, 1)
            lp.con([(v, 1) for v, _ in inz[q, t]] + [(zs, -1), (p, -1)], -1, np.inf)
            lp.con([(v, 1) for v, _ in inz[q, t + 1]] + [(zt, -1), (p, 1)], -np.inf, 1)
            lp.con([(v, 1) for v, _ in inz[q, t + 1]] + [(zt, -1), (p, -1)], -1, np.inf)
            # cost: d >= a*dx + b*dy - M(1-p)
            yold = [(rz[q, t, y], float(y)) for y in Y]
            for (ca, sb) in dirs:
                terms = [(d, 1)]
                terms += [(v, -ca * w) for v, w in px[q, t + 1]] + [(v, ca * w) for v, w in px[q, t]]
                terms += [(v, -sb * w) for v, w in ynew] + [(v, sb * w) for v, w in yold]
                terms.append((p, -30))
                lp.con(terms, -30, np.inf)
        lp.con([(v, 1) for v in slot_picks] + [(a, -1)], 0, np.inf)                 # an active slot moves an atom
    # pulses
    for k, s in enumerate(pulses_at):
        mem = members_at[k]
        for q in act:
            iz = inz[q, s]
            if q in mem:
                lp.con(iz, 1, 1)
            else:
                lp.con(iz, 0, 0)
        for q, site in fixed.items():
            if (q in mem) != M.in_zone(site):
                return None
        for (a_, b_) in pairs_at[k]:
            if a_ not in act or b_ not in act:
                if a_ in fixed and b_ in fixed:
                    continue
                return None
            for y in Y:
                lp.con([(rz[a_, s, y], 1), (rz[b_, s, y], -1)], 0, 0)
            o = lp.var()
            lp.con([(cz[a_, s, x], 1) for x in X if x % 2 == 0] + [(o, -1)], 0, 0)
            for j in range(5):
                lp.con([(cz[b_, s, 2 * j + 1], 1), (cz[a_, s, 2 * j], -1), (o, 1)], -np.inf, 1)
                lp.con([(cz[b_, s, 2 * j + 1], 1), (cz[a_, s, 2 * j], -1), (o, -1)], -1, np.inf)
                lp.con([(cz[b_, s, 2 * j], 1), (cz[a_, s, 2 * j + 1], -1), (o, -1)], -np.inf, 0)
                lp.con([(cz[b_, s, 2 * j], 1), (cz[a_, s, 2 * j + 1], -1), (o, 1)], 0, np.inf)
    for q, s0, s1 in need_storage:
        lp.con([(rz[q, s, y], 1) for s in range(s0, s1 + 1) for y in Y if y <= 9], 1, np.inf)
    # symmetry/tightening: idle slots last within the window is not imposed (pulses fix slots)
    if ub is not None:
        lp.con([(d, 1) for d in dvars], -np.inf, ub)
    build = time.time() - t0
    res = lp.solve(time_limit=time_limit, gap=gap, verbose=verbose)
    if res.x is None:
        return {"moves": None, "msg": res.message, "status": res.status, "build_s": build,
                "solve_s": time.time() - t0 - build, "obj": None}
    xs = res.x
    # decode positions per state and moves
    pos = []
    for s in range(S):
        cfg = dict(fixed)
        for q in act:
            y = max(Y, key=lambda y: xs[rz[q, s, y]])
            x = max(X, key=lambda x: xs[cz[q, s, x]])
            cfg[q] = (x, y)
        pos.append(cfg)
    moves = []
    for t in range(T):
        g = {q: pos[t + 1][q] for q in act if xs[pick[q, t]] > 0.5}
        moves.append(g)
    return {"moves": moves, "pos": pos, "obj": res.fun, "status": res.status, "msg": res.message,
            "build_s": build, "solve_s": time.time() - t0 - build, "bound": getattr(res, "mip_dual_bound", None)}
