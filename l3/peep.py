#!/usr/bin/env python3
"""Peephole optimiser on a move plan (fast, MILP-free). Operators found by the MIP-LNS:

  P1 merge     : move j joins move i (i < j) or move i joins move j
  P2 trim      : an atom that rides a down+up round trip it does not need leaves both moves
  P3 waypoint  : an intermediate site of an atom (target of one move, source of a later one) is
                 re-chosen among nearby free sites
  P4 drop-atom : an atom is removed from a move and given its own move (enables other merges)

Every candidate rewrite of a gap is replayed exactly: legal moves, same end configuration, and the
required storage visits (B_k). Accept if the gap gets cheaper. Repeat until no operator helps.

    python l3/peep.py <cached pkl> [more pkls]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chain as C  # noqa: E402
import model as M  # noqa: E402


def replay(start, moves, need, end):
    """Exact cost of a gap, or None if illegal / wrong end / missing storage visit."""
    st = M.State(start)
    seen = {q for q in need if not M.in_zone(st.pos[q])}
    c = 0.0
    try:
        for g in moves:
            if not g:
                continue
            c += st.apply(g)[1]
            for q in need:
                if q not in seen and not M.in_zone(st.pos[q]):
                    seen.add(q)
    except M.MoveError:
        return None
    if st.pos != end or len(seen) != len(need):
        return None
    return c


def gap_need(sch, k, start, end, B):
    if k < sch.K:
        return set(B[k])
    fo = sch.final_owe()
    return {q for q in fo if M.in_zone(start[q]) and M.in_zone(end[q])}


LAST_EVALS = [0]


def optimise_gap(start, moves, need, end, max_span=8, budget=4000, stay_op=False):
    """Greedy first-improvement over the operators until no change. Returns (moves, cost);
    the number of replays used is left in LAST_EVALS[0]."""
    moves = [dict(g) for g in moves if g]
    best = replay(start, moves, need, end)
    LAST_EVALS[0] = 0
    if best is None:
        return moves, None
    evals = 0
    improved = True
    while improved and evals < budget:
        improved = False
        n = len(moves)
        # P1 merge
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + max_span)):
                if set(moves[i]) & set(moves[j]):
                    continue
                for into in (i, j):
                    cand = [dict(g) for g in moves]
                    cand[into] = {**moves[i], **moves[j]}
                    del cand[j if into == i else i]
                    evals += 1
                    c = replay(start, cand, need, end)
                    if c is not None and c < best - 1e-9:
                        moves, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # P2 trim: atom q down in move i and back to the same site in move j, not needed
        states = [M.State(start)]
        for g in moves:
            s2 = states[-1].copy()
            s2.apply(g)
            states.append(s2)
        for i in range(n):
            for q, t in moves[i].items():
                if q in need or not (M.in_zone(states[i].pos[q]) != M.in_zone(t)):
                    continue
                for j in range(i + 1, n):
                    if q in moves[j]:
                        if moves[j][q] == states[i].pos[q]:
                            cand = [dict(g) for g in moves]
                            del cand[i][q]
                            del cand[j][q]
                            cand = [g for g in cand if g]
                            evals += 1
                            c = replay(start, cand, need, end)
                            if c is not None and c < best - 1e-9:
                                moves, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # P3 waypoint: atom q targeted by move i and moved again by a later move j
        for i in range(n):
            for q in list(moves[i]):
                j = next((j for j in range(i + 1, n) if q in moves[j]), None)
                if j is None:
                    continue
                cur = moves[i][q]
                occ_all = set()
                for s in states[i:j + 1]:
                    occ_all |= set(s.occ)
                cands = []
                for x in range(10):
                    for y in range(20):
                        s = (x, y)
                        if s == cur or s in occ_all or M.zone_of(s) != M.zone_of(cur):
                            continue
                        dd = M.dist(states[i].pos[q], s) + M.dist(s, moves[j][q])
                        cands.append((dd, s))
                cands.sort()
                for _, s in cands[:12]:
                    cand = [dict(g) for g in moves]
                    cand[i][q] = s
                    evals += 1
                    c = replay(start, cand, need, end)
                    if c is not None and c < best - 1e-9:
                        moves, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # P2b collective trim: move j reverses move i on a subset; drop all unneeded atoms of it
        # at once and send the needed ones to the nearest free storage site
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + max_span)):
                rt = [q for q, t in moves[i].items() if q in moves[j] and moves[j][q] == states[i].pos[q]
                      and M.in_zone(states[i].pos[q]) and not M.in_zone(t)]
                if len(rt) < 2:
                    continue
                cand = [dict(g) for g in moves]
                keep = [q for q in rt if q in need]
                for q in rt:
                    if q not in need:
                        del cand[i][q]
                        del cand[j][q]
                if keep:
                    occ_all = set()
                    for s in states[i:j + 2]:
                        occ_all |= {site for site, a in s.occ.items() if a not in rt}
                    occ_all |= {states[i].pos[a] for a in rt}
                    for q in keep:
                        best_s = min(((M.dist(states[i].pos[q], (x, y)), (x, y)) for x in range(10) for y in range(10)
                                      if (x, y) not in occ_all), default=None)
                        if best_s:
                            cand[i][q] = best_s[1]
                            occ_all.add(best_s[1])
                cand = [g for g in cand if g]
                evals += 1
                c = replay(start, cand, need, end)
                if c is not None and c < best - 1e-9:
                    moves, best, improved = cand, c, True
                    break
            if improved:
                break
        if improved:
            continue
        # P5 ride-along: an atom with a single-atom move at i rides with a translation move j
        # (j < i: rides first, then fixes up; j > i: goes to t - v first, then rides to t)
        for i in range(n):
            if len(moves[i]) != 1:
                continue
            (q, t), = moves[i].items()
            p = states[i].pos[q]
            for j in range(max(0, i - max_span), min(n, i + 1 + max_span)):
                if j == i or q in moves[j]:
                    continue
                vs = {(tt[0] - states[j].pos[a][0], tt[1] - states[j].pos[a][1]) for a, tt in moves[j].items()}
                if len(vs) != 1:
                    continue
                (vx, vy), = vs
                if any(q in moves[h] for h in range(min(i, j) + 1, max(i, j))):
                    continue
                cand = [dict(g) for g in moves]
                if j < i:
                    w = (p[0] + vx, p[1] + vy)
                    if not (0 <= w[0] <= 9 and 0 <= w[1] <= 19):
                        continue
                    cand[j][q] = w
                    cand[i] = {q: t}
                else:
                    w = (t[0] - vx, t[1] - vy)
                    if not (0 <= w[0] <= 9 and 0 <= w[1] <= 19):
                        continue
                    cand[i] = {q: w}
                    cand[j][q] = t
                cand = [g for g in cand if g]
                evals += 1
                c = replay(start, cand, need, end)
                if c is not None and c < best - 1e-9:
                    moves, best, improved = cand, c, True
                    break
            if improved:
                break
        if improved:
            continue
        # P5b staging absorbed: single-atom move i of q, later translation move j also moving q:
        # skip the staging move, ride with j from q's original position, fix up after j
        for i in range(n):
            if len(moves[i]) != 1:
                continue
            (q, t), = moves[i].items()
            p = states[i].pos[q]
            j = next((j for j in range(i + 1, n) if q in moves[j]), None)
            if j is None or len(moves[j]) < 2:
                continue
            vs = {(tt[0] - states[j].pos[a][0], tt[1] - states[j].pos[a][1]) for a, tt in moves[j].items() if a != q}
            if len(vs) != 1:
                continue
            (vx, vy), = vs
            w = (p[0] + vx, p[1] + vy)
            if not (0 <= w[0] <= 9 and 0 <= w[1] <= 19):
                continue
            final = moves[j][q]
            cand = [dict(g) for g in moves]
            cand[j][q] = w
            cand.insert(j + 1, {q: final})
            del cand[i]
            evals += 1
            c = replay(start, cand, need, end)
            if c is not None and c < best - 1e-9:
                moves, best, improved = cand, c, True
                break
        if improved:
            continue
        # P6 general ride-along: single-atom move {q: t} at i becomes a short pre-move {q: w} at i
        # plus q joining a later move j with target t (non-uniform maps allowed; replay decides)
        for i in range(n):
            if len(moves[i]) != 1:
                continue
            (q, t), = moves[i].items()
            p = states[i].pos[q]
            old_i = M.dist(p, t)
            for j in range(i + 1, min(n, i + 1 + max_span)):
                if q in moves[j] or any(q in moves[h] for h in range(i + 1, j)):
                    continue
                tried_here = 0
                for dx in (-2, -1, 0, 1, 2):
                    for dy in (-2, -1, 0, 1, 2):
                        w = (p[0] + dx, p[1] + dy)
                        if w == p or not (0 <= w[0] <= 9 and 0 <= w[1] <= 19) or M.zone_of(w) != M.zone_of(p):
                            continue
                        if M.dist(p, w) >= old_i - 1e-9:
                            continue
                        cand = [dict(g) for g in moves]
                        cand[i] = {q: w}
                        cand[j][q] = t
                        evals += 1
                        tried_here += 1
                        c = replay(start, cand, need, end)
                        if c is not None and c < best - 1e-9:
                            moves, best, improved = cand, c, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # P7 stay (ZAC-style reuse): an atom in the zone at both ends of the gap that does not need
        # storage skips its storage trip; it gets one zone->zone move (or none), optionally via a
        # waypoint, inserted at the best position
        if stay_op:
            zq = [q for q in start if M.in_zone(start[q]) and M.in_zone(end[q]) and q not in need
                  and any(q in g for g in moves) and any(not M.in_zone(st.pos[q]) for st in states)]
            for q in zq:
                base_moves = [{a: b for a, b in g.items() if a != q} for g in moves]
                if start[q] == end[q]:
                    variants = [[]]
                else:
                    variants = [[{q: end[q]}]]
                    x, y = start[q]
                    for w in ((x, y + 1), (x, y + 2)):
                        if w[1] <= 19:
                            variants.append([{q: w}, {q: end[q]}])
                done = False
                for var in variants:
                    positions = range(len(base_moves) + 1)
                    for p1 in positions:
                        for p2 in (positions if len(var) == 2 else [None]):
                            if p2 is not None and p2 < p1:
                                continue
                            cand = [dict(g) for g in base_moves]
                            if var:
                                if p2 is None:
                                    cand.insert(p1, dict(var[0]))
                                else:
                                    cand.insert(p2, dict(var[1]))
                                    cand.insert(p1, dict(var[0]))
                            cand = [g for g in cand if g]
                            evals += 1
                            c = replay(start, cand, need, end)
                            if c is not None and c < best - 1e-9:
                                moves, best, improved, done = cand, c, True, True
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
        if improved:
            continue
        # P8 compose (MILP pattern on relay_l3): atoms moved by move i and again by a later move j
        # go to their move-j targets already in move i (jointly: the column/row maps must stay
        # monotone), and leave move j
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + max_span)):
                common = [q for q in moves[i] if q in moves[j]
                          and not any(q in moves[h] for h in range(i + 1, j))]
                if not common:
                    continue
                subsets = [common] + ([[q] for q in common] if len(common) > 1 else [])
                for sub in subsets:
                    cand = [dict(g) for g in moves]
                    for q in sub:
                        cand[i][q] = moves[j][q]
                        del cand[j][q]
                    cand = [g for g in cand if g]
                    evals += 1
                    c = replay(start, cand, need, end)
                    if c is not None and c < best - 1e-9:
                        moves, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # P9 stack-lift (MILP pattern): two storage->zone moves i < j with disjoint atoms become one
        # move where move i's atoms land one zone row higher, plus a zone fix-up move for them
        for i in range(n):
            if not moves[i] or any(M.in_zone(states[i].pos[q]) or not M.in_zone(t) for q, t in moves[i].items()):
                continue
            for j in range(i + 1, min(n, i + 1 + max_span)):
                if set(moves[i]) & set(moves[j]):
                    continue
                if any(M.in_zone(states[j].pos[q]) or not M.in_zone(t) for q, t in moves[j].items()):
                    continue
                for dy in (1, -1, 2):
                    lifted = {q: (t[0], t[1] + dy) for q, t in moves[i].items()}
                    if any(not (10 <= t[1] <= 19) for t in lifted.values()):
                        continue
                    cand = [dict(g) for g in moves]
                    cand[j] = {**moves[j], **lifted}
                    cand.insert(j + 1, dict(moves[i]))
                    del cand[i]
                    evals += 1
                    c = replay(start, cand, need, end)
                    if c is not None and c < best - 1e-9:
                        moves, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # P10 round-trip retarget (MILP pattern on relay_l3): move i takes a set of zone atoms down and
        # a later move j brings exactly them back to the same sites: shift the storage sites of both
        for i in range(n):
            if len(moves[i]) < 2 or not all(M.in_zone(states[i].pos[q]) and not M.in_zone(t) for q, t in moves[i].items()):
                continue
            for j in range(i + 1, n):
                if set(moves[j]) != set(moves[i]):
                    if set(moves[j]) & set(moves[i]):
                        break
                    continue
                if any(moves[j][q] != states[i].pos[q] for q in moves[i]):
                    break
                for dx, dy in ((0, 1), (0, 2), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1)):
                    down = {q: (t[0] + dx, t[1] + dy) for q, t in moves[i].items()}
                    if any(not (0 <= t[0] <= 9 and 0 <= t[1] <= 9) for t in down.values()):
                        continue
                    cand = [dict(g) for g in moves]
                    cand[i] = down
                    evals += 1
                    c = replay(start, cand, need, end)
                    if c is not None and c < best - 1e-9:
                        moves, best, improved = cand, c, True
                        break
                break
            if improved:
                break
        if improved:
            continue
        # P6b ride-along with an earlier move / map-preimage waypoints: single-atom move {q: t} at i;
        # for another move j (before or after i, q not moved in between) the waypoint w is the
        # preimage of t under move j's row/col maps (or t itself for j < i); q goes to w by its own
        # move (inserted at the right place) and rides move j
        for i in range(n):
            if len(moves[i]) != 1:
                continue
            (q, t), = moves[i].items()
            p = states[i].pos[q]
            for j in range(max(0, i - max_span), min(n, i + 1 + max_span)):
                if j == i or q in moves[j]:
                    continue
                lo, hi = min(i, j), max(i, j)
                if any(q in moves[h] for h in range(lo + 1, hi)):
                    continue
                src = states[j].pos
                rmap = {src[a][1]: b[1] for a, b in moves[j].items()}
                cmap = {src[a][0]: b[0] for a, b in moves[j].items()}
                rinv = {v: k for k, v in rmap.items()}
                cinv = {v: k for k, v in cmap.items()}
                ws = set()
                for wy in ({rinv[t[1]]} if t[1] in rinv else set()) | {t[1] - 1, t[1] + 1, p[1]}:
                    for wx in ({cinv[t[0]]} if t[0] in cinv else set()) | {p[0], t[0]}:
                        ws.add((wx, wy))
                for w in ws:
                    if w == p or not (0 <= w[0] <= 9 and 0 <= w[1] <= 19) or M.zone_of(w) != M.zone_of(p):
                        continue
                    cand = [dict(g) for g in moves]
                    cand[j] = {**moves[j], q: t}
                    if j > i:
                        cand[i] = {q: w}
                    else:
                        del cand[i]
                        cand.insert(j, {q: w})
                    cand = [g for g in cand if g]
                    evals += 1
                    c = replay(start, cand, need, end)
                    if c is not None and c < best - 1e-9:
                        moves, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # P4 split an atom off a multi-atom move into its own move (right after), then hope P1 merges
        for i in range(n):
            if len(moves[i]) < 2:
                continue
            for q in list(moves[i]):
                for pos in (i, i + 1):
                    cand = [dict(g) for g in moves]
                    t = cand[i].pop(q)
                    cand.insert(pos if pos == i else i + 1, {q: t})
                    evals += 1
                    c = replay(start, cand, need, end)
                    if c is not None and c < best - 1e-9:
                        moves, best, improved = cand, c, True
                        break
                if improved:
                    break
            if improved:
                break
    LAST_EVALS[0] = evals
    return moves, best


def peep(path, log=None):
    c, l = M.load_cached(path)
    s = M.extract(c)
    X, gaps = C.incumbent(c, l, s)
    B = C.binding(s)
    t0 = time.time()
    base = new = 0.0
    for k in range(s.K + 1):
        need = gap_need(s, k, X[k], X[k + 1], B)
        old = C.gap_cost(X[k], gaps[k])
        mv, cst = optimise_gap(X[k], gaps[k], need, X[k + 1])
        base += old
        if cst is not None and cst < old - 1e-9:
            gaps[k] = mv
            new += cst
        else:
            new += old
    out, load, d = C.assemble(s, l, gaps)
    return {"file": Path(path).stem, "base": round(base, 3), "peep": round(d, 3), "secs": round(time.time() - t0, 1)}, out, load, s, gaps


if __name__ == "__main__":
    from check import check
    tb = tn = 0
    for p in sys.argv[1:]:
        r, out, load, s, gaps = peep(p)
        v = check(out, load)
        assert M.signature(out) == M.schedule_signature(s)
        tb += r["base"]
        tn += v["distance"]
        print(r, "validated", round(v["distance"], 3), flush=True)
    print("TOTAL", round(tb, 1), round(tn, 1))
