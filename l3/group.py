"""Rectangle grouping: execute a set of transfers {atom: target} as few, cheap legal moves.

Greedy: repeatedly take the executable transfers (target free now), grow maximal legal groups
from a few seeds, apply the group with the best saving. Cycles (every target occupied by another
mover) are broken by parking one mover on a free temporary site.
"""
from __future__ import annotations

import model as M


def _free_sites(st, zone, near, limit=40):
    """Free sites in `zone` ('s' or 'e') sorted by distance to `near`."""
    ys = range(0, 10) if zone == "s" else range(10, 20)
    out = []
    for y in ys:
        for x in range(10):
            if (x, y) not in st.occ:
                out.append((M.dist((x, y), near), (x, y)))
    out.sort()
    return [s for _, s in out[:limit]]


def execute(em, transfers, seeds=6, order=None):
    """Apply transfers through the emitter `em` (which owns the state). Returns total cost."""
    st = em.st
    rem = {q: t for q, t in transfers.items() if st.pos[q] != t}
    total = 0.0
    guard = 0
    while rem:
        guard += 1
        if guard > 4 * len(transfers) + 50:
            raise M.MoveError("grouping did not converge")
        exe = [q for q, t in rem.items() if t not in st.occ]
        if not exe:
            # cycle: park one mover whose target is held by another mover
            q = next(iter(rem))
            blocker = st.occ[rem[q]]
            tmp = next(s for s in _free_sites(st, M.zone_of(rem[blocker]), rem[blocker]))
            if M.zone_of(tmp) != M.zone_of(st.pos[blocker]) and M.zone_of(st.pos[blocker]) != M.zone_of(rem[blocker]):
                pass
            total += em.move({blocker: tmp})
            continue
        d = {q: M.dist(st.pos[q], rem[q]) for q in exe}
        cand = sorted(exe, key=lambda q: -d[q])
        best = None
        for seed in cand[:seeds]:
            g = {seed: rem[seed]}
            sx, sy = st.pos[seed]
            others = sorted((q for q in exe if q != seed),
                            key=lambda q: abs(st.pos[q][0] - sx) + abs(st.pos[q][1] - sy))
            for q in others:
                g[q] = rem[q]
                if st.ok(g) is None:
                    del g[q]
            res = st.ok(g)
            cost = res[1]
            saving = sum(d[q] for q in g) - cost
            key = (saving, len(g), -cost)
            if best is None or key > best[0]:
                best = (key, g)
        g = best[1]
        total += em.move(g)
        for q in g:
            del rem[q]
    return total
