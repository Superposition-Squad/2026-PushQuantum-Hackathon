"""Workbench planner: realise a fixed pulse schedule with legal moves.

Per gap (before pulse k):
  1. stayers: atoms in the zone that are in pulse k and owe no 1q gates may stay (policy);
  2. everyone else in the zone drops to storage (sites chosen below / near its next use);
  3. pairs get zone slots: anchored at a stayer, else a free slot near the atoms;
  4. zone re-pairs (zone->zone), then lifts (storage->zone), all through the grouping engine.
"""
from __future__ import annotations

import model as M
from group import execute

SLOTS = [(2 * j, y) for y in range(10, 20) for j in range(5)]   # left site of each pair


def sib(s):
    x, y = s
    return (x + 1, y) if x % 2 == 0 else (x - 1, y)


class Planner:
    def __init__(self, sch, loading, stay=True, drop_rows=(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)):
        self.sch = sch
        self.em = M.Emitter(sch, loading)
        self.stay = stay
        self.drop_rows = drop_rows
        self.mem = sch.members()
        self.next_use = self._next_use()

    def _next_use(self):
        """next_use[k][q] = first pulse >= k that contains q (or K)."""
        K, n = self.sch.K, self.sch.n
        nu = [None] * (K + 1)
        cur = {q: K for q in range(n)}
        nu[K] = dict(cur)
        for k in range(K - 1, -1, -1):
            for q in self.mem[k]:
                cur[q] = k
            nu[k] = dict(cur)
        return nu

    # ------------------------------------------------------------ helpers
    def zone_atoms(self):
        return {q for q, s in self.em.st.pos.items() if M.in_zone(s)}

    def drop_targets(self, qs):
        """Storage sites for atoms leaving the zone: keep column order, nearest free sites."""
        st = self.em.st
        taken = set()
        out = {}
        for q in sorted(qs, key=lambda q: (st.pos[q][1], st.pos[q][0])):
            px = M.phys(st.pos[q])[0]
            best = None
            for y in self.drop_rows:
                for x in range(10):
                    s = (x, y)
                    if s in st.occ or s in taken:
                        continue
                    c = M.dist(st.pos[q], s)
                    if best is None or c < best[0]:
                        best = (c, s)
                if best is not None and best[0] < 1.5:
                    break
            out[q] = best[1]
            taken.add(best[1])
        return out

    # ------------------------------------------------------------ one gap
    def gap(self, k):
        em, st = self.em, self.em.st
        P = self.sch.pulses[k]
        mem = self.mem[k]
        Z = self.zone_atoms()
        stay = {q for q in Z & mem if not em.pending(q)} if self.stay else set()
        partner = {}
        for a, b in P:
            partner[a], partner[b] = b, a
        # a stayer whose partner is already beside it on a pair: keep both
        # decide anchors: for pairs with both stayers, one moves to the other (or both keep)
        targets = {}       # final site for every member of pulse k
        reserved = set()   # zone sites claimed by final positions
        movers_z = {}      # zone->zone transfers
        for a, b in P:
            if a in stay and b in stay:
                if st.pos[b] == sib(st.pos[a]):
                    targets[a], targets[b] = st.pos[a], st.pos[b]
                else:
                    targets[a] = st.pos[a]   # b joins a (cheaper side chosen below)
                    targets[b] = sib(st.pos[a])
                    alt_a, alt_b = sib(st.pos[b]), st.pos[b]
                    if M.dist(st.pos[a], alt_a) < M.dist(st.pos[b], targets[b]):
                        targets[a], targets[b] = alt_a, alt_b
            elif a in stay or b in stay:
                s, l = (a, b) if a in stay else (b, a)
                targets[s] = st.pos[s]
                targets[l] = sib(st.pos[s])
        # conflicts: two pairs claiming the same site -> move the later pair to a free slot
        claimed = {}
        conflict_pairs = []
        for a, b in P:
            if a in targets:
                if targets[a] in claimed or targets[b] in claimed:
                    conflict_pairs.append((a, b))
                    del targets[a], targets[b]
                    continue
                claimed[targets[a]] = a
                claimed[targets[b]] = b
        leave = Z - stay
        # sites still occupied by stayers that are not moving keep them reserved
        # free slots for remaining pairs
        def slot_free(s):
            for t in (s, sib(s)):
                if t in claimed:
                    return False
                o = st.occ.get(t)
                if o is not None and o in stay and o not in targets:
                    return False
            return True

        rest = [(a, b) for a, b in P if a not in targets]
        for a, b in rest:
            # choose slot near the two atoms' current positions
            best = None
            for s in SLOTS:
                if not slot_free(s):
                    continue
                for (l, r) in ((a, b), (b, a)):
                    c = max(M.dist(st.pos[l], s), M.dist(st.pos[r], sib(s)))
                    c += 0.01 * (s[1] - 10)
                    if best is None or c < best[0]:
                        best = (c, s, l, r)
            _, s, l, r = best
            targets[l], targets[r] = s, sib(s)
            claimed[s], claimed[sib(s)] = l, r
        # 1. drops (leavers), 2. gates are emitted by the emitter, 3. zone->zone, 4. lifts
        execute(em, self.drop_targets(leave))
        zz = {q: t for q, t in targets.items() if M.in_zone(st.pos[q]) and st.pos[q] != t}
        execute(em, zz)
        # atoms still in the zone owing gates (stayers never owe) -> none
        up = {q: t for q, t in targets.items() if not M.in_zone(st.pos[q])}
        # binding atoms (pending) that are in storage get gates on arrival; lift
        execute(em, up)
        em.pulse()

    def run(self):
        for k in range(self.sch.K):
            self.gap(k)
        em, st = self.em, self.em.st
        Z = self.zone_atoms()
        need = {q for q in Z if em.pending(q)}
        if need:
            execute(em, self.drop_targets(need))
        return em.finish()


def plan(sch, loading, **kw):
    return Planner(sch, loading, **kw).run()
