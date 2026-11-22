"""Planner v2: zone-resident workbench.

* Members of consecutive pulses stay in the zone (re-pair there); only leavers go down.
* Atoms that owe 1q gates (rotor mid-rotations) dip: the zone rows holding them shift down rigidly
  into free storage rows right below the zone (one move) and come back (one move).
* Leavers park in storage rows <= park_top (rows above stay free for dips); atoms needed soon
  park higher.
* Pairs get slots anchored at stayers; newcomers are lifted into them.
"""
from __future__ import annotations

import model as M
from group import execute
from planner import SLOTS, sib


class Planner2:
    def __init__(self, sch, loading, park_top=7, dip=True, stay=True, seeds=6):
        self.sch = sch
        self.em = M.Emitter(sch, loading)
        self.park_top = park_top
        self.dip = dip
        self.stay = stay
        self.seeds = seeds
        self.mem = sch.members()
        K, n = sch.K, sch.n
        self.nxt = [None] * (K + 1)          # nxt[k][q] = first pulse >= k containing q
        cur = {q: K for q in range(n)}
        self.nxt[K] = dict(cur)
        for k in range(K - 1, -1, -1):
            for q in self.mem[k]:
                cur[q] = k
            self.nxt[k] = dict(cur)

    def zone_atoms(self):
        return {q for q, s in self.em.st.pos.items() if M.in_zone(s)}

    # ---------------------------------------------------------------- parking
    def park_targets(self, qs, k):
        """Leavers -> storage rows <= park_top; sooner-needed atoms higher; keep column order."""
        st = self.em.st
        taken = set()
        out = {}
        for q in sorted(qs, key=lambda q: (self.nxt[k][q], st.pos[q][1], st.pos[q][0])):
            best = None
            for y in range(self.park_top, -1, -1):
                for x in range(10):
                    s = (x, y)
                    if s in st.occ or s in taken:
                        continue
                    c = M.dist(st.pos[q], s)
                    if best is None or c < best[0] - 1e-9:
                        best = (c, s)
            if best is None:   # storage rows <= park_top full: anywhere
                for y in range(9, -1, -1):
                    for x in range(10):
                        s = (x, y)
                        if s not in st.occ and s not in taken:
                            c = M.dist(st.pos[q], s)
                            if best is None or c < best[0]:
                                best = (c, s)
            out[q] = best[1]
            taken.add(best[1])
        return out

    # ---------------------------------------------------------------- dips
    def do_dip(self, pend):
        """Rigid shift of the zone rows holding `pend` into free storage rows; gates; back."""
        em, st = self.em, self.em.st
        rows = sorted({st.pos[q][1] for q in pend})
        block = [q for q, s in st.pos.items() if M.in_zone(s) and s[1] in rows]
        h = len(rows)
        for top in range(9, h - 2, -1):          # storage rows top-h+1 .. top
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
            em.move(down)
            em.move(back)
            return True
        return False

    # ---------------------------------------------------------------- one gap
    def gap(self, k):
        em, st = self.em, self.em.st
        P = self.sch.pulses[k]
        mem = self.mem[k]
        Z = self.zone_atoms()
        leave = Z - mem if self.stay else set(Z)
        execute(em, self.park_targets(leave, k), seeds=self.seeds)
        Z = self.zone_atoms()
        pend = {q for q in Z if em.pending(q)}
        if pend and self.dip:
            if not self.do_dip(pend):
                execute(em, self.park_targets(pend, k), seeds=self.seeds)
        elif pend:
            execute(em, self.park_targets(pend, k), seeds=self.seeds)
        Z = self.zone_atoms()
        stay = Z
        targets, claimed = {}, {}
        for a, b in P:
            if a in stay and b in stay:
                if st.pos[b] == sib(st.pos[a]):
                    ta, tb = st.pos[a], st.pos[b]
                else:
                    ta, tb = st.pos[a], sib(st.pos[a])
                    if M.dist(st.pos[a], sib(st.pos[b])) < M.dist(st.pos[b], tb):
                        ta, tb = sib(st.pos[b]), st.pos[b]
            elif a in stay or b in stay:
                s, l = (a, b) if a in stay else (b, a)
                ta, tb = (st.pos[s], sib(st.pos[s])) if s == a else (sib(st.pos[s]), st.pos[s])
            else:
                continue
            if ta in claimed or tb in claimed:
                continue
            targets[a], targets[b] = ta, tb
            claimed[ta], claimed[tb] = a, b

        def slot_free(s):
            for t in (s, sib(s)):
                if t in claimed:
                    return False
                o = st.occ.get(t)
                if o is not None and o not in targets:
                    return False
            return True

        for a, b in P:
            if a in targets:
                continue
            best = None
            for s in SLOTS:
                if not slot_free(s):
                    continue
                for l, r in ((a, b), (b, a)):
                    c = max(M.dist(st.pos[l], s), M.dist(st.pos[r], sib(s))) + 0.01 * (s[1] - 10)
                    if best is None or c < best[0]:
                        best = (c, s, l, r)
            _, s, l, r = best
            targets[l], targets[r] = s, sib(s)
            claimed[s], claimed[sib(s)] = l, r
        zz = {q: t for q, t in targets.items() if M.in_zone(st.pos[q]) and st.pos[q] != t}
        execute(em, zz, seeds=self.seeds)
        up = {q: t for q, t in targets.items() if not M.in_zone(st.pos[q])}
        execute(em, up, seeds=self.seeds)
        em.pulse()

    def run(self):
        for k in range(self.sch.K):
            self.gap(k)
        em = self.em
        need = {q for q in self.zone_atoms() if em.pending(q)}
        if need:
            execute(em, self.park_targets(need, self.sch.K))
        return em.finish()


def plan(sch, loading, **kw):
    return Planner2(sch, loading, **kw).run()
