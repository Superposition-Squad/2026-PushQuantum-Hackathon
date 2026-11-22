"""Rollout lookahead for the band choice (approximate DP), on relay_plus.

1. Run relay_plus's normal search, recording every run_candidate (cost, cand, home, steps).
2. Take the best run; replay its step sequence (forced schedule) where at every round step the band
   plan rank r in 0..K-1 is chosen by simulating this round plus the next L steps (rank 0) on a clone.
   Rank 0 everywhere must reproduce the base cost (sanity check).

    python l3/exp/rollout.py K L <qasm> [<qasm> ...]
"""
import copy
import importlib.util
import logging
import sys
import time
from pathlib import Path

logging.disable(logging.WARNING)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "vendor"))
spec = importlib.util.spec_from_file_location("rr", str(HERE / "relay_roll.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
from qiskit import QuantumCircuit  # noqa: E402

DW = R.DEPTH_W
RUNS = []
_orig_run_candidate = R.run_candidate


def _rec_run_candidate(n, blocks, preds, succs, home, pol, lam, mu, opts=(), emit=False, allowed=None):
    ex = _orig_run_candidate(n, blocks, preds, succs, home, pol, lam, mu, opts, emit, allowed)
    if not emit:
        RUNS.append((ex.m.dist + DW * (max(ex.m.depth) + 1), (pol, lam, mu, opts), dict(home),
                     [(k, list(b)) for k, b in ex.steps], blocks, allowed))
    return ex


R.run_candidate = _rec_run_candidate


class Replay:
    """Forced replay state: executor + per-step bookkeeping (copied for rollouts)."""

    def __init__(self, n, blocks, home, cand, allowed):
        pol, lam, mu, opts = cand
        hub = n - 1
        frame = dict(opts).get("FRAME", "S")
        if frame != "S" and any(b.kind != "G" for b in blocks):
            blocks = [R.Block(b.c, b.t, b.th, b.idx) for b in blocks]
        self.hub = hub
        ex = R.Executor(n, hub, home, emit=False, frame=frame)
        ex.allowed = allowed
        ex.LOOK = lam
        for k, v in opts:
            setattr(ex, k, v)
        ex.m.LAZY = ex.LAZY_FRAME
        if ex.SEED is not None:
            import numpy as np
            ex.rng = np.random.default_rng(ex.SEED)
        ex.steps = []
        self.ex = ex
        self.byq = {}
        for b in blocks:
            for q in b.qubits():
                self.byq.setdefault(q, []).append(b)
        self.ptr = {q: 0 for q in self.byq}
        self.done = set()
        self.hub_left = {}
        for b in blocks:
            if hub in (b.c, b.t) and b.kind != "P":
                o = b.t if b.c == hub else b.c
                self.hub_left[o] = self.hub_left.get(o, 0) + 1
        self.blocks = blocks
        self.idx = {b.idx: b for b in blocks}

    def clone(self):
        c = copy.copy(self)
        c.ex = self.ex.clone()
        c.ptr = dict(self.ptr)
        c.done = set(self.done)
        c.hub_left = dict(self.hub_left)
        return c

    def step(self, kind, blk_ids, rank=0):
        ex, hub = self.ex, self.hub
        blks = [self.idx[i] for i in blk_ids]
        cur = set(blk_ids)
        nxt = {}
        for b in blks:
            for q in b.qubits():
                lst = self.byq[q]
                k = self.ptr[q]
                while k < len(lst) and lst[k].idx in self.done:
                    k += 1
                self.ptr[q] = k
                while k < len(lst) and (lst[k].idx in self.done or lst[k].idx in cur):
                    k += 1
                if k < len(lst):
                    o = lst[k]
                    nxt[q] = o.t if o.c == q else o.c
        ex.nxt = nxt
        if ex.EVPULL:
            na = {}
            for q, lst in self.byq.items():
                k = self.ptr[q]
                while k < len(lst) and (lst[k].idx in self.done or lst[k].idx in cur):
                    k += 1
                self.ptr[q] = k if not any(lst[j].idx in cur for j in range(self.ptr[q], k)) else self.ptr[q]
                if k < len(lst):
                    o = lst[k]
                    na[q] = o.t if o.c == q else o.c
            ex.nxt_all = na
        ex.steps.append((kind, blks))
        for b in blks:
            if hub in (b.c, b.t) and b.kind != "P":
                self.hub_left[b.t if b.c == hub else b.c] -= 1
        ex.hub_left = self.hub_left
        ex.PLAN_RANK = rank
        try:
            (ex.run_round_band if kind == "round" else ex.run_hub)(blks)
        finally:
            ex.PLAN_RANK = 0
        self.done |= cur

    def cost(self):
        return self.ex.m.dist + DW * (max(self.ex.m.depth) + 1)


def replay(n, run, K=1, L=0):
    _, cand, home, steps, blocks, allowed = run
    ids = [(k, [b.idx for b in bl]) for k, bl in steps]
    st = Replay(n, blocks, home, cand, allowed)
    changed = 0
    for t, (kind, bl) in enumerate(ids):
        rank = 0
        if kind == "round" and K > 1:
            best = None
            for r in range(K):
                c = st.clone()
                try:
                    c.step(kind, bl, r)
                    for k2, b2 in ids[t + 1: t + 1 + L]:
                        c.step(k2, b2, 0)
                except R.Illegal:
                    continue
                v = c.cost()
                if best is None or v < best[0] - 1e-9:
                    best = (v, r)
            rank = best[1] if best else 0
            changed += rank != 0
        st.step(kind, bl, rank)
    st.ex.m.finish()
    return st.cost(), changed


if __name__ == "__main__":
    K, L = int(sys.argv[1]), int(sys.argv[2])
    from pushquantum_validator.spec import load_trusted_spec  # noqa: F401
    for f in sys.argv[3:]:
        RUNS.clear()
        qc = QuantumCircuit.from_qasm_file(f)
        parsed = R.parse_ihva(qc)
        n, blocks = parsed
        t = time.time()
        R.compile_ihva(n, blocks, emit=False)
        base = min(RUNS, key=lambda r: r[0])
        t1 = time.time()
        c0, _ = replay(n, base, 1, 0)
        c1, ch = replay(n, base, K, L)
        print(f"{Path(f).stem[9:]:24s} search-best {base[0]:8.1f}  replay(rank0) {c0:8.1f}  rollout K={K} L={L}: {c1:8.1f} "
              f"({100 * (c1 - base[0]) / base[0]:+.2f} %, {ch} rounds changed)  search {t1 - t:.0f}s rollout {time.time() - t1:.0f}s",
              flush=True)
