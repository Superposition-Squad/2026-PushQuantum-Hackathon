#!/usr/bin/env python3
"""MIP-based large neighbourhood search on an incumbent plan.

Slide a window of W consecutive moves (inside one gap) over the incumbent; fix the configurations
before and after; re-solve the window exactly (milp.solve_window, T = W slots, idle slots allowed);
splice the result in if its exact cost is lower. Storage visits that the incumbent makes only
inside the window are required in the window too (1q gates must still find their atom in storage).

    python l3/milp_lns.py <cached pkl> [W=4] [time_limit=20] [passes=1]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chain as C  # noqa: E402
import milp as MI  # noqa: E402
import model as M  # noqa: E402
from check import check  # noqa: E402


def states_of(start, moves):
    st = M.State(start)
    out = [dict(st.pos)]
    for g in moves:
        st.apply(g)
        out.append(dict(st.pos))
    return out


def window_need(states, i, W, B, members_end):
    """Atoms of B whose storage states in this gap all lie strictly inside the window."""
    need = []
    for q in B:
        sto = [s for s, cfg in enumerate(states) if not M.in_zone(cfg[q])]
        if sto and all(i < s < i + W for s in sto):
            need.append((q, 1, W - 1))
    return need


PAD = 1


def lns(path, W=4, tl=20, passes=1, log=print, record=None, ckpt=None, every=25):
    """ckpt: path of a checkpoint file (pickle). Saved every `every` windows (gaps, position,
    counters, records); an existing checkpoint is resumed. Delete it to start over."""
    import pickle
    c, l = M.load_cached(path)
    s = M.extract(c)
    X, gaps = C.incumbent(c, l, s)
    B = C.binding(s)
    base = sum(C.gap_cost(X[k], gaps[k]) for k in range(s.K + 1))
    gain = 0.0
    tried = solved = improved = 0
    t0 = time.time()
    pos0 = (0, 0, 0)
    if record is None:
        record = []
    if ckpt and Path(ckpt).exists():
        with open(ckpt, "rb") as f:
            st_ = pickle.load(f)
        gaps, pos0, gain, tried, solved, improved = st_["gaps"], st_["pos"], st_["gain"], st_["tried"], st_["solved"], st_["improved"]
        record[:] = st_["records"]
        log(f"  resumed at pass/gap/move {pos0}, gain so far {gain:.2f}")

    def save(p, k, i):
        if ckpt:
            tmp = str(ckpt) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"gaps": gaps, "pos": (p, k, i), "gain": gain, "tried": tried, "solved": solved,
                             "improved": improved, "records": record, "base": base}, f)
            Path(tmp).replace(ckpt)

    for p in range(pos0[0], passes):
        for k in range(pos0[1] if p == pos0[0] else 0, s.K + 1):
            i = pos0[2] if (p, k) == pos0[:2] else 0
            while i + 2 <= len(gaps[k]):
                if tried and tried % every == 0:
                    save(p, k, i)
                w = min(W, len(gaps[k]) - i)
                states = states_of(X[k], gaps[k])
                a, b = states[i], states[i + w]
                act = {q for g in gaps[k][i:i + w] for q in g}
                old = sum(M.State(states[i + j]).check(gaps[k][i + j])[1] for j in range(w))
                need = window_need(states, i, w, B[k] if k < len(B) else set(), None)
                tried += 1
                ys = [states[i + j][q][1] for j in range(w + 1) for q in act]
                rows = range(max(0, min(ys) - PAD), min(19, max(ys) + PAD) + 1)
                r = MI.solve_window(a, [], [], [], need, end=b, T=w, active=act, time_limit=tl, ub=old + 1e-6, rows=rows)
                if r.get("moves") is not None:
                    solved += 1
                    mv = [g for g in r["moves"] if g]
                    try:
                        new = C.gap_cost(a, mv)
                    except M.MoveError:
                        new = None
                    if new is not None and new < old - 1e-6:
                        if True:
                            def desc(start, ms):
                                st = M.State(start)
                                out = []
                                for g in ms:
                                    _, c = st.check(g)
                                    out.append((round(c, 3), {q: (st.pos[q], t) for q, t in g.items()}))
                                    st.apply(g)
                                return out
                            record.append({"gap": k, "i": i, "w": w, "old": old, "new": new, "need": need,
                                           "old_moves": desc(a, gaps[k][i:i + w]), "new_moves": desc(a, mv)})
                        gaps[k] = gaps[k][:i] + mv + gaps[k][i + w:]
                        gain += old - new
                        improved += 1
                        log(f"  gap {k} moves {i}..{i+w-1}: {old:.3f} -> {new:.3f} ({len(mv)} moves)  total gain {gain:.2f}")
                        continue  # retry at same position
                i += 1
    save(passes, 0, 0)
    out, load, d = C.assemble(s, l, gaps)
    r = check(out, load)
    return {"file": Path(path).stem, "base": base, "final": r["distance"], "tried": tried, "solved": solved,
            "improved": improved, "secs": round(time.time() - t0, 1)}, out, load


if __name__ == "__main__":
    path = sys.argv[1]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 20
    passes = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    res, out, load = lns(path, W, tl, passes)
    print(json.dumps(res))
