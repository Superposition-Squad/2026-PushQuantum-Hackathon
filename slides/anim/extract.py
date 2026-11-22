#!/usr/bin/env python3
"""Extract a clean event stream (moves, CZ pulses, 1q gate groups) from a cached compiled plan,
with chapter markers, for slides/anim/relay_anim.html.

    python slides/anim/extract.py l3/cache/relay_best_l3/bus_stop_20q_ihva_stagger_p1.pkl > slides/anim/data.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "l3"))
import model as M  # noqa: E402

LABEL = "metric_circuit_13"  # bench label of 20q iHVA stagger p1 (naive baseline key)
CHAPTERS = [  # (first pulse index, title, text, layers)
    (0, "Band rounds", "Commuting blocks are grouped into rounds (matchings). Each block is CZ · rx(θ) · CZ in "
        "the S-frame: the whole band is lifted rigidly, pulsed, the rotors dip to storage for rx(θ), pulsed again.", ["L1", "L2", "L3"]),
    (6, "Relay: collect parity", "The hub's fan-out only depends on the partners' parity. Partner pairs pulse in "
        "parallel (a controlled-Y each) and fold their parity into one root per pair.", ["L1"]),
    (7, "Hub meets only the roots", "The hub now needs one CZ per pair instead of one per partner: "
        "8 serial hub pulses instead of 16.", ["L1", "L3"]),
    (15, "Uncompute, rotate, recompute", "The pairs undo the parity relay, the partners get their rx(θ) in "
        "storage, and the relay is rebuilt for the second half of the fan-out.", ["L1"]),
    (17, "Second pass", "Same trick with the other atom of each pair as root, so every input pair interacts "
        "(validator rule), then the parity relay is undone.", ["L1", "L3"]),
    (26, "Last round", "The remaining blocks run as one more band round and a final hub block.", ["L2", "L3"]),
]


def main(path):
    circ, loading = M.load_cached(path)
    qi = {b: i for i, b in enumerate(circ.qubits)}
    st = M.State(loading)
    n = circ.num_qubits
    events, cur_pulse, cur_1q = [], None, None
    pulse_idx = 0

    def flush():
        nonlocal cur_pulse, cur_1q, pulse_idx
        if cur_pulse:
            events.append({"t": "cz", "pairs": cur_pulse, "k": pulse_idx})
            pulse_idx += 1
        cur_pulse = None
        if cur_1q:
            events.append({"t": "g", "q": sorted(cur_1q)})
        cur_1q = None

    for ins in circ.data:
        name = ins.operation.name
        qs = [qi[q] for q in ins.qubits]
        if name == "cz":
            if cur_1q:
                events.append({"t": "g", "q": sorted(cur_1q)}); cur_1q = None
            if cur_pulse is None:
                cur_pulse = []
            if any(q in p for p in cur_pulse for q in qs):   # second pulse of the same layer
                events.append({"t": "cz", "pairs": cur_pulse, "k": pulse_idx}); pulse_idx += 1; cur_pulse = []
            cur_pulse.append(qs)
        elif name in ("rx", "rz"):
            if cur_pulse:
                events.append({"t": "cz", "pairs": cur_pulse, "k": pulse_idx}); pulse_idx += 1; cur_pulse = None
            cur_1q = (cur_1q or set()) | {qs[0]}
        elif name == "move":
            flush()
            rows, cols, nr, nc = ins.operation.params
            occ_r = [y for y in rows if any(st.pos[q][1] == y for q in qs)]
            occ_c = [x for x in cols if any(st.pos[q][0] == x for q in qs)]
            if len(nr) == len(rows):
                occ_r = list(rows)
            if len(nc) == len(cols):
                occ_c = list(cols)
            rm, cm = dict(zip(occ_r, nr)), dict(zip(occ_c, nc))
            grp = {q: (cm[st.pos[q][0]], rm[st.pos[q][1]]) for q in qs}
            atoms = []
            for q in qs:
                (x0, y0), (x1, y1) = M.phys(st.pos[q]), M.phys(grp[q])
                atoms.append([q, round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)])
            d = max(M.dist(st.pos[q], grp[q]) for q in qs)
            st.apply(grp)
            events.append({"t": "m", "a": atoms, "d": round(d, 4)})
        else:
            flush()
    flush()
    # chapter marker: before the first event of the chapter's first pulse (including its moves)
    first_ev_of_pulse = {}
    last_cz_before = -1
    for i, e in enumerate(events):
        if e["t"] == "cz" and e["k"] not in first_ev_of_pulse:
            j = i
            while j - 1 > last_cz_before and events[j - 1]["t"] != "cz":
                j -= 1
            first_ev_of_pulse[e["k"]] = j
        if e["t"] == "cz":
            last_cz_before = i
    chapters = [{"ev": 0 if k == 0 else first_ev_of_pulse[k], "title": t, "text": x, "layers": L} for k, t, x, L in CHAPTERS]
    sites = sorted({tuple(M.phys((x, y))) for x in range(10) for y in range(20)})
    load = {q: M.phys(s) for q, s in loading.items()}
    naive = json.load(open(ROOT / "vendor/pushquantum_validator/naive_baseline.json"))
    nv = naive.get(LABEL)
    out = {
        "title": "relay_best_l3 · 20-qubit iHVA (stagger, p = 1)",
        "n": n, "hub": n - 1,
        "load": [[round(load[q][0], 3), round(load[q][1], 3)] for q in range(n)],
        "sites": [[round(x, 3), round(y, 3)] for x, y in sites],
        "events": events, "chapters": chapters,
        "total_dist": round(sum(e["d"] for e in events if e["t"] == "m"), 2),
        "naive_dist": round(nv["metrics"]["move_distance"], 1) if nv else None,
    }
    json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1])
