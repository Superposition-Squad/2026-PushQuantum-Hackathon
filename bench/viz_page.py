"""Pure rendering for the compiled-circuit visualiser (``bench/viz.py`` produces the data).

Two public entry points, both returning one self-contained HTML document (inline CSS + plain
ES2017 JavaScript, no CDN, no modules, works from ``file://``):

    render_page(data)  -> str   # one compiled circuit (atom view + wire diagram + controls)
    render_index(rows) -> str   # overview over every compiler x circuit

Self-test CLI::

    python bench/viz_page.py --demo  -o /tmp/demo.html
    python bench/viz_page.py --demo-index -o /tmp/demo_index.html
    python bench/viz_page.py --demo-big -o /tmp/demo_big.html
    python bench/viz_page.py --data FILE.json -o out.html

The data schema is authoritative in ``docs/ideas/viz.md``.  Every optional field is treated
defensively: a ``compile_error`` page (no ``ops``/``moves``) still renders a clear error card.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- CSS
CSS = r"""
*{box-sizing:border-box}
html,body{margin:0;padding:0;height:100%;overflow:hidden;background:#f4f6f8;color:#1b2733;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:13px}
button{font:inherit;color:inherit}
#app{display:flex;flex-direction:column;height:100vh}
header#hdr{flex:0 0 auto;display:flex;align-items:center;gap:12px;padding:6px 12px;
  background:#1f2b38;color:#e8eef4;min-height:38px}
#hdr .name{font-weight:700;color:#9fd0ff}
#hdr .sep{color:#5b6b7b}
#hdr .file{color:#9fb0c0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:44vw}
#hdrleft{display:flex;align-items:center;gap:8px;min-width:0;flex:1 1 auto}
#hdrright{display:flex;align-items:center;gap:8px;flex:0 0 auto}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;
  letter-spacing:.02em;line-height:1.5}
.badge.ok{background:#1e7a3c;color:#fff}
.badge.warn{background:#a86a00;color:#fff}
.badge.bad{background:#b3261e;color:#fff}
#banner{flex:0 0 auto;margin:6px 8px 0;padding:8px 10px;border:1px solid #e2b3ae;background:#fdecea;
  color:#8c1d18;border-radius:6px;max-height:20vh;overflow:auto}
#banner .etitle{font-weight:700;margin-bottom:2px}
#banner pre{margin:4px 0;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:11px}
#banner details{margin-top:4px}
#banner ul{margin:4px 0 0 18px;padding:0}
#panels{flex:1 1 auto;display:flex;gap:8px;padding:8px;min-height:0}
#left{flex:0 0 45%;display:flex;flex-direction:column;min-width:0;min-height:0;
  background:#fff;border:1px solid #dbe2e9;border-radius:8px;overflow:hidden}
#atomwrap{flex:1 1 auto;min-height:0;position:relative}
#atomsvg{width:100%;height:100%;display:block}
#atomsvg .lbl{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;pointer-events:none}
#atomsvg.nolabels .lbl{display:none}
#chips{flex:0 0 auto;display:flex;flex-wrap:wrap;gap:6px;padding:5px 8px;border-top:1px solid #eef1f4;
  min-height:24px;align-items:center}
.chip{font-size:11px;padding:1px 7px;border-radius:9px;background:#eef2f6;color:#445566}
.chip.warn{background:#fdecea;color:#a3241d;font-weight:600}
.chip.good{background:#e6f4ea;color:#1e7a3c}
.chip.mismatch{background:#b3261e;color:#fff;font-weight:700}
#right{flex:1 1 55%;display:flex;flex-direction:column;min-width:0;min-height:0;
  background:#fff;border:1px solid #dbe2e9;border-radius:8px;overflow:hidden;position:relative}
#wirebox{flex:1 1 auto;position:relative;min-height:0;overflow:hidden}
#wirec{position:absolute;left:0;top:0;display:block;z-index:1}
#wirewrap{position:absolute;inset:0;overflow:auto;z-index:2}
#scrollpad{width:10px;height:10px}
#tip{position:fixed;z-index:20;pointer-events:none;background:#1f2b38;color:#f0f4f8;
  padding:4px 7px;border-radius:5px;font-size:11px;line-height:1.35;max-width:320px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;opacity:.96}
#controls{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:6px 12px;
  border-top:1px solid #dbe2e9;background:#fbfcfd}
#controls button{background:#eef2f6;border:1px solid #cdd7e0;border-radius:6px;padding:3px 10px;
  cursor:pointer;min-width:34px}
#controls button:hover{background:#e2e9f0}
#controls button:disabled{opacity:.45;cursor:default}
#controls label{display:flex;align-items:center;gap:5px;color:#44515f;font-size:12px;white-space:nowrap}
#controls input[type=range]{vertical-align:middle}
#scrub{flex:1 1 240px;min-width:120px}
#stats{flex:0 0 auto;padding:5px 12px;border-top:1px solid #dbe2e9;background:#1f2b38;color:#dbe6f0;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;white-space:nowrap;
  overflow-x:auto;display:flex;gap:10px;align-items:center;min-height:26px}
#stats .k{color:#7f93a6}
#stats .v{color:#fff}
.hidden{display:none!important}
#emptycard{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  padding:24px;text-align:center;color:#8c1d18;font-size:13px}
/* ---- index page ---- */
#idx{height:100vh;overflow:auto;padding:0 0 40px}
#idx header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:12px;flex-wrap:wrap;
  padding:10px 16px;background:#1f2b38;color:#e8eef4}
#idx header h1{font-size:16px;margin:0;font-weight:700}
#idx header .gen{color:#9fb0c0;font-size:11px;font-family:ui-monospace,Menlo,monospace}
#statusline{padding:6px 16px;background:#fff7e0;border-bottom:1px solid #efdca8;color:#7a5b00;
  font-family:ui-monospace,Menlo,monospace;font-size:12px}
#statusline.on{display:block}
#cards{display:flex;flex-wrap:wrap;gap:10px;padding:12px 16px}
.card{background:#fff;border:1px solid #dbe2e9;border-radius:8px;padding:10px 14px;min-width:230px;
  box-shadow:0 1px 2px rgba(20,40,60,.05)}
.card .cname{font-weight:700;font-size:14px;margin-bottom:4px;display:flex;align-items:center;gap:8px}
.card .metrics{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#33414f;line-height:1.6}
.card .metrics b{color:#0f1c28}
.card a.regen{font-size:11px;color:#7f93a6;text-decoration:none;border-bottom:1px dotted #b9c6d2}
.card a.regen:hover{color:#2f7fd0}
#filterbar{display:flex;gap:12px;align-items:center;padding:4px 16px 10px;flex-wrap:wrap}
#filter{flex:1 1 280px;max-width:520px;padding:6px 9px;border:1px solid #c6d1dc;border-radius:6px;font:inherit}
#filterbar .count{color:#6b7785;font-size:12px}
table{width:calc(100% - 32px);margin:0 16px;border-collapse:collapse;background:#fff;
  border:1px solid #dbe2e9;border-radius:8px;overflow:hidden;font-size:12px}
thead th{position:sticky;top:52px;background:#eef2f6;text-align:left;padding:6px 8px;cursor:pointer;
  border-bottom:1px solid #cdd7e0;white-space:nowrap;user-select:none}
thead th:hover{background:#e2e9f0}
thead th .ar{color:#8fa1b2;font-size:10px;margin-left:3px}
tbody td{padding:4px 8px;border-bottom:1px solid #eef1f4;white-space:nowrap;
  font-family:ui-monospace,Menlo,monospace}
tbody tr:hover{background:#f7fafd}
td.num{text-align:right}
td .filecell{max-width:430px;overflow:hidden;text-overflow:ellipsis;display:inline-block;
  vertical-align:bottom;color:#55636f}
.neg{color:#1e7a3c;font-weight:600}
.pos{color:#b3261e;font-weight:600}
td a{color:#1f5fbf;text-decoration:none}
td a:hover{text-decoration:underline}
"""

# --------------------------------------------------------------------------- page template
PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<div id="app">
  <header id="hdr"><div id="hdrleft"></div><div id="hdrright"></div></header>
  <div id="banner" class="hidden"></div>
  <main id="panels">
    <section id="left">
      <div id="atomwrap"><svg id="atomsvg" xmlns="http://www.w3.org/2000/svg"></svg></div>
      <div id="chips"></div>
    </section>
    <section id="right">
      <div id="wirebox">
        <canvas id="wirec"></canvas>
        <div id="wirewrap"><div id="scrollpad"></div></div>
      </div>
    </section>
  </main>
  <div id="controls">
    <button id="btnReset" title="reset (Home)">&#9198;</button>
    <button id="btnBack" title="step back (Left)">&#9664;</button>
    <button id="btnPlay" title="play / pause (Space)">&#9654;</button>
    <button id="btnFwd" title="step forward (Right)">&#9654;&#9654;</button>
    <label>speed <input id="speed" type="range" min="0" max="100" value="47" style="width:110px">
      <span id="speedv" class="k">12/s</span></label>
    <label>grain <select id="gran">
      <option value="instruction">instruction</option>
      <option value="frame">frame</option></select></label>
    <input id="scrub" type="range" min="0" max="1" value="0">
    <label><input id="lblToggle" type="checkbox" checked> labels</label>
    <label>zoom <input id="zoom" type="range" min="0.4" max="3" step="0.05" value="1"
      style="width:90px"><span id="zoomv" class="k">1.0x</span></label>
  </div>
  <div id="stats"></div>
</div>
<div id="tip" class="hidden"></div>
<script>
"use strict";
/*DATA_START*/const DATA = __DATA__;/*DATA_END*/
__JS__
</script>
</body>
</html>
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<div id="idx">
  <header><h1>compiled-circuit visualiser</h1><span class="gen" id="gen"></span></header>
  <div id="statusline" class="hidden"></div>
  <div id="cards"></div>
  <div id="filterbar">
    <input id="filter" type="search" placeholder="filter compiler / circuit / file / family">
    <span class="count" id="count"></span>
    <span class="count">click a column header to sort</span>
  </div>
  <table id="tbl">
    <thead><tr>
      <th data-k="compiler">compiler<span class="ar"></span></th>
      <th data-k="label">circuit<span class="ar"></span></th>
      <th data-k="file">file<span class="ar"></span></th>
      <th data-k="family">family<span class="ar"></span></th>
      <th data-k="qubits">qubits<span class="ar"></span></th>
      <th data-k="dist">distance<span class="ar"></span></th>
      <th data-k="ddist">&#916;% vs naive<span class="ar"></span></th>
      <th data-k="depth">depth<span class="ar"></span></th>
      <th data-k="ddepth">&#916;% vs naive<span class="ar"></span></th>
      <th data-k="valid">validity<span class="ar"></span></th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>
<script>
"use strict";
/*ROWS_START*/const ROWS = __ROWS__;/*ROWS_END*/
__JS__
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- helpers
def _safe_json(obj) -> str:
    """JSON safe to embed in a <script> element."""
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=True, default=str)
    return (
        s.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _title(data: dict) -> str:
    meta = data.get("meta") if isinstance(data, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    comp = meta.get("compiler") or "compiler"
    label = meta.get("label") or "circuit"
    return "%s \u00b7 %s" % (comp, label)


def render_page(data: dict) -> str:
    """Full HTML document for one compiled circuit."""
    if not isinstance(data, dict):
        data = {}
    out = PAGE_HTML
    out = out.replace("__TITLE__", html.escape(_title(data)))
    out = out.replace("__DATA__", _safe_json(data))
    out = out.replace("__CSS__", CSS)
    out = out.replace("__JS__", JS_PAGE)
    return out


def render_index(rows: dict) -> str:
    """Full HTML document for the overview table."""
    if not isinstance(rows, dict):
        rows = {}
    gen = rows.get("generated")
    title = "compiled circuits \u00b7 %s" % gen if gen else "compiled circuits"
    out = INDEX_HTML
    out = out.replace("__TITLE__", html.escape(title))
    out = out.replace("__ROWS__", _safe_json(rows))
    out = out.replace("__CSS__", CSS)
    out = out.replace("__JS__", JS_INDEX)
    return out


# --------------------------------------------------------------------------- demo data
def _zone_dicts() -> list:
    storage_sites = [[gx, gy, float(gx), float(gy)] for gx in range(10) for gy in range(10)]

    def ex(gx: int) -> float:
        return 2 * (gx // 2) + 0.3 * (gx % 2)

    ent_sites = [[gx, gy, ex(gx), float(gy)] for gx in range(10) for gy in range(10, 20)]
    return [
        {"name": "storage", "x": [0, 9], "y": [0, 9], "spacing": 1.0, "pair_axis": None,
         "pair_distance": None, "operations": ["rx", "rz"], "sites": storage_sites},
        {"name": "entangling", "x": [0, 9], "y": [10, 19], "spacing": 1.0, "pair_axis": "x",
         "pair_distance": 0.3, "operations": ["cz"], "sites": ent_sites},
    ]


class _DemoBuilder:
    """Builds a schema-1 payload with a plausible ASAP column layout and prefix arrays."""

    def __init__(self, nq: int):
        self.nq = nq
        self.zones = _zone_dicts()
        self.phys = {}
        for z in self.zones:
            for s in z["sites"]:
                self.phys[(s[0], s[1])] = (s[2], s[3])
        self.ops: list = []
        self.angles: list = []
        self.cols: list = []
        self.moves: list = []
        self.warnings: list = []
        self.qcol = [0] * nq
        self.maxcol = -1
        # loading: storage, fill rows left to right
        self.loading = [[i % 10, i // 10] for i in range(nq)]
        self.qsite = [list(s) for s in self.loading]

    def _p(self, site) -> tuple:
        return self.phys.get((site[0], site[1]), (float(site[0]), float(site[1])))

    def _col(self, qubits) -> int:
        c = max([self.qcol[q] for q in qubits] or [0])
        return c

    def gate(self, code: int, qubits, angle: float = 0.0, cz_count: dict | None = None):
        c = self._col(qubits)
        self.ops.append([code] + list(qubits) + [-1] * (2 - len(qubits)))
        self.angles.append(angle)
        self.cols.append(c)
        for q in qubits:
            self.qcol[q] = c + 1
        self.maxcol = max(self.maxcol, c)
        if cz_count is not None and code == 2:
            a, b = qubits
            key = tuple(sorted([tuple(self.qsite[a]), tuple(self.qsite[b])]))
            cz_count[key] = cz_count.get(key, 0) + 1

    def move(self, qubits, targets, warn: bool = False):
        b = [list(self.qsite[q]) for q in qubits]
        a = [list(t) for t in targets]
        c = max(self.qcol) + 1
        self.ops.append([3, len(self.moves), -1])
        self.angles.append(0.0)
        self.cols.append(c)
        d = 0.0
        for j in range(len(qubits)):
            pb, pa = self._p(b[j]), self._p(a[j])
            d += math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        rows = sorted({s[1] for s in b})
        cols = sorted({s[0] for s in b})
        nrows = sorted({s[1] for s in a})
        ncols = sorted({s[0] for s in a})
        self.moves.append({
            "i": len(self.ops) - 1, "q": list(qubits), "b": b, "a": a,
            "rows": rows, "cols": cols, "nrows": nrows, "ncols": ncols,
            "d": round(d, 5), "s": 1 if warn else 0, "w": 1 if warn else 0,
        })
        if warn:
            self.warnings.append([len(self.moves) - 1,
                                  "Moving cols %s -> %s: spacing not proportional." % (cols, ncols)])
        for j, q in enumerate(qubits):
            self.qsite[q] = list(a[j])
            self.qcol[q] = c + 1
        for q in range(self.nq):
            self.qcol[q] = max(self.qcol[q], c + 1)
        self.maxcol = max(self.maxcol, c)

    def finish(self, meta_extra: dict | None = None) -> dict:
        n = len(self.ops)
        ncols = self.maxcol + 1
        colmove = [0] * ncols
        for mv in self.moves:
            ci = self.cols[mv["i"]]
            if 0 <= ci < ncols:
                colmove[ci] = 1
        depth = 0
        depth_at = []
        for o in self.ops:
            if o[0] not in (3, 5):
                depth += 1
            depth_at.append(depth)
        total_d = round(sum(m["d"] for m in self.moves), 5)
        meta = {
            "compiler": "demo", "compiler_path": "/demo/compiler.py:compile_circuit",
            "label": "demo_circuit", "file": "metric_benchmarks/demo.qasm", "family": "hea",
            "qubits": self.nq, "status": "ok", "error": None, "error_index": None,
            "violations": [], "physics_ok": True,
            "physics": "transversal CZ ok (%d segments)" % max(1, len(self.moves)),
            "metrics": {"num_moves": len(self.moves), "move_steps": len(self.moves),
                        "move_distance": total_d, "move_time": total_d, "depth": depth,
                        "num_operations": n, "spacing_warnings": len(self.warnings)},
            "metrics_partial": False,
            "naive": {"move_distance": round(total_d * 1.9, 5), "depth": int(depth * 1.05)},
            "compile_s": 0.42, "generated": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if meta_extra:
            meta.update(meta_extra)
        return {
            "schema": 1, "meta": meta,
            "spec": {"loading_zone": "storage", "zones": self.zones},
            "loading": self.loading, "hub": max(0, self.nq - 1), "degree": [0] * self.nq,
            "ops": self.ops, "angles": self.angles, "col": self.cols, "ncols": ncols,
            "colmove": colmove, "depth_at": depth_at, "moves": self.moves,
            "warnings": self.warnings,
        }


def _demo_data(nq: int = 12, big: bool = False) -> dict:
    """Synthetic schema-1 payload exercising every overlay (moves, cz, rz/rx, measure, warning)."""
    if big:
        return _demo_big(50, moves_target=2800)
    b = _DemoBuilder(nq)
    cz_count: dict = {}
    # 1) single-qubit layer in storage
    for q in range(nq):
        b.gate(0, [q], angle=0.3 + 0.01 * q)
    for q in range(nq):
        b.gate(1, [q], angle=1.5708)
    # 2) move everything into entangling pairs (row 10 / 11)
    targets = []
    for i in range(nq):
        row = 10 if i < 10 else 11
        k = i if i < 10 else i - 10
        targets.append([2 * (k // 2) + (k % 2), row])
    b.move(list(range(nq)), targets, warn=False)
    # 3) one CZ pulse over every pair + another one on the first pair only (under-pulse demo)
    pairs = []
    seen = set()
    for q in range(nq):
        s = b.qsite[q]
        key = (s[1], s[0] - (s[0] % 2))
        if key not in seen and s[0] % 2 == 0:
            seen.add(key)
            pairs.append((q,))
    for q in range(nq):
        s = b.qsite[q]
        if s[0] % 2 == 0:
            other = None
            for q2 in range(nq):
                s2 = b.qsite[q2]
                if s2[0] == s[0] + 1 and s2[1] == s[1]:
                    other = q2
                    break
            if other is not None:
                b.gate(2, [q, other], cz_count=cz_count)
    # extra pulse on only one pair -> under-pulsed in the *current* segment if replayed mid-way
    if nq >= 2:
        b.gate(2, [0, 1], cz_count=cz_count)
    # 4) rz on a few atoms (targeted cross), rx, measure
    for q in range(0, min(nq, 6)):
        b.gate(0, [q], angle=0.7)
    for q in range(0, min(nq, 3)):
        b.gate(1, [q], angle=0.2)
    # 5) a second (warning) move: shuffle three atoms to a fresh row (non-proportional -> warning)
    mvq = list(range(0, min(nq, 6), 2))
    tgt = [[2 * j, 12] for j in range(len(mvq))]
    b.move(mvq, tgt, warn=True)
    # 6) cz again, then one more clean move, then measure all
    for q in range(0, min(nq, 6), 2):
        s = b.qsite[q]
        other = None
        for q2 in range(nq):
            if b.qsite[q2][0] == s[0] + 1 and b.qsite[q2][1] == s[1]:
                other = q2
                break
        if other is not None:
            b.gate(2, [q, other], cz_count=cz_count)
    if nq >= 4:
        b.move([nq - 2, nq - 1], [[8, 14], [9, 14]])
    for q in range(nq):
        b.gate(4, [q])
    degrees = [0] * nq
    for i in range(nq):
        degrees[i] = 2 + (i % 5)
    degrees[max(0, nq - 1)] = max(degrees) + 4
    data = b.finish()
    data["degree"] = degrees
    data["hub"] = max(0, nq - 1) if nq < 40 else 49
    return data


def _demo_big(nq: int = 50, moves_target: int = 2800) -> dict:
    """~``moves_target`` moves and ~4x as many gates: exercises wire virtualisation."""
    b = _DemoBuilder(nq)
    import random as _random
    rng = _random.Random(7)
    ent = [tuple(s) for s in b.zones[1]["sites"]]
    # start in storage
    for q in range(nq):
        b.gate(0, [q], angle=0.1)
    # one big move into the entangling zone
    tg = []
    for i in range(nq):
        row = 10 + (i // 10)
        k = i % 10
        tg.append([2 * (k // 2) + (k % 2), row])
    b.move(list(range(nq)), tg)
    cur = [list(s) for s in b.qsite]
    used_rows = sorted({s[1] for s in cur})
    for m in range(moves_target):
        # move two random qubits to random fresh entangling sites
        q1 = rng.randrange(nq)
        q2 = rng.randrange(nq)
        if q1 == q2:
            q2 = (q2 + 1) % nq
        t1 = ent[rng.randrange(len(ent))]
        t2 = ent[rng.randrange(len(ent))]
        # keep them paired most of the time
        if m % 3 != 0:
            t1 = (t1[0] - t1[0] % 2, t1[1])
            t2 = (t1[0] + 1, t1[1])
        b.move([q1, q2], [t1, t2], warn=(m % 97 == 0))
        for _ in range(3):
            code = rng.choice([0, 0, 1, 2, 4])
            if code == 2:
                # only legal-ish pairs from current positions
                q = rng.randrange(nq)
                s = b.qsite[q]
                partner = None
                for q2b in range(nq):
                    if b.qsite[q2b][0] == (s[0] ^ 1) and b.qsite[q2b][1] == s[1]:
                        partner = q2b
                        break
                if partner is not None:
                    b.gate(2, [q, partner])
                else:
                    b.gate(0, [q], angle=rng.random())
            elif code == 4:
                b.gate(4, [rng.randrange(nq)])
            else:
                b.gate(code, [rng.randrange(nq)], angle=rng.random())
    for q in range(nq):
        b.gate(4, [q])
    degrees = [10 + (i % 7) for i in range(nq)]
    data = b.finish({"label": "demo_big_50q"})
    data["degree"] = degrees
    data["hub"] = 49
    return data


def _demo_index() -> dict:
    now = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    compilers = []
    for ci, (name, scale) in enumerate([("naive", 1.0), ("zonedwell", 0.53)]):
        circuits = []
        for k in range(1, 7):
            q = [12, 20, 30, 40, 50, 50][k - 1]
            base = 1200.0 * q / 50 + 300
            naive = round(base, 2)
            dist = round(base * scale, 2)
            circuits.append({
                "label": "metric_circuit_%02d" % k,
                "file": "metric_benchmarks/%s/circuit_%02d.qasm" % (["hea", "ihva", "qaoa"][k % 3], k),
                "family": ["hea", "ihva", "qaoa"][k % 3], "qubits": q,
                "status": "ok" if k != 4 else ("invalid" if ci == 1 else "ok"),
                "error": None if k != 4 else "ArchitectureError[pair]: Instruction 412 (cz on [3, 7])",
                "page": "%s/metric_circuit_%02d.html" % (name, k),
                "metrics": {"num_moves": 900 + 10 * k, "move_steps": 900 + 10 * k,
                            "move_distance": dist, "move_time": dist,
                            "depth": int(700 + 20 * k), "num_operations": 4000,
                            "spacing_warnings": 0},
                "naive": {"move_distance": naive, "depth": int(760 + 21 * k)},
                "physics_ok": k != 5, "compile_s": 0.4 + 0.1 * k,
                "bytes": 900000 + 1000 * k,
            })
        summ = {
            "n_metric": len(circuits),
            "distance": round(sum(c["metrics"]["move_distance"] for c in circuits), 2),
            "naive_distance": round(sum(c["naive"]["move_distance"] for c in circuits), 2),
            "depth": sum(c["metrics"]["depth"] for c in circuits),
            "naive_depth": sum(c["naive"]["depth"] for c in circuits),
            "valid": all(c["status"] == "ok" and c["physics_ok"] for c in circuits),
        }
        summ["distance_improvement"] = round(1 - summ["distance"] / summ["naive_distance"], 4)
        summ["depth_improvement"] = round(1 - summ["depth"] / summ["naive_depth"], 4)
        summ["score"] = 0.0 if not summ["valid"] else round(1 - (summ["distance"] / summ["naive_distance"]
                                                                 + summ["depth"] / summ["naive_depth"]) / 2, 4)
        compilers.append({"compiler": name,
                          "compiler_path": "/home/hackathon/challenge/compilers/%s/compiler.py" % name,
                          "set": "all", "generated": now, "circuits": circuits, "summary": summ})
    return {"generated": now, "compilers": compilers}


# --------------------------------------------------------------------------- static JS check
_SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.I)


def _strip_js(src: str) -> str:
    """Remove string literals and comments (small, sufficient for a brace/paren balance check)."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'`":
            quote = c
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i < n - 1 and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _check_js(document: str) -> list:
    """Static checks only (no JS engine here): balanced brackets + embedded JSON parses."""
    problems = []
    blocks = _SCRIPT_RE.findall(document)
    if not blocks:
        problems.append("no <script> block found")
    for bi, block in enumerate(blocks):
        stripped = _strip_js(block)
        pairs = {")": "(", "]": "[", "}": "{"}
        stack = []
        for ch in stripped:
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack or stack[-1] != pairs[ch]:
                    problems.append("script %d: unbalanced %r" % (bi, ch))
                    stack = []
                    break
                stack.pop()
        if stack:
            problems.append("script %d: unclosed %r" % (bi, stack[-1]))
    for marker in ("DATA", "ROWS"):
        m = re.search(r"/\*%s_START\*/const %s = (.*?);/\*%s_END\*/" % (marker, marker, marker),
                      document, re.S)
        if m:
            try:
                json.loads(m.group(1))
            except Exception as exc:  # noqa: BLE001 - report, never raise in the CLI
                problems.append("%s payload does not parse: %s" % (marker, exc))
    return problems


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="render visualiser HTML pages (pure rendering)")
    ap.add_argument("--demo", action="store_true", help="synthetic 12-qubit page")
    ap.add_argument("--demo-big", action="store_true", help="synthetic 50-qubit, ~11k instruction page")
    ap.add_argument("--demo-index", action="store_true", help="synthetic index page")
    ap.add_argument("--data", metavar="FILE.json", help="render a real data file")
    ap.add_argument("-o", "--out", default=None, help="output html path")
    ap.add_argument("--check", action="store_true", help="run the static JS check on the output")
    args = ap.parse_args(argv)

    if args.demo:
        data, default = _demo_data(12), "/tmp/demo.html"
    elif args.demo_big:
        data, default = _demo_data(50, big=True), "/tmp/demo_big.html"
    elif args.demo_index:
        data, default = _demo_index(), "/tmp/demo_index.html"
    elif args.data:
        data, default = json.loads(Path(args.data).read_text(encoding="utf-8")), "/tmp/page.html"
    else:
        ap.print_help()
        return 2
    out = Path(args.out or default)
    if args.demo_index:
        document = render_index(data)
    else:
        document = render_page(data)
    problems = _check_js(document)
    if problems:
        print("static JS check FAILED:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
    else:
        print("static JS check: ok (balanced brackets, embedded JSON parses)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    print("wrote %s (%d bytes)" % (out, len(document.encode("utf-8"))))
    return 1 if problems else 0


# --------------------------------------------------------------------------- JS: page
JS_PAGE = r"""(function(){
"use strict";
var SVGNS="http://www.w3.org/2000/svg";
function $(id){return document.getElementById(id);}
function clamp(v,a,b){return v<a?a:(v>b?b:v);}
function num(v,d){if(v===null||v===undefined||isNaN(v))return "\u2013";return Number(v).toFixed(d);}
function sv(tag,attrs){var e=document.createElementNS(SVGNS,tag);if(attrs){for(var k in attrs){if(Object.prototype.hasOwnProperty.call(attrs,k))e.setAttribute(k,attrs[k]);}}return e;}

var D=(typeof DATA!=="undefined"&&DATA)?DATA:{};
var meta=(D.meta&&typeof D.meta==="object")?D.meta:{};
var spec=(D.spec&&typeof D.spec==="object")?D.spec:{};
var zonesRaw=spec.zones;
var zones=[];
if(Array.isArray(zonesRaw)){zones=zonesRaw;}
else if(zonesRaw&&typeof zonesRaw==="object"){
  for(var zk in zonesRaw){ if(Object.prototype.hasOwnProperty.call(zonesRaw,zk)){
    var zr=zonesRaw[zk]||{}; var pa=zr.pairs&&zr.pairs.axis?zr.pairs.axis:null;
    var pd=zr.pairs&&typeof zr.pairs.distance==="number"?zr.pairs.distance:null;
    zones.push({name:zk,x:zr.x,y:zr.y,spacing:zr.spacing,pair_axis:pa,pair_distance:pd,
      operations:zr.operations,sites:zr.sites});
  }}
}
zones=zones.filter(function(z){return z&&Array.isArray(z.x)&&Array.isArray(z.y);});
var ops=Array.isArray(D.ops)?D.ops:[];
var angles=Array.isArray(D.angles)?D.angles:[];
var colOf=Array.isArray(D.col)?D.col:[];
var moves=Array.isArray(D.moves)?D.moves:[];
var depthAt=Array.isArray(D.depth_at)?D.depth_at:[];
var colmove=Array.isArray(D.colmove)?D.colmove:[];
var loading=Array.isArray(D.loading)?D.loading:[];
var warnings=Array.isArray(D.warnings)?D.warnings:[];
var N=ops.length;
var nq=0;
if(typeof meta.qubits==="number"&&meta.qubits>0){nq=Math.floor(meta.qubits);}
else{nq=loading.length;}
for(var qi=0;qi<N;qi++){
  var qo=ops[qi];
  if(!qo||!qo.length){continue;}
  if(qo[0]===2){nq=Math.max(nq,(qo[1]|0)+1,(qo[2]|0)+1);}
  else if(qo[0]===0||qo[0]===1||qo[0]===4){nq=Math.max(nq,(qo[1]|0)+1);}
}
if(!(nq>0)){nq=Math.max(1,loading.length);}
var hub=(typeof D.hub==="number")?D.hub:-1;

/* ------------------------------------------------------------------ physics helpers */
var sitePhys={};
for(var zi=0;zi<zones.length;zi++){
  var zs=zones[zi].sites;
  if(Array.isArray(zs)){for(var si=0;si<zs.length;si++){var s0=zs[si];
    if(Array.isArray(s0)&&s0.length>=4){sitePhys[s0[0]+","+s0[1]]=[s0[2],s0[3]];}}}
}
function phys(zone,line,axis){
  if(!zone){return line;}
  var sp=(typeof zone.spacing==="number"&&zone.spacing>0)?zone.spacing:1;
  if(zone.pair_axis!==axis||zone.pair_distance===null||zone.pair_distance===undefined){return line*sp;}
  var start=(axis==="x")?zone.x[0]:zone.y[0];
  var dd=Math.floor((line-start)/2);
  var slot=((line-start)%2+2)%2;
  return start*sp+dd*2*sp+slot*zone.pair_distance;
}
function zoneOf(site){
  if(!site){return null;}
  for(var i=0;i<zones.length;i++){var z=zones[i];
    if(site[0]>=z.x[0]&&site[0]<=z.x[1]&&site[1]>=z.y[0]&&site[1]<=z.y[1]){return z;}}
  return null;
}
function physicalOfSite(site){
  if(!site){return [0,0];}
  var p=sitePhys[site[0]+","+site[1]];
  if(p){return p;}
  var z=zoneOf(site);
  if(z){return [phys(z,site[0],"x"),phys(z,site[1],"y")];}
  return [site[0],site[1]];
}

/* ------------------------------------------------------------------ precomputed indexes */
var ncols=(typeof D.ncols==="number"&&D.ncols>0)?Math.floor(D.ncols):0;
if(!ncols){ncols=0;for(var nc=0;nc<N;nc++){ncols=Math.max(ncols,(colOf[nc]||0)+1);}}
var order=[],colStart=[],colFirst=[];
(function buildCols(){
  order=new Array(N);
  for(var i=0;i<N;i++){order[i]=i;}
  order.sort(function(a,b){var ca=colOf[a]||0,cb=colOf[b]||0;return ca-cb||a-b;});
  colStart=new Array(ncols+1);
  colFirst=new Array(ncols);
  for(var c=0;c<ncols;c++){colFirst[c]=-1;}
  var p=0;
  for(c=0;c<ncols;c++){
    while(p<N&&(colOf[order[p]]||0)<c){p++;}
    colStart[c]=p;
  }
  colStart[ncols]=N;
  for(p=0;p<N;p++){var cc=colOf[order[p]]||0;if(cc>=0&&cc<ncols&&colFirst[cc]<0){colFirst[cc]=order[p];}}
})();
var moveInstr=[],moveDist=[],moveWarn=[],mp=[0],mw=[0];
(function buildMoves(){
  for(var k=0;k<moves.length;k++){
    var mv=moves[k]||{};
    moveInstr.push(typeof mv.i==="number"?mv.i:k);
    moveDist.push(typeof mv.d==="number"?mv.d:0);
    moveWarn.push(typeof mv.w==="number"?mv.w:(mv.s?1:0));
    mp.push(mp[k]+moveDist[k]);
    mw.push(mw[k]+moveWarn[k]);
  }
})();
var czPrefix=new Array(N+1);czPrefix[0]=0;
for(var ci0=0;ci0<N;ci0++){var co=ops[ci0];czPrefix[ci0+1]=czPrefix[ci0]+((co&&co[0]===2)?1:0);}
var segStart=new Array(N);
(function(){var lm=-1;for(var i=0;i<N;i++){if(ops[i]&&ops[i][0]===3){lm=i;}segStart[i]=lm+1;}})();
function countLess(arr,v){var lo=0,hi=arr.length;while(lo<hi){var mid=(lo+hi)>>1;if(arr[mid]<v){lo=mid+1;}else{hi=mid;}}return lo;}

/* ------------------------------------------------------------------ svg build (once) */
var svg=$("atomsvg");
var gZones=null,gPairs=null,gTraps=null,gAod=null,gTrail=null,gAtoms=null,gOverlay=null;
var atomG=[],atomBadge=[];
var basePx=[],basePy=[];
var zoneGeom={};
var ATOM_R=0.145,TRAP_R=0.145;
var bbox={x0:0,x1:10,y0:0,y1:20};
function buildSvg(){
  while(svg.firstChild){svg.removeChild(svg.firstChild);}
  var xs=[],ys=[];
  for(var i=0;i<zones.length;i++){var z=zones[i];var ss=z.sites;
    if(Array.isArray(ss)){for(var j=0;j<ss.length;j++){var s=ss[j];
      if(Array.isArray(s)&&s.length>=4){xs.push(s[2]);ys.push(s[3]);}}}}
  if(!xs.length){for(i=0;i<zones.length;i++){xs.push(phys(zones[i],zones[i].x[0],"x"),phys(zones[i],zones[i].x[1],"x"));
    ys.push(zones[i].y[0],zones[i].y[1]);}}
  if(!xs.length){bbox={x0:-1,x1:11,y0:-1,y1:11};}
  else{
    var x0=Math.min.apply(null,xs)-1.2,x1=Math.max.apply(null,xs)+1.2;
    var y0=Math.min.apply(null,ys)-0.9,y1=Math.max.apply(null,ys)+0.9;
    bbox={x0:x0,x1:x1,y0:y0,y1:y1};
  }
  var W=bbox.x1-bbox.x0,H=bbox.y1-bbox.y0;
  svg.setAttribute("viewBox",bbox.x0+" "+(-bbox.y1)+" "+W+" "+H);
  svg.setAttribute("preserveAspectRatio","xMidYMid meet");
  gZones=sv("g",{});gPairs=sv("g",{});gTraps=sv("g",{});gAod=sv("g",{});
  gTrail=sv("g",{});gAtoms=sv("g",{});gOverlay=sv("g",{});
  svg.appendChild(gZones);svg.appendChild(gPairs);svg.appendChild(gTraps);svg.appendChild(gAod);
  svg.appendChild(gTrail);svg.appendChild(gAtoms);svg.appendChild(gOverlay);
  zoneGeom={};
  for(i=0;i<zones.length;i++){
    var zz=zones[i];
    var pxa=phys(zz,zz.x[0],"x")-0.45,pxb=phys(zz,zz.x[1],"x")+0.45;
    var pya=phys(zz,zz.y[0],"y")-0.45,pyb=phys(zz,zz.y[1],"y")+0.45;
    zoneGeom[zz.name||("zone"+i)]={x0:pxa,x1:pxb,y0:pya,y1:pyb};
    var orange=(zz.pair_axis!==null&&zz.pair_axis!==undefined);
    gZones.appendChild(sv("rect",{x:pxa,y:-pyb,width:pxb-pxa,height:pyb-pya,rx:0.3,ry:0.3,
      fill:orange?"#fbe3d0":"#dde9f5",stroke:orange?"#e07b2a":"#7fa8cc","stroke-width":0.07}));
    var cy=-(pya+pyb)/2;
    var t1=sv("text",{x:pxa+0.26,y:cy,fill:orange?"#b45309":"#2f6fa8","font-size":0.48,
      "text-anchor":"middle","font-weight":600,transform:"rotate(-90 "+ (pxa+0.26)+" "+cy+")"});
    t1.textContent=orange?"Entangling zone":"Storage zone";gZones.appendChild(t1);
    var t2=sv("text",{x:pxb-0.26,y:cy,fill:orange?"#b45309":"#2f6fa8","font-size":0.48,
      "text-anchor":"middle","font-weight":600,transform:"rotate(90 "+ (pxb-0.26)+" "+cy+")"});
    t2.textContent=orange?"Transverse CZ":"Global Rx";gZones.appendChild(t2);
    if(Array.isArray(zz.sites)){for(var sj=0;sj<zz.sites.length;sj++){var site=zz.sites[sj];
      if(!Array.isArray(site)||site.length<4){continue;}
      gTraps.appendChild(sv("circle",{cx:site[2],cy:-site[3],r:TRAP_R,fill:"#ffffff",stroke:"#9fb3c8","stroke-width":0.04}));
    }}
  }
  atomG=[];atomBadge=[];
  for(i=0;i<nq;i++){
    var g=sv("g",{});
    var isHub=(hub===i);
    var c=sv("circle",{r:ATOM_R,fill:isHub?"#7b3fb5":"#1f5fbf",
      stroke:isHub?"#4a1f73":"#12407f","stroke-width":isHub?0.11:0.05});
    g.appendChild(c);
    var lbl=sv("text",{"class":"lbl","font-size":0.32,x:0.20,y:-0.17,fill:"#12324f"});
    lbl.textContent="q"+i;g.appendChild(lbl);
    var badge=sv("rect",{x:-0.085,y:-0.085,width:0.17,height:0.17,fill:"#6b7785",display:"none"});
    g.appendChild(badge);
    gAtoms.appendChild(g);atomG.push(g);atomBadge.push(badge);
    basePx.push(0);basePy.push(0);
  }
}
buildSvg();
if(!zones.length){
  var ec=document.createElement("div");ec.id="emptycard";
  ec.textContent=(meta.error||"no architecture spec in this page")+"";
  $("atomwrap").appendChild(ec);
}
function setAtomPos(q,px,py){
  basePx[q]=px;basePy[q]=py;
  if(atomG[q]){atomG[q].setAttribute("transform","translate("+px+","+(-py)+")");}
}

/* ------------------------------------------------------------------ state */
var gran="instruction";
var frames=[];
function buildFrames(){
  frames=[];
  var i=0;
  while(i<N){
    if(ops[i]&&ops[i][0]===3){frames.push({start:i,end:i+1});i++;}
    else{var j=i;while(j<N&&!(ops[j]&&ops[j][0]===3)){j++;}frames.push({start:i,end:j});i=j;}
  }
}
buildFrames();
var step=0;
var playing=false,speed=12,lastT=0,acc=0,rafPlay=0;
var animState={ci:-1,t:1,k:-1,qs:[],pre:[],post:[]};
var rafAnim=0,animT0=0;
var trail=null;
var zoom=1.0;
var errCol=-1;
if(typeof meta.error_index==="number"&&meta.error_index>=0&&meta.error_index<N){
  errCol=colOf[meta.error_index]||0;
}
var hoverInstr=-1;

function nstepsFor(){return gran==="frame"?(frames.length+1):(N+1);}
function posFor(s){
  if(gran==="frame"){
    if(s<=0||!frames.length){return 0;}
    return frames[Math.min(s,frames.length)-1].end;
  }
  return clamp(s,0,N);
}
function stepForPos(p){
  if(gran==="frame"){
    if(p<=0){return 0;}
    for(var f=0;f<frames.length;f++){if(p<=frames[f].end){return f+1;}}
    return frames.length;
  }
  return clamp(p,0,N);
}
function replay(limit){
  var sites=new Array(nq);
  for(var i=0;i<nq;i++){var s=loading[i];sites[i]=(Array.isArray(s)&&s.length>=2)?[s[0],s[1]]:[0,0];}
  for(var k=0;k<moves.length;k++){
    var mv=moves[k];if(!mv){continue;}
    var ii=(typeof mv.i==="number")?mv.i:k;
    if(ii>=limit){continue;}
    if(Array.isArray(mv.q)&&Array.isArray(mv.a)){
      for(var j=0;j<mv.q.length;j++){
        var q=mv.q[j];var a=mv.a[j];
        if(q>=0&&q<nq&&Array.isArray(a)&&a.length>=2){sites[q]=[a[0],a[1]];}
      }
    }
  }
  return sites;
}
function computeState(){
  var p=posFor(step);
  var cidx=p-1;
  var curIsMove=(cidx>=0&&ops[cidx]&&ops[cidx][0]===3);
  return {p:p,ci:cidx,curIsMove:curIsMove,sites:replay(curIsMove?cidx:p)};
}

/* ------------------------------------------------------------------ atom view rendering */
function updateAtomBase(st){
  for(var q=0;q<nq;q++){var ph=physicalOfSite(st.sites[q]);setAtomPos(q,ph[0],ph[1]);}
}
function beginMoveAnim(st){
  var k=ops[st.ci][1];
  var mv=moves[k];
  animState={ci:st.ci,t:0,k:k,qs:[],pre:[],post:[]};
  if(!mv||!Array.isArray(mv.q)){animState.t=1;return;}
  for(var j=0;j<mv.q.length;j++){
    var q=mv.q[j];
    var b=physicalOfSite(mv.b?mv.b[j]:null);
    var a=physicalOfSite(mv.a?mv.a[j]:null);
    animState.qs.push(q);animState.pre.push(b);animState.post.push(a);
  }
}
function updateMovedAtoms(){
  var t=animState.t;
  for(var j=0;j<animState.qs.length;j++){
    var q=animState.qs[j],b=animState.pre[j],a=animState.post[j];
    if(q<0||q>=nq){continue;}
    setAtomPos(q,b[0]+(a[0]-b[0])*t,b[1]+(a[1]-b[1])*t);
  }
}
function startAnimLoop(){
  if(rafAnim){cancelAnimationFrame(rafAnim);}
  animT0=0;
  function frame(now){
    if(!animT0){animT0=now;}
    var t=clamp((now-animT0)/460,0,1);
    animState.t=t;
    updateMovedAtoms();
    if(t<1){rafAnim=requestAnimationFrame(frame);}else{rafAnim=0;}
  }
  rafAnim=requestAnimationFrame(frame);
}

function drawPairs(st){
  while(gPairs.firstChild){gPairs.removeChild(gPairs.firstChild);}
  var pairs={},pairList=[];
  for(var i=0;i<zones.length;i++){
    var z=zones[i];
    if(z.pair_axis!=="x"&&z.pair_axis!=="y"){continue;}
    var byKey={};
    for(var q=0;q<nq;q++){
      var s=st.sites[q];if(!s){continue;}
      if(!(s[0]>=z.x[0]&&s[0]<=z.x[1]&&s[1]>=z.y[0]&&s[1]<=z.y[1])){continue;}
      var axis0=(z.pair_axis==="x")?0:1;
      var line=s[axis0];var start=(axis0===0)?z.x[0]:z.y[0];
      var slot=((line-start)%2+2)%2;
      var low=line-slot;
      var key=(axis0===0)?(s[1]+":"+low):(low+":"+s[0]);
      if(!byKey[key]){byKey[key]={key:key,zone:z,axis0:axis0,low:low,row:(axis0===0?s[1]:s[0]),qs:[null,null]};}
      byKey[key].qs[slot]=q;
    }
    for(var kk in byKey){if(Object.prototype.hasOwnProperty.call(byKey,kk)){
      var pr=byKey[kk];
      if(pr.qs[0]!==null&&pr.qs[1]!==null){pairs[kk]=pr;pairList.push(pr);}
    }}
  }
  var counts={};
  var maxc=0;
  var cidx=st.ci;
  if(cidx>=0&&!(ops[cidx]&&ops[cidx][0]===3)){
    var a0=segStart[cidx]||0;
    for(var j=a0;j<=cidx;j++){
      var o=ops[j];if(!o||o[0]!==2){continue;}
      var qa=o[1],qb=o[2];
      var sa=st.sites[qa],sb=st.sites[qb];
      if(!sa||!sb){continue;}
      var wkey=null;
      for(var pj=0;pj<pairList.length;pj++){
        var pl=pairList[pj];
        if((pl.qs[0]===qa&&pl.qs[1]===qb)||(pl.qs[0]===qb&&pl.qs[1]===qa)){wkey=pl.key;break;}
      }
      if(wkey!==null){counts[wkey]=(counts[wkey]||0)+1;maxc=Math.max(maxc,counts[wkey]);}
    }
  }
  var under=0;
  for(var p2=0;p2<pairList.length;p2++){
    var pair=pairList[p2];
    var zz=pair.zone;
    var pxa,pxb,pya,pyb;
    if(pair.axis0===0){
      pxa=phys(zz,pair.low,"x")-0.20;pxb=phys(zz,pair.low+1,"x")+0.20;
      pya=phys(zz,pair.row,"y")-0.28;pyb=pya+0.56;
    }else{
      pya=phys(zz,pair.low,"y")-0.20;pyb=phys(zz,pair.low+1,"y")+0.20;
      pxa=phys(zz,pair.row,"x")-0.28;pxb=pxa+0.56;
    }
    var cnt=counts[pair.key]||0;
    var bad=(maxc>0&&cnt<maxc);
    if(bad){under++;}
    gPairs.appendChild(sv("rect",{x:pxa,y:-pyb,width:pxb-pxa,height:pyb-pya,rx:0.18,ry:0.18,
      fill:"#8a97a4",opacity:0.12,stroke:bad?"#d93025":"none","stroke-width":bad?0.08:0}));
  }
  drawPairs.under=under;drawPairs.maxc=maxc;
}

function addRing(px,py,color){
  gOverlay.appendChild(sv("circle",{cx:px,cy:-py,r:ATOM_R+0.11,fill:"none",stroke:color,"stroke-width":0.07}));
}
function addCross(zoneName,px,py){
  var gm=zoneGeom[zoneName];if(!gm){return;}
  gOverlay.appendChild(sv("line",{x1:gm.x0,x2:gm.x1,y1:-py,y2:-py,stroke:"#6f9e2b",
    "stroke-width":0.045,"stroke-dasharray":"0.28 0.26",opacity:0.85}));
  gOverlay.appendChild(sv("line",{x1:px,x2:px,y1:-gm.y1,y2:-gm.y0,stroke:"#6f9e2b",
    "stroke-width":0.045,"stroke-dasharray":"0.28 0.26",opacity:0.85}));
}
function addCzPair(qa,qb,sites){
  var sa=sites[qa],sb=sites[qb];if(!sa||!sb){return;}
  var pa=physicalOfSite(sa),pb=physicalOfSite(sb);
  var zz=zoneOf(sa)||zoneOf(sb);
  var ang=(zz&&zz.pair_axis==="y")?90:0;
  var cx=(pa[0]+pb[0])/2,cy=-(pa[1]+pb[1])/2;
  var rx=Math.hypot(pa[0]-pb[0],pa[1]-pb[1])/2+0.30;
  gOverlay.appendChild(sv("ellipse",{cx:cx,cy:cy,rx:rx,ry:0.30,fill:"#f2a05a",opacity:0.55,
    stroke:"#e07b2a","stroke-width":0.07,transform:"rotate("+ang+" "+cx+" "+cy+")"}));
}
function currentOps(){
  var p=posFor(step);
  if(p<=0){return [];}
  var out=[];
  if(gran==="frame"){
    if(step<=0){return [];}
    var f=frames[step-1];if(!f){return [];}
    for(var i=f.start;i<f.end&&out.length<200;i++){out.push(i);}
  }else{out.push(p-1);}
  return out;
}
function drawOverlay(st){
  while(gOverlay.firstChild){gOverlay.removeChild(gOverlay.firstChild);}
  for(var b=0;b<atomBadge.length;b++){atomBadge[b].setAttribute("display","none");}
  var list=currentOps();
  for(var i=0;i<list.length;i++){
    var j=list[i],o=ops[j];if(!o){continue;}
    var code=o[0];
    if(code===4){
      var qm=o[1];if(qm>=0&&qm<nq&&atomBadge[qm]){atomBadge[qm].setAttribute("display","inline");}
    }else if(code===0||code===1){
      var qg=o[1];if(!(qg>=0&&qg<nq)){continue;}
      var ph=physicalOfSite(st.sites[qg]);
      addRing(ph[0],ph[1],code===0?"#6f9e2b":"#2f7fd0");
      if(code===0){var zn=zoneOf(st.sites[qg]);addCross(zn?zn.name:null,ph[0],ph[1]);}
    }else if(code===2){
      addCzPair(o[1],o[2],st.sites);
    }
  }
}

function drawAod(st){
  while(gAod.firstChild){gAod.removeChild(gAod.firstChild);}
  if(!st.curIsMove){return;}
  var k=ops[st.ci][1];var mv=moves[k];if(!mv){return;}
  var zz=zoneOf((mv.a&&mv.a[0])?mv.a[0]:null)||zoneOf((mv.b&&mv.b[0])?mv.b[0]:null);
  function vline(px,op){gAod.appendChild(sv("line",{x1:px,x2:px,y1:-bbox.y1,y2:-bbox.y0,
    stroke:"#c026a6",opacity:op,"stroke-width":0.05,"stroke-dasharray":"0.32 0.24"}));}
  function hline(py,op){gAod.appendChild(sv("line",{x1:bbox.x0,x2:bbox.x1,y1:-py,y2:-py,
    stroke:"#c026a6",opacity:op,"stroke-width":0.05,"stroke-dasharray":"0.32 0.24"}));}
  var cols=mv.cols||[],rows=mv.rows||[],ncols2=mv.ncols||[],nrows2=mv.nrows||[];
  for(var i=0;i<cols.length;i++){if(zz){vline(phys(zz,cols[i],"x"),0.75);}}
  for(i=0;i<rows.length;i++){if(zz){hline(phys(zz,rows[i],"y"),0.75);}}
  for(i=0;i<ncols2.length;i++){if(zz){vline(phys(zz,ncols2[i],"x"),0.32);}}
  for(i=0;i<nrows2.length;i++){if(zz){hline(phys(zz,nrows2[i],"y"),0.32);}}
  if(typeof mv.d==="number"&&mv.q&&mv.q.length){
    var q0=mv.q[0];
    var pa=physicalOfSite((mv.a&&mv.a[0])?mv.a[0]:null);
    var t=sv("text",{x:pa[0]+0.25,y:-pa[1]-0.18,"font-size":0.4,fill:"#a01f88","font-weight":600});
    t.textContent="d="+mv.d.toFixed(2);gAod.appendChild(t);
  }
}

function drawTrail(st){
  while(gTrail.firstChild){gTrail.removeChild(gTrail.firstChild);}
  if(!trail){return;}
  var mv=moves[trail.k];if(!mv||!Array.isArray(mv.q)){return;}
  var op=st.curIsMove?0.5:Math.max(0,0.36-0.08*trail.age);
  if(op<=0.02){return;}
  for(var j=0;j<mv.q.length;j++){
    var b=physicalOfSite(mv.b?mv.b[j]:null),a=physicalOfSite(mv.a?mv.a[j]:null);
    gTrail.appendChild(sv("line",{x1:b[0],y1:-b[1],x2:a[0],y2:-a[1],stroke:"#c026a6",
      opacity:op,"stroke-width":0.035}));
    for(var g=1;g<=3;g++){
      var f=g/4;
      gTrail.appendChild(sv("circle",{cx:b[0]+(a[0]-b[0])*f,cy:-(b[1]+(a[1]-b[1])*f),
        r:ATOM_R*0.8,fill:"none",stroke:"#c026a6",opacity:op*0.8*(1-f)+0.05,"stroke-width":0.03}));
    }
  }
}

/* ------------------------------------------------------------------ chips */
function chip(text,cls){
  var s=document.createElement("span");s.className="chip"+(cls?(" "+cls):"");
  s.textContent=text;return s;
}
function drawChips(st){
  var box=$("chips");
  while(box.firstChild){box.removeChild(box.firstChild);}
  box.appendChild(chip((meta.physics&&meta.physics!=="not checked")?String(meta.physics):"physics: not checked",
    meta.physics_ok===false?"warn":"good"));
  if(meta.status==="ok"&&meta.physics_ok===false){box.appendChild(chip("transversal CZ check failed","warn"));}
  var under=drawPairs.under||0,maxc=drawPairs.maxc||0;
  if(under>0){
    box.appendChild(chip("transversal CZ: "+under+" pair"+(under===1?"":"s")+" under-pulsed","warn"));
  }
  var ws=warnings.length;
  if(ws){box.appendChild(chip("spacing warnings: "+ws,"warn"));}
  if(!ops.length){box.appendChild(chip("no instructions (status: "+(meta.status||"?")+")","warn"));}
}

/* ------------------------------------------------------------------ wire diagram */
var wrap=$("wirewrap"),cvs=$("wirec"),pad=$("scrollpad");
var cw=10,rh=12,gutterW=46;
var wireScheduled=false;
function requestWire(){
  if(wireScheduled){return;}
  wireScheduled=true;
  requestAnimationFrame(function(){wireScheduled=false;drawWire();});
}
function reflow(){
  pad.style.width=Math.max(10,gutterW+Math.ceil(ncols*cw))+"px";
  pad.style.height=Math.max(10,Math.ceil(nq*rh)+2)+"px";
}
function colX(c,sx){return gutterW+c*cw-sx;}
function cursorCol(){
  var p=posFor(step);var cidx=p-1;
  if(cidx>=0&&colOf[cidx]!==undefined){return colOf[cidx];}
  return colOf[0]||0;
}
function drawWire(){
  if(!cvs){return;}
  var W=wrap.clientWidth,H=wrap.clientHeight;
  if(W<=0||H<=0){return;}
  var dpr=window.devicePixelRatio||1;
  var pxW=Math.round(W*dpr),pxH=Math.round(H*dpr);
  if(cvs.width!==pxW||cvs.height!==pxH){cvs.width=pxW;cvs.height=pxH;cvs.style.width=W+"px";cvs.style.height=H+"px";}
  var g=cvs.getContext("2d");
  g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,W,H);
  g.fillStyle="#fbfcfd";g.fillRect(0,0,W,H);
  if(!N){
    g.fillStyle="#8c1d18";g.font="13px sans-serif";g.textAlign="left";
    g.fillText("no instructions to draw",12,H/2);
    return;
  }
  rh=clamp(Math.floor((H-2)/Math.max(1,nq)),8,22);
  reflow();
  var sx=wrap.scrollLeft,sy=wrap.scrollTop;
  var c0=Math.max(0,Math.floor(sx/cw)-1);
  var c1=Math.min(ncols-1,Math.ceil((sx+(W-gutterW))/cw)+1);
  var r0=Math.max(0,Math.floor(sy/rh)-1);
  var r1=Math.min(nq-1,Math.ceil((sy+H)/rh)+1);
  var qy=function(q){return q*rh+rh/2-sy;};
  /* column backgrounds */
  for(var c=c0;c<=c1;c++){
    if(colmove&&colmove[c]){
      g.fillStyle="rgba(192,38,166,0.055)";
      g.fillRect(gutterW+c*cw-sx,0,cw,H);
    }
  }
  /* wires */
  g.strokeStyle="#d7dee6";g.lineWidth=1;
  for(var r=r0;r<=r1;r++){
    var y=Math.round(qy(r))+0.5;
    g.beginPath();g.moveTo(gutterW,y);g.lineTo(W,y);g.stroke();
  }
  /* instructions in visible columns */
  var lo=colStart[c0]||0;
  var hi=colStart[Math.min(ncols,c1+1)];
  if(hi===undefined){hi=N;}
  g.textAlign="center";
  for(var zz=lo;zz<hi;zz++){
    var idx=order[zz];if(idx===undefined){continue;}
    var o=ops[idx];if(!o){continue;}
    var x=colX(colOf[idx]||0,sx)+cw/2;
    if(x<-cw||x>W+cw){continue;}
    var code=o[0];
    if(code===0||code===1){
      var q=o[1];if(!(q>=r0&&q<=r1)){continue;}
      var yy=qy(q);
      g.fillStyle=code===0?"#6f9e2b":"#2f7fd0";
      var hh=Math.max(3,rh*0.46);
      g.fillRect(x-2.5,yy-hh/2,5,hh);
    }else if(code===2){
      var qa=o[1],qb=o[2];
      if(!((qa>=r0&&qa<=r1)||(qb>=r0&&qb<=r1))){continue;}
      g.strokeStyle="#e07b2a";g.lineWidth=1.4;
      g.beginPath();g.moveTo(x,qy(qa));g.lineTo(x,qy(qb));g.stroke();
      g.fillStyle="#e07b2a";
      g.beginPath();g.arc(x,qy(qa),2.1,0,6.2832);g.fill();
      g.beginPath();g.arc(x,qy(qb),2.1,0,6.2832);g.fill();
    }else if(code===4){
      var qm=o[1];if(!(qm>=r0&&qm<=r1)){continue;}
      var ym=qy(qm);
      g.fillStyle="#aeb8c2";g.fillRect(x-2.5,ym-rh*0.2,5,rh*0.4);
      g.fillStyle="#4b5560";g.fillRect(x-2.5,ym+rh*0.2-1.5,5,1.5);
    }else if(code===5){
      g.strokeStyle="#b9c4ce";g.lineWidth=1;g.setLineDash([2,3]);
      g.beginPath();g.moveTo(x,0);g.lineTo(x,H);g.stroke();g.setLineDash([]);
    }else if(code===3){
      var mv=moves[o[1]];if(!mv||!Array.isArray(mv.q)||!mv.q.length){continue;}
      var qmin=Infinity,qmax=-Infinity;
      for(var mj=0;mj<mv.q.length;mj++){var qq=mv.q[mj];if(qq<qmin){qmin=qq;}if(qq>qmax){qmax=qq;}}
      if(qmax<r0||qmin>r1){continue;}
      g.strokeStyle="#c026a6";g.lineWidth=1.6;
      g.beginPath();g.moveTo(x,qy(qmin));g.lineTo(x,qy(qmax));g.stroke();
      g.fillStyle="#c026a6";
      for(mj=0;mj<mv.q.length;mj++){
        if(mv.q[mj]>=r0&&mv.q[mj]<=r1){g.fillRect(x-2,qy(mv.q[mj])-1,4,2);}
      }
    }else{
      var qo2=o[1];
      if(qo2>=r0&&qo2<=r1){g.fillStyle="#9aa7b3";g.fillRect(x-2,qy(qo2)-2,4,4);}
    }
  }
  /* error column hatch */
  if(errCol>=0){
    var ex=colX(errCol,sx);
    if(ex>-cw&&ex<W){
      g.fillStyle="rgba(217,48,37,0.13)";g.fillRect(ex,0,cw,H);
      g.strokeStyle="#d93025";g.lineWidth=1.5;
      g.beginPath();g.moveTo(ex+cw/2,0);g.lineTo(ex+cw/2,H);g.stroke();
    }
  }
  /* gutter over everything */
  g.fillStyle="#f2f6f9";g.fillRect(0,0,gutterW,H);
  g.strokeStyle="#dde4ea";g.lineWidth=1;
  g.beginPath();g.moveTo(gutterW-0.5,0);g.lineTo(gutterW-0.5,H);g.stroke();
  g.fillStyle="#6b7785";g.font="10px ui-monospace,Menlo,monospace";g.textAlign="right";
  for(r=r0;r<=r1;r++){g.fillText("q"+r,gutterW-7,qy(r)+3.5);}
  /* cursor */
  var cc=cursorCol();
  var cx=colX(cc,sx)+cw/2;
  if(cx>-cw&&cx<W){
    g.fillStyle="rgba(31,95,191,0.06)";g.fillRect(colX(cc,sx),0,cw,H);
    g.strokeStyle="#1f5fbf";g.lineWidth=1.5;
    g.beginPath();g.moveTo(cx,0);g.lineTo(cx,H);g.stroke();
  }
  /* hover column */
  if(hoverInstr>=0&&colOf[hoverInstr]!==undefined){
    var hx=colX(colOf[hoverInstr],sx);
    if(hx>-cw&&hx<W){g.fillStyle="rgba(31,95,191,0.07)";g.fillRect(hx,0,cw,H);}
  }
}
function ensureCursor(){
  var cc=cursorCol();
  var x=gutterW+cc*cw-wrap.scrollLeft;
  var avail=wrap.clientWidth-gutterW;
  if(x<gutterW||x>wrap.clientWidth-cw*2){
    wrap.scrollLeft=Math.max(0,gutterW+cc*cw-gutterW-0.35*avail);
    requestWire();
  }
}

/* ------------------------------------------------------------------ tooltip / clicks */
var tip=$("tip");
var instrName=["rz","rx","cz","move","measure","barrier","other"];
function instrAt(col,row){
  if(col<0||col>=ncols){return -1;}
  var lo=colStart[col]||0,hi=colStart[Math.min(ncols,col+1)];
  if(hi===undefined){hi=N;}
  var fallback=-1;
  for(var z=lo;z<hi;z++){
    var idx=order[z];var o=ops[idx];if(!o){continue;}
    if(fallback<0){fallback=idx;}
    if(o[0]===2&&(o[1]===row||o[2]===row)){return idx;}
    if((o[0]===0||o[0]===1||o[0]===4||o[0]===6)&&o[1]===row){return idx;}
    if(o[0]===3){var mv=moves[o[1]];if(mv&&Array.isArray(mv.q)&&mv.q.indexOf(row)>=0){return idx;}}
  }
  return fallback;
}
function showTip(idx,clientX,clientY){
  if(idx<0||idx>=N){tip.classList.add("hidden");return;}
  var o=ops[idx]||[];var code=o[0]|0;
  var txt="#"+idx+" "+(instrName[code]||"other");
  if(code===0||code===1){txt+=" q"+o[1]+"  \u03b8="+num(angles[idx],4);}
  else if(code===2){txt+=" q"+o[1]+",q"+o[2];}
  else if(code===4){txt+=" q"+o[1];}
  else if(code===3){var mv=moves[o[1]];txt+=" q"+((mv&&mv.q)?mv.q.join(",q"):"?")+"  d="+num(mv&&mv.d,3);}
  txt+="\ncol "+num(colOf[idx],0)+(depthAt[idx]!==undefined?("  depth "+depthAt[idx]):"");
  tip.textContent=txt;
  tip.style.whiteSpace="pre";
  tip.classList.remove("hidden");
  var bw=tip.offsetWidth||200,bh=tip.offsetHeight||40;
  var x=clientX+14,y=clientY+14;
  if(x+bw>window.innerWidth-6){x=clientX-bw-10;}
  if(y+bh>window.innerHeight-6){y=clientY-bh-10;}
  tip.style.left=Math.max(4,x)+"px";tip.style.top=Math.max(4,y)+"px";
}
function hideTip(){tip.classList.add("hidden");}
function jumpToInstr(i){
  if(i<0||i>=N){return;}
  pause();
  if(gran==="instruction"){setStep(i+1);}
  else{
    for(var f=0;f<frames.length;f++){if(i<frames[f].end){setStep(f+1);break;}}
  }
}
wrap.addEventListener("mousemove",function(ev){
  var rect=wrap.getBoundingClientRect();
  var sx=wrap.scrollLeft,sy=wrap.scrollTop;
  var lx=ev.clientX-rect.left,ly=ev.clientY-rect.top;
  if(lx<gutterW){
    hoverInstr=-1;hideTip();requestWire();return;
  }
  var col=Math.floor((lx-gutterW+sx)/cw);
  var row=Math.floor((ly+sy)/rh);
  if(row<0||row>=nq){hoverInstr=-1;hideTip();requestWire();return;}
  var idx=instrAt(col,row);
  if(idx!==hoverInstr){hoverInstr=idx;requestWire();}
  showTip(idx,ev.clientX,ev.clientY);
});
wrap.addEventListener("mouseleave",function(){hoverInstr=-1;hideTip();requestWire();});
wrap.addEventListener("click",function(ev){
  var rect=wrap.getBoundingClientRect();
  var lx=ev.clientX-rect.left;
  if(lx<gutterW){return;}
  var col=Math.floor((lx-gutterW+wrap.scrollLeft)/cw);
  if(col<0||col>=ncols){return;}
  var first=(colFirst[col]!==undefined&&colFirst[col]>=0)?colFirst[col]:-1;
  if(first<0){var lo=colStart[col]||0;first=(lo<N)?order[lo]:-1;}
  jumpToInstr(first);
});
wrap.addEventListener("scroll",function(){requestWire();});
window.addEventListener("resize",function(){requestWire();});

/* ------------------------------------------------------------------ stats / header */
function statusInfo(){
  if(meta.status==="ok"&&meta.physics_ok!==false){return {cls:"ok",text:"valid"};}
  if(meta.status==="ok"){return {cls:"warn",text:"physics failed"};}
  return {cls:"bad",text:"INVALID"};
}
function buildHeader(){
  var hl=$("hdrleft"),hr=$("hdrright");
  function span(cls,txt){var s=document.createElement("span");if(cls){s.className=cls;}s.textContent=txt;return s;}
  hl.appendChild(span("name",meta.compiler||"compiler"));
  hl.appendChild(span("sep","\u00b7"));
  hl.appendChild(span("",meta.label||"circuit"));
  if(meta.file){hl.appendChild(span("file",String(meta.file)));}
  var si=statusInfo();
  var b=document.createElement("span");b.className="badge "+si.cls;b.textContent=si.text;hr.appendChild(b);
  var ph=document.createElement("span");ph.className="file";
  ph.textContent=(meta.physics_ok===false?"physics: "+String(meta.physics||"failed"):"physics ok");
  hr.appendChild(ph);
}
function buildBanner(){
  var ban=$("banner");
  var vio=Array.isArray(meta.violations)?meta.violations:[];
  if(meta.status&&meta.status!=="ok"){
    ban.classList.remove("hidden");
    var t=document.createElement("div");t.className="etitle";
    t.textContent=(meta.status==="compile_error"?"compile error":"INVALID compiled output");
    ban.appendChild(t);
    if(meta.error){var pre=document.createElement("pre");pre.textContent=String(meta.error);ban.appendChild(pre);}
    if(vio.length){
      var det=document.createElement("details");var sm=document.createElement("summary");
      sm.textContent="violations ("+vio.length+")";det.appendChild(sm);
      var ul=document.createElement("ul");
      for(var i=0;i<vio.length;i++){var li=document.createElement("li");li.textContent=String(vio[i]);ul.appendChild(li);}
      det.appendChild(ul);ban.appendChild(det);
    }
  }else{ban.classList.add("hidden");}
}
function updateStats(st){
  var p=st.p;
  var em=countLess(moveInstr,p);
  var cum=mp[em]||0;
  var thisMove=null;
  if(st.ci>=0&&ops[st.ci]&&ops[st.ci][0]===3){
    var mv=moves[ops[st.ci][1]];thisMove=mv?(mv.d||0):0;
  }
  var depth=p<=0?0:(depthAt[p-1]||0);
  var cz=czPrefix[p]||0;
  var sp=mw[em]||0;
  var naive=meta.naive&&typeof meta.naive.move_distance==="number"?meta.naive.move_distance:null;
  var parts=[];
  function add(k,v){parts.push([k,v]);}
  add("instruction",Math.min(p,N)+"/"+N);
  add("move",em+"/"+moveInstr.length);
  add("this move",thisMove===null?"\u2013":"d="+num(thisMove,3));
  add("cum. distance",num(cum,1)+(naive!==null?(" ("+num(naive,1)+", "+(naive>0?num(100*cum/naive,1):"\u2013")+"% of it)"):""));
  add("depth",depth+"/"+((meta.metrics&&typeof meta.metrics.depth==="number")?meta.metrics.depth:"\u2013"));
  add("CZ",String(cz));
  add("spacing",String(sp));
  add("steps",String(em));
  var stats=$("stats");
  while(stats.firstChild){stats.removeChild(stats.firstChild);}
  for(var i=0;i<parts.length;i++){
    var k=document.createElement("span");k.className="k";k.textContent=parts[i][0]+": ";
    var v=document.createElement("span");v.className="v";v.textContent=parts[i][1];
    stats.appendChild(k);stats.appendChild(v);
    if(i<parts.length-1){var s=document.createElement("span");s.className="k";s.textContent="\u00b7";stats.appendChild(s);}
  }
  if(meta.metrics_partial){var mp2=document.createElement("span");mp2.className="chip warn";
    mp2.textContent="partial (replay stopped at the first violation)";stats.appendChild(mp2);}
  var mismatch=false;
  if(p>=N&&meta.metrics){
    if(typeof meta.metrics.move_distance==="number"&&Math.abs(cum-meta.metrics.move_distance)>1e-6){mismatch=true;}
    if(typeof meta.metrics.depth==="number"&&depth!==meta.metrics.depth){mismatch=true;}
  }
  var mm=document.createElement("span");
  if(mismatch){mm.className="chip mismatch";mm.textContent="MISMATCH";}
  else if(p>=N&&meta.metrics){mm.className="chip good";mm.textContent="\u2713 matches metrics";}
  else{mm.className="chip";mm.textContent="";}
  if(mm.textContent){stats.appendChild(mm);}
}

/* ------------------------------------------------------------------ render */
function render(){
  var st=computeState();
  updateAtomBase(st);
  drawPairs(st);
  if(st.curIsMove){
    var k=ops[st.ci][1];
    if(animState.ci!==st.ci||animState.k!==k){beginMoveAnim(st);startAnimLoop();}
  }else{
    animState.ci=-1;animState.qs=[];animState.t=1;
    if(rafAnim){cancelAnimationFrame(rafAnim);rafAnim=0;}
  }
  updateMovedAtoms();
  drawOverlay(st);
  drawAod(st);
  drawTrail(st);
  drawChips(st);
  updateStats(st);
  requestWire();
}
function syncControls(){
  var ns=nstepsFor();
  var sc=$("scrub");
  sc.max=String(Math.max(0,ns-1));sc.value=String(step);
  $("speedv").textContent=Math.round(speed)+"/s";
  $("zoomv").textContent=zoom.toFixed(1)+"x";
  var dis=N===0;
  $("btnPlay").disabled=dis;$("btnBack").disabled=dis;$("btnFwd").disabled=dis;
  $("btnReset").disabled=dis;
  $("btnPlay").textContent=playing?"\u23f8":"\u25b6";
}
function onStepChanged(){
  var st=computeState();
  if(st.curIsMove){trail={k:ops[st.ci][1],age:0};}
  else if(trail){trail.age++;if(trail.age>4){trail=null;}}
}
function setStep(s,keepScroll){
  var ns=nstepsFor();
  step=clamp(Math.round(s),0,ns-1);
  if(step>=ns-1&&playing){pause();}
  onStepChanged();
  syncControls();
  render();
  if(!keepScroll){ensureCursor();}
}

/* ------------------------------------------------------------------ playback */
function playTick(now){
  if(!playing){return;}
  if(!lastT){lastT=now;}
  var dt=(now-lastT)/1000;lastT=now;
  acc+=dt*speed;
  var adv=Math.floor(acc);
  if(adv>0){
    acc-=adv;
    var ns=nstepsFor();
    var ns2=Math.min(ns-1,step+adv);
    setStep(ns2);
    if(ns2>=ns-1){pause();return;}
  }
  rafPlay=requestAnimationFrame(playTick);
}
function play(){
  if(N===0){return;}
  if(playing){return;}
  var ns=nstepsFor();
  if(step>=ns-1){setStep(0);}
  playing=true;lastT=0;acc=0;
  syncControls();
  rafPlay=requestAnimationFrame(playTick);
}
function pause(){
  playing=false;
  if(rafPlay){cancelAnimationFrame(rafPlay);rafPlay=0;}
  syncControls();
}
function togglePlay(){if(playing){pause();}else{play();}}

/* ------------------------------------------------------------------ wiring controls */
function speedFromSlider(v){return Math.pow(200,clamp(v,0,100)/100);}
$("speed").addEventListener("input",function(){speed=speedFromSlider(parseFloat(this.value));syncControls();});
$("gran").addEventListener("change",function(){
  var p=posFor(step);
  gran=this.value==="frame"?"frame":"instruction";
  step=clamp(stepForPos(p),0,nstepsFor()-1);
  onStepChanged();syncControls();render();ensureCursor();
});
$("scrub").addEventListener("input",function(){var v=parseFloat(this.value);pause();setStep(v);});
$("zoom").addEventListener("input",function(){
  zoom=clamp(parseFloat(this.value)||1,0.4,3);
  $("zoomv").textContent=zoom.toFixed(1)+"x";
  cw=10*zoom;reflow();requestWire();ensureCursor();
});
$("lblToggle").addEventListener("change",function(){
  if(this.checked){svg.classList.remove("nolabels");}else{svg.classList.add("nolabels");}
});
$("btnPlay").addEventListener("click",togglePlay);
$("btnBack").addEventListener("click",function(){pause();setStep(step-1);});
$("btnFwd").addEventListener("click",function(){pause();setStep(step+1);});
$("btnReset").addEventListener("click",function(){pause();setStep(0);});
document.addEventListener("keydown",function(ev){
  var tag=(ev.target&&ev.target.tagName)?ev.target.tagName.toUpperCase():"";
  if(tag==="INPUT"||tag==="SELECT"||tag==="TEXTAREA"||ev.target&&ev.target.isContentEditable){return;}
  var k=ev.key;
  if(k===" "||k==="Spacebar"){ev.preventDefault();togglePlay();return;}
  if(k==="ArrowLeft"){ev.preventDefault();pause();setStep(step-(ev.shiftKey?10:1));return;}
  if(k==="ArrowRight"){ev.preventDefault();pause();setStep(step+(ev.shiftKey?10:1));return;}
  if(k==="Home"){ev.preventDefault();pause();setStep(0);return;}
  if(k==="End"){ev.preventDefault();pause();setStep(nstepsFor()-1);return;}
});

/* ------------------------------------------------------------------ init */
speed=speedFromSlider(parseFloat($("speed").value));
cw=10*zoom;
$("lblToggle").checked=nq<25;
if(nq>=25){svg.classList.add("nolabels");}
buildHeader();
buildBanner();
syncControls();
render();
/* initial position: error column or start */
if(errCol>=0){
  var avail=wrap.clientWidth-gutterW;
  wrap.scrollLeft=Math.max(0,gutterW+errCol*cw-gutterW-0.2*Math.max(avail,0));
  requestWire();
}
window.addEventListener("load",function(){requestWire();});
})();"""

# --------------------------------------------------------------------------- JS: index
JS_INDEX = r"""(function(){
"use strict";
function $(id){return document.getElementById(id);}
var R=(typeof ROWS!=="undefined"&&ROWS)?ROWS:{};
if(typeof R==="string"){try{R=JSON.parse(R);}catch(e){R={};}}
var compilers=Array.isArray(R.compilers)?R.compilers:[];
var gen=$("gen");
gen.textContent=R.generated?("generated "+R.generated):"";

/* ---- flatten every compiler x circuit into one record ---- */
var recs=[];
function numOrNull(v){return (typeof v==="number"&&!isNaN(v))?v:null;}
for(var i=0;i<compilers.length;i++){
  var c=compilers[i]||{};
  var name=c.compiler||"?";
  var circuits=Array.isArray(c.circuits)?c.circuits:[];
  for(var j=0;j<circuits.length;j++){
    var x=circuits[j]||{};
    var m=x.metrics||null;
    var nv=x.naive||null;
    var dist=m?numOrNull(m.move_distance):null;
    var depth=m?numOrNull(m.depth):null;
    var ndist=nv?numOrNull(nv.move_distance):null;
    var ndepth=nv?numOrNull(nv.depth):null;
    var ok=(x.status==="ok")&&(x.physics_ok!==false);
    recs.push({
      compiler:name,
      label:x.label||"",
      file:x.file||"",
      family:x.family||"",
      qubits:numOrNull(x.qubits),
      dist:dist,ndist:ndist,
      ddist:(dist!==null&&ndist)?100*(dist-ndist)/ndist:null,
      depth:depth,ndepth:ndepth,
      ddepth:(depth!==null&&ndepth)?100*(depth-ndepth)/ndepth:null,
      status:x.status||"",valid:ok,
      page:x.page||null,error:x.error||null,physics_ok:x.physics_ok!==false,
      compile_s:numOrNull(x.compile_s),bytes:numOrNull(x.bytes)
    });
  }
}

/* ---- summary cards ---- */
var fmt=function(v,d){return (v===null||v===undefined||isNaN(v))?"\u2013":Number(v).toFixed(d);};
var pct=function(a,b){return (a===null||!b)?null:100*a/b;};
var cards=$("cards");
for(i=0;i<compilers.length;i++){
  var cc=compilers[i]||{};
  var sm=cc.summary||{};
  var card=document.createElement("div");card.className="card";
  var head=document.createElement("div");head.className="cname";
  var nm=document.createElement("span");nm.textContent=cc.compiler||"?";head.appendChild(nm);
  var bad=document.createElement("span");
  var okc=(sm.valid!==false);
  bad.className="badge "+(okc?"ok":"bad");bad.textContent=okc?"valid":"INVALID";
  head.appendChild(bad);card.appendChild(head);
  var met=document.createElement("div");met.className="metrics";
  var pdist=pct(numOrNull(sm.distance),numOrNull(sm.naive_distance));
  var pdep=pct(numOrNull(sm.depth),numOrNull(sm.naive_depth));
  var metLines=[
    ["score",numOrNull(sm.score)===null?"\u2013":Number(sm.score).toFixed(4)],
    ["distance",(pdist===null?"\u2013":pdist.toFixed(1)+"% of naive")+
       " ("+(numOrNull(sm.distance_improvement)===null?"\u2013":(100*sm.distance_improvement).toFixed(1)+"% better")+")"],
    ["depth",(pdep===null?"\u2013":pdep.toFixed(1)+"% of naive")+
       " ("+(numOrNull(sm.depth_improvement)===null?"\u2013":(100*sm.depth_improvement).toFixed(1)+"% better")+")"],
    ["circuits",String(Array.isArray(cc.circuits)?cc.circuits.length:0)]
  ];
  for(var ml=0;ml<metLines.length;ml++){
    var d=document.createElement("div");d.textContent=metLines[ml][0]+": "+metLines[ml][1];met.appendChild(d);
  }
  card.appendChild(met);
  var regen=document.createElement("a");regen.className="regen";
  regen.href="/regen?compiler="+encodeURIComponent(cc.compiler_path||cc.compiler||"");
  regen.title="only works when served by bench/viz.py serve";
  regen.textContent="regenerate \u21bb";card.appendChild(regen);
  cards.appendChild(card);
}

/* ---- sortable / filterable table ---- */
var state={filter:"",key:"label",dir:1};
var tbody=$("tbody");
var COLCOUNT=10;
function cmp(a,b,key,dir){
  var va=a[key],vb=b[key];
  if(key==="valid"){va=va?1:0;vb=vb?1:0;}
  var an=(va===null||va===undefined),bn=(vb===null||vb===undefined);
  if(an&&bn){return 0;}
  if(an){return 1;}
  if(bn){return -1;}
  if(typeof va==="number"&&typeof vb==="number"){return (va-vb)*dir;}
  var sa=String(va).toLowerCase(),sb=String(vb).toLowerCase();
  return sa<sb?-dir:(sa>sb?dir:0);
}
function deltaCell(v){
  var td=document.createElement("td");td.className="num";
  if(v===null||v===undefined||isNaN(v)){td.textContent="\u2013";return td;}
  td.textContent=(v>0?"+":"")+v.toFixed(1)+"%";
  td.className="num "+(v<0?"neg":(v>0?"pos":""));
  return td;
}
function mkTd(text,cls){var e=document.createElement("td");e.textContent=text;if(cls){e.className=cls;}return e;}
function renderRows(){
  while(tbody.firstChild){tbody.removeChild(tbody.firstChild);}
  var f=state.filter.toLowerCase();
  var rows=[];
  for(var i=0;i<recs.length;i++){
    var r=recs[i];
    if(f){
      var hay=(r.compiler+" "+r.label+" "+r.file+" "+r.family).toLowerCase();
      if(hay.indexOf(f)<0){continue;}
    }
    rows.push(r);
  }
  rows.sort(function(a,b){return cmp(a,b,state.key,state.dir);});
  for(i=0;i<rows.length;i++){
    var r=rows[i];
    var tr=document.createElement("tr");
    tr.appendChild(mkTd(r.compiler));
    var tdl=document.createElement("td");
    if(r.page){var a=document.createElement("a");a.href=r.page;a.textContent=r.label;tdl.appendChild(a);}
    else{tdl.textContent=r.label;}
    tr.appendChild(tdl);
    var tdf=document.createElement("td");
    var sp=document.createElement("span");sp.className="filecell";sp.title=r.file;sp.textContent=r.file;tdf.appendChild(sp);
    tr.appendChild(tdf);
    tr.appendChild(mkTd(r.family));
    tr.appendChild(mkTd(r.qubits===null?"\u2013":String(r.qubits),"num"));
    tr.appendChild(mkTd(fmt(r.dist,1),"num"));
    tr.appendChild(deltaCell(r.ddist));
    tr.appendChild(mkTd(fmt(r.depth,0),"num"));
    tr.appendChild(deltaCell(r.ddepth));
    var tdv=document.createElement("td");
    var b=document.createElement("span");
    if(r.status==="ok"&&r.valid){b.className="badge ok";b.textContent="valid";}
    else if(r.status==="ok"){b.className="badge warn";b.textContent="physics";}
    else{b.className="badge bad";b.textContent="INVALID";}
    if(r.error){b.title=String(r.error);}
    tdv.appendChild(b);tr.appendChild(tdv);
    tbody.appendChild(tr);
  }
  var cnt=$("count");cnt.textContent=rows.length+" / "+recs.length+" circuits";
  if(rows.length===0){
    var tr2=document.createElement("tr");var td2=document.createElement("td");
    td2.colSpan=COLCOUNT;td2.textContent="no matching circuits";td2.style.textAlign="center";
    td2.style.color="#6b7785";tr2.appendChild(td2);tbody.appendChild(tr2);
  }
}
var ths=document.querySelectorAll("#tbl thead th");
for(i=0;i<ths.length;i++){
  ths[i].addEventListener("click",function(){
    var k=this.getAttribute("data-k");
    if(state.key===k){state.dir=-state.dir;}else{state.key=k;state.dir=1;}
    for(var t=0;t<ths.length;t++){ths[t].querySelector(".ar").textContent="";}
    var ar=this.querySelector(".ar");if(ar){ar.textContent=state.dir>0?"\u25b2":"\u25bc";}
    renderRows();
  });
}
$("filter").addEventListener("input",function(){state.filter=this.value;renderRows();});
(function markFirst(){
  for(var t=0;t<ths.length;t++){
    if(ths[t].getAttribute("data-k")===state.key){
      var ar=ths[t].querySelector(".ar");if(ar){ar.textContent="\u25b2";}
    }
  }
})();
renderRows();

/* ---- live status (regeneration), best effort: silently absent on file:// ---- */
var statusline=$("statusline");
var wasRunning=false;
function poll(){
  var p;
  try{p=fetch("status.json",{cache:"no-store"});}catch(e){return;}
  p.then(function(r){return r.json();}).then(function(s){
    if(!s||typeof s!=="object"){return;}
    if(s.running){
      wasRunning=true;
      statusline.classList.remove("hidden");statusline.classList.add("on");
      statusline.textContent="regenerating "+(s.compiler||"?")+": "+(s.done||0)+"/"+(s.total||"?")+" \u2026";
    }else{
      statusline.classList.add("hidden");statusline.classList.remove("on");
      if(wasRunning){wasRunning=false;location.reload();}
    }
  }).catch(function(){});
}
setInterval(poll,2000);
})();"""


if __name__ == "__main__":
    raise SystemExit(main())
