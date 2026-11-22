#!/usr/bin/env python3
"""MIP-LNS on cached plans, recording every improving rewrite (for pattern mining).
    python l3/mine.py <out json> <W> <tl> <pkl> [<pkl> ...]   (one process per pkl)
Stoppable: per circuit, <out>/<stem>.ckpt is saved every 25 windows and resumed on restart;
finished circuits are kept in <out>/<stem>.json, live improvements in <out>/<stem>.log."""
import concurrent.futures as cf, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def one(a):
    path, W, tl, outdir = a
    import logging; logging.disable(logging.WARNING)
    import milp_lns as L
    stem = Path(path).stem
    done = Path(outdir) / (stem + ".json")
    if done.exists():                      # finished earlier: reuse
        return json.loads(done.read_text())
    logf = open(Path(outdir) / (stem + ".log"), "a")
    rec = []
    res, out, load = L.lns(path, W, tl, 1, log=lambda s: (logf.write(s + "\n"), logf.flush()), record=rec,
                           ckpt=Path(outdir) / (stem + ".ckpt"))
    res["records"] = rec
    done.write_text(json.dumps(res, default=str))
    return res

if __name__ == "__main__":
    outp, W, tl, files = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
    outdir = Path(outp).with_suffix("")      # per-circuit results, logs and checkpoints
    outdir.mkdir(parents=True, exist_ok=True)
    allr = []
    with cf.ProcessPoolExecutor(min(6, len(files))) as ex:
        for r in ex.map(one, [(f, W, tl, str(outdir)) for f in files]):
            print(r["file"], round(r["base"], 2), "->", round(r["final"], 2), f"({len(r['records'])} rewrites, {r['secs']} s)", flush=True)
            allr.append(r)
    json.dump(allr, open(outp, "w"), default=str)
