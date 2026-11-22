"""Build the relay pitch deck: template.html + numbers from results/ -> index.html.

    python slides/relay/build.py

Every number on the slides comes from results/relay.json, results/solo2.json, results/naive.json
and docs/bounds.md, so the deck cannot drift from the measured results.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NNBSP = " "


def load(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def num(x, digits=0):
    return f"{x:,.{digits}f}".replace(",", NNBSP)


def depth_bounds():
    """file -> the depth lower bound of docs/bounds.md (2 cz pulses per hub block)."""
    out = {}
    for line in (ROOT / "docs" / "bounds.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 9 and cells[1].endswith(".qasm"):
            out[cells[1]] = int(cells[8])
    return out


def main():
    relay, solo2, naive = load("relay"), load("solo2"), load("naive")
    final = load("relay_best_l3")
    fs = final["summary"]
    by = lambda run: {r["label"]: r for r in run["records"]}
    r_rec, s_rec, n_rec = by(relay), by(solo2), by(naive)
    bounds = depth_bounds()

    circuits = []
    for label, r in r_rec.items():
        m = re.search(r"(\d+)q_(hea|ihva)_(.+)_p(\d)\.qasm$", r["file"])
        rel = r["file"].split("metric_benchmarks/")[-1]
        circuits.append({
            "label": label, "n": int(m.group(1)), "family": m.group(2), "variant": m.group(3),
            "p": int(m.group(4)), "bound": bounds.get(rel),
            "naive": [n_rec[label]["metrics"]["move_distance"], n_rec[label]["metrics"]["depth"]],
            "solo2": [s_rec[label]["metrics"]["move_distance"], s_rec[label]["metrics"]["depth"]],
            "relay": [r["metrics"]["move_distance"], r["metrics"]["depth"]],
            "compile_s": r["compile_s"],
        })

    rs, ss = relay["summary"], solo2["summary"]
    assert fs["valid"] and not final["failed"]
    assert rs["valid"] and not relay["failed"] and relay["mode"] == "full"
    big = next(c for c in circuits if (c["n"], c["variant"], c["p"]) == (50, "stagger", 3))
    hub_deg = big["n"] - 1
    values = {
        "relay_score": f"{rs['score']:.2f}",
        "final_score": f"{fs['score']:.2f}",
        "final_dist": num(fs["distance"]),
        "final_depth": num(fs["depth"]),
        "solo2_score": f"{ss['score']:.2f}",
        "score_gain": f"{rs['score'] - ss['score']:.2f}",
        "relay_depth": num(rs["depth"]),
        "solo2_depth": num(ss["depth"]),
        "depth_cut": f"{100 * (1 - rs['depth'] / ss['depth']):.0f}",
        "relay_dist": num(rs["distance"]),
        "solo2_dist": num(ss["distance"]),
        "dist_change": f"{100 * (rs['distance'] / ss['distance'] - 1):+.1f}",
        "dist_impr": f"{rs['distance_improvement']:.1f}",
        "depth_impr": f"{rs['depth_improvement']:.1f}",
        "n_valid": f"{rs['n'] - len(relay['failed'])}{NNBSP}/{NNBSP}{rs['n']}",
        "big_solo2": str(big["solo2"][1]),
        "big_relay": str(big["relay"][1]),
        "big_bound": str(big["bound"]),
        "big_cut": f"{100 * (1 - big['relay'][1] / big['solo2'][1]):.0f}",
        "hub_deg": str(hub_deg),
        "hub_pulses": str(2 * hub_deg),
        "max_compile": f"{max(c['compile_s'] for c in circuits):.0f}",
    }
    assert big["bound"] == 2 * hub_deg * big["p"]

    html = (HERE / "template.html").read_text()
    used = set(re.findall(r"\{\{(\w+)\}\}", html))
    missing = used - set(values)
    if missing:
        raise SystemExit(f"template uses unknown placeholders: {sorted(missing)}")
    for k in used:
        html = html.replace("{{" + k + "}}", values[k])
    data = {"circuits": circuits, "summary": {"relay": rs, "solo2": ss}}
    html = html.replace("__DECK_DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (HERE / "index.html").write_text(html)
    print(f"wrote {HERE / 'index.html'} ({len(html) / 1024:.0f} kB)")
    for k in sorted(values):
        print(f"  {k:12} {values[k]}")


if __name__ == "__main__":
    main()
