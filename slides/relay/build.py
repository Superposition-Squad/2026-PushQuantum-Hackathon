"""Build the relay pitch deck: template.html + numbers from results/ -> index.html.

    python slides/relay/build.py

Every number on the slides comes from results/<BEST>.json, results/naive.json and docs/bounds.md, so
the deck cannot drift from the measured results. The deck presents one compiler (BEST) against the
official naive baseline only.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BEST = "relay_best_l3"
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
    best, naive = load(BEST), load("naive")
    n_rec = {r["label"]: r for r in naive["records"]}
    bounds = depth_bounds()

    circuits = []
    for r in best["records"]:
        m = re.search(r"(\d+)q_(hea|ihva)_(.+)_p(\d)\.qasm$", r["file"])
        rel = r["file"].split("metric_benchmarks/")[-1]
        nm = n_rec[r["label"]]["metrics"]
        circuits.append({
            "label": r["label"], "n": int(m.group(1)), "family": m.group(2), "variant": m.group(3),
            "p": int(m.group(4)), "bound": bounds.get(rel),
            "naive": [nm["move_distance"], nm["depth"]],
            "best": [r["metrics"]["move_distance"], r["metrics"]["depth"]],
        })

    s = best["summary"]
    assert s["valid"] and not best["failed"] and best["mode"] == "full"
    big = next(c for c in circuits if (c["n"], c["variant"], c["p"]) == (50, "stagger", 3))
    hub_deg = big["n"] - 1
    assert big["bound"] == 2 * hub_deg * big["p"]
    # slide 6 says that every 50-qubit circuit lands below the assumed floor
    fifty = [c for c in circuits if c["n"] == 50 and c["family"] == "ihva"]
    assert len(fifty) == 6 and all(c["best"][1] < c["bound"] for c in fifty), "a 50q circuit is not below its floor"
    assert sum(c["n"] == 12 for c in circuits) == 8  # "all eight circuits" on slide 5

    values = {
        "score": f"{s['score']:.2f}",
        "dist_impr": f"{s['distance_improvement']:.1f}",
        "depth_impr": f"{s['depth_improvement']:.1f}",
        "naive_dist": num(s["naive_distance"]),
        "best_dist": num(s["distance"]),
        "naive_depth": num(s["naive_depth"]),
        "best_depth": num(s["depth"]),
        "n_circ": str(s["n"]),
        "n_valid": f"{s['n'] - len(best['failed'])}{NNBSP}/{NNBSP}{s['n']}",
        "big_naive": num(big["naive"][1]),
        "big_best": str(big["best"][1]),
        "big_bound": str(big["bound"]),
        "hub_deg": str(hub_deg),
        "hub_pulses": str(2 * hub_deg),
    }

    html = (HERE / "template.html").read_text()
    used = set(re.findall(r"\{\{(\w+)\}\}", html))
    missing = used - set(values)
    if missing:
        raise SystemExit(f"template uses unknown placeholders: {sorted(missing)}")
    for k in used:
        html = html.replace("{{" + k + "}}", values[k])
    data = {"compiler": BEST, "circuits": circuits}
    html = html.replace("__DECK_DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (HERE / "index.html").write_text(html)
    print(f"wrote {HERE / 'index.html'} ({len(html) / 1024:.0f} kB) from results/{BEST}.json")
    for k in sorted(values):
        print(f"  {k:12} {values[k]}")


if __name__ == "__main__":
    main()
