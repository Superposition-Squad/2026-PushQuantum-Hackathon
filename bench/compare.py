"""Rank compilers by their latest full metric run in results/<name>.json.

Usage: uv run python bench/compare.py [names...] [--per-circuit]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(names: list[str]) -> list[dict]:
    files = [ROOT / "results" / f"{n}.json" for n in names] if names else sorted((ROOT / "results").glob("*.json"))
    return [json.loads(f.read_text()) for f in files if f.is_file()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--per-circuit", action="store_true", help="distance/depth per circuit, side by side")
    args = ap.parse_args(argv)
    runs = load(args.names)
    if not runs:
        print("no results")
        return 1

    def key(r):
        s = r["summary"]
        return s.get("score", float("-inf")) if s.get("valid") else float("-inf")

    runs.sort(key=key, reverse=True)
    print(f"{'rank':<5}{'name':<28}{'score':>9}{'dist%':>9}{'depth%':>9}  {'hea d/p':>15}{'ihva d/p':>15}{'qaoa d/p':>15}  mode     when")
    for i, r in enumerate(runs, 1):
        s = r["summary"]
        if s.get("valid"):
            fam = "".join(
                f"{s['families'].get(f, {}).get('distance_improvement', 0):>+8.1f}/{s['families'].get(f, {}).get('depth_improvement', 0):<+6.1f}"
                for f in ("hea", "ihva", "qaoa")
            )
            print(f"{i:<5}{r['name']:<28}{s['score']:>+8.2f}%{s['distance_improvement']:>+8.2f}%{s['depth_improvement']:>+8.2f}%  {fam}  {r['mode']:<8} {r['timestamp']}")
        else:
            print(f"{i:<5}{r['name']:<28}{'INVALID':>9}  failed: {', '.join(r['failed'][:6])}  {r['mode']:<8} {r['timestamp']}")

    if args.per_circuit:
        labels = sorted({rec["label"] for r in runs for rec in r["records"] if rec["label"].startswith("metric_")})
        print()
        print(f"{'circuit':<20}{'file':<44}" + "".join(f"{r['name'][:22]:>24}" for r in runs))
        for lab in labels:
            recs = [next((x for x in r["records"] if x["label"] == lab), None) for r in runs]
            f = next((x["file"] for x in recs if x), "")
            cells = "".join(
                f"{(f'{x['metrics']['move_distance']:.0f}/{x['metrics']['depth']}' if x and x['ok'] else 'FAIL'):>24}" for x in recs
            )
            print(f"{lab:<20}{f:<44}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
