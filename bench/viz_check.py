"""Smoke checks for bench/viz.py pages (dev-only: needs the `playwright` and `esprima` dev packages).

Usage (from the repo root):
    python bench/viz_check.py viz/naive/metric_circuit_01.html [more pages ...] [--static-only]
    python bench/viz_check.py --compare viz/<name> [results/runs/<run>.json | naive]

- static: the inline scripts parse (esprima), every element id the script uses exists, the embedded
  JSON parses, and the page totals are self-consistent (Σ move distances, last depth, move count).
- browser (headless Chromium via playwright): the page loads without console errors, stepping,
  frame mode, play, a click on the circuit, the scrubber and the error jump move the cursor.
  Chromium needs a few shared libraries; if they are not installed system-wide, point
  LD_LIBRARY_PATH at a directory that has them (see docs/viz.md).
- --compare: every sidecar's metrics equal the bench run's records (or the naive baseline) exactly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RE = re.compile(r'<script id="viz-data" type="application/json">(.*?)</script>', re.S)
KEYS = ["num_moves", "move_steps", "move_distance", "move_time", "depth", "num_operations", "spacing_warnings"]


def static_check(path: Path) -> list[str]:
    import esprima

    problems = []
    text = path.read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    for script in re.findall(r"<script>(.*?)</script>", text, re.S):
        try:
            esprima.parseScript(script)
        except Exception as error:  # noqa: BLE001
            problems.append(f"script does not parse: {error}")
        used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', script)) | set(re.findall(r'getElementById\("([A-Za-z0-9_-]+)"\)', script))
        if used - ids:
            problems.append(f"missing element ids: {sorted(used - ids)}")
    match = DATA_RE.search(text)
    if not match:
        return problems + ["no embedded viz-data"]
    data = json.loads(match.group(1))
    tot = data["totals"]
    if len(data["moves"]) != tot["num_moves"]:
        problems.append(f"{len(data['moves'])} moves vs totals {tot['num_moves']}")
    if abs(round(sum(m["d"] for m in data["moves"]), 6) - tot["move_distance"]) > 1e-6:
        problems.append("sum of move distances != totals.move_distance")
    depth = data["inst"]["depth"]
    if depth and not tot.get("partial") and depth[-1] != tot["depth"]:
        problems.append(f"last depth {depth[-1]} != totals.depth {tot['depth']}")
    return problems


def browser_check(paths: list[Path]) -> dict[str, list[str]]:
    from playwright.sync_api import sync_playwright

    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for path in paths:
            problems: list[str] = []
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.on("pageerror", lambda e, pr=problems: pr.append(f"pageerror: {e}"))
            page.on("console", lambda m, pr=problems: m.type == "error" and pr.append(f"console: {m.text}"))
            page.goto(path.resolve().as_uri())
            page.wait_for_function("window.__viz !== undefined", timeout=60000)
            state = lambda: page.evaluate("window.__viz.state()")  # noqa: E731
            n = state()["N"]
            page.wait_for_timeout(100)
            if not page.evaluate("window.__viz.sizesOk()"):
                problems.append("a canvas is not sized to its box (stretched drawing)")
            if n:
                page.keyboard.press("ArrowRight")
                if state()["k"] != 0:
                    problems.append("ArrowRight did not step to instruction 0")
                page.keyboard.press("End")
                if state()["k"] != n - 1:
                    problems.append("End did not reach the last instruction")
                page.keyboard.press("Home")
                page.keyboard.press("g")
                page.keyboard.press("ArrowRight")
                page.keyboard.press("g")
                if state()["k"] < 0:
                    problems.append("frame step did not advance")
                target = n // 2
                page.evaluate(f"window.__viz.screenOfInst({target})")
                page.wait_for_timeout(50)
                xy = page.evaluate(f"window.__viz.screenOfInst({target})")
                box = page.locator("#circCv").bounding_box()
                page.mouse.click(box["x"] + xy["x"], box["y"] + xy["y"])
                if state()["k"] != target:
                    problems.append(f"click on instruction {target} gave cursor {state()['k']}")
                page.locator("#scrub").fill(str(n))
                if state()["k"] != n - 1:
                    problems.append("scrubber to the end failed")
                page.keyboard.press("Home")
                page.evaluate("window.__setSpeed(30)")
                page.keyboard.press(" ")
                page.wait_for_timeout(800)
                after = state()
                if after["playing"]:
                    page.keyboard.press(" ")  # pause (a short circuit may already have stopped at its end)
                if after["k"] < min(1, n - 1):
                    problems.append("play did not advance")
            if page.locator("#errJump").is_visible():
                page.locator("#errJump").click()
                if state()["k"] < 0:
                    problems.append("error jump did not move the cursor")
            out[str(path)] = problems
            page.close()
        browser.close()
    return out


def compare(directory: Path, reference: str) -> int:
    if reference == "naive":
        base = json.loads((ROOT / "vendor" / "pushquantum_validator" / "naive_baseline.json").read_text())
        ref = {k: {**v["metrics"], "spacing_warnings": v["spacing_warnings"]} for k, v in base.items()}
    else:
        ref = {r["label"]: r.get("metrics") for r in json.loads(Path(reference).read_text())["records"]}
    compared = bad = 0
    for sidecar in sorted(directory.glob("*.json")):
        s = json.loads(sidecar.read_text())
        if s["label"] not in ref:
            continue
        compared += 1
        r, m = ref[s["label"]], s.get("metrics")
        diff = [k for k in KEYS if r is None or m is None or m[k] != r[k]]
        if diff:
            bad += 1
            print(f"MISMATCH {s['label']}: {diff}  viz={m}  ref={r}")
    print(f"{directory.name}: {compared} circuits compared, {bad} mismatches")
    return 1 if bad or not compared else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="*", type=Path)
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("VIZ_DIR", "RUN_JSON|naive"))
    args = ap.parse_args()
    if args.compare:
        return compare(Path(args.compare[0]), args.compare[1])
    failed = 0
    for page in args.pages:
        problems = static_check(page)
        print(f"static  {page}: {'OK' if not problems else problems}")
        failed += bool(problems)
    if not args.static_only and args.pages:
        for page, problems in browser_check(args.pages).items():
            print(f"browser {page}: {'OK' if not problems else problems}")
            failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
