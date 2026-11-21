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
- --nav URL: comparison sidebar + split view on a served viz/ (`bench/viz.py serve --port 9123`):
  live manifest, compiler switch keeps the progress, [ ] and , . shortcuts, split view in sync.
  Needs at least two compilers with pages for the same circuits; `--nav-page name/label.html`.
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
                box = page.locator(".view .circWrap canvas").first.bounding_box()
                page.mouse.click(box["x"] + xy["x"], box["y"] + xy["y"])
                if state()["k"] != target:
                    problems.append(f"click on instruction {target} gave cursor {state()['k']}")
                page.locator("#scrub").fill(str(state()["S"]))
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


def nav_check(base: str, start: str) -> list[str]:
    from playwright.sync_api import sync_playwright

    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
        ready = lambda: page.wait_for_function("window.__viz !== undefined", timeout=60000)  # noqa: E731
        state = lambda: page.evaluate("window.__viz.state()")  # noqa: E731
        progress = lambda s: (s["s"] + 1) / s["S"] if s["S"] else 0  # noqa: E731  (display steps: a move = 2)
        page.goto(f"{base.rstrip('/')}/{start}")
        ready()
        page.wait_for_function("window.__viz.state().manifestLive", timeout=15000)
        comps = page.evaluate("window.__viz.compilations()")
        s0 = state()
        others = [c["name"] for c in comps if c["name"] != s0["name"]]
        if not others:
            return problems + ["no other compilation of this circuit: generate pages for a second compiler"]
        page.evaluate(f"window.__viz.set({int(s0['N'] * 0.42) - 1})")
        p0 = progress(state())
        page.locator(f'.crow[data-name="{others[0]}"]').click()
        ready()
        s1 = state()
        if s1["name"] != others[0]:
            problems.append(f"sidebar click went to {s1['name']}, not {others[0]}")
        if abs(progress(s1) - p0) > 2 / max(1, min(s0["S"], s1["S"])) + 1e-9:
            problems.append(f"progress not kept: {p0:.4f} -> {progress(s1):.4f}")
        page.keyboard.press("]")
        ready()
        if state()["name"] == s1["name"]:
            problems.append("] did not switch the compiler")
        page.keyboard.press("[")
        ready()
        label = state()["label"]
        page.keyboard.press(".")
        ready()
        if state()["label"] == label:
            problems.append(". did not switch the circuit")
        page.keyboard.press(",")
        ready()
        if state()["label"] != label:
            problems.append(", did not come back to the previous circuit")
        page.keyboard.press("c")
        page.wait_for_function("window.__viz.state().kB !== null || window.__viz.state().splitMsg", timeout=60000)
        s2 = state()
        if s2["kB"] is None:
            problems.append(f"split view did not load: {s2['splitMsg']}")
        else:
            page.keyboard.press("ArrowRight")
            s3 = state()
            if abs(progress(s3) - (s3["sB"] + 1) / s3["SB"]) > 2 / max(1, min(s3["S"], s3["SB"])) + 1e-9:
                problems.append("split views out of sync")
            page.wait_for_timeout(200)
            if not page.evaluate("window.__viz.sizesOk()"):
                problems.append("split view: a canvas is not sized to its box")
        page.keyboard.press("c")
        if state()["split"]:
            problems.append("C did not close the split view")
        browser.close()
    return problems


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
    ap.add_argument("--nav", metavar="URL", help="base URL of a served viz/ directory")
    ap.add_argument("--nav-page", default="gates/metric_circuit_07.html")
    args = ap.parse_args()
    if args.nav:
        problems = nav_check(args.nav, args.nav_page)
        print(f"nav {args.nav}/{args.nav_page}: {'OK' if not problems else problems}")
        return 1 if problems else 0
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
