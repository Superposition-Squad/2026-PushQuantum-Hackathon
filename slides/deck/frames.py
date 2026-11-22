import json
from pathlib import Path
from playwright.sync_api import sync_playwright
import subprocess
here = Path(__file__).resolve().parent
anim = here.parent / "anim"
d = json.load(open(anim / "data.json"))
url = (anim / "relay_anim.html").as_uri() + "?export=1"
def cz_index(k):
    return next(i for i, e in enumerate(d["events"]) if e["t"] == "cz" and e["k"] == k)
shots = {"band_round": cz_index(0), "relay_parity": cz_index(6), "hub_root": cz_index(9)}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(url); pg.wait_for_timeout(400)
    for name, i in shots.items():
        t = pg.evaluate(f"window.animEvStart({i})") + 0.27
        pg.evaluate(f"window.renderAt({t})"); pg.wait_for_timeout(60)
        full = here / "img" / f"{name}_full.png"
        pg.screenshot(path=str(full), clip={"x": 0, "y": 0, "width": 1920, "height": 1080})
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-vf", "crop=1110:830:70:160", str(here / "img" / f"{name}.png")], check=True)
        full.unlink()
    b.close()
print("ok", sorted(x.name for x in (here / "img").iterdir()))
