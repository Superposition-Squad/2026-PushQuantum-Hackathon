"""Build slides/talk_5min.pdf (8 pages, 1280x720) and docs/slides/talk_5min.pdf.

    python slides/talk/build.py

Pages 1,2,3,4,7 are pages 1,8,3,4,6 of slides/relay/relay.pdf (rendered here from
slides/relay/index.html, run slides/relay/build.py first), embedded as images.
Pages 5, 6, 8 are written in talk.html. QR code: slides/deck/img/visualiser_qr.png.
Needs: chromium, pdftoppm, playwright (repo .venv).
"""
import shutil, subprocess, tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHROME = shutil.which("chromium") or "/usr/bin/chromium"

with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    (t / "hi").mkdir()
    relay = ROOT / "slides/relay/relay.pdf"
    subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={relay}", f"file://{ROOT}/slides/relay/index.html"], check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "144", str(relay), str(t / "hi/p")], check=True)
    shutil.copy(ROOT / "docs/img/viz_diff_split.png", t / "viz.png")
    shutil.copy(ROOT / "slides/deck/img/visualiser_qr.png", t / "qr.png")
    shutil.copy(HERE / "talk.html", t / "talk.html")
    out = ROOT / "slides/talk_5min.pdf"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = b.new_page()
        pg.goto(f"file://{t}/talk.html")
        pg.wait_for_timeout(800)
        pg.pdf(path=str(out), width="1280px", height="720px", print_background=True)
        b.close()
shutil.copy(out, ROOT / "docs/slides/talk_5min.pdf")
shutil.copy(relay, ROOT / "docs/slides/relay_full.pdf")
print("wrote", out)
