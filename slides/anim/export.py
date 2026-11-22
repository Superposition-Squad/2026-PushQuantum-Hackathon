#!/usr/bin/env python3
"""Render relay_anim.html frame by frame (deterministic renderAt) and encode MP4 + GIF.
    python slides/anim/export.py <frames dir> [fps=30]"""
import subprocess, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

here = Path(__file__).resolve().parent
frames = Path(sys.argv[1]); frames.mkdir(parents=True, exist_ok=True)
fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
url = (here / "relay_anim.html").as_uri() + "?export=1"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(url); pg.wait_for_timeout(500)
    total = pg.evaluate("window.animTotal")
    n = int(total * fps) + 1
    for f in range(n):
        pg.evaluate(f"window.renderAt({f / fps})")
        pg.screenshot(path=str(frames / f"f{f:05d}.png"), clip={"x": 0, "y": 0, "width": 1920, "height": 1080})
        if f % 300 == 0:
            print("frame", f, "/", n, flush=True)
    b.close()
mp4 = here / "relay_anim.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
gif = here / "relay_anim.gif"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                str(gif)], check=True)
print("wrote", mp4, mp4.stat().st_size // 1024, "KB;", gif, gif.stat().st_size // 1024, "KB")
