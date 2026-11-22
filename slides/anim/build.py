#!/usr/bin/env python3
"""Inline data.json into template.html -> relay_anim.html (single self-contained file)."""
from pathlib import Path
here = Path(__file__).resolve().parent
html = (here / "template.html").read_text().replace("__DATA__", (here / "data.json").read_text().replace("</", "<\\/"))
(here / "relay_anim.html").write_text(html)
print("wrote", here / "relay_anim.html", len(html), "bytes")
