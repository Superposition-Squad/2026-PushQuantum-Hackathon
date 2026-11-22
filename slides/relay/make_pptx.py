"""Turn the HTML deck (index.html) into relay.pptx, laid out for import into Google Slides.

    LD_LIBRARY_PATH=~/.cache/viz-chromium-libs/lib \
      uv run --no-sync --with python-pptx python slides/relay/make_pptx.py

The HTML deck is the single source: this script opens it in headless Chromium, reads the position
and style of every text block and panel of each slide, and rebuilds them as native (editable) text
boxes and shapes at the same place. Diagrams are exported as PNG, the animated ones (data-gif in the
template) are recorded into looping GIFs. Speaker notes are copied. Slides with reveal steps become
one slide per step.

Run build.py first. To use the result in Google Slides: upload relay.pptx to Drive and open it with Slides.
"""
import io
import re
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
W, H = 1280, 720          # the deck's stage in CSS px; 1 px = 9525 EMU (96 dpi), so the slide is 13.33 x 7.5 in
PX = 9525
BG = (0x0A, 0x0F, 0x1E)
FONT, MONO = "Inter", "Roboto Mono"   # both available in Google Slides
GRADIENT_TEXT = (0xFF, 0xB5, 0x4A)    # stand-in for the gradient-filled title (no gradient text in Slides)
SCALE = 2                 # device scale factor of exported PNGs
GIF_SCALE = 1.5

# Runs in the page: describe the visible content of the active slide as boxes, text blocks and images.
EXTRACT = r"""
() => {
  const stage = document.getElementById('stage').getBoundingClientRect();
  const slide = document.querySelector('.slide.active');
  const items = [];
  let cap = 0;
  const rel = (r) => [r.left - stage.left, r.top - stage.top, r.width, r.height];
  const alpha = (c) => { const m = c.match(/rgba\(([^)]+)\)/); return m ? parseFloat(m[1].split(',')[3]) : 1; };
  const tag = (e) => { e.setAttribute('data-cap', ++cap); return cap; };
  const inline = (e) => getComputedStyle(e).display.startsWith('inline');
  const style = (e) => {
    const cs = getComputedStyle(e);
    return { size: parseFloat(cs.fontSize), bold: parseInt(cs.fontWeight, 10) >= 600, color: cs.color,
             mono: /mono/i.test(cs.fontFamily), spc: parseFloat(cs.letterSpacing) || 0, upper: cs.textTransform === 'uppercase' };
  };
  function runs(node, st, out) {            // node: a text node or an inline element
    if (node.nodeType === 3) {
      let t = node.textContent.replace(/[ \t\n\r]+/g, ' ');
      if (st.upper) t = t.toUpperCase();
      const link = node.parentElement.closest('a[href]');
      if (t) out.push({ t, ...st, href: link?.href });
    } else if (node.nodeType === 1) {
      if (node.tagName === 'BR') out.push({ br: true });
      else if (node.tagName !== 'svg') { const s2 = style(node); for (const ch of node.childNodes) runs(ch, s2, out); }
    }
  }
  function text(nodes, parent, rect) {
    const out = [], st = style(parent);
    nodes.forEach((n) => runs(n, st, out));
    if (!out.some((r) => r.t && r.t.trim())) return;
    const cs = getComputedStyle(parent);
    const lh = parseFloat(cs.lineHeight) || 1.2 * parseFloat(cs.fontSize);
    items.push({ type: 'text', rect, runs: out, lh, align: cs.textAlign });
  }
  function walk(e) {
    if (e.tagName === 'ASIDE') return;
    const cs = getComputedStyle(e);
    if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) === 0) return;
    const r = rel(e.getBoundingClientRect());
    if (e.dataset.gif) { items.push({ type: 'gif', name: e.dataset.gif, ms: parseInt(e.dataset.gifMs || '0', 10), rect: r, cap: tag(e) }); return; }
    if (e.tagName === 'svg' || 'img' in e.dataset) { items.push({ type: 'img', rect: r, cap: tag(e) }); return; }
    const bt = parseFloat(cs.borderTopWidth), bl = parseFloat(cs.borderLeftWidth), bb = parseFloat(cs.borderBottomWidth);
    if (alpha(cs.backgroundColor) > 0 && !cs.backgroundClip.includes('text'))
      items.push({ type: 'box', rect: r, fill: cs.backgroundColor, line: bt > 0 ? cs.borderTopColor : null, lw: bt, radius: parseFloat(cs.borderTopLeftRadius) || 0 });
    else if (bl >= 3 && bt === 0) items.push({ type: 'box', rect: [r[0], r[1], bl, r[3]], fill: cs.borderLeftColor, radius: 0 });
    else if (bb > 0 && bt === 0) items.push({ type: 'box', rect: [r[0], r[1] + r[3] - bb, r[2], bb], fill: cs.borderBottomColor, radius: 0 });
    const be = getComputedStyle(e, '::before');                       // list markers drawn with ::before
    if (be.position === 'absolute' && /^"/.test(be.content)) {
      const size = parseFloat(be.fontSize);
      items.push({ type: 'text', rect: [r[0] + parseFloat(be.left), r[1] + parseFloat(be.top), size * 1.6, size * 1.3], lh: size * 1.2, align: 'start',
                   runs: [{ t: JSON.parse(be.content), size, bold: parseInt(be.fontWeight, 10) >= 600, color: be.color, mono: false, spc: 0 }] });
    }
    const kids = [...e.childNodes].filter((n) => n.nodeType === 1 ? n.tagName !== 'ASIDE' : n.nodeType === 3);
    const isBlock = (n) => n.nodeType === 1 && n.tagName !== 'BR' && (!inline(n) || n.tagName === 'svg' || 'img' in n.dataset || 'gif' in n.dataset);
    if (!kids.some(isBlock)) {                                        // a leaf block: one text box in its content box
      const pl = parseFloat(cs.paddingLeft) + bl, pt = parseFloat(cs.paddingTop) + bt;
      const pr = parseFloat(cs.paddingRight) + parseFloat(cs.borderRightWidth), pb = parseFloat(cs.paddingBottom) + bb;
      text(kids, e, [r[0] + pl, r[1] + pt, r[2] - pl - pr, r[3] - pt - pb]);
      return;
    }
    let group = [];                                                   // mixed content: runs of inline nodes between blocks
    const flush = () => {
      if (group.some((n) => n.textContent.trim())) {
        const range = document.createRange();
        range.setStartBefore(group[0]); range.setEndAfter(group[group.length - 1]);
        text(group, e, rel(range.getBoundingClientRect()));
      }
      group = [];
    };
    for (const n of kids) { if (isBlock(n)) { flush(); walk(n); } else group.push(n); }
    flush();
  }
  [...slide.children].forEach(walk);
  return { items, notes: slide.querySelector('aside.notes').textContent.replace(/\s+/g, ' ').trim(),
           steps: new Set([...slide.querySelectorAll('.frag')].map((f, i) => f.dataset.g || 'f' + i)).size };
}
"""


def rgb(css, over=BG):
    """CSS rgb()/rgba() -> (r, g, b), blended over the slide background; None if fully transparent."""
    v = [float(x) for x in re.findall(r"[\d.]+", css)]
    a = v[3] if len(v) > 3 else 1.0
    if a == 0:
        return None
    return tuple(round(a * v[i] + (1 - a) * over[i]) for i in range(3))


def emu(px):
    return Emu(int(round(px * PX)))


# ------------------------------------------------------------------ capture
def shot(page, cap):
    return page.locator(f'[data-cap="{cap}"]').screenshot(type="png")


def record_gif(page, item, path):
    """Record one loop of an animated region into a GIF (real time, global palette)."""
    x, y, w, h = item["rect"]
    clip = {"x": x, "y": y, "width": w, "height": h}
    name, frames, stamps = item["name"], [], []
    if item["ms"]:                                   # a CSS animation with a fixed period
        done = lambda t: t >= item["ms"] / 1000
    else:                                            # a scripted scene: restart it and record until the loop ends
        page.evaluate(f"() => {{ const s = window.__scenes['{name}']; s.stop(); s.loops = 0; s.start(); }}")
        done = lambda t: page.evaluate(f"() => window.__scenes['{name}'].loops") >= 1 or t > 60
    t0 = time.monotonic()
    while True:
        a = time.monotonic()
        png = page.screenshot(type="png", clip=clip)
        b = time.monotonic()
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
        stamps.append((a + b) / 2 - t0)
        if done(b - t0):
            break
    total = item["ms"] / 1000 if item["ms"] else stamps[-1] + (stamps[-1] - stamps[-2])
    durs = [max(20, round(1000 * (n - s))) for s, n in zip(stamps, stamps[1:] + [total])]
    # one palette for all frames (no flicker): sample frames across the loop, including the pulses
    step = max(1, len(frames) // 24)
    sample = frames[::step]
    sheet = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, f in enumerate(sample):
        sheet.paste(f, (0, i * f.height))
    pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=False, disposal=1)
    return len(frames), sum(durs) / 1000


def capture(tmp):
    """-> list of slides: {items, notes}; images written to tmp, items of type img/gif get a 'path'."""
    url = "file://" + str(HERE / "index.html")
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=SCALE)
        gctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=GIF_SCALE)
        page, gpage = ctx.new_page(), gctx.new_page()
        errors = []
        for pg in (page, gpage):
            pg.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url + "#1&cap=1")
        n_slides = page.locator(".slide").count()
        for k in range(1, n_slides + 1):
            page.goto("about:blank")
            page.goto(f"{url}#{k}&cap=1")
            page.wait_for_timeout(1500)               # fonts, chart bars
            steps = page.evaluate(EXTRACT)["steps"]
            for step in range(steps + 1):
                if step:
                    page.keyboard.press("ArrowRight")
                    page.wait_for_timeout(700)
                data = page.evaluate(EXTRACT)
                for it in data["items"]:
                    if it["type"] == "img":
                        it["path"] = tmp / f"s{k}_{step}_{it['cap']}.png"
                        it["path"].write_bytes(shot(page, it["cap"]))
                    elif it["type"] == "gif":
                        it["path"] = tmp / f"s{k}_{it['name']}.gif"
                        if not it["path"].exists():
                            gpage.goto("about:blank")
                            gpage.goto(f"{url}#{k}&cap=1")
                            gpage.wait_for_timeout(800)
                            n, secs = record_gif(gpage, it, it["path"])
                            print(f"  slide {k}: {it['name']}.gif  {n} frames, {secs:.1f} s, {it['path'].stat().st_size / 1e6:.1f} MB")
                out.append({"items": data["items"], "notes": data["notes"], "slide": k, "step": step})
            print(f"slide {k}: {steps + 1} step(s), {len(data['items'])} items")
        browser.close()
    if errors:
        raise SystemExit(f"page errors: {errors}")
    return out


# ------------------------------------------------------------------ pptx
def add_box(slide, it):
    x, y, w, h = it["rect"]
    fill = rgb(it["fill"])
    if fill is None or w <= 0 or h <= 0:
        return
    rounded = it.get("radius", 0) > 0
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, emu(x), emu(y), emu(w), emu(h))
    if rounded:
        sh.adjustments[0] = min(0.5, it["radius"] / min(w, h))
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor(*fill)
    line = rgb(it["line"]) if it.get("line") else None
    if line:
        sh.line.color.rgb = RGBColor(*line)
        sh.line.width = Pt(0.75 * it["lw"])
    else:
        sh.line.fill.background()


def add_text(slide, it):
    x, y, w, h = it["rect"]
    runs = it["runs"]
    # trim like HTML does: no leading/trailing space in a block, none around a line break
    for i, r in enumerate(runs):
        if "t" not in r:
            continue
        if i == 0 or "br" in runs[i - 1] or (runs[i - 1].get("t", "x").endswith(" ")):
            r["t"] = r["t"].lstrip(" ")
        if i == len(runs) - 1 or "br" in runs[i + 1]:
            r["t"] = r["t"].rstrip(" ")
    runs = [r for r in runs if "br" in r or r["t"]]
    if not runs:
        return
    lines = max(1, round(h / it["lh"]))
    align = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT, "end": PP_ALIGN.RIGHT}.get(it["align"], PP_ALIGN.LEFT)
    if lines == 1:
        # the fonts differ between browsers and Slides: give single lines room so that they never wrap
        extra = 0.3 * w + 24
        if align == PP_ALIGN.LEFT:
            w = min(w + extra, W - x)
        elif align == PP_ALIGN.CENTER:
            x, w = x - extra / 2, w + extra
        else:
            x, w = x - extra, w + extra
    else:
        w += 6
    tb = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = MSO_ANCHOR.TOP
    para = tf.paragraphs[0]
    para.alignment = align
    para.line_spacing = Pt(0.75 * it["lh"])
    for r in runs:
        if "br" in r:
            para.add_line_break()
            continue
        run = para.add_run()
        run.text = r["t"]
        if r.get("href"):
            run.hyperlink.address = r["href"]
            run.font.underline = True
        f = run.font
        f.size = Pt(round(0.75 * r["size"] * 2) / 2)
        f.bold = bool(r["bold"])
        f.name = MONO if r["mono"] else FONT
        f.color.rgb = RGBColor(*(rgb(r["color"]) or GRADIENT_TEXT))
        if r.get("spc"):
            run._r.get_or_add_rPr().set("spc", str(int(round(75 * r["spc"]))))   # hundredths of a point


def build(slides, out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = emu(W), emu(H)
    blank = prs.slide_layouts[6]
    for sd in slides:
        s = prs.slides.add_slide(blank)
        fill = s.background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(*BG)
        for it in sd["items"]:
            if it["type"] == "box":
                add_box(s, it)
            elif it["type"] == "text":
                add_text(s, it)
            else:
                x, y, w, h = it["rect"]
                s.shapes.add_picture(str(it["path"]), emu(x), emu(y), emu(w), emu(h))
        s.notes_slide.notes_text_frame.text = sd["notes"]
    prs.save(out)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "relay.pptx"
    with tempfile.TemporaryDirectory() as tmp:
        slides = capture(Path(tmp))
        build(slides, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(slides)} slides)")


if __name__ == "__main__":
    main()
