# relay_deck.pptx

11-slide deck built from `slides/deck_brief.md` (speaker notes included). Import into Google Slides
with File → Import slides (or upload the .pptx to Drive and open it). Insert the animation
(`slides/anim/relay_anim.mp4`) by hand on slide 5 or 6 if wanted.

Rebuild: `frames.py` renders the animation frames into `img/` (Playwright + Chromium + ffmpeg), then
`NODE_PATH=<dir with node_modules/pptxgenjs> node build_deck.js <pptx skill>/scripts/apply_theme.js`.
