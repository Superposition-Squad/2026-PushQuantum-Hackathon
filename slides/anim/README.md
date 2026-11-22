# Animation: how the compiler executes a circuit

`relay_anim.html`: one self-contained page (no external files), relay_best_l3 on the 20-qubit iHVA
stagger p = 1 circuit, built from the real compiled plan (106 moves, 30 CZ pulses). Six chapters
(L1/L2/L3 tagged): band rounds, relay parity collection, hub meets only the roots, uncompute/rotate/
recompute, second pass, last round. Keys: Space play/pause, 1-6 jump to a chapter, ← / → ±2 s.

| step | command |
|---|---|
| event data from a cached plan (`l3/cache.py relay_best_l3`) | `python slides/anim/extract.py l3/cache/relay_best_l3/bus_stop_20q_ihva_stagger_p1.pkl > slides/anim/data.json` |
| inline into the page | `python slides/anim/build.py` |
| MP4 (1920×1080, 30 fps, ~1:27) + GIF (960 px, 12 fps) | `python slides/anim/export.py <frames dir>` (Playwright + system Chromium, ffmpeg) |

Google Slides cannot embed HTML: insert the MP4 via Insert → Video → Google Drive, or the GIF via
Insert → Image (plays and loops automatically). MP4/GIF are not committed (generated media).
