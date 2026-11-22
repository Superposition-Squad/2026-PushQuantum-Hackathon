# Talk plan (5 minutes): decision 2026-11-22

Two decks exist: `slides/relay/relay.pptx` (teammate; story around the relay idea, polished, animated
GIFs, honesty slide, 6 draft identity slides) and `slides/deck/relay_deck.pptx` (ours; broader:
three layers, formal L3 / exact solver / peephole, how we worked, results history, MP4).

## Comparison

| | better in relay.pptx (theirs) | better in relay_deck.pptx (ours) |
|---|---|---|
| narrative, hook | one clear idea, strong title | |
| the relay itself | step-by-step insight, animated mechanism | |
| validation / honesty | dedicated slide | only in notes |
| how the compiler fits together | | three-layer slide (L1/L2/L3) |
| how we worked, tools | | process + visualiser diff slide |
| movement layer (formal model, exact solver, peephole) | one sentence | two slides |
| history systolic → relay_best_l3 | | results chart |
| a full circuit executing | | 1:27 MP4 |

Ours is too long for 5 minutes (11 slides). Theirs is the better talk.

## Decision: their deck as backbone, one slide from ours, the rest is backup

| # | slide | source | time |
|---|---|---|---|
| 1 | Title + headline numbers (+92.45 %, −93.2 % travel, −89.4 % depth, 20/20) | theirs, slide 1 | 0:30 |
| 2 | The bottleneck: one hub, 98 pulses per layer | theirs, slide 2 | 0:45 |
| 3 | The insight: the hub only needs the parity (reveal steps = one slide) | theirs, slides 3-5 | 1:00 |
| 4 | The mechanism: all handovers at once, hub meets only roots | theirs, slide 6 (optionally play 10 s of our MP4 from 0:19) | 0:45 |
| 5 | How we worked: one harness, exact solver → fast peephole pass, visualiser diff, merging lines of work | ours, slide 9 | 0:40 |
| 6 | Results + takeaway | theirs, slide 8 (one takeaway sentence from slide 9 in the notes) | 0:50 |

Total ≈ 4:30, leaving 30 s buffer.

Backup (after a "Backup" divider, for questions): theirs "Keeping it honest" (slide 7), theirs takeaway
(slide 9), theirs 6 draft identity slides, ours "Three layers" (slide 4), ours "From exact solutions
to a fast pass" (slide 8), ours results chart (slide 10), the MP4 `slides/anim/relay_anim.mp4`.

## How to assemble (Google Slides, no file edits)

1. File → Import slides → upload `slides/relay/relay.pptx` → select all → Import.
2. File → Import slides → upload `slides/deck/relay_deck.pptx` → select only slide 9 (and the backup
   slides 4, 8, 10) → Import; drag slide 9 between their mechanism and results slides.
3. Move their slide 7, 9 and the draft slides behind a "Backup" title slide.
4. Optional: Insert → Video → Drive → `relay_anim.mp4` on the mechanism slide (start at 0:19).
