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

## Decision (updated): fewer hub slides, one slide on the movement layer

Their deck spends five slides on the hub. For 5 minutes: two hub slides (insight, mechanism), then
one slide on how the atoms move (L3, ours, new) and one on how we worked (ours).

| # | slide | source | time |
|---|---|---|---|
| 1 | Title + headline numbers (+92.45 %, −93.2 % travel, −89.4 % depth, 20/20) | theirs, slide 1 | 0:25 |
| 2 | The hub only needs the parity (final reveal state; say the bottleneck in one sentence: 98 hub pulses per layer at 50 qubits) | theirs, slide 5 | 1:00 |
| 3 | The mechanism: all handovers at once, hub meets only roots | theirs, slide 6 | 0:40 |
| 4 | **L3: moving atoms cheaply**: staging row + row rotation (diagram), rigid band rounds, hub slot hopping, gather lifts, dip trimming, exact solver → fast pass | ours, slide 7 | 0:55 |
| 5 | How we worked: one harness, exact solver → peephole, visualiser diff, merging lines of work | ours, slide 10 | 0:40 |
| 6 | Results (one takeaway sentence in the notes) | theirs, slide 8 | 0:40 |

Total ≈ 4:20, leaving 40 s buffer.

Backup (behind a "Backup" title slide): theirs slides 2 (bottleneck), 3-4 (reveal steps), 7 (keeping it
honest), 9 (takeaway), the 6 draft identity slides; ours slides 4 (three layers), 6 (rounds and band),
8 (formal L3), 9 (exact solutions → fast pass), 11 (results chart); the MP4.

## How to assemble (Google Slides, no file edits)

1. File → Import slides → upload `slides/relay/relay.pptx` → select all → Import.
2. File → Import slides → upload `slides/deck/relay_deck.pptx` → select slides 7 and 10 (and the backup
   slides 4, 6, 8, 9, 11) → Import; put 7 and 10 between their mechanism and results slides.
3. Move their slides 2, 3, 4, 7, 9 and the draft slides behind a "Backup" title slide.
4. Optional: Insert → Video → Drive → `relay_anim.mp4` on the mechanism slide (start at 0:19).
   The slide also links to the online animation,
   `https://superposition-squad.github.io/2026-PushQuantum-Hackathon/anim/relay_anim.html` (GitHub Pages).
5. The workflow slide links (URL and QR code beside the screenshot) to the GitHub Pages landing page,
   `https://superposition-squad.github.io/2026-PushQuantum-Hackathon/viz/` (visualiser of all compilers).
   Check that both pages load before presenting.
