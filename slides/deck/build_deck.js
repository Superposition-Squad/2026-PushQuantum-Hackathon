// Build slides/deck/relay_deck.pptx from slides/deck_brief.md content (pptxgenjs).
//   NODE_PATH=<dir with node_modules/pptxgenjs> node slides/deck/build_deck.js <apply_theme.js path>
const path = require("path");
const pptxgen = require("pptxgenjs");
const HERE = __dirname;
const IMG = (f) => path.join(HERE, "img", f);
const SITE_URL = "https://superposition-squad.github.io/2026-PushQuantum-Hackathon/";
const SITE_LABEL = "superposition-squad.github.io/2026-PushQuantum-Hackathon";
const ANIMATION_URL = SITE_URL + "anim/relay_anim.html";
const VISUALISER_URL = SITE_URL;  // landing page: visualiser, animation, slides

const THEME = {
  name: "Relay",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1D2733", lt1: "FFFFFF", dk2: "3A4552", lt2: "F4F6F8",
    accent1: "2F6DB5", accent2: "E0A100", accent3: "E8762D", accent4: "7C4DC4", accent5: "2F8FD8", accent6: "6B7684",
    hlink: "2F6DB5", folHlink: "7C4DC4",
  },
};
const HEX = { ink: "1D2733", ink2: "3A4552", paper: "F4F6F8", blue: "2F6DB5", gold: "E0A100", orange: "E8762D", purple: "7C4DC4", sky: "2F8FD8", muted: "6B7684", zone: "FCEBDD", sto: "E6EEF8", white: "FFFFFF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Compiling iHVA circuits for a zoned neutral-atom machine";
pres.author = "Superposition Squad";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 8.6, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.05, w: 8.6, h: 0.9, fontSize: 18, color: "CADCFC", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text1, valign: "middle" }, text: "" } },
    { text: { text: "relay_best_l3 · planqc compiler track · PushQuantum 2026", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.45, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK_CONTENT",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.background1, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.45, h: 0.3, fontSize: 10, color: "9AA4B0", align: "right" },
});

// ------------------------------------------------------------------ helpers
let objN = 0;
const nm = (s) => `${s}_${++objN}`;
function chip(slide, x, y, label, color, d = 0.55, fs = 15) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 }, objectName: nm("chip") });
  slide.addText(label, { x, y, w: d, h: d, fontSize: fs, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: nm("chipText") });
}
function bullets(slide, items, x, y, w, h, fs = 16) {
  const runs = items.map((t, i) => {
    const parts = Array.isArray(t) ? t : [{ text: t }];
    return parts.map((p, j) => ({ text: p.text, options: { bullet: j === 0 ? { indent: 16 } : undefined, bold: !!p.bold, color: p.color || undefined, fontFace: p.mono ? "Courier New" : undefined, breakLine: j === parts.length - 1 && i < items.length - 1, paraSpaceAfter: 8 } }));
  }).flat();
  slide.addText(runs, { x, y, w, h, fontSize: fs, color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: nm("bullets") });
}
function stat(slide, x, y, w, big, small, color, bigFs = 40) {
  slide.addText(big, { x, y, w, h: 0.75, fontSize: bigFs, bold: true, color, fontFace: THEME.headFontFace, margin: 0, valign: "bottom", isTextBox: true, objectName: nm("statBig") });
  slide.addText(small, { x, y: y + 0.8, w, h: 0.6, fontSize: 13, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: nm("statSmall") });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { color: fill || C.background2, width: 0 }, objectName: nm("card") });
}
function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.45, fontSize: 11, italic: true, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: nm("caption") });
}
function picture(slide, file, x, y, w, aspect, alt) {
  slide.addImage({ path: IMG(file), x, y, w, h: w / aspect, altText: alt, objectName: nm("picture") });
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: w / aspect, rectRadius: 0.06, fill: { type: "none" }, line: { color: "D5DBE2", width: 1 }, objectName: nm("frame") });
}
function webLink(slide, url, x, y, w, label = SITE_LABEL) {
  // two lines: the github.io address does not fit one line in the 4-inch box of the visualiser slide
  slide.addText(label.replace(".io/", ".io/\n"), {
    x, y, w, h: 0.6, fontSize: 13, bold: true, color: HEX.blue, margin: 0,
    hyperlink: { url }, isTextBox: true, objectName: nm("webLink"),
  });
}
const FRAME_ASPECT = 1110 / 830, VIZ_ASPECT = 1600 / 950;

// ------------------------------------------------------------------ 0 title
pres.addSection({ title: "Intro" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Intro" });
s.addText("Compiling iHVA circuits for a zoned neutral-atom machine", { placeholder: "title" });
s.addText("Superposition Squad · PushQuantum Hackathon 2026 · planqc compiler track", { placeholder: "body" });
[[10.2, 1.4, HEX.gold, 0.9], [11.35, 2.15, HEX.blue, 0.6], [10.55, 2.75, HEX.blue, 0.6], [11.75, 3.1, HEX.orange, 0.45], [10.05, 3.55, HEX.blue, 0.45]].forEach(([x, y, col, d]) => s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: col }, line: { color: col, width: 0 }, objectName: nm("atomDeco") }));
s.addText("+92.45 %", { x: 0.7, y: 5.25, w: 5, h: 0.9, fontSize: 54, bold: true, color: HEX.gold, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: "titleScore" });
s.addText("relay_best_l3 · full validation, 20/20 scored circuits + 13/13 sanity", { x: 0.7, y: 6.15, w: 8, h: 0.4, fontSize: 14, color: "CADCFC", margin: 0, isTextBox: true, objectName: "titleScoreLabel" });
s.addNotes("Our best compiler, relay_best_l3, scores +92.45 % on the 20 scored circuits with full validation (official validator, QCEC equivalence up to 19 qubits, interaction-pair check above). The talk walks through the three layers of the compiler and how we iterated on them.");

// ------------------------------------------------------------------ 1 machine
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("The machine and the score", { placeholder: "title" });
bullets(s, [
  [{ text: "Storage zone: ", bold: true }, { text: "rx, rz only. " }, { text: "Entangling zone: ", bold: true }, { text: "cz only, on pairs (0.3 apart; pairs 2.0 apart)" }],
  [{ text: "Transversal pulse: ", bold: true }, { text: "every occupied pair gets a CZ. Idle rule: every zone atom must be in a CZ" }],
  [{ text: "Move = one AOD rectangle: ", bold: true }, { text: "contiguous rows × columns, every atom inside moves, order preserved, one zone each side" }],
  [{ text: "Cost of a move = its longest single-atom path, ", bold: true }, { text: "however many atoms it carries" }],
], 0.6, 1.45, 6.4, 3.6, 16);
card(s, 0.6, 5.2, 6.4, 1.45);
s.addText([{ text: "Score = 0.8 · distance reduction + 0.2 · depth reduction", options: { bold: true, color: C.text1, breakLine: true } }, { text: "vs the naive compiler, over 20 circuits (18 iHVA at 12/20/50 qubits, 2 HEA). 1 depth unit ≈ 3.06 distance units.", options: { color: C.text2 } }], { x: 0.85, y: 5.35, w: 6.0, h: 1.2, fontSize: 14, margin: 0, valign: "middle", isTextBox: true, objectName: "scoreFormula" });
// diagram
const dx = 7.55, dy = 1.45, dw = 5.15;
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: dx, y: dy, w: dw, h: 2.15, rectRadius: 0.12, fill: { color: HEX.zone }, line: { color: HEX.zone, width: 0 }, objectName: "zoneRect" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: dx, y: dy + 2.3, w: dw, h: 2.6, rectRadius: 0.12, fill: { color: HEX.sto }, line: { color: HEX.sto, width: 0 }, objectName: "storageRect" });
s.addText("ENTANGLING ZONE · cz", { x: dx + 0.2, y: dy + 0.12, w: 3.5, h: 0.3, fontSize: 12, bold: true, color: "B8662A", margin: 0, isTextBox: true, objectName: "zoneLabel" });
s.addText("STORAGE · rx, rz", { x: dx + 0.2, y: dy + 4.5, w: 3.5, h: 0.3, fontSize: 12, bold: true, color: "3D6EA8", margin: 0, isTextBox: true, objectName: "stoLabel" });
for (let p = 0; p < 5; p++) for (let r = 0; r < 2; r++) for (let k = 0; k < 2; k++) s.addShape(pres.shapes.OVAL, { x: dx + 0.4 + p * 0.95 + k * 0.22, y: dy + 0.65 + r * 0.7, w: 0.07, h: 0.07, fill: { color: "C9D1DA" }, line: { color: "C9D1DA", width: 0 }, objectName: nm("site") });
for (let c = 0; c < 10; c++) for (let r = 0; r < 4; r++) s.addShape(pres.shapes.OVAL, { x: dx + 0.4 + c * 0.47, y: dy + 2.55 + r * 0.45, w: 0.07, h: 0.07, fill: { color: "C9D1DA" }, line: { color: "C9D1DA", width: 0 }, objectName: nm("site") });
[[0, 0], [1, 0], [2, 0], [3, 0], [4, 1], [5, 1], [6, 0], [7, 2], [8, 1], [9, 0], [1, 2], [3, 1]].forEach(([c, r]) => s.addShape(pres.shapes.OVAL, { x: dx + 0.4 + c * 0.47 - 0.11, y: dy + 2.55 + r * 0.45 - 0.11, w: 0.29, h: 0.29, fill: { color: HEX.blue }, line: { color: HEX.white, width: 1 }, objectName: nm("atom") }));
s.addShape(pres.shapes.OVAL, { x: dx + 0.4 + 0.95 * 2 - 0.11, y: dy + 1.35 - 0.11, w: 0.29, h: 0.29, fill: { color: HEX.gold }, line: { color: HEX.white, width: 1 }, objectName: "hubAtom" });
s.addShape(pres.shapes.OVAL, { x: dx + 0.4 + 0.95 * 2 + 0.22 - 0.11, y: dy + 1.35 - 0.11, w: 0.29, h: 0.29, fill: { color: HEX.blue }, line: { color: HEX.white, width: 1 }, objectName: "partnerAtom" });
s.addShape(pres.shapes.LINE, { x: dx + 0.4 + 0.95 * 2 + 0.25, y: dy + 1.62, w: 0.7, h: 1.2, flipH: false, line: { color: HEX.purple, width: 2, dashType: "dash", endArrowType: "none", beginArrowType: "triangle" }, objectName: "moveArrow" });
s.addText("one move, cost = longest path", { x: dx + 2.85, y: dy + 1.85, w: 2.2, h: 0.35, fontSize: 11, italic: true, color: HEX.purple, margin: 0, isTextBox: true, objectName: "moveLabel" });
s.addNotes("The metric rewards carrying many atoms in one short rigid move; parallel moves are worth nothing. The naive compiler pays two long trips per CZ (163 364 total distance, 13 364 depth on the scored set). Equivalence is checked exactly (QCEC up to 19 qubits, interaction-pair check above).");

// ------------------------------------------------------------------ 2 circuits
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("The circuits: one hub against everyone", { placeholder: "title" });
const gx = 3.3, gy = 4.05, R = 2.15, N = 12;
const node = (i) => [gx + R * Math.cos((2 * Math.PI * i) / N - Math.PI / 2), gy + R * Math.sin((2 * Math.PI * i) / N - Math.PI / 2)];
const lineBetween = (a, b, color, width, name) => {
  const [x1, y1] = a, [x2, y2] = b;
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(0.01, Math.abs(x2 - x1)), h: Math.max(0.01, Math.abs(y2 - y1)), flipH: (x2 - x1) * (y2 - y1) < 0, line: { color, width }, objectName: nm(name) });
};
for (let i = 0; i < N; i++) lineBetween([gx, gy], node(i), "F0D48A", 1.25, "spoke");
for (let i = 0; i < N; i++) lineBetween(node(i), node((i + 1) % N), "9DB7D9", 2, "ringEdge");
[[0, 5], [2, 8], [3, 10], [6, 11]].forEach(([a, b]) => lineBetween(node(a), node(b), "9DB7D9", 2, "chord"));
for (let i = 0; i < N; i++) { const [x, y] = node(i); s.addShape(pres.shapes.OVAL, { x: x - 0.2, y: y - 0.2, w: 0.4, h: 0.4, fill: { color: HEX.blue }, line: { color: HEX.white, width: 1.5 }, objectName: nm("node") }); }
s.addShape(pres.shapes.OVAL, { x: gx - 0.32, y: gy - 0.32, w: 0.64, h: 0.64, fill: { color: HEX.gold }, line: { color: HEX.white, width: 2 }, objectName: "hubNode" });
s.addText("H", { x: gx - 0.32, y: gy - 0.32, w: 0.64, h: 0.64, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "hubLabel" });
caption(s, "iHVA interaction graph (12 qubits): ring + random chords, plus the hub", 1.0, 6.45, 5);
stat(s, 7.2, 1.45, 5.5, "2(n − 1)", "serial CZs on the hub per layer with a naive decomposition (98 at 50 qubits): sets depth and pulse count", HEX.gold);
stat(s, 7.2, 3.15, 5.5, "52 %", "of distance is re-sorting atoms between rounds (measured on solo's plans)", HEX.orange);
bullets(s, ["Blocks are Pauli rotations exp(−iθ/2 Z⊗Y)", "Non-hub graph needs up to 6 matchings per layer; consecutive matchings share no structure"], 7.2, 4.95, 5.5, 1.6, 15);
s.addNotes("These two facts drive everything: the hub chain sets the depth and the number of pulses; re-pairing between rounds sets most of the distance.");

// ------------------------------------------------------------------ 3 layers
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
s.addText("Three layers, three questions", { placeholder: "title" });
const L = [
  ["L1", "Gate form", "Which native gates realise each block?", "fixes the gate list and depth", HEX.gold],
  ["L2", "Schedule", "Which pairs pulse together, in which order?", "fixes the pulse sequence", HEX.orange],
  ["L3", "Movement", "Where are the atoms, and which rectangle moves get them there?", "changes distance only", HEX.blue],
];
L.forEach(([tag, name, q, fx, col], i) => {
  const x = 0.6 + i * 4.2;
  card(s, x, 1.7, 3.7, 3.7);
  chip(s, x + 0.3, 1.95, tag, col, 0.75, 20);
  s.addText(name, { x: x + 1.2, y: 2.0, w: 2.3, h: 0.65, fontSize: 24, bold: true, color: C.text1, fontFace: THEME.headFontFace, margin: 0, valign: "middle", isTextBox: true, objectName: nm("layerName") });
  s.addText(q, { x: x + 0.3, y: 3.0, w: 3.1, h: 1.4, fontSize: 18, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: nm("layerQ") });
  s.addText(fx, { x: x + 0.3, y: 4.55, w: 3.1, h: 0.6, fontSize: 14, italic: true, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: nm("layerFix") });
  if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 3.78, y: 3.35, w: 0.34, h: 0.4, fill: { color: "C9D1DA" }, line: { color: "C9D1DA", width: 0 }, objectName: nm("arrow") });
});
s.addText("Every improvement in this talk belongs to exactly one layer, so lines of work could run in parallel and be merged.", { x: 0.6, y: 5.8, w: 12.1, h: 0.6, fontSize: 17, color: C.text1, margin: 0, isTextBox: true, objectName: "layersNote" });
s.addNotes("L1 fixes the gate list and its depth; L2 fixes the pulse sequence; L3 only changes distance (depth is measured with moves removed). Keeping the layers separate let several people and agents work in parallel on different layers and merge the results.");

// ------------------------------------------------------------------ 4 L1 relay
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
s.addText("L1: blocks without basis changes, and the relay", { placeholder: "title" });
chip(s, 0.6, 1.45, "L1", HEX.gold, 0.5, 13);
bullets(s, [
  [{ text: "S-frame: ", bold: true }, { text: "conjugating by S turns Z⊗Y into Z⊗X = CZ · rx(θ) · CZ. Only the rotor visits storage between the pulses" }],
  [{ text: "Special angles: ", bold: true }, { text: "±π/2 → one CZ + local Cliffords; π → local Paulis; 0 dropped" }],
  [{ text: "Relayed hub fan-out: ", bold: true }, { text: "F = ∏ cz(hub, p) depends only on the partners' parity. Pairs fold their parity into one root in parallel; the hub meets one root per group" }],
], 1.25, 1.45, 5.75, 3.4, 15);
card(s, 0.6, 4.85, 6.4, 1.0);
s.addText("cz(hub, l) · cz(hub, r) = P · cz(hub, r) · P,   P = rx_r(−π/2) · cz(l, r) · rx_r(π/2)", { x: 0.8, y: 4.95, w: 6.0, h: 0.8, fontSize: 13, fontFace: "Courier New", color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "relayIdentity" });
stat(s, 0.6, 5.85, 6.4, "depth 1 856 → 1 486", "solo2 → relay on the scored set (+91.56 → +92.01)", HEX.gold, 26);
picture(s, "relay_parity.png", 7.35, 1.45, 5.35, FRAME_ASPECT, "Seven partner pairs pulse in parallel in the entangling zone while the hub waits in storage");
caption(s, "Animation 0:19 · 7 partner pairs collect parity in parallel; the hub (gold) waits in storage. Next: 8 hub pulses instead of 16.", 7.35, 5.5, 5.35);
webLink(s, ANIMATION_URL, 7.35, 6.05, 5.35);
s.addNotes("This is the step from solo2 (+91.56) to relay (+92.01): total depth 1 856 → 1 486 on the scored set. A hub batch is F · ∏ rx(θ_p) · F with F = ∏ cz(hub, p). Partner pairs (or paths of four) fold their parity into one root; roots alternate between the two passes so every input pair still interacts (validator rule above 19 qubits). The identity was checked numerically on 6 qubits before any compiler work, and every compiled circuit passes the official equivalence check.");

// ------------------------------------------------------------------ 5 L2 + L3
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
s.addText("L2 + L3: rounds, hub batches and the storage band", { placeholder: "title" });
picture(s, "band_round.png", 0.6, 1.45, 5.35, FRAME_ASPECT, "A band of ten pairs pulsing together in the entangling zone");
caption(s, "Animation 0:01 · a band round: 10 pairs lifted as one rectangle and pulsed together.", 0.6, 5.5, 5.35);
chip(s, 6.4, 1.45, "L2", HEX.orange, 0.5, 13);
bullets(s, [
  [{ text: "Exact commutation DAG ", bold: true }, { text: "of the Pauli strings" }],
  [{ text: "Round: ", bold: true }, { text: "a matching of ready blocks. " }, { text: "Hub batch: ", bold: true }, { text: "cz #1 for every partner, one rx layer, cz #2 in reverse" }],
  [{ text: "Multi-start search, ", bold: true }, { text: "cost = distance + 3.06 · depth" }],
], 7.05, 1.45, 5.65, 2.2, 15);
chip(s, 6.4, 3.75, "L3", HEX.blue, 0.5, 13);
bullets(s, [
  [{ text: "Dynamic storage band: ", bold: true }, { text: "no fixed homes; Hungarian assignment of pairs to a storage rectangle, then one rigid lift for all pairs" }],
  [{ text: "Hub stream: ", bold: true }, { text: "hub in zone row 10, partners from storage row 9, rows rotate through row 9" }],
], 7.05, 3.75, 5.65, 1.9, 15);
stat(s, 6.4, 5.55, 6.3, "≈ 2.1 per band move", "for up to 10 pairs (2-row band); the hub stream is near its floor at ≈ 3.2 per hub pulse", HEX.blue, 26);
s.addNotes("The rigid band is why moves are cheap per CZ: one 2-row band move costs about 2.1 for up to 10 pairs. Per round: lift, pulse, rotors dip for rx(θ), lift, pulse, drop, about 4 moves for all pairs. The hub stream is close to its floor (about 3.2 per hub pulse: drop, hop, lift).");

// ------------------------------------------------------------------ 5b L3 movement summary (for the 5-minute talk)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
s.addText("L3: moving atoms cheaply", { placeholder: "title" });
// left: staging row + row rotation diagram
card(s, 0.6, 1.4, 5.0, 5.35);
chip(s, 0.85, 1.65, "L3", HEX.blue, 0.5, 13);
s.addText("Hub stream: staging row + row rotation", { x: 1.5, y: 1.65, w: 4.0, h: 0.5, fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "l3dTitle" });
const zx = 0.95, zy = 2.4, zw = 4.3;
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: zx, y: zy, w: zw, h: 0.8, rectRadius: 0.08, fill: { color: HEX.zone }, line: { color: HEX.zone, width: 0 }, objectName: "l3zone" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: zx, y: zy + 0.9, w: zw, h: 1.85, rectRadius: 0.08, fill: { color: HEX.sto }, line: { color: HEX.sto, width: 0 }, objectName: "l3sto" });
s.addText("entangling zone, row 10", { x: zx + 0.1, y: zy + 0.04, w: 2.4, h: 0.24, fontSize: 10, color: "B8662A", margin: 0, isTextBox: true, objectName: "l3zoneLbl" });
const hubX = zx + 1.6, rowY = [zy + 1.1, zy + 1.65, zy + 2.2], rowCol = [HEX.blue, "7FA6D6", "B9CCE6"], rowName = ["row 9 · staging", "row 8", "row 7"];
s.addShape(pres.shapes.OVAL, { x: hubX, y: zy + 0.33, w: 0.36, h: 0.36, fill: { color: HEX.gold }, line: { color: HEX.white, width: 1 }, objectName: "l3hub" });
s.addShape(pres.shapes.OVAL, { x: hubX + 0.4, y: zy + 0.33, w: 0.36, h: 0.36, fill: { color: HEX.blue }, line: { color: HEX.white, width: 1 }, objectName: "l3partner" });
rowY.forEach((y, r) => {
  for (let k = 0; k < 7; k++) s.addShape(pres.shapes.OVAL, { x: zx + 0.15 + k * 0.4, y, w: 0.3, h: 0.3, fill: { color: rowCol[r] }, line: { color: HEX.white, width: 1 }, objectName: nm("l3row") });
  s.addText(rowName[r], { x: zx + 3.05, y: y - 0.02, w: 1.2, h: 0.34, fontSize: 10, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: nm("l3rowLbl") });
});
s.addShape(pres.shapes.LINE, { x: hubX + 0.58, y: zy + 0.72, w: 0.01, h: rowY[0] - zy - 0.72, line: { color: HEX.purple, width: 2, dashType: "dash", beginArrowType: "triangle" }, objectName: "l3lift" });
bullets(s, [
  "The hub waits in the zone; partners rise one by one from row 9 and drop back (≈ 1.1 each)",
  "Row 9 used up: one move shifts all rows down by one, one move lifts the next row into row 9",
  "≈ 6 per row of 10 partners at 50 qubits",
], 0.85, 5.2, 4.55, 1.5, 12);
// right: the other movement tricks, one row each
const TRICKS = [
  ["Rigid band rounds", "A round's pairs sit as one band in storage; one lift carries all of them, rotors dip and come back.", "≈ 2.1 / move, 10 pairs", HEX.blue],
  ["Hub slot hopping", "The hub moves between the two sites of its pair (0.3 apart), so partners rise straight up from two columns.", "0.3 per hop", HEX.gold],
  ["Gather lifts", "Small rounds: each row segment of members goes straight to its pair slots in one move, no shuffling first.", "rounds ≤ 5 pairs", HEX.orange],
  ["Dip trimming", "Only atoms that owe a rotation go to storage, straight down, instead of taking whole rows along.", "1.22 → 1.04 / move", HEX.sky],
  ["Exact solver → fast pass", "MILP on 3-move windows of finished plans; the patterns became rewrite rules that run in seconds.", "−4.1 % distance", HEX.purple],
];
TRICKS.forEach(([t, d, num, col], i) => {
  const y = 1.4 + i * 1.08;
  card(s, 5.85, y, 6.85, 0.95);
  chip(s, 6.05, y + 0.22, String(i + 1), col, 0.5, 14);
  s.addText(t, { x: 6.75, y: y + 0.08, w: 3.6, h: 0.36, fontSize: 15, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: nm("trickTitle") });
  s.addText(d, { x: 6.75, y: y + 0.44, w: 4.25, h: 0.48, fontSize: 11, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: nm("trickText") });
  s.addText(num, { x: 11.05, y: y + 0.08, w: 1.5, h: 0.8, fontSize: 13, bold: true, color: col, align: "right", margin: 0, valign: "middle", isTextBox: true, objectName: nm("trickNum") });
});
s.addNotes("Between the pulses, the movement layer decides where atoms wait and which rigid moves bring them up. The picture on the left is the hub stream: the hub waits in the zone and its partners rise one at a time from storage row 9, the staging row. When that row is used up, one move shifts every stored row down by one and one move lifts the next row into row 9: about six per row of ten partners at 50 qubits. On the right the other tricks: a round's pairs are lifted as one band, about 2.1 per move for up to ten pairs; the hub hops between the two sites of its pair so partners come straight up from two columns; small rounds gather each row segment straight into its slots; only atoms that owe a rotation dip, straight down. And we solved small windows of finished plans exactly with a MILP and turned the patterns into a clean-up pass: 4 percent less distance on top of relay, gates untouched.");

// ------------------------------------------------------------------ 6 formal L3
pres.addSection({ title: "Our L3 work" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our L3 work" });
s.addText("L3 as a formal optimisation problem", { placeholder: "title" });
card(s, 0.6, 1.45, 6.6, 2.55);
s.addText([
  { text: "min  Σ_k g_k(X_{k−1}, X_k)", options: { bold: true, breakLine: true } },
  { text: "X_k  : atom configuration at pulse k", options: { breakLine: true } },
  { text: "s.t. pulse pairs on interaction pairs, all others in storage", options: { breakLine: true } },
  { text: "     atoms owing 1q gates touch storage between pulses", options: { breakLine: true } },
  { text: "g_k  : cheapest legal rectangle-move sequence", options: {} },
], { x: 0.85, y: 1.6, w: 6.2, h: 2.3, fontSize: 15, fontFace: "Courier New", color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "formulation" });
bullets(s, [
  [{ text: "Exact model: ", bold: true }, { text: "time-expanded MILP (HiGHS): one-hot positions, picked row/column intervals, monotone integer maps, ghost rule built in, single-zone, occupancy, polygonal Euclidean norm" }],
  [{ text: "Full instances do not scale, ", bold: true }, { text: "so the exact model runs on sliding windows of 3 moves over a complete plan (MIP-based large neighbourhood search)" }],
  [{ text: "Gates and per-qubit order never change: ", bold: true }, { text: "depth and equivalence preserved by construction; every rewrite replayed with the validator's rules" }],
], 0.6, 4.25, 6.6, 2.5, 14);
stat(s, 7.9, 1.6, 4.8, "−4 to −7 %", "distance from exact 3-move windows on complete 12-qubit plans", HEX.purple);
stat(s, 7.9, 3.3, 4.8, "ground truth", "the exact model shows what is locally possible; it is not run inside the compiler", HEX.blue, 30);
s.addNotes("The point of the formal model is ground truth for what is locally possible, not running it in the compiler. g_k is a set-partitioning problem: cover the required atom transfers with legal rectangle moves. Windows: fix the configurations before and after 3 consecutive moves, re-solve exactly, splice in if cheaper.");

// ------------------------------------------------------------------ 7 peephole
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our L3 work" });
s.addText("From exact solutions to a fast pass", { placeholder: "title" });
s.addText("We recorded every window the MILP improved and turned the recurring patterns into peephole operators: no solver, seconds per circuit.", { x: 0.6, y: 1.35, w: 12.1, h: 0.55, fontSize: 16, color: C.text2, margin: 0, isTextBox: true, objectName: "peepIntro" });
const OPS = [
  ["Merge", "two moves that form one rigid rectangle become one move"],
  ["Dip trim", "only atoms that owe 1q gates go to storage, straight down"],
  ["Ride-along", "a long move is split so half of it shares another move's rectangle"],
  ["Compose", "a drop and the following shuffle become one drop with a re-targeted column map"],
  ["Stack-lift", "two lifts from different rows become one two-row lift"],
  ["Round-trip retarget", "a rigid dip is shifted to free rows closer to the zone"],
];
OPS.forEach(([t, d], i) => {
  const x = 0.6 + (i % 3) * 2.75, y = 2.1 + Math.floor(i / 3) * 2.05;
  card(s, x, y, 2.55, 1.85);
  chip(s, x + 0.2, y + 0.2, String(i + 1), HEX.purple, 0.42, 13);
  s.addText(t, { x: x + 0.75, y: y + 0.2, w: 1.7, h: 0.42, fontSize: 15, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: nm("opTitle") });
  s.addText(d, { x: x + 0.2, y: y + 0.75, w: 2.2, h: 1.0, fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: nm("opText") });
});
stat(s, 9.1, 2.1, 3.6, "11 770 → 11 284", "relay's total distance with the pass (−4.1 %), depth unchanged", HEX.blue, 28);
bullets(s, ["Every rewrite replayed exactly: legal moves, same end configuration, required storage visits", "Compiler-agnostic: wraps any compiler; ≤ 25 s at 50 qubits"], 9.1, 3.75, 3.6, 2.4, 13);
s.addNotes("Two mining rounds: first on solo's plans, then again on the already-optimised relay plans, which surfaced four new operators (compose, stack-lift, round-trip retarget, map-preimage ride-along). Each candidate rewrite is kept only if the exact replay is cheaper.");

// ------------------------------------------------------------------ 8 process
pres.addSection({ title: "Process & results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Process & results" });
s.addText("How we worked: iterate, measure, merge", { placeholder: "title" });
const STEPS = ["Profile", "Hypothesis", "Experiment", "Validate", "Merge"];
STEPS.forEach((t, i) => {
  const x = 0.6 + i * 1.3;
  chip(s, x + 0.2, 1.45, String(i + 1), [HEX.blue, HEX.gold, HEX.orange, HEX.purple, HEX.ink][i], 0.55, 15);
  s.addText(t, { x, y: 2.05, w: 0.95, h: 0.35, fontSize: 12, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: nm("stepLabel") });
});
bullets(s, [
  [{ text: "Parallel lines of work ", bold: true }, { text: "on separate branches (team and AI coding agents), one shared harness: official validator + transversal-CZ and zone-span checks" }],
  [{ text: "Every result validated ", bold: true }, { text: "on 20 scored + 13 sanity circuits before merging" }],
  [{ text: "Merging lines: ", bold: true }, { text: "relay forked before later solo improvements; a 3-way merge combined them (+92.16), then search variants and the peephole pass" }],
  [{ text: "Visualiser with diff mode: ", bold: true }, { text: "two plans aligned on their CZ pulses, differences red/green, click jumps both to the equivalent state" }],
], 0.6, 2.7, 6.2, 4.0, 14);
picture(s, "viz_diff_split.png", 7.1, 1.45, 5.6, VIZ_ASPECT, "Visualiser split view in diff mode comparing two compilations");
caption(s, "Split view, diff mode: relay_best_l3 (left) vs relay_l3 (right); red = only left, green = only right.", 7.1, 4.85, 5.6);
s.addImage({
  path: IMG("visualiser_qr.png"), x: 7.1, y: 5.35, w: 1.4, h: 1.4,
  altText: "QR code for the interactive visualiser", hyperlink: { url: VISUALISER_URL },
  objectName: nm("visualiserQr"),
});
s.addText("Try the visualiser", {
  x: 8.7, y: 5.63, w: 4.0, h: 0.35, fontSize: 17, bold: true, color: C.text1,
  margin: 0, isTextBox: true, objectName: nm("visualiserLabel"),
});
webLink(s, VISUALISER_URL, 8.7, 6.05, 4.0);
s.addNotes("Most ideas were judged by watching both plans side by side, for example the tall storage bands at 50 qubits (5 rows, so at least 5 per band move) and dips that took whole rows down when one atom needed storage. Those observations turned into the band-width search option and the dip-trim operator.");

// ------------------------------------------------------------------ 9 results
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Process & results" });
s.addText("Results", { placeholder: "title" });
const names = ["systolic", "blockingcz", "solo", "solo2", "relay", "relay_l3", "relay_best_l3"];
const vals = [85.34, 90.48, 91.48, 91.56, 92.01, 92.25, 92.45];
s.addChart(pres.charts.BAR, [{ name: "score (%)", labels: names, values: vals }], {
  x: 0.6, y: 1.4, w: 7.4, h: 5.2, barDir: "bar", catAxisOrientation: "maxMin",
  valAxisMinVal: 84, valAxisMaxVal: 93, valAxisMajorUnit: 1,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 12, dataLabelColor: HEX.ink,
  chartColors: [HEX.blue], showLegend: false, showTitle: true, title: "Score vs naive (%), 20 scored circuits", titleFontSize: 14, titleColor: HEX.ink,
  catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 13, valAxisLabelFontSize: 11,
  valGridLine: { color: "E3E7EC", size: 1 }, catGridLine: { style: "none" },
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  objectName: "resultsChart",
});
stat(s, 8.5, 1.45, 4.2, "−93.2 %", "total move distance vs naive (11 113 vs 163 364)", HEX.blue);
stat(s, 8.5, 3.05, 4.2, "−89.4 %", "depth vs naive (1 411 vs 13 364)", HEX.orange);
stat(s, 8.5, 4.65, 4.2, "20/20 · 13/13", "valid in full mode · sanity circuits; ≤ ~200 s per 50-qubit circuit", HEX.gold, 32);
s.addNotes("Progression: systolic +85.34 (storage homes + hub stream), blockingcz +90.48 (S-frame, matchings), solo +91.48 (dynamic storage band, special angles), solo2 +91.56 (hub as rotor, gather lifts), relay +92.01 (relayed hub fan-out), relay_l3 +92.25 (peephole pass), relay_best_l3 +92.45 (later solo improvements merged, search variants, rollout finalists, peephole). Example from the animation: 20-qubit iHVA, 106 moves, distance 199 vs 3 274 naive (−93.9 %). Wall-clock guards skip optional stages on slow machines; on an idle machine the score with guards is identical.");

// ------------------------------------------------------------------ 10 next
s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Process & results" });
s.addText("What limits the next step", { placeholder: "title" });
const NEXT = [
  ["Local search is saturated", "Exact windows, the peephole pass and full-horizon rollout lookahead agree: the greedy band choice is locally optimal.", HEX.gold],
  ["Re-sorting dominates", "The remaining distance is re-sorting atoms between unrelated matchings under order-preserving rectangle moves.", HEX.orange],
  ["Next levers", "Joint L2 + L3 search (pick the next round by its real movement cost) and a storage layout that keeps rows near the zone free for wide, shallow bands.", HEX.sky],
];
NEXT.forEach(([t, d, col], i) => {
  const x = 0.6 + i * 4.2;
  card(s, x, 1.8, 3.8, 4.2, "26323F");
  chip(s, x + 0.3, 2.1, String(i + 1), col, 0.6, 16);
  s.addText(t, { x: x + 0.3, y: 2.95, w: 3.2, h: 0.8, fontSize: 22, bold: true, color: C.background1, fontFace: THEME.headFontFace, margin: 0, valign: "top", isTextBox: true, objectName: nm("nextTitle") });
  s.addText(d, { x: x + 0.3, y: 3.85, w: 3.2, h: 2.0, fontSize: 15, color: "CADCFC", margin: 0, valign: "top", isTextBox: true, objectName: nm("nextText") });
});
s.addNotes("Shallow bands would cost about 2.1 instead of at least 5 per band move at 50 qubits; the open question is the cost of keeping storage arranged for them.");

// ------------------------------------------------------------------ write
(async () => {
  const out = path.join(HERE, "relay_deck.pptx");
  await pres.writeFile({ fileName: out });
  const { applyTheme } = require(process.argv[2]);
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
