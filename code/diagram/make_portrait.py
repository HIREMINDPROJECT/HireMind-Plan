"""Portrait pipeline diagram (large text) -> pipeline_portrait.html, .png, .pdf"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
W = 920
LX, LW = 20, 540           # main column
RX, RW = 600, 270          # side column
parts = []
def box(x, y, w, h, title, lines, cls="b"):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="{cls}"/>')
    parts.append(f'<text x="{x+18}" y="{y+32}" class="t">{title}</text>')
    for i, l in enumerate(lines):
        parts.append(f'<text x="{x+18}" y="{y+60+i*24}" class="s">{l}</text>')
def arrow(d, cls="l"): parts.append(f'<path d="{d}" class="{cls}"/>')

y = 96; GAP = 30; ys = {}
chain = [
    ("in", "Inputs", ["CV text + job description", "(one pair = one case)"]),
    ("b1", "B1 Extraction (rule-based, schema v3.1)", ["requirements and CV claims: skills, education,", "years, certificates, dates — each with its text position"]),
    ("b3", "B3 Matching", ["exact match + DistilBERT similarity per requirement", "found / not mentioned / explicitly negative"]),
    ("b4", "B4 Evidence input", ["requirement sentences + linked CV evidence", "within 512 tokens, plus the Knowledge v1 line"]),
    ("cl", "Classifier (frozen in E1)", ["M4 BERT-Tiny + Knowledge v1, original labels, seed 46", "compared: M1, M2, M3, M5 DistilBERT (+ Knowledge", "through E0–E2), M6, M8"]),
    ("su", "Suggestion", ["No / Potential / Good Fit with probabilities", "low confidence → abstain; P(Good Fit) ≥ 0.30 → review flag"]),
    ("hr", "HR review (prototype v3) — the human decides", ["evidence for every requirement; “not mentioned” ≠ lacks", "“Check these first” list; reason required to override", "no automatic hiring or rejection"]),
]
for i, (k, t, ls) in enumerate(chain):
    h = 52 + 24 * len(ls) + 14
    box(LX, y, LW, h, t, ls, "h" if k == "hr" else "b")
    ys[k] = (y, h)
    if i: arrow(f"M{LX+LW//2},{prev_end}V{y-4}")
    prev_end = y + h
    y += h + GAP
end_chain = y - GAP

# training labels (side), feeding the classifier
cy, ch = ys["cl"]
ly = cy - 150
box(RX, ly, RW, 236, "Training labels (A)", ["D0 original labels kept", "A4: 400 TRAIN cases,", "2 reviewers + adjudication", "→ 255 approved corrections", "→ D1 overlay (A5)", "→ trains the classifier", "test set never used"], "r")
arrow(f"M{RX},{cy+ch//2}H{LX+LW+4}", "d")
# evidence spans to HR
by, bh = ys["b3"]; hy, hh = ys["hr"]
arrow(f"M{LX+LW},{by+bh//2}H{RX+RW+20}V{hy+hh//2}H{LX+LW+4}")
parts.append(f'<text x="{RX}" y="{by+bh//2-10}" class="m">evidence spans shown to HR</text>')

# evaluation box
ey = end_chain + 34
ev = ["B2: extractor vs human annotation (P 0.51 / R 0.38)",
      "B3: similarity is not proof (AUC 0.58) · B4: evidence kept 33% → 80%",
      "E0–E5 on the frozen VALIDATION split, mean ± SD over seeds",
      "Selection rule fixed in advance: beat M4 on accuracy AND Good Fit",
      "recall, with no drop > 2 points in GF precision, macro-F1, Potential recall",
      "E1: weights SHA-256, re-loaded result 56.97 · E3: prototype 33/33 tests,",
      "5 HR participants, platform 106/106 tests with the frozen model"]
box(LX, ey, W - 2 * LX + 10, 52 + 24 * len(ev) + 14, "Evaluation and checks (Phases B–E)", ev)
H = ey + 52 + 24 * len(ev) + 14 + 16

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W+10}" height="{H}" viewBox="0 0 {W+10} {H}">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#444"/></marker></defs>
<style>.b{{fill:#f6f4ee;stroke:#555;stroke-width:1.5}}.h{{fill:#e8f1fb;stroke:#1c5fa8;stroke-width:1.8}}.r{{fill:#fdf0e6;stroke:#a15c00;stroke-width:1.8}}
.t{{font-size:20px;font-weight:bold;fill:#1c1b19}}.s{{font-size:17px;fill:#333}}.m{{font-size:15px;fill:#444;font-style:italic}}
.l{{stroke:#444;stroke-width:2;fill:none;marker-end:url(#a)}}.d{{stroke:#a15c00;stroke-width:2;fill:none;stroke-dasharray:6 4;marker-end:url(#a)}}
.c{{font-size:26px;font-weight:bold;fill:#1c1b19}}.st{{font-size:17px;fill:#555}}</style>
<text x="20" y="38" class="c">HireMind pipeline</text>
<text x="20" y="66" class="st">Advice only: the model suggests, HR decides. Phase letters refer to the supervisor's plan.</text>
{"".join(parts)}
</svg>'''
html = '<!doctype html><meta charset="utf-8"><style>body{margin:0;background:#fff;font-family:Arial,Helvetica,sans-serif}</style>' + svg
(out / "pipeline_portrait.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": W + 10, "height": H}, device_scale_factor=2)
    pg.set_content(html); pg.locator("svg").screenshot(path=str(out / "HireMind_pipeline_diagram_portrait.png"))
    pg.pdf(path=str(out / "HireMind_pipeline_diagram_portrait.pdf"), width=f"{W+10}px", height=f"{H+2}px", print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()
print("size", W + 10, H)
