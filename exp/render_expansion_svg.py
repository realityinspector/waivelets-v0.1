#!/usr/bin/env python3
"""Render the model-expansion bar-chart SVG (matching model_ladder.svg style).

Reads /tmp/exp/expansion_results.json and writes
/tmp/waivelets-v0.1/web/assets/model_expansion.svg.

Bars in green if they beat the 76% baseline by >= 3 percentage points.
"""
import json, pathlib

BASE = 0.76
GREEN_GAIN_PP = 3.0
UPPER = 0.896  # raw 384-D centroid upper bound

res = json.loads(pathlib.Path("/tmp/exp/expansion_results.json").read_text())

ORDER = [
    ("A_gut+llm_baseline",     "A · Gut + LLM-PCA (baseline)"),
    ("B_lda_alone",            "B · LDA basis alone"),
    ("C_triple_spectrometer",  "C · Triple (Gut+LLM+LDA)"),
    ("D_contrastive_alone",    "D · Contrastive basis alone"),
    ("E_quad_spectrometer",    "E · Quad (+contrastive)"),
    ("F_adversarial_alone",    "F · Adversarial basis alone"),
    ("G_quint_spectrometer",   "G · Quint (all 5 bases)"),
]

rows = []
for k, label in ORDER:
    v = res[k]
    rows.append((label, v["acc"], v["dim"], v["null_mean"], v["null_max"]))

# Dimensions
W = 920
H = 100 + 52 * len(rows) + 110
LEFT = 290
BAR_W_MAX = 540
TOP = 110

def x_of(frac):
    return LEFT + BAR_W_MAX * frac

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="Inter,system-ui,sans-serif">')
svg.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
svg.append(f'<text x="{W/2:.1f}" y="40" fill="#f0f0f8" font-size="20" '
           f'font-weight="700" text-anchor="middle">Adding label-aware bases to the spectrometer</text>')
svg.append(f'<text x="{W/2:.1f}" y="62" fill="#9090b0" font-size="12" '
           f'text-anchor="middle">8-way model-ID accuracy · leave-one-out · '
           f'nested CV for every label-aware basis · 96 generations</text>')
gainers = [r for r in rows if r[1] >= BASE + GREEN_GAIN_PP / 100]
if gainers:
    top = max(gainers, key=lambda r: r[1])
    msg = f"Best new condition: {top[0].split(' · ')[0]} at {top[1]*100:.1f}% "\
          f"(+{(top[1]-BASE)*100:.1f}pp over baseline)"
    msg_color = "#34d399"
else:
    msg = "No new condition exceeds baseline by 3pp — labeled bases don't add over Gut+LLM-PCA."
    msg_color = "#f59e0b"
svg.append(f'<text x="{W/2:.1f}" y="82" fill="{msg_color}" font-size="13" '
           f'font-weight="600" text-anchor="middle">{msg}</text>')

# Reference lines: baseline (76%) and upper bound (89.6%)
for ref_frac, ref_label, color in [
    (BASE,  "76% baseline",     "#f87171"),
    (UPPER, "89.6% upper bound","#22d3ee"),
]:
    xr = x_of(ref_frac)
    svg.append(f'<line x1="{xr:.1f}" y1="{TOP-8:.1f}" x2="{xr:.1f}" '
               f'y2="{TOP + 52*len(rows) + 6:.1f}" stroke="{color}" '
               f'stroke-width="1.4" stroke-dasharray="4 4" opacity="0.65"/>')
    svg.append(f'<text x="{xr:.1f}" y="{TOP-12:.1f}" fill="{color}" '
               f'font-size="10.5" text-anchor="middle" opacity="0.95">{ref_label}</text>')

for i, (label, acc, dim, nm, nmax) in enumerate(rows):
    y = TOP + i * 52
    bar_w = BAR_W_MAX * acc
    is_green = acc >= BASE + GREEN_GAIN_PP / 100
    is_baseline = i == 0
    if is_green:
        color = "#34d399"
    elif is_baseline:
        color = "#a78bfa"
    else:
        color = "#f59e0b"
    # Null overlay (mean)
    null_w = BAR_W_MAX * nm
    svg.append(f'<rect x="{LEFT}" y="{y+4:.1f}" width="{null_w:.1f}" '
               f'height="34" fill="#3a3a55" opacity="0.55" rx="3"/>')
    # Main bar
    svg.append(f'<rect x="{LEFT}" y="{y:.1f}" width="{bar_w:.1f}" height="34" '
               f'fill="{color}" opacity="0.88" rx="3"/>')
    # Label
    svg.append(f'<text x="{LEFT-14}" y="{y+22:.1f}" fill="#d0d0e0" '
               f'font-size="12.5" text-anchor="end">{label}</text>')
    # Acc value
    svg.append(f'<text x="{LEFT + bar_w + 8:.1f}" y="{y+22:.1f}" fill="#f0f0f8" '
               f'font-size="13.5" font-weight="700" '
               f'font-family="JetBrains Mono,monospace">{acc*100:.1f}%</text>')
    # dim annotation
    svg.append(f'<text x="{LEFT + bar_w + 64:.1f}" y="{y+22:.1f}" fill="#707090" '
               f'font-size="10.5">{dim}-D · null {nm*100:.1f}/{nmax*100:.1f}%</text>')

# Axis
axis_y = TOP + 52 * len(rows) + 8
svg.append(f'<line x1="{LEFT}" y1="{axis_y:.1f}" x2="{LEFT+BAR_W_MAX:.1f}" '
           f'y2="{axis_y:.1f}" stroke="#404060" stroke-width="1"/>')
for frac, lbl in [(0,"0%"), (0.25,"25%"), (0.50,"50%"), (0.75,"75%"), (1.0,"100%")]:
    xr = x_of(frac)
    svg.append(f'<line x1="{xr:.1f}" y1="{axis_y:.1f}" x2="{xr:.1f}" '
               f'y2="{axis_y+6:.1f}" stroke="#606080"/>')
    svg.append(f'<text x="{xr:.1f}" y="{axis_y+20:.1f}" fill="#707090" '
               f'font-size="10.5" text-anchor="middle">{lbl}</text>')

# Caption
cap_y = axis_y + 48
svg.append(f'<text x="{W/2:.1f}" y="{cap_y:.1f}" fill="#9090b0" font-size="11" '
           f'text-anchor="middle">grey ribbon = permutation-null mean (50 perms) · '
           f'green = beats 76% baseline by ≥3pp</text>')

svg.append("</svg>")

out = pathlib.Path("/tmp/waivelets-v0.1/web/assets/model_expansion.svg")
out.write_text("\n".join(svg))
print(f"Wrote {out}")
