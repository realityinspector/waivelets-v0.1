#!/usr/bin/env python3
"""Final consolidated ladder SVG showing every method tested."""
import pathlib

OUT = pathlib.Path("/tmp/waivelets-v0.1/web/assets/model_ladder_final.svg")

# Ordered worst-to-best for clarity
rows = [
    ("Random guess",                   0.125, "#606080", "8-way chance"),
    ("Original 7-D fingerprint card",  0.116, "#606080", "the published summary"),
    ("Per-sentence + majority vote",   0.531, "#a78bfa", "doc-level vote of LDA on sentences"),
    ("Gutenberg basis · 78-D readout", 0.583, "#f59e0b", "literary structure axes"),
    ("LLM-PCA basis · 78-D readout",   0.698, "#22d3ee", "unsupervised LLM corpus axes"),
    ("Spec Gut + LLM-PCA · 156-D",     0.760, "#34d399", "← prior headline"),
    ("Spec Gut + Wavelet · 156-D",     0.844, "#6ee7b7", "wavelet methodology, no labels"),
    ("LDA basis alone · 21-D",         0.854, "#818cf8", "supervised — labels in basis"),
    ("Contrastive triplet · 48-D",     0.833, "#a5b4fc", "metric learning"),
    ("Adversarial topic-debiased",     0.885, "#34d399", "← new headline · 48-D"),
    ("Raw 384-D centroid (ceiling)",   0.896, "#22d3ee", "all-features upper bound"),
]

W, H = 1100, 660
pad_l, pad_b, pad_t = 320, 100, 120

plot_w = W - pad_l - 130
plot_h = H - pad_t - pad_b

s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
s.append(f'<text x="{W/2}" y="42" fill="#f0f0f8" font-size="22" font-weight="700" text-anchor="middle">Closing the gap to the upper bound</text>')
s.append(f'<text x="{W/2}" y="66" fill="#9090b0" font-size="12" text-anchor="middle">All conditions: leave-one-document-out · 8-way model identification · 96 generations from 8 frontier LLMs</text>')
s.append(f'<text x="{W/2}" y="88" fill="#34d399" font-size="14" font-weight="600" text-anchor="middle">88.5% with the adversarial topic-debiased basis — 48 dimensions, 7× chance, within 1pp of the 384-D ceiling.</text>')

def sy(i): return pad_t + i * (plot_h / len(rows))
bar_h = (plot_h / len(rows)) * 0.62

for i, (lab, val, col, note) in enumerate(rows):
    y = sy(i) + (plot_h / len(rows)) * 0.19
    s.append(f'<text x="{pad_l-14}" y="{y+bar_h/2+5}" fill="#d0d0e0" font-size="13" text-anchor="end">{lab}</text>')
    bw = (val / 1.0) * plot_w
    opacity = "0.95" if "headline" in note or "←" in note else "0.7"
    s.append(f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{col}" opacity="{opacity}" rx="3"/>')
    s.append(f'<text x="{pad_l+bw+10}" y="{y+bar_h/2+5}" fill="#f0f0f8" font-size="14" font-weight="700" font-family="JetBrains Mono,monospace">{val*100:.1f}%</text>')
    s.append(f'<text x="{pad_l+bw+85}" y="{y+bar_h/2+5}" fill="#606080" font-size="10.5">{note}</text>')

# axis ticks
yt = pad_t + plot_h + 12
for tv in [0, 0.25, 0.5, 0.75, 1.0]:
    tx = pad_l + tv * plot_w
    s.append(f'<line x1="{tx}" y1="{yt-4}" x2="{tx}" y2="{yt+4}" stroke="#606080"/>')
    s.append(f'<text x="{tx}" y="{yt+18}" fill="#606080" font-size="10" text-anchor="middle">{int(tv*100)}%</text>')

# chance line
cx = pad_l + (1/8) * plot_w
s.append(f'<line x1="{cx}" y1="{pad_t-2}" x2="{cx}" y2="{pad_t+plot_h+2}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.4"/>')
s.append(f'<text x="{cx+5}" y="{pad_t-2}" fill="#f87171" font-size="10" opacity="0.7">chance</text>')

# ceiling line
cy = pad_l + 0.896 * plot_w
s.append(f'<line x1="{cy}" y1="{pad_t-2}" x2="{cy}" y2="{pad_t+plot_h+2}" stroke="#22d3ee" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.4"/>')
s.append(f'<text x="{cy-5}" y="{pad_t-2}" fill="#22d3ee" font-size="10" opacity="0.7" text-anchor="end">ceiling</text>')

s.append(f'<text x="{pad_l + plot_w/2}" y="{H-25}" fill="#9090b0" font-size="11" text-anchor="middle">8-way model-ID accuracy (leave-one-DOCUMENT-out cross-validation)</text>')
s.append('</svg>')

OUT.write_text("\n".join(s))
print(f"wrote {OUT}")
