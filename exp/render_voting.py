#!/usr/bin/env python3
"""Render per-sentence vote vs spectrometer SVG."""
import json, pathlib

R = json.loads(pathlib.Path("/tmp/exp/per_sent_vote_v2.json").read_text())
OUT = pathlib.Path("/tmp/waivelets-v0.1/web/assets/model_voting.svg")

W, H = 920, 480
pad_l, pad_b, pad_t = 240, 100, 110

rows = [
    ("Random guess",                 R["chance"],                          "#606080", "1 in 8"),
    ("Per-sentence (single sent)",   R["per_sentence_acc"],                "#f59e0b", f"n={R['n_sentences']} sentences"),
    ("Confidence-weighted vote",     R["confwt_doc_acc"],                  "#a78bfa", "mean LDA probability"),
    ("Majority vote",                R["majority_vote_doc_acc"],           "#22d3ee", "argmax of vote counts"),
    ("Doc-level spectrometer",       R["spectrometer_baseline_156d"],      "#34d399", "current 156-D · ←"),
]

plot_w = W - pad_l - 100
plot_h = H - pad_t - pad_b

s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
s.append(f'<text x="{W/2}" y="40" fill="#f0f0f8" font-size="20" font-weight="700" text-anchor="middle">Does per-sentence + voting help?</text>')
s.append(f'<text x="{W/2}" y="64" fill="#9090b0" font-size="12" text-anchor="middle">Each sentence gets classified independently; the document\'s prediction = majority vote across its sentences.</text>')
s.append(f'<text x="{W/2}" y="86" fill="#f87171" font-size="13" font-weight="600" text-anchor="middle">No — voting trails the doc-level spectrometer by ~23 pp.</text>')

def sy(i): return pad_t + i * (plot_h / len(rows))
bar_h = (plot_h / len(rows)) * 0.65

for i, (lab, val, col, note) in enumerate(rows):
    y = sy(i) + (plot_h / len(rows)) * 0.18
    s.append(f'<text x="{pad_l-14}" y="{y+bar_h/2+4}" fill="#d0d0e0" font-size="13" text-anchor="end">{lab}</text>')
    bw = (val / 1.0) * plot_w
    s.append(f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{col}" opacity="0.85" rx="3"/>')
    s.append(f'<text x="{pad_l+bw+10}" y="{y+bar_h/2+5}" fill="#f0f0f8" font-size="15" font-weight="700" font-family="JetBrains Mono,monospace">{val*100:.1f}%</text>')
    s.append(f'<text x="{pad_l+bw+80}" y="{y+bar_h/2+5}" fill="#606080" font-size="11">{note}</text>')

# axis ticks
yt = pad_t + plot_h + 12
for tv in [0, 0.25, 0.5, 0.75, 1.0]:
    tx = pad_l + tv * plot_w
    s.append(f'<line x1="{tx}" y1="{yt-4}" x2="{tx}" y2="{yt+4}" stroke="#606080"/>')
    s.append(f'<text x="{tx}" y="{yt+18}" fill="#606080" font-size="10" text-anchor="middle">{int(tv*100)}%</text>')

# chance line
cx = pad_l + (1/8) * plot_w
s.append(f'<line x1="{cx}" y1="{pad_t-2}" x2="{cx}" y2="{pad_t+plot_h+2}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.5"/>')

s.append(f'<text x="{pad_l + plot_w/2}" y="{H-25}" fill="#9090b0" font-size="11" text-anchor="middle">8-way model-ID accuracy (leave-one-DOCUMENT-out)</text>')
s.append('</svg>')

OUT.write_text("\n".join(s))
print(f"wrote {OUT}")
