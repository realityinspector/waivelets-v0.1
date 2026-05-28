#!/usr/bin/env python3
"""Render the new (clearer) visualizations:
  1. model_confusion.svg — 8x8 confusion matrix with family blocks
  2. model_ladder.svg — chance vs achieved accuracy bar
  3. model_per_model.svg — per-model recall bar chart
"""
import json, pathlib
import numpy as np

C = json.loads(pathlib.Path("/tmp/exp/confusion.json").read_text())
OUT = pathlib.Path("/tmp/waivelets-v0.1/web/assets")
OUT.mkdir(parents=True, exist_ok=True)

SHORT = C["short"]
order = C["order"]
SN = [SHORT[m] for m in order]
FAMILY = C["family"]
FAMILY_COLOR = {"anthropic": "#f59e0b", "openai": "#22d3ee", "google": "#a78bfa", "meta": "#34d399"}

# ─── 1. Confusion matrix ───
def confusion_svg():
    M = np.array(C["spectrometer"]["confusion_matrix"])
    n = len(order)
    cell = 58
    pad_l, pad_t = 170, 170
    W, H = pad_l + n*cell + 60, pad_t + n*cell + 90
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="36" fill="#f0f0f8" font-size="20" font-weight="700" text-anchor="middle">Which model wrote this? — confusion matrix</text>')
    s.append(f'<text x="{W/2}" y="60" fill="#9090b0" font-size="11.5" text-anchor="middle">Rows = true model · columns = spectrometer prediction · brighter cell = more times the row model was called the column model.</text>')
    s.append(f'<text x="{W/2}" y="78" fill="#9090b0" font-size="11.5" text-anchor="middle">Diagonal = correct. <tspan fill="#34d399" font-weight="600">76% of all calls land on the diagonal.</tspan></text>')

    # family-block backgrounds
    fams = [FAMILY[m] for m in order]
    # row blocks (left labels)
    cur_fam, start = fams[0], 0
    blocks = []
    for i in range(1, n+1):
        if i == n or fams[i] != cur_fam:
            blocks.append((cur_fam, start, i))
            if i < n:
                cur_fam, start = fams[i], i
    for fam, s_idx, e_idx in blocks:
        y_top = pad_t + s_idx * cell
        h = (e_idx - s_idx) * cell
        c = FAMILY_COLOR[fam]
        s.append(f'<rect x="{pad_l-6}" y="{y_top}" width="3" height="{h-4}" fill="{c}" opacity="0.85"/>')
        s.append(f'<rect x="{pad_l + s_idx*cell}" y="{pad_t-12}" width="{(e_idx-s_idx)*cell-2}" height="3" fill="{c}" opacity="0.85"/>')

    # cells
    vmax = M.max()
    for i in range(n):
        row_sum = M[i].sum()
        for j in range(n):
            x = pad_l + j*cell
            y = pad_t + i*cell
            v = M[i, j]
            frac = v / row_sum if row_sum else 0
            # color: black background, white text for diagonal, grayscale for off-diagonal
            if i == j:
                # green intensity proportional to recall
                intensity = v / row_sum if row_sum else 0
                g_int = int(60 + intensity * 180)
                fill = f"rgb(0,{g_int},90)"
                txt = "#f0f0f8"
            elif v == 0:
                fill = "#0a0a14"
                txt = "#2a2a4a"
            else:
                # red intensity for confusion
                intensity = v / vmax
                r_int = int(80 + intensity * 160)
                fill = f"rgb({r_int},60,80)"
                txt = "#f0f0f8"
            s.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}" stroke="#1e1e3a" stroke-width="0.5"/>')
            if v > 0:
                s.append(f'<text x="{x + cell/2}" y="{y + cell/2 - 2}" fill="{txt}" font-size="13" font-weight="700" font-family="JetBrains Mono,monospace" text-anchor="middle">{v}</text>')
                s.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 14}" fill="{txt}" font-size="9" font-family="JetBrains Mono,monospace" text-anchor="middle" opacity="0.8">{int(frac*100)}%</text>')

    # row labels (true)
    for i, sn in enumerate(SN):
        s.append(f'<text x="{pad_l - 12}" y="{pad_t + i*cell + cell/2 + 4}" fill="#d0d0e0" font-size="12" text-anchor="end" font-family="JetBrains Mono,monospace">{sn}</text>')
    # col labels (rotated)
    for j, sn in enumerate(SN):
        x = pad_l + j*cell + cell/2
        s.append(f'<g transform="translate({x},{pad_t - 16}) rotate(-45)"><text fill="#d0d0e0" font-size="12" text-anchor="start" font-family="JetBrains Mono,monospace">{sn}</text></g>')

    # axis hints
    s.append(f'<text x="{pad_l-12}" y="{pad_t + n*cell + 30}" fill="#606080" font-size="10" text-anchor="end" font-style="italic">true model →</text>')
    s.append(f'<g transform="translate({pad_l - 80}, {pad_t + n*cell/2}) rotate(-90)"><text fill="#606080" font-size="10" text-anchor="middle" font-style="italic">predicted →</text></g>')

    s.append('</svg>')
    return "\n".join(s)

(OUT / "model_confusion.svg").write_text(confusion_svg())
print(f"wrote {OUT}/model_confusion.svg")

# ─── 2. Accuracy ladder ───
def ladder_svg():
    L = C["ladder"]
    rows = [
        ("Random guess",              L["chance_8"],        "#606080", "1 in 8"),
        ("Gutenberg basis only",      L["gut_only_8"],      "#f59e0b", "78-D readout"),
        ("LLM basis only",            L["llm_only_8"],      "#a78bfa", "78-D readout"),
        ("Spectrometer (both)",       L["spectrometer_8"],  "#34d399", "156-D · ←"),
        ("Raw 384-D upper bound",     L["raw_384_8"],       "#22d3ee", "full embedding"),
    ]
    W, H = 920, 460
    pad_l, pad_b, pad_t = 230, 100, 100
    plot_w = W - pad_l - 100
    plot_h = H - pad_t - pad_b
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="40" fill="#f0f0f8" font-size="20" font-weight="700" text-anchor="middle">Can we tell the models apart?</text>')
    s.append(f'<text x="{W/2}" y="62" fill="#9090b0" font-size="12" text-anchor="middle">8-way model-identification accuracy (leave-one-out LDA) · 12 generations per model · 96 total</text>')
    s.append(f'<text x="{W/2}" y="80" fill="#34d399" font-size="14" font-weight="600" text-anchor="middle">Yes — the spectrometer gets 76% right. ~6× better than random.</text>')

    def sy(i): return pad_t + i * (plot_h / len(rows))
    bar_h = (plot_h / len(rows)) * 0.65

    for i, (label, val, color, note) in enumerate(rows):
        y = sy(i) + (plot_h / len(rows)) * 0.18
        s.append(f'<text x="{pad_l - 14}" y="{y + bar_h/2 + 4}" fill="#d0d0e0" font-size="13" text-anchor="end">{label}</text>')
        bw = (val / 1.0) * plot_w
        s.append(f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}" opacity="0.85" rx="3"/>')
        # text inside or outside bar
        text_x = pad_l + bw + 10
        s.append(f'<text x="{text_x}" y="{y + bar_h/2 + 5}" fill="#f0f0f8" font-size="15" font-weight="700" font-family="JetBrains Mono,monospace">{val*100:.1f}%</text>')
        s.append(f'<text x="{text_x + 75}" y="{y + bar_h/2 + 5}" fill="#606080" font-size="11">{note}</text>')

    # x-axis ticks
    yt = pad_t + plot_h + 12
    for tv in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = pad_l + tv * plot_w
        s.append(f'<line x1="{tx}" y1="{yt-4}" x2="{tx}" y2="{yt+4}" stroke="#606080"/>')
        s.append(f'<text x="{tx}" y="{yt+18}" fill="#606080" font-size="10" text-anchor="middle">{int(tv*100)}%</text>')
    # dashed chance line through all bars
    cx = pad_l + (1/8) * plot_w
    s.append(f'<line x1="{cx}" y1="{pad_t-2}" x2="{cx}" y2="{pad_t+plot_h+2}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.5"/>')
    s.append(f'<text x="{cx+5}" y="{pad_t+10}" fill="#f87171" font-size="10" opacity="0.7">chance</text>')

    s.append(f'<text x="{pad_l + plot_w/2}" y="{H-25}" fill="#9090b0" font-size="11" text-anchor="middle">accuracy on 8-way model identification</text>')
    s.append('</svg>')
    return "\n".join(s)

(OUT / "model_ladder.svg").write_text(ladder_svg())
print(f"wrote {OUT}/model_ladder.svg")

# ─── 3. Per-model recall bar chart ───
def per_model_svg():
    accs = C["spectrometer"]["per_model_recall"]
    W, H = 920, 460
    pad_l, pad_b, pad_t = 180, 100, 100
    plot_w = W - pad_l - 80
    plot_h = H - pad_t - pad_b

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="40" fill="#f0f0f8" font-size="20" font-weight="700" text-anchor="middle">Which models does it spot most reliably?</text>')
    s.append(f'<text x="{W/2}" y="62" fill="#9090b0" font-size="12" text-anchor="middle">Per-model recall — fraction of that model\'s generations correctly identified by the spectrometer</text>')
    s.append(f'<text x="{W/2}" y="82" fill="#34d399" font-size="13" text-anchor="middle">Anthropic models cleanest (75–92%). Reasoning-trained + open-source models confused with each other.</text>')

    sorted_order = sorted(order, key=lambda m: -accs[m])

    def sy(i): return pad_t + i * (plot_h / len(order))
    bar_h = (plot_h / len(order)) * 0.7
    for i, m in enumerate(sorted_order):
        a = accs[m]
        c = FAMILY_COLOR[FAMILY[m]]
        y = sy(i) + (plot_h / len(order)) * 0.15
        s.append(f'<text x="{pad_l - 14}" y="{y + bar_h/2 + 4}" fill="#d0d0e0" font-size="13" text-anchor="end" font-family="JetBrains Mono,monospace">{SHORT[m]}</text>')
        bw = (a / 1.0) * plot_w
        s.append(f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{c}" opacity="0.85" rx="3"/>')
        s.append(f'<text x="{pad_l + bw + 10}" y="{y + bar_h/2 + 5}" fill="#f0f0f8" font-size="14" font-weight="700" font-family="JetBrains Mono,monospace">{a*100:.0f}%</text>')

    # chance line
    cx = pad_l + (1/8) * plot_w
    s.append(f'<line x1="{cx}" y1="{pad_t-2}" x2="{cx}" y2="{pad_t+plot_h+2}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.5"/>')
    s.append(f'<text x="{cx+5}" y="{pad_t+10}" fill="#f87171" font-size="10" opacity="0.7">chance (12.5%)</text>')

    # x ticks
    yt = pad_t + plot_h + 12
    for tv in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = pad_l + tv * plot_w
        s.append(f'<line x1="{tx}" y1="{yt-4}" x2="{tx}" y2="{yt+4}" stroke="#606080"/>')
        s.append(f'<text x="{tx}" y="{yt+18}" fill="#606080" font-size="10" text-anchor="middle">{int(tv*100)}%</text>')

    s.append(f'<text x="{pad_l + plot_w/2}" y="{H-25}" fill="#9090b0" font-size="11" text-anchor="middle">per-model recall (of 12 generations)</text>')
    s.append('</svg>')
    return "\n".join(s)

(OUT / "model_per_model.svg").write_text(per_model_svg())
print(f"wrote {OUT}/model_per_model.svg")
print("DONE")
