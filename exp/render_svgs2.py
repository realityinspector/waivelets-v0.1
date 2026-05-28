#!/usr/bin/env python3
"""Render LDA scatter (family-level) + feature-space bar chart."""
import json, pathlib
import numpy as np

A = json.loads(pathlib.Path("/tmp/exp/analysis2.json").read_text())
OUT_DIR = pathlib.Path("/tmp/waivelets-v0.1/web/assets")

FAMILY_COLOR = {"anthropic": "#f59e0b", "openai": "#22d3ee", "google": "#a78bfa", "meta": "#34d399"}
FAMILY_LABEL = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "meta": "Meta"}

# ─── SVG 1: LDA family scatter (2-D) ───
def lda_fam_svg():
    coords = A["X_lda_fam_coords"]
    families = sorted(coords.keys())
    W, H = 900, 560
    pad = 70

    all_pts = [p for f in families for p in coords[f]]
    arr = np.array(all_pts)
    if arr.shape[1] < 2:
        # pad with zeros
        arr = np.column_stack([arr, np.zeros(len(arr))])
        for f in families:
            coords[f] = [[p[0], 0.0] for p in coords[f]]

    x_min, x_max = arr[:,0].min(), arr[:,0].max()
    y_min, y_max = arr[:,1].min(), arr[:,1].max()
    span_x = max(x_max - x_min, 1e-3)
    span_y = max(y_max - y_min, 1e-3)
    x_min -= 0.1*span_x; x_max += 0.1*span_x
    y_min -= 0.15*span_y; y_max += 0.15*span_y

    def sx(v): return pad + 30 + (v - x_min) / (x_max - x_min) * (W - 2*pad - 220)
    def sy(v): return H - pad - (v - y_min) / (y_max - y_min) * (H - 2*pad - 40)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="32" fill="#f0f0f8" font-size="18" font-weight="700" text-anchor="middle">Model families separate in raw MiniLM embedding space</text>')
    s.append(f'<text x="{W/2}" y="54" fill="#9090b0" font-size="11.5" text-anchor="middle">LDA projection of 384-D embedding centroids (PCA→50, then 2-D LDA). 4-way family LOO accuracy <tspan fill="#34d399" font-weight="600">57.3%</tspan> vs 37.5% majority baseline · p &lt; 0.01.</text>')

    # axes
    s.append(f'<line x1="{pad+30}" y1="{H-pad}" x2="{W-pad-220}" y2="{H-pad}" stroke="#1e1e3a"/>')
    s.append(f'<line x1="{pad+30}" y1="{pad+30}" x2="{pad+30}" y2="{H-pad}" stroke="#1e1e3a"/>')
    s.append(f'<text x="{(W-220)/2}" y="{H-20}" fill="#606080" font-size="10" text-anchor="middle">LDA-1</text>')
    s.append(f'<g transform="translate(20,{(H+30)/2}) rotate(-90)"><text fill="#606080" font-size="10" text-anchor="middle">LDA-2</text></g>')

    # points
    for f in families:
        c = FAMILY_COLOR[f]
        for pt in coords[f]:
            s.append(f'<circle cx="{sx(pt[0]):.1f}" cy="{sy(pt[1]):.1f}" r="6" fill="{c}" opacity="0.55"/>')

    # centroids
    for f in families:
        pts = np.array(coords[f])
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        c = FAMILY_COLOR[f]
        s.append(f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="14" fill="none" stroke="{c}" stroke-width="3"/>')
        s.append(f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="4" fill="{c}"/>')

    # legend
    lx = W - 200
    ly = 100
    s.append(f'<text x="{lx}" y="{ly-14}" fill="#9090b0" font-size="10" letter-spacing="1">FAMILY</text>')
    for i, f in enumerate(families):
        c = FAMILY_COLOR[f]
        n = len(coords[f])
        s.append(f'<circle cx="{lx+8}" cy="{ly+i*26+4}" r="6" fill="{c}"/>')
        s.append(f'<text x="{lx+22}" y="{ly+i*26+8}" fill="#d0d0e0" font-size="12">{FAMILY_LABEL[f]} <tspan fill="#606080">(n={n})</tspan></text>')

    s.append('</svg>')
    return "\n".join(s)

(OUT_DIR / "model_lda_family.svg").write_text(lda_fam_svg())
print(f"wrote {OUT_DIR}/model_lda_family.svg")

# ─── SVG 2: feature-space comparison bar chart ───
def feature_compare_svg():
    W, H = 920, 480
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="32" fill="#f0f0f8" font-size="18" font-weight="700" text-anchor="middle">The signal lives in the raw embedding, not the eigenbasis projection</text>')
    s.append(f'<text x="{W/2}" y="54" fill="#9090b0" font-size="11.5" text-anchor="middle">LDA leave-one-out accuracy at the 4-way family level · dashed line = majority-class baseline (37.5%)</text>')

    fam = A["family_results"]
    order = ["X_26 (attractor mean)", "X_26a (|attractor| mean)", "X_basin (visit histogram)",
             "X_combined (26+26+26)", "X_emb → PCA(20)", "X_emb → PCA(50)", "X_emb (raw centroid)"]
    pretty = {
        "X_26 (attractor mean)": "26-D attractor coords (mean)",
        "X_26a (|attractor| mean)": "26-D attractor coords (|mean|)",
        "X_basin (visit histogram)": "26-D basin visit histogram",
        "X_combined (26+26+26)": "78-D combined (26+26+26)",
        "X_emb → PCA(20)": "384-D embedding → PCA(20)",
        "X_emb → PCA(50)": "384-D embedding → PCA(50)",
        "X_emb (raw centroid)": "384-D raw embedding centroid",
    }
    dim_max = max(fam[k]["dim"] for k in order)
    label_x = 270
    bar_x = 280
    bar_w_max = W - bar_x - 160
    y0 = 110

    # majority baseline
    maj = list(fam.values())[0]["majority"]
    mx = bar_x + maj * bar_w_max / 1.0
    # Actually scale bars 0..0.6 to make differences visible
    scale = 0.7
    mx = bar_x + (maj / scale) * bar_w_max
    s.append(f'<line x1="{mx}" y1="{y0-20}" x2="{mx}" y2="{y0+len(order)*42+10}" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4 4"/>')
    s.append(f'<text x="{mx+4}" y="{y0-25}" fill="#a78bfa" font-size="10">majority baseline 37.5%</text>')

    for i, k in enumerate(order):
        d = fam[k]
        y = y0 + i*42
        s.append(f'<text x="{label_x}" y="{y+5}" fill="#d0d0e0" font-size="11" text-anchor="end">{pretty[k]}</text>')
        # dim chip
        s.append(f'<text x="{label_x+8}" y="{y+5}" fill="#606080" font-size="9" font-family="JetBrains Mono,monospace">{d["dim"]}D</text>')
        bw = (d["acc"] / scale) * bar_w_max
        # color by significance
        if d["p"] <= 0.01:
            color = "#34d399"
        elif d["p"] <= 0.05:
            color = "#a78bfa"
        else:
            color = "#606080"
        s.append(f'<rect x="{bar_x}" y="{y-14}" width="{bw:.0f}" height="24" fill="{color}" opacity="0.75"/>')
        s.append(f'<text x="{bar_x + bw + 8}" y="{y+5}" fill="#d0d0e0" font-size="12" font-family="JetBrains Mono,monospace">{d["acc"]:.3f}</text>')
        s.append(f'<text x="{bar_x + bw + 65}" y="{y+5}" fill="#606080" font-size="10">p={d["p"]:.2f}</text>')

    # x-axis ticks
    yt = y0 + len(order)*42 + 18
    for tv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        tx = bar_x + (tv / scale) * bar_w_max
        s.append(f'<line x1="{tx}" y1="{yt-4}" x2="{tx}" y2="{yt+4}" stroke="#606080"/>')
        s.append(f'<text x="{tx}" y="{yt+18}" fill="#606080" font-size="10" text-anchor="middle">{tv:.1f}</text>')
    s.append(f'<text x="{bar_x + bar_w_max/2}" y="{yt+38}" fill="#9090b0" font-size="10" text-anchor="middle">LDA family-level accuracy (leave-one-out)</text>')

    # legend for colors
    leg_y = 80
    s.append(f'<rect x="{W-130}" y="{leg_y}" width="10" height="10" fill="#34d399"/>')
    s.append(f'<text x="{W-114}" y="{leg_y+9}" fill="#d0d0e0" font-size="10">p ≤ 0.01</text>')
    s.append(f'<rect x="{W-130}" y="{leg_y+18}" width="10" height="10" fill="#606080"/>')
    s.append(f'<text x="{W-114}" y="{leg_y+27}" fill="#d0d0e0" font-size="10">not significant</text>')

    s.append('</svg>')
    return "\n".join(s)

(OUT_DIR / "model_feature_compare.svg").write_text(feature_compare_svg())
print(f"wrote {OUT_DIR}/model_feature_compare.svg")
print("DONE")
