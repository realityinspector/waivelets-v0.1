#!/usr/bin/env python3
"""Render SVGs from analysis.json: distance matrix heatmap + PCA scatter + LDA scatter + bar charts."""
import json, pathlib
import numpy as np

OUT_DIR = pathlib.Path("/tmp/waivelets-v0.1/web/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)
A = json.loads(pathlib.Path("/tmp/exp/analysis.json").read_text())

SHORT = A["model_short"]
models = A["models"]
SN = [SHORT[m] for m in models]
COLORS = ["#f59e0b", "#a78bfa", "#22d3ee", "#34d399", "#f87171", "#818cf8", "#6ee7b7", "#a5b4fc"]
COLOR_OF = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}

# Family colors (override individual colors with family for some plots)
FAMILY = {
    "anthropic/claude-sonnet-4": "anthropic", "anthropic/claude-sonnet-4.5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic", "openai/gpt-4-turbo": "openai",
    "openai/gpt-4o": "openai", "openai/gpt-5": "openai",
    "google/gemini-2.5-pro": "google", "meta-llama/llama-3.3-70b-instruct": "meta",
}
FAMILY_COLOR = {"anthropic": "#f59e0b", "openai": "#22d3ee", "google": "#a78bfa", "meta": "#34d399"}

# ─── SVG 1: Pairwise centroid distance heatmap ───
def heatmap_svg():
    D = np.array(A["dist_matrix"])
    n = len(models)
    cell = 56
    pad_l, pad_t = 130, 130
    W, H = pad_l + n*cell + 80, pad_t + n*cell + 80
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="32" fill="#f0f0f8" font-size="18" font-weight="700" text-anchor="middle">Pairwise centroid distance (z-space)</text>')
    s.append(f'<text x="{W/2}" y="54" fill="#9090b0" font-size="11" text-anchor="middle">Closer = more similar 7-D fingerprint. Within-cluster spread averages <tspan fill="#f87171" font-weight="600">{A["mean_within"]:.2f}</tspan>; cross-model centroids average <tspan fill="#22d3ee" font-weight="600">{A["mean_between"]:.2f}</tspan>.</text>')

    vmax = D.max()
    for i in range(n):
        for j in range(n):
            x = pad_l + j*cell
            y = pad_t + i*cell
            v = D[i,j]
            # blue-to-red scale, low=red(close)=accent, high=blue(far)
            if i == j:
                fill = "#1e1e3a"
                txt = "#606080"
            else:
                t = v / vmax
                # interpolate amber (close) -> dark blue (far)
                r = int(245 - t * 200)
                g = int(158 - t * 130)
                b = int(11 + t * 100)
                fill = f"rgb({r},{g},{b})"
                txt = "#f0f0f8" if t < 0.7 else "#9090b0"
            s.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}"/>')
            s.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 4}" fill="{txt}" font-size="11" font-family="JetBrains Mono,monospace" text-anchor="middle">{v:.2f}</text>')
    # row labels
    for i, sn in enumerate(SN):
        fam_c = FAMILY_COLOR[FAMILY[models[i]]]
        s.append(f'<text x="{pad_l - 10}" y="{pad_t + i*cell + cell/2 + 4}" fill="#d0d0e0" font-size="11" text-anchor="end" font-family="JetBrains Mono,monospace">{sn}</text>')
        s.append(f'<rect x="{pad_l - 8}" y="{pad_t + i*cell + cell/2 - 2}" width="4" height="4" fill="{fam_c}"/>')
    # col labels (rotated)
    for j, sn in enumerate(SN):
        x = pad_l + j*cell + cell/2
        s.append(f'<g transform="translate({x},{pad_t - 14}) rotate(-45)"><text fill="#d0d0e0" font-size="11" text-anchor="start" font-family="JetBrains Mono,monospace">{sn}</text></g>')
    s.append('</svg>')
    return "\n".join(s)

(OUT_DIR / "model_distance_matrix.svg").write_text(heatmap_svg())
print(f"wrote {OUT_DIR}/model_distance_matrix.svg")

# ─── SVG 2: PCA scatter ───
def pca_svg():
    W, H = 920, 560
    pad = 50
    pca = A["pca_coords"]
    all_pts = [pt for m in models for pt in pca[m]]
    arr = np.array(all_pts)
    x_min, x_max = arr[:,0].min(), arr[:,0].max()
    y_min, y_max = arr[:,1].min(), arr[:,1].max()
    span_x, span_y = x_max - x_min, y_max - y_min
    x_min -= 0.1*span_x; x_max += 0.1*span_x
    y_min -= 0.1*span_y; y_max += 0.1*span_y

    def sx(v): return pad + 40 + (v - x_min) / (x_max - x_min) * (W - 2*pad - 200)
    def sy(v): return H - pad - (v - y_min) / (y_max - y_min) * (H - 2*pad - 40)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="28" fill="#f0f0f8" font-size="17" font-weight="700" text-anchor="middle">PCA of 7-D fingerprints — 95 generations, 8 models</text>')
    s.append(f'<text x="{W/2}" y="48" fill="#9090b0" font-size="11" text-anchor="middle">PC1 explains {A["pca_explained"][0]*100:.1f}% · PC2 explains {A["pca_explained"][1]*100:.1f}% · models overlap heavily; <tspan fill="#f87171" font-weight="600">no clean clusters by model identity</tspan></text>')
    # axes
    s.append(f'<line x1="{pad+40}" y1="{H-pad}" x2="{W-pad-200}" y2="{H-pad}" stroke="#1e1e3a"/>')
    s.append(f'<line x1="{pad+40}" y1="{pad+30}" x2="{pad+40}" y2="{H-pad}" stroke="#1e1e3a"/>')
    s.append(f'<text x="{(W-200)/2}" y="{H-15}" fill="#606080" font-size="10" text-anchor="middle">PC1 ({A["pca_explained"][0]*100:.1f}%)</text>')
    s.append(f'<g transform="translate(20,{(H+30)/2}) rotate(-90)"><text fill="#606080" font-size="10" text-anchor="middle">PC2 ({A["pca_explained"][1]*100:.1f}%)</text></g>')

    # points by family color, opacity by family
    for m in models:
        c = FAMILY_COLOR[FAMILY[m]]
        for pt in pca[m]:
            s.append(f'<circle cx="{sx(pt[0]):.1f}" cy="{sy(pt[1]):.1f}" r="5" fill="{c}" opacity="0.55"/>')

    # centroids as big rings, label each
    cen_x = {}
    cen_y = {}
    for m in models:
        pts = np.array(pca[m])
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        cen_x[m], cen_y[m] = sx(cx), sy(cy)
        c = FAMILY_COLOR[FAMILY[m]]
        s.append(f'<circle cx="{cen_x[m]:.1f}" cy="{cen_y[m]:.1f}" r="11" fill="none" stroke="{c}" stroke-width="2.5"/>')
        s.append(f'<circle cx="{cen_x[m]:.1f}" cy="{cen_y[m]:.1f}" r="3" fill="{c}"/>')

    # legend right
    lx = W - 180
    ly = 90
    s.append(f'<text x="{lx}" y="{ly-12}" fill="#9090b0" font-size="10" letter-spacing="1">MODELS</text>')
    for i, m in enumerate(models):
        c = FAMILY_COLOR[FAMILY[m]]
        s.append(f'<circle cx="{lx+8}" cy="{ly+i*22+4}" r="5" fill="{c}"/>')
        s.append(f'<text x="{lx+22}" y="{ly+i*22+8}" fill="#d0d0e0" font-size="11" font-family="JetBrains Mono,monospace">{SHORT[m]}</text>')
    s.append('</svg>')
    return "\n".join(s)

(OUT_DIR / "model_pca.svg").write_text(pca_svg())
print(f"wrote {OUT_DIR}/model_pca.svg")

# ─── SVG 3: separability bar chart (vs chance) ───
def sep_svg():
    W, H = 760, 360
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="32" fill="#f0f0f8" font-size="17" font-weight="700" text-anchor="middle">Separability of LLMs in the 7-D human-literary fingerprint space</text>')
    s.append(f'<text x="{W/2}" y="54" fill="#9090b0" font-size="11" text-anchor="middle">LDA leave-one-out accuracy vs random-permutation null (n=200)</text>')

    items = [
        ("8-way model id", A["lda_acc"], A["lda_chance"], "vs 1/8 chance"),
        ("4-way family",   A["lda_family_acc"] if A.get("lda_family_acc") else 0.0, 0.379, "vs majority-class"),
        ("permutation null mean", A["null_mean"], None, "null distribution mean"),
        ("permutation null max",  A["null_max"], None, "best null result"),
    ]
    bar_x = 200
    bar_w_max = W - bar_x - 80
    y0 = 100
    for i, (lab, v, ref, sub) in enumerate(items):
        y = y0 + i*60
        s.append(f'<text x="{bar_x-10}" y="{y+6}" fill="#d0d0e0" font-size="12" text-anchor="end">{lab}</text>')
        bw = v * bar_w_max
        color = "#34d399" if (ref and v > ref*1.2) else "#9090b0" if not ref else "#f87171"
        s.append(f'<rect x="{bar_x}" y="{y-12}" width="{bw:.0f}" height="22" fill="{color}" opacity="0.7"/>')
        s.append(f'<text x="{bar_x + bw + 10}" y="{y+5}" fill="#d0d0e0" font-size="13" font-family="JetBrains Mono,monospace">{v:.3f}</text>')
        s.append(f'<text x="{bar_x + bw + 60}" y="{y+5}" fill="#606080" font-size="10">{sub}</text>')
        if ref:
            rx = bar_x + ref * bar_w_max
            s.append(f'<line x1="{rx}" y1="{y-18}" x2="{rx}" y2="{y+18}" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="3 3"/>')

    s.append(f'<text x="{W/2}" y="{H-20}" fill="#a78bfa" font-size="10" text-anchor="middle">— dashed = chance / baseline · observed bars not exceeding null = not separable</text>')
    s.append('</svg>')
    return "\n".join(s)

(OUT_DIR / "model_separability.svg").write_text(sep_svg())
print(f"wrote {OUT_DIR}/model_separability.svg")
print("DONE")
