#!/usr/bin/env python3
"""Render the spectrometer result: bar chart comparing 5 feature spaces side-by-side
with permutation-null floors. Plus basis overlap diagram."""
import json, pathlib
import numpy as np

R = json.loads(pathlib.Path("/tmp/exp/multibasis_results.json").read_text())
OUT = pathlib.Path("/tmp/waivelets-v0.1/web/assets")

# ─── Bar chart: 5 feature spaces × {family, model} ───
def bar_svg():
    res = R["results"]
    fam_keys = [k for k in res if not k.endswith("[8-way]")]
    fam_keys = ["Gutenberg 78-D (existing baseline)", "LLM-PCA 78-D (new basis)",
                "Spectrometer 156-D (Gutenberg+LLM)", "Raw 384-D centroid", "Raw 384-D → PCA(50)"]
    mod_keys = [k + " [8-way]" for k in fam_keys]
    short = {
        "Gutenberg 78-D (existing baseline)": "Gutenberg basis",
        "LLM-PCA 78-D (new basis)":           "LLM-PCA basis",
        "Spectrometer 156-D (Gutenberg+LLM)": "Spectrometer (both)",
        "Raw 384-D centroid":                 "Raw 384-D centroid",
        "Raw 384-D → PCA(50)":                "Raw 384-D → PCA(50)",
    }
    colors = ["#f59e0b", "#a78bfa", "#34d399", "#22d3ee", "#818cf8"]

    W, H = 1000, 580
    pad_l, pad_b, pad_t = 200, 80, 110
    plot_w = W - pad_l - 60
    plot_h = H - pad_t - pad_b
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="32" fill="#f0f0f8" font-size="18" font-weight="700" text-anchor="middle">Spectrometer: two bases beat one — by a lot</text>')
    s.append(f'<text x="{W/2}" y="54" fill="#9090b0" font-size="11.5" text-anchor="middle">Leave-one-out LDA accuracy. Family = 4-way (Anthropic/OpenAI/Google/Meta). Model = 8-way (individual model).</text>')
    s.append(f'<text x="{W/2}" y="74" fill="#9090b0" font-size="11" text-anchor="middle">Dashed lines = mean of 100-permutation null. Solid bars = observed.</text>')

    # group bars: per feature space, two bars (family, model)
    n_groups = len(fam_keys)
    group_w = plot_w / n_groups
    bar_w = group_w / 3.6
    y0 = pad_t + plot_h

    def sy(v): return y0 - (v / 1.0) * plot_h

    # gridlines
    for tv in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gy = sy(tv)
        s.append(f'<line x1="{pad_l}" y1="{gy}" x2="{pad_l+plot_w}" y2="{gy}" stroke="#1e1e3a" stroke-dasharray="2 3"/>')
        s.append(f'<text x="{pad_l-10}" y="{gy+4}" fill="#606080" font-size="10" text-anchor="end">{tv:.1f}</text>')

    for i, k in enumerate(fam_keys):
        cx = pad_l + (i + 0.5) * group_w
        fam = res[k]
        mod = res[mod_keys[i]]
        col = colors[i]

        # family bar (left)
        bx1 = cx - bar_w - 4
        s.append(f'<rect x="{bx1}" y="{sy(fam["acc"])}" width="{bar_w}" height="{y0 - sy(fam["acc"])}" fill="{col}" opacity="0.85"/>')
        s.append(f'<line x1="{bx1-2}" y1="{sy(fam["null_mean"])}" x2="{bx1+bar_w+2}" y2="{sy(fam["null_mean"])}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="3 2"/>')
        s.append(f'<text x="{bx1 + bar_w/2}" y="{sy(fam["acc"]) - 6}" fill="#d0d0e0" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">{fam["acc"]:.2f}</text>')

        # model bar (right)
        bx2 = cx + 4
        s.append(f'<rect x="{bx2}" y="{sy(mod["acc"])}" width="{bar_w}" height="{y0 - sy(mod["acc"])}" fill="{col}" opacity="0.55"/>')
        s.append(f'<line x1="{bx2-2}" y1="{sy(mod["null_mean"])}" x2="{bx2+bar_w+2}" y2="{sy(mod["null_mean"])}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="3 2"/>')
        s.append(f'<text x="{bx2 + bar_w/2}" y="{sy(mod["acc"]) - 6}" fill="#9090b0" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">{mod["acc"]:.2f}</text>')

        # x label
        label = short[k]
        s.append(f'<g transform="translate({cx},{y0+18}) rotate(-15)"><text fill="#d0d0e0" font-size="11" text-anchor="middle">{label}</text></g>')
        s.append(f'<text x="{cx}" y="{y0+50}" fill="#606080" font-size="10" text-anchor="middle">{fam["dim"]}-D</text>')

    # legend
    lx = pad_l + 20
    ly = pad_t - 28
    s.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#22d3ee" opacity="0.85"/>')
    s.append(f'<text x="{lx+22}" y="{ly+11}" fill="#d0d0e0" font-size="11">4-way family</text>')
    s.append(f'<rect x="{lx+140}" y="{ly}" width="14" height="14" fill="#22d3ee" opacity="0.55"/>')
    s.append(f'<text x="{lx+162}" y="{ly+11}" fill="#d0d0e0" font-size="11">8-way model</text>')
    s.append(f'<line x1="{lx+280}" y1="{ly+7}" x2="{lx+300}" y2="{ly+7}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="3 2"/>')
    s.append(f'<text x="{lx+306}" y="{ly+11}" fill="#d0d0e0" font-size="11">permutation null mean</text>')

    # axis label
    s.append(f'<g transform="translate(28,{pad_t+plot_h/2}) rotate(-90)"><text fill="#9090b0" font-size="11" text-anchor="middle">LDA leave-one-out accuracy</text></g>')
    s.append('</svg>')
    return "\n".join(s)

(OUT / "model_spectrometer.svg").write_text(bar_svg())
print(f"wrote {OUT}/model_spectrometer.svg")

# ─── Basis overlap diagram ───
def overlap_svg():
    bo = R["basis_overlap"]
    W, H = 760, 280
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#06060c"/>')
    s.append(f'<text x="{W/2}" y="36" fill="#f0f0f8" font-size="17" font-weight="700" text-anchor="middle">The two bases are nearly orthogonal</text>')
    s.append(f'<text x="{W/2}" y="58" fill="#9090b0" font-size="11.5" text-anchor="middle">|cos| between each Gutenberg eigenvector and its best-matched LLM-PCA eigenvector</text>')

    cx_gut, cx_llm = 180, W - 180
    cy = 170
    r = 60
    s.append(f'<circle cx="{cx_gut}" cy="{cy}" r="{r}" fill="#f59e0b" opacity="0.18" stroke="#f59e0b" stroke-width="2"/>')
    s.append(f'<text x="{cx_gut}" y="{cy+5}" fill="#f0f0f8" font-size="13" font-weight="700" text-anchor="middle">Gutenberg</text>')
    s.append(f'<text x="{cx_gut}" y="{cy+22}" fill="#9090b0" font-size="11" text-anchor="middle">100 literary texts</text>')

    s.append(f'<circle cx="{cx_llm}" cy="{cy}" r="{r}" fill="#a78bfa" opacity="0.18" stroke="#a78bfa" stroke-width="2"/>')
    s.append(f'<text x="{cx_llm}" y="{cy+5}" fill="#f0f0f8" font-size="13" font-weight="700" text-anchor="middle">LLM-PCA</text>')
    s.append(f'<text x="{cx_llm}" y="{cy+22}" fill="#9090b0" font-size="11" text-anchor="middle">3084 LLM sentences</text>')

    # arrow between
    s.append(f'<line x1="{cx_gut+r+8}" y1="{cy}" x2="{cx_llm-r-8}" y2="{cy}" stroke="#34d399" stroke-width="2" stroke-dasharray="4 4"/>')
    s.append(f'<text x="{W/2}" y="{cy-12}" fill="#34d399" font-size="13" font-weight="600" text-anchor="middle">mean |cos| = {bo["per_gut_axis_best_abs_cos_mean"]:.3f}</text>')
    s.append(f'<text x="{W/2}" y="{cy+18}" fill="#9090b0" font-size="10.5" text-anchor="middle">range {bo["per_gut_axis_best_abs_cos_min"]:.3f}–{bo["per_gut_axis_best_abs_cos_max"]:.3f}  ·  basis vectors are essentially perpendicular</text>')

    s.append('</svg>')
    return "\n".join(s)

(OUT / "model_basis_overlap.svg").write_text(overlap_svg())
print(f"wrote {OUT}/model_basis_overlap.svg")
print("DONE")
