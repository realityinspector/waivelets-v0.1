"""Core wavelet analysis engine for text embeddings."""

import json
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text_units: list[str]) -> np.ndarray:
    return get_model().encode(text_units)


def compute_power_spectra(embeddings: np.ndarray, wavelet="morl", max_scale=None):
    n_units = embeddings.shape[0]
    if max_scale is None:
        max_scale = max(min(n_units // 2, 512), 8)
    scales = np.arange(1, max_scale + 1)
    power_spectra = []
    for i in range(embeddings.shape[1]):
        coef, _ = pywt.cwt(embeddings[:, i], scales, wavelet)
        power_spectra.append((abs(coef)) ** 2)
    return power_spectra, scales


def rank_dimensions_by_power(power_spectra):
    ranked = [(i, float(np.mean(ps)), float(np.std(ps)), float(np.max(ps)))
              for i, ps in enumerate(power_spectra)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def load_text_catalog(catalog_path="texts/catalog.json"):
    with open(catalog_path) as f:
        return json.load(f)["texts"]


def load_text_units(text_id, catalog):
    entry = catalog[text_id]
    with open(entry["source"]) as f:
        data = json.load(f)
    for k in entry["source_key"].split("."):
        data = data[k]
    return data


def load_precomputed_embeddings(path):
    with open(path) as f:
        data = json.load(f)
    for k in ("embeddings", "embedding", "embed"):
        if k in data:
            return np.array(data[k])
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return np.array(v)
    raise ValueError(f"Cannot find embeddings array in {path}")


# ---------------------------------------------------------------------------
# Plotting — separate, focused figures for a scrolling research layout
# ---------------------------------------------------------------------------

def make_heatmap(power_spectrum_2d, scales, n_units, dim_index, power_max,
                 unit_label="unit"):
    """Full-width power spectrum heatmap."""
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        power_spectrum_2d, cmap="coolwarm", vmin=0, vmax=power_max,
        aspect="auto", origin="lower",
        extent=[0, n_units, scales[0], scales[-1]]
    )
    ax.set_xlabel(f"{unit_label.title()} Index", fontsize=10)
    ax.set_ylabel("Period (scale)", fontsize=10)
    ax.set_title(f"Wavelet Power Spectrum — Dimension {dim_index + 1}",
                 fontsize=12, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("Power", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def make_magnitude(embedding_column, dim_index, n_units, unit_label="unit"):
    """Embedding magnitude trace for one dimension."""
    fig, ax = plt.subplots(figsize=(14, 2.5))
    x = np.arange(n_units)
    ax.fill_between(x, embedding_column, alpha=0.25, color="#1f77b4")
    ax.plot(x, embedding_column, linewidth=1, color="#1f77b4",
            marker="o", markersize=2.5)
    ax.axhline(0, color="#999", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"{unit_label.title()} Index", fontsize=10)
    ax.set_ylabel("Magnitude", fontsize=10)
    ax.set_title(f"Embedding Values — Dimension {dim_index + 1}",
                 fontsize=11, pad=6)
    ax.set_xlim(0, n_units - 1)
    fig.tight_layout()
    return fig


def make_overview_heatmap(power_spectra, n_top=30):
    """Mean power per dimension (top N), as a quick overview bar chart."""
    ranked = rank_dimensions_by_power(power_spectra)
    top = ranked[:n_top]
    dims = [f"D{r[0]+1}" for r in top]
    powers = [r[1] for r in top]

    fig, ax = plt.subplots(figsize=(14, 3))
    bars = ax.barh(range(len(dims)), powers, color="#4c78a8", edgecolor="none")
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Wavelet Power", fontsize=10)
    ax.set_title(f"Top {n_top} Most Active Embedding Dimensions", fontsize=11, pad=8)

    for bar, (_, mean_p, std_p, max_p) in zip(bars, top):
        ax.text(bar.get_width() + max(powers) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"  max={max_p:.4f}", va="center", fontsize=7, color="#666")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Topology visualization
# ---------------------------------------------------------------------------

def make_correlation_matrix(corr_matrix):
    """Heatmap of dimension-dimension correlation."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xlabel("Embedding Dimension", fontsize=10)
    ax.set_ylabel("Embedding Dimension", fontsize=10)
    ax.set_title("Dimension Correlation (from Wavelet Spectra)", fontsize=11, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    return fig


def make_cross_scale_corr(cs_corr, scales):
    """Cross-scale correlation heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cs_corr, cmap="magma", vmin=0, vmax=1, aspect="equal",
                   extent=[scales[0], scales[-1], scales[-1], scales[0]])
    ax.set_xlabel("Scale", fontsize=10)
    ax.set_ylabel("Scale", fontsize=10)
    ax.set_title("Cross-Scale Self-Similarity", fontsize=11, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    return fig


def make_comparison_chart(comparison):
    """Bar chart comparing actual vs shuffled baseline for key features."""
    keys = list(comparison.keys())
    actual = [comparison[k]["actual"] for k in keys]
    baseline = [comparison[k]["baseline_mean"] for k in keys]
    baseline_err = [comparison[k]["baseline_std"] for k in keys]
    z_scores = [comparison[k]["z_score"] for k in keys]

    labels = [k.replace("_", " ").title() for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4),
                                    gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(keys))
    w = 0.35
    ax1.bar(x - w/2, actual, w, label="Actual text", color="#4c78a8")
    ax1.bar(x + w/2, baseline, w, yerr=baseline_err, label="Shuffled baseline",
            color="#ccc", capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax1.set_title("Actual vs Shuffled Baseline", fontsize=11, pad=8)
    ax1.legend(fontsize=8)

    colors = ["#c44e52" if z < -2 else "#4c78a8" if z > 2 else "#999"
              for z in z_scores]
    ax2.barh(x, z_scores, color=colors)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.axvline(0, color="#333", linewidth=0.5)
    ax2.axvline(2, color="#4c78a8", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.axvline(-2, color="#c44e52", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Z-score (vs shuffled)", fontsize=9)
    ax2.set_title("Statistical Significance", fontsize=11, pad=8)

    fig.tight_layout()
    return fig


def make_adjacency_graph(adj, threshold_edges=500):
    """Visualize the adjacency structure using a circular layout."""
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                edges.append((i, j, adj[i, j]))

    if len(edges) > threshold_edges:
        edges.sort(key=lambda e: e[2], reverse=True)
        edges = edges[:threshold_edges]

    fig, ax = plt.subplots(figsize=(8, 8))

    if not edges:
        ax.text(0.5, 0.5, "No edges above threshold", ha="center", va="center",
                fontsize=12, color="#999", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)

    max_w = max(e[2] for e in edges)
    for i, j, w in edges:
        alpha = 0.05 + 0.3 * (w / max_w)
        ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
                color="#4c78a8", alpha=alpha, linewidth=0.3)

    degrees = adj.astype(bool).sum(axis=1)
    sizes = 2 + 15 * (degrees / max(degrees.max(), 1))
    ax.scatter(pos_x, pos_y, s=sizes, c=degrees, cmap="viridis",
               edgecolors="white", linewidth=0.3, zorder=5)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Dimension Correlation Graph ({len(edges)} edges)", fontsize=11, pad=8)
    fig.tight_layout()
    return fig
