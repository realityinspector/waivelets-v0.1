"""
Topological analysis of wavelet power spectra.

Adapted from teeny-tiny-t9 IMT topology.py — instead of analysing
capability graphs of neural tasks, we build graphs from the correlation
structure of wavelet power spectra across embedding dimensions and
extract the same topological invariants.

If the waivelets hypothesis is correct (text has fractal-like self-similar
structure), we should see:
  - Higher genus (cycle rank) than shuffled baselines
  - Distinct spectral gap / Fiedler values for poetry vs prose vs technical
  - Cross-scale correlations that survive shuffling
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Graph Laplacian (from t9 topology.py)
# ---------------------------------------------------------------------------

def graph_laplacian(adj_matrix):
    """Compute normalized graph Laplacian."""
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix
    d_inv_sqrt = np.zeros_like(D)
    diag = np.diag(D)
    mask = diag > 0
    d_inv_sqrt[mask, mask] = 1.0 / np.sqrt(diag[mask])
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    return L, L_norm


# ---------------------------------------------------------------------------
# Build graphs from wavelet spectra
# ---------------------------------------------------------------------------

def spectra_correlation_matrix(power_spectra):
    """
    Compute pairwise correlation between the 384 dimensions' power spectra.
    Each spectrum is flattened (scales x units) into a vector.
    Returns 384x384 correlation matrix.
    """
    flat = np.array([ps.ravel() for ps in power_spectra])  # (384, scales*units)
    # Pearson correlation
    means = flat.mean(axis=1, keepdims=True)
    centered = flat - means
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = centered / norms
    corr = normed @ normed.T
    return corr


def correlation_to_adjacency(corr, threshold=0.5):
    """
    Threshold absolute correlation to build an adjacency matrix.
    Diagonal is zeroed. Edge weight = |correlation| above threshold.
    """
    adj = np.abs(corr).copy()
    np.fill_diagonal(adj, 0)
    adj[adj < threshold] = 0
    return adj


def adjacency_to_edges(adj):
    """Convert adjacency matrix to edge list + weights."""
    edges = []
    weights = {}
    idx = 0
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edges.append((i, j))
                weights[idx] = adj[i, j]
                idx += 1
    return edges, weights


# ---------------------------------------------------------------------------
# Topological feature extraction (adapted from t9 topology.py)
# ---------------------------------------------------------------------------

def extract_topology(adj, edges=None):
    """
    Extract topological features from an adjacency matrix.
    Returns a dict of scalar features.
    """
    n = adj.shape[0]
    if edges is None:
        edges, _ = adjacency_to_edges(adj)
    n_edges = len(edges)

    # Connected components (for cycle rank)
    visited = set()
    n_components = 0
    for start in range(n):
        if start in visited:
            continue
        if adj[start].sum() == 0:
            # isolated node — skip or count as component
            visited.add(start)
            n_components += 1
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            neighbors = np.where(adj[node] > 0)[0]
            for nb in neighbors:
                if nb not in visited:
                    stack.append(nb)
        n_components += 1

    # Cycle rank = E - V + C  (first Betti number)
    n_active = sum(1 for i in range(n) if adj[i].sum() > 0)
    cycle_rank = max(0, n_edges - n_active + n_components)

    # Euler characteristic (for thickened graph as surface)
    euler_char = 2 - 2 * cycle_rank

    # Degree distribution
    degrees = adj.astype(bool).astype(float).sum(axis=1)  # unweighted degree
    active_degrees = degrees[degrees > 0]

    # Graph Laplacian spectrum
    if n_active > 1:
        L, L_norm = graph_laplacian(adj)
        try:
            eigs_norm = np.sort(np.linalg.eigvalsh(L_norm))
        except Exception:
            eigs_norm = np.zeros(n)
        try:
            eigs_raw = np.sort(np.linalg.eigvalsh(L))
        except Exception:
            eigs_raw = np.zeros(n)
        spectral_gap = float(eigs_raw[1]) if len(eigs_raw) > 1 else 0.0
        fiedler = spectral_gap  # algebraic connectivity
    else:
        eigs_norm = np.zeros(n)
        spectral_gap = 0.0
        fiedler = 0.0

    return {
        "n_nodes": n,
        "n_active_nodes": n_active,
        "n_edges": n_edges,
        "n_components": n_components,
        "cycle_rank": cycle_rank,
        "euler_char": euler_char,
        "spectral_gap": spectral_gap,
        "fiedler": fiedler,
        "mean_degree": float(active_degrees.mean()) if len(active_degrees) > 0 else 0,
        "max_degree": float(active_degrees.max()) if len(active_degrees) > 0 else 0,
        "degree_variance": float(active_degrees.var()) if len(active_degrees) > 0 else 0,
        "density": n_edges / max(1, n_active * (n_active - 1) / 2),
        "eigenvalues": eigs_norm,
    }


# ---------------------------------------------------------------------------
# Cross-scale self-similarity
# ---------------------------------------------------------------------------

def cross_scale_correlation(power_spectra, scales):
    """
    Measure self-similarity: for each dimension, correlate the power
    spectrum at different scales. Average across dimensions.
    Returns a (n_scales x n_scales) cross-scale correlation matrix.
    """
    # Stack all spectra: shape (384, n_scales, n_units)
    stack = np.array(power_spectra)
    n_dims, n_scales, n_units = stack.shape

    # Average across dimensions to get mean power at each scale
    mean_power = stack.mean(axis=0)  # (n_scales, n_units)

    # Cross-scale correlation
    corr = np.corrcoef(mean_power)  # (n_scales, n_scales)
    return corr


def fractal_dimension_estimate(power_spectra, scales):
    """
    Estimate fractal dimension via power-law scaling of wavelet power.
    For each dimension, fit log(mean_power) vs log(scale).
    Returns mean and std of estimated scaling exponents.
    """
    log_scales = np.log(scales.astype(float))
    exponents = []
    for ps in power_spectra:
        mean_power_per_scale = ps.mean(axis=1)  # average over text positions
        mean_power_per_scale = np.maximum(mean_power_per_scale, 1e-20)
        log_power = np.log(mean_power_per_scale)
        # Linear fit: log(P) = beta * log(s) + c
        if len(log_scales) > 2:
            coeffs = np.polyfit(log_scales, log_power, 1)
            exponents.append(coeffs[0])
    exponents = np.array(exponents)
    return float(exponents.mean()), float(exponents.std())


# ---------------------------------------------------------------------------
# Shuffled baseline
# ---------------------------------------------------------------------------

def _cwt_one_dim(args):
    """Worker for parallel CWT computation."""
    import pywt
    signal, scales, wavelet = args
    coef, _ = pywt.cwt(signal, scales, wavelet)
    return (np.abs(coef)) ** 2


def _run_one_baseline_trial(args):
    """Worker for one shuffled baseline trial."""
    embeddings, scales, wavelet, threshold, seed = args
    rng = np.random.RandomState(seed)
    perm = rng.permutation(embeddings.shape[0])
    shuffled = embeddings[perm]

    # CWT per dimension (sequential within worker — already parallelized at trial level)
    import pywt
    power_spectra = []
    for i in range(shuffled.shape[1]):
        coef, _ = pywt.cwt(shuffled[:, i], scales, wavelet)
        power_spectra.append((np.abs(coef)) ** 2)

    corr = spectra_correlation_matrix(power_spectra)
    adj = correlation_to_adjacency(corr, threshold=threshold)
    features = extract_topology(adj)

    fd_mean, fd_std = fractal_dimension_estimate(power_spectra, scales)
    features["fractal_exponent_mean"] = fd_mean
    features["fractal_exponent_std"] = fd_std
    features.pop("eigenvalues", None)
    return features


def shuffled_baseline(embeddings, wavelet="morl", max_scale=None, n_trials=5,
                      threshold=0.5, parallel=True):
    """
    Generate topological features from shuffled text (destroy word order,
    preserve word identity). This is the null hypothesis: if patterns are
    just from vocabulary, not structure, shuffled text should look similar.

    Uses multiprocessing to run trials in parallel on large texts.
    """
    n_units, n_dims = embeddings.shape
    if max_scale is None:
        max_scale = max(min(n_units // 2, 256), 8)  # capped at 256 for safety
    max_scale = min(max_scale, 256)
    scales = np.arange(1, max_scale + 1)

    # SAFETY: avoid multiprocessing on macOS — fork() with large numpy arrays
    # causes memory duplication and can lock up the machine. Run sequentially.
    if True:
        all_features = []
        for trial in range(n_trials):
            feat = _run_one_baseline_trial(
                (embeddings, scales, wavelet, threshold, 42 + trial))
            all_features.append(feat)

    # Average across trials
    avg = {}
    for key in all_features[0]:
        vals = [f[key] for f in all_features]
        avg[key] = float(np.mean(vals))
        avg[f"{key}_std"] = float(np.std(vals))
    return avg


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_topology(power_spectra, scales, embeddings, threshold=0.5,
                     run_baseline=True):
    """
    Full topological analysis pipeline.

    Returns dict with:
      - 'features': topological features of actual text
      - 'fractal': fractal dimension estimates
      - 'cross_scale_corr': cross-scale correlation matrix
      - 'correlation_matrix': dimension correlation matrix
      - 'adjacency': thresholded adjacency matrix
      - 'baseline': averaged features from shuffled text (if run_baseline)
      - 'comparison': dict comparing actual vs baseline
    """
    # Build correlation graph
    corr = spectra_correlation_matrix(power_spectra)
    adj = correlation_to_adjacency(corr, threshold=threshold)
    features = extract_topology(adj)

    # Cross-scale correlation
    cs_corr = cross_scale_correlation(power_spectra, scales)

    # Fractal dimension
    fd_mean, fd_std = fractal_dimension_estimate(power_spectra, scales)
    features["fractal_exponent_mean"] = fd_mean
    features["fractal_exponent_std"] = fd_std

    result = {
        "features": features,
        "cross_scale_corr": cs_corr,
        "correlation_matrix": corr,
        "adjacency": adj,
    }

    if run_baseline:
        baseline = shuffled_baseline(embeddings, max_scale=len(scales))
        comparison = {}
        for key in ["cycle_rank", "spectral_gap", "fiedler", "density",
                     "mean_degree", "fractal_exponent_mean"]:
            actual = features.get(key, 0)
            base_val = baseline.get(key, 0)
            base_std = baseline.get(f"{key}_std", 0.001)
            z_score = (actual - base_val) / max(base_std, 0.001)
            comparison[key] = {
                "actual": actual,
                "baseline_mean": base_val,
                "baseline_std": base_std,
                "z_score": z_score,
            }
        result["baseline"] = baseline
        result["comparison"] = comparison

    return result
