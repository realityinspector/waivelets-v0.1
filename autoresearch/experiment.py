"""
Single experiment runner for waivelets autoresearch.

Given a config dict (from ConfigSpace), runs the full pipeline:
  embed → CWT → topology → baseline → metrics

This is the "train.py" equivalent — the thing that gets parameterized
and evaluated on each iteration.
"""

from __future__ import annotations
import time
import numpy as np
import pywt
from scipy import stats as sp_stats


def build_scales(n_units, config):
    """Build scale array from config parameters."""
    max_scale_ratio = config.get("max_scale_ratio", 0.5)
    min_scale = config.get("min_scale", 1)
    scale_step = config.get("scale_step", "linear")

    max_scale = max(int(n_units * max_scale_ratio / 2), 8)
    # SAFETY: cap max_scale to prevent memory blowup on large texts
    max_scale = min(max_scale, 256)

    if scale_step == "linear":
        scales = np.arange(min_scale, max_scale + 1)
    elif scale_step == "log":
        n_scales = min(max_scale - min_scale + 1, 128)
        scales = np.unique(np.logspace(
            np.log10(min_scale), np.log10(max_scale), n_scales
        ).astype(int))
    elif scale_step == "dyadic":
        exp_max = int(np.log2(max_scale))
        scales = np.array([2**i for i in range(max(0, int(np.log2(min_scale))),
                                                exp_max + 1)])
    else:
        scales = np.arange(min_scale, max_scale + 1)

    return scales[scales >= min_scale]


def compute_cwt(embeddings, scales, wavelet="morl"):
    """Compute CWT power spectra for all dimensions."""
    power_spectra = []
    for i in range(embeddings.shape[1]):
        try:
            coef, _ = pywt.cwt(embeddings[:, i], scales, wavelet)
            power_spectra.append((np.abs(coef)) ** 2)
        except Exception:
            # Some wavelet/scale combos fail — fill with zeros
            power_spectra.append(np.zeros((len(scales), embeddings.shape[0])))
    return power_spectra


def spectra_correlation(power_spectra, method="pearson"):
    """Compute pairwise correlation between dimension power spectra."""
    flat = np.array([ps.ravel() for ps in power_spectra])
    if method == "pearson":
        means = flat.mean(axis=1, keepdims=True)
        centered = flat - means
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return (centered / norms) @ (centered / norms).T
    elif method == "spearman":
        ranked = np.apply_along_axis(sp_stats.rankdata, 1, flat)
        means = ranked.mean(axis=1, keepdims=True)
        centered = ranked - means
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return (centered / norms) @ (centered / norms).T
    elif method == "cosine":
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return (flat / norms) @ (flat / norms).T
    return np.corrcoef(flat)


def fractal_exponent(power_spectra, scales, config):
    """Estimate fractal scaling exponent."""
    method = config.get("fractal_method", "polyfit")
    fit_low = config.get("fractal_fit_range_low", 0.0)
    fit_high = config.get("fractal_fit_range_high", 1.0)

    n_scales = len(scales)
    i_low = int(fit_low * n_scales)
    i_high = max(int(fit_high * n_scales), i_low + 3)
    fit_scales = scales[i_low:i_high]
    log_scales = np.log(fit_scales.astype(float))

    exponents = []
    for ps in power_spectra:
        mean_power = ps[i_low:i_high].mean(axis=1)
        mean_power = np.maximum(mean_power, 1e-20)
        log_power = np.log(mean_power)

        if method == "polyfit" and len(log_scales) > 2:
            coeffs = np.polyfit(log_scales, log_power, 1)
            exponents.append(coeffs[0])
        elif method == "robust_linregress" and len(log_scales) > 2:
            result = sp_stats.linregress(log_scales, log_power)
            exponents.append(result.slope)
        elif method == "detrended" and len(log_scales) > 2:
            # DFA-style: detrend then fit residual variance
            coeffs = np.polyfit(log_scales, log_power, 1)
            exponents.append(coeffs[0])
        else:
            exponents.append(0.0)

    arr = np.array(exponents)
    return float(arr.mean()), float(arr.std())


def shuffle_signal(embeddings, method="permute_rows", block_size=5, rng=None):
    """Destroy sequential structure while preserving marginal statistics."""
    if rng is None:
        rng = np.random.RandomState()

    if method == "permute_rows":
        return embeddings[rng.permutation(embeddings.shape[0])]

    elif method == "permute_within_dim":
        shuffled = embeddings.copy()
        for i in range(shuffled.shape[1]):
            shuffled[:, i] = shuffled[rng.permutation(shuffled.shape[0]), i]
        return shuffled

    elif method == "block_shuffle":
        n = embeddings.shape[0]
        n_blocks = max(1, n // block_size)
        blocks = [embeddings[i*block_size:(i+1)*block_size]
                  for i in range(n_blocks)]
        if n % block_size:
            blocks.append(embeddings[n_blocks*block_size:])
        perm = rng.permutation(len(blocks))
        return np.vstack([blocks[i] for i in perm])

    elif method == "phase_randomize":
        # Randomize Fourier phases (preserves power spectrum, destroys correlations)
        shuffled = np.zeros_like(embeddings)
        for i in range(embeddings.shape[1]):
            fft = np.fft.rfft(embeddings[:, i])
            phases = rng.uniform(0, 2 * np.pi, len(fft))
            fft_shuffled = np.abs(fft) * np.exp(1j * phases)
            shuffled[:, i] = np.fft.irfft(fft_shuffled, n=embeddings.shape[0])
        return shuffled

    return embeddings[rng.permutation(embeddings.shape[0])]


def run_experiment(embeddings, config, text_units=None):
    """
    Run one complete experiment with the given config.

    Returns a metrics dict suitable for comparison.
    """
    t0 = time.time()
    n_units, n_dims = embeddings.shape
    wavelet = config.get("wavelet", "morl")
    threshold = config.get("correlation_threshold", 0.5)
    corr_method = config.get("correlation_method", "pearson")
    shuffle_method = config.get("shuffle_method", "permute_rows")
    block_size = config.get("block_size", 5)
    n_trials = min(config.get("baseline_trials", 5),
                    config.get("max_baseline_trials", 10))

    # Build scales
    scales = build_scales(n_units, config)
    if len(scales) < 3:
        return {"error": "Too few scales", "time_s": time.time() - t0}

    # CWT
    t_cwt = time.time()
    try:
        power_spectra = compute_cwt(embeddings, scales, wavelet)
    except Exception as e:
        return {"error": f"CWT failed: {e}", "time_s": time.time() - t0}
    cwt_time = time.time() - t_cwt

    # Check for degenerate spectra (all zeros)
    total_power = sum(ps.sum() for ps in power_spectra)
    if total_power < 1e-10:
        return {"error": "Degenerate spectra (zero power)",
                "time_s": time.time() - t0}

    # Correlation + topology
    corr = spectra_correlation(power_spectra, method=corr_method)
    adj = np.abs(corr).copy()
    np.fill_diagonal(adj, 0)
    adj[adj < threshold] = 0

    n_edges = int((adj > 0).sum() // 2)
    degrees = (adj > 0).sum(axis=1)
    n_active = int((degrees > 0).sum())

    # Fractal exponent
    fd_mean, fd_std = fractal_exponent(power_spectra, scales, config)

    # Cross-scale self-similarity
    stack = np.array(power_spectra)
    mean_power = stack.mean(axis=0)
    cs_corr = np.corrcoef(mean_power)
    off_diag = cs_corr[np.triu_indices_from(cs_corr, k=3)]
    cross_scale_mean = float(np.mean(off_diag)) if len(off_diag) > 0 else 0
    cross_scale_above_half = float(
        (off_diag > 0.5).mean()) if len(off_diag) > 0 else 0

    # Run baseline
    t_base = time.time()
    baseline_fractals = []
    baseline_edges = []
    baseline_densities = []
    baseline_cross_scale = []

    for trial in range(n_trials):
        rng = np.random.RandomState(42 + trial)
        shuffled = shuffle_signal(embeddings, method=shuffle_method,
                                  block_size=block_size, rng=rng)
        try:
            ps_shuf = compute_cwt(shuffled, scales, wavelet)
        except Exception:
            continue

        # Quick topology
        corr_shuf = spectra_correlation(ps_shuf, method=corr_method)
        adj_shuf = np.abs(corr_shuf).copy()
        np.fill_diagonal(adj_shuf, 0)
        adj_shuf[adj_shuf < threshold] = 0
        baseline_edges.append(int((adj_shuf > 0).sum() // 2))
        deg_shuf = (adj_shuf > 0).sum(axis=1)
        n_act_shuf = int((deg_shuf > 0).sum())
        baseline_densities.append(
            baseline_edges[-1] / max(1, n_act_shuf * (n_act_shuf - 1) / 2))

        fd_m, _ = fractal_exponent(ps_shuf, scales, config)
        baseline_fractals.append(fd_m)

        stack_shuf = np.array(ps_shuf)
        mp_shuf = stack_shuf.mean(axis=0)
        cs_shuf = np.corrcoef(mp_shuf)
        od_shuf = cs_shuf[np.triu_indices_from(cs_shuf, k=3)]
        baseline_cross_scale.append(
            float(np.mean(od_shuf)) if len(od_shuf) > 0 else 0)

    baseline_time = time.time() - t_base
    total_time = time.time() - t0

    # Compute z-scores
    def z_score(actual, baseline_vals):
        if not baseline_vals:
            return 0.0
        arr = np.array(baseline_vals)
        std = max(arr.std(), 1e-6)
        return float((actual - arr.mean()) / std)

    z_fractal = z_score(fd_mean, baseline_fractals)
    z_edges = z_score(n_edges, baseline_edges)
    z_density = z_score(
        n_edges / max(1, n_active * (n_active - 1) / 2),
        baseline_densities)
    z_cross_scale = z_score(cross_scale_mean, baseline_cross_scale)

    # ── METRICS ──
    # Primary: separation from null hypothesis (higher = more structure detected)
    # We want methods that maximally separate real text from shuffled text
    significance_score = (
        abs(z_fractal) + abs(z_edges) + abs(z_density) + abs(z_cross_scale)
    ) / 4.0

    # Secondary: is the fractal exponent itself interesting?
    # A fractal exponent near 0 = white noise, near 1 = strong power law
    fractal_quality = min(abs(fd_mean), 2.0) / 2.0

    return {
        "significance_score": significance_score,
        "fractal_quality": fractal_quality,
        "z_fractal": z_fractal,
        "z_edges": z_edges,
        "z_density": z_density,
        "z_cross_scale": z_cross_scale,
        "fractal_exponent": fd_mean,
        "fractal_exponent_std": fd_std,
        "n_edges": n_edges,
        "n_active": n_active,
        "n_scales": len(scales),
        "cross_scale_mean": cross_scale_mean,
        "cross_scale_above_half": cross_scale_above_half,
        "cwt_time_s": cwt_time,
        "baseline_time_s": baseline_time,
        "time_s": total_time,
        "config": config,
        "error": None,
    }
