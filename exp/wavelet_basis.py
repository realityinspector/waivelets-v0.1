#!/usr/bin/env python3
"""Fit an LLM-corpus eigenbasis via the waivelets v0.1 wavelet methodology.

Method:
  1. Treat each generation's sentence embeddings as 384 time series of length n.
  2. CWT (Morlet) per dimension at log-spaced scales.
  3. Wavelet power |CWT|^2 averaged over scales gives a per-dim feature; full
     spectrogram flattened (n*S per dim) is concatenated across generations.
  4. Cross-dimension Pearson correlation across the pooled wavelet representation.
  5. Subtract a permute-within-dim null correlation matrix.
  6. Top-26 eigenvectors of the resulting symmetric matrix via np.linalg.eigh.
"""
import sys, time
import numpy as np
import pywt

RNG = np.random.default_rng(0)
N_SCALES = 16
N_EIG = 26
WAVELET = "morl"

def cwt_power_per_dim(emb, n_scales=N_SCALES):
    """emb: (n_sent, 384). Returns (384, n_sent * n_scales) wavelet power features."""
    n, D = emb.shape
    if n < 4:
        return None
    smax = max(2.0, n / 2.0)
    scales = np.logspace(np.log10(1.0), np.log10(smax), n_scales)
    out = np.empty((D, n * n_scales), dtype=np.float32)
    for d in range(D):
        coef, _ = pywt.cwt(emb[:, d], scales, WAVELET)   # (n_scales, n)
        power = (np.abs(coef) ** 2).astype(np.float32)   # (n_scales, n)
        out[d] = power.ravel()
    return out

def pooled_wavelet_matrix(all_embs, gen_ranges, shuffle=False):
    """Return (384, T_total) wavelet-power features pooled across generations.
    If shuffle=True, permute each dim's time series independently per generation."""
    pieces = []
    for (s, e, _, _) in gen_ranges:
        emb = all_embs[int(s):int(e)]
        if shuffle:
            emb = emb.copy()
            n = emb.shape[0]
            for d in range(emb.shape[1]):
                emb[:, d] = emb[RNG.permutation(n), d]
        feat = cwt_power_per_dim(emb)
        if feat is not None:
            pieces.append(feat)
    return np.concatenate(pieces, axis=1)   # (384, T_total)

def corr_matrix(X):
    """X: (D, T). Returns (D, D) Pearson correlation across columns."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    Xn = Xc / norm
    return Xn @ Xn.T

def main():
    t0 = time.time()
    z = np.load("/tmp/exp/all_sent_embs.npz", allow_pickle=True)
    all_embs = z["embs"].astype(np.float32)
    gen_ranges = z["ranges"]
    print(f"embs={all_embs.shape} gens={len(gen_ranges)}", file=sys.stderr)

    print("CWT pooling (real)...", file=sys.stderr)
    W = pooled_wavelet_matrix(all_embs, gen_ranges, shuffle=False)
    print(f"  W shape={W.shape}  in {time.time()-t0:.1f}s", file=sys.stderr)

    print("CWT pooling (shuffled null)...", file=sys.stderr)
    t1 = time.time()
    Wn = pooled_wavelet_matrix(all_embs, gen_ranges, shuffle=True)
    print(f"  Wn shape={Wn.shape}  in {time.time()-t1:.1f}s", file=sys.stderr)

    C = corr_matrix(W)
    Cn = corr_matrix(Wn)
    D = C - Cn
    # symmetrize numerically
    D = 0.5 * (D + D.T)
    print(f"  C diag mean={np.diag(C).mean():.3f}  off-diag |mean|={np.abs(C - np.diag(np.diag(C))).mean():.4f}", file=sys.stderr)
    print(f"  D off-diag |mean|={np.abs(D - np.diag(np.diag(D))).mean():.4f}", file=sys.stderr)

    w, V = np.linalg.eigh(D)
    # eigh returns ascending; take largest 26
    idx = np.argsort(w)[::-1][:N_EIG]
    wavelet_basis = V[:, idx].astype(np.float32)   # (384, 26)
    eigvals = w[idx].astype(np.float32)
    print(f"  top-6 eigvals: {eigvals[:6].round(3).tolist()}", file=sys.stderr)
    print(f"  wavelet_basis shape={wavelet_basis.shape}  is_real={np.isrealobj(wavelet_basis)}", file=sys.stderr)

    # Compare with PCA basis
    pca = np.load("/tmp/exp/llm_basis.npz")["basis"]   # (384, 26)
    M = np.abs(wavelet_basis.T @ pca)
    per_axis = M.max(axis=1)
    print(f"  per-axis best |cos| vs PCA: mean={per_axis.mean():.3f} median={np.median(per_axis):.3f} max={per_axis.max():.3f}", file=sys.stderr)

    np.savez("/tmp/exp/llm_wavelet_basis.npz",
             basis=wavelet_basis,
             eigvals=eigvals,
             corr_real_diag=np.diag(C).astype(np.float32),
             null_subtracted=True,
             n_scales=N_SCALES,
             wavelet=WAVELET)
    print(f"wrote /tmp/exp/llm_wavelet_basis.npz   total {time.time()-t0:.1f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
