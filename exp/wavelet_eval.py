#!/usr/bin/env python3
"""Evaluate the LLM-wavelet basis against PCA baseline via leave-one-out LDA.

Conditions:
  - Gutenberg 78-D
  - LLM-PCA 78-D
  - LLM-wavelet 78-D (NEW)
  - Spectrometer Gutenberg+LLM-PCA 156-D
  - Spectrometer Gutenberg+LLM-wavelet 156-D (NEW)
  - Triple-spectrometer 234-D (NEW)
For 4-way family and 8-way model labels. 100-permutation null per condition.
"""
import json, sys, pathlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score

z = np.load("/tmp/exp/all_sent_embs.npz", allow_pickle=True)
all_embs = z["embs"].astype(np.float32)
gen_ranges = z["ranges"]

gut_basis = np.load("/tmp/waivelets-v0.1/basis.npz")["eigvecs"]          # (384, 26)
pca_basis = np.load("/tmp/exp/llm_basis.npz")["basis"]                   # (384, 26)
wav_basis = np.load("/tmp/exp/llm_wavelet_basis.npz")["basis"]           # (384, 26)
print(f"gut={gut_basis.shape} pca={pca_basis.shape} wav={wav_basis.shape}", file=sys.stderr)

# basis-overlap metrics
def overlap(A, B):
    M = np.abs(A.T @ B)
    pa = M.max(axis=1)
    return {"mean": float(pa.mean()), "median": float(np.median(pa)),
            "min": float(pa.min()), "max": float(pa.max())}

ov_wav_pca = overlap(wav_basis, pca_basis)
ov_wav_gut = overlap(wav_basis, gut_basis)
ov_pca_gut = overlap(pca_basis, gut_basis)
print(f"wavelet vs PCA: {ov_wav_pca}", file=sys.stderr)
print(f"wavelet vs Gutenberg: {ov_wav_gut}", file=sys.stderr)
print(f"PCA vs Gutenberg: {ov_pca_gut}", file=sys.stderr)

def feat_basis(basis, all_embs, gen_ranges):
    feats = []
    for (s, e, _, _) in gen_ranges:
        emb = all_embs[int(s):int(e)]
        proj = emb @ basis
        mean = proj.mean(axis=0)
        amean = np.abs(proj).mean(axis=0)
        basin = np.argmax(np.abs(proj), axis=1)
        hist = np.bincount(basin, minlength=basis.shape[1]).astype(float)
        s_ = hist.sum()
        hist = hist / s_ if s_ > 0 else hist
        feats.append(np.concatenate([mean, amean, hist]))
    return np.array(feats)

X_gut = feat_basis(gut_basis, all_embs, gen_ranges)
X_pca = feat_basis(pca_basis, all_embs, gen_ranges)
X_wav = feat_basis(wav_basis, all_embs, gen_ranges)
X_spec_pca = np.concatenate([X_gut, X_pca], axis=1)
X_spec_wav = np.concatenate([X_gut, X_wav], axis=1)
X_triple = np.concatenate([X_gut, X_pca, X_wav], axis=1)

y = np.array([m for _,_,m,_ in gen_ranges])
FAMILY = {
    "anthropic/claude-sonnet-4": "anthropic", "anthropic/claude-sonnet-4.5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic", "openai/gpt-4-turbo": "openai",
    "openai/gpt-4o": "openai", "openai/gpt-5": "openai",
    "google/gemini-2.5-pro": "google", "meta-llama/llama-3.3-70b-instruct": "meta",
}
y_fam = np.array([FAMILY[m] for m in y])

def standardize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

def lda_loo(X, y):
    Xs = standardize(X)
    return cross_val_score(LinearDiscriminantAnalysis(), Xs, y, cv=LeaveOneOut()).mean()

def perm_null(X, y, n_perm=100):
    rng = np.random.default_rng(7)
    accs = [lda_loo(X, rng.permutation(y)) for _ in range(n_perm)]
    return float(np.mean(accs)), float(np.max(accs))

specs = [
    ("Gutenberg 78-D",                 X_gut),
    ("LLM-PCA 78-D",                   X_pca),
    ("LLM-wavelet 78-D",               X_wav),
    ("Spectrometer Gut+PCA 156-D",     X_spec_pca),
    ("Spectrometer Gut+Wavelet 156-D", X_spec_wav),
    ("Triple-spectrometer 234-D",      X_triple),
]

results = {"family_4way": {}, "model_8way": {}}
for label, ylab in [("family_4way", y_fam), ("model_8way", y)]:
    print(f"\n=== {label} ===", file=sys.stderr)
    print(f"{'condition':36s} {'dim':>5s}  {'acc':>5s}  {'null_mean':>10s}  {'null_max':>9s}  {'p':>6s}", file=sys.stderr)
    for name, X in specs:
        acc = lda_loo(X, ylab)
        nm, nmax = perm_null(X, ylab, n_perm=100)
        rng = np.random.default_rng(7)
        ge = sum(1 for _ in range(100) if lda_loo(X, rng.permutation(ylab)) >= acc)
        p = (ge + 1) / 101
        results[label][name] = {"dim": X.shape[1], "acc": float(acc),
                                "null_mean": nm, "null_max": nmax, "p": float(p)}
        print(f"{name:36s} {X.shape[1]:>5d}  {acc:.3f}  {nm:>10.3f}  {nmax:>9.3f}  {p:>6.3f}", file=sys.stderr)

results["basis_overlap"] = {
    "wavelet_vs_pca": ov_wav_pca,
    "wavelet_vs_gutenberg": ov_wav_gut,
    "pca_vs_gutenberg": ov_pca_gut,
}
pathlib.Path("/tmp/exp/wavelet_results.json").write_text(json.dumps(results, indent=2))
print("\nwrote /tmp/exp/wavelet_results.json", file=sys.stderr)
