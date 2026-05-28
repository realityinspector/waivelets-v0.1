#!/usr/bin/env python3
"""Re-run analysis on richer feature spaces: 26-D attractor coords, basin histogram, 384-D embedding mean."""
import json, pathlib, sys
import numpy as np
from collections import defaultdict, Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster

records = [json.loads(l) for l in pathlib.Path("/tmp/exp/coords26.jsonl").open()]
print(f"Loaded {len(records)} records", file=sys.stderr)

SHORT = {
    "anthropic/claude-sonnet-4": "sonnet-4",
    "anthropic/claude-sonnet-4.5": "sonnet-4.5",
    "anthropic/claude-haiku-4.5": "haiku-4.5",
    "openai/gpt-4-turbo": "gpt-4-turbo",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-5": "gpt-5",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
}
FAMILY = {
    "anthropic/claude-sonnet-4": "anthropic", "anthropic/claude-sonnet-4.5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic", "openai/gpt-4-turbo": "openai",
    "openai/gpt-4o": "openai", "openai/gpt-5": "openai",
    "google/gemini-2.5-pro": "google", "meta-llama/llama-3.3-70b-instruct": "meta",
}

models = sorted(set(r["model"] for r in records))
y = np.array([r["model"] for r in records])
y_fam = np.array([FAMILY[m] for m in y])

# Build feature matrices
def stack(key):
    return np.array([r[key] for r in records])

X_26 = stack("coord_mean")            # (N, 26)  attractor coords mean
X_26a = stack("coord_mean_abs")        # (N, 26)  |attractor coord| mean
X_basin = stack("basin_hist")          # (N, 26)  basin visit histogram
X_emb = stack("emb_mean")              # (N, 384) raw embedding centroid
X_combined = np.concatenate([X_26, X_26a, X_basin], axis=1)  # (N, 78)

# Combined + emb_mean reduced
print(f"Shapes: X_26={X_26.shape}  X_emb={X_emb.shape}  X_combined={X_combined.shape}", file=sys.stderr)

def standardize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

# PCA-then-LDA for high-dim features
def lda_loo(X, y, n_pca=None, label=""):
    Xs = standardize(X)
    if n_pca and Xs.shape[1] > n_pca:
        Xs = PCA(n_components=n_pca, random_state=0).fit_transform(Xs)
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, Xs, y, cv=LeaveOneOut())
    return scores.mean()

# Permutation null
def perm_null(X, y, n_pca=None, n_perm=100):
    rng = np.random.default_rng(7)
    accs = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        accs.append(lda_loo(X, yp, n_pca))
    return np.mean(accs), np.max(accs)

print("\n=== LDA accuracy (leave-one-out, 8-way model id) ===", file=sys.stderr)
print(f"{'feature space':30s} {'dim':>5s}  {'acc':>5s}  {'null_mean':>10s}  {'null_max':>9s}  {'p':>5s}", file=sys.stderr)
specs = [
    ("X_26 (attractor mean)",       X_26,       None),
    ("X_26a (|attractor| mean)",    X_26a,      None),
    ("X_basin (visit histogram)",   X_basin,    None),
    ("X_combined (26+26+26)",       X_combined, None),
    ("X_emb (raw centroid)",        X_emb,      None),
    ("X_emb → PCA(20)",             X_emb,      20),
    ("X_emb → PCA(50)",             X_emb,      50),
]
results = {}
for name, X, n_pca in specs:
    acc = lda_loo(X, y, n_pca)
    nm, nmax = perm_null(X, y, n_pca, n_perm=100)
    p = (nm >= acc)  # rough
    # better: compute fraction null >= acc
    rng = np.random.default_rng(7)
    cnt = 0
    null_accs = []
    for _ in range(100):
        yp = rng.permutation(y)
        a = lda_loo(X, yp, n_pca)
        null_accs.append(a)
        if a >= acc:
            cnt += 1
    p_val = cnt / 100
    print(f"{name:30s} {X.shape[1]:>5d}  {acc:.3f}  {np.mean(null_accs):>10.3f}  {np.max(null_accs):>9.3f}  {p_val:>5.2f}", file=sys.stderr)
    results[name] = {"dim": X.shape[1], "acc": float(acc), "null_mean": float(np.mean(null_accs)), "null_max": float(np.max(null_accs)), "p": float(p_val)}

# Family-level (4-way)
print("\n=== LDA accuracy (4-way family) ===", file=sys.stderr)
family_results = {}
for name, X, n_pca in specs:
    acc = lda_loo(X, y_fam, n_pca)
    rng = np.random.default_rng(7)
    cnt = 0
    null_accs = []
    for _ in range(100):
        yp = rng.permutation(y_fam)
        a = lda_loo(X, yp, n_pca)
        null_accs.append(a)
        if a >= acc:
            cnt += 1
    p_val = cnt / 100
    majority = max(Counter(y_fam).values()) / len(y_fam)
    print(f"{name:30s} {X.shape[1]:>5d}  acc={acc:.3f}  null_mean={np.mean(null_accs):.3f}  null_max={np.max(null_accs):.3f}  p={p_val:.2f}  (maj={majority:.3f})", file=sys.stderr)
    family_results[name] = {"dim": X.shape[1], "acc": float(acc), "null_mean": float(np.mean(null_accs)), "null_max": float(np.max(null_accs)), "p": float(p_val), "majority": float(majority)}

# Best feature: 384-D PCA(20) likely. Let's get LDA viz from that.
best_name = max(results, key=lambda k: results[k]["acc"])
print(f"\nBest model-id feature: {best_name}", file=sys.stderr)

# Project to 2-D LDA space for best
def get_X(name):
    for nn, X, p in specs:
        if nn == name:
            return X, p
    return None, None
X_best, npca_best = get_X(best_name)
Xs = standardize(X_best)
if npca_best:
    Xs = PCA(n_components=npca_best, random_state=0).fit_transform(Xs)
lda_proj = LinearDiscriminantAnalysis(n_components=min(2, len(set(y))-1))
X_lda = lda_proj.fit_transform(Xs, y)
# Also family-LDA proj
lda_proj_fam = LinearDiscriminantAnalysis(n_components=min(2, len(set(y_fam))-1))
X_lda_fam = lda_proj_fam.fit_transform(Xs, y_fam)

# Unsupervised clustering on best
Z = linkage(Xs, method="ward")
cluster_quality = {}
for k in range(2, 9):
    labels = fcluster(Z, t=k, criterion="maxclust")
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    pur = sum(Counter(y[labels == c]).most_common(1)[0][1] for c in set(labels)) / len(y)
    cluster_quality[k] = {"ari": float(ari), "nmi": float(nmi), "purity": float(pur)}
    print(f"  unsup k={k}  ARI={ari:.3f}  NMI={nmi:.3f}  purity={pur:.3f}", file=sys.stderr)

# Per-model centroid in best feature space (for the "yes there is signal" claim)
centroids = {}
for m in models:
    mask = (y == m)
    centroids[m] = X_lda[mask].mean(axis=0).tolist() if X_lda.ndim == 2 else [float(X_lda[mask].mean())]

# Save
out = {
    "n_records": len(records),
    "models": models,
    "short": SHORT,
    "family": FAMILY,
    "model_id_results": results,
    "family_results": family_results,
    "best_feature": best_name,
    "X_lda_coords": {m: X_lda[y == m].tolist() for m in models},
    "X_lda_fam_coords": {f: X_lda_fam[y_fam == f].tolist() for f in sorted(set(y_fam))},
    "centroids_lda": centroids,
    "cluster_quality": cluster_quality,
}
pathlib.Path("/tmp/exp/analysis2.json").write_text(json.dumps(out, indent=2))
print("\nWrote /tmp/exp/analysis2.json", file=sys.stderr)
