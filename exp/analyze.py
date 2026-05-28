#!/usr/bin/env python3
"""Analyze model fingerprints: centroids, separability, unsupervised clustering, LDA."""
import json, pathlib, sys
import numpy as np
from collections import defaultdict, Counter

RESULTS = pathlib.Path("/tmp/exp/results.jsonl")
OUT = pathlib.Path("/tmp/exp/analysis.json")

# Feature order (must be consistent)
FEATS = ["smoothness_mean", "smoothness_std", "exposition", "interiority",
         "formal_structure", "scene_narration", "basin_entropy"]

records = [json.loads(l) for l in RESULTS.open()]
print(f"Loaded {len(records)} records", file=sys.stderr)

# Group by model
by_model = defaultdict(list)
for r in records:
    fp = r["fingerprint"]
    vec = np.array([fp[k] for k in FEATS], dtype=float)
    by_model[r["model"]].append({"vec": vec, "mode": r["mode"], "n_sent": r["n_sentences"], "prompt_idx": r["prompt_idx"]})

# Short model names for display
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

models = sorted(by_model.keys())
print(f"Models: {len(models)}", file=sys.stderr)
for m in models:
    print(f"  {SHORT[m]:18s} n={len(by_model[m])}", file=sys.stderr)

# Build X, y
X_rows, y_rows, prompts = [], [], []
for m in models:
    for r in by_model[m]:
        X_rows.append(r["vec"])
        y_rows.append(m)
        prompts.append(r["prompt_idx"])
X = np.array(X_rows)              # (N, 7)
y = np.array(y_rows)
N, D = X.shape
print(f"X shape: {X.shape}", file=sys.stderr)

# Standardize across all samples (z-score per feature)
Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

# ─── 1. Per-model centroids (in standardized space) ───
centroids = {}
spreads = {}
for m in models:
    mask = (y == m)
    centroids[m] = Xz[mask].mean(axis=0)
    spreads[m] = Xz[mask].std(axis=0)

# Pairwise centroid distance matrix
dist_matrix = np.zeros((len(models), len(models)))
for i, mi in enumerate(models):
    for j, mj in enumerate(models):
        dist_matrix[i, j] = np.linalg.norm(centroids[mi] - centroids[mj])

print("\n=== Pairwise centroid distances (z-space) ===", file=sys.stderr)
sn = [SHORT[m] for m in models]
hdr = " " * 14 + "  ".join(f"{s:>13s}" for s in sn)
print(hdr, file=sys.stderr)
for i, m in enumerate(models):
    row = "  ".join(f"{dist_matrix[i,j]:13.2f}" for j in range(len(models)))
    print(f"{SHORT[m]:14s} {row}", file=sys.stderr)

# Within-model spread (mean intra-cluster distance) — for context
within = {}
for m in models:
    mask = (y == m)
    c = centroids[m]
    d = np.linalg.norm(Xz[mask] - c, axis=1).mean()
    within[m] = d

# Average inter-centroid distance / average within-cluster distance
mean_within = np.mean(list(within.values()))
off_diag = dist_matrix[~np.eye(len(models), dtype=bool)]
mean_between = off_diag.mean()
print(f"\nMean within-cluster distance:  {mean_within:.2f}", file=sys.stderr)
print(f"Mean between-centroid distance: {mean_between:.2f}", file=sys.stderr)
print(f"Ratio between/within (>1 = separable): {mean_between/mean_within:.2f}", file=sys.stderr)

# ─── 2. PCA ───
from sklearn.decomposition import PCA
pca = PCA(n_components=min(D, 4))
X_pca = pca.fit_transform(Xz)
print(f"\n=== PCA explained variance ===", file=sys.stderr)
for i, ev in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ev:.3f}  cumulative {pca.explained_variance_ratio_[:i+1].sum():.3f}", file=sys.stderr)

# ─── 3. Unsupervised hierarchical clustering ───
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
Z = linkage(Xz, method="ward")
print(f"\n=== Unsupervised clustering vs model labels ===", file=sys.stderr)
print(f"  k    ARI    NMI    purity", file=sys.stderr)
cluster_results = {}
for k in range(2, min(9, len(models) + 1)):
    labels = fcluster(Z, t=k, criterion="maxclust")
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    # cluster purity: in each cluster, fraction matching the dominant true label
    pur = 0.0
    for c in set(labels):
        mask = labels == c
        if mask.sum():
            mc = Counter(y[mask]).most_common(1)[0][1]
            pur += mc
    pur /= len(y)
    cluster_results[k] = {"ari": ari, "nmi": nmi, "purity": pur}
    print(f"  {k}    {ari:.3f}  {nmi:.3f}  {pur:.3f}", file=sys.stderr)

# ─── 4. LDA — supervised separability ───
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
lda = LinearDiscriminantAnalysis()
loo = LeaveOneOut()
lda_scores = cross_val_score(lda, Xz, y, cv=loo)
lda_acc = lda_scores.mean()
n_classes = len(models)
chance = 1.0 / n_classes
print(f"\n=== LDA leave-one-out accuracy ===", file=sys.stderr)
print(f"  LDA acc: {lda_acc:.3f}  (chance: {chance:.3f}, n_classes: {n_classes})", file=sys.stderr)

# Also: 2-class collapsed: family (anthropic / openai / other)
FAMILY = {
    "anthropic/claude-sonnet-4": "anthropic",
    "anthropic/claude-sonnet-4.5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic",
    "openai/gpt-4-turbo": "openai",
    "openai/gpt-4o": "openai",
    "openai/gpt-5": "openai",
    "google/gemini-2.5-pro": "google",
    "meta-llama/llama-3.3-70b-instruct": "meta",
}
y_fam = np.array([FAMILY[m] for m in y])
fam_classes = sorted(set(y_fam))
n_fam = len(fam_classes)
if n_fam >= 2:
    lda_fam_scores = cross_val_score(lda, Xz, y_fam, cv=loo)
    lda_fam_acc = lda_fam_scores.mean()
    fam_chance = max(Counter(y_fam).values()) / len(y_fam)  # majority-class baseline
    print(f"  LDA family acc: {lda_fam_acc:.3f}  (majority-class baseline: {fam_chance:.3f}, families: {fam_classes})", file=sys.stderr)
else:
    lda_fam_acc = None

# LDA-projected coords (for viz)
lda_full = LinearDiscriminantAnalysis(n_components=min(n_classes - 1, 2))
X_lda = lda_full.fit_transform(Xz, y)

# ─── 5. Permutation null ───
rng = np.random.default_rng(7)
null_accs = []
for _ in range(200):
    y_perm = rng.permutation(y)
    s = cross_val_score(lda, Xz, y_perm, cv=loo)
    null_accs.append(s.mean())
null_p = (np.array(null_accs) >= lda_acc).mean()
print(f"\n=== LDA permutation null (n=200) ===", file=sys.stderr)
print(f"  null mean: {np.mean(null_accs):.3f}  null max: {np.max(null_accs):.3f}", file=sys.stderr)
print(f"  observed: {lda_acc:.3f}   p ≈ {null_p:.3f}", file=sys.stderr)

# ─── 6. Feature importance from LDA ───
# Compute |coef|.mean across classes per feature
coef = np.abs(lda_full.coef_).mean(axis=0) if n_classes > 2 else np.abs(lda_full.coef_).ravel()
# Standardize for relative comparison
coef_n = coef / coef.sum()
print(f"\n=== Per-feature LDA |coef| (normalized) ===", file=sys.stderr)
for f, c in sorted(zip(FEATS, coef_n), key=lambda x: -x[1]):
    print(f"  {f:20s} {c:.3f}", file=sys.stderr)

# ─── 7. Mode-label confusion ───
mode_dist = defaultdict(Counter)
for r in records:
    mode_dist[r["model"]][r["mode"]] += 1

# ─── Persist analysis ───
out = {
    "models": models,
    "model_short": SHORT,
    "n_per_model": {m: len(by_model[m]) for m in models},
    "centroids": {m: centroids[m].tolist() for m in models},
    "within_spread": {m: float(within[m]) for m in models},
    "dist_matrix": dist_matrix.tolist(),
    "mean_within": float(mean_within),
    "mean_between": float(mean_between),
    "ratio": float(mean_between / mean_within),
    "pca_explained": pca.explained_variance_ratio_.tolist(),
    "pca_coords": {m: X_pca[y == m].tolist() for m in models},
    "lda_coords": {m: X_lda[y == m].tolist() for m in models},
    "lda_acc": float(lda_acc),
    "lda_chance": float(chance),
    "lda_family_acc": float(lda_fam_acc) if lda_fam_acc else None,
    "cluster_results": cluster_results,
    "null_mean": float(np.mean(null_accs)),
    "null_max": float(np.max(null_accs)),
    "null_p": float(null_p),
    "feat_coef_norm": dict(zip(FEATS, coef_n.tolist())),
    "mode_dist": {m: dict(mode_dist[m]) for m in models},
    "feats": FEATS,
}
OUT.write_text(json.dumps(out, indent=2))
print(f"\nWrote {OUT}", file=sys.stderr)
