#!/usr/bin/env python3
"""Fit an LLM-corpus basis from the 96 generations, then compare model discrimination
across: Gutenberg basis (existing), LLM-PCA basis (new), combined spectrometer, raw 384-D.

Method: extract all sentence embeddings, fit PCA(26) for an unsupervised LLM basis.
Then per-generation aggregate features in each basis. Compare via leave-one-out LDA.
"""
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, "/tmp/waivelets-v0.1")
from fastprint import _load_basis, _get_model, split_sentences
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from collections import Counter

# ── 1. Load all 96 generations and embed all sentences ──
records = [json.loads(l) for l in pathlib.Path("/tmp/exp/results.jsonl").open()]
print(f"Records: {len(records)}", file=sys.stderr)

gut_basis, _ = _load_basis()       # (384, 26)
model = _get_model()
print(f"Gutenberg basis: {gut_basis.shape}", file=sys.stderr)

CACHE = pathlib.Path("/tmp/exp/all_sent_embs.npz")
if CACHE.exists():
    print(f"Loading cached embeddings", file=sys.stderr)
    z = np.load(CACHE, allow_pickle=True)
    all_embs = z["embs"]
    gen_ranges = z["ranges"]       # list of (start, end, model, prompt_idx)
    gen_meta = list(z["meta"])
else:
    all_embs_list = []
    gen_ranges = []
    gen_meta = []
    t0 = time.time()
    for i, r in enumerate(records):
        sents = split_sentences(r["text"])
        if len(sents) < 3:
            continue
        emb = model.encode(sents, batch_size=64, show_progress_bar=False)
        start = len(all_embs_list)
        all_embs_list.append(emb)
        end = start + len(emb)
        gen_ranges.append((start, end, r["model"], r["prompt_idx"]))
        gen_meta.append((r["model"], r["prompt_idx"]))
        if (i+1) % 12 == 0:
            print(f"  embedded {i+1}/{len(records)}  total_sents={sum(e.shape[0] for e in all_embs_list)}", file=sys.stderr)
    all_embs = np.concatenate(all_embs_list, axis=0)
    np.savez(CACHE, embs=all_embs, ranges=np.array(gen_ranges, dtype=object), meta=np.array(gen_meta, dtype=object))
    print(f"Embeddings: {all_embs.shape}  in {time.time()-t0:.1f}s", file=sys.stderr)

# ── 2. Fit LLM-corpus basis: PCA(26) on all LLM sentence embeddings ──
pca_llm = PCA(n_components=26, random_state=0)
pca_llm.fit(all_embs)
llm_basis = pca_llm.components_.T   # (384, 26)
print(f"LLM basis: {llm_basis.shape}", file=sys.stderr)
print(f"  LLM PCA explained variance (top 6): {pca_llm.explained_variance_ratio_[:6].round(3)}", file=sys.stderr)
print(f"  cumulative @ 26 components: {pca_llm.explained_variance_ratio_.sum():.3f}", file=sys.stderr)

# How orthogonal are LLM and Gutenberg bases? Compute principal angles.
# A simple summary: |cos| between best-matched pairs
M = gut_basis.T @ llm_basis    # (26, 26) — cosines
abs_M = np.abs(M)
# Hungarian-style: for each Gutenberg axis, best LLM-axis match
best_match = abs_M.max(axis=1)
print(f"  per-Gutenberg-axis best |cos| with LLM basis: mean={best_match.mean():.3f}  median={np.median(best_match):.3f}  min={best_match.min():.3f}  max={best_match.max():.3f}", file=sys.stderr)
print(f"  → bases overlap: {'high' if best_match.mean() > 0.7 else 'moderate' if best_match.mean() > 0.4 else 'low'} similarity", file=sys.stderr)

# ── 3. Aggregate per-generation features under each basis ──
def feat_basis(basis, all_embs, gen_ranges):
    """For each generation, project sentences into basis and compute aggregates.
    Returns (N, 26+26+26)=78-D feature: mean, |mean|, basin_hist."""
    feats = []
    for (s, e, _, _) in gen_ranges:
        emb = all_embs[s:e]
        proj = emb @ basis   # (n, 26)
        mean = proj.mean(axis=0)
        amean = np.abs(proj).mean(axis=0)
        basin = np.argmax(np.abs(proj), axis=1)
        hist = np.bincount(basin, minlength=basis.shape[1]).astype(float)
        hist /= hist.sum() if hist.sum() > 0 else 1.0
        feats.append(np.concatenate([mean, amean, hist]))
    return np.array(feats)

# Also: raw 384-D centroid per generation
centroids_384 = np.array([all_embs[s:e].mean(axis=0) for s,e,_,_ in gen_ranges])
print(f"\ncentroids_384 shape: {centroids_384.shape}", file=sys.stderr)

X_gut = feat_basis(gut_basis, all_embs, gen_ranges)
X_llm = feat_basis(llm_basis, all_embs, gen_ranges)
X_both = np.concatenate([X_gut, X_llm], axis=1)
print(f"X_gut: {X_gut.shape}  X_llm: {X_llm.shape}  X_both: {X_both.shape}", file=sys.stderr)

# ── 4. Labels ──
y = np.array([m for _,_,m,_ in gen_ranges])
FAMILY = {
    "anthropic/claude-sonnet-4": "anthropic", "anthropic/claude-sonnet-4.5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic", "openai/gpt-4-turbo": "openai",
    "openai/gpt-4o": "openai", "openai/gpt-5": "openai",
    "google/gemini-2.5-pro": "google", "meta-llama/llama-3.3-70b-instruct": "meta",
}
y_fam = np.array([FAMILY[m] for m in y])

# ── 5. LDA evaluation ──
def standardize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

def lda_loo(X, y, n_pca=None):
    Xs = standardize(X)
    if n_pca and Xs.shape[1] > n_pca:
        Xs = PCA(n_components=n_pca, random_state=0).fit_transform(Xs)
    return cross_val_score(LinearDiscriminantAnalysis(), Xs, y, cv=LeaveOneOut()).mean()

def perm_null(X, y, n_pca=None, n_perm=100):
    rng = np.random.default_rng(7)
    accs = []
    for _ in range(n_perm):
        accs.append(lda_loo(X, rng.permutation(y), n_pca))
    return float(np.mean(accs)), float(np.max(accs))

print("\n=== Family-level (4-way) LDA leave-one-out ===", file=sys.stderr)
print(f"{'feature space':38s} {'dim':>5s}  {'acc':>5s}  {'null_mean':>10s}  {'null_max':>9s}", file=sys.stderr)
specs = [
    ("Gutenberg 78-D (existing baseline)", X_gut, None),
    ("LLM-PCA 78-D (new basis)", X_llm, None),
    ("Spectrometer 156-D (Gutenberg+LLM)", X_both, None),
    ("Raw 384-D centroid", centroids_384, None),
    ("Raw 384-D → PCA(50)", centroids_384, 50),
]
results = {}
for name, X, n_pca in specs:
    acc = lda_loo(X, y_fam, n_pca)
    nm, nmax = perm_null(X, y_fam, n_pca)
    results[name] = {"dim": X.shape[1], "acc": acc, "null_mean": nm, "null_max": nmax}
    print(f"{name:38s} {X.shape[1]:>5d}  {acc:.3f}  {nm:>10.3f}  {nmax:>9.3f}", file=sys.stderr)

print("\n=== Model-level (8-way) LDA leave-one-out ===", file=sys.stderr)
for name, X, n_pca in specs:
    acc = lda_loo(X, y, n_pca)
    nm, nmax = perm_null(X, y, n_pca)
    results[name + " [8-way]"] = {"dim": X.shape[1], "acc": acc, "null_mean": nm, "null_max": nmax}
    print(f"{name:38s} {X.shape[1]:>5d}  {acc:.3f}  {nm:>10.3f}  {nmax:>9.3f}", file=sys.stderr)

# ── 6. Save basis + results ──
np.savez("/tmp/exp/llm_basis.npz",
         basis=llm_basis,
         explained_variance_ratio=pca_llm.explained_variance_ratio_)
pathlib.Path("/tmp/exp/multibasis_results.json").write_text(json.dumps({
    "results": results,
    "basis_overlap": {
        "per_gut_axis_best_abs_cos_mean": float(best_match.mean()),
        "per_gut_axis_best_abs_cos_median": float(np.median(best_match)),
        "per_gut_axis_best_abs_cos_min": float(best_match.min()),
        "per_gut_axis_best_abs_cos_max": float(best_match.max()),
    },
    "llm_pca_top6_variance": pca_llm.explained_variance_ratio_[:6].tolist(),
    "llm_pca_total_26": float(pca_llm.explained_variance_ratio_.sum()),
}, indent=2))
print("\nWrote /tmp/exp/multibasis_results.json", file=sys.stderr)
print("DONE", file=sys.stderr)
