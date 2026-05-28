#!/usr/bin/env python3
"""Compute confusion matrix from spectrometer-basis LDA LOO predictions.
Also produces simple accuracy ladder data."""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, "/tmp/waivelets-v0.1")
from fastprint import _load_basis, _get_model, split_sentences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA

gut_basis, _ = _load_basis()
llm_basis = np.load("/tmp/exp/llm_basis.npz")["basis"]
print(f"gut={gut_basis.shape}  llm={llm_basis.shape}", file=sys.stderr)

# Load cached embeddings
z = np.load("/tmp/exp/all_sent_embs.npz", allow_pickle=True)
all_embs = z["embs"]
gen_ranges = z["ranges"]
print(f"all_embs={all_embs.shape}  gens={len(gen_ranges)}", file=sys.stderr)

def feat_basis(basis):
    feats = []
    for (s, e, _, _) in gen_ranges:
        emb = all_embs[s:e]
        proj = emb @ basis
        mean = proj.mean(axis=0)
        amean = np.abs(proj).mean(axis=0)
        basin = np.argmax(np.abs(proj), axis=1)
        hist = np.bincount(basin, minlength=basis.shape[1]).astype(float)
        hist /= hist.sum() if hist.sum() > 0 else 1.0
        feats.append(np.concatenate([mean, amean, hist]))
    return np.array(feats)

X_gut = feat_basis(gut_basis)
X_llm = feat_basis(llm_basis)
X_both = np.concatenate([X_gut, X_llm], axis=1)
centroids_384 = np.array([all_embs[s:e].mean(axis=0) for s,e,_,_ in gen_ranges])

y = np.array([m for _,_,m,_ in gen_ranges])
SHORT = {
    "anthropic/claude-sonnet-4": "sonnet-4",
    "anthropic/claude-sonnet-4.5": "sonnet-4.5",
    "anthropic/claude-haiku-4.5": "haiku-4.5",
    "openai/gpt-4-turbo": "gpt-4-turbo",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-5": "gpt-5",
    "google/gemini-2.5-pro": "gemini-2.5",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3",
}
FAMILY = {
    "anthropic/claude-sonnet-4": "anthropic", "anthropic/claude-sonnet-4.5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic", "openai/gpt-4-turbo": "openai",
    "openai/gpt-4o": "openai", "openai/gpt-5": "openai",
    "google/gemini-2.5-pro": "google", "meta-llama/llama-3.3-70b-instruct": "meta",
}
# Order: anthropic family, openai family, google, meta
order = [
    "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5",
    "openai/gpt-4-turbo", "openai/gpt-4o", "openai/gpt-5",
    "google/gemini-2.5-pro", "meta-llama/llama-3.3-70b-instruct"
]

def standardize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

def loo_predictions(X, y):
    Xs = standardize(X)
    preds = np.empty(len(y), dtype=object)
    for train, test in LeaveOneOut().split(Xs):
        lda = LinearDiscriminantAnalysis()
        lda.fit(Xs[train], y[train])
        preds[test] = lda.predict(Xs[test])
    return preds

preds_spec = loo_predictions(X_both, y)
preds_384 = loo_predictions(centroids_384, y)
preds_gut = loo_predictions(X_gut, y)
preds_llm = loo_predictions(X_llm, y)

def conf_matrix(y_true, y_pred, order):
    n = len(order)
    idx = {m: i for i, m in enumerate(order)}
    M = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M

def per_model_acc(y_true, y_pred, order):
    accs = {}
    for m in order:
        mask = y_true == m
        if mask.sum():
            accs[m] = float((y_pred[mask] == m).sum() / mask.sum())
    return accs

# Family-level
y_fam = np.array([FAMILY[m] for m in y])
preds_spec_fam = loo_predictions(X_both, y_fam)
fam_order = ["anthropic", "openai", "google", "meta"]
def conf_fam(y_true, y_pred, order):
    n = len(order)
    idx = {m: i for i, m in enumerate(order)}
    M = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M

cm_spec = conf_matrix(y, preds_spec, order)
cm_fam = conf_fam(y_fam, preds_spec_fam, fam_order)
acc_spec = per_model_acc(y, preds_spec, order)

print("\n=== Spectrometer per-model recall ===", file=sys.stderr)
for m in order:
    print(f"  {SHORT[m]:14s}  {acc_spec[m]*100:5.1f}%", file=sys.stderr)

print("\n=== Spectrometer confusion matrix (rows=true, cols=predicted) ===", file=sys.stderr)
print(" " * 16 + "  ".join(f"{SHORT[m][:10]:>10s}" for m in order), file=sys.stderr)
for i, m in enumerate(order):
    print(f"{SHORT[m]:16s}" + "  ".join(f"{cm_spec[i,j]:>10d}" for j in range(len(order))), file=sys.stderr)

out = {
    "order": order,
    "short": SHORT,
    "family": FAMILY,
    "fam_order": fam_order,
    "n_per_model": {m: int((y == m).sum()) for m in order},
    "spectrometer": {
        "confusion_matrix": cm_spec.tolist(),
        "per_model_recall": acc_spec,
        "overall_acc": float((preds_spec == y).mean()),
    },
    "spectrometer_family": {
        "confusion_matrix": cm_fam.tolist(),
        "overall_acc": float((preds_spec_fam == y_fam).mean()),
    },
    "ladder": {
        "chance_8": 1/8,
        "gut_only_8":  float((preds_gut == y).mean()),
        "llm_only_8":  float((preds_llm == y).mean()),
        "spectrometer_8": float((preds_spec == y).mean()),
        "raw_384_8":   float((preds_384 == y).mean()),
        "chance_fam": 1/4,
        "majority_fam": float(max((y_fam == f).sum() for f in fam_order) / len(y_fam)),
        "spectrometer_fam": float((preds_spec_fam == y_fam).mean()),
    },
}
pathlib.Path("/tmp/exp/confusion.json").write_text(json.dumps(out, indent=2))
print(f"\nOverall 8-way acc: {out['spectrometer']['overall_acc']:.3f}", file=sys.stderr)
print(f"Overall family acc: {out['spectrometer_family']['overall_acc']:.3f}", file=sys.stderr)
print("Wrote /tmp/exp/confusion.json", file=sys.stderr)
