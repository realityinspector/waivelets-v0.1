#!/usr/bin/env python3
"""Per-sentence classification + majority vote, LOO at the DOCUMENT level.
Replaces broken v1. Critical: training set must contain ZERO sentences from
the held-out document; labels must stay attached to features through standardization."""
import json, pathlib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import Counter

z = np.load("/tmp/exp/all_sent_embs.npz", allow_pickle=True)
all_embs = z["embs"]
ranges = list(z["ranges"])

gut_basis = np.load("/tmp/waivelets-v0.1/basis.npz")["eigvecs"]
llm_basis = np.load("/tmp/exp/llm_basis.npz")["basis"]

# Build per-sentence feature matrix: [Gutenberg-proj, LLM-PCA-proj] = 52-D
# Track which doc each sentence belongs to + the doc's model label
sent_features = []
sent_doc_idx = []
sent_model = []
doc_model = []
for di, (s, e, m, _) in enumerate(ranges):
    emb = all_embs[s:e]
    proj_g = emb @ gut_basis  # (n, 26)
    proj_l = emb @ llm_basis  # (n, 26)
    feats = np.concatenate([proj_g, proj_l], axis=1)  # (n, 52)
    sent_features.append(feats)
    sent_doc_idx.append(np.full(len(feats), di))
    sent_model.append(np.full(len(feats), m, dtype=object))
    doc_model.append(m)

X = np.concatenate(sent_features, axis=0)
doc_idx = np.concatenate(sent_doc_idx)
y = np.concatenate(sent_model)
doc_model = np.array(doc_model)
n_docs = len(ranges)

print(f"X: {X.shape}  n_docs: {n_docs}  n_models: {len(set(y))}")
print(f"per-class sentence counts:")
for m, c in Counter(y).most_common():
    print(f"  {m:42s}  {c}")

# LOO at the document level
per_sent_correct = 0
per_sent_total = 0
doc_correct_majority = 0
doc_correct_confwt = 0
preds_per_doc = []
acc_per_doc = []

for di in range(n_docs):
    train_mask = (doc_idx != di)
    test_mask = (doc_idx == di)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    # Standardize using ONLY training stats
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-9
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tr_s, y_tr)

    # Per-sentence predictions
    sent_preds = lda.predict(X_te_s)
    n_correct = int((sent_preds == y_te).sum())
    per_sent_correct += n_correct
    per_sent_total += len(y_te)

    true_model = doc_model[di]
    # Majority vote
    maj = Counter(sent_preds).most_common(1)[0][0]
    if maj == true_model:
        doc_correct_majority += 1

    # Confidence-weighted vote (average class probabilities)
    probs = lda.predict_proba(X_te_s)  # (n_sent_in_doc, n_classes)
    mean_probs = probs.mean(axis=0)
    confwt_pred = lda.classes_[mean_probs.argmax()]
    if confwt_pred == true_model:
        doc_correct_confwt += 1

    preds_per_doc.append({"true": true_model, "majority": maj, "confwt": confwt_pred,
                          "sent_correct": n_correct, "n_sent": int(len(y_te))})
    acc_per_doc.append(n_correct / len(y_te))

per_sent_acc = per_sent_correct / per_sent_total
maj_acc = doc_correct_majority / n_docs
confwt_acc = doc_correct_confwt / n_docs

print(f"\n=== Results ===")
print(f"per-sentence accuracy:        {per_sent_acc:.3f}   (n={per_sent_total} sentences)")
print(f"majority-vote doc accuracy:   {maj_acc:.3f}   ({doc_correct_majority}/{n_docs})")
print(f"confwt-vote doc accuracy:     {confwt_acc:.3f}   ({doc_correct_confwt}/{n_docs})")
print(f"current 156-D spectrometer:   0.760     (baseline for comparison)")
print(f"chance (uniform 8-class):     0.125")

# Per-model breakdown
mod_per_sent = {}
mod_per_doc_maj = {}
mod_per_doc_confwt = {}
for m in sorted(set(doc_model)):
    docs_of_m = [d for d in preds_per_doc if d["true"] == m]
    mod_per_sent[m] = sum(d["sent_correct"] for d in docs_of_m) / max(1, sum(d["n_sent"] for d in docs_of_m))
    mod_per_doc_maj[m] = sum(1 for d in docs_of_m if d["majority"] == m) / max(1, len(docs_of_m))
    mod_per_doc_confwt[m] = sum(1 for d in docs_of_m if d["confwt"] == m) / max(1, len(docs_of_m))

print(f"\n=== Per-model ===")
print(f"{'model':46s} {'per-sent':>10s} {'maj-vote':>10s} {'confwt':>10s}")
for m in sorted(set(doc_model)):
    print(f"{m:46s} {mod_per_sent[m]:>10.3f} {mod_per_doc_maj[m]:>10.3f} {mod_per_doc_confwt[m]:>10.3f}")

# Permutation null (50 perms)
rng = np.random.default_rng(7)
null_per_sent = []
null_maj = []
for _ in range(50):
    # Permute doc_model assignments (preserving doc structure)
    y_perm = doc_model.copy()
    rng.shuffle(y_perm)
    # Expand to per-sentence labels
    y_sent_perm = np.array([y_perm[di] for di in doc_idx])
    correct_sent = 0
    total_sent = 0
    correct_maj = 0
    for di in range(n_docs):
        tr = doc_idx != di
        te = doc_idx == di
        X_tr = X[tr]; X_te = X[te]
        mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-9
        X_tr_s = (X_tr - mu)/sd; X_te_s = (X_te - mu)/sd
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_tr_s, y_sent_perm[tr])
            p = lda.predict(X_te_s)
            correct_sent += int((p == y_sent_perm[te]).sum())
            total_sent += len(p)
            maj = Counter(p).most_common(1)[0][0]
            if maj == y_perm[di]:
                correct_maj += 1
        except Exception:
            pass
    null_per_sent.append(correct_sent / max(1, total_sent))
    null_maj.append(correct_maj / n_docs)

print(f"\n=== Permutation null (n=50) ===")
print(f"per-sent: mean={np.mean(null_per_sent):.3f}  max={np.max(null_per_sent):.3f}")
print(f"maj-vote: mean={np.mean(null_maj):.3f}  max={np.max(null_maj):.3f}")
null_p_sent = float((np.array(null_per_sent) >= per_sent_acc).mean())
null_p_maj = float((np.array(null_maj) >= maj_acc).mean())
print(f"p(per-sent): {null_p_sent:.3f}")
print(f"p(maj-vote): {null_p_maj:.3f}")

# Save
pathlib.Path("/tmp/exp/per_sent_vote_v2.json").write_text(json.dumps({
    "per_sentence_acc": float(per_sent_acc),
    "majority_vote_doc_acc": float(maj_acc),
    "confwt_doc_acc": float(confwt_acc),
    "spectrometer_baseline_156d": 0.760,
    "chance": 0.125,
    "per_model_per_sent": {k: float(v) for k,v in mod_per_sent.items()},
    "per_model_maj_vote": {k: float(v) for k,v in mod_per_doc_maj.items()},
    "per_model_confwt": {k: float(v) for k,v in mod_per_doc_confwt.items()},
    "null_per_sent_mean": float(np.mean(null_per_sent)),
    "null_per_sent_max": float(np.max(null_per_sent)),
    "null_maj_mean": float(np.mean(null_maj)),
    "null_maj_max": float(np.max(null_maj)),
    "null_p_per_sent": null_p_sent,
    "null_p_maj_vote": null_p_maj,
    "n_sentences": int(per_sent_total),
    "n_docs": int(n_docs),
}, indent=2))
print(f"\nWrote /tmp/exp/per_sent_vote_v2.json")
