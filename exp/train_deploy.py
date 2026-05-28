#!/usr/bin/env python3
"""Train the deployable cascade artifacts:
  - adversarial basis W (384, 16)
  - per-doc 48-D feature standardization stats
  - final LDA classifier (linear coefficients) on per-doc features

Output: /tmp/exp/identify_artifacts.npz with all components for inference.
"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, "/tmp/waivelets-v0.1")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Inputs
z = np.load("/tmp/exp/all_sent_embs.npz", allow_pickle=True)
all_embs = z["embs"]            # (N_sent, 384)
ranges = list(z["ranges"])      # [(start, end, model_id, prompt_idx), ...]

# Restrict to the 8 main models (drop free-model adds)
MAIN_MODELS = [
    "anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "openai/gpt-5",
    "google/gemini-2.5-pro",
    "meta-llama/llama-3.3-70b-instruct",
]
cls_to_idx = {m: i for i, m in enumerate(MAIN_MODELS)}
mask = [m in cls_to_idx for (_,_,m,_) in ranges]
ranges_main = [r for r, ok in zip(ranges, mask) if ok]
N_CLS = len(MAIN_MODELS)
print(f"docs={len(ranges_main)}  models={N_CLS}", file=sys.stderr)

# Per-sentence features + per-sentence labels
sent_X = []
sent_y = []
sent_p = []
for s, e, m, p in ranges_main:
    sent_X.append(all_embs[s:e])
    sent_y.extend([cls_to_idx[m]] * (e - s))
    sent_p.extend([int(p) % 12] * (e - s))  # prompt_idx mod 12 (paid retries use 100+idx)
sent_X = np.concatenate(sent_X, axis=0).astype(np.float32)
sent_y = np.array(sent_y, dtype=np.int64)
sent_p = np.array(sent_p, dtype=np.int64)
print(f"sentences={sent_X.shape}  unique_prompts={len(set(sent_p))}", file=sys.stderr)

# ── Train adversarial basis on ALL sentences ──
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
rng = np.random.default_rng(0)
K = 16
EPOCHS = 200
BATCH = 256
MARGIN = 0.5
LAM = 1.0

device = torch.device("cpu")
Xt = torch.from_numpy(sent_X).to(device)
yt = torch.from_numpy(sent_y).to(device)
pt = torch.from_numpy(sent_p).to(device)

W = nn.Parameter(torch.randn(384, K, device=device) * 0.05)
model_head = nn.Linear(K, N_CLS).to(device)
n_prompts = int(sent_p.max() + 1)
prompt_head = nn.Linear(K, n_prompts).to(device)
opt = torch.optim.Adam([W] + list(model_head.parameters()) + list(prompt_head.parameters()), lr=1e-3)
by_lab = {c: np.where(sent_y == c)[0] for c in range(N_CLS)}

for ep in range(EPOCHS):
    idx = rng.integers(0, sent_X.shape[0], size=BATCH)
    anchor_lab = sent_y[idx]
    pos_idx = np.array([rng.choice(by_lab[c]) for c in anchor_lab])
    neg_lab = (anchor_lab + rng.integers(1, N_CLS, size=BATCH)) % N_CLS
    neg_idx = np.array([rng.choice(by_lab[c]) for c in neg_lab])

    a = Xt[idx] @ W; pp = Xt[pos_idx] @ W; nn_ = Xt[neg_idx] @ W
    a_n = F.normalize(a, dim=1); p_n = F.normalize(pp, dim=1); n_n = F.normalize(nn_, dim=1)
    d_pos = 1 - (a_n * p_n).sum(dim=1)
    d_neg = 1 - (a_n * n_n).sum(dim=1)
    triplet = F.relu(d_pos - d_neg + MARGIN).mean()

    z = Xt[idx] @ W
    loss_model = F.cross_entropy(model_head(z), yt[idx])
    loss_prompt = F.cross_entropy(prompt_head(z), pt[idx])
    loss = triplet + loss_model + (-LAM) * loss_prompt

    opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 40 == 0:
        print(f"ep {ep+1:3d}  triplet={triplet.item():.3f}  model={loss_model.item():.3f}  prompt={loss_prompt.item():.3f}", file=sys.stderr)

W_np = W.detach().cpu().numpy().astype(np.float32)
print(f"W: {W_np.shape}", file=sys.stderr)

# ── Per-doc 48-D feature ──
def doc_feature(emb, W):
    proj = emb @ W  # (n, 16)
    mean = proj.mean(axis=0)
    amean = np.abs(proj).mean(axis=0)
    basin = np.argmax(np.abs(proj), axis=1)
    hist = np.bincount(basin, minlength=W.shape[1]).astype(np.float32)
    hist /= hist.sum() if hist.sum() > 0 else 1.0
    return np.concatenate([mean, amean, hist])

X_doc = np.array([doc_feature(all_embs[s:e], W_np) for s,e,m,p in ranges_main]).astype(np.float32)
y_doc = np.array([cls_to_idx[m] for s,e,m,p in ranges_main])
print(f"X_doc: {X_doc.shape}  y_doc range: {y_doc.min()}..{y_doc.max()}", file=sys.stderr)

# Standardize
feat_mean = X_doc.mean(axis=0)
feat_std = X_doc.std(axis=0) + 1e-9
X_doc_s = (X_doc - feat_mean) / feat_std

# Fit final LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_doc_s, y_doc)

# Sanity: training accuracy (not LOO)
acc = (lda.predict(X_doc_s) == y_doc).mean()
print(f"train acc: {acc:.3f}", file=sys.stderr)

# Save artifacts
np.savez("/tmp/exp/identify_artifacts.npz",
         W=W_np,                                       # (384, 16) adversarial basis
         feat_mean=feat_mean.astype(np.float32),       # (48,)
         feat_std=feat_std.astype(np.float32),         # (48,)
         lda_coef=lda.coef_.astype(np.float32),        # (8, 48)
         lda_intercept=lda.intercept_.astype(np.float32),  # (8,)
         lda_classes=np.array(MAIN_MODELS, dtype=object),
         )
print("Wrote /tmp/exp/identify_artifacts.npz", file=sys.stderr)
print(f"  W: 384x{K}")
print(f"  feat_mean/std: 48")
print(f"  lda_coef: {lda.coef_.shape}")
print(f"  classes: {len(MAIN_MODELS)}")
