#!/usr/bin/env python3
"""Triple/quad/quint spectrometer evaluation with strict nested LOO CV.

Conditions:
  A. Current 2-basis spectrometer  Gut+LLM-PCA (156-D) — baseline (~76%)
  B. LDA-basis alone (21-D from 7-D supervised basis)
  C. Triple-spectrometer (177-D) Gut+LLM+LDA
  D. Contrastive basis alone (48-D, 16-dim×3)
  E. Quad-spectrometer (225-D) Gut+LLM+LDA+Contrastive
  F. Adversarial basis alone (48-D)
  G. Quint-spectrometer (273-D) all 5 bases

For any basis that consumes labels (LDA / contrastive / adversarial), it is
re-fit per fold using ONLY the training fold's sentences.

Data convention (matching `fit_llm_basis.py`): each generation's sentences
are `embs[s:e]` where `(s, e)` come from `all_sent_embs.npz`'s `ranges` field
as-is. This reproduces the 76% spectrometer baseline.
"""
import json, pathlib, sys, time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ---------- Load inputs ----------
z = np.load("/tmp/exp/all_sent_embs.npz", allow_pickle=True)
all_embs = z["embs"].astype(np.float32)           # (3084, 384)
raw_ranges = list(z["ranges"])
gen_info = [(int(r[0]), int(r[1]), str(r[2]), int(r[3])) for r in raw_ranges]
N_GEN = len(gen_info)
print(f"[load] {N_GEN} generations, sentence-matrix={all_embs.shape}", file=sys.stderr)

gut_basis = np.load("/tmp/waivelets-v0.1/basis.npz")["eigvecs"].astype(np.float32)
llm_basis = np.load("/tmp/exp/llm_basis.npz")["basis"].astype(np.float32)

MODELS = sorted({g[2] for g in gen_info})
M2I = {m: i for i, m in enumerate(MODELS)}
N_CLS = len(MODELS)
y_gen = np.array([M2I[g[2]] for g in gen_info], dtype=np.int64)


def per_gen_feature_from_indexed(indexed_proj, K):
    """indexed_proj[i] is (n_i, K) for gen i. Build per-gen 3K feature."""
    feats = np.zeros((N_GEN, 3 * K), dtype=np.float32)
    for i, block in enumerate(indexed_proj):
        mean = block.mean(axis=0)
        amean = np.abs(block).mean(axis=0)
        basin = np.argmax(np.abs(block), axis=1)
        hist = np.bincount(basin, minlength=K).astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        feats[i] = np.concatenate([mean, amean, hist])
    return feats


def per_gen_feature(basis):
    """Project each gen's sentence window through `basis` and build 3K feature."""
    K = basis.shape[1]
    out = []
    for s, e, _, _ in gen_info:
        out.append(all_embs[s:e] @ basis)
    return per_gen_feature_from_indexed(out, K)


# ---------- Fixed (unsupervised) per-gen features ----------
X_GUT = per_gen_feature(gut_basis)
X_LLM = per_gen_feature(llm_basis)
print(f"[feats] X_GUT={X_GUT.shape}  X_LLM={X_LLM.shape}", file=sys.stderr)


# ---------- Build training-fold sentence pool ----------
def fold_train_sentences(held_gen):
    """Return (X_sent, y_sent, p_sent) — concatenation of `embs[s:e]` across
    all gens except `held_gen`."""
    Xs = []
    ys = []
    ps = []
    for i, (s, e, m, p) in enumerate(gen_info):
        if i == held_gen:
            continue
        Xs.append(all_embs[s:e])
        n = e - s
        ys.append(np.full(n, M2I[m], dtype=np.int64))
        ps.append(np.full(n, p, dtype=np.int64))
    return (np.concatenate(Xs, axis=0),
            np.concatenate(ys, axis=0),
            np.concatenate(ps, axis=0))


# ---------- Fold-aware label-bases ----------
def fit_lda_basis(X, y):
    """Return (384,7) LDA scaling matrix."""
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(X, y)
    return lda.scalings_[:, :N_CLS - 1].astype(np.float32)


def fit_contrastive_basis(X, y, p_labels=None, K=16, epochs=120, batch_size=256,
                          margin=0.5, seed=0, adversarial=False, lam=1.0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = Xt.shape[0]

    W = nn.Parameter(torch.randn(384, K, device=device) * 0.05)
    params = [W]

    if adversarial:
        model_head = nn.Linear(K, N_CLS).to(device)
        n_prompts = int(p_labels.max() + 1)
        prompt_head = nn.Linear(K, n_prompts).to(device)
        pt = torch.from_numpy(p_labels).to(device)
        params += list(model_head.parameters()) + list(prompt_head.parameters())

    opt = torch.optim.Adam(params, lr=1e-3)

    by_lab = {c: np.where(y == c)[0] for c in range(N_CLS)}

    for ep in range(epochs):
        idx = rng.integers(0, n, size=batch_size)
        anchor_lab = y[idx]
        pos_idx = np.array([rng.choice(by_lab[c]) for c in anchor_lab])
        neg_lab = (anchor_lab + rng.integers(1, N_CLS, size=batch_size)) % N_CLS
        neg_idx = np.array([rng.choice(by_lab[c]) for c in neg_lab])

        a = Xt[idx] @ W
        pp = Xt[pos_idx] @ W
        nn_ = Xt[neg_idx] @ W

        a_n = F.normalize(a, dim=1)
        p_n = F.normalize(pp, dim=1)
        n_n = F.normalize(nn_, dim=1)

        d_pos = 1 - (a_n * p_n).sum(dim=1)
        d_neg = 1 - (a_n * n_n).sum(dim=1)
        loss = F.relu(d_pos - d_neg + margin).mean()

        if adversarial:
            z = Xt[idx] @ W
            yh = model_head(z)
            ph = prompt_head(z)
            loss_model = F.cross_entropy(yh, yt[idx])
            loss_prompt = F.cross_entropy(ph, pt[idx])
            loss = loss + loss_model + (-lam) * loss_prompt

        opt.zero_grad()
        loss.backward()
        opt.step()

    return W.detach().cpu().numpy().astype(np.float32)


# ---------- Caches ----------
_BASIS_CACHE = {}


def basis_for_fold(kind, held_gen):
    key = (kind, held_gen)
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    Xs, ys, ps = fold_train_sentences(held_gen)
    if kind == "lda":
        W = fit_lda_basis(Xs, ys)
    elif kind == "contrastive":
        W = fit_contrastive_basis(Xs, ys, K=16, epochs=120, seed=0,
                                  adversarial=False)
    elif kind == "adversarial":
        W = fit_contrastive_basis(Xs, ys, p_labels=ps, K=16, epochs=120,
                                  seed=0, adversarial=True, lam=1.0)
    else:
        raise ValueError(kind)
    _BASIS_CACHE[key] = W
    return W


# ---------- Per-fold feature builders ----------
def standardize_train_test(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def builder_fixed(X_full):
    def b(train_idx, test_idx):
        return X_full[train_idx], X_full[test_idx:test_idx + 1]
    return b


def builder_labelbasis(kind, K_expected):
    def b(train_idx, test_idx):
        W = basis_for_fold(kind, test_idx)
        # Project per-gen and build feature
        proj_per_gen = [all_embs[s:e] @ W for (s, e, _, _) in gen_info]
        X_full = per_gen_feature_from_indexed(proj_per_gen, W.shape[1])
        return X_full[train_idx], X_full[test_idx:test_idx + 1]
    return b


def loo_eval(builders, y, label=""):
    n = len(y)
    correct = 0
    preds = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        tr_blocks, te_blocks = [], []
        for bldr in builders:
            tr, te = bldr(train_idx, i)
            tr_blocks.append(tr)
            te_blocks.append(te)
        X_tr = np.concatenate(tr_blocks, axis=1)
        X_te = np.concatenate(te_blocks, axis=1)
        X_tr, X_te = standardize_train_test(X_tr, X_te)
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_tr, y[train_idx])
        p = clf.predict(X_te)[0]
        preds[i] = p
        if p == y[i]:
            correct += 1
        if (i + 1) % 12 == 0:
            print(f"  [{label}] fold {i+1}/{n}  acc_so_far={correct/(i+1):.3f}  "
                  f"elapsed={time.time()-t0:.1f}s", file=sys.stderr)
    acc = correct / n
    print(f"  [{label}] FINAL acc={acc:.4f}  elapsed={time.time()-t0:.1f}s",
          file=sys.stderr)
    return acc, preds


# ---------- Null (permutation) via fixed-feature LOO ----------
def fixed_feature_loo(X, y):
    n = len(y)
    correct = 0
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        sc = StandardScaler()
        Xt = sc.fit_transform(X[tr])
        Xv = sc.transform(X[i:i + 1])
        clf = LinearDiscriminantAnalysis()
        clf.fit(Xt, y[tr])
        if clf.predict(Xv)[0] == y[i]:
            correct += 1
    return correct / n


def perm_null_fixed(X, y, n_perm=50, seed=7):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_perm):
        accs.append(fixed_feature_loo(X, rng.permutation(y)))
    return float(np.mean(accs)), float(np.max(accs))


def materialise_label_features(kind, K):
    """For each gen i, build the per-gen feature using the fold-fit basis where
    i was held-out — what the nested-CV evaluator sees on its test point."""
    out = np.zeros((N_GEN, 3 * K), dtype=np.float32)
    for i in range(N_GEN):
        W = basis_for_fold(kind, i)
        proj_per_gen = [all_embs[s:e] @ W for (s, e, _, _) in gen_info]
        feats = per_gen_feature_from_indexed(proj_per_gen, W.shape[1])
        out[i] = feats[i]
    return out


def materialise_train_view(kind, K, sample_fold=0):
    """A static feature matrix using ONE fold's basis for everybody. Used only
    for permutation-null estimation (treating the basis as fixed). Slight
    leakage acceptable for null."""
    W = basis_for_fold(kind, sample_fold)
    proj_per_gen = [all_embs[s:e] @ W for (s, e, _, _) in gen_info]
    return per_gen_feature_from_indexed(proj_per_gen, W.shape[1])


# ---------- Run conditions ----------
results = {}

print("\n=== A: Gut+LLM-PCA baseline (156-D) ===", file=sys.stderr)
X_A = np.concatenate([X_GUT, X_LLM], axis=1)
acc_A = fixed_feature_loo(X_A, y_gen)
nm_A, nmax_A = perm_null_fixed(X_A, y_gen)
results["A_gut+llm_baseline"] = {"dim": int(X_A.shape[1]), "acc": acc_A,
                                  "null_mean": nm_A, "null_max": nmax_A}
print(f"  acc={acc_A:.4f}  null_mean={nm_A:.3f}  null_max={nmax_A:.3f}",
      file=sys.stderr)

print("\n=== B: LDA basis alone (21-D) ===", file=sys.stderr)
K_LDA = N_CLS - 1
acc_B, _ = loo_eval([builder_labelbasis("lda", K_LDA)], y_gen, label="B")
X_B_train = materialise_train_view("lda", K_LDA)
nm_B, nmax_B = perm_null_fixed(X_B_train, y_gen)
results["B_lda_alone"] = {"dim": 3 * K_LDA, "acc": acc_B,
                           "null_mean": nm_B, "null_max": nmax_B}

print("\n=== C: Triple-spectrometer Gut+LLM+LDA ===", file=sys.stderr)
acc_C, _ = loo_eval([builder_fixed(X_GUT), builder_fixed(X_LLM),
                     builder_labelbasis("lda", K_LDA)], y_gen, label="C")
X_C_train = np.concatenate([X_GUT, X_LLM, X_B_train], axis=1)
nm_C, nmax_C = perm_null_fixed(X_C_train, y_gen)
results["C_triple_spectrometer"] = {"dim": int(X_C_train.shape[1]), "acc": acc_C,
                                     "null_mean": nm_C, "null_max": nmax_C}

pathlib.Path("/tmp/exp/lda_basis_results.json").write_text(json.dumps({
    "A": results["A_gut+llm_baseline"],
    "B": results["B_lda_alone"],
    "C": results["C_triple_spectrometer"],
}, indent=2))
print("[save] /tmp/exp/lda_basis_results.json", file=sys.stderr)

print("\n=== D: Contrastive basis alone (48-D) ===", file=sys.stderr)
K_CON = 16
acc_D, _ = loo_eval([builder_labelbasis("contrastive", K_CON)], y_gen, label="D")
X_D_train = materialise_train_view("contrastive", K_CON)
nm_D, nmax_D = perm_null_fixed(X_D_train, y_gen)
results["D_contrastive_alone"] = {"dim": 3 * K_CON, "acc": acc_D,
                                   "null_mean": nm_D, "null_max": nmax_D}

print("\n=== E: Quad-spectrometer Gut+LLM+LDA+Contrastive ===", file=sys.stderr)
acc_E, _ = loo_eval([builder_fixed(X_GUT), builder_fixed(X_LLM),
                     builder_labelbasis("lda", K_LDA),
                     builder_labelbasis("contrastive", K_CON)], y_gen, label="E")
X_E_train = np.concatenate([X_GUT, X_LLM, X_B_train, X_D_train], axis=1)
nm_E, nmax_E = perm_null_fixed(X_E_train, y_gen)
results["E_quad_spectrometer"] = {"dim": int(X_E_train.shape[1]), "acc": acc_E,
                                   "null_mean": nm_E, "null_max": nmax_E}

pathlib.Path("/tmp/exp/contrastive_results.json").write_text(json.dumps({
    "D": results["D_contrastive_alone"],
    "E": results["E_quad_spectrometer"],
}, indent=2))
print("[save] /tmp/exp/contrastive_results.json", file=sys.stderr)

print("\n=== F: Adversarial basis alone (48-D) ===", file=sys.stderr)
acc_F, _ = loo_eval([builder_labelbasis("adversarial", K_CON)], y_gen, label="F")
X_F_train = materialise_train_view("adversarial", K_CON)
nm_F, nmax_F = perm_null_fixed(X_F_train, y_gen)
results["F_adversarial_alone"] = {"dim": 3 * K_CON, "acc": acc_F,
                                   "null_mean": nm_F, "null_max": nmax_F}

print("\n=== G: Quint-spectrometer (all 5 bases) ===", file=sys.stderr)
acc_G, _ = loo_eval([builder_fixed(X_GUT), builder_fixed(X_LLM),
                     builder_labelbasis("lda", K_LDA),
                     builder_labelbasis("contrastive", K_CON),
                     builder_labelbasis("adversarial", K_CON)], y_gen, label="G")
X_G_train = np.concatenate([X_GUT, X_LLM, X_B_train, X_D_train, X_F_train], axis=1)
nm_G, nmax_G = perm_null_fixed(X_G_train, y_gen)
results["G_quint_spectrometer"] = {"dim": int(X_G_train.shape[1]), "acc": acc_G,
                                    "null_mean": nm_G, "null_max": nmax_G}

pathlib.Path("/tmp/exp/adversarial_results.json").write_text(json.dumps({
    "F": results["F_adversarial_alone"],
    "G": results["G_quint_spectrometer"],
}, indent=2))
print("[save] /tmp/exp/adversarial_results.json", file=sys.stderr)


def pvalue(obs, null_max, n_perm=50):
    if obs > null_max:
        return 1.0 / (n_perm + 1)
    return None


combined = {}
for k, v in results.items():
    v = dict(v)
    v["p_one_sided_approx"] = pvalue(v["acc"], v["null_max"])
    combined[k] = v

pathlib.Path("/tmp/exp/expansion_results.json").write_text(json.dumps(combined, indent=2))
print("[save] /tmp/exp/expansion_results.json", file=sys.stderr)

# Pretty summary
print("\n" + "=" * 72, file=sys.stderr)
print(f"{'Condition':40s} {'dim':>5s}  {'acc':>6s}  {'null_mean':>9s}  {'null_max':>9s}",
      file=sys.stderr)
for k, v in results.items():
    print(f"{k:40s} {v['dim']:>5d}  {v['acc']:.4f}  {v['null_mean']:>9.3f}  {v['null_max']:>9.3f}",
          file=sys.stderr)
print("DONE", file=sys.stderr)
