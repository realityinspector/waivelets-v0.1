#!/usr/bin/env python3
"""
Char-Code Wavelet Test
======================

Can we skip embeddings entirely? What if we just map characters to numbers
(a=1, b=2, ...) and run the same wavelet / spectral analysis on that signal?

If the structural modes are real dynamical properties of text, they might
show up even in a signal this primitive — because the dynamics are about
*how meaning unfolds over time*, and even character distributions carry
temporal structure (paragraph length, sentence rhythm, vocabulary complexity).

Test: fingerprint with MiniLM embeddings vs char-code "embeddings",
see if both find the same modes.
"""

import numpy as np
import time
import json
from pathlib import Path


def charcode_embed_sentence(sentence: str) -> np.ndarray:
    """
    Map each character to its ordinal, average over the sentence.
    Returns a fixed-size vector from character statistics.

    We compute several features per sentence:
    - Mean char code (captures vocabulary register)
    - Std char code (captures character diversity)
    - Fraction uppercase (captures formality)
    - Fraction punctuation (captures syntactic density)
    - Word length mean (captures vocabulary complexity)
    - Word length std (captures word length variety)
    - Sentence length in chars (captures breath/pacing)
    - Fraction vowels (captures phonetic texture)
    - Fraction spaces (inverse of word length, captures density)
    - Bigram entropy (captures local character predictability)
    """
    if not sentence.strip():
        return np.zeros(10)

    codes = np.array([ord(c) for c in sentence], dtype=np.float32)
    chars = list(sentence)
    words = sentence.split()

    # Basic char stats
    mean_code = np.mean(codes)
    std_code = np.std(codes)

    # Character class fractions
    n = len(sentence) or 1
    frac_upper = sum(1 for c in chars if c.isupper()) / n
    frac_punct = sum(1 for c in chars if c in '.,;:!?-\'"()[]{}') / n
    frac_vowel = sum(1 for c in chars if c.lower() in 'aeiou') / n
    frac_space = sum(1 for c in chars if c == ' ') / n

    # Word-level stats
    if words:
        wlens = [len(w) for w in words]
        wlen_mean = np.mean(wlens)
        wlen_std = np.std(wlens) if len(wlens) > 1 else 0
    else:
        wlen_mean = 0
        wlen_std = 0

    sent_len = len(sentence)

    # Bigram entropy (2-char transition predictability)
    if len(sentence) > 2:
        bigrams = [sentence[i:i+2] for i in range(len(sentence)-1)]
        from collections import Counter
        bg_counts = Counter(bigrams)
        total = sum(bg_counts.values())
        bg_entropy = -sum((c/total) * np.log2(c/total) for c in bg_counts.values())
    else:
        bg_entropy = 0

    return np.array([
        mean_code, std_code, frac_upper, frac_punct,
        wlen_mean, wlen_std, sent_len, frac_vowel,
        frac_space, bg_entropy
    ], dtype=np.float32)


def charcode_fingerprint(text: str) -> dict:
    """
    Compute a fingerprint using only character-level features.
    No neural network. No embeddings. Just characters → numbers.

    We still compute smoothness, oscillation, and entropy of
    the trajectory through feature space — same dynamical analysis,
    different signal.
    """
    # Split sentences (same regex as fastprint)
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]

    if len(sentences) < 3:
        return {"error": "need 3+ sentences"}

    # Cap at 500 sentences for consistency
    sentences = sentences[:500]

    # Embed each sentence using char-code features
    embeddings = np.array([charcode_embed_sentence(s) for s in sentences])

    # Normalize to zero mean unit variance per feature (like z-scoring)
    means = embeddings.mean(axis=0)
    stds = embeddings.std(axis=0)
    stds[stds < 1e-8] = 1  # avoid div by zero
    embeddings_z = (embeddings - means) / stds

    # Smoothness: cosine similarity between consecutive normalized vectors
    norms = np.linalg.norm(embeddings_z, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1
    normed = embeddings_z / norms

    if len(normed) > 1:
        cos_sims = np.sum(normed[:-1] * normed[1:], axis=1)
        smoothness_mean = float(np.mean(cos_sims))
        smoothness_std = float(np.std(cos_sims))
    else:
        smoothness_mean = 0
        smoothness_std = 0

    # Basin entropy: which feature dimension dominates each sentence
    # (argmax of absolute z-scored embedding = which "basin" it's in)
    basins = np.argmax(np.abs(embeddings_z), axis=1)
    from collections import Counter
    basin_counts = Counter(basins.tolist())
    total = sum(basin_counts.values())
    basin_entropy = -sum(
        (c/total) * np.log2(c/total)
        for c in basin_counts.values()
        if c > 0
    )

    # Feature-group activations (analogous to cluster activations)
    # Group our 10 features into semantic-ish categories
    # [0,1] = char stats, [2,3] = formality, [4,5] = word complexity, [6] = pacing, [7,8,9] = texture
    char_activation = float(np.mean(np.abs(embeddings_z[:, :2])))
    formality_activation = float(np.mean(np.abs(embeddings_z[:, 2:4])))
    complexity_activation = float(np.mean(np.abs(embeddings_z[:, 4:6])))
    texture_activation = float(np.mean(np.abs(embeddings_z[:, 7:])))

    return {
        "smoothness_mean": round(smoothness_mean, 4),
        "smoothness_std": round(smoothness_std, 4),
        "char_activation": round(char_activation, 4),
        "formality_activation": round(formality_activation, 4),
        "complexity_activation": round(complexity_activation, 4),
        "texture_activation": round(texture_activation, 4),
        "basin_entropy": round(basin_entropy, 4),
        "n_sentences": len(sentences),
    }


def run_comparison():
    """
    Compare char-code fingerprints to MiniLM fingerprints.
    Do they find the same structure?
    """
    from eval_ai_vs_human import AI_SAMPLES

    # Also load some human texts
    corpus_path = Path("mega_corpus_results.jsonl")
    human_texts_needed = []
    if corpus_path.exists():
        for line in corpus_path.open():
            r = json.loads(line)
            if r.get("mode"):
                human_texts_needed.append(r)
                if len(human_texts_needed) >= 20:
                    break

    print("=" * 70)
    print("CHAR-CODE WAVELET TEST")
    print("Can character-to-number mapping reveal structural spectra?")
    print("=" * 70)

    # ── Fingerprint AI texts with charcode ──
    print("\nFingerprinting AI texts (char-code)...")
    ai_cc_fps = []
    for sample in AI_SAMPLES:
        fp = charcode_fingerprint(sample["text"])
        if "error" not in fp:
            ai_cc_fps.append(fp)
            print(f"  AI  {sample['id']:30s} sm={fp['smoothness_mean']:+.3f} osc={fp['smoothness_std']:.3f} ent={fp['basin_entropy']:.2f}")

    # ── Fetch and fingerprint human texts with charcode ──
    print("\nFingerprinting human texts (char-code, fetching from Gutenberg)...")
    import urllib.request
    human_cc_fps = []

    # We need the actual text, not just the cached fingerprints
    # Fetch a handful of diverse texts
    test_gids = [
        (1342, "Pride and Prejudice"),
        (84, "Frankenstein"),
        (1661, "Sherlock Holmes"),
        (1524, "Hamlet"),
        (2009, "Origin of Species"),
        (76, "Huckleberry Finn"),
        (174, "Dorian Gray"),
        (5827, "Problems of Philosophy"),
        (1513, "Midsummer Night's Dream"),
        (1321, "Paradise Lost"),
        (11, "Alice in Wonderland"),
        (98, "Tale of Two Cities"),
        (1260, "Jane Eyre"),
        (16, "Peter Pan"),
        (345, "Dracula"),
        (2701, "Moby Dick"),
        (100, "Complete Shakespeare"),
        (120, "Treasure Island"),
        (219, "Heart of Darkness"),
        (768, "Wuthering Heights"),
    ]

    for gid, title in test_gids:
        try:
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="replace")

            # Strip header/footer
            for m in ["*** START OF", "***START OF"]:
                idx = raw.find(m)
                if idx >= 0:
                    raw = raw[raw.index("\n", idx) + 1:]
                    break
            for m in ["*** END OF", "***END OF"]:
                idx = raw.find(m)
                if idx >= 0:
                    raw = raw[:idx]
                    break

            # Take first ~3000 chars of real text
            text = raw[:3000]
            fp = charcode_fingerprint(text)
            if "error" not in fp:
                human_cc_fps.append(fp)
                print(f"  HUM {title:30s} sm={fp['smoothness_mean']:+.3f} osc={fp['smoothness_std']:.3f} ent={fp['basin_entropy']:.2f}")
        except Exception as e:
            print(f"  HUM {title:30s} ERROR: {e}")

    # ── Analysis ──
    print(f"\n{'=' * 70}")
    print(f"CHAR-CODE ANALYSIS: AI={len(ai_cc_fps)}, Human={len(human_cc_fps)}")
    print("=" * 70)

    fp_keys = ["smoothness_mean", "smoothness_std", "char_activation",
               "formality_activation", "complexity_activation",
               "texture_activation", "basin_entropy"]

    ai_arr = np.array([[fp[k] for k in fp_keys] for fp in ai_cc_fps])
    hum_arr = np.array([[fp[k] for k in fp_keys] for fp in human_cc_fps])

    print(f"\n{'Feature':24s} {'AI mean':>10s} {'AI std':>10s} {'Hum mean':>10s} {'Hum std':>10s} {'Δ/σ':>8s}")
    print("─" * 74)
    for i, key in enumerate(fp_keys):
        ai_m = np.mean(ai_arr[:, i])
        ai_s = np.std(ai_arr[:, i])
        hum_m = np.mean(hum_arr[:, i])
        hum_s = np.std(hum_arr[:, i])
        pooled = np.sqrt((ai_s**2 + hum_s**2) / 2)
        d = abs(ai_m - hum_m) / pooled if pooled > 0 else 0
        marker = " ***" if d > 1.0 else " **" if d > 0.5 else " *" if d > 0.3 else ""
        print(f"{key:24s} {ai_m:10.4f} {ai_s:10.4f} {hum_m:10.4f} {hum_s:10.4f} {d:8.2f}{marker}")

    # Centroid analysis
    ai_cent = np.mean(ai_arr, axis=0)
    hum_cent = np.mean(hum_arr, axis=0)
    inter = np.linalg.norm(ai_cent - hum_cent)
    ai_intra = np.mean([np.linalg.norm(fp - ai_cent) for fp in ai_arr])
    hum_intra = np.mean([np.linalg.norm(fp - hum_cent) for fp in hum_arr])
    ratio = inter / ((ai_intra + hum_intra) / 2)

    print(f"\nCluster separation:")
    print(f"  Inter-centroid distance: {inter:.4f}")
    print(f"  AI intra-class: {ai_intra:.4f}")
    print(f"  Human intra-class: {hum_intra:.4f}")
    print(f"  Separation ratio: {ratio:.3f}")

    if ratio > 1.5:
        print(f"  → STRONG separation even with char-codes!")
    elif ratio > 0.8:
        print(f"  → MODERATE separation — char-codes carry some signal")
    elif ratio > 0.3:
        print(f"  → WEAK separation — mostly noise")
    else:
        print(f"  → NO separation — char-codes don't distinguish")

    # Best single feature classifier
    all_arr = np.vstack([ai_arr, hum_arr])
    labels = np.array([1]*len(ai_arr) + [0]*len(hum_arr))
    best_acc = 0
    best_feat = ""
    for i, key in enumerate(fp_keys):
        vals = all_arr[:, i]
        for pct in range(5, 96):
            thresh = np.percentile(vals, pct)
            for d in [1, -1]:
                preds = (vals * d > thresh * d).astype(int)
                acc = np.mean(preds == labels)
                if acc > best_acc:
                    best_acc = acc
                    best_feat = key

    print(f"\n  Best single-feature classifier: {best_feat}")
    print(f"  Accuracy: {best_acc:.1%}")

    # Comparison summary
    print(f"\n{'=' * 70}")
    print("COMPARISON: MiniLM embeddings vs Char-codes")
    print("=" * 70)
    print(f"  MiniLM separation ratio:   4.877  (from eval_ai_vs_human.py)")
    print(f"  Char-code separation ratio: {ratio:.3f}")
    print(f"  MiniLM best accuracy:      97.3%")
    print(f"  Char-code best accuracy:   {best_acc:.1%}")
    if ratio > 1.0:
        print(f"\n  → CHAR-CODES WORK! The structural signal is real and present")
        print(f"    even in character-level statistics. Embeddings sharpen it,")
        print(f"    but the dynamical structure exists at multiple resolutions.")
    elif ratio > 0.5:
        print(f"\n  → PARTIAL SIGNAL. Char-codes catch some of the structure")
        print(f"    but embeddings are needed for reliable classification.")
    else:
        print(f"\n  → CHAR-CODES MISS IT. The structural signal lives in semantic")
        print(f"    space, not character space. Embeddings are essential.")


if __name__ == "__main__":
    run_comparison()
