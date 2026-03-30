#!/usr/bin/env python3
"""
Fast-Path Attractor Fingerprint
================================

Text in → 7 numbers out → instant genre/style classification.

No wavelets at runtime. The eigenbasis was precomputed from wavelet
correlation analysis of three reference texts. At runtime it's just:

  embed sentences (~1ms each)
  → project onto 26 eigenvectors (one matmul)
  → accumulate smoothness, oscillation, cluster activations (running counters)

The 7-number fingerprint:
  1. smoothness_mean    — how much the text stays in one semantic mode
  2. smoothness_std     — how much the rhythm varies (oscillation)
  3. exposition         — cluster activation for explanation/comparison
  4. interiority        — cluster activation for psychological/inner states
  5. formal_structure   — cluster activation for complex clause structure
  6. scene_narration    — cluster activation for bodies-in-space
  7. basin_entropy      — how many attractor basins are visited (diversity)

Usage:
  # As a library
  from fastprint import fingerprint, classify, explain
  fp = fingerprint("Your text here. Multiple sentences work best.")
  label = classify(fp)
  explain(fp)

  # From command line
  python fastprint.py "Paste text here..."
  python fastprint.py --file mytext.txt
  python fastprint.py --compare file1.txt file2.txt
  echo "pipe text in" | python fastprint.py --stdin
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Precomputed basis (37KB) ────────────────────────────────────────

_BASIS_DIR = Path(__file__).parent
_eigvecs = None
_clusters = None
_model = None


def _load_basis():
    global _eigvecs, _clusters
    if _eigvecs is None:
        data = np.load(_BASIS_DIR / "basis.npz")
        _eigvecs = data["eigvecs"]  # (384, 26)
    if _clusters is None:
        with open(_BASIS_DIR / "basis_clusters.json") as f:
            raw = json.load(f)
        _clusters = {k: np.array(v) for k, v in raw.items()}
    return _eigvecs, _clusters


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ── Sentence splitting ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple but effective."""
    import re
    text = text.strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r'(?<=[.!?])\s+', text)
    # Also split on double newlines (paragraph breaks)
    result = []
    for part in parts:
        sub = part.split('\n\n')
        result.extend(s.strip() for s in sub if s.strip())
    # Filter out very short fragments
    return [s for s in result if len(s) > 10]


# ── The fingerprint ─────────────────────────────────────────────────

class Fingerprint(NamedTuple):
    smoothness_mean: float
    smoothness_std: float
    exposition: float
    interiority: float
    formal_structure: float
    scene_narration: float
    basin_entropy: float

    def to_dict(self) -> dict:
        return self._asdict()

    def to_array(self) -> np.ndarray:
        return np.array(self)

    def __repr__(self):
        parts = [f"{k}={v:+.4f}" if k.startswith('smooth') else f"{k}={v:.4f}"
                 for k, v in self._asdict().items()]
        return f"Fingerprint({', '.join(parts)})"


def fingerprint(text: str, sentences: list[str] | None = None,
                embeddings: np.ndarray | None = None) -> Fingerprint:
    """
    Compute the 7-number attractor fingerprint of a text.

    Args:
        text: Raw text string (will be sentence-split and embedded)
        sentences: Pre-split sentences (skips splitting)
        embeddings: Pre-computed embeddings (skips embedding)

    Returns:
        Fingerprint namedtuple with 7 values
    """
    eigvecs, clusters = _load_basis()

    # Step 1: Get embeddings
    if embeddings is None:
        if sentences is None:
            sentences = split_sentences(text)
        if len(sentences) < 3:
            raise ValueError(f"Need at least 3 sentences, got {len(sentences)}")
        model = _get_model()
        embeddings = model.encode(sentences, batch_size=256, show_progress_bar=False)

    n = embeddings.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 embeddings, got {n}")

    # Step 2: Project onto eigenbasis (one matmul — microseconds)
    proj = embeddings @ eigvecs  # (n, 26)

    # Step 3: Smoothness — cosine similarity between consecutive projections
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = proj / norms

    cos_sim = np.sum(normed[:-1] * normed[1:], axis=1)  # (n-1,)
    smoothness_mean = float(cos_sim.mean())
    smoothness_std = float(cos_sim.std())

    # Step 4: Cluster activations (mean absolute activation per cluster)
    cluster_acts = {}
    for cname, cdims in clusters.items():
        cluster_acts[cname] = float(np.abs(embeddings[:, cdims]).mean())

    # Step 5: Basin entropy (how diverse is the attractor visit pattern?)
    basin = np.argmax(np.abs(proj), axis=1)
    counts = np.bincount(basin, minlength=eigvecs.shape[1]).astype(float)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    basin_entropy = float(-np.sum(probs * np.log2(probs)))

    return Fingerprint(
        smoothness_mean=smoothness_mean,
        smoothness_std=smoothness_std,
        exposition=cluster_acts.get("exposition", 0),
        interiority=cluster_acts.get("interiority", 0),
        formal_structure=cluster_acts.get("formal_structure", 0),
        scene_narration=cluster_acts.get("scene_narration", 0),
        basin_entropy=basin_entropy,
    )


# ── Classification ──────────────────────────────────────────────────

# Dynamical structural modes derived from 25-text Gutenberg corpus analysis.
# These are NOT genres — they describe HOW meaning unfolds, not WHAT the text
# is about. Genres map to modes many-to-many.
#
# Centroids are in z-scored space. Raw centroids provided for reference.
# Z-scoring uses corpus statistics to weight all features equally.

# Corpus statistics for z-scoring (from 25-text Gutenberg batch)
_CORPUS_MEAN = np.array([0.2866, 0.2300, 0.0403, 0.0390, 0.0451, 0.0414, 4.2923])
_CORPUS_STD  = np.array([0.0712, 0.0207, 0.0011, 0.0014, 0.0015, 0.0020, 0.2682])

MODE_PROFILES = {
    "convergent": {
        "centroid": Fingerprint(
            smoothness_mean=0.399, smoothness_std=0.224,
            exposition=0.0412, interiority=0.0387,
            formal_structure=0.0473, scene_narration=0.0423,
            basin_entropy=3.61,
        ),
        "description": (
            "Text that narrows to few attractor basins and holds. "
            "High coherence, concentrated semantic range. "
            "Liturgical, aphoristic, legalistic, or meditative structure."
        ),
        "examples": "Bhagavad Gita, Dickinson, Locke's Second Treatise",
    },
    "contemplative": {
        "centroid": Fingerprint(
            smoothness_mean=0.339, smoothness_std=0.215,
            exposition=0.0403, interiority=0.0426,
            formal_structure=0.0434, scene_narration=0.0377,
            basin_entropy=4.30,
        ),
        "description": (
            "Text that reflects. High interiority, low scene narration. "
            "Self-aware analysis — thinking about thinking. "
            "Steady pace with wide semantic range."
        ),
        "examples": "Russell's Problems of Philosophy, Swift's Modest Proposal",
    },
    "discursive": {
        "centroid": Fingerprint(
            smoothness_mean=0.262, smoothness_std=0.225,
            exposition=0.0408, interiority=0.0391,
            formal_structure=0.0442, scene_narration=0.0421,
            basin_entropy=4.44,
        ),
        "description": (
            "The wide river of extended prose. Moderate pace, "
            "many attractor basins visited, balanced semantic domains. "
            "Sustained shaped discourse — narrative, argument, or epic."
        ),
        "examples": "Austen, Doyle, Whitman, Plato, Dante, Tocqueville",
    },
    "dialectical": {
        "centroid": Fingerprint(
            smoothness_mean=0.260, smoothness_std=0.255,
            exposition=0.0396, interiority=0.0384,
            formal_structure=0.0462, scene_narration=0.0415,
            basin_entropy=4.33,
        ),
        "description": (
            "Rapid oscillation between registers. High rhythmic variation, "
            "high formal structure. Dramatic mode-switching — "
            "dialogue, confrontation, or systematic contrast."
        ),
        "examples": "Shakespeare, Dickens, Melville, Darwin, Wilde (drama)",
    },
}


def classify(fp: Fingerprint) -> tuple[str, float]:
    """
    Classify a fingerprint into a dynamical structural mode.

    Uses z-scored L2 distance to mode centroids. Returns (mode_name, distance)
    where lower distance = closer match.
    """
    fp_z = (fp.to_array() - _CORPUS_MEAN) / _CORPUS_STD

    best_label = ""
    best_dist = float("inf")

    for label, profile in MODE_PROFILES.items():
        ref_z = (profile["centroid"].to_array() - _CORPUS_MEAN) / _CORPUS_STD
        dist = float(np.linalg.norm(fp_z - ref_z))
        if dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label, best_dist


def distance(fp1: Fingerprint, fp2: Fingerprint) -> float:
    """L2 distance between two fingerprints."""
    return float(np.linalg.norm(fp1.to_array() - fp2.to_array()))


# ── Pretty printing ─────────────────────────────────────────────────

def explain(fp: Fingerprint, name: str = ""):
    """Print a human-readable explanation of a fingerprint."""
    label, dist = classify(fp)
    profile = MODE_PROFILES[label]

    if name:
        print(f"\n  {name}")
        print(f"  {'─' * len(name)}")
    print(f"  Mode: {label.upper()} (distance: {dist:.2f})")
    print(f"  {profile['description']}")
    print()

    # Show all 4 mode distances
    fp_z = (fp.to_array() - _CORPUS_MEAN) / _CORPUS_STD
    print(f"  Mode distances:")
    for mode_name, mode_prof in MODE_PROFILES.items():
        ref_z = (mode_prof["centroid"].to_array() - _CORPUS_MEAN) / _CORPUS_STD
        d = float(np.linalg.norm(fp_z - ref_z))
        marker = " ◀" if mode_name == label else ""
        bar_len = max(0, int((6 - d) / 6 * 20))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {mode_name:16s} {bar} {d:.2f}{marker}")
    print()

    # Smoothness interpretation
    if fp.smoothness_mean > 0.40:
        sm_desc = "HIGH — stays in one semantic mode"
    elif fp.smoothness_mean > 0.28:
        sm_desc = "SUSTAINED — coherent movement between modes"
    elif fp.smoothness_mean > 0.20:
        sm_desc = "FLOWING — regular mode transitions"
    else:
        sm_desc = "VOLATILE — rapid mode-switching"

    print(f"  Smoothness:        {fp.smoothness_mean:+.3f}  {sm_desc}")

    if fp.smoothness_std > 0.28:
        osc_desc = "HIGH — dramatic oscillation between registers"
    elif fp.smoothness_std > 0.23:
        osc_desc = "MODERATE — rhythmic variation"
    else:
        osc_desc = "STEADY — consistent pace"

    print(f"  Oscillation:       {fp.smoothness_std:.3f}   {osc_desc}")
    print()

    # Cluster activations
    clusters = [
        ("Exposition",        fp.exposition,        "explaining, comparing"),
        ("Interiority",       fp.interiority,       "inner states, psychology"),
        ("Formal Structure",  fp.formal_structure,  "complex syntax, meta-text"),
        ("Scene Narration",   fp.scene_narration,   "bodies in space, action"),
    ]

    print(f"  Semantic domains:")
    max_act = max(c[1] for c in clusters)
    for name_, val, desc in clusters:
        bar_len = int(val / max(max_act, 0.001) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {name_:20s} {bar} {val:.4f}  ({desc})")

    print()
    print(f"  Basin entropy:     {fp.basin_entropy:.2f} bits  ", end="")
    if fp.basin_entropy > 4.2:
        print("(WIDE — traverses many attractor basins)")
    elif fp.basin_entropy > 3.5:
        print("(MODERATE — uses several basins)")
    else:
        print("(CONCENTRATED — few basins dominate)")
    print()


# ── Streaming version ───────────────────────────────────────────────

class StreamingFingerprint:
    """
    Accumulates a fingerprint incrementally as sentences arrive.
    Call .update(embedding) for each sentence, .result() at any time.
    """

    def __init__(self):
        eigvecs, clusters = _load_basis()
        self._eigvecs = eigvecs
        self._clusters = clusters
        self._prev_normed = None
        self._smoothness_sum = 0.0
        self._smoothness_sq_sum = 0.0
        self._smoothness_n = 0
        self._cluster_sums = {k: 0.0 for k in clusters}
        self._basin_counts = np.zeros(eigvecs.shape[1], dtype=int)
        self._n = 0

    def update(self, embedding: np.ndarray):
        """Process one sentence embedding (384-dim vector)."""
        proj = embedding @ self._eigvecs  # (26,)
        norm = np.linalg.norm(proj)
        if norm < 1e-10:
            return
        normed = proj / norm

        # Smoothness
        if self._prev_normed is not None:
            cos = float(np.dot(self._prev_normed, normed))
            self._smoothness_sum += cos
            self._smoothness_sq_sum += cos * cos
            self._smoothness_n += 1
        self._prev_normed = normed

        # Basin
        basin = int(np.argmax(np.abs(proj)))
        self._basin_counts[basin] += 1

        # Clusters
        for cname, cdims in self._clusters.items():
            self._cluster_sums[cname] += float(np.abs(embedding[cdims]).mean())

        self._n += 1

    def result(self) -> Fingerprint | None:
        """Get current fingerprint. Returns None if <3 sentences processed."""
        if self._n < 3 or self._smoothness_n < 2:
            return None

        sm_mean = self._smoothness_sum / self._smoothness_n
        sm_var = (self._smoothness_sq_sum / self._smoothness_n) - sm_mean ** 2
        sm_std = max(0, sm_var) ** 0.5

        counts = self._basin_counts[self._basin_counts > 0].astype(float)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log2(probs)))

        return Fingerprint(
            smoothness_mean=sm_mean,
            smoothness_std=sm_std,
            exposition=self._cluster_sums.get("exposition", 0) / self._n,
            interiority=self._cluster_sums.get("interiority", 0) / self._n,
            formal_structure=self._cluster_sums.get("formal_structure", 0) / self._n,
            scene_narration=self._cluster_sums.get("scene_narration", 0) / self._n,
            basin_entropy=entropy,
        )


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast-path attractor fingerprint: text in → 7 numbers out")
    parser.add_argument("text", nargs="?", help="Text to fingerprint (inline)")
    parser.add_argument("--file", "-f", nargs="+", help="File(s) to fingerprint")
    parser.add_argument("--compare", nargs=2, metavar="FILE",
                        help="Compare two files")
    parser.add_argument("--stdin", action="store_true",
                        help="Read from stdin")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on built-in texts")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    if args.compare:
        texts = []
        for path in args.compare:
            with open(path) as f:
                texts.append((path, f.read()))
        fp1 = fingerprint(texts[0][1])
        fp2 = fingerprint(texts[1][1])
        explain(fp1, texts[0][0])
        explain(fp2, texts[1][0])
        print(f"  Distance: {distance(fp1, fp2):.4f}")
        return

    # Collect text
    if args.stdin:
        text = sys.stdin.read()
    elif args.file:
        for path in args.file:
            with open(path) as f:
                text = f.read()
            t0 = time.time()
            fp = fingerprint(text)
            elapsed = time.time() - t0
            if args.json:
                print(json.dumps({"file": path, "fingerprint": fp.to_dict(),
                                  "classification": classify(fp)[0],
                                  "time_ms": round(elapsed * 1000, 1)}))
            else:
                explain(fp, f"{path} ({elapsed*1000:.0f}ms)")
        return
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        return

    t0 = time.time()
    fp = fingerprint(text)
    elapsed = time.time() - t0

    if args.json:
        print(json.dumps({"fingerprint": fp.to_dict(),
                          "classification": classify(fp)[0],
                          "time_ms": round(elapsed * 1000, 1)}))
    else:
        explain(fp, f"Input ({elapsed*1000:.0f}ms)")


def run_benchmark():
    """Benchmark on built-in texts."""
    sys.path.insert(0, str(Path(__file__).parent))
    import wavelet_engine as we

    catalog = we.load_text_catalog()
    print("FAST-PATH FINGERPRINT BENCHMARK")
    print("=" * 60)

    for tid in ["yeats_circus_animals", "oreilly_graph_db", "gatsby"]:
        entry = catalog[tid]
        units = we.load_text_units(tid, catalog)
        precomputed = entry.get("precomputed_embeddings")
        if precomputed and Path(precomputed).exists():
            emb = we.load_precomputed_embeddings(precomputed)
        else:
            emb = we.embed_text(units)

        # Time just the fingerprint computation (no embedding)
        t0 = time.time()
        for _ in range(100):
            fp = fingerprint("", embeddings=emb)
        elapsed_fp = (time.time() - t0) / 100

        # Time including embedding
        t0 = time.time()
        fp = fingerprint("", sentences=units)
        elapsed_full = time.time() - t0

        explain(fp, f"{entry['title']} ({len(units)} sentences)")
        print(f"  Timing:")
        print(f"    Fingerprint only: {elapsed_fp*1000:.2f}ms ({elapsed_fp*1e6:.0f}us)")
        print(f"    With embedding:   {elapsed_full*1000:.0f}ms")
        print(f"    Embedding is {elapsed_full/max(elapsed_fp,1e-9):.0f}x the bottleneck")
        print()


if __name__ == "__main__":
    main()
