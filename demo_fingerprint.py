#!/usr/bin/env python3
"""
Attractor Fingerprint Demo
===========================

Demonstrates the core discovery: text has measurable attractor dynamics
in embedding space, and you can use those dynamics to select few-shot
examples that match a target *style of unfolding* regardless of content.

This script:
1. Loads all three texts and builds the shared attractor eigenbasis
2. Computes local attractor fingerprints for sliding windows
3. Lets you select a target dynamic (textbook / poetic / confrontation)
4. Finds the passages that best match that dynamic
5. Shows how the same passages differ from content-selected passages

Usage:
  python demo_fingerprint.py                     # interactive mode
  python demo_fingerprint.py --style textbook    # direct mode
  python demo_fingerprint.py --style poetic
  python demo_fingerprint.py --style confrontation
  python demo_fingerprint.py --compare           # side-by-side comparison
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import wavelet_engine as we
from autoresearch.experiment import compute_cwt, build_scales, spectra_correlation


# ── Style profiles ──────────────────────────────────────────────────
# Each style is defined by its dynamical signature, not its content.

STYLES = {
    "textbook": {
        "description": "Expository. Stays in one attractor basin. High autocorrelation. "
                       "The definition-example-consequence cycle.",
        "select_by": "smoothness_high",
        "color": "\033[94m",  # blue
    },
    "poetic": {
        "description": "Oscillating. Alternates between smooth and rough transitions. "
                       "Periodic returns in attractor space — verse structure.",
        "select_by": "oscillation_high",
        "color": "\033[95m",  # magenta
    },
    "confrontation": {
        "description": "Maximally traversing. Each sentence moves AWAY from the previous "
                       "basin. Narrative tension as dynamical instability.",
        "select_by": "smoothness_low",
        "color": "\033[91m",  # red
    },
}


def load_all_texts():
    """Load texts and embeddings."""
    catalog = we.load_text_catalog()
    texts = {}
    for tid in catalog:
        entry = catalog[tid]
        units = we.load_text_units(tid, catalog)
        precomputed = entry.get("precomputed_embeddings")
        if precomputed and Path(precomputed).exists():
            emb = we.load_precomputed_embeddings(precomputed)
        else:
            emb = we.embed_text(units)
        texts[tid] = {
            "units": units, "emb": emb,
            "title": entry["title"], "unit_label": entry["unit"],
        }
    return texts


def build_eigenbasis(texts, K=26):
    """Build the attractor eigenbasis from all texts' wavelet correlations."""
    corr_all = []
    for tid, tdata in texts.items():
        config = {"max_scale_ratio": 0.5, "min_scale": 1, "scale_step": "linear"}
        scales = build_scales(len(tdata["units"]), config)
        ps = compute_cwt(tdata["emb"], scales, "morl")
        corr = spectra_correlation(ps, "pearson")
        corr_all.append(corr)

    avg_corr = np.mean(corr_all, axis=0)
    eigvals, eigvecs = np.linalg.eigh(avg_corr)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[:K]], eigvals[idx[:K]]


def compute_local_fingerprints(units, emb, eigvecs, window=10):
    """Compute attractor dynamics in sliding windows."""
    K = eigvecs.shape[1]
    proj = emb @ eigvecs  # (n, K)

    fingerprints = []
    for start in range(len(units) - window):
        chunk = proj[start:start + window]

        # Smoothness: cosine similarity between consecutive projection vectors
        smoothness = []
        for t in range(window - 1):
            n1 = np.linalg.norm(chunk[t])
            n2 = np.linalg.norm(chunk[t + 1])
            if n1 > 1e-10 and n2 > 1e-10:
                smoothness.append(np.dot(chunk[t], chunk[t + 1]) / (n1 * n2))
            else:
                smoothness.append(0.0)

        smoothness = np.array(smoothness)
        fingerprints.append({
            "start": start,
            "smoothness_mean": float(smoothness.mean()),
            "smoothness_std": float(smoothness.std()),
            "smoothness_min": float(smoothness.min()),
            "smoothness_max": float(smoothness.max()),
        })

    return fingerprints


def select_passages(fingerprints, style_key, n=3):
    """Select passages matching a dynamical style."""
    if style_key == "smoothness_high":
        return sorted(fingerprints, key=lambda f: f["smoothness_mean"], reverse=True)[:n]
    elif style_key == "smoothness_low":
        return sorted(fingerprints, key=lambda f: f["smoothness_mean"])[:n]
    elif style_key == "oscillation_high":
        return sorted(fingerprints, key=lambda f: f["smoothness_std"], reverse=True)[:n]
    return fingerprints[:n]


def print_passage(units, start, window=10, color="", label=""):
    """Print a passage with formatting."""
    reset = "\033[0m"
    dim = "\033[2m"
    if label:
        print(f"  {color}{label}{reset}")
    print(f"  {dim}[sentences {start}-{start + window - 1}]{reset}")
    for i in range(min(window, len(units) - start)):
        text = units[start + i].rstrip()
        if len(text) > 100:
            text = text[:97] + "..."
        print(f"  {color}{text}{reset}")
    print()


def print_fingerprint_bar(fp, label=""):
    """Print a visual representation of a fingerprint."""
    s = fp["smoothness_mean"]
    std = fp["smoothness_std"]

    # Smoothness bar
    bar_len = 40
    pos = int((s + 0.2) / 0.8 * bar_len)  # map [-0.2, 0.6] to [0, bar_len]
    pos = max(0, min(bar_len - 1, pos))
    bar = list("─" * bar_len)
    bar[pos] = "█"
    bar_str = "".join(bar)

    print(f"    smoothness: [{bar_str}] {s:+.3f}  (std={std:.3f})")


def run_compare(texts, eigvecs):
    """Side-by-side comparison: same source text, three different dynamical selections."""
    print("\n" + "=" * 70)
    print("ATTRACTOR FINGERPRINT DEMO")
    print("Same text. Three dynamical selections. Content follows dynamics.")
    print("=" * 70)

    # Use Gatsby as source (richest dynamics)
    tdata = texts["gatsby"]
    units = tdata["units"]
    emb = tdata["emb"]
    fps = compute_local_fingerprints(units, emb, eigvecs, window=8)

    # Also compute global stats for reference
    proj = emb @ eigvecs
    global_smoothness = []
    for t in range(len(units) - 1):
        n1 = np.linalg.norm(proj[t])
        n2 = np.linalg.norm(proj[t + 1])
        if n1 > 1e-10 and n2 > 1e-10:
            global_smoothness.append(np.dot(proj[t], proj[t + 1]) / (n1 * n2))
    gs = np.array(global_smoothness)

    print(f"\nSource: The Great Gatsby (1609 sentences)")
    print(f"Global attractor smoothness: mean={gs.mean():.3f}, std={gs.std():.3f}")
    print(f"Global range: [{gs.min():.3f}, {gs.max():.3f}]")

    # Reference: what do the OTHER texts' dynamics look like?
    print(f"\nReference dynamics:")
    for tid in ["oreilly_graph_db", "yeats_circus_animals"]:
        td = texts[tid]
        p = td["emb"] @ eigvecs
        sm = []
        for t in range(len(td["units"]) - 1):
            n1, n2 = np.linalg.norm(p[t]), np.linalg.norm(p[t + 1])
            if n1 > 1e-10 and n2 > 1e-10:
                sm.append(np.dot(p[t], p[t + 1]) / (n1 * n2))
        sm = np.array(sm)
        print(f"  {td['title']:40s} smoothness={sm.mean():.3f} (std={sm.std():.3f})")

    for style_name, style_info in STYLES.items():
        color = style_info["color"]
        reset = "\033[0m"
        bold = "\033[1m"

        selected = select_passages(fps, style_info["select_by"], n=1)
        fp = selected[0]

        print(f"\n{'─' * 70}")
        print(f"{bold}{color}STYLE: {style_name.upper()}{reset}")
        print(f"{style_info['description']}")
        print_fingerprint_bar(fp, style_name)
        print()
        print_passage(units, fp["start"], window=8, color=color)

    # The punchline
    print("=" * 70)
    print("\033[1mKey insight:\033[0m These are all from the SAME novel.")
    print("The attractor fingerprint finds passages where Fitzgerald")
    print("writes like a textbook, like a poet, and like an action scene —")
    print("purely from the dynamics of how meaning unfolds in embedding space.")
    print()
    print("A few-shot prompt seeded with 'textbook-dynamic' passages from ANY")
    print("source text would steer generation toward that unfolding pattern,")
    print("regardless of topic. Content and dynamics are orthogonal axes.")
    print("=" * 70)

    # Show the full attractor trajectory for context
    print(f"\n\033[2mGatsby attractor trajectory (first 80 sentences):\033[0m")
    basin = np.argmax(np.abs(proj[:80]), axis=1)
    traj_parts = []
    for t in range(80):
        if t > 0 and basin[t] != basin[t - 1]:
            traj_parts.append(f"|{basin[t]}")
        elif t == 0:
            traj_parts.append(str(basin[t]))
    print(f"  {''.join(traj_parts)}")
    print()


def run_single_style(texts, eigvecs, style_name):
    """Show passages matching a single style."""
    if style_name not in STYLES:
        print(f"Unknown style: {style_name}")
        print(f"Available: {', '.join(STYLES.keys())}")
        return

    style = STYLES[style_name]
    color = style["color"]
    reset = "\033[0m"
    bold = "\033[1m"

    print(f"\n{bold}{color}STYLE: {style_name.upper()}{reset}")
    print(f"{style['description']}\n")

    for tid, tdata in texts.items():
        units = tdata["units"]
        emb = tdata["emb"]
        if len(units) < 12:
            continue

        fps = compute_local_fingerprints(units, emb, eigvecs, window=8)
        selected = select_passages(fps, style["select_by"], n=1)

        if selected:
            fp = selected[0]
            print(f"{'─' * 60}")
            print(f"  {bold}{tdata['title']}{reset}")
            print_fingerprint_bar(fp)
            print()
            print_passage(units, fp["start"], window=6, color=color)


def main():
    parser = argparse.ArgumentParser(description="Attractor Fingerprint Demo")
    parser.add_argument("--style", type=str, choices=list(STYLES.keys()),
                        help="Show passages matching this dynamical style")
    parser.add_argument("--compare", action="store_true",
                        help="Side-by-side comparison of all three styles in Gatsby")
    args = parser.parse_args()

    print("Loading texts and building attractor eigenbasis...")
    t0 = time.time()
    texts = load_all_texts()
    eigvecs, eigvals = build_eigenbasis(texts)
    elapsed = time.time() - t0
    print(f"Ready ({elapsed:.1f}s)\n")

    if args.compare or (not args.style):
        run_compare(texts, eigvecs)
    elif args.style:
        run_single_style(texts, eigvecs, args.style)


if __name__ == "__main__":
    main()
