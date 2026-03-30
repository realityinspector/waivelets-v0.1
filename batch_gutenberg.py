#!/usr/bin/env python3
"""
Batch Gutenberg Fingerprint Test
=================================

Pulls diverse texts from Project Gutenberg, fingerprints them all,
and tests whether the 7-number signature generalizes across a broad
corpus of human writing.

Genres covered:
  - Novels (literary fiction, adventure, gothic, romance, satire)
  - Poetry (Romantic, Victorian, Modernist, epic)
  - Philosophy / essays
  - Science / natural history
  - Drama / plays
  - Religious / mythological texts
  - Political / speeches

Usage:
  python batch_gutenberg.py                  # run full batch
  python batch_gutenberg.py --limit 5        # quick test with 5 texts
  python batch_gutenberg.py --output results.json
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fastprint import (fingerprint, classify, explain, Fingerprint, split_sentences,
                       MODE_PROFILES)


# ── Gutenberg corpus ────────────────────────────────────────────────
# Curated for genre diversity. Each entry: (id, title, author, genre, expected_class)

# Each entry: (gutenberg_id, title, author, genre_label, expected_mode)
# genre_label = traditional genre (informational)
# expected_mode = dynamical structural mode (what we're testing)
#   convergent: narrows and holds — liturgical, aphoristic, legalistic
#   contemplative: thinks about thinking — analytical, reflective
#   discursive: wide river of prose — narrative, argument, epic
#   dialectical: oscillates between registers — dramatic, contrastive

CORPUS = [
    # ── Novels ──
    (1342, "Pride and Prejudice", "Austen", "novel", "discursive"),
    (11, "Alice in Wonderland", "Carroll", "novel", "discursive"),
    (84, "Frankenstein", "Shelley", "novel", "discursive"),
    (1400, "Great Expectations", "Dickens", "novel", "dialectical"),
    (2701, "Moby Dick", "Melville", "novel", "dialectical"),
    (174, "The Picture of Dorian Gray", "Wilde", "novel", "discursive"),
    (1661, "Sherlock Holmes", "Doyle", "novel", "discursive"),
    (98, "A Tale of Two Cities", "Dickens", "novel", "discursive"),
    (345, "Dracula", "Stoker", "novel", "discursive"),
    (16328, "Dubliners", "Joyce", "novel", "discursive"),
    (1952, "The Yellow Wallpaper", "Gilman", "novel", "discursive"),
    (76, "Huckleberry Finn", "Twain", "novel", "discursive"),
    (120, "Treasure Island", "Stevenson", "novel", "discursive"),
    (36, "War of the Worlds", "Wells", "novel", "discursive"),
    (244, "A Study in Scarlet", "Doyle", "novel", "discursive"),
    (1260, "Jane Eyre", "Bronte", "novel", "discursive"),
    (219, "Heart of Darkness", "Conrad", "novel", "discursive"),
    (766, "David Copperfield", "Dickens", "novel", "discursive"),
    (5200, "Metamorphosis", "Kafka", "novel", "discursive"),
    (4300, "Ulysses", "Joyce", "novel", None),  # genuinely unknown — let's see

    # ── Poetry ──
    (1321, "Paradise Lost", "Milton", "poetry", "discursive"),
    (4800, "Leaves of Grass", "Whitman", "poetry", "discursive"),
    (8800, "Divine Comedy: Inferno", "Dante (tr.)", "poetry", "discursive"),
    (1065, "Poems", "Dickinson", "poetry", "convergent"),
    (1524, "Hamlet", "Shakespeare", "drama", "dialectical"),
    (1533, "Macbeth", "Shakespeare", "drama", "dialectical"),
    (1515, "Romeo and Juliet", "Shakespeare", "drama", None),
    (1519, "The Tempest", "Shakespeare", "drama", None),
    (23042, "The Importance of Being Earnest", "Wilde", "drama", "dialectical"),

    # ── Philosophy / Essays ──
    (1232, "The Prince", "Machiavelli", "philosophy", "discursive"),
    (1497, "The Republic", "Plato", "philosophy", "discursive"),
    (5827, "The Problems of Philosophy", "Russell", "philosophy", "contemplative"),
    (1080, "A Modest Proposal", "Swift", "essay", "contemplative"),
    (7370, "Second Treatise of Government", "Locke", "philosophy", "convergent"),
    (5740, "Tractatus Logico-Philosophicus", "Wittgenstein", "philosophy", None),
    (4280, "Critique of Pure Reason", "Kant", "philosophy", None),
    (3600, "Thus Spake Zarathustra", "Nietzsche", "philosophy", None),
    (7205, "Leviathan", "Hobbes", "philosophy", None),

    # ── Science ──
    (2009, "Origin of Species", "Darwin", "science", "dialectical"),
    (14725, "Descent of Man", "Darwin", "science", None),

    # ── Religious / Mythological ──
    (8001, "Bhagavad Gita", "Vyasa (tr.)", "religious", "convergent"),
    (8300, "Tao Te Ching", "Lao Tzu (tr.)", "religious", None),
    (10, "Bible (KJV)", "Various", "religious", None),

    # ── Political ──
    (815, "Democracy in America (v1)", "Tocqueville", "political", "discursive"),
    (61, "Communist Manifesto", "Marx/Engels", "political", None),

    # ── Misc ──
    (1184, "The Count of Monte Cristo", "Dumas", "novel", None),
    (2600, "War and Peace", "Tolstoy", "novel", None),
    (28054, "Brothers Karamazov", "Dostoevsky", "novel", None),
    (2554, "Crime and Punishment", "Dostoevsky", "novel", None),
    (74, "Tom Sawyer", "Twain", "novel", None),
]


def fetch_gutenberg(gid: int, max_chars: int = 80000) -> str | None:
    """Fetch plain text from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "waivelets-research/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8-sig", errors="replace")
    except Exception as e:
        print(f"  FETCH FAILED: {e}")
        return None

    # Strip Gutenberg header/footer
    start_markers = ["*** START OF THE PROJECT GUTENBERG",
                     "*** START OF THIS PROJECT GUTENBERG",
                     "***START OF THE PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG",
                   "*** END OF THIS PROJECT GUTENBERG",
                   "***END OF THE PROJECT GUTENBERG",
                   "End of the Project Gutenberg",
                   "End of Project Gutenberg"]

    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the end of the marker line
            nl = text.find("\n", idx)
            if nl != -1:
                text = text[nl + 1:]
            break

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


def run_batch(limit=None, output_path=None):
    corpus = CORPUS[:limit] if limit else CORPUS

    print("=" * 70)
    print("BATCH GUTENBERG FINGERPRINT TEST")
    print(f"Testing {len(corpus)} texts across {len(set(c[3] for c in corpus))} genres")
    print("=" * 70)

    results = []
    genre_fps = {}  # genre -> list of fingerprints
    correct = 0
    total = 0

    for i, (gid, title, author, genre, expected) in enumerate(corpus):
        print(f"\n[{i+1}/{len(corpus)}] {title} ({author}) — {genre}")

        # Fetch
        t0 = time.time()
        text = fetch_gutenberg(gid)
        fetch_time = time.time() - t0

        if not text:
            results.append({"gid": gid, "title": title, "author": author,
                           "genre": genre, "error": "fetch_failed"})
            continue

        sentences = split_sentences(text)
        if len(sentences) < 10:
            print(f"  Only {len(sentences)} sentences — skipping")
            results.append({"gid": gid, "title": title, "author": author,
                           "genre": genre, "error": f"too_few_sentences ({len(sentences)})"})
            continue

        # Cap at 500 sentences for speed (still representative)
        if len(sentences) > 500:
            sentences = sentences[:500]

        print(f"  {len(sentences)} sentences (fetched in {fetch_time:.1f}s)")

        # Fingerprint
        t0 = time.time()
        try:
            fp = fingerprint("", sentences=sentences)
        except Exception as e:
            print(f"  FINGERPRINT FAILED: {e}")
            results.append({"gid": gid, "title": title, "author": author,
                           "genre": genre, "error": str(e)})
            continue
        fp_time = time.time() - t0

        # Classify
        label, dist = classify(fp)
        if expected is None:
            match = "NEW"
            # Don't count unknown expected modes in accuracy
        elif label == expected:
            match = "OK"
            correct += 1
            total += 1
        else:
            match = "MISS"
            total += 1

        print(f"  Fingerprint: {fp_time*1000:.0f}ms")
        exp_str = expected or "?"
        print(f"  Mode: {label} (dist={dist:.2f}) "
              f"[expected: {exp_str}] {match}")
        print(f"  smooth={fp.smoothness_mean:+.3f}  osc={fp.smoothness_std:.3f}  "
              f"expo={fp.exposition:.4f}  inter={fp.interiority:.4f}  "
              f"formal={fp.formal_structure:.4f}  scene={fp.scene_narration:.4f}  "
              f"entropy={fp.basin_entropy:.2f}")

        result = {
            "gid": gid, "title": title, "author": author,
            "genre": genre, "expected": expected,
            "classified_as": label, "distance": dist,
            "match": label == expected if expected else None,
            "fingerprint": fp.to_dict(),
            "n_sentences": len(sentences),
            "time_ms": round(fp_time * 1000, 1),
        }
        results.append(result)

        if genre not in genre_fps:
            genre_fps[genre] = []
        genre_fps[genre].append(fp)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total: {total}  Correct: {correct}  Accuracy: {correct/max(total,1)*100:.1f}%")

    # Per-genre averages
    print(f"\n  Genre averages:")
    print(f"  {'Genre':<14s} {'N':>3s}  {'Smooth':>7s}  {'Oscil':>6s}  {'Expo':>6s}  "
          f"{'Inter':>6s}  {'Formal':>6s}  {'Scene':>6s}  {'Entropy':>7s}")
    print(f"  {'─'*14} {'─'*3}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")

    for genre in sorted(genre_fps.keys()):
        fps = genre_fps[genre]
        n = len(fps)
        avg = Fingerprint(*(np.mean([f[i] for f in fps]) for i in range(7)))
        print(f"  {genre:<14s} {n:3d}  {avg.smoothness_mean:+.3f}  {avg.smoothness_std:.3f}  "
              f"{avg.exposition:.4f}  {avg.interiority:.4f}  {avg.formal_structure:.4f}  "
              f"{avg.scene_narration:.4f}  {avg.basin_entropy:.2f}")

    # Mode distribution by genre
    print(f"\n  Mode distribution by genre:")
    genres_seen = sorted(set(r.get("genre", "") for r in results if "classified_as" in r))

    for genre in genres_seen:
        genre_results = [r for r in results if r.get("genre") == genre and "classified_as" in r]
        if not genre_results:
            continue
        counts = {}
        for r in genre_results:
            c = r["classified_as"]
            counts[c] = counts.get(c, 0) + 1
        dist_str = ", ".join(f"{c}: {n}" for c, n in sorted(counts.items()))
        print(f"    {genre:<14s} → {dist_str}")

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump({"results": results, "summary": {
                "total": total, "correct": correct,
                "accuracy": correct / max(total, 1),
            }}, f, indent=2, default=str)
        print(f"\n  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Gutenberg fingerprint test")
    parser.add_argument("--limit", type=int, help="Limit number of texts")
    parser.add_argument("--output", "-o", default="batch_results.json",
                        help="Output JSON path")
    args = parser.parse_args()
    run_batch(limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()
