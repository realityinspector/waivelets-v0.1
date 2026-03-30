#!/usr/bin/env python3
"""
Waivelets Autoresearch Loop

Karpathy-style iterative search for the analysis configuration that
maximally reveals self-similar structure in text embeddings.

Primary metric: significance_score — average |z-score| across topology,
fractal exponent, graph density, and cross-scale self-similarity,
comparing actual text against shuffled baselines.

Higher = this config finds more real structure that survives shuffling.

Usage:
  # Quick test (3 iterations, Yeats only):
  python -m autoresearch.run --iterations 3 --text yeats_circus_animals

  # Full sweep (50 iterations, all texts):
  python -m autoresearch.run --iterations 50

  # Specific cluster:
  python -m autoresearch.run --cluster wavelet --iterations 20

  # Resume from previous results:
  python -m autoresearch.run --iterations 50 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure parent dir is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config_space import ConfigSpace, ALL_CLUSTERS
from autoresearch.experiment import run_experiment
import wavelet_engine as we

RESULTS_DIR = Path(__file__).parent / "results"


def load_texts():
    """Load all built-in texts and their embeddings."""
    catalog = we.load_text_catalog()
    texts = {}
    for text_id, entry in catalog.items():
        units = we.load_text_units(text_id, catalog)
        precomputed = entry.get("precomputed_embeddings")
        if precomputed and Path(precomputed).exists():
            emb = we.load_precomputed_embeddings(precomputed)
        else:
            emb = we.embed_text(units)
        texts[text_id] = {
            "units": units,
            "embeddings": emb,
            "title": entry["title"],
            "unit_label": entry["unit"],
        }
    return texts


def load_single_text(text_id):
    """Load one text."""
    catalog = we.load_text_catalog()
    entry = catalog[text_id]
    units = we.load_text_units(text_id, catalog)
    precomputed = entry.get("precomputed_embeddings")
    if precomputed and Path(precomputed).exists():
        emb = we.load_precomputed_embeddings(precomputed)
    else:
        emb = we.embed_text(units)
    return {text_id: {
        "units": units, "embeddings": emb,
        "title": entry["title"], "unit_label": entry["unit"],
    }}


def run_loop(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load texts
    print("Loading texts and embeddings...")
    if args.text:
        texts = load_single_text(args.text)
    elif args.skip_gatsby:
        catalog = we.load_text_catalog()
        texts = {}
        for tid in catalog:
            if tid == "gatsby":
                continue
            texts.update(load_single_text(tid))
    else:
        texts = load_texts()

    print(f"  Loaded {len(texts)} texts: {', '.join(texts.keys())}")

    # Config space
    clusters = [args.cluster] if args.cluster else None
    space = ConfigSpace(clusters=clusters, seed=args.seed)

    # Results log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"run_{timestamp}.jsonl"
    summary_path = RESULTS_DIR / f"summary_{timestamp}.json"

    # Load previous best if resuming
    best_score = 0.0
    best_config = None
    best_results = []

    if args.resume:
        prev = sorted(RESULTS_DIR.glob("summary_*.json"))
        if prev:
            with open(prev[-1]) as f:
                prev_data = json.load(f)
            best_score = prev_data.get("best_significance_score", 0)
            best_config = prev_data.get("best_config")
            print(f"  Resuming from {prev[-1].name}, best score: {best_score:.3f}")

    # Baseline: run with default config first
    print("\n--- Baseline (default config) ---")
    default_config = {
        "wavelet": "morl", "max_scale_ratio": 0.5, "min_scale": 1,
        "scale_step": "linear", "correlation_threshold": 0.5,
        "correlation_method": "pearson", "fractal_method": "polyfit",
        "fractal_fit_range_low": 0.0, "fractal_fit_range_high": 1.0,
        "shuffle_method": "permute_rows", "baseline_trials": 5,
    }
    for tid, tdata in texts.items():
        result = run_experiment(tdata["embeddings"], default_config)
        if result.get("error"):
            print(f"  {tid}: ERROR — {result['error']}")
        else:
            print(f"  {tid}: significance={result['significance_score']:.3f}  "
                  f"z_fractal={result['z_fractal']:+.2f}  "
                  f"fractal={result['fractal_exponent']:.3f}  "
                  f"time={result['time_s']:.1f}s")
            with open(log_path, "a") as f:
                entry = {"iteration": 0, "text_id": tid,
                         "config": default_config, **{k: v for k, v in result.items()
                                                       if k != "config"}}
                f.write(json.dumps(entry, default=str) + "\n")

    # Main loop
    print(f"\n{'='*60}")
    print(f"Starting autoresearch ({args.iterations} iterations)")
    print(f"Clusters: {clusters or 'all'}")
    print(f"Texts: {list(texts.keys())}")
    print(f"{'='*60}\n")

    n_improved = 0
    n_failed = 0
    all_scores = []

    for i in range(1, args.iterations + 1):
        config = space.sample()
        # Merge with defaults so all keys exist
        full_config = {**default_config, **config}

        mutated_keys = list(config.keys())
        print(f"[{i}/{args.iterations}] Mutated: {mutated_keys}")
        for k, v in config.items():
            print(f"    {k} = {v}")

        scores = []
        for tid, tdata in texts.items():
            result = run_experiment(tdata["embeddings"], full_config)
            if result.get("error"):
                print(f"  {tid}: ERROR — {result['error']}")
                n_failed += 1
                continue

            sig = result["significance_score"]
            scores.append(sig)
            status = ""
            if sig > best_score:
                status = " *** NEW BEST ***"
                best_score = sig
                best_config = full_config.copy()
                n_improved += 1

            print(f"  {tid}: significance={sig:.3f}  "
                  f"z_fractal={result['z_fractal']:+.2f}  "
                  f"fractal={result['fractal_exponent']:.3f}  "
                  f"time={result['time_s']:.1f}s{status}")

            with open(log_path, "a") as f:
                entry = {"iteration": i, "text_id": tid,
                         "config": full_config,
                         **{k: v for k, v in result.items() if k != "config"}}
                f.write(json.dumps(entry, default=str) + "\n")

        if scores:
            avg = np.mean(scores)
            all_scores.append(avg)
            print(f"  Avg significance: {avg:.3f}  "
                  f"(best so far: {best_score:.3f})\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"  Iterations: {args.iterations}")
    print(f"  Improvements: {n_improved}")
    print(f"  Failures: {n_failed}")
    print(f"  Best significance score: {best_score:.3f}")
    if best_config:
        print(f"  Best config:")
        for k, v in sorted(best_config.items()):
            print(f"    {k}: {v}")
    print(f"{'='*60}")

    summary = {
        "timestamp": timestamp,
        "iterations": args.iterations,
        "n_improved": n_improved,
        "n_failed": n_failed,
        "best_significance_score": best_score,
        "best_config": best_config,
        "all_scores": all_scores,
        "texts_used": list(texts.keys()),
        "clusters": clusters,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults: {log_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Waivelets Autoresearch")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of experiment iterations")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text ID to test (e.g. yeats_circus_animals)")
    parser.add_argument("--skip-gatsby", action="store_true",
                        help="Skip Gatsby (slow)")
    parser.add_argument("--cluster", type=str, default=None,
                        choices=list(ALL_CLUSTERS.keys()),
                        help="Restrict mutations to one cluster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous best")
    args = parser.parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
