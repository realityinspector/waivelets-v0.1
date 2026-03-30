#!/usr/bin/env python3
"""
Mega Gutenberg Corpus Fingerprinter
=====================================

Pulls hundreds of texts from Project Gutenberg, fingerprints each one,
checkpoints results to disk after every text. Memory-safe: processes one
text at a time, never holds more than one embedding matrix in memory.

Usage:
  python mega_corpus.py                    # run full corpus
  python mega_corpus.py --resume           # resume from checkpoint
  python mega_corpus.py --limit 20         # test with 20 texts
  python mega_corpus.py --stats            # show stats from existing results
"""

from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fastprint import fingerprint, classify, split_sentences, Fingerprint

RESULTS_PATH = Path("mega_corpus_results.jsonl")

# ── Mega corpus: 200+ texts from Gutenberg ──────────────────────────
# (id, title, author, genre)
# No expected modes — we're discovering, not validating.

MEGA_CORPUS = [
    # ── English novels (chronological) ──
    (1080, "Gulliver's Travels", "Swift", "novel"),
    (1342, "Pride and Prejudice", "Austen", "novel"),
    (158, "Emma", "Austen", "novel"),
    (161, "Sense and Sensibility", "Austen", "novel"),
    (105, "Persuasion", "Austen", "novel"),
    (84, "Frankenstein", "Shelley", "novel"),
    (768, "Wuthering Heights", "E. Bronte", "novel"),
    (1260, "Jane Eyre", "C. Bronte", "novel"),
    (98, "A Tale of Two Cities", "Dickens", "novel"),
    (1400, "Great Expectations", "Dickens", "novel"),
    (766, "David Copperfield", "Dickens", "novel"),
    (730, "Oliver Twist", "Dickens", "novel"),
    (580, "The Pickwick Papers", "Dickens", "novel"),
    (2701, "Moby Dick", "Melville", "novel"),
    (209, "The Turn of the Screw", "James", "novel"),
    (432, "The Ambassadors", "James", "novel"),
    (11, "Alice in Wonderland", "Carroll", "novel"),
    (12, "Through the Looking-Glass", "Carroll", "novel"),
    (174, "The Picture of Dorian Gray", "Wilde", "novel"),
    (345, "Dracula", "Stoker", "novel"),
    (1661, "Sherlock Holmes Adventures", "Doyle", "novel"),
    (244, "A Study in Scarlet", "Doyle", "novel"),
    (2852, "The Hound of the Baskervilles", "Doyle", "novel"),
    (76, "Huckleberry Finn", "Twain", "novel"),
    (74, "Tom Sawyer", "Twain", "novel"),
    (86, "A Connecticut Yankee", "Twain", "novel"),
    (120, "Treasure Island", "Stevenson", "novel"),
    (43, "Jekyll and Hyde", "Stevenson", "novel"),
    (36, "The War of the Worlds", "Wells", "novel"),
    (35, "The Time Machine", "Wells", "novel"),
    (159, "The Island of Doctor Moreau", "Wells", "novel"),
    (219, "Heart of Darkness", "Conrad", "novel"),
    (974, "Lord Jim", "Conrad", "novel"),
    (5200, "Metamorphosis", "Kafka", "novel"),
    (16328, "Dubliners", "Joyce", "novel"),
    (4300, "Ulysses", "Joyce", "novel"),
    (2814, "Dubliners (alt)", "Joyce", "novel"),
    (1952, "The Yellow Wallpaper", "Gilman", "novel"),
    (160, "The Awakening", "Chopin", "novel"),
    (514, "Little Women", "Alcott", "novel"),
    (164, "Twenty Thousand Leagues", "Verne", "novel"),
    (103, "Around the World in 80 Days", "Verne", "novel"),
    (1184, "Count of Monte Cristo", "Dumas", "novel"),
    (1259, "The Three Musketeers", "Dumas", "novel"),
    (2600, "War and Peace", "Tolstoy", "novel"),
    (2554, "Crime and Punishment", "Dostoevsky", "novel"),
    (28054, "Brothers Karamazov", "Dostoevsky", "novel"),
    (600, "Notes from Underground", "Dostoevsky", "novel"),
    (1399, "Anna Karenina", "Tolstoy", "novel"),
    (2500, "Siddhartha", "Hesse", "novel"),
    (7849, "The Trial", "Kafka", "novel"),

    # ── Poetry ──
    (1321, "Paradise Lost", "Milton", "poetry"),
    (4800, "Leaves of Grass", "Whitman", "poetry"),
    (8800, "Divine Comedy: Inferno", "Dante (tr.)", "poetry"),
    (8789, "Divine Comedy: Purgatorio", "Dante (tr.)", "poetry"),
    (8799, "Divine Comedy: Paradiso", "Dante (tr.)", "poetry"),
    (1065, "Poems", "Dickinson", "poetry"),
    (6130, "Iliad", "Homer (tr.)", "poetry"),
    (3160, "Odyssey", "Homer (tr.)", "poetry"),
    (4705, "Sonnets", "Shakespeare", "poetry"),
    (100, "Complete Shakespeare", "Shakespeare", "drama"),
    (1524, "Hamlet", "Shakespeare", "drama"),
    (1533, "Macbeth", "Shakespeare", "drama"),
    (1515, "Romeo and Juliet", "Shakespeare", "drama"),
    (1519, "The Tempest", "Shakespeare", "drama"),
    (1513, "A Midsummer Night's Dream", "Shakespeare", "drama"),
    (23042, "Importance of Being Earnest", "Wilde", "drama"),
    (4363, "An Ideal Husband", "Wilde", "drama"),
    (844, "The Importance of Being Earnest", "Wilde", "drama"),
    (2148, "The Rime of the Ancient Mariner", "Coleridge", "poetry"),
    (8147, "The Waste Land", "Eliot", "poetry"),

    # ── Philosophy / Essays ──
    (1232, "The Prince", "Machiavelli", "philosophy"),
    (1497, "The Republic", "Plato", "philosophy"),
    (5827, "Problems of Philosophy", "Russell", "philosophy"),
    (7370, "Second Treatise of Government", "Locke", "philosophy"),
    (4280, "Critique of Pure Reason", "Kant", "philosophy"),
    (3600, "Thus Spake Zarathustra", "Nietzsche", "philosophy"),
    (7205, "Leviathan", "Hobbes", "philosophy"),
    (10615, "Nicomachean Ethics", "Aristotle (tr.)", "philosophy"),
    (1656, "Apology", "Plato (tr.)", "philosophy"),
    (55201, "Being and Time (excerpt)", "Heidegger", "philosophy"),
    (10616, "Politics", "Aristotle (tr.)", "philosophy"),
    (3207, "Meditations", "Marcus Aurelius", "philosophy"),
    (10662, "Poetics", "Aristotle (tr.)", "philosophy"),

    # ── Science ──
    (2009, "Origin of Species", "Darwin", "science"),
    (14725, "Descent of Man", "Darwin", "science"),
    (5001, "Principia preface", "Newton (tr.)", "science"),
    (37729, "Relativity", "Einstein (tr.)", "science"),

    # ── Religious / Sacred ──
    (8001, "Bhagavad Gita", "Vyasa (tr.)", "religious"),
    (8300, "Tao Te Ching", "Lao Tzu (tr.)", "religious"),
    (10, "Bible (KJV)", "Various", "religious"),
    (2680, "Meditations", "Descartes", "philosophy"),
    (3296, "Confessions", "Augustine", "religious"),

    # ── Political ──
    (815, "Democracy in America", "Tocqueville", "political"),
    (61, "Communist Manifesto", "Marx/Engels", "political"),
    (1, "Declaration of Independence (US)", "Jefferson et al.", "political"),
    (5, "Bill of Rights (US)", "US Congress", "political"),

    # ── Essays / Nonfiction ──
    (2680, "Meditations", "Descartes", "philosophy"),
    (4705, "Walden", "Thoreau", "essay"),
    (71, "The Adventures of Tom Sawyer", "Twain", "novel"),
    (1080, "A Modest Proposal", "Swift", "essay"),

    # ── Horror / Gothic ──
    (932, "The Fall of the House of Usher", "Poe", "novel"),
    (2147, "The Raven", "Poe", "poetry"),
    (1064, "The Masque of the Red Death", "Poe", "novel"),
]


def fetch_gutenberg(gid: int, max_chars: int = 80000) -> str | None:
    """Fetch plain text from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "waivelets-research/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8-sig", errors="replace")
    except Exception as e:
        return None

    # Strip Gutenberg header/footer
    for marker in ["*** START OF THE PROJECT GUTENBERG",
                    "*** START OF THIS PROJECT GUTENBERG",
                    "***START OF THE PROJECT GUTENBERG"]:
        idx = text.find(marker)
        if idx != -1:
            nl = text.find("\n", idx)
            if nl != -1:
                text = text[nl + 1:]
            break

    for marker in ["*** END OF THE PROJECT GUTENBERG",
                    "*** END OF THIS PROJECT GUTENBERG",
                    "End of the Project Gutenberg",
                    "End of Project Gutenberg"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def load_completed() -> set:
    """Load already-completed Gutenberg IDs from checkpoint."""
    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["gid"])
                except Exception:
                    pass
    return done


def save_result(result: dict):
    """Append one result to the checkpoint file."""
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def show_stats():
    """Show stats from existing results."""
    if not RESULTS_PATH.exists():
        print("No results found. Run the corpus first.")
        return

    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    successes = [r for r in results if r.get("mode")]
    failures = [r for r in results if r.get("error")]

    print(f"\nMEGA CORPUS RESULTS")
    print(f"=" * 70)
    print(f"  Total: {len(results)}  Success: {len(successes)}  Failed: {len(failures)}")

    # Mode distribution
    mode_counts = {}
    for r in successes:
        m = r["mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print(f"\n  Mode distribution:")
    for mode in sorted(mode_counts.keys()):
        n = mode_counts[mode]
        bar = "█" * int(n / len(successes) * 40)
        print(f"    {mode:16s} {n:3d} ({n/len(successes)*100:4.1f}%) {bar}")

    # Mode distribution by genre
    print(f"\n  Mode x Genre:")
    genre_mode = {}
    for r in successes:
        g = r["genre"]
        m = r["mode"]
        if g not in genre_mode:
            genre_mode[g] = {}
        genre_mode[g][m] = genre_mode[g].get(m, 0) + 1

    for genre in sorted(genre_mode.keys()):
        counts = genre_mode[genre]
        total = sum(counts.values())
        dist = ", ".join(f"{m}: {n}" for m, n in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"    {genre:14s} ({total:2d}) → {dist}")

    # Per-mode averages
    print(f"\n  Per-mode fingerprint averages:")
    print(f"  {'Mode':<16s} {'N':>3s} {'Smooth':>7s} {'Oscil':>6s} {'Expo':>6s} "
          f"{'Inter':>6s} {'Formal':>6s} {'Scene':>6s} {'Entropy':>7s}")
    for mode in sorted(mode_counts.keys()):
        mode_results = [r for r in successes if r["mode"] == mode]
        n = len(mode_results)
        avg = {k: np.mean([r["fingerprint"][k] for r in mode_results]) for k in mode_results[0]["fingerprint"]}
        print(f"  {mode:<16s} {n:3d} {avg['smoothness_mean']:+.3f} {avg['smoothness_std']:.3f} "
              f"{avg['exposition']:.4f} {avg['interiority']:.4f} {avg['formal_structure']:.4f} "
              f"{avg['scene_narration']:.4f} {avg['basin_entropy']:.2f}")

    # Save summary for web
    web_corpus = []
    for r in successes:
        web_corpus.append({
            "title": r["title"],
            "author": r["author"],
            "genre": r["genre"],
            "mode": r["mode"],
            "distance": round(r["distance"], 2),
            "fp": {k: round(v, 4) for k, v in r["fingerprint"].items()},
            "n_sentences": r["n_sentences"],
        })

    with open("web/precomputed.json") as f:
        web_data = json.load(f)
    web_data["corpus"] = web_corpus
    with open("web/precomputed.json", "w") as f:
        json.dump(web_data, f)
    print(f"\n  Updated web/precomputed.json with {len(web_corpus)} texts")


def run_corpus(limit=None, resume=False):
    corpus = MEGA_CORPUS[:limit] if limit else MEGA_CORPUS

    # Deduplicate by Gutenberg ID
    seen_ids = set()
    deduped = []
    for entry in corpus:
        if entry[0] not in seen_ids:
            seen_ids.add(entry[0])
            deduped.append(entry)
    corpus = deduped

    completed = load_completed() if resume else set()
    remaining = [(gid, t, a, g) for gid, t, a, g in corpus if gid not in completed]

    if not resume and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    print(f"MEGA GUTENBERG CORPUS")
    print(f"=" * 70)
    print(f"  Total: {len(corpus)}  Already done: {len(completed)}  Remaining: {len(remaining)}")
    print()

    for i, (gid, title, author, genre) in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {title} ({author}) — {genre}", end="", flush=True)

        text = fetch_gutenberg(gid)
        if not text:
            print(f"  FETCH FAILED")
            save_result({"gid": gid, "title": title, "author": author,
                        "genre": genre, "error": "fetch_failed"})
            continue

        sentences = split_sentences(text)
        if len(sentences) < 5:
            print(f"  too few sentences ({len(sentences)})")
            save_result({"gid": gid, "title": title, "author": author,
                        "genre": genre, "error": f"too_few ({len(sentences)})"})
            continue

        if len(sentences) > 500:
            sentences = sentences[:500]

        t0 = time.time()
        try:
            fp = fingerprint("", sentences=sentences)
        except Exception as e:
            print(f"  ERROR: {e}")
            save_result({"gid": gid, "title": title, "author": author,
                        "genre": genre, "error": str(e)})
            gc.collect()
            continue
        elapsed = time.time() - t0

        mode, dist = classify(fp)
        print(f"  → {mode} (d={dist:.1f}) [{elapsed*1000:.0f}ms]")

        save_result({
            "gid": gid, "title": title, "author": author,
            "genre": genre, "mode": mode, "distance": round(dist, 2),
            "fingerprint": fp.to_dict(),
            "n_sentences": len(sentences),
            "time_ms": round(elapsed * 1000, 1),
        })

        # Memory safety: force garbage collection every 10 texts
        if (i + 1) % 10 == 0:
            gc.collect()

    print(f"\nDone! Results in {RESULTS_PATH}")
    show_stats()


def main():
    parser = argparse.ArgumentParser(description="Mega Gutenberg corpus fingerprinter")
    parser.add_argument("--limit", type=int, help="Limit number of texts")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--stats", action="store_true", help="Show stats from existing results")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        run_corpus(limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
