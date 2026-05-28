#!/usr/bin/env python3
"""Extract the 26-D attractor coordinates for each generation using local fastprint.
Writes /tmp/exp/coords26.jsonl with per-generation aggregates.
"""
import json, pathlib, sys, re, time
import numpy as np
sys.path.insert(0, "/tmp/waivelets-v0.1")
from fastprint import _load_basis, _get_model, split_sentences

RESULTS = pathlib.Path("/tmp/exp/results.jsonl")
OUT = pathlib.Path("/tmp/exp/coords26.jsonl")

print("Loading basis + model…", file=sys.stderr, flush=True)
eigvecs, clusters = _load_basis()
model = _get_model()
print(f"eigvecs: {eigvecs.shape}", file=sys.stderr)

records = [json.loads(l) for l in RESULTS.open()]
print(f"Records: {len(records)}", file=sys.stderr)

# Resume
done = set()
if OUT.exists():
    for line in OUT.open():
        try:
            r = json.loads(line)
            done.add((r["model"], r["prompt_idx"]))
        except Exception:
            pass

with OUT.open("a") as fout:
    for i, r in enumerate(records):
        key = (r["model"], r["prompt_idx"])
        if key in done:
            continue
        text = r["text"]
        sentences = split_sentences(text)
        if len(sentences) < 3:
            print(f"[{i}] skip (n={len(sentences)})", file=sys.stderr)
            continue
        t0 = time.time()
        emb = model.encode(sentences, batch_size=64, show_progress_bar=False)  # (n, 384)
        proj = emb @ eigvecs  # (n, 26)
        # Aggregates
        coord_mean = proj.mean(axis=0)
        coord_std = proj.std(axis=0)
        coord_mean_abs = np.abs(proj).mean(axis=0)
        # Basin visit distribution: fraction of sentences with argmax(|proj|) == basin j
        basin = np.argmax(np.abs(proj), axis=1)
        basin_hist = np.bincount(basin, minlength=eigvecs.shape[1]).astype(float)
        basin_hist /= basin_hist.sum()
        # Embedding mean (384-D)
        emb_mean = emb.mean(axis=0)
        rec = {
            "model": r["model"], "prompt_idx": r["prompt_idx"],
            "n_sentences": len(sentences),
            "coord_mean": coord_mean.tolist(),
            "coord_std": coord_std.tolist(),
            "coord_mean_abs": coord_mean_abs.tolist(),
            "basin_hist": basin_hist.tolist(),
            "emb_mean": emb_mean.tolist(),
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        print(f"[{i+1}/{len(records)}] {r['model']:42s} #{r['prompt_idx']:2d}  n={len(sentences):3d}  {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

print("DONE", file=sys.stderr)
