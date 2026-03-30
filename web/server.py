"""
Waivelets Web — Interactive blog post + live fingerprinting API.

Lightweight FastAPI server:
  - Serves the static HTML blog post
  - /api/fingerprint endpoint for live text analysis
  - Preloads the embedding model on startup
"""

import json
import os
import time
import html as html_lib
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from fastprint import (
    fingerprint, classify, split_sentences, Fingerprint,
    MODE_PROFILES, _CORPUS_MEAN, _CORPUS_STD, _get_model, _load_basis
)

app = FastAPI(title="Waivelets", docs_url=None, redoc_url=None)

WEB_DIR = Path(__file__).parent

# Load AI detector params
_DETECTOR = None
_detector_path = WEB_DIR.parent / "ai_detector_params.json"
if _detector_path.exists():
    with open(_detector_path) as f:
        _DETECTOR = json.load(f)
    _DET_MEAN = np.array(_DETECTOR["z_mean"])
    _DET_STD = np.array(_DETECTOR["z_std"])
    _DET_WEIGHTS = np.array(_DETECTOR["weights"])
    _DET_THRESH = _DETECTOR["threshold"]
    _DET_AI_CENT = np.array(_DETECTOR["centroid"]["ai"])
    _DET_HUM_CENT = np.array(_DETECTOR["centroid"]["human"])


@app.on_event("startup")
async def warmup():
    """Preload model and basis on startup."""
    print("Warming up model...")
    t0 = time.time()
    _load_basis()
    model = _get_model()
    model.encode(["warmup"], batch_size=1, show_progress_bar=False)
    print(f"Model ready in {time.time() - t0:.1f}s")


class TextInput(BaseModel):
    text: str


@app.post("/api/fingerprint")
async def api_fingerprint(body: TextInput):
    """Fingerprint arbitrary text. Returns mode + 7-number signature."""
    text = body.text.strip()
    if len(text) < 20:
        return JSONResponse({"error": "Text too short (need at least a few sentences)"}, 400)

    # Sanitize — strip HTML, limit length
    text = html_lib.escape(text)
    if len(text) > 50000:
        text = text[:50000]

    sentences = split_sentences(text)
    if len(sentences) < 3:
        return JSONResponse({"error": f"Only found {len(sentences)} sentences. Need at least 3."}, 400)

    # Cap at 500 sentences for safety
    if len(sentences) > 500:
        sentences = sentences[:500]

    t0 = time.time()
    fp = fingerprint("", sentences=sentences)
    fp_time = time.time() - t0

    mode, dist = classify(fp)
    profile = MODE_PROFILES[mode]

    # Compute all mode distances
    fp_z = (fp.to_array() - _CORPUS_MEAN) / _CORPUS_STD
    mode_distances = {}
    for name, prof in MODE_PROFILES.items():
        ref_z = (prof["centroid"].to_array() - _CORPUS_MEAN) / _CORPUS_STD
        mode_distances[name] = round(float(np.linalg.norm(fp_z - ref_z)), 2)

    return {
        "mode": mode,
        "distance": round(dist, 2),
        "description": profile["description"],
        "fingerprint": {k: round(v, 4) for k, v in fp.to_dict().items()},
        "mode_distances": mode_distances,
        "n_sentences": len(sentences),
        "time_ms": round(fp_time * 1000, 1),
    }


@app.post("/api/detect")
async def api_detect(body: TextInput):
    """Detect whether text is AI-generated or human-written using structural fingerprint."""
    if _DETECTOR is None:
        return JSONResponse({"error": "AI detector not configured"}, 500)

    text = body.text.strip()
    if len(text) < 20:
        return JSONResponse({"error": "Text too short (need at least a few sentences)"}, 400)

    text = html_lib.escape(text)
    if len(text) > 50000:
        text = text[:50000]

    sentences = split_sentences(text)
    if len(sentences) < 3:
        return JSONResponse({"error": f"Only found {len(sentences)} sentences. Need at least 3."}, 400)
    if len(sentences) > 500:
        sentences = sentences[:500]

    t0 = time.time()
    fp = fingerprint("", sentences=sentences)
    fp_time = time.time() - t0

    mode, dist = classify(fp)
    fp_arr = fp.to_array()

    # Composite weighted score
    z = (fp_arr - _DET_MEAN) / _DET_STD
    score = float(z @ _DET_WEIGHTS)

    # Centroid distances
    dist_ai = float(np.linalg.norm(fp_arr - _DET_AI_CENT))
    dist_human = float(np.linalg.norm(fp_arr - _DET_HUM_CENT))

    # Short-text penalty: basin entropy is naturally lower with fewer sentences
    # The classifier was calibrated on ~30-50 sentence passages
    # Reduce confidence for short texts, never increase it
    n_sent = len(sentences)
    short_text_warning = None
    if n_sent < 15:
        short_text_warning = f"Only {n_sent} sentences — detection is unreliable below ~15 sentences. Add more text for better accuracy."
    if n_sent < 30:
        # Shift score toward threshold (reduce certainty)
        damping = max(0.3, n_sent / 30.0)
        score = _DET_THRESH + (score - _DET_THRESH) * damping

    # Classification
    is_ai = score > _DET_THRESH
    # Confidence: how far past the threshold (scaled to 0-1)
    margin = abs(score - _DET_THRESH)
    confidence = min(1.0, margin / 4.0)  # 4.0 = reasonable max margin

    # Basin entropy is the strongest single signal
    entropy_signal = "low" if fp.basin_entropy < 3.38 else "normal"

    return {
        "prediction": "ai" if is_ai else "human",
        "confidence": round(confidence, 3),
        "score": round(score, 3),
        "threshold": round(_DET_THRESH, 3),
        "signals": {
            "basin_entropy": round(fp.basin_entropy, 3),
            "basin_entropy_signal": entropy_signal,
            "formal_structure": round(fp.formal_structure, 4),
            "smoothness_mean": round(fp.smoothness_mean, 4),
            "smoothness_std": round(fp.smoothness_std, 4),
        },
        "centroid_distances": {
            "to_ai": round(dist_ai, 3),
            "to_human": round(dist_human, 3),
        },
        "mode": mode,
        "mode_distance": round(dist, 2),
        "fingerprint": {k: round(v, 4) for k, v in fp.to_dict().items()},
        "warning": short_text_warning,
        "n_sentences": len(sentences),
        "time_ms": round(fp_time * 1000, 1),
    }


@app.get("/api/precomputed")
async def api_precomputed():
    with open(WEB_DIR / "precomputed.json") as f:
        return JSONResponse(json.load(f))


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/whitepaper", response_class=HTMLResponse)
async def whitepaper():
    # Serve the MIDWAY_REPORT as rendered HTML
    try:
        with open(WEB_DIR.parent / "MIDWAY_REPORT.md") as f:
            md_content = f.read()
        # Simple markdown rendering fallback
        return HTMLResponse(f"""<!DOCTYPE html><html><head>
        <meta charset="UTF-8"><title>Waivelets — Whitepaper</title>
        <style>body{{font-family:Georgia,serif;max-width:800px;margin:40px auto;padding:0 20px;
        color:#222;line-height:1.7}}pre{{background:#f4f4f4;padding:16px;overflow-x:auto;
        border-radius:4px}}code{{background:#f4f4f4;padding:2px 4px;border-radius:3px}}
        table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;
        text-align:left}}h1,h2,h3{{color:#333}}</style></head>
        <body><pre style="white-space:pre-wrap;font-family:Georgia,serif;background:none">{md_content}</pre></body></html>""")
    except Exception:
        return HTMLResponse("<h1>Whitepaper not found</h1>", 404)
