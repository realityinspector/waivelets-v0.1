# Waivelets

**Does text have measurable dynamical structure in embedding space?**

A research prototype exploring that question. Embed sentences with MiniLM (22M param sentence transformer), project onto a wavelet-derived eigenbasis (38KB), measure 7 scalar statistics about how the trajectory unfolds. On a 100-text Gutenberg corpus from Homer to Kafka, four structural modes emerge from clustering — and they separate AI-generated text from human writing at 93.7% on a 79-sample eval. A neat hack worth tinkering with.

**[Live demo](https://waivelets-production.up.railway.app)** | **[Whitepaper](https://waivelets-production.up.railway.app/whitepaper)**

---

## What is this?

Waivelets discovers that text has measurable *structural dynamics* in embedding space. When you embed sentences with a neural network and analyze how the trajectory unfolds over time, four distinct modes emerge:

| Mode | What it does | Examples |
|------|-------------|----------|
| **Convergent** | Narrows to few attractor basins and holds. Liturgical, aphoristic, legalistic. | KJV Bible, Bhagavad Gita, Dickinson |
| **Contemplative** | Thinks about thinking. High interiority, low scene narration. | Darwin (Descent), Russell, Hobbes |
| **Discursive** | The wide river. Sustained shaped discourse across many basins. | Austen, Dostoevsky, Plato, Dante |
| **Dialectical** | Rapid oscillation between registers. Systematic contrast. | Shakespeare (plays), Darwin (Origin), Melville |

These are not genres. Shakespeare and Darwin share a mode (dialectical) because they both do rapid systematic contrast. The Bible and a graph database textbook share a mode (convergent) because they both narrow to few semantic basins and repeat.

## The numbers

| Metric | Value |
|--------|-------|
| **End-to-end (raw text → mode)** | **~5–50ms on CPU** (embed + classify) |
| Classify only (pre-embedded) | **68 microseconds** (matrix multiply) |
| With GPU embedding (500 sentences) | **~140ms** |
| Throughput (pre-embedded) | **10,000+ docs/sec** on CPU |
| Fingerprint size | **28 bytes** (7 float32s) |
| Classifier size | **38 KB** (eigenbasis + clusters + centroids) |
| Embedding model | all-MiniLM-L6-v2 (22M params, not included in 38KB) |
| Corpus validated | **100 texts**, 7 genres, 2,500 years |
| Compression ratio | **22 million to 1** |

> **On speed claims:** The 68μs number is the classification step only — the matrix multiply after embeddings are computed. The full pipeline (MiniLM embedding + classification) runs ~5–50ms on CPU. For pre-embedded corpora (common in search/retrieval pipelines), you skip embedding and get 10K+ docs/sec. We compare against fine-tuned BERT (~200ms) as the closest apples-to-apples, not LLM inference which is a fundamentally different task.

## How it works

```
Text → Sentences → Embed (384-dim) → Project (384×26 eigenbasis) → Measure → 7 numbers
```

1. **Embed** each sentence to a 384-dim vector via MiniLM (22M params)
2. **Project** onto a precomputed 26-dim eigenbasis (37KB matrix multiply)
3. **Measure** smoothness, oscillation, 4 semantic activations, basin entropy

The eigenbasis was discovered through continuous wavelet transforms on embedding sequences, building 384x384 correlation matrices, and extracting topological invariants. That's the expensive telescope. The eigenbasis is the map it produced. You don't need the telescope once you have the map.

## Quick start

```bash
# Install
pip install sentence-transformers numpy

# Fingerprint a text
python fastprint.py "path/to/text.txt"

# Run the web server
pip install fastapi uvicorn
uvicorn web.server:app --port 8080

# Run the full Gutenberg corpus
python mega_corpus.py
python mega_corpus.py --resume    # resume from checkpoint
python mega_corpus.py --stats     # show stats from results
```

### Python API

```python
from fastprint import fingerprint, classify, explain

fp = fingerprint("Your text here. It needs at least three sentences. The more text, the sharper the fingerprint.")
mode, distance = classify(fp)
explain(fp, name="My Text")
```

## The 7-number fingerprint

```python
Fingerprint(
    smoothness_mean,    # Cosine similarity between consecutive projections
    smoothness_std,     # Oscillation (variance of smoothness)
    exposition,         # Cluster activation: explanation, analysis
    interiority,        # Cluster activation: inner states, psychology
    formal_structure,   # Cluster activation: complex syntax, meta-text
    scene_narration,    # Cluster activation: bodies in space, action
    basin_entropy,      # Shannon entropy of attractor basin visits (bits)
)
```

## Project structure

```
fastprint.py              # Core fingerprinting engine (zero-dep beyond numpy + sentence-transformers)
basis.npz                 # 384×26 eigenbasis matrix (37KB)
basis_clusters.json       # 6 semantic domain mappings
mega_corpus.py            # Gutenberg corpus runner (100+ texts)
mega_corpus_results.jsonl # Full corpus results
web/
  server.py               # FastAPI server (live fingerprinting + static site)
  index.html              # Interactive frontend (D3.js)
  precomputed.json        # Pre-computed corpus data for frontend
Dockerfile                # Railway deployment
MIDWAY_REPORT.md          # Full research whitepaper
```

## AI text detection

The fingerprint can detect AI-generated text with **93.7% accuracy** on a 79-sample eval (39 AI, 40 human).

**The signal:** AI text visits fewer attractor basins (basin entropy: AI=2.77 vs Human=3.61, Cohen's d=1.79). AI text is structurally *convergent* — it stays in more constrained semantic territory even when trying to be varied.

```bash
# Run the eval yourself
python eval_ai_human_v2.py

# API endpoint
curl -X POST https://waivelets-production.up.railway.app/api/detect \
  -H 'Content-Type: application/json' \
  -d '{"text": "Your text here..."}'
```

| Metric | Value |
|--------|-------|
| **5-fold CV accuracy** | **92.7% ± 3.1%** |
| **AUC (ROC)** | **0.991 ± 0.006** |
| Best single feature (basin entropy) | 87.0% |
| Strongest signal | Basin entropy (d=2.02) |
| vs majority class baseline | +27.0pp |
| Eval set | 58 LLM-generated + 111 Gutenberg texts |

**Limitations:** Short texts (<15 sentences) are unreliable. Trained on one LLM family — needs multi-model validation. Human text from unusual genres (legal, liturgical) can false-positive. A strong signal, not a standalone verdict.

## What you can build with this

- **AI text detection** — 93.7% accuracy from structural dynamics alone, near-zero compute
- **Style transfer by dynamics** — Select few-shot examples by fingerprint, not topic
- **Persona fingerprints** — Each author/persona has a mode distribution
- **Edge classification** — 38KB classifier runs on a microcontroller

## Updates

### March 30, 2026 — AI Detection

Built and shipped AI vs human text detection. Ran comprehensive eval: 39 AI-generated texts (including 7 adversarial samples — casual tone, Hemingway pastiche, diary entries) against 40 fresh Gutenberg excerpts. Basin entropy emerged as the killer signal (d=1.79). Composite 7-feature weighted classifier hits 93.7%. Added `/api/detect` endpoint, interactive detection UI with confidence gauge and structural signal cards, entropy distribution histogram, and honest limitations callout. Short-text damping prevents false positives on passages under 30 sentences.

### March 29, 2026 — Launch

Took Waivelets from a research notebook to a live, interactive research tool. Starting from the original repo — which contained the raw wavelet analysis code, a 49-text validation corpus, and a static Shiny app — we rebuilt the entire pipeline: wrote `fastprint.py` (a zero-dependency fingerprinter), scaled the Gutenberg corpus from 49 to 100 texts across 7 genres and 2,500 years of writing, built a FastAPI server with a live fingerprinting API, designed an interactive frontend with D3.js visualizations, and deployed everything to Railway with auto-deploy from GitHub. The four dynamical modes held up at scale: drama went 100% dialectical, religious texts clustered convergent, and Darwin's two books split across modes exactly as the theory predicts.

## Credits

Research by Sean McDonald — March 2026

Built with: [sentence-transformers](https://www.sbert.net/), [FastAPI](https://fastapi.tiangolo.com/), [D3.js](https://d3js.org/), [Railway](https://railway.app/)

## Origin

This project started from a [Twitter conversation](https://twitter.com/Caldwbr/status/1765892610751692863) about applying continuous wavelet transforms to language model token sequences. The hypothesis: wavelet analysis would reveal self-similar patterns, idiomatic phrases, and argument structures recurring at different timescales — like overtones in music. It did. And the overtones turned out to be structural modes of meaning.
