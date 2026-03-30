# Waivelets

**Every text has a structural fingerprint. We found it.**

7 numbers. 68 microseconds. 28 bytes. That's all it takes to classify 2,500 years of human writing into four dynamical modes that cut across every genre boundary. No LLM. No fine-tuning. Just linear algebra on embeddings.

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
| Corpus validated | **100 texts**, 8 genres, 2,500 years |
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

## What you can build with this

- **AI text detection** — If LLM text has different attractor dynamics, detect it at zero cost
- **Style transfer by dynamics** — Select few-shot examples by fingerprint, not topic
- **Persona fingerprints** — Each author/persona has a mode distribution
- **Edge classification** — 38KB classifier runs on a microcontroller

## Update — March 29, 2026

Today we took Waivelets from a research notebook to a live, interactive research tool. Starting from the original repo — which contained the raw wavelet analysis code, a 49-text validation corpus, and a static Shiny app — we rebuilt the entire pipeline: wrote `fastprint.py` (a zero-dependency fingerprinter that classifies text in 68us), scaled the Gutenberg corpus from 49 to 100 texts across 8 genres and 2,500 years of writing, built a FastAPI server with a live fingerprinting API, designed an interactive frontend with D3.js visualizations, and deployed everything to Railway with auto-deploy from GitHub. The four dynamical modes held up at scale: drama went 100% dialectical, religious texts clustered convergent, and Darwin's two books split across modes exactly as the theory predicts.

## Credits

Research by Sean McDonald + Claude (Opus 4.6) — March 2026

Built with: [sentence-transformers](https://www.sbert.net/), [FastAPI](https://fastapi.tiangolo.com/), [D3.js](https://d3js.org/), [Railway](https://railway.app/)

## Origin

This project started from a [Twitter conversation](https://twitter.com/Caldwbr/status/1765892610751692863) about applying continuous wavelet transforms to language model token sequences. The hypothesis: wavelet analysis would reveal self-similar patterns, idiomatic phrases, and argument structures recurring at different timescales — like overtones in music. It did. And the overtones turned out to be structural modes of meaning.
