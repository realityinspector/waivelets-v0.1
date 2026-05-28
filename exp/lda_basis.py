#!/usr/bin/env python3
"""EXP 1 — Supervised LDA basis as a third spectrometer module.

Fits a per-fold sklearn LinearDiscriminantAnalysis on PER-SENTENCE MiniLM
embeddings with model labels, giving (n_classes-1)=7 supervised axes. The
resulting (384, 7) projection is treated as another spectrometer module and
its per-gen 21-D feature (mean + |mean| + 7-bin basin-argmax histogram) is
concatenated with the existing Gutenberg-78 and LLM-PCA-78.

Strict nested LOO CV: for each held-out generation the LDA basis is re-fit on
the remaining 95 documents' sentences only.

The full A/B/C/D/E/F/G grid lives in `expansion_eval.py`; this script is the
isolated EXP-1 view that reports conditions A (baseline), B (LDA alone),
C (triple-spectrometer).
"""
import json, pathlib, subprocess, sys

# Re-use the combined evaluator for consistency.
HERE = pathlib.Path(__file__).parent
COMBINED = HERE / "expansion_eval.py"
print("EXP 1 runs as part of /tmp/waivelets-v0.1/exp/expansion_eval.py", file=sys.stderr)
print("(Re-running here for an isolated A/B/C report.)\n", file=sys.stderr)
subprocess.check_call([sys.executable, str(COMBINED)])
res = json.loads(pathlib.Path("/tmp/exp/lda_basis_results.json").read_text())
print(json.dumps(res, indent=2))
