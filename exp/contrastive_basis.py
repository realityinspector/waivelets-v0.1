#!/usr/bin/env python3
"""EXP 2 — Contrastive (triplet) linear basis.

Trains a (384, K=16) linear projection W with cosine-distance triplet loss
(margin=0.5): same-model sentences pull together, different-model push apart.
Strict nested LOO CV: W is re-trained on each fold's training sentences only.

Per-doc 48-D feature = mean + |mean| + 16-bin basin-argmax histogram.

Conditions:
  D. Contrastive basis alone (48-D)
  E. Quad-spectrometer Gut + LLM-PCA + LDA + Contrastive
"""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
COMBINED = HERE / "expansion_eval.py"
print("EXP 2 runs as part of /tmp/waivelets-v0.1/exp/expansion_eval.py", file=sys.stderr)
subprocess.check_call([sys.executable, str(COMBINED)])
res = json.loads(pathlib.Path("/tmp/exp/contrastive_results.json").read_text())
print(json.dumps(res, indent=2))
