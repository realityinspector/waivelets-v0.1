#!/usr/bin/env python3
"""EXP 3 — Adversarial topic-debiased basis.

Same linear-projection backbone as EXP 2 plus two heads on the projected
features:
  - Model head: predict model identity from W·embedding (maximised, standard CE).
  - Prompt head: predict prompt_idx (0..11) from W·embedding (MINIMISED via
    gradient-reversal i.e. loss term `-λ · CE(prompt)` with λ=1.0).

Triplet loss on model labels is kept as the primary contrastive signal so the
adversarial setup is "triplet + classify model + un-classify prompt".

Conditions:
  F. Adversarial basis alone
  G. Quint-spectrometer (Gut + LLM-PCA + LDA + Contrastive + Adversarial)
"""
import json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
COMBINED = HERE / "expansion_eval.py"
print("EXP 3 runs as part of /tmp/waivelets-v0.1/exp/expansion_eval.py", file=sys.stderr)
subprocess.check_call([sys.executable, str(COMBINED)])
res = json.loads(pathlib.Path("/tmp/exp/adversarial_results.json").read_text())
print(json.dumps(res, indent=2))
