#!/usr/bin/env python3
"""Generate model outputs via OpenRouter, then fingerprint each via waivelets /api/fingerprint."""
import json, os, sys, time, pathlib
import urllib.request, urllib.error

ROOT = pathlib.Path("/tmp/exp")
OUT = ROOT / "results.jsonl"
KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
FP_URL = "https://waivelets-production.up.railway.app/api/fingerprint"

prompts = json.loads((ROOT / "prompts.json").read_text())
models = json.loads((ROOT / "models.json").read_text())

# Resume: load done (model, prompt_idx) pairs
done = set()
if OUT.exists():
    for line in OUT.open():
        try:
            r = json.loads(line)
            done.add((r["model"], r["prompt_idx"]))
        except Exception:
            pass

def call(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

total = len(models) * len(prompts)
i = 0
with OUT.open("a") as fout:
    for model in models:
        for pi, prompt in enumerate(prompts):
            i += 1
            if (model, pi) in done:
                print(f"[{i}/{total}] SKIP {model} #{pi}", flush=True)
                continue
            t0 = time.time()
            try:
                resp = call(OR_URL, {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 900,
                    "temperature": 0.7,
                }, {
                    "Authorization": f"Bearer {KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://waivelets-production.up.railway.app",
                    "X-Title": "waivelets-model-profiling",
                }, timeout=180)
                text = resp["choices"][0]["message"]["content"] or ""
                gen_time = time.time() - t0
            except Exception as e:
                print(f"[{i}/{total}] FAIL gen {model} #{pi}: {e}", flush=True)
                continue

            if len(text.split()) < 50:
                print(f"[{i}/{total}] SHORT {model} #{pi}: {len(text.split())} words — skip", flush=True)
                continue

            # fingerprint
            try:
                fp = call(FP_URL, {"text": text}, {"Content-Type": "application/json"}, timeout=60)
            except Exception as e:
                print(f"[{i}/{total}] FAIL fp {model} #{pi}: {e}", flush=True)
                continue

            if "error" in fp:
                print(f"[{i}/{total}] FP_ERR {model} #{pi}: {fp['error']}", flush=True)
                continue

            record = {
                "model": model,
                "prompt_idx": pi,
                "prompt": prompt,
                "text": text,
                "gen_time": round(gen_time, 1),
                "fingerprint": fp.get("fingerprint"),
                "mode": fp.get("mode"),
                "n_sentences": fp.get("n_sentences"),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            print(f"[{i}/{total}] OK {model:50s} #{pi:2d}  mode={fp.get('mode'):14s}  n_sent={fp.get('n_sentences'):3d}  {gen_time:.1f}s", flush=True)

print("DONE", flush=True)
