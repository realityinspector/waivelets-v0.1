#!/usr/bin/env python3
"""Retry just gpt-5 and gemini-2.5-pro with higher max_tokens (reasoning models)."""
import json, os, time, pathlib
import urllib.request

ROOT = pathlib.Path("/tmp/exp")
OUT = ROOT / "results.jsonl"
KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
FP_URL = "https://waivelets-production.up.railway.app/api/fingerprint"

prompts = json.loads((ROOT / "prompts.json").read_text())
models_retry = ["openai/gpt-5", "google/gemini-2.5-pro"]

# Drop any existing short/empty records for these models — easier to just regenerate cleanly
lines = []
if OUT.exists():
    for line in OUT.open():
        try:
            r = json.loads(line)
            if r["model"] not in models_retry or r.get("n_sentences", 0) >= 15:
                lines.append(line)
        except Exception:
            pass
OUT.write_text("".join(lines))
print(f"After cleanup: {len(lines)} records kept", flush=True)

done = set()
for line in OUT.open():
    r = json.loads(line)
    done.add((r["model"], r["prompt_idx"]))

def call(url, payload, headers, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

total = len(models_retry) * len(prompts)
i = 0
with OUT.open("a") as fout:
    for model in models_retry:
        for pi, prompt in enumerate(prompts):
            i += 1
            if (model, pi) in done:
                print(f"[{i}/{total}] SKIP {model} #{pi} (already done)", flush=True)
                continue
            t0 = time.time()
            try:
                resp = call(OR_URL, {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.7,
                }, {
                    "Authorization": f"Bearer {KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://waivelets-production.up.railway.app",
                    "X-Title": "waivelets-model-profiling",
                }, timeout=300)
                text = resp["choices"][0]["message"]["content"] or ""
                gen_time = time.time() - t0
            except Exception as e:
                print(f"[{i}/{total}] FAIL gen {model} #{pi}: {e}", flush=True)
                continue

            wc = len(text.split())
            if wc < 50:
                print(f"[{i}/{total}] SHORT {model} #{pi}: {wc} words — skip", flush=True)
                continue

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
            print(f"[{i}/{total}] OK {model:42s} #{pi:2d}  mode={fp.get('mode'):14s}  n_sent={fp.get('n_sentences'):3d}  wc={wc}  {gen_time:.1f}s", flush=True)

print("DONE", flush=True)
