#!/usr/bin/env python3
"""Add a few more free-model generations to expand the free-model subset."""
import json, os, time, pathlib, urllib.request

KEY = os.environ["OPENROUTER_API_KEY"]
OR = "https://openrouter.ai/api/v1/chat/completions"
FP = "https://waivelets-production.up.railway.app/api/fingerprint"

prompts = json.loads(pathlib.Path("/tmp/exp/prompts.json").read_text())
new_models = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "deepseek/deepseek-v4-flash:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-nano-9b-v2:free",
]
prompt_indices = [0, 3, 6, 9]  # 4 per model = 16 generations

def call(url, payload, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

added = 0
with open("/tmp/exp/results.jsonl", "a") as fout:
    for m in new_models:
        for pi in prompt_indices:
            t0 = time.time()
            try:
                resp = call(OR, {
                    "model": m,
                    "messages": [{"role": "user", "content": prompts[pi]}],
                    "max_tokens": 2000,
                    "temperature": 0.7,
                }, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                    "HTTP-Referer": "https://waivelets-production.up.railway.app",
                    "X-Title": "waivelets-model-profiling"}, timeout=240)
                text = resp["choices"][0]["message"]["content"] or ""
                gen_time = time.time() - t0
            except Exception as e:
                print(f"FAIL gen {m} #{pi}: {e}", flush=True)
                continue
            if len(text.split()) < 50:
                print(f"SHORT {m} #{pi}: {len(text.split())} words — skip", flush=True)
                continue
            try:
                fp = call(FP, {"text": text}, {"Content-Type": "application/json"}, timeout=60)
            except Exception as e:
                print(f"FAIL fp {m} #{pi}: {e}", flush=True)
                continue
            if "error" in fp:
                print(f"FP_ERR {m} #{pi}: {fp['error']}", flush=True)
                continue
            rec = {"model": m, "prompt_idx": pi, "prompt": prompts[pi], "text": text,
                   "gen_time": round(gen_time, 1), "fingerprint": fp.get("fingerprint"),
                   "mode": fp.get("mode"), "n_sentences": fp.get("n_sentences")}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            added += 1
            print(f"OK {m:48s} #{pi:2d} mode={fp.get('mode'):14s} n_sent={fp.get('n_sentences'):3d} {gen_time:.1f}s", flush=True)
print(f"\nAdded {added} new generations", flush=True)
