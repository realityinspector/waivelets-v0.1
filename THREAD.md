# Waivelets — X/Twitter thread

Drop SVGs (in `web/assets/`) into tweets 1, 3, and 4 respectively.

---

**1/** I built a thing that fingerprints text dynamics in 7 numbers and detects AI-generated writing at 92.7% accuracy.

It also discovered that the Bible and a graph-database textbook belong to the same structural mode.

Here's what's going on. 🧵

📎 `pipeline.svg`

---

**2/** Every text has structural dynamics — how meaning *moves* through it sentence by sentence.

I ran MiniLM embeddings through wavelet analysis on 100 books from Project Gutenberg (Homer → Kafka, 7 genres) and found something weird:

The natural taxonomy isn't genre. It's *dynamics*.

---

**3/** Four modes emerged:

🟧 **CONVERGENT** — narrows to a few attractors and holds (liturgy, code docs)
🟪 **CONTEMPLATIVE** — thinks about thinking (essays, philosophy)
🟦 **DISCURSIVE** — the wide river (novels, sustained narrative)
🟥 **DIALECTICAL** — oscillates between registers (drama, scientific argument)

📎 `modes.svg`

---

**4/** The pairings broke my brain:

📕 Bible + graph-database textbook → both CONVERGENT
🎭 Shakespeare + Darwin → both DIALECTICAL
📜 Plato + Austen → both DISCURSIVE

Genre is what a text is *about*. Mode is what it *does*. They're orthogonal.

📎 `pairings.svg`

---

**5/** The full pipeline uses continuous wavelet transforms across 384 embedding dimensions to find a 26-mode attractor landscape — Hopfield-like, with one universal attractor shared across all English text.

That's the telescope. Then I distilled it into a 37KB eigenbasis.

---

**6/** At runtime:

`sentence → embedding → matmul → 7 numbers`

<1ms per text. 28 bytes per fingerprint.

The wavelets were the telescope. The eigenbasis is the map. You don't need the telescope once you have the map.

---

**7/** AI text has measurably different dynamics. It visits fewer attractor basins than human writing.

Pair the 7-number structural fingerprint with surface formatting features (lists, bolds, em-dashes) and you get **92.7% detection accuracy** under 5-fold CV. AUC 0.991.

n=169 (58 AI, 111 human).

---

**8/** Just ran it passage-by-passage on the new papal encyclical *Magnifica Humanitas* — Pope Leo XIV's 2026 letter on AI.

40 passages. ~45k words. Meta-irony intended.

→ https://waivelets-production.up.railway.app/magnifica

---

**9/** Live demo + open source:

🔗 App: https://waivelets-production.up.railway.app
🔗 Whitepaper: https://waivelets-production.up.railway.app/whitepaper
🔗 Repo: https://github.com/realityinspector/waivelets-v0.1

Paste in any text. Tell me what mode you live in.
