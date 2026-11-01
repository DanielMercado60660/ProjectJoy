# Phase 0: independent judge (Whisper large-v3-turbo, MLX)

Corpus CER / WER %, 100 clips each (val = seed-0 sample of LJ041-044, test = project 27's 100 clips). Shared
normalizer (`judge.py`). Source: `results/judge/<split>_<system>.json`.

| system | val CER | val WER | test CER | test WER |
|---|---|---|---|---|
| real recording | 0.23 | 1.06 | 0.54 | 1.60 |
| real mel -> 28's vocoder (mel-path ceiling) | 0.27 | 1.06 | 0.43 | 1.54 |
| project 27 TTS (FastSpeech + vocoder) | – | – | 1.14 | 2.84 |
| Mimi round trip, 32 cb | 0.31 | 1.18 | 0.44 | 1.42 |
| Mimi, 16 cb | 0.40 | 1.29 | 0.61 | 1.83 |
| Mimi, 8 cb | 0.55 | 1.51 | 0.74 | 2.07 |
| Mimi, 4 cb (r3's ceiling) | 0.70 | 2.35 | 1.09 | 2.96 |
| Mimi, 2 cb | 2.90 | 5.76 | 2.34 | 5.32 |
| Mimi, cb0 only | 26.16 | 39.56 | 27.43 | 40.21 |
| **Gemma TTS r3** (4 cb, t0.2) | 12.70 | 22.83 | 13.03 | 23.83 |
| same, random init (t0.4) | 76.81 | 112.59 | 67.80 | 97.52 |
| Gemma -> 27's mel (mel1) | 78.20 | 115.17 | – | – |
