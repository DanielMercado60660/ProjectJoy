# 26 · Neural Orchestra — a compact generative drum machine

A small **style-conditional VAE (MLX)** that generates and morphs 2-bar drum grooves, trained only on
**procedurally generated** patterns (no downloaded music). Patterns are symbolic grids; a separate numpy
synth turns them into WAVs, and `media/demo.html` is a self-contained step sequencer that runs the VAE
decoder in JavaScript (the seed for an iPad instrument).

**Open `media/demo.html` in a browser** (works offline from `file://`), press Play, pick a style and hit
"New neural groove", drag the morph slider or the latent pad.

## Pieces

| file | what it does |
|---|---|
| `data.py` | procedural drum "drummer rules" for 8 styles (rock, funk, house, breakbeat, hip-hop, bossa, half-time, reggaeton): 32 sixteenth steps x 9 drums (kick, snare, clap, closed hat, open hat, low tom, high tom, ride, crash) with velocity and microtiming; de-duplicated and split 80/10/10 → `data/drums.npz` |
| `synth.py` | numpy drum synth (sine-sweep kick/toms, filtered-noise snare/hats/clap/crash, square-cluster ride), renders grids to 44.1 kHz WAV |
| `model.py` | MLX models: `VAE` (MLP encoder/decoder, optional style conditioning), `ARVAE` (same encoder, autoregressive GRU decoder), `StyleClassifier` |
| `train.py` | VAE training with checkpoint/resume (weights, optimizer state, step, numpy RNG state, args) |
| `train_classifier.py` | separate style classifier used to score conditional control |
| `baselines.py` | Bernoulli-per-cell, position-aware column Markov, bar-aware Markov ("markov_bar"), copy-a-training-pattern |
| `metrics.py` | novelty (nearest-neighbour Hamming), diversity, pulse/timing stats, swing |
| `eval.py` | everything vs everything under matched conditions → `results/eval.json` (also picks the main VAE by validation NLL and the demo decode mode by C2ST) |
| `report.py` | `results/eval.json` → `results/tables.md` (the tables in RESULTS.md) |
| `make_demo.py` | WAVs, PNGs, blind A/B folder and `media/demo.html` from the saved checkpoint |
| `score_listening.py` | scores the blind A/B sheet once you've filled it in |
| `web/` | `template.html`, `core.js` (decoder + synth in JS), `ui.js` (sequencer), `check_decoder.js` (node check that JS decoder == MLX) |
| `sweep.sh` | the training runs used for the results |
| `tests/test_smoke.py` | tiny end-to-end pipeline test |

## How to run (from this folder)

```bash
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY data.py                                   # ~5 s  -> data/drums.npz (8,792 distinct patterns)
$PY train_classifier.py                       # ~30 s -> checkpoints/classifier/
../bin/gpu_run ./sweep.sh > logs/sweep.log 2>&1   # ~6 min: 3 beta MLP VAEs, unconditional, AR VAE (+ a check-eval to results/eval_sweep.json;
                                                  #   that output path was added after the run, so it isn't on disk)
# final evaluation as reported (≈2 min; the AR VAE wins on val NLL, the beta=1 MLP VAE is the demo model):
../bin/gpu_run $PY eval.py --vaes checkpoints/vae_cond_b1.0/best checkpoints/vae_cond_b0.5/best \
    checkpoints/vae_cond_b0.25/best checkpoints/vae_uncond_b0.5/best checkpoints/arvae_cond_fb10/best \
    checkpoints/vae_quick_b1.0 --demo_vae checkpoints/vae_cond_b1.0/best > logs/eval.log 2>&1
$PY report.py                                 # -> results/tables.md, spliced into RESULTS.md
$PY make_demo.py                              # -> media/ from the saved checkpoints, no training (~20 s)
$PY -m pytest -q                              # smoke test, < 1 min
node web/check_decoder.js media/demo.html     # JS decoder agrees with MLX
```

`checkpoints/vae_quick_b1.0` is an earlier 200-epoch run kept for comparison
(`$PY train.py --epochs 200 --out checkpoints/vae_quick_b1.0`; final-epoch weights, no `best/`).
Resume a training run: add `--resume` with the same `--out`.

## Layout

```
data/drums.npz            dataset + fixed split
checkpoints/<run>/        vae.safetensors, vae_config.json, optimizer.safetensors, state.json
checkpoints/classifier/   style classifier
results/eval.json         all measured numbers;  results/tables.md  same as markdown
logs/                     training/eval logs
media/demo.html           the instrument
media/grooves/            generated grooves per style (VAE)
media/compare/            same style from real data / VAE / bar-Markov / Bernoulli
media/interp/             latent walks style A -> style B (tempo glides too)
media/pattern_grid.png, media/interpolation_grid.png
media/listening_ab/       blind A/B pairs + listening_sheet.csv; answer key hidden in _answer_key/
```
