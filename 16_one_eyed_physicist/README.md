# 16 — The One-Eyed Physicist

Estimate a ball's **coefficient of restitution** `e` from a short synthetic video of it bouncing,
with known scene scale and camera geometry. Two estimators, both giving a value and an uncertainty:

* **Network (MLX):** a per-frame CNN (with x/y coordinate channels) → temporal 1-D conv stack →
  pooled MLP head that outputs a mean and log-variance, trained with a Gaussian NLL on freshly
  rendered clips (infinite synthetic data, rendered on the GPU each step). 471k parameters.
* **Classical baseline:** threshold + centroid tracking of the ball, known-g parabola fits to the
  frames before and after the first bounce, contact times where each parabola meets the ground,
  `e = |v_out| / |v_in|`. Uncertainty from a residual bootstrap of the two fits. No training.

**No recorded (real) video exists for this project**, so the design doc's "gap between synthetic and
recorded video" Measure item is **not measured**. Everything here is synthetic.

## The scene (`scene.py`)
Side view, orthographic camera, 64×64 px, 32 px/m, ground at image row 60, ball radius 0.07 m,
g = 9.81 m/s², 30 fps, 32 frames (1.07 s) per clip. Analytic multi-bounce physics (no air drag,
no spin, horizontal velocity constant). Per clip, uniformly random: restitution e ∈ [0.2, 0.95],
start height 0.5–1.5 m, vertical velocity ±1 m/s, horizontal start/velocity, first impact between
0.2 and 0.55 s, **motion blur** (exposure 0–0.8 of the frame interval, 5 sub-frame samples),
**pixel noise** σ ∈ [0, 0.12], background and ball brightness; 8-bit quantised frames.
Neither method is told the exposure or noise level.

Held-out sets (fixed seeds, never seen in training; training draws from a separate RNG stream):
* `main`: 2000 clips from the training distribution (seed 1001)
* `hard`: 1000 clips with noise σ ∈ [0.25, 0.40], outside the training range (seed 2002)

## Commands (run from this folder)
```bash
../.venv/bin/python baseline.py                     # quick classical-baseline check on 300 clips
../bin/gpu_run ../.venv/bin/python train.py --minutes 15 --out checkpoints/main   # main run (15 min)
../.venv/bin/python evaluate.py --ckpt checkpoints/main      # -> results.json, media/calibration.png, media/scatter.png
../.venv/bin/python demo.py --ckpt checkpoints/main          # -> media/demo.gif, media/demo_frame.png
./smoke.sh                                                        # end-to-end tiny check (< 2 min)
```
Set `OMP_NUM_THREADS=4` when running on the shared machine.

## Files
| path | what |
|---|---|
| `scene.py` | physics (analytic bounce table), parameter sampling, MLX/NumPy renderer, deterministic dataset maker |
| `model.py` | `OneEyed` network, checkpoint loader, batched predict |
| `train.py` | time-budgeted training (MSE warm-up then Gaussian NLL), val every ~22 s, best-by-val-NLL checkpoint in `best/`, final weights at the top level |
| `baseline.py` | classical tracker + parabola-fit estimator with bootstrap uncertainty |
| `evaluate.py` | held-out metrics: parameter error, re-simulation (prediction) error, calibration; plots |
| `demo.py` | GIF of three held-out clips with both estimates overlaid and a 1 s forecast re-simulated from each |
| `smoke.sh` | tiny end-to-end check |
| `checkpoints/main/` | final weights (step 8070; also the best val NLL) used for every reported number: `model.safetensors`, `config.json`, `train_log.jsonl` |
| `checkpoints/main/best/` | best periodic-val checkpoint (step 7875), kept for reference only |
| `results.json` | every number in RESULTS.md |
| `media/` | `demo.gif`, `demo_frame.png`, `calibration.png`, `scatter.png` |
| `logs/` | training stdout |
| `review_check.py`, `results_review.json` | cross-review: reproduction, paired-bootstrap CIs, clipped-classical check (`../bin/gpu_run ../.venv/bin/python review_check.py`, ~1 min) |
