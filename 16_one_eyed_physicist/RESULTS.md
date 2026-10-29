# Results — 16 The One-Eyed Physicist (first satisfying version)

**Not measured up front: the synthetic-to-recorded gap.** No real recorded video exists for this
project, so every number below is on synthetic clips. The Measure item "gap between synthetic and
recorded video" is **not measured**.

All numbers come from `results.json` (written by `evaluate.py --ckpt checkpoints/main`) unless
another file is named. Held-out sets were rendered with fixed seeds the training stream never uses.
* **main**: 2000 clips, training distribution (noise σ 0–0.12, exposure 0–0.8 frame, e ∈ [0.2, 0.95])
* **hard**: 1000 clips, noise σ 0.25–0.40, **outside** the network's training range (everything else the same)

Training (from `checkpoints/main/train_log.jsonl`, `logs/train_main.log`): one run, 15 min wall,
8070 steps × 64 freshly rendered clips (≈516k clips, none repeated), 471k params, 9.0 steps/s
*under load* (ten builders sharing the machine). MSE warm-up for 400 steps, then Gaussian NLL.
Val MAE (512 clips): 0.19 at step 1 → 0.0099 at 113 s → 0.0020 at the end. Val NLL was still improving
slowly at the end of the cosine schedule (−4.61 at step 7875, −4.62 final, learning rate by then ~2 % of
peak). A longer run would probably help a little.

## Parameter error (restitution e)

| set | method | MAE | median AE | RMSE | p95 AE | fit failures |
|---|---|---|---|---|---|---|
| main | **network** | 0.0021 | 0.0015 | 0.0033 | 0.0061 | – |
| main | **classical** | **0.0020** | **0.0009** | 0.0048 | 0.0070 | 0 % |
| main | prior mean (0.575) | 0.188 | 0.190 | 0.216 | 0.357 | – |
| hard | network | **0.021** | 0.012 | **0.033** | **0.068** | – |
| hard | classical | 0.036 | **0.0034** | 0.236 | 0.204 | 5.9 % (fall back to prior) |
| hard | classical, fitted clips only | 0.023 | 0.0031 | 0.234 | 0.053 | – |

In distribution the two are **tied on MAE** (0.0021 vs 0.0020; paired bootstrap 95% CI of the
network−classical MAE difference [−0.00007, +0.00037], so not distinguishable); the classical fit has the better
median (CI excludes zero) and the network has the better RMSE (fewer large misses; CI excludes zero). The classical
estimate is closer on 61% of clips. Neither is told the noise level or the exposure time. CIs from
`results_review.json` (reviewer rerun, which reproduced every main-set number above exactly).

**Reviewer note on the hard set: the classical row includes unphysical estimates.** 5 of the 941 "fitted"
hard clips return e up to 6.5 (max |err| 6.2), which is what makes the fitted-only RMSE 0.234. Clipping the
classical estimate to the physical range [0, 1] (no tuning, no prior knowledge of the training range) gives
(`results_review.json`, `cl_clip01_*`):

| set | method | MAE | RMSE |
|---|---|---|---|
| hard | classical, clipped to [0, 1] | 0.027 | 0.082 |
| hard | classical, clipped, fitted clips only | **0.0137** | 0.052 |
| hard | network, on the same fitted clips | 0.0199 | – |

So on the hard set the network's lower MAE (0.021 vs 0.036; paired CI of the difference [−0.032, −0.003])
comes from the classical method's 5.9% prior fallbacks **and** five wild fits. On the clips the classical
method can fit, a one-line physical clip makes it clearly better than the network (0.014 vs 0.020).
The tables above are left unclipped as originally computed; the forecast metric already clips e to [0, 0.999].

They fail in different places (main set, MAE by true e):

| true e | network | classical |
|---|---|---|
| 0.2–0.4 | **0.0019** | 0.0047 |
| 0.4–0.6 | 0.0016 | **0.0012** |
| 0.6–0.8 | 0.0019 | **0.0010** |
| 0.8–0.95 | 0.0034 | **0.0009** |

* With low e the ball barely rises after the bounce: the post-bounce parabola gets only ~3–6 frames
  before the second bounce, so the classical fit is short of data. The network uses the whole clip,
  including the later bounces.
* With high e the classical fit has long clean arcs and wins easily. The network is at its worst at
  e > 0.8, the edge of the training range (the scatter in `media/scatter.png` shows mild squashing there).
* Noise and exposure (motion blur) barely matter in distribution for either method: per-bin MAE stays
  between 0.0015 and 0.0025 (see `bins` in `results.json`).

## Prediction error: re-simulate with the estimate

Each clip is re-simulated from its **true** initial state using the **estimated** e, and compared with
the true trajectory over the 1 s after the clip ends (30 frames; mean |Δy| of the ball centre). The
true state is used on purpose, so this isolates how much the restitution error matters for forecasting.

| set | method | mean (cm) | median (cm) | p95 (cm) |
|---|---|---|---|---|
| main | network | 0.32 | 0.036 | 1.58 |
| main | classical | **0.10** | **0.017** | **0.47** |
| main | prior mean | 9.6 | 2.9 | 38 |
| hard | network | 2.58 | 0.23 | 13.6 |
| hard | classical | **1.41** | **0.079** | **6.5** |

The parameter MAEs are tied, but the classical method forecasts 3× better (paired 95% CI of the mean
difference 0.19–0.26 cm; also better on median and p95, so it does not depend on the statistic). By true e
(`results_review.json`, `resim_mean_by_e`): network vs classical 0.31 vs 0.15 cm at e 0.6–0.8 and 1.12 vs 0.29 cm
at e 0.8–0.95; below e 0.6 both are under 0.04 cm, because the ball has nearly stopped by the forecast window, so
the network's low-e advantage barely registers here. Forecast error is most
sensitive to e when e is high (big, long arcs, many bounces to compound over), and high e is exactly
where the classical fit is 4× more accurate. Parameter error alone hides this, which is why the
prediction metric is worth having.

## Calibration of the uncertainty (`media/calibration.png`)

Coverage = fraction of held-out clips whose true e falls inside the central Gaussian interval
μ ± z·σ. "calib MAE" is the mean |observed − nominal| over the levels 0.1 … 0.99.

| set | method | 50 % | 80 % | 90 % | 95 % | calib MAE | NLL | mean σ | corr(σ, \|err\|) |
|---|---|---|---|---|---|---|---|---|---|
| main | **network** | 0.56 | 0.82 | 0.90 | 0.95 | **0.029** | −4.56 | 0.0028 | 0.50 |
| main | classical (bootstrap) | 0.85 | 0.97 | 0.99 | 1.00 | 0.210 | −4.61 | 0.0042 | 0.70 |
| hard | network | **0.09** | **0.18** | **0.21** | **0.25** | 0.453 | **+50.8** | 0.0029 | 0.27 |
| hard | **classical (bootstrap)** | 0.49 | 0.77 | 0.90 | 0.96 | **0.013** | −3.36 | 0.034 | 0.66 |

Post-hoc check (added after the write-up): rescale each method's σ by a single factor fitted on the
**validation** set (seed 7, the 512 clips `train.py` validates on; RMS of z-scores). Factors:
network ×0.93, classical ×0.52 (`val_sigma_scale` in `results.json`).

| set | method, σ rescaled on val | 50 % | 90 % | 95 % | calib MAE | NLL |
|---|---|---|---|---|---|---|
| main | network ×0.93 | 0.53 | 0.88 | 0.93 | 0.017 | −4.56 |
| main | classical ×0.52 | 0.59 | 0.92 | 0.94 | 0.050 | **−4.91** |
| hard | network ×0.93 | 0.09 | 0.20 | 0.23 | 0.461 | +59.2 |
| hard | classical ×0.52 | 0.29 | 0.59 | 0.66 | 0.211 | −2.46 |

Rescaling fixes the classical method in distribution: its NLL becomes the best of all (−4.91).
On the hard set, though, it turns over-confident. The quarter-pixel σ floor that made it
under-confident in distribution (the floor is active on 95 % of the 512 validation clips, seed 7; reviewer measurement) was exactly what
kept it honest under heavy noise. The network barely changes; it was already calibrated in
distribution and is still badly over-confident out of distribution.

* **In distribution the network's Gaussian NLL head is well calibrated** (90 % interval covers
  90.4 %, PIT histogram close to flat). The classical bootstrap is **under-confident**: its intervals are
  too wide (50 % nominal covers 85 %), because of the quarter-pixel floor I put on the residual
  scale (active on 95 % of the 512 validation clips, seed 7; reviewer measurement). Its σ still ranks errors better (correlation 0.70 vs 0.50).
* **The good kind of failure: the network stays confident outside its training noise range.** On the
  hard set its error grows 10× (MAE 0.0021 → 0.021) but its predicted σ does not move at all
  (mean 0.0028 → 0.0029). Its 95 % intervals cover only 25 % of clips, its NLL goes from −4.6 to
  +50.8, and its PIT histogram is a U shape (bottom-right panel). The heteroscedastic head learned
  "how hard is this clip *within* the training distribution", not "is this clip like my training data".
  The classical bootstrap measures the actual residual noise of each fit, so it widens automatically
  (σ 0.004 → 0.034) and is almost perfectly calibrated on the hard set (calib MAE 0.013).
  It still fails outright on 5.9 % of hard clips, where tracking breaks and it falls back to the prior.

## Speed (busy-machine numbers, under load)

Measured during the `evaluate.py` run that wrote `results.json` (`sec_per_clip_under_load`; `uptime` field:
load averages 11.87 / 13.22 / 10.85, ten builders running): network 0.15 ms per clip on the main set and
0.24 ms on the hard set (batched 128 on the GPU); classical 4.9 / 4.8 ms per clip (single-threaded NumPy,
including 64 bootstrap refits). Training ran at 9.0 steps/s × 64 clips under similar load.
These are not idle-machine numbers. *(Reviewer fix: the earlier text quoted 0.46 ms / 12.3 ms at load
13.98, from an earlier evaluate run whose output was overwritten; it did not match `results.json`.
The reviewer rerun at load 13.84 / 14.88 / 12.76 measured 0.46 / 0.39 ms and 9.8 / 9.8 ms
(`results_review.json`), so under tonight's load these timings move by 2–3× between runs; only the
order of magnitude (network ~20–30× faster per clip when batched) is meaningful.)*

## What worked
* Rendering on the GPU inside the training loop (analytic bounce table in NumPy, soft-disk
  rasterisation, sub-frame motion blur, noise and 8-bit quantisation in MLX) gives infinite,
  never-repeated data at almost no cost. Overfitting simply isn't a concern.
* Coordinate channels plus a per-frame CNN, a temporal conv and mean+max pooling: the error fell to
  1 % of the e range in under 2 minutes.
* A known-g parabola fit, with contact times solved against the known ground plane, is a very strong
  baseline. Excluding frames whose (maximum) exposure window overlaps the estimated impact handles
  motion blur without knowing the exposure time.

## What didn't / caveats
* The network does not beat the physics-informed baseline in distribution. It ties on MAE and loses
  3× on forecasting. It wins at low e. Under heavy out-of-range noise its mean error is lower (0.021 vs
  0.036), but that gap comes from the classical method's 5.9 % prior fallbacks and five unphysical fits
  (e up to 6.5). On the clips the classical method fits, clipped to [0, 1], it beats the network
  (0.014 vs 0.020 on the same clips; reviewer check). The classical median is 4× better (0.0034 vs 0.012),
  and the network's uncertainty is useless there. *(Reviewer fix: the original "near-tie on fitted clips
  (0.021 vs 0.023)" compared the network on all clips with the unclipped classical fit on fitted clips.)*
* The classical baseline is under-confident in distribution: the σ floor is a hand choice, not tuned
  on the test set. A single validation-fitted rescale fixes the main set but breaks the hard set (see the
  calibration section). No single σ scale is right for both. `media/calibration.png` shows the raw σ.
* The classical fit over-estimates low e slightly (visible in `media/scatter.png` for e < 0.35): there
  are few post-bounce frames, and the blurred frames next to the contact get excluded.
* This scene is easy on purpose ("clearly visible bounce", known scale and camera): orthographic camera,
  no drag, no spin, no friction, a single ball on a plain background, and the same geometry in every clip.
* Single training seed. No seed variance measured. The paired confidence intervals in the review section
  are over held-out clips only, not over training seeds.
* Baseline design (reviewer note): both methods see the same frames, but the classical fit only uses the
  arcs either side of the **first** bounce and ignores the later bounces the network can use. That is a
  design choice, not an information handicap, but the network's low-e win partly reflects it; a
  multi-bounce classical fit would probably close some of that gap (not measured).
* The demo GIF uses held-out clips drawn with the test seed but a different count (60), so they are
  not literally rows of the 2000-clip test set. They are still never seen in training.

## Surprises
* The tie on MAE and the 3× gap on forecasting: the two methods spend their error budgets on different
  values of e, and the forecast metric weights high e heavily.
* The network's predicted σ is almost exactly independent of out-of-range noise. I expected it to at
  least widen somewhat. It did not (0.0028 vs 0.0029 mean).

## Next steps
* Make the classical σ noise-aware, since a fixed rescale can't fix both sets.
* Give the network a way to widen its σ: train with noise augmentation beyond the test range, or add an explicit OOD signal (for example
  the residual of the classical track), so the network's uncertainty can widen.
* Hybrid: feed the classical track (centroids) to the network, or use the network only where the
  post-bounce arc is short (low e).
* Go-deeper items: randomised cameras and scale (where scale and g trade off and the posterior should
  turn multimodal), drag and spin, and real phone recordings to finally measure the synthetic-to-real gap.

## Demo
* `media/demo.gif`: three held-out clips (e ≈ 0.30, 0.60, 0.90), with the classical track (orange dots)
  and both estimates ± σ overlaid. A 1 s forecast follows, with rings re-simulated from each estimate
  over the true future ball. `media/demo_frame.png` is a still from it.
* `media/calibration.png`: coverage curves and PIT histograms for both methods on both sets.
* `media/scatter.png`: estimated vs true e.
