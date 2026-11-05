# R8 Gate C (navigation): ViNT / NoMaD zero-shot on the duck's camera

> **Lead's checks (2026-11-05):**
> - Every gate number in the results table matches `zeroshot.json` (ρ, Δρ, sign tests, bearing errors, per-gate
>   flags); its verdict field is `does_not_transfer` for both models.
> - Protocol timing: written 15:29, amendments 15:32, recording started 15:35:35. The amendment's timestamp text was
>   corrected at 16:54 (disclosed above); the file has no history, so that "no other change" rests on the
>   builder's word.
> - Downloads are only the two approved checkpoints (nomad.pth 76 MB; vint.pth 430 MB because it pickles whole
>   Python objects, 119 MB once converted). Both are read with `weights_only=True`.
> - `r8_job.sh` is unchanged (mtime 15:05); the builder's free-slot pre-check lives in its own scripts.
> - The builder left a `tail -f` monitor running; the lead stopped it.
> - The TF32 finding matches the lab's earlier one (31 Silicon Decathlon: MLX 0.32.4 GPU float32 matmuls are TF32
>   unless `MLX_ENABLE_TF32=0` is set first).

**Verdict: does not transfer** (both models, both cameras, by the pre-registered rule). The waypoints are a
goal-independent walk-forward prior. The distance heads only recognise arrival in the last ~2–5 s.

## Built
- `r8_nav/weights.py`: reads both checkpoints only via `torch.load(weights_only=True)`; vint.pth's pickled classes become inert stubs.
- `r8_nav/mlx_models.py`: the MLX ports — EfficientNet-b0 with efficientnet_pytorch's static 224 px padding (BN / GN), ViNT, NoMaD with goal mask, UNet1D and a 10-step DDPM.
- `torch_ref.py`, `parity.py`, `episodes.py`, `zeroshot.py`, `posthoc.py`, `visual.py`.
- `tests/test_r8_nav.py` (10 tests), `media/r8_nav_strip.png`.
- Weights and provenance with sha256 in `data/pretrained/`.

## Parity (`parity.json`)
Torch references were written from the official sources, loaded with the official weights `strict=True`, and run on random inputs. Tolerance, fixed beforehand: max|Δ| ≤ 1e-3·max(1,|ref|), and 2e-3 for the sampler with shared noise.

Measured max abs errors:
- ViNT: distance 8.1e-6 (|ref| ≤ 7.5); waypoints 2.6e-6.
- NoMaD: context 1.6e-5 (≤ 21); distance 1.5e-5; UNet 9.5e-7.
- Full DDPM waypoints: 3.4e-5 (≤ 10.4).

All pass **with `MLX_ENABLE_TF32=0`**. MLX's default on this M5 does float32 matmuls in TF32, which gives 2–4e-3 relative error and fails 6 of 11 checks. The runtime disables it.

## Setup (`protocol.json`, written before any duck frame, as were amendments A1–A4; the amendment timestamp was later corrected to the file's mtime, nothing else touched)
- **Episodes:** 80 press-needed set_lamp episodes on train-0000..0039, from the servo (its ORACLE latch is on the expert; no truth enters a model) and the bot. 69 reached the plate.
- **Frames:** 4 Hz, rendered inside the experts' ticks (episodes are bit-identical with or without this), cropped 4:3, resized to 85×64 / 96×96.
- **Goal G1:** the same episode's on-plate frame.
- **Camera:** 0.20–0.27 m high (RECON's config: 0.95 m). The head pitches 0.1–0.55 rad, so G1 looks ~28° down at floor.
- **Queries:** 6,688 frames up to 40 s before the goal, with 0.25 s context.

## Results (`zeroshot.json`): G1, 0.25 s
Thresholds: ρ ≥ 0.5, ≤ 30°, clearly beat the wrong-room goal (Δρ ≥ 0.2 and ≥ 10°, sign test), beat straight-ahead.

| model, camera | ρ (wrong-room) | Δρ, sign test | bearing error: goal / wrong / masked / straight |
|---|---|---|---|
| ViNT, normal | 0.22 (0.06) | 0.15, 47/69, p=0.004 | 16.7° / 16.8° / – / 14.0° |
| ViNT, wide 100° | 0.27 (0.15) | 0.12, 44/69, p=0.03 | 16.8° / 16.7° / – / 14.0° |
| NoMaD, normal | 0.22 (0.10) | 0.11, 49/69, p=0.0006 | 17.6° / 17.7° / 17.2° / 14.0° |
| NoMaD, wide 100° | 0.38 (0.12) | **0.22, 51/69, p=9e-5** | 16.2° / 16.6° / 16.1° / 14.0° |

- **Gates:**
  - ρ ≥ 0.5 fails everywhere; ≤ 30° passes everywhere.
  - The bearing control and the straight-ahead reading fail everywhere.
  - Only NoMaD-wide's Δρ is met.
- **Waypoints:** the ≤ 30° pass is an artifact of the experts facing the plate. Straight-ahead beats every model in 50–59 of 69 episodes (p ≤ 2.4e-4), and the wrong-room or masked goal gives the same error. Post hoc, normal camera: with the plate in view, 8.8° / 9.6° vs 7.5° straight-ahead; out of view, ~70° for all. The error against the expert's next 1 s of motion is ~11° everywhere, controls included.
- **Distance:** over the last 5 s (A1), ρ is 0.42 / 0.38 against ~0 for wrong-room. With the wide camera it is 0.63 / 0.58 against 0.31 / 0.24. Post hoc: the output drops mainly in the last 2 s (11.2 / 10.6, against ~13.6–14.0 beyond 5 s).
- **Secondary:**
  - G2 (placed on the plate, head level): ρ −0.15 to 0.06.
  - 0.5 / 1 s context: no gain.
- **Exploratory:** ViNT's context encoder outputs near-constant features for every input tried: duck frames, noise, smooth fields (std 0.002, vs 0.20 for its goal encoder; the torch reference agrees). So context frames have no effect. The goal encoder shares the architecture, loader and BN eps and varies normally, which excludes the port and the loading path. What remains is the weights in the file.

## What fine-tuning needs
- **Data:** 4 Hz duck trajectories. The recorder exists (~6 s wall per episode on 4 processes); 300 configs × several starts gives ≈ 2–5 h of driving.
- **Labels:**
  - temporal distance from frame offsets (free);
  - waypoints from leg odometry over 1–2 s (sensors-only; ~13 cm drift per 30 s), or disclosed ORACLE poses;
  - hindsight goals, including the crossing;
  - wrong-room negatives.
- **Units:** the duck moves ~3 cm per 0.25 s, so use 0.5–1 s steps and refit ACTION_STATS and the distance range.
- **Camera:** a fixed navigation head pose and the wide render.
- **Gate:** add a lamp/done head. Compare checkpoint + fine-tune against scratch with 2 seeds, then go closed loop via a waypoint layer.

## Compute
- Downloads 506 MB; parity 21 s; recording 748 s on 4 processes.
- Zero-shot (shared GPU): 743 s for the primary stage plus 1,314 s for the rest, which used a batched DDPM (equals per-episode sampling to 2.8e-5).
- Slot waits ~28 min. `smoke.sh`: 170 passed in 173 s at load ~18 (over the 2 min target, combined suite; mine take ~12 s).

## Deviations
- torch is used only for the weights and the parity reference.
- Drive's virus-scan page: I re-requested its own GET form with curl (confirm=t). No browser.
- I killed the job after the primary stage and relaunched it with `--merge`. After losing four slot races I added a 1 s free-slot check before `r8_job.sh`; the cap is unchanged.
- The BAM track edited duck.py mid-run. My episodes reproduce the pre-change entry ticks.
- Before the protocol, exploration runs measured the experts' timing (no nav frames, no models); they set the 5 s start and the 40 s window.
- A 3-episode code check (shapes only) ran before the full run.

## Open issues / next
- Open:
  - fisheye not tried;
  - open loop only;
  - ViNT's collapsed context encoder is unexplained;
  - head-pitch confound.
- Next: re-record with a fixed head pose and the wide camera, then fine-tune vs scratch.
