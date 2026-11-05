# LeWorldModel (LeWM) for the Microduck: feasibility study

Written 2026-11-04 (R6). This study trained nothing and downloaded nothing. Every number below comes from one of four places:

- the paper, arXiv 2603.19312, HTML v3 (2026-06-03; v1 and v2 spot-checked, same key numbers);
- the official repo's README, configs and source as web pages;
- Hugging Face file listings (sizes from the HF API; no files fetched);
- timings measured on this Mac today. The timing script is `lewm/bench_step.py`, and the load average was 6–9 while it ran.

Owner's prompt: *"reconsider LeWorld model, it's only 15M parameters, take the pre-trained one and work from there."*

**Short answer.**
- **The pretrained checkpoints don't help.** Every released LeWM checkpoint was trained on one toy environment's pixels and action space, so we train from scratch.
- **Porting is cheap.** The port to MLX is about 500 lines with nothing missing from MLX. A LeWM-sized train step is about 0.25–0.29 s on 64–96 px frames.
- **We have enough data to try.** We already hold about 1.33M action-labelled training frames, about the size of LeWM's TwoRoom set. The weak spot is coverage: only 300 training rooms.
- **First use: the representation, not the planner.** The world model's encoder and predictor would serve as a predictive representation, gated on probe and prediction tests. Latent planning comes later and narrower.
  - Its native planning reach is 2.5 s, against plate approaches of 10–50 s.
  - It needs a goal image, and a fresh room has none.
  - The hand-written skills already solve short sub-goals.

---

## 1. What LeWM is, exactly

| Part | Paper | Code (`lucas-maes/le-wm`, configs as read 2026-11-04) |
|---|---|---|
| **Encoder** | ViT-Tiny, patch 14, 12 layers, 3 heads, width 192, ~5M params (Sec 3.1, App D). Input 224×224 (App D). One frame per encoding; no frame stacking. | ImageNet normalisation. The embedding is the last-layer [CLS] token passed through a projector: Linear 192→2048, BatchNorm1d, GELU, Linear 2048→192. The paper says BatchNorm is needed because the ViT's final LayerNorm blocks SIGReg (Sec 3.1). |
| **Latent** | 192-d (Sec 5.1). Performance drops below ~184 and saturates above (App G, Fig 15). | — |
| **Predictor** | 6-layer transformer, 16 heads, dropout 0.1, ~10M params. Learned positional embeddings, called "ViT-S" (App D, Table 6). | Width 192, 16 heads × 64 (inner 1024), MLP 2048. Causal over time (`is_causal=True`). A `pred_proj` MLP follows, same shape as the projector. |
| **Action conditioning** | AdaLN-zero in every layer, zero-initialised (Sec 3.1). | Action encoder: Conv1d(k=1) then an MLP (4× expansion, SiLU) to 192 dims. |
| **History, frameskip** | History N = 3 frames for PushT and Cube, 1 for TwoRoom. Frameskip 5, with the 5 actions concatenated into one block (App D). | `input_dim = frameskip × action_dim`. The released TwoRoom config has `num_frames: 3`. |
| **Training windows** | 4 frames and 4 action blocks, batch 128 (App D). Teacher forcing (Eq 1). | The predictor sees true embeddings for frames 0–2 and regresses frames 1–3. |
| **Losses** | L = ‖ẑ_{t+1} − z_{t+1}‖² + λ·SIGReg(Z), with λ = 0.1 and M = 1024 random directions (Sec 3.1, Eqs 1–3). SIGReg runs the Epps–Pulley normality test on 1-D random projections of the batch of embeddings, per time step (App A; from LeJEPA, arXiv 2511.08544). No EMA and no stop-gradient: training is end to end. | λ = 0.09 (Fig 16: best near 0.09, above 80% success for λ in [0.01, 0.2]). Quadrature: 17 knots on [0, 3], window exp(−t²/2). |
| **Parameters** | "~15M" = 5M + 10M (Abstract, Sec 3.1). | Each HF `weights.pt` is 72.29 MB, about 18.07M fp32 values. My MLX re-build counts **17.96M**. The two projectors (~0.8M each) and the action encoder make up the difference. |
| **Optimiser** | Not stated. The paper says "single GPU, a few hours" (Abstract) and 10 epochs (App E). | AdamW, lr 5e-5, weight decay 1e-3, linear warm-up then cosine (stepped per epoch). bf16, gradient clip 1.0, batch 128. `max_epochs: 100`. Lightning + Hydra + stable-pretraining. |
| **Planner** | Latent CEM with terminal cost C = ‖ẑ_H − z_goal‖², where z_goal encodes the goal image (Eq 4, Sec 3.2). 300 samples, 30 elites, μ = 0, σ = 1. 30 iterations for PushT and 10 for the others (App D; App B says 30 for all). Horizon 5 blocks = 25 env steps; the whole plan runs before replanning. | `solver/cem.yaml`: 30 iterations. |
| **Evaluation** | 50 episodes, each starting from a dataset state, with the goal 25 steps later in the *same* episode and a 50-step budget (App F.1). | Seed 42. |
| **Planning speed** | "48× faster", under 1 s per plan (Sec 4.2). The project-page figure gives 0.98 s for LeWM vs 47 s for DINO-WM. | — |

**Datasets (App E).** All images are 224 px. Transition counts are episodes × average length, my arithmetic.

| Environment | Episodes × avg length | Transitions | Collection | Action dim |
|---|---|---|---|---|
| TwoRoom | 10,000 × 92 | ≈0.92M | Noisy heuristic | 2 |
| PushT | 20,000 × 196 | ≈3.9M | Expert (the DINO-WM set) | 2 |
| OGBench-Cube | 10,000 × 200 | ≈2.0M | OGBench heuristic | 5 |
| Reacher | 10,000 × 200 | ≈2.0M | "SAC policy"; the eval config names `dmc/reacher_random` (unresolved) | 2 |

**Results (Fig 6 bars, read from the project-page PNGs).**
- LeWM success: PushT 96, TwoRoom 87, Reacher 86, Cube 74.
- PLDM: 78 / 97 / 78 / 65.
- DINO-WM without proprioception: 74 / 100 / 79 / 86.

**PushT ablations (App G).**
- Predictor size: tiny 80.7, small 96, base 86.7 (Table 6).
- Adding a pixel-reconstruction loss: 86 (Table 7).
- ResNet-18 encoder: 94 (Table 8).
- Predictor dropout: 0 gives 78, 0.1 gives 96, 0.5 gives 66.7 (Table 9).

**Probes (Tables 1, 3, 4).**
- Positions are linear in the latent: PushT r 0.97–0.99, TwoRoom 0.996.
- Velocities and yaw are not: Cube joint velocity r 0.095, overall Cube 0.48 vs DINO-WM's 0.73.

**Surprise.** Prediction error jumps when objects teleport (p < 0.01, Sec 5.2).

**Stated limits (Sec 6).**
- Short planning horizons; the authors suggest hierarchy as the fix.
- It needs offline data with good coverage. SIGReg struggles on low-diversity, low-dimensional data, which is their explanation for TwoRoom's 87.
- It needs action labels.

**Paper vs code disagreements.**
- λ: 0.1 in the paper, 0.09 in code.
- Epochs: 10 vs 100.
- TwoRoom history: 1 vs 3.
- CEM iterations: 30/10 vs 30.
- SIGReg nodes: [0.2, 4] vs [0, 3] with 17 knots.
- The "ViT-S" label doesn't match the 192-wide predictor.
- Parameter count: 15M stated vs ~18M in the weights.
- The README's checkpoint instructions are older than `eval.py`.
- Dataset format: HDF5 in the README vs Lance in the current PushT config.

**Not verified.**
- The Google Drive folder's contents and sizes.
- GPU model and training wall time (stated nowhere).
- The arXiv Comments field.
- Error bars in Fig 6.

## 2. Official code and checkpoints, and whether they help

**Official repo.** The official repo is **https://github.com/lucas-maes/le-wm**, MIT licence.
- The paper's own "Code" link (HTML v3) points to it, as does le-wm.github.io.
- `github.com/linzhongzi/le-worldmodel` is **not** official. The GitHub API marks it as a fork of lucas-maes/le-wm created 2026-03-30; it is a stale copy that inherited the description.

**What was released.**
- **The Google Drive folder** is linked from the official README, but it holds the *baseline* checkpoints: PLDM, LeJEPA, IVL, IQL, GCBC and DINO-WM.
- **LeWM checkpoints** are on Hugging Face (collection `quentinll/lewm`, MIT). Each model has `weights.pt` (a state_dict) and `config.json`:

  | HF repo | `weights.pt` size |
  |---|---|
  | `quentinll/lewm-pusht` | 72,290,721 B |
  | `quentinll/lewm-cube` | 72,291,425 B |
  | `quentinll/lewm-tworooms` | 72,290,849 B |
  | `quentinll/lewm-reacher` | 72,290,849 B |

- **Loading.** `config.json` points at `stable_worldmodel.wm.lewm.*` classes. The code loads with `torch.load(map_location='cpu')` and no `weights_only` flag, then `hydra.utils.instantiate` and `load_state_dict`. The README's older conversion path pickles the whole module (`torch.save(model)`).
- **Datasets.** The datasets are 3.4–46 GB zstd archives, for example `tworoom.tar.zst` at 3,425,937,909 B. They are not needed here.

**Does a pretrained checkpoint help the Microduck? No.**

- **Predictor.** Its AdaLN action encoder expects a 10–25-d block (frameskip 5 × 2 or 5 action dims) with that environment's meaning: pusher xy, reacher torques. Microduck's action is vx, wz and 4 head deltas at 10 Hz. That half of the network (~10.8M of 18M parameters) is dead weight. Its dynamics knowledge (a top-down point agent, a T-block, a fixed arm camera) also has nothing to do with a walking egocentric camera.
- **Encoder.**
  - Shape mismatch: ViT-Tiny at patch 14 and 224 px. Patch 14 does not divide 96 (stored frames) or 64 (the policy's frame). Using it means upsampling 96 px frames to 224 or 112, which adds compute but no information, and interpolating its positional embeddings.
  - What it learned: it saw one synthetic environment (flat 2-D renders or a fixed-camera MuJoCo arm) for a few GPU-hours. Its features carry no ImageNet-style generality. The ViT-Tiny it starts from is randomly initialised; the paper stresses "no pretrained encoder".
  - Data on our side: we have about as many frames of our own (1.33M) as their TwoRoom set. As an initialisation it might speed early training slightly. It will not supply knowledge we lack.
- **Verdict.** From scratch on Microduck data is the realistic path. An encoder-init ablation (PushT or TwoRoom encoder vs random) is possible only after S1 works, and only if the owner wants it.

**If a `.pt` ever had to be read.** Torch is not installed and installs are forbidden. A modern `torch.save` file is a zip containing `data.pkl` plus raw tensor storages under `data/<key>`. A restricted `pickle.Unpickler` could read it into numpy in about 100 lines:
- whitelist only `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and the storage-type names;
- map `persistent_load` to `np.frombuffer` over the zip member, with shape, stride and offset;
- run it with `python -I` inside a new, empty directory, treating the file as untrusted.

This should work for the HF `weights.pt`, which is a state_dict. It is **untested** because I could not fetch the file. It would *not* be safe for the README's whole-module `_object.ckpt`: that file needs arbitrary class construction, so refuse it.

## 3. Microduck data inventory

**Episode format.** One compressed `.npz` per episode, gitignored, defined in the `record.py` docstring. Per 10 Hz tick it stores:
- `frames`: 96×96×3 uint8, box-downsampled from the 128×128 live render. The P2 policies use 64×64.
- `body` (34), `imu_sub` (5×6).
- `cmd_exec` (8) = [vx, vy, wz, neck_p, head_p, head_y, head_r, done]: the command actually applied.
- `cmd_bot` (8): the bot's own command, which is the label in DAgger rounds.
- `skill`, `activity`, `observed`, `memory` (JSON: the bot's confirmed tracks, lamp, last ball sighting), `truth` (7, replay only), `header`.

**Action label for a world model: `cmd_exec`, not `cmd_bot`.**
- vy is always 0 and `done` ends the episode rather than moving anything, so the dynamics action is **6-d**: vx, wz and the 4 head deltas.
- At frameskip k the action block is 6k-d.
- Bounds: vx ∈ [−0.3, 0.4] m/s, wz ∈ [−0.6, 0.6] rad/s, head deltas within 0.8 × caps.

**Counts, from the index files; all configs are on the training split.**

| Source | Episodes | Ticks | Mean / min / max length | Who drove |
|---|---|---|---|---|
| M4 bot demos (`index.jsonl`) | 1,029 (343 per task) | 447,216 | 435 / 101 / 1,200 | Reference bot, ~8% perturbed |
| DAgger round 1 (β 0.5) | 450 | 158,261 | 352 / 102 / 1,200 | Mixed policy and bot |
| DAgger round 2 (β 0) | 450 | 287,041 | 638 / 14 / 1,200 | M5 policy |
| DAgger round 3 (set_lamp) | 300 | 171,963 | 573 / 11 / 900 | P2-A′ policy |
| Recovery and takeover (set_lamp) | 500 (`p2_takeover` 300 + `p2_recovery` 200) | 260,591 | 521 / 11 / 1,300 | Bot alone, or the policy then a bot takeover at tick 60–250 (`p2_takeover`: `bot_alone`, `fresh_250`, `takeover_100`) |
| **Total (training split)** | **2,729** | **≈1.33M** | | |

`*_filtered_index.jsonl` and `dagger3_train_index.jsonl` list the same episodes as their parents (identical counts and ticks); they are not extra data.

**Exclude from training.**
- `p2_dev` (80 set_lamp bot demos: instances-dev 0010–0029, others 0000–0019), `p2_shadow_dev` (440), `p2_look` (4) and `m6_clips`. These are dev or test rooms.
- `p2_dev` is the natural **prediction and probe check set on unseen dev rooms**. It does not overlap the 80 confirmation rooms (instances 0030–0049, others 0020–0039).

**Against LeWM's datasets.**
- **Size is comparable.** 1.33M transitions vs TwoRoom's 0.92M (PushT 3.9M). Mean episode length at frameskip 5 is about 87 blocks, vs TwoRoom's 92.
- **Action diversity is better than a pure-expert set.** About 0.88M ticks come from the DAgger and recovery rounds, where a weak policy drove some or all of the time: wrong-way turns, standing, collisions. That is the off-expert coverage a dynamics model needs.
- **Room diversity is the weak point.**
  - All 2,729 episodes come from the same **300 training configs**, about 9 episodes per room.
  - The training families fix one partition orientation (wall_NS) and one appearance family.
  - The dev rooms differ by design: wall_EW partition, different floor and wall hues, light tilt 30–45°. A world model trained on training rooms will be judged out of family, and appearance shift is exactly what SIGReg-trained encoders were never shown to handle. LeWM's own colour-change surprise result was not significant.
- **The observation regime is harder.** LeWM's four environments use fixed external cameras on fully observed scenes, as I understand them from the paper's figures. The duck's camera bobs on a walking head and sees a fraction of the room. With 3 frames of history, out-of-view objects are not in the predictor's input at all.

**How much more can be recorded.**
- Measured today with `record_episode` into the scratchpad, on train-0000 set_lamp and train-0001 go_and_return: **37.8 ticks/s per process**, at load 7–9.
- Earlier rounds, from `wall_s` in the indexes: bot demos 29.7, DAgger 20.6–45.7 ticks/s per process.
- With the lab's 4-process cap: **about 150 ticks/s ≈ 540k ticks/h ≈ 1,200 bot episodes/h ≈ 2 GB/h** of npz (3.4–3.9 KB per tick).
- Doubling the dataset costs about 2.5 h of recording.
- Recommended kind of extra data, if any: **noisy-bot episodes**, mirroring TwoRoom's "noisy heuristic". Hold random commands for 3–10 ticks on ~20% of ticks, then let the bot recover, on new k seeds of the training configs. This widens action coverage around purposeful paths.
- New training-family *rooms* would widen scene coverage more, but that means adding configs to the splits. It is the owner's call, and `check_splits` must pass.

## 4. MLX port cost

**Code to write**, all in a new `lewm/` folder; standalone, no torch:

| Module | Lines (est.) | Notes |
|---|---|---|
| Model | ~180 | ViT-tiny encoder, BatchNorm projectors, AdaLN-zero causal predictor, action encoder. `lewm/bench_step.py` already has a 145-line working version. |
| SIGReg | ~25 | Done in `bench_step.py`. |
| Frame cache and window sampler | ~100 | Decode every npz once into a uint8 memmap under `episodes/cache/lewm_*`. 64 px: 16 GB; 96 px: 37 GB. Disk use asks the owner first, per the cleanup rule. Windows are T=4 frames at frameskip k, with random phase; actions come from `cmd_exec`. |
| Training | ~150 | AdamW, warm-up then cosine by step, resume holding weights, optimiser state, step, RNG and sampler state, a 45-min guard, collapse logging (per-dimension std, effective rank, SIGReg statistic). |
| Evaluation and probes | ~150 | Prediction against controls, action ranking, linear probes. |
| CEM (S3 only) | ~60 | Measured below. |

That is about 500–650 lines for S1, plus 60 for CEM.

**Missing ops: none.** MLX 0.32.4 has `nn.BatchNorm`, `mx.fast.scaled_dot_product_attention` with causal masks, LayerNorm, GELU and SiLU. SIGReg needs only matmul, cos and sin.

**Traps already known on this stack (MICRODUCK_LESSONS 12):**
- `mx.compile` must get `inputs`/`outputs` = state, or the weights freeze silently. The benchmark ran eager for that reason.
- Put an epsilon on norms.
- `MLX_ENABLE_TF32=0` for any parity check.
- BatchNorm in MLX needs `model.train()`/`model.eval()` handled explicitly in probes and planning.

**Measured step times.** Setup:
- one full train step: forward, backward, clip, AdamW;
- 17.96M parameters, batch 128 windows × 4 frames = 512 frames per step;
- synthetic data, eager mode;
- M5 Max on AC power, load 6–9 from other jobs. These are lower bounds on speed.

| Frame / patch (tokens) | Precision | Median step | Frames/s | Peak memory |
|---|---|---|---|---|
| 96 / 8 (144) | fp32 | 0.50 s | 1,019 | **18.8 GB** (over the 16 GB cap) |
| 96 / 8 (144) | bf16 (whole model) | 0.29 s | 1,763 | 9.5 GB |
| 96 / 12 (64) | fp32 | 0.24 s | 2,114 | 9.6 GB |
| **64 / 8 (64)** | **fp32** | **0.235 s** | **2,177** | **9.5 GB** |
| 96 / 8 (144), batch 64 | fp32 | 0.28 s | 927 | 9.8 GB |

**S1 configuration: 64 px, patch 8, fp32.** It fits memory, avoids bf16 numerics in a first port, and Phase 2's lamp probe showed 64 px is enough (a small CNN reads the lamp at 0.953 at 64 px). 96 px bf16 is the fallback if 64 px probes fall short.

**Training time.**
- One 45-min run ≈ 11k steps at 0.235 s ≈ 1.4M windows.
- **Plan three resumable 45-min runs, ≈30k steps ≈ 2.3 h of compute.** That is about 3 passes over the ~1.33M start positions.
- Whether this matches the paper's budget is unknowable: the paper gives no step counts or GPU. "10 epochs" over PushT's 3.9M frames would be roughly 300k steps if an epoch means every window. So we may be 10× under; the S1 gates decide whether that matters.
- Real-data throughput will be lower than synthetic until the frame cache exists. Decoding npz in the loop would dominate.
- Building the cache takes an estimated 20–40 min on 4 processes; not measured.

**Planning cost (measured).**
- One CEM plan as in the paper (300 samples, 30 elites, 30 iterations, horizon 5, history 3) takes **1.44–1.58 s** eager in MLX with action blocks of 6–35 dims. That is comparable to the paper's 0.98 s.
- The simulator waits for the controller, so this is affordable at one plan per 2.5 s. Each step's observation age must still be logged (lesson 5).

## 5. What the world model is for here: recommended first use

**Recommendation: (b), a predictive representation first.**
- Train LeWM from scratch on the 1.33M existing frames.
- Gate it on prediction, action sensitivity and probes.
- If it passes, use its frozen encoder (optionally with the predictor's latent) as the image features of the Phase 2 recurrent policy and of the coming skill selector.
- Latent planning (a) becomes a later, narrow stretch (S3).

**Why not (a) first.** Three facts from the paper and our data decide it:

1. **Horizon.** LeWM plans 5 blocks × frameskip 5 = 25 steps. At 10 Hz that is **2.5 s, about 0.2–0.4 m** of duck travel (achieved speed is ~0.44× the 0.4 m/s command). Reaching the plate takes 10–50 s, often around a partition. The paper names short horizons as its main limit and points to hierarchy, which is our skill layer.
2. **Goal images.** LeWM's goal is a frame 25 steps later *in the same episode* (App F.1). In a fresh dev room there is no goal image without an oracle. A goal frame from a train-room demo would encode that room's walls, floor and lighting into the latent, and dev rooms deliberately change exactly those.
3. **Ceiling.** The hand-written skills already solve the short sub-goals a 2.5 s planner could handle (`press_button` 40/40; skill battery 100%). At best, (a) ties them in the skill-selection plan, where skills are given.

**Why (b) is worth testing.**
- The image encoder is a measured weak link: a ReLU encoder died (P2-A); the best lamp probe on frozen policy features reaches 0.837 (P2-A′) against 0.953 for a supervised 64 px CNN; P2-C′ reached 0.626.
- A JEPA encoder trained on action consequences is exactly what LeWM's probes say it is good at: positions linear in the latent.
- It is cheap to check without any closed loop.

**Honest caveat.** Astra's diagnosis and steps 2–5 point to *state, memory and stage*, not perception, as the main bottleneck. A better encoder may raise lamp scores without producing presses. S2's bars therefore count stages (reached plate, pressed, said done after a press), not just AUC. A no-gain result there would itself say the encoder isn't the limit. If S1's probes fail, stop before S2.

**Baselines and controls.**

*Prediction quality.* All prediction tests use held-out training configs 270–299, kept out of LeWM training, plus the `p2_dev` demos on unseen dev rooms.
- **Copy-last-latent**: ẑ_{t+1} = z_t. Report the predictor's error relative to this baseline within each encoder's own space. Raw MSE across encoders is not comparable.
- **Frozen random encoder + trained predictor**, same size and steps. This tests whether the encoder learned anything beyond what a random projection carries.
- **Action-ranking test** (Astra's option-6 check). Score the true action block against 15 blocks drawn from other ticks of other episodes. Also check that swapping left vs right turn blocks moves a yaw-rate probe in the right direction.

*Representations.* The same frames, labels and linear-probe code are used on every encoder:
- the LeWM encoder;
- the random-encoder control;
- P2-A′ s0 and s1 frozen features (`runs/p2_a2*/ckpt_016000`);
- R3 s1 (best discrete-head pick);
- the supervised 64 px lamp CNN (0.953) as the ceiling.

| Probe | Label | Measure |
|---|---|---|
| Lamp on/off | Existing `p2_enc_probe` protocol: bot fresh sightings, fit on configs 0–199, test 200–299 | Accuracy; current best 0.837 |
| Plate/button in view, and bearing | Bot `memory` track seen this tick (sensors-only label) | AUC; bearing r |
| Ball in view | Bot `memory` (sensors-only) | AUC |
| Pose: yaw change and head pose | Δyaw from `truth`, labelled evaluation-only and never a training signal; head angles from `body` | r |

*Behaviour (S2).*
- The **reference bot**: dev 82–97%, set-lamp press-needed 65–81%.
- The **best Phase 2 arm on 2 seeds**: R3 and H picks, 0–1 press-needed of 40 on the confirmation rooms.
- The **same recipe with its CNN** vs **with the frozen LeWM encoder**, matched steps, samplers and seeds.

**Evaluation sets.**
- Checkpoint selection uses only dev rooms outside the confirmation set, by the existing P2 selection rule.
- `p2_dev` (instances-dev 0010–0029, others 0000–0019, set_lamp only) is the prediction and probe check set.
- The **80 confirmation rooms** (instances-dev 0030–0049, others 0020–0039) are used once, for the S2 verdict. Phase 2 steps 4–5 already looked at them once, so S2 is a *second* look. That makes it a fair paired comparison against R3/H measured on the same rooms, but not a headline.
- The reserved test split stays untouched; M6 already spent the old one. A headline claim needs a new reserved split drawn first.

## 6. Staged plan

**S0: owner decisions (no compute).**
- Approve about 16 GB of disk for a 64 px frame cache under `episodes/cache/` (37 GB if 96 px).
- Choose whether noisy-bot recording (about 1 h, about 2 GB) happens before S1 or only if S1's action-ranking gate fails.
- No download is needed for this plan.

**S1: port, train and gate.** About 1 working day. Roughly half a day of code, a 20–40 min cache build, 3 × 45 min training through `gpu_run`, and about 30 min of evaluation.

Work:
- Write the MLX LeWM (64 px, patch 8, latent 192, history 3).
- Frameskip 5 (6×5 = 30-d action blocks), with frameskip 2 as a one-run ablation.
- λ 0.09, M 1024, AdamW 5e-5, batch 128.
- Hold out configs 270–299.
- Verify save/kill/resume before the first run over 20 minutes.
- Run a one-run random-encoder control with matched steps.

Done bars, all required:
- (i) No collapse: per-dimension std of z in [0.5, 2], effective rank ≥ 100 of 192, SIGReg statistic falling.
- (ii) One-block prediction error ≤ 0.7 × copy-last on held-out train configs, and ≤ 0.85 × copy-last on `p2_dev` dev rooms.
- (iii) Action ranking: top-1 ≥ 0.5 of 16, and ≥ 3× the random-encoder control.
- (iv) Lamp probe ≥ 0.90, against 0.837 for P2-A′. Plate-in-view AUC must be at least P2-A′'s on the same frames.

If (iii) fails: record noisy-bot data and retrain once. If (iv) fails: report and stop. Option (b) is then dead, and (a) would inherit the same weak latent.

**S2: frozen LeWM encoder inside the Phase 2 policy.** About 1 day.

Work:
- Swap P2's CNN for the frozen LeWM encoder (192-d features, the same width as now).
- Keep the R3/H recipe: discrete heads and the H sampler.
- 2 seeds × 16k steps through `gpu_run`, matched against the existing R3/H picks.
- Dev selection by the existing rule, then one confirmation pass on the 80 rooms.
- Pre-register the protocol before any S2 training.

Done bar, pre-registered against seed noise (the same recipe spans no-press 7–17 and AUC 0.58–0.84):
- each seed reaches ≥ 2 press-needed successes on the 80 rooms, and the two seeds sum to ≥ 5;
- R3/H reached 0–1;
- no rise in press-needed "said done without a press", the metric that exposed E's false gain;
- stage counts (reached plate, pressed, done after press) no worse.

If skill selection is built first, the same encoder can feed the selector instead. That is a separate one-day branch with the same probes.

S2 option (owner, 2026-11-04): a small latent world-action model, DreamZero-style; see `design/WAM_S2_OPTION.md`. **S1 outcome (RESULTS 'Round 6'): S1 failed its gates, so neither S2 arm runs.** S1's latents are not usable as WAM targets: lamp probe 0.50; action ranking 0.265 vs 0.283 for the random-encoder control; rollout error vs copy-last 0.77–0.80 over 1–5 blocks on held-out training rooms, 0.82–0.88 on dev rooms.

**S3 (stretch, only if S1 passes): narrow latent MPC.** About 1.5 days.

Task: the "final approach" sub-goal. Start from bot-driven prefixes in dev rooms 0000–0009, taking over at the first tick the plate is confirmed in view.

Planner:
- CEM over 6-d command blocks: horizon 5 × frameskip 5, replan each block.
- Cost from a learned *probe head* on the predicted latent: plate bearing and distance, fitted to bot `memory` labels. No goal images and no `truth`.
- An ensemble of 3 predictors with a disagreement penalty, and the fooled rate reported (lesson 1).

Baselines: the hand-written `go_to` skill (the ceiling) and the best P2 policy from the same starts.

Done bar: plate reached in at least 70% of the go_to skill's successes from the same starts, with a fooled rate ≤ 20%.

**Risks.**
1. **Appearance shift.** Dev rooms change floor and wall hues and light, and the encoder sees only 300 train-family rooms. Probes may pass on train configs and fail on dev rooms; S1 bar (ii) tests this early.
2. **Ego-motion dominates prediction.** Gait bob and head motion may let "predict the camera shake" satisfy the loss while object identity stays weak. The action-ranking test and the object probes guard against this.
3. **Budget.** About 30k steps may be ≥10× under the paper's budget.
4. **The bottleneck may not be the encoder.** S2 can fail with good probes. That is still an informative result.
5. **Short-horizon planner.** S3 is useful only inside the skill hierarchy; the paper's own limit.
6. **Disk.** A 16–37 GB cache comes on top of the existing 28 GB in `episodes/`.
7. **Paper-vs-code discrepancies.** λ, epochs, history and knot ranges disagree; we follow the code's values and say so.

---
Files: `lewm/bench_step.py` (timing only; `python lewm/bench_step.py --img 64 --patch 8`). The recording measurement wrote only to the session scratchpad.
