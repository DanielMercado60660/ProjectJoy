# 28 · Vocoder Thunderdome — round 6: a GAN-trained vocoder on the fixed mel

**Headline.** A GAN fine-tune of the round-3 generator on full LJSpeech (19.4 h) lifts real-mel → vocoder UTMOS on 27's 100
test clips from **2.59 to 3.05** (paired +0.46 [+0.42, +0.51], higher on 98/100 clips). On 28's 24 clips it goes from 2.66 to
3.16 (+0.50 [+0.40, +0.60]). The adversarial loss is what does it: the same generator, data, batches and 24k steps with the
old reconstruction loss gain only +0.06. The shipped model (`checkpoints/r6_gan`, GAN + 10 × MR-STFT, 24k steps) is a
drop-in on the same mel, the same 7.74 M architecture and the same speed.
**It does not close most of the gap** to the recording (4.40). It closes about a quarter (0.46 of 1.81 UTMOS).
**It is worse on pitch.** YIN gross pitch error is 5.5% vs 4.9% for the old vocoder (+0.59 [+0.33, +0.87]) and 4.2% for
Griffin–Lim (+1.36 [+1.09, +1.62]). The GPE ≤ GL bar is **not met**, a documented gap of ~1.4 points (1.7 on 28's clips).
**Why the gap remains** is the round's clearest finding. Swapping in the recording's true phase lifts the *old* generator's
own magnitudes to UTMOS 4.20. The whole gap is phase, and 24k GAN steps recover only part of it.
By the pre-registered swap criterion (UTMOS CI > 0 *and* GPE not significantly worse), the swap into 27 is **the owner's
call after listening**, not an automatic yes.

## Setup actually run

**The mel is unchanged** (16 kHz, n_fft 1024, Hann 1024, hop 256, 80 Slaney bands 0–8 kHz, ln(max(·, 1e-5))), so the
generator drops into 27 and 33-laptop as is. `r6_vocoder.py` carries its own copy of the mel maths; `test_r6.py` checks
it is bit-identical to `mel.py`, and that the round-3 checkpoint loads into it and gives identical output.

**Data** (`r6_prepare.py` → `cache_r6/`, 3.5 GB): every clip of LJSpeech chapters LJ001–LJ040 (10,674 clips, 19.4 h; earlier
rounds used 400 or 3,000 of them). Nothing from LJ041–LJ050 is trained on, so 27's validation (LJ041–044) and test
(LJ045–050) clips and 28's own 24 held-out clips (LJ045–050) are all unseen. **Multi-speaker data was not used for training**
(two arms plus pilots filled the GPU slots). Two unseen-speaker sets are scored instead: LibriTTS-R train-clean-100, the 6
speakers 27/33 hold out (52 clips; one speaker had fewer than 10 clips of 2–10 s; LibriTTS-R is CC BY 4.0, there is no licence
file in `data/libritts_r`), and Hi-Fi TTS (CC BY 4.0, `LICENSE.txt` in the folder), 5 test-manifest clips for each of its 10
speakers (50 clips). No vocoder here has seen any of those speakers.

**Arms** (`r6_train.py`). All warm-start from `checkpoints/followup_val3k/best`, the round-3 generator (7.74 M params,
dim 384 × 8 ConvNeXt blocks), which is exactly what 27 ships as `vocoder28`. All use the same crops and batches
(`np.random.default_rng([seed, step])`, batch 8 × 1.02 s) and the same G optimiser (AdamW 2e-4, betas 0.8/0.99, cosine to
2e-5 over 24k steps). Each arm runs 24k steps unless noted.
- **recon** (the equal-budget non-adversarial control): the round-1/3 loss (MR-STFT at 512/1024/2048 + 2 × mel L1).
- **gan**: Vocos recipe. MPD (periods 2/3/5/7/11, channels 32-128-256-512-512) + MRD (log-|STFT| at 512/1024/2048,
  16 channels), all convs hand-weight-normalised. Hinge losses; G = adv(MPD) + 0.1·adv(MRD) + FM(MPD) + 0.1·FM(MRD) +
  45 × log-mel L1. D has 10.9 M params, AdamW 2e-4.
- **gan_mrstft**: gan + 10 × MR-STFT (SC + log-magnitude at 512/1024/2048). One variable changed vs gan. Launched after the
  plain GAN's pitch error rose; the reason is in "Why gan_mrstft" below.
- pilots: **gan_phase** (gan + 2 × APNet-style phase loss, stopped at 4,007 steps), **recon_phase** (recon + phase loss,
  24k steps). Both are in "Things that did not work".

**Compute.** GAN step 2.1–4.5 steps/s depending on load; the MRD's 2-D convs dominate. A 1-D-conv MPD and fp16/bf16 were
not faster in MLX. Recon runs at 35–90 steps/s. Peak memory is 2.1 GB per process. Wall time: gan 153 min, gan_mrstft
109 min, recon 12 min (each including in-training validation), in ≤43-min chunks chained by `r6_chain.sh`. `r6_bench.py` and the demo build ran outside `gpu_run` (each under 2 min).
Resume was verified before the long runs: SIGTERM at step 104, resume, and losses at steps 150/200 matched an
uninterrupted run within GPU noise (mel L1 0.1606 vs 0.1603). `r6_smoke.sh` repeats a stop/resume (stop at step 6, resume to 12) at tiny size.

**Measurement.** UTMOS22-strong (`r6_utmos.py`) runs with torch in `../.venv-sac`. It loads only from the local hub cache,
with network calls blocked in-process; nothing was downloaded. It reproduces 27's numbers on 27's test clips: recording
4.396 (27: 4.40), GL 64 2.03 (2.03), old vocoder 2.589 (2.59). Pitch uses 28's YIN (`metrics.py`, 60–400 Hz, 10 ms hop).
GPE is the share of frames voiced in both signals with >20% f0 error. mel L1 and MR-STFT are as in rounds 1/3. mel L1 is
GL's own objective and is in every neural loss. MR-STFT is in the recon and gan_mrstft losses. On test28, GL 64 and the old
vocoder reproduce RESULTS.md (GPE 3.56% and 4.52%). test27 and test28 are both drawn from LJ045–050 and share
3 clips (LJ045-0120, LJ048-0021, LJ048-0032), so they are not fully independent replications. **UTMOS is a proxy** trained on other systems' listening tests, with known
biases. 27 found it rates a TTS above the real mel through the same vocoder. It is also the selection criterion here
(double use; see Caveats). No human listening test was run.

**Selection (pre-registered in STATUS.md at 18:19, before any val score).** Candidates per arm are the snapshots every 4k
steps plus the last one. Each arm's pick is the candidate with the highest val100 UTMOS (27's 100 validation clips, real
mel). The shipped generator is the best of those picks over the GAN arms. Test sets were scored once at the end. The
swap-into-27 criterion was also written in STATUS.md before any new-model test number: recommend swapping only if the
test27 paired UTMOS (shipped − old) CI is > 0 *and* the paired GPE CI is not entirely > 0.

## 1. The diagnosis: the whole gap is phase (`r6_oracle.py`, val100)

The generator emits, per 16 ms frame, a log-magnitude and a phase for 513 bins on exactly the analysis STFT's frame grid
(`istft(target_stft(y)) == y` to 3.6e-7, `test_r6.py`). So either half can be swapped with the recording's own STFT:

| val100 (100 clips, LJ041–044) | UTMOS [95% CI] | GPE % |
|---|---|---|
| recording | 4.40 [4.38, 4.42] | – |
| true magnitude + true phase (sanity: equals the recording) | 4.40 | 0.0 |
| **old generator's magnitude + true phase** | **4.20 [4.16, 4.23]** | 5.2 |
| true magnitude + old generator's phase | 2.33 [2.27, 2.39] | 8.4 |
| old generator as is | 2.58 [2.51, 2.64] | 5.6 |
| GAN arm at 4k: its magnitude + true phase / true magnitude + its phase | 4.12 / 2.46 | 7.0 / 9.2 |
| Griffin–Lim 64 (NNLS magnitude) | 2.09 [2.03, 2.16] | 4.3 |

With the right phase, the old vocoder's magnitudes score 4.2; its own phase drags it to 2.6. With true phase, those
magnitudes come out a near-constant +3.1 dB too loud (median per-clip log-mel offset 0.36 ln, sd 0.06; mel L1 0.374 → 0.179
once the offset is removed). The likely reason: overlap-adding frames whose phases disagree cancels energy, and the head
learned to inflate magnitudes to compensate. On a training batch the generator's phase output was barely related to the
true phase (APNet phase distance 4.53, vs 4.72 for uniformly random phase). Every non-adversarial round so far was therefore
capped by phase, which no magnitude loss constrains directly.

## 2. Training arms on validation (val100 UTMOS; GPE / mel L1 / MR-STFT on the same 100 clips)

| arm | step | val UTMOS [95% CI] | GPE % | mel L1 | MR-STFT |
|---|---|---|---|---|---|
| old (= step 0 of every arm) | 0 | 2.578 [2.51, 2.64] | 5.61 | 0.148 | 0.833 |
| recon | 4k / 8k / 12k / **16k** / 20k / 24k | 2.578 / 2.608 / 2.597 / **2.645** / 2.638 / 2.645 | 5.85 → 5.01 | 0.153 → 0.135 | 0.835 → 0.809 |
| gan | 4k | 2.649 [2.60, 2.70] | 7.83 | 0.165 | 0.997 |
| gan | 8k | 2.795 [2.74, 2.85] | 7.29 | 0.167 | 0.998 |
| gan | 12k | 2.890 [2.83, 2.94] | 7.10 | 0.162 | 0.990 |
| gan | 16k | 2.979 [2.92, 3.04] | 6.90 | 0.159 | 0.985 |
| gan | 20k | 3.027 [2.97, 3.08] | 6.81 | 0.156 | 0.986 |
| gan | **24k** | **3.036 [2.98, 3.09]** | 6.95 | 0.155 | 0.986 |
| gan_mrstft | 4k | 2.841 [2.78, 2.90] | 6.40 | 0.164 | 0.913 |
| gan_mrstft | 8k | 2.909 [2.85, 2.97] | 6.37 | 0.167 | 0.920 |
| gan_mrstft | 12k | 2.977 [2.92, 3.03] | 6.36 | 0.162 | 0.918 |
| gan_mrstft | 16k | 3.034 [2.98, 3.09] | 6.16 | 0.160 | 0.914 |
| gan_mrstft | 20k | 3.063 [3.01, 3.12] | 6.29 | 0.156 | 0.913 |
| gan_mrstft | **24k (shipped)** | **3.068 [3.01, 3.13]** | 6.22 | 0.155 | 0.913 |
| gan_phase (pilot) | 4k | 2.603 [2.55, 2.65] | 7.96 | 0.170 | 1.002 |
| recon_phase | 24k | 2.302 [2.23, 2.37] | 5.46 | 0.152 | 0.831 |

Curves: `media/r6_val_curves.png`. The recon rule tie (16k and 24k both 2.645 to 3 decimals) went to the earlier
snapshot, as the rule's `max` does; the unselected last snapshot is reported on test too.

- **The GAN is what moves UTMOS.** With identical init, data, batches and steps, the reconstruction-loss control gains
  +0.07 on val (all of it data + steps) while the GAN gains +0.46. The GAN's gain was still rising at the end of its
  schedule but flattening (+0.05 from 16k to 20k, +0.01 from 20k to 24k as the LR annealed to 2e-5).
- **The GAN costs pitch accuracy at first.** Plain GAN val GPE jumped from 5.6% to 7.8% within 4k steps and only crept back
  to ~6.9%.
- **Why gan_mrstft.** The Vocos recipe drops the MR-STFT loss. Its 2048-point term spans 8 hops of the output, so it is only
  low when consecutive frames' phases cohere. That was the old model's only (indirect) phase supervision. Putting it back
  alongside the GAN gives higher UTMOS at every matched step (+0.19, +0.11, +0.09, +0.06, +0.04, +0.03 at 4/8/12/16/20/24k), lower GPE
  (6.2–6.4% vs 6.8–7.8%) and MR-STFT ~0.07 better. The UTMOS lead shrinks as both arms train.

## 3. Held-out results (scored once)

Real mel → vocoder; every system scored once on each set after selection on val. Full tables with paired CIs vs the old
vocoder, the no-GAN control and the plain GAN: `results_r6/tables_test.md` (+ `.json`). The shipped model vs GL 64:
`results_r6/tables_vs_gl.md`.

**27's 100 test clips (LJ045–050)**

| system | UTMOS [95% CI] | Δ UTMOS vs old [95% CI] | GPE % | Δ GPE vs old | f0 RMSE ¢ | mel L1 | MR-STFT |
|---|---|---|---|---|---|---|---|
| recording | 4.40 [4.38, 4.41] | +1.81 | – | – | – | – | – |
| Griffin–Lim 64 | 2.03 [1.96, 2.10] | −0.56 [−0.61, −0.51] | **4.17** | −0.76 [−1.03, −0.50] | **35.0** | **0.101** | 0.934 |
| old neural (round 3; what 27 uses) | 2.59 [2.53, 2.64] | 0 | 4.93 | 0 | 36.9 | 0.143 | 0.820 |
| no-GAN control, 16k (val-selected) | 2.64 [2.59, 2.70] | +0.06 [+0.04, +0.07] | 4.71 | −0.22 [−0.43, −0.01] | 36.2 | 0.133 | 0.803 |
| no-GAN control, 24k (last) | 2.66 [2.60, 2.71] | +0.07 [+0.05, +0.09] | 4.44 | −0.49 [−0.71, −0.27] | 35.4 | 0.129 | **0.797** |
| GAN, Vocos recipe, 24k | 3.01 [2.95, 3.07] | +0.42 [+0.37, +0.46] | 6.63 | +1.70 [+1.39, +2.03] | 41.0 | 0.150 | 0.976 |
| **GAN + MR-STFT, 24k (shipped)** | **3.05 [3.00, 3.10]** | **+0.46 [+0.42, +0.51]** | 5.52 | +0.59 [+0.33, +0.87] | 38.5 | 0.150 | 0.897 |

**28's 24 held-out clips (LJ045–050)**

| system | UTMOS [95% CI] | Δ UTMOS vs old | GPE % | Δ GPE vs old | mel L1 | MR-STFT |
|---|---|---|---|---|---|---|
| recording | 4.40 [4.36, 4.43] | +1.74 | – | – | – | – |
| Griffin–Lim 64 | 2.14 [1.98, 2.29] | −0.53 [−0.64, −0.42] | **3.56** | −0.95 [−1.47, −0.44] | **0.100** | 0.936 |
| old neural | 2.66 [2.55, 2.78] | 0 | 4.52 | 0 | 0.142 | 0.825 |
| no-GAN control, 16k | 2.72 [2.62, 2.83] | +0.06 [+0.02, +0.11] | 4.18 | −0.33 [−0.69, +0.02] | 0.132 | 0.810 |
| GAN, Vocos recipe, 24k | 3.13 [3.02, 3.23] | +0.46 [+0.38, +0.55] | 6.29 | +1.77 [+1.20, +2.41] | 0.149 | 0.984 |
| **GAN + MR-STFT, 24k (shipped)** | **3.16 [3.04, 3.27]** | **+0.50 [+0.40, +0.60]** | 5.27 | +0.75 [+0.25, +1.31] | 0.149 | 0.901 |

**Unseen speakers** (no vocoder here trained on any of them; LJ-only training)

| set | recording | GL 64 | old | no-GAN control | GAN | **shipped** | shipped − old (UTMOS) | GPE % old / shipped / GL |
|---|---|---|---|---|---|---|---|---|
| LibriTTS-R, 6 held-out speakers (52 clips) | 4.24 | 1.88 | 2.07 | 2.13 | 2.63 | **2.70** | +0.63 [+0.55, +0.72] | 17.1 / 13.7 / 10.2 |
| Hi-Fi TTS, 10 speakers (50 clips) | 3.98 | 1.62 | 1.95 | 1.96 | 2.44 | **2.49** | +0.54 [+0.45, +0.63] | 9.3 / 9.0 / 4.5 |

What the held-out numbers say:
- **The GAN's contribution, isolated:** shipped vs the equal-step no-GAN control is +0.41 [+0.36, +0.45] UTMOS on test27 (95/100
  clips). Data + steps alone (control vs old) is +0.06. Plain GAN vs control is +0.36. Adding MR-STFT to the GAN adds
  +0.04 [+0.01, +0.07] UTMOS and cuts GPE by 1.1 points [−1.40, −0.84] vs the plain GAN (test27); on test28 +0.04 [−0.03, +0.11] and −1.0.
- **Pitch is the cost.** Every GAN arm has higher GPE and f0 RMSE than the old vocoder and GL on LJ. MR-STFT recovers about
  two thirds of the plain GAN's GPE damage, not all of it. Voicing error (VDE) is unchanged within noise (+0.09 [−0.05, +0.23]).
- **mel L1 is flat** (0.150 vs 0.143) and MR-STFT is worse (0.897 vs 0.820). The GAN trades exact spectral match for texture
  that UTMOS likes. Both measures are partly training objectives (see Measurement), so neither is a neutral quality judge.
- **Unseen speakers gain the most UTMOS** (+0.54 to +0.63) and the shipped model is *better* than the old one on LibriTTS-R
  GPE (−3.5 points). Every LJ-trained neural vocoder is still far from GL's pitch on unseen voices, and the absolute
  scores (2.5–2.7) say this is still an LJ vocoder.

## 4. Things that did not work (all reported)

| attempt | what | val100 UTMOS | val GPE % |
|---|---|---|---|
| recon_phase | recon loss + direct phase regression (APNet anti-wrapping IP/GD/IAF, weights 0.2/1/1), 24k steps | 2.30 [2.24, 2.37] | 5.46 |
| GL on neural magnitude | old generator's magnitude, 32 fast-GL iterations started from its own phase | 2.21 [2.14, 2.27] | 7.87 |
| GL on neural magnitude | same with the recon-24k generator | 2.31 [2.25, 2.38] | 7.69 |
| gan_phase | gan + 2 × phase loss (weights 1/1/1), stopped at 4,007 steps | 2.60 [2.55, 2.65] | 7.96 |

- Direct phase regression makes things *worse* (2.30 < old 2.58), even though the oracle says phase is the whole gap. A
  plausible reading (not tested further): true phase is only partly predictable from an 80-band mel, because absolute phase
  depends on where glottal pulses sit inside a 16 ms hop. A per-bin distance to a partly unpredictable target then pulls the
  output toward a compromise. The adversarial loss asks only for *plausible* phase, which is what the oracle says is missing.
- Classical phase retrieval on the neural magnitudes is worse too. Those magnitudes are ~3 dB inflated and not a consistent
  spectrogram, so GL drifts (GPE 7.7–7.9%).
- Adding the phase loss to the GAN gave nothing at 4k (2.60 vs 2.65 for gan at 4k, GPE 8.0 vs 7.8%). It was stopped to free
  the slot. Its phase term only fell from 4.3 to 3.9.

## 5. Speed (`results_r6/bench_rtf.json`, 28's 24 clips, median of 3, under load: load average 7.3–9.1)

| method | RTF | × real time |
|---|---|---|
| Griffin–Lim 64 (numpy, CPU, 4 threads) | 0.057 | 17× |
| round-6 generator, MLX GPU | 0.00042 | 2,380× |
| round-6 generator, MLX CPU | 0.0070 | 142× |

The architecture is unchanged from round 3 (7.74 M params, frame-rate ConvNeXt + iSTFT), so speed is unchanged: the
round-3 idle numbers (GPU RTF 0.00030, CPU 0.0035) are the better reference. These numbers were taken at about twice the load.
Discriminators are training-only.

## 6. Listening demo

`media/demo_r6.html` (self-contained, offline): 6 held-out LJ clips (test27, evenly spaced through the id list, no picking
on scores) plus 2 clips from unseen LibriTTS-R speakers. Each clip has the recording, GL 64, the old neural vocoder, the
no-GAN control and the shipped GAN vocoder, shuffled per clip with the recording's slot balanced, all at −23 LUFS
(BS.1770). A reveal button shows labels, per-clip UTMOS / GPE / mel L1 and spectrograms. Listener preference:
**not measured**.

## Caveats

- **Double use of UTMOS.** Checkpoints were selected by val UTMOS, and UTMOS is the headline. Test is held out, but the
  selection favours whatever UTMOS likes. The rule mostly picked the last or near-last snapshot. The unselected last
  snapshots are reported next to the picks.
- **GPE and UTMOS disagree.** Every GAN arm is worse than the old vocoder and than GL on YIN GPE. YIN's gross errors are
  octave/subharmonic jumps. A GAN that adds plausible aperiodic detail can trip them without being audibly off-pitch, but it
  may also be audibly off-pitch. That is what the demo and a listening test are for. Not resolved here.
- One seed per arm. The CIs cover clip sampling only, not training-seed variance.
- Unseen-speaker sets are small (52 and 50 clips) and every vocoder here is LJ-only, so read them as robustness checks,
  not as a multi-speaker vocoder.
- I briefly ran 3 jobs at once (≈19:35–19:38: the oracle's UTMOS tail overlapped the recon_phase launch), against the
  2-job limit.

## Not done / next

- **TTS mels.** All numbers are real mel → vocoder. Whether the gain carries to 27's flow-matching mels (27 found its short
  GAN fine-tune helped v1's mels but not B/C's) is **not measured**. 27 keeps only vocoded wavs, not mels, and this round
  could not run 27's code.
- Longer GAN training. Both GAN arms were still improving slowly at 24k steps, and Vocos-style recipes usually run ~1M steps.
- A larger generator (not tried; this round kept the warm-startable 7.7 M architecture) and a full-width MPD (1024 channels).
- Multi-speaker training (LibriTTS-R / Hi-Fi TTS are on disk and licensed CC BY 4.0) to lift the unseen-speaker numbers.
- A human listening test on `media/demo_r6.html`.
