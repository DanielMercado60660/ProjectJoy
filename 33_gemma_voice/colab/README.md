# 33 on Colab: Gemma 3 270M + a Ming-style flow head (PyTorch)

A PyTorch rebuild of 33 for a Colab A100, with Ming-3.0-TTS's ideas:
- Ming's 64-d VAE latent (`inclusionAI/Ming-omni-tts-tokenizer-12Hz`) as the target, with Mimi as the fallback.
- A 0.1B transformer flow head that also sees the previous frame.
- Ming's 10-step sampler, with guidance applied by zeroing the backbone state.

Open `gemma_voice_colab.ipynb` in Colab. The run order and every setting are in its first two cells.

## Staged data expansion (2026-11-03)

Use **`gemma_voice_large.ipynb`** for the larger experiment. The existing live
Colab notebook has the same expansion and training gate. `gemma_voice_colab.ipynb`
is preserved as the original smoke workflow. The earlier `gemma_voice_10kh.ipynb`
is a historical snapshot of the initial plan. Do not run either PyTorch variant
locally under the current Joy Lab MLX-only rules.

Emilia-YODAS English is added from `amphion/Emilia-Dataset` at revision
`d7f2f7340a6385696f3766c8049fa920a4707c07`. Access was approved by the user and
verified with Colab's HF identity. The loader selects English, DNSMOS >= 3.3,
0.8–20 s, nonempty transcripts, finite decodable audio, and unique clip IDs.
The source archive is never extracted onto the filesystem. At most 512 clips'
decoded audio are retained per chunk. Mean/std posteriors are stored as fp16;
completed JSON/audio pairs are backed up to the existing private dataset repo.
Source video hashes assign 98% train / 1% val / 1% test. This groups clips from
one video; it does not prove speaker identity or content is disjoint across
different videos or corpora. Transcript accuracy has not been independently
verified with ASR, and DNSMOS is a quality proxy.

The current target is **2,500 actual accepted training hours**, excluding held-out
audio and LJ repetitions. No target is inferred from dataset labels, nominal
corpus size, or the 6,000-token **per-batch** memory budget. The three pilot
chunks contain 1,536 clips / 4.2102 training hours. Fresh remote download plus
SHA256 verification passed; continuation skipped these saved chunks correctly.

The user chose this 2,000–3,000-hour milestone before committing to 10,000+ hours
and explicitly authorized continuous preparation beyond the Joy Lab short-run
limit. Original corpora now finish first; Emilia tops up the mixed corpus to
2,500 hours. The final complete archive can overshoot the target slightly.
Later expansion reuses the same revision/policy and completed chunks. Quality
will be judged from held-out generated speech and listening before scaling.

In Colab:

1. Run configuration, setup and definitions. The HF account needs Gemma and
   Emilia access, plus `HF_TOKEN` in Colab Secrets.
2. Run the preparation cell, which calls `prepare_large_data(minutes=None)`.
   It continues to the target without timed pauses. Completed pairs restore from HF after a
   runtime loss. An interruption can require redoing the uncommitted chunks of
   the current archive (uploads occur at most every 16 chunks and at a clean stop).
3. `audit_large_data()` checks the target, original corpus completion, duplicate
   Emilia IDs and backup presence. Training remains blocked while incomplete.
4. Run the training cell. It restores an existing checkpoint, or first verifies
   exact recovery after forced worker termination and runs 100 calibration updates.
   It then automatically trains through the **first complete epoch (45,143 steps)**.
   Speech reviews occur at 5,000, 11,286, 22,572, 33,857 and 45,143 steps.
   Checkpoints save every ten minutes; bounded worker segments resume automatically
   inside the same cell, with no manual 40-minute pause. Review the first epoch's
   speech before extending to epochs two and three. The cosine learning-rate
   horizon is three epochs / 135,431 updates, with 500 warmup updates.
5. The last cell re-evaluates the saved full-data checkpoint without training.
   For a fresh Colab runtime, restore the data using the preparation cell first,
   then run the updated training cell in `gemma_voice_large.ipynb`. The old smoke
   notebook's training function is not compatible with this new checkpoint format.

Estimated latent storage from the format alone is about 29 GB per 2,500 h
(115 GB per 10,000 h), plus metadata, holdouts, downloads and model checkpoints. Actual total
preparation finished at 2,504.268 accepted train hours in a continuous invocation
of 293.44 minutes (some data existed before it started). A 100-update A100 calibration
measured 0.2995 seconds/update, 14.17 GiB peak GPU allocation, and 36,699 speaker IDs.
One epoch is estimated at 4–5 hours including checkpoint/evaluation overhead;
three at 12–14 hours. These estimates are not completed-run measurements.
Training quality on the expanded data is **not measured yet**.

Rebuild and check from `/Users/danielmercado/ProjectJoy/33_gemma_voice`:

```sh
../.venv/bin/python colab/build_large_notebook.py
../.venv/bin/python -m pytest -q colab/test_emilia_expansion.py
```

The builder combines the original notebook, `emilia_expansion.py`, and the saved
configuration/gate cells in `runs/2026-11-03/`. It embeds the expansion, so the
generated notebook needs no local Python files after upload. Eight CPU checks
cover corruption detection, grouping, filters, unique-hour counts, duplicate IDs
and retry after an interrupted upload, plus optional versus continuous deadlines.
These do not qualify model audio quality.

Sources: [official Emilia dataset card](https://huggingface.co/datasets/amphion/Emilia-Dataset)
and [Hugging Face upload documentation](https://huggingface.co/docs/huggingface_hub/guides/upload).

| file | what |
|---|---|
| `gemma_voice_colab.py` | the notebook source (`# %%` cells), also runnable on the laptop |
| `gemma_voice_colab.ipynb` | built from the .py by `python3 build_notebook.py` |
| `build_notebook.py` | .py → .ipynb |
| `mimi_parity.py` | transformers' Mimi vs 33's moshi_mlx Mimi on stored latents (see below) |

**Laptop smoke test** (Mimi codec, tiny head, Whisper tiny, local LJ; about 2 minutes):
```sh
GV_LOCAL=1 GV_WORK=/tmp/gv_local PYTORCH_ENABLE_MPS_FALLBACK=1 ../../.venv-xy/bin/python gemma_voice_colab.py
```

**Verified on the laptop (2026-11-03):**
- The full smoke path runs: codec gate, encode, train 15 steps, resume at 15 (the LR schedule continues), finish 30, generate, judge.
- PyTorch Gemma 3 270M gives the same greedy text as 33's MLX port.
- Batched generation matches single-clip generation within 2e-6.
- Encoding:
  - Mimi through transformers changes short clips when they are batched with right padding (11–16% error). The notebook checks this for any codec and falls back to one clip at a time.
  - transformers' Mimi encoder differs from moshi_mlx's on the semantic half of the latent (74%; the acoustic half is within 5%, and the projection weights are identical). So 33's stored Mimi latents are not reused.
- **Not run here:** the Ming codec path. The permission check blocks running Ming's cloned code on the laptop, so it runs first on Colab. The gate cell asserts that the batched encoder matches Ming's own forward before scoring.

## Verified full-data training (2026-11-04)

Run `g270m_2500_v1` uses BF16 autocast, fused AdamW, SDPA, length buckets,
backbone LR 5e-5 / new-layer LR 2e-4, gradient norm cap 1, and joint backbone/head
training with the text embedding table frozen. A 6,000 padded-token budget fits
comfortably in A100 memory. Megatron's multi-GPU partitioning is not needed here;
activation recomputation/offload are deferred unless profiling demonstrates a need.
No Megatron package or optimizer experiment was installed.

`training_runtime.py` freezes metadata hashes, shard sizes, vocabulary, speaker
mapping, normalization and the training policy. Its checkpoint contains optimizer,
Python/NumPy/Torch/CUDA RNG, exact epoch/batch cursor and observed audio frames.
Posterior samples are keyed by update, independent of process/session length.
The explicit cursor handles the measured unequal epoch lengths [45143,45144,45144].
The recovery test saved step 3, computed an uninterrupted step 4, terminated the
worker with SIGTERM, then reproduced step 4 in a fresh process byte-for-byte over
all trained weights, optimizer tensors, RNG and cursor. This is a process-recovery
check on the same Colab filesystem; a full VM replacement has not been simulated.
Six local CPU checks cover frozen-identity changes and posterior-sampling recovery.

`training_evaluation.py` retains WAVs and CER/WER/UTMOS, length ratio and runaways
for LJ val (24 early / 100 at an epoch boundary) plus 16 unseen Emilia transcripts
spoken with LJ's trained voice. The latter is text generalization, not a zero-shot
speaker test. Each report records the HF revision containing its checkpoint, so
an earlier model can be recovered if a later model worsens. Held-out test clips
are not used for these training decisions.

The worker and evaluation source are copied into the private checkpoint repo
under `g270m_2500_v1/code/`. Fresh-runtime recovery also requires the encoded data
and the installed Colab dependencies. Rebuild the embedded worker and notebook:

```sh
../.venv/bin/python colab/build_large_notebook.py
../.venv/bin/python -m pytest -q colab/test_training_runtime.py --basetemp=colab/.test_tmp
```

The current first-epoch cell is `runs/2026-11-04/train_first_epoch.py`; the combined
restore/check/calibrate/train cell is embedded in `gemma_voice_large.ipynb`.

## Epoch-two continuation (2026-11-04)

The user approved the second epoch only, ending at step **90,287**, while retaining
the existing three-epoch learning-rate horizon (135,431). On a replacement Colab VM,
paste/run `runs/2026-11-04/restore_epoch_two.py` first, then
`runs/2026-11-04/train_second_epoch.py`. The former restores the saved checkpoint,
the exact verified worker/evaluator and the 1,683 encoded shards from private HF;
it does not re-encode audio. It restores the original Colab dependency version and
pinned Ming source. These scripts target Colab, not local MLX execution.

The driver saves every ten minutes, automatically continues bounded worker
segments, checks speech at 56,429 / 67,715 / 79,001 / 90,287, and stops before
epoch three. A failed speech review is retried before more training. The final
comparison checks that the complete LJ-100 and Emilia-16 panels have identical
IDs and reference texts, then displays the same four previously selected clips
from epochs one and two. Speaker, batching, random seed and decoding are unchanged.
The first-epoch baseline is pinned to HF revision
`7dd93c8f0005e72ab32725acf534da189123a118`; its WAVs are also saved locally in
`runs/2026-11-04/listening/`.

The replacement VM received an L4 because Colab reported the A100 unavailable.
Training throughput and numerical behavior across that hardware change still
need live verification; the same-GPU exact recovery check does not establish
bitwise equivalence across GPUs. The launcher itself passed CPU checks:

```sh
../.venv/bin/python colab/runs/2026-11-04/test_epoch_two_driver.py
```


## Epoch-three continuation (2026-11-04)

The user approved epoch three and an A100 runtime. The live runtime reports
NVIDIA A100-SXM4-80GB. The previous L4 run finished epoch two at 90,287; its
checkpoint/evaluation revision is `06e601de2adf13957ee55d1f7719ad7a669c7909`.
Paste/run `runs/2026-11-04/restore_epoch_three.py` in cell 16, then queue
`runs/2026-11-04/train_third_epoch.py` in cell 15. Do not run all notebook cells.
Restoration reuses encoded shards and restores the exact frozen worker/evaluator.
A readiness marker prevents queued training if restoration fails.

Epoch three stops at **135,431** with epoch=3, batch=0, preserving the original
three-epoch learning-rate schedule, token budget, dataset identity and optimizer.
Speech reviews occur at 101,573 / 112,859 / 124,145 / 135,431. The final review
uses LJ-100 and Emilia-text/LJ-voice-16, compares the same four listening IDs
against epoch two, uploads `epoch_three_comparison.json`, then prints
`THIRD_EPOCH_READY`. No epoch four is authorized. Intermediate LJ panels use 24
clips. Automated scores do not establish broad unseen-speaker performance;
GPU hardware differs between epochs two and three.

Epoch three completed at step 135,431. The final A100-SXM4-80GB worker segment
measured 0.287 s/update median (historical L4: 0.752 s/update). BF16, SDPA and fused
AdamW remained enabled with the unchanged recipe. Final scores, caveats and
provenance are in `../RESULTS.md`; audio and reports are under
`runs/2026-11-04/listening/`, with `epoch_three_listening_manifest.json` identifying
the same four comparison clips. The training monitor is paused for review.

Three CPU controller checks cover the final boundary, failed-review retry before
training, and no additional training after completion:

```sh
../.venv/bin/python colab/runs/2026-11-04/test_epoch_three_driver.py
```
