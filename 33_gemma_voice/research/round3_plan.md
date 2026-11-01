# Round 3 plan: better codec, T5Gemma 2, faster training, more data (2026-10-31)

User direction: Gemma looks like the better long-term bet after 27 v2. Before building, research: speed (nanochat-mlx),
codec (Qwen3-12Hz or XY), T5Gemma (proven for expressive TTS by Aratako), and whether to get more data or add epochs.

## 1. Codec: Mimi at 4 codebooks caps how natural we can sound

New measurement (UTMOS22-strong via 27's `mos.py`, CPU; val-100 round trips; `results/utmos/val_*.json`), next to the
existing Whisper ceilings (`results/phase0_summary.md`, STATUS 18:05):

| audio (val-100) | UTMOS | Whisper CER / WER % | streams per 12.5 Hz frame | runs on |
|---|---|---|---|---|
| real recording | 4.40 | 0.23 / 1.06 | – | – |
| **Qwen3-TTS-Tokenizer-12Hz, 16 cb** | **4.44** | **0.32 / 1.18** | 16 × 2048 | MLX (mlx-audio), 24 kHz in/out |
| SAC (semantic 12.5 Hz + acoustic 25 Hz) | 4.44 | 0.76 / 1.57 | 3 | torch CPU, 1.7 s/clip encode, 16 kHz |
| XY-Tokenizer, 8 cb | 4.26 | 0.41 / 1.23 | 8 × 1024 | torch CPU, 16 kHz in |
| Mimi, 16 cb | 4.23 | 0.40 / 1.29 | 16 × 2048 | MLX |
| Mimi, 8 cb | 3.92 | 0.55 / 1.51 | 8 | MLX |
| **Mimi, 4 cb (what r5 uses)** | **3.11** | 0.70 / 2.35 | 4 | MLX |
| real mel → 28's vocoder (27's audio path) | 2.58 | 0.27 / 1.06 | – | – |
| r5 snap 16000, t0.4, single take | 2.30 | 2.00 / 4.14 | – | – |
| r5 snap 16000, t0.4, best-of-4 | 2.34 | 1.84 / 3.92 | – | – |

- **Mimi-4's ceiling is 3.1**; r5 sits at 2.3 under it. 27 v2's best (test) is 2.96, under its vocoder's 2.58–2.73
  ceiling. 27 can't rise much without a new vocoder; 33 can, by changing what it predicts.
- Caveat: UTMOS resamples to 16 kHz, so it can't hear detail above 8 kHz. It also isn't checked against listeners
  (27 v2 noted it rates some vocoded audio above real mel). Listening is the final check.

Three separate questions, which the table mixes:

**(a) How many codebooks.** Most of the gap is codebook count, not codec: Mimi 4 → 8 → 16 is 3.11 → 3.92 → 4.23, and
Qwen-16 is 4.44, only 0.2 above Mimi-16. `data/*codes32.npz` already store 32 Mimi codebooks, so `--nq 8` is a flag
on existing data: no re-encode, no new code. Cheapest test of "more codebooks, same model".

**(b) Which codec at equal codebooks.** Qwen3-12Hz vs Mimi at 16 (4.44 vs 4.23 UTMOS, 0.32 vs 0.40 CER). Both are
near-lossless, so the deciding question is which one Gemma can *predict* better: a short text-to-token probe on the
same data. For Qwen: native 24 kHz, MLX, encodes as fast as Mimi (20 clips in 0.51 s measured today; Mimi did 283 h in
74 min), Apache-2.0, and its first codebook is semantic. Its decoder needs all 16 codebooks (Mimi can stop at any
count). Against XY: CPU torch, 16 kHz input, 8 codebooks, 4.26. SAC: ~1.7 s/clip CPU encode (days for 1k h), 16 kHz out.
Switching codec means re-encoding everything (~2 h for what we have); shangeth's Mimi codes then stop mattering.

**(c) How to emit many codebooks per frame.** Today: delay pattern, one head per codebook, cb k delayed k frames.
At 16 codebooks that's 15 frames (1.2 s) of extra delay and 16 independent heads. Alternative, used by Qwen3-TTS
(verified in mlx-audio: a 28-layer "talker" plus a 5-layer `code_predictor`, `qwen3_tts/config.py:33`) and by Moshi:
the backbone emits one hidden state per 12.5 Hz frame and a small *depth* transformer predicts cb1..cbN in order
within the frame. Applies to either codec. Worth it at 8+ codebooks, where Mimi's late residuals were already hard.

**(d) Skip codebooks: predict Mimi's continuous latent (user link, 2026-10-31).** `gradium/phonon-31m-en` (Apache-2.0,
24 kHz, 31M params, builds on Kyutai's Pocket TTS + Mimi): a flow-matching transformer predicts the *continuous* Mimi
latent one 80 ms frame at a time, and Mimi's decoder renders it. That removes the codebook-count ceiling entirely
(no quantization), keeps 12.5 Hz streaming, and needs no re-encode beyond saving Mimi's pre-quantizer latents. It
reports 1.5% LLM-WER on its hard set and Elo 2098 from 8,290 blind votes; training data and recipe aren't disclosed.
For us: Gemma backbone + a small per-frame flow head in place of the codebook heads (27 v2's flow-matching decoder
already works locally). Strong candidate to test against (b)+(c).
KittenTTS (other link): a 1.7B 1-bit speech LM plus 15–80M ONNX models; no training code, data or architecture
details released, so nothing to borrow for 33 beyond "1-bit/quantized TTS runs on CPU" (more a project-35 note).

## 2. Backbone: T5Gemma 2 270M-270M

- `google/t5gemma-2-270m-270m`, gated under the Gemma terms (same as Gemma 3 270M, already accepted). 1.57 GB bf16, of
  which ~835 MB is a frozen SigLIP vision tower we drop. Text model: encoder 100M + decoder 100M + one shared 168M
  embedding.
- Each stack has exactly Gemma 3 270M's shape (18 layers, d 640, 4q/1kv, head_dim 256, 5 sliding + 1 global). Both
  were initialized from Gemma 3 270M and adapted with UL2 on ~2T tokens. Global RoPE has linear scaling 8.
- **Merged attention**: each decoder layer has one attention whose keys/values are [decoder tokens; encoder states],
  through the same W_k/W_v, one softmax. Only decoder q/k are rotated; encoder keys get no RoPE and no window. For
  PM-RoPE (Aratako's load-bearing piece) we add progress rotation to encoder keys ourselves.
- No MLX port exists (mlx-audio has encoder-only T5Gemma 2 code). Our `gemma.py` blocks cover both stacks; new work is
  the merged attention, bidirectional encoder, weight loading and a parity check vs transformers.
- Compute per step ≈ today's: the encoder runs once over ~100 text tokens; the decoder is the same size as now.
- Our decoder-only `--prefix_lm 1 --progress 2000` flags (built, verified) are a cheap preview of the same mechanism.
- Aratako's facts, for scale: ~170k h (EN ~100k Libriheavy), ~143k steps × ~240k tokens on 8 × MI300X for 2 weeks; no
  scaling curves and no smaller backbone tried. (The training script in the repo uses ScaledAdam lr 0.035; the report
  says AdamW 1e-4. Not resolved; doesn't matter for us.)

## 3. Speed: about 1.5x is available (nanochat-mlx measurements)

r5 ran 27k padded tok/s ≈ 18 TFLOP/s ≈ 30% of the 60 TFLOP/s bf16 peak; nanochat's d12 (98M, same class) sustains 46%.

| change | nanochat measured | status in 33 |
|---|---|---|
| `mx.compile` the loss+grad and optimizer step | 1.2–1.5x | not used (train.py:170–197) |
| fixed shape buckets (needed for compile) | – | r5 makes 454 distinct (B, Lt, S) shapes per epoch |
| less padding | – | 24% of r5's tokens are padding (fewer shapes means more padding: tune the two together) |
| `mask="causal"` instead of mask arrays | baseline path | uses additive mask arrays; all T ≤ 406 < window 512 |
| fused cross-entropy | +2% at this size | skip |
| bigger batch, accumulation, optimizer tricks | no gain (±2%) | skip |
| thermal throttling | −20% within 4 min | A/B speed changes interleaved, same process |

Also: peak memory 19 GB is over the 16 GB `set_memory_limit`. Check whether that stalls allocations; the user is open to
raising limits when the machine is otherwise idle.

## 4. More data or more epochs? More data.

- r5: 380k clips (~400 h with LJ ×4), one pass ≈ 3,610 steps ≈ 35 min. Whisper val CER by pass: 1.1 → 4.96%,
  2.5 → 2.92%, 2.8 → 2.28%, 3.3 → 2.35%, **4.4 → 2.00% (best)**, 5.0 → 2.15%, 5.2 → 2.31%.
- Whisper CER flattened after ~3 passes (100-clip scores are noisy by a few tenths, and the cosine schedule was still
  decaying, so this is direction, not a precise turning point). LJ is repeated ×4, so by step 16k each LJ clip had been
  seen ~18 times; LJ val loss rising after ~2.6 passes is a secondary hint of memorizing (val loss is a poor proxy here).
  The LJ-only fine-tune (r6) made things worse, consistent with that.
- That matches the data-constrained LM scaling result (Muennighoff et al. 2023): up to ~4 epochs is about as good as
  fresh data; beyond that returns fall off fast. We're at that limit, so extra epochs on the same 400 h buy little.
- **So: grow the data so a long run is 2–3 passes over much more audio**, and cut LJ to ×1–2.

| tier | add | hours | download | notes |
|---|---|---|---|---|
| have | LJ 24 h, LibriTTS-R clean-100 38 h, Hi-Fi TTS 283 h | 345 | – | re-encode with Qwen (~2 h) |
| A | LibriTTS-R clean-360 + other-500 (audio) | +501 | 74 GB | cleanest multi-speaker read speech; cased text |
| A | Jenny | +30 | 3.7 GB | varied text (news, recipes, chat) |
| B | HiFiTTS-2, filtered slice (book text, CER < 2%, bandwidth ≥ 11 kHz, ≤ 1 h/speaker) | +600–1,000 | 17.8 GB metadata + ~45–75 GB from LibriVox | biggest clean read source |
| C | Expresso (read + improvised) | +46 | 13.8 GB | expressive styles, studio |
| C | Emilia-YODAS EN, DNSMOS ≥ 3.3 | +500–1,000 | ~10–20 GB | conversational prosody; CC BY; gated |

Tier A gives ~900 h (2.2x); A+B ~1.8k h; all tiers ~2.5–3k h. At today's speed one pass over 3k h ≈ 4.4 h; with the
speed work ≈ 3 h, so 2–3 passes fit an overnight run. Disk: 416 GB free.

## Proposed order
Every probe: 45 min from r5 snap 16000 (or fresh where the model changes), same data, judged on val-100 by Whisper
CER/WER + UTMOS.
1. **T5Gemma mechanism A/B** (built and verified, no downloads): control vs `--progress 2000 --prefix_lm 1`.
   Gates step 5: if bidirectional text + progress positions don't help here, the full port is a weaker bet.
2. **Codebook probe**: `--nq 8` on the existing Mimi codes (delay pattern), vs nq 4.
   Starts while downloads run.
3. **Speed pass**: compile + fixed buckets + causal mask; loss unchanged, tok/s measured interleaved. (GPU ~1 h.)
4. **Downloads** (confirm each): T5Gemma 2 270M-270M 1.57 GB, LibriTTS-R 360/500 audio 74 GB, Jenny 3.7 GB.
   HiFiTTS-2 and Emilia later.
5. **Codec + emission**: Qwen-16 vs Mimi-16 on the same data, with a depth head; T5Gemma 2 port (+ PM-RoPE on encoder
   keys, parity vs transformers) if step 1 is positive.
6. **Long run** (overnight) on the winning stack with tier-A (+B) data, LJ ×1–2, snapshots judged by Whisper + UTMOS.

Supersedes the earlier "download shangeth Mimi codes, run the A/B alongside 27 v2" question: 27 v2 is done (GPU free),
and those codes only matter if we stay on Mimi.
