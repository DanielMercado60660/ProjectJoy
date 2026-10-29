# 30 · Nanochat on MLX

A clean, pure-MLX port of [karpathy/nanochat](https://github.com/karpathy/nanochat)
(tokenizer → GPT → Muon pretraining → SFT → eval → KV-cache inference), a measured
throughput benchmark, a ≤2 h pilot pretraining run on the shared M5 Max, and a tiny job for
the model: **Grid Courier**, a 5×5 text world where the model must deliver items into
containers by emitting actions such as `move north`, `pick gem`, `put gem in basket`.

Results, numbers and the hand-off plan for the M4 Pro mini are in [RESULTS.md](RESULTS.md).
Demo: open `media/demo.html` (self-contained, no network). It has the loss curves,
generations at 4 checkpoints, step-by-step replays of the model playing held-out episodes,
and the result tables. Also see `media/episode_transcripts.md`, `media/loss_curve.png` and
`media/sft_curve.png`.

**Headline:** a 16.9M-parameter d6 pilot reached 1.167 val bits/byte after 157M ClimbMix
tokens in 52 min on a shared M5 Max (~51k tok/s). After SFT on sandbox traces it solves **100%**
of tasks built from held-out item/container combinations. The non-language MLP baseline,
trained on the same traces, solves **0%** of them.
**Read with these caveats from the cross-review:**
- The held-out action is a verbatim copy of words in the task line, and a from-scratch LM also
  gets 100%. That makes this a copying result, not compositional reasoning.
- The MLP's flat action head never saw the held-out `put` classes as targets, so its 0% is by
  construction.
- All throughput and latency numbers were measured on a heavily shared GPU. An idle M5 Max is
  ~4× faster at training and ~6× faster at decoding, and the "30–39% MFU" used a contended
  peak. bpb with corrected byte counts is 1.1685.

## How to run (from this folder)

All commands use the shared venv. Anything longer than a minute goes through the GPU slot lock.

```bash
PY=../.venv/bin/python

# 0. data + tokenizer (ClimbMix shards from HF, last shard = validation)
$PY data.py --download 12                      # 12 train shards + val shard (~1.2 GB) -> data/climbmix
RAYON_NUM_THREADS=4 $PY tokenizer.py --train --vocab-size 8192 --max-chars 100000000
RAYON_NUM_THREADS=4 $PY data.py --tokenize     # -> data/tokens/*.npy (uint16, ~63M tokens per shard)

# 1. benchmark + pilot pretraining (+ the post stage below), all in one GPU slot
../bin/gpu_run bash run_pilot.sh > logs/pilot.log 2>&1
#    or by hand:
../bin/gpu_run $PY bench.py --out media/bench.json
../bin/gpu_run $PY train.py --depth 6 --total-batch 131072 --device-batch 32 \
    --num-iterations 1195 --eval-every 100 --sample-every 298 --out checkpoints/pilot --resume   # the pilot as run

# 2. post stage: inference bench, SFT (pretrained and from-scratch), sandbox evals, plots, demo
../bin/gpu_run bash run_post.sh > logs/post.log 2>&1
#    pieces:
$PY engine.py --ckpt checkpoints/pilot/last                       # first-token latency, decode tok/s
$PY sft.py --base checkpoints/pilot/last --out checkpoints/sft --episodes 8000 --epochs 3   # SFT on sandbox traces (as run; default --epochs is 2)
$PY baseline.py                                                   # non-language MLP baseline
$PY eval_sandbox.py --ckpt checkpoints/sft/last --policies expert,random,mlp,lm,lm_constrained
$PY plot.py && $PY make_demo.py --ckpt checkpoints/sft/last

# 3. extra experiments: data efficiency (300/1000 episodes, pretrained vs scratch), 3-delivery tasks
../bin/gpu_run bash run_extra.sh > logs/extra.log 2>&1
SEED=1 ../bin/gpu_run bash run_seeds.sh > logs/seeds.log 2>&1     # more SFT seeds; also SEED=2

# 4. smoke test (< 1 min) and unit tests. Needs tokenized data; on a fresh clone the minimal setup is
#    $PY data.py --download 1 && RAYON_NUM_THREADS=4 $PY data.py --tokenize   (~40 s)
./smoke.sh
$PY -m pytest -q tests
```

Resume: rerun the same `train.py` command with `--resume`; it continues from
`checkpoints/<out>/last` (model, optimizer state, step, schedule; the data order is a pure
function of (seed, step), so resume is bit-exact; see RESULTS.md).

## File layout

| file | what |
|---|---|
| `tokenizer.py` | byte-level BPE (HF `tokenizers`) with upstream's split regex + special tokens, chat rendering with loss mask |
| `data.py` | ClimbMix download, tokenization to flat uint16 arrays, random-window train loader, fixed val batches |
| `gpt.py` | the GPT (rotary, QK-norm, ReLU², untied, softcap, resid/x0 scalars), KV cache, FLOP accounting |
| `optim.py` | MuonAdamW: hand-written Muon (Polar Express, NorMuon, cautious WD) + AdamW groups |
| `train.py` | pretraining loop: compiled step, grad accumulation, upstream LR/momentum/WD schedules, bpb eval, checkpoint/resume, time budget |
| `checkpoint.py` | save/load model + optimizer state (safetensors) + JSON meta |
| `bench.py` | tokens/s, MFU vs measured matmul peak, peak memory; sizes × compile × dtype |
| `fused_ce.py` | fused logit-softcap + cross-entropy Metal kernel (default `--ce-mode kernel`; `--ce-mode ref` = original path) |
| `efficiency/` | afternoon efficiency pass: `bench_step.py` (times the real accumulation-4 training step; use this, not `bench.py`, to size runs), `parity.py`, `profile_step.py`, `LOG.tsv`, `REPORT.md` |
| `engine.py` | KV-cache generation, constrained-decoding hook, latency benchmark |
| `sandbox.py` | Grid Courier world, expert policy, held-out pair split, structured features |
| `sft.py` | SFT on expert traces in upstream's chat format, packed rows, assistant-only loss |
| `baseline.py` | non-language baseline: MLP classifier on structured state |
| `eval_sandbox.py` | success / validity / latency for expert, random, MLP, LM, grammar-constrained LM |
| `plot.py`, `make_demo.py`, `summarize.py` | loss PNGs, `media/demo.html`, `media/episode_transcripts.md`, data-efficiency table |
| `run_pilot.sh`, `run_post.sh`, `run_extra.sh`, `run_seeds.sh` | the exact pipeline that produced the results (logs in `logs/`) |
| `smoke.sh`, `tests/` | end-to-end tiny run and unit tests (causality, KV cache, sandbox split, Muon) |
| `checkpoints/` | tokenizer, pilot (`pilot/last` = weights + optimizer, resumable; `pilot/step_*` = samples), SFT models (`sft/last` is the demo model), MLP baseline. Committed weights: `pilot/last/model.safetensors`, `sft/last/model.safetensors` (67 MB each) and the MLP. All other `*.safetensors` are gitignored, including `pilot/last/optim.safetensors`, so resuming pretraining from a fresh clone isn't possible but reloading for eval/demo is |
| `media/` | demo.html, loss curves, bench/eval JSON, transcripts |
| `upstream/`, `data/` | reference clone of nanochat and the downloaded data (gitignored) |

## What was ported, simplified, skipped

| upstream piece | here |
|---|---|
| rustbpe + tiktoken tokenizer, GPT-4-style split, 32768 vocab | **simplified**: HF `tokenizers` byte-level BPE, same split regex and special tokens, **8192** vocab (smaller lm_head for a 30M model) |
| FineWeb-Edu → ClimbMix-400B shards, last shard = val | **ported** (ClimbMix, as upstream uses since Mar 2026; 12 train shards) |
| BOS-aligned best-fit document packing | **simplified**: uniformly random windows of the BOS-delimited token stream |
| GPT: rotary (base 1e5), QK norm ×1.2, ReLU², untied wte/lm_head, norm after embed, no biases, param-free RMSNorm, softcap 15, resid/x0 lambdas, upstream init | **ported** (rope via `mx.fast.rope`, attention via `mx.fast.scaled_dot_product_attention`) |
| GQA | ported (config `n_kv_head`), pilot uses full MHA like upstream |
| value embeddings (ResFormer), smear gate, backout | **skipped** (complexity; mostly speedrun tweaks) |
| sliding-window attention (SSSL) | **skipped**: all layers full context ("L") |
| fp32 master weights, bf16 matmuls | ported (cast weights to the activation dtype in `Linear`) |
| MuonAdamW: Polar Express, NorMuon, MuonEq, Muon+, cautious WD, lr·√max(1,m/n) | **ported** by hand (no param stacking/sharding; single device) |
| LR warmup/warmdown, Muon momentum ramp, cosine WD, batch-size √ LR scaling, WD ∝ D_ref/D | ported |
| bits-per-byte val metric | ported |
| CORE / ARC / MMLU / GSM8K / HumanEval evals | **skipped**: replaced by the sandbox evaluation (the pilot is far too small for these) |
| SFT on SmolTalk etc. with assistant-only loss | **replaced** by SFT on Grid Courier traces (same chat format and masking) |
| RL (GRPO-style on GSM8K) | skipped ("Go deeper" stage) |
| Engine with KV cache, tool use (calculator) | KV cache ported (batch 1); tool use skipped |
| DDP, FP8, Flash Attention 3, wandb | not applicable on one Mac |

## Round 5: revise a plan (results in RESULTS.md)

Round 5 replaces the copying task with a harder courier world:
- parcel locations and destination bindings must be discovered by inspecting rooms
- actions can fail (a locked receptacle, a parcel that moves)
- a request can change mid-episode
- tests include unseen 3–4 delivery sequences

It compares six methods: pilot-SFT, scratch-SFT, success-filtered post-training, a
continued-SFT control, a factored-action MLP and a tuned symbolic planner. Design:
[DESIGN_R5.md](DESIGN_R5.md). Earlier
commands and results are unchanged.

**Headline** (turn 2, fresh test seeds; mean success over three training seeds):

| method | success |
|---|---:|
| planner | 100% |
| factored MLP | 61% |
| scratch-SFT LM | 22.5% |
| pretrained-SFT LM | 10% |
| success-filtered fine-tune | 13% |
| continued-SFT control | 12.5% |

The LMs choose the right command only 77–96% of the time even with teacher forcing, and
get stuck repeating illegal commands. See RESULTS.md, "Round 5".

### Commands actually run (2026-10-24, AC power, from this folder)

```sh
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 TOKENIZERS_PARALLELISM=false
../.venv/bin/python -m pytest -q tests/test_r5_numpy.py              # 20 tests (turn 1), 25 (turn 2)
../bin/gpu_run sh smoke_r5.sh                                         # logs/r5/claude_smoke*.log, claude_t2_smoke.log

# turn 1: test seeds 81,241,000+
../bin/gpu_run ../.venv/bin/python r5_suite.py --tag battery_s51 --seed 51 --max-minutes 32.5
../bin/gpu_run ../.venv/bin/python r5_suite.py --ac --tag ac_s52 --seed 52 --max-minutes 32.5
../bin/gpu_run ../.venv/bin/python r5_suite.py --ac --tag ac_s53 --seed 53 --max-minutes 32.5

# turn 2: full validation monitor, fresh test seeds 91,024,000+, old range as (incomplete) secondary
for s in 51 52 53; do
  ../bin/gpu_run ../.venv/bin/python r5_suite.py --ac --tag t2_ac_s$s --seed $s --max-minutes 32.5 \
    --test-seed-start 91024000 --secondary-seed-start 81241000 --secondary-episodes 4
done

# review diagnostics: NumPy/JSON only, ~50 s, no GPU
OMP_NUM_THREADS=4 ../.venv/bin/python r5_review_analysis.py   # -> results/r5/review_t2_diagnostics.json
```

Notes on the runs:
- `battery_s51` ran without `--ac` but on AC power (its `finished.json` records it).
- Before the turn-1 suites, the first smoke failed on MLX 0.32's RNG restore. The runner
  patched `r5_checkpoint.restore` and the second smoke passed (`logs/r5/claude_smoke2.log`).
- Each suite takes about 28 minutes (1,625–1,692 s).
- The turn-2 secondary evaluation is under-budgeted (0.8 min) and finished only 31–36 of
  80 scenarios. Do not read it as a test.

### Round-5 files

| file / destination | purpose |
|---|---|
| `r5_world.py` | NumPy environment, shared public notebook, planner, recovery metrics |
| `r5_data.py`, `r5_protocol.py` | planner tuning, teacher data, the one shared prompt/command wire format and grammar |
| `r5_train.py`, `r5_checkpoint.py`, `r5_validation.py` | MLX LM/factored-MLP training, time guards, resumable generations, full-set validation |
| `r5_eval.py` | grammar-only policies, success-filtered collection, paired tests, self-contained HTML |
| `r5_suite.py` | one-slot sequential experiment with power checks and a total wall guard |
| `r5_smoke.py`, `smoke_r5.sh`, `r5_decode_check.py`, `tests/test_r5_numpy.py` | kill/resume + decoder-parity smoke; NumPy unit gate |
| `r5_audit.py`, `r5_numpy_reference.py` | Astra's CPU serialization audit and FP32 NumPy forward pass of saved LM weights |
| `r5_review_analysis.py` | reviewer diagnostics: closed-loop teacher agreement, loops, by-verb accuracy |
| `results/r5/<tag>/`, `logs/r5/<tag>/` | manifests, exact commands, paired episode receipts, per-arm logs |
| `checkpoints/r5_<tag>/<arm>/selected` | final per-method export (weights git-ignored; about 194 MB per LM) |
| `media/r5_t2_ac_s5x.html` | demo: first success and first failure per method, with full transcripts |

Only the final paths in each run's `selected.json` need distribution. Generated training
arrays (`results/r5/*/data/*.npz`, 64 MB each) are reproducible from `r5_data.py` and
are git-ignored. Do not commit them, or the smoke/trial/resume weights.
