# 04 · Alien Radio: two agents invent a language from nothing

A **sender** sees one object and can transmit only a few discrete symbols. A **receiver** sees K = 5
candidate objects in shuffled order and must point at the one the sender saw. Neither agent is told
what any symbol means, so whatever "language" appears is invented during training.

* **Stage 1 (first satisfying version):** 8 colours, 1 symbol from a vocabulary of 8.
* **Stage 2 (go deeper):** colour (8) × shape (4) × location (4) = 128 objects, a message of
  **3 symbols from a vocabulary of 8** (9 bits of capacity against the 7 bits needed). **26 of the 128
  attribute combinations (20%) are held out** and never appear in training, not even as distractors.
  Every individual colour, shape and location value still appears in training.
* Two learning rules: **straight-through Gumbel-softmax** (end-to-end differentiable) and
  **REINFORCE** (sender gets a scalar reward with a baseline and an entropy bonus; the receiver is
  trained with cross-entropy).
* Baselines with the same receiver network, steps and data: **no communication** (chance level), a
  **hand-designed compositional code** (symbol j = attribute j, the upper reference), and a **random
  holistic code** (every object gets an arbitrary unique message: perfect on seen objects, useless on
  new ones).
* Analysis: accuracy on seen and held-out combinations, side-channel controls, message efficiency,
  topographic similarity, positional disentanglement, **partner swapping**, and **symbol interventions**
  that build the receiver's dictionary.

Results, numbers and caveats are in [RESULTS.md](RESULTS.md).

## Run it

All commands are run from this folder and use the shared venv.

```bash
# smoke check: unit tests + a tiny train -> analyze -> demo pipeline in a temp dir (~1 min)
./smoke.sh
# or only the unit tests (splits, side-channel controls, metrics, exact checkpoint resume)
../.venv/bin/python -m pytest -q

# stage 1 (colour only): ~15 s per method, run directly
../.venv/bin/python train.py --stage color --method gumbel reinforce nocomm --P 8 --fresh
bash sweep_color.sh > runs/sweep_color.log          # stage-1 sensitivity sweep (14 x ~15 s)

# stage 2 + variants + analysis + demo (one GPU-lock job, ~15 min of compute)
../bin/gpu_run bash run_all.sh > runs/run_all.log 2>&1
# follow-up: harder game, K=20 candidates (then re-runs analysis + demo)
../bin/gpu_run bash run_deeper.sh > runs/run_deeper.log 2>&1

# single pieces
../bin/gpu_run ../.venv/bin/python train.py --stage full --method gumbel reinforce nocomm compositional holistic --P 5
../.venv/bin/python analyze.py      # -> results/metrics.json, media/*.png
../.venv/bin/python make_demo.py    # -> media/demo.html (reads checkpoints/ + results/)
node tests/check_demo.js media/demo.html   # optional: headless check of the demo's JS forward pass
```

`train.py` resumes from `checkpoints/<run>.json` automatically (pass `--fresh` to restart). Batches and
noise are a pure function of (seed, step), so a resumed run reproduces the uninterrupted one; the
test suite checks this.

## Demo

Open **`media/demo.html`** in any browser. It is fully offline and about 1.9 MB. It opens on Gumbel seed 0,
the seed with the best held-out accuracy (about 94% never-seen vs a 5-seed mean of 86%); use the partner
selectors to look at the others. You can:
* deal a round of seen objects or **never-seen combinations**, or build a target from chips;
* read the sender's 3-glyph message and watch the receiver's probabilities over the 5 candidates;
* **click a glyph to replace it** (an intervention) and see how the receiver's reading changes;
* **swap partners**: pair any sender with any receiver (other seeds, the other learning rule, or a
  receiver trained on the hand-designed code). Accuracy for the pair is measured live over 400 rounds;
* see the receiver's **dictionary**, recomputed live by intervention for the chosen pair.

Senders are exported as their deterministic codebooks (the argmax symbol per position for each of the
128 objects). Receivers are exported as weights and run in a small JS forward pass, which the page
checks against choices computed in Python (footer line).

Static figures: `media/example_rounds.png` (a strip of rounds: target → message → receiver's probabilities),
`media/training_curves.png`, `media/summary_bars.png`, `media/partner_swap.png`,
`media/dictionary_gumbel.png`, `media/dictionary_reinforce.png`, `media/dictionary_compositional.png`.

## Files

| path | what |
|---|---|
| `alien.py` | world (objects, held-out split, episode sampler), population-batched sender/receiver nets, fixed codes, metrics |
| `train.py` | training (Gumbel / REINFORCE / receiver-only baselines), evaluation helpers, checkpointing |
| `analyze.py` | full evaluation, partner swaps, interventions, figures, `results/metrics.json` |
| `make_demo.py` | builds the self-contained `media/demo.html` |
| `run_all.sh`, `run_deeper.sh`, `sweep_color.sh` | the exact pipelines behind the reported numbers (stage 2 + variants; K=20 follow-up; stage-1 sweep) |
| `smoke.sh`, `tests/` | pytest suite + end-to-end smoke; `tests/check_demo.js` runs the demo's script under node |
| `checkpoints/` | `full_<method>_s0.*`, `color_<method>_s0.*` (weights, Adam state, config, step, log); tagged variants `full_<method>_<tag>_s0.*`; `sweep/` = stage-1 sweep logs (JSON only; weights deleted to save space) |
| `results/metrics.json` | every number in RESULTS.md (plus the codebooks of all agents) |
| `runs/` | training logs |

**Population training.** Each run trains P independent agent pairs at once via a leading "population"
axis on every weight matrix. Member p has its own initialisation, its own data batches and its own
noise, and the members' losses are simply summed, so this is exactly P separate runs. It is just much
cheaper on a GPU shared with 12 other projects. "Seed p" in the write-up means population member p.
