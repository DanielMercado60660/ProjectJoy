# 37. The Optimizer Is a Chatbot — OPRO on the Joy Lab shelf
Status: idea, not built (written 2026-10-30)
Scope: Expandable

**Build:** "Optimization by PROmpting" (Yang et al., *Large Language Models as Optimizers*, ICLR 2024,
arXiv 2309.03409): the optimizer is an LLM. Each step it sees a meta-prompt with a task description and
the best earlier solutions with their scores. It proposes new solutions, a harness scores them, and they
go back into the prompt. Here the proposer is Astra (Codex, `gpt-6-astra`) and the scorer is our own
harness on the shelf's projects.

**Why it fits here:**
- The proposer only writes text (parameters, rules, prompts or code edits). It never runs MLX, so the
  Codex sandbox's missing Metal device doesn't matter: Claude's harness runs the scoring (`bin/gpu_run`
  for anything on the GPU). Use `codex exec -s read-only`.
- The lab's most repeated review finding was "the hand-written baseline was under-tuned" (8 of 10 in
  wave 2). An LLM baseline-tuner tests that directly.
- The GPU and the scorer's run time set the pace, not Codex tokens.

**Tinker with:** solutions per step (OPRO used 8), how many past solutions sit in the prompt and in what
order (best last), showing scores vs ranks, adding the failure clips/logs to the prompt, temperature,
plain parameter vectors vs editing a rule function's code (FunSearch / AlphaEvolve-style).

**First satisfying version:** tune one existing CPU-only scripted baseline (e.g. 06's longest-queue +
min-green traffic rule, or 08's ray-following driver) with OPRO. Compare it, at the **same number of
evaluations**, against random search and CMA-ES (or a simple evolution strategy) on the same parameter
space, scored on **held-out seeds/tracks** the optimizer never saw. Report best score vs evaluations,
wall clock, and Codex tokens used.

**Go deeper:**
- Code mode: the LLM edits the heuristic's source, not just numbers. Does it beat the shelf's learned
  policy at matched compute?
- Nanochat recipe search: candidates are hyperparameters or small `train.py` edits, scored by d6 val bpb
  after a fixed ~5-minute run (~60M tokens at ~200k tok/s, so ~10 trials/hour, ~100 overnight).
- Classic OPRO: optimize the instruction prompt for a small local model (Gemma 3 270M from 33) on a
  task with a train/test split.
- Reproduce the paper's toy cases (linear regression, small TSP) against a real solver, to see where an
  LLM optimizer is plainly worse.

**Measure:** held-out score vs number of evaluations (against random search / CMA-ES), the gap between
optimization and held-out scores (overfitting), Codex tokens and wall clock per improvement, and how
diverse the proposals are (does it get stuck re-proposing near-duplicates?).

**Honesty traps:** the score the LLM sees must come from a training split, with the final result on
held-out seeds; random search with the same budget is the baseline to beat; and LLM priors can leak
known-good settings (say so; that's part of what the LLM brings).

**Shelf notes:** `bin/astra_turn` already drives Codex turns. Never let it pip install, download, or run
git. Keep one scorer job on the GPU at a time.

**Links:** https://arxiv.org/abs/2309.03409
