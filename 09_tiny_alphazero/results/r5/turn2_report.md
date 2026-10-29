## Round 5: Learn when to think longer

**Turn 2, measured on 2026-10-24. TL;DR: the learned mistake predictor did not beat fixed-budget MCTS.** At equal simulations it tied fixed search on the small test (11 errors each) and made one extra error on the larger test (61 vs 60), while taking 3.55× / 3.69× as long. At matched time it made 14 vs 11 errors and 73 vs 60. The larger rule was the best adaptive time allocator (65 errors), still behind fixed. Games scored slightly below 0.5. Better calibration did not establish better allocation.

Sources: `results/r5/{main,ac_large}/evaluation.json` and `games.json`; independent recomputation/provenance checks in `results/r5/turn2_analysis.json` from `r5_analyze.py`. `media/r5_review.png` summarizes the measured results. AC/load-average 3–7 and concurrent projects are Claude's reported run conditions; the timings below are recorded wall times.

### Setup, coverage and selection

Both runs freeze the same gen-191 backbone and train six small mistake heads. The target is whether the **8-simulation probe's move loses game-theoretic value**, not whether extra search will repair it. The allocators rank unlabelled blocks of 64 positions. Fixed search gets 32 simulations per board; adaptive search distributes exactly the same total between a floor of 8 and a cap of 128. It is batch scheduling, not an online one-board latency test. Every board receives a legal move; no abstention.

| Run | Train / validation / test | Recorded train / eval seconds | Final steps per candidate | Selected head / step | Validation drawn positions |
|---|---|---|---|---|---|
| main | 29,491 / 384 / 384 | 1438.01 / 479.12 | 86,528–86,656 | width 64, seed 73 / 512 | 22 |
| ac_large | 60,355 / 1,536 / 1,536 | 2638.01 / 718.30 | 337,664–337,792 | width 128, seed 73 / 1,664 | 90 |

Data counts/coverage: each run's `frozen.json`; steps/times: `logs/r5/{main,ac_large}_{train,eval}.log`; selected weights/config: `checkpoints/r5_{main,ac_large}/selected/head.json`. Selection used only equal-update rounds; the wall guard interrupted the final unselected round. Smoke passed in **11.736 s**, including exact RNG/optimizer resume and SIGKILL/restart (`logs/r5/claude_smoke.log`; the reported `smoke_driver.log` filename does not exist).

**These are nested-data runs, not independent seeds.** All 384 main validation boards and all 384 main test boards occur in their respective larger splits. Training overlaps on 29,225 boards, so the train sets are largely shared but not strict prefixes. Five larger-test boards and one larger-validation board occur in the main head's training set; they do **not** occur in the larger head's training set. Do not evaluate the main head on the whole larger test and call it held out. The selected heads also differ in width and learning rate; the larger test changes risk prevalence and block composition, so this does not isolate a data-size effect.

The audit verified dataset/checkpoint hashes, zero within-run split overlap, and zero overlap with all **six retained replay buffers** and the old solver split, including mirrors. Overwritten historical replay remains unverifiable. All accepted labels are exact; 9/393 main test candidates and 37/1,573 larger test candidates exceeded the 3M-node cap. Every timeout was in the early group: 9/137 and 37/549 early candidates. Thus the reported population excludes some of the hardest early boards. Accepted validation/test sets each contain all three ply groups; none was trimmed during selection or evaluation (`frozen.json`, `data/unsolved.jsonl`).

On the **shared 384 test boards**, both runs have 11 learned and 11 fixed equal-simulation errors. Main/larger learned matched-time errors on those boards are 14/16. The rule changes from 10 to 12 equal-simulation errors on the shared subset. These cached comparisons (`turn2_analysis.json.shared_test_errors`) still mix different adaptive block compositions and measured time allowances; they do not isolate training-data size.

### Allocation accuracy and efficiency

`uncertainty` is 1−|value|; `disagreement` measures policy/visit disagreement; `rule` is the validation-tuned alpha-beta/uncertainty/threat formula. `logistic` is an additional trained classical predictor. Error rates include 130 / 491 already-lost boards, where no legal move can further lower the win/draw/loss value; the group results expose this class imbalance.

**Net avoided = rescued probe errors minus newly introduced errors**, relative to each board's 8-simulation probe, not relative to fixed-32. The probe made 18 / 89 errors (raw policy: 36 / 146). Milliseconds are total batched decision time, including features, predictor, tree allocation/sorting and final move extraction. `Avoided/1k sim` is the requested per-simulation rate multiplied by 1,000. Negative comparisons with fixed are visible in the error column; higher per-simulation efficiency at a smaller total simulation count is not an equal-simulation win.

**main:** source `results/r5/main/evaluation.json`, fields `summary` and `records`.

| Budget / signal | Errors | Net avoided | Simulations | Total ms | Avoided/ms | Avoided/1k sim |
|---|---:|---:|---:|---:|---:|---:|
| equal sims / fixed | 11 | 7 | 12,288 | 126.93 | 0.05515 | 0.5697 |
| equal sims / learned | 11 | 7 | 12,288 | 450.68 | 0.01553 | 0.5697 |
| equal sims / entropy | 16 | 2 | 12,288 | 442.26 | 0.00452 | 0.1628 |
| equal sims / uncertainty | 15 | 3 | 12,288 | 448.04 | 0.00670 | 0.2441 |
| equal sims / disagreement | 16 | 2 | 12,288 | 437.21 | 0.00457 | 0.1628 |
| equal sims / rule | 10 | 8 | 12,288 | 651.88 | 0.01227 | 0.6510 |
| equal sims / logistic | 11 | 7 | 12,288 | 452.58 | 0.01547 | 0.5697 |
| time / fixed | 11 | 7 | 12,288 | 126.93 | 0.05515 | 0.5697 |
| time / learned | 14 | 4 | 4,931 | 128.96 | 0.03102 | 0.8112 |
| time / entropy | 17 | 1 | 5,035 | 128.56 | 0.00778 | 0.1986 |
| time / uncertainty | 17 | 1 | 5,035 | 129.19 | 0.00774 | 0.1986 |
| time / disagreement | 16 | 2 | 5,048 | 128.56 | 0.01556 | 0.3962 |
| time / rule **UNMATCHED** | 18 | 0 | 3,072 | 240.44 | 0.00000 | 0.0000 |
| time / logistic | 14 | 4 | 5,009 | 129.02 | 0.03100 | 0.7986 |

**ac_large:** source `results/r5/ac_large/evaluation.json`, fields `summary` and `records`.

| Budget / signal | Errors | Net avoided | Simulations | Total ms | Avoided/ms | Avoided/1k sim |
|---|---:|---:|---:|---:|---:|---:|
| equal sims / fixed | 60 | 29 | 49,152 | 514.90 | 0.05632 | 0.5900 |
| equal sims / learned | 61 | 28 | 49,152 | 1899.36 | 0.01474 | 0.5697 |
| equal sims / entropy | 85 | 4 | 49,152 | 1850.51 | 0.00216 | 0.0814 |
| equal sims / uncertainty | 74 | 15 | 49,152 | 1834.77 | 0.00818 | 0.3052 |
| equal sims / disagreement | 82 | 7 | 49,152 | 1866.98 | 0.00375 | 0.1424 |
| equal sims / rule | 61 | 28 | 49,152 | 1872.63 | 0.01495 | 0.5697 |
| equal sims / logistic | 65 | 24 | 49,152 | 1863.72 | 0.01288 | 0.4883 |
| time / fixed | 60 | 29 | 49,152 | 514.90 | 0.05632 | 0.5900 |
| time / learned | 73 | 16 | 19,932 | 524.50 | 0.03051 | 0.8027 |
| time / entropy | 87 | 2 | 20,075 | 522.93 | 0.00382 | 0.0996 |
| time / uncertainty | 78 | 11 | 20,036 | 521.59 | 0.02109 | 0.5490 |
| time / disagreement | 86 | 3 | 20,049 | 521.28 | 0.00576 | 0.1496 |
| time / rule | 65 | 24 | 19,321 | 520.83 | 0.04608 | 1.2422 |
| time / logistic | 75 | 14 | 20,023 | 521.70 | 0.02684 | 0.6992 |

Main's AB8 rule exceeded the time allowance by **89.4%**, exceeded it in all six blocks, and performed no simulations beyond the probe. That row is an observed failure, not an equal-time comparison. Every other main time arm and every larger time arm was within 3.8% on every block; aggregate overruns were ≤1.9%. Timings are brief measurements during a shared-machine run, not isolated latency distributions.

In the larger simulation comparison, fixed rescues 36 errors and introduces 7; learned rescues 30 and introduces 2. The learned gate makes fewer new errors but fails to repair enough existing ones. It improves the drawn subgroup (11/89 errors vs fixed's 15/89) while worsening won positions (50/956 vs 45/956). At matched time learned loses that draw benefit (15/89) and makes 58 won-position errors. On the small test, learned/fixed simulation errors by early/middle/late group are 5/2/4 versus 7/1/3. All group counts are retained in the evaluation files.

Paired exploratory checks: learned vs fixed has 3 improvements / 3 regressions on main and 17 / 18 on the larger simulation test (exact McNemar p=1 in both). At larger matched time the counts are 2 / 15 (p=0.00235, unadjusted across comparisons). The raw error counts support no learned advantage. The one-error rule advantage on main is likewise not persuasive.

The cost gap is not mainly the tiny head: entropy/disagreement also take roughly 3.5–3.6× fixed's time at equal simulations. The implementation runs up to 128 serial tree steps on a shrinking active batch instead of 32 steps on the full batch. This explains a throughput penalty structurally; timings do not separately identify dispatch, tree-copy and predictor costs. The time arm also preallocates 1,024-simulation trees. These results qualify this implementation, not every possible adaptive-search system.

### Game results

Sources: each `games.json`, independently recomputed in `turn2_analysis.json`. Both colours per start; learned and logistic share starts within paired batches. Costs are **32 simulations per decision on average across the active batch**, not equal total game time or necessarily 32 for each move. All pairs finished; no censoring. Confidence intervals bootstrap paired-start batches (16 starts / 32 games), preserving within-batch dependence.

| Run / allocator vs fixed | W / D / L | Games | Score | Batch-bootstrap 95% CI | Total decision-time ratio |
|---|---:|---:|---:|---|---:|
| main / learned | 1567 / 192 / 1601 | 3,360 | 0.49494 | 0.48631–0.50327 | 3.17× |
| main / logistic | 1549 / 192 / 1619 | 3,360 | 0.48958 | 0.48021–0.49836 | 3.15× |
| ac_large / learned | 2486 / 291 / 2567 | 5,344 | 0.49242 | 0.48503–0.49963 | 3.14× |
| ac_large / logistic | 2472 / 311 / 2561 | 5,344 | 0.49167 | 0.48456–0.49860 | 3.17× |

Logistic, not the rule, was the validation-selected classical game opponent in both runs. Rule/entropy/value/disagreement game scores and time-matched game scores are **not measured**. The first 105 paired-start batches of the larger run reuse main's start seeds; these game results are also not independent replication. Neither learned game score exceeds 0.5; the larger estimate is slightly below it.

### Calibration and overtraining

Sources: `evaluation.json.calibration`; the constant predictor uses the **training** error prevalence, computed in `turn2_analysis.json`. Lower NLL/Brier/ECE is better; ECE uses ten bins and can look small when positives are rare.

| Run / predictor | NLL | Brier | ECE | Draw-group NLL / ECE |
|---|---:|---:|---:|---|
| main / learned | 0.18271 | 0.04791 | 0.03130 | 0.52434 / 0.16674 |
| main / rule | 0.18278 | 0.04358 | 0.03693 | 0.48840 / 0.12030 |
| main / logistic | 0.18789 | 0.04722 | 0.02151 | 0.53273 / 0.11249 |
| main / constant train prevalence | 0.19170 | 0.04495 | 0.01643 | not measured |
| ac_large / learned | 0.18148 | 0.04915 | 0.01697 | 0.43817 / 0.07488 |
| ac_large / rule | 0.21216 | 0.05330 | 0.04188 | 0.48461 / 0.07722 |
| ac_large / logistic | 0.19559 | 0.05149 | 0.00777 | 0.45330 / 0.11120 |
| ac_large / constant train prevalence | 0.22148 | 0.05461 | 0.00492 | not measured |

Main's learned Brier score is worse than the constant and rule baselines, despite better NLL than the constant. The larger head improves NLL/Brier over both trained classical predictors, while logistic has the lowest ECE among the three nonconstant predictors. Draw calibration remains worse than overall calibration (20 / 89 test draws). A low global calibration error is not evidence that the ranking identifies *repairable* mistakes.

The clock budgets were filled, but most optimization was unnecessary: the selected checkpoints appeared at **round 4 / 13**, followed by **673 / 2,626 rounds** without a new selection. Final training losses were near zero. Observed median end-to-end data rates were **38.94 / 40.01 boards/s**; aggregate head-update rates were **874 / 2,124 steps/s** (`*_train.log`, `turn2_analysis.json`). Filling a budget is not the same as improving generalization; further training is not the next useful experiment.

### Baseline and test-selection audit

**No test labels selected the models or classical settings.** Selection functions receive train/validation arrays only; full validation includes early, middle, late and drawn boards. Test boards are featurized before training and later used for unlabelled score ranking, warm-up and measured time allowances; no test mistake count enters selection. The best head was chosen over many repeated validation checks, so validation overfitting is possible even though the test remains held out within each run.

Classical tuning was broad but not perfectly aligned with the question: 576 calibration settings span AB depths 2/4/6/8, three signals, four weights, four biases and three scales. Bias/scale largely preserve ordering, so these are at most 48 underlying ranking formulas, not 576 different allocators. Main chose AB8 + 0.5×value uncertainty; the larger run chose AB4 + 1×value uncertainty. Both include the fixed 0.25×threat term. Logistic tested six ridge values (.0001–10), all converged; selected ridge .0001 took 41 / 32 L-BFGS iterations. Both classical selectors used the same complete-validation macro-NLL as the head, not downstream mistakes or time cost (`selected/baselines.json`, `r5_baselines.py`).

**The fixed baseline was not tuned as hard.** Its 32-simulation allowance, PUCT=1.5, and the adaptive 8/128 floor/cap were fixed in advance. Only one fixed operating point was evaluated. Thus the stronger requirement to tune fixed search and allocation settings on the downstream objective was not fully met in turn 1. This omission cannot establish an adaptive win—fixed already wins—but limits claims about an optimal operating point.

Secondary old-split results (1,000 boards, known historical replay overlap): fixed makes 27 errors; main learned/rule/logistic make 24/22/25, and larger learned/rule/logistic make 26/22/26. Entropy/value/disagreement make 41/35/42 in both runs. These are secondary comparisons, not new held-out evidence (`secondary.json`).

### Demo, failures and next step

The original demos were written successfully: `media/r5_main.html` (26,969 bytes) and `media/r5_ac_large.html` (29,222 bytes). Silence came from the renderer having no print statement. It now emits `demo_written` with path, byte count and failure-board count, and supports `--demo-out` to preserve earlier renders. New self-contained copies `media/r5_main_review.html` and `media/r5_ac_large_review.html` include calibration, game costs and 11 / 12 failure boards. The original media, checkpoints, logs and measured results were preserved.

What worked: exact simulation accounting, full-validation selection, resumable training, a useful risk predictor on the larger test, and preserved failures. What did not: the aggressively concentrated allocation, throughput efficiency, and prolonged training after early validation selection. The AB8 time failure and good calibration without better play are the useful surprises.

**Turn-2 follow-up is prepared, not measured:** `r5_followup.py` freezes the larger head and logistic weights; excludes both earlier runs' six splits plus retained replay/old solver positions; uses fresh validation/test seeds; tunes PUCT for fixed and every gate, and tunes floors 8/24 and caps 64/128 against complete-validation move errors. The rule gets 48 formulas over the same search grid. Only after selection are fresh test blocks scored, filling a six-minute measured-rate allowance. No retraining or reuse of the old tests for selection. See RUNME's Turn 2 commands and caveats. MLX execution of this new path remains for Claude; NumPy mechanics/tests and static checks were run by Astra.

CPU verification: **32 tests passed** (`logs/r5/astra_turn2_tests.log`), including cached-versus-direct search continuation, the complete 639-trial tuning grid, fresh-seed/exclusion resume, demo acknowledgement and smoke-sized game accounting. HTML structure and original/new demo hashes are recorded in `results/r5/turn2_demo_check.json`; the summary PNG was visually inspected. No MLX execution occurred in Astra's turn.
