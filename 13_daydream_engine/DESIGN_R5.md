# Round 5 — Turn 2: trustworthy dreams against misspecified physics

**Evidence and hypotheses.** Turn 1 is preliminary because its test seeds informed this redesign.
Risk planning helped at 256–4096 shots, but not 64. Its “fitted” law exactly reproduced the
simulator. H1: a genuinely misspecified classical fit remains competitive. H2: more training,
contact-focused loss and explicit geometry reduce ball-contact errors. H3: disagreement remains
useful on untouched tasks. Neither neural improvement nor an advantage over classical fitting
is assumed. Verified numbers are in RESULTS.md.

**Classical methods.** Retain known walls and the recovered exact-law fit, now named
`oracle_exact`: a ceiling over a fixed random candidate set (CEM is only a reference search).
Add `soft_contact`: simultaneous finite spring/damper forces, with no instantaneous impulse or
position projection. Fit stiffness k∈[80,32000], damping ratio ζ∈[0,1.2], effective contact radius
r∈[.04,.09]; walls retain the known radius. Damping is c=ζ√(2k). Bounded finite stiffness and
finite contact duration exclude the true hard-contact law from the family. Use a 48-point grid
plus 144 random proposals, then 64 local proposals around the eight best training fits. Rank
256 trials on 2,048 collision-balanced five-frame training windows; validate the top 32 on
128 natural 40-frame validation trajectories. Select mean position error across all horizons, also used for neural selection. Retain every trial, bounds and hashes.

**Learned trials.** Two five-member ensembles, each trained from scratch for the same wall-clock
budget: width 64, three hidden layers, batch 256/member, unroll five. Control uses the original
12 edge features; treatment adds 12 geometry features (safe distances, normals, approach,
tangential velocity and closest approach). Corrections remain shared and antisymmetric. The original stop-gradient gate fix stays; differentiated norms add epsilon
inside sqrt, including self edges. Sample 75% contact windows and weight ball-contact frames
3×. LR starts .0008, follows a step-indexed 100,000-step cosine to .00008, then stays there.
A 2,000,000-step ceiling leaves the time guard in control. Save every 5,000 updates, sample loss
history every 100, and retain complete optimizer/RNG/config state. Resume is deterministic.
Trials match time, not updates or parameter count.

**Validation only.** Reuse exactly the turn-1 train/validation data (seeds 51001/51002). Select
between the two final new checkpoints and the frozen turn-1 h32 fallback by 128-scene forecast
error. Tune beta∈{0,.25,.5,1,2} on 64 new validation planning tasks (seed 53003), 256 shots,
horizon 40; maximize real success, then minimize distance, then beta. Freeze the selection before
testing. If the old checkpoint wins, report that the new training did not improve validation.
The frozen turn-1 model/beta is a paired reference on new tasks.

**Fresh evaluation and metrics.** Reserve scene/action/target/forecast seeds
62001/62002/62101/62201; the turn-2 guard rejects old streams and legacy data. AC: 200 tasks and
256 independent forecast scenes; battery minimum: 64 tasks. Preserve random shots 64→4096 and
CEM population 64, iterations 1/2/4; share tasks and random numbers. Classical methods also get
5× random shots for a member-frame-count control, not equal FLOPs. Report real/dream success,
fooled and conditional fooled rates, coverage (always 100%), error means/medians by collision
group, Spearman/AUROC/bins, costs and invalid counts. Primary contrast: selected risk vs mean at
256 shots; frozen-reference comparisons are secondary. Use task-paired McNemar, bootstrap
intervals and Holm adjustment. Retain complete task rows and a checkpoint-reloaded failure clip.

**Budget / minimum.** Battery guards: smoke 1.9 min, fit 3, train 2×5, tune 2, eval 14, render .8:
31.7 min plus startup margin, total ≤35, no new job below 50%. AC: stage 1 smoke+fit+2×14-minute
training ≤35; stage 2 tune ≤3 and eval ≤35; stage 3 optional eval continuation ≤35 and render
≤.8. Total ≤109 minutes, leaving room within 135 including the preceding turn's ~15 minutes.
Use gpu_run and ≤4 CPU threads; arrays remain small. Minimum complete: both trials, 256 classical
proposals, frozen selection, 64 paired tasks, full pressure/cost sweeps, forecasts and demo.
Timeouts stay partial. New MLX/VJP/resume smoke must pass first. **Turn-2
training, classical fit quality and headline outcomes remain pending runner.**
