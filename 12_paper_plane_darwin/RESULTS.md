# Results: Paper Plane Darwin

All numbers come from `results/eval.json` (audit and held-out evaluation, `evaluate.py`) and
`results/search_log.json` / `results/search_stdout.txt` (main run, `search.py`), produced on
2026-10-22. One main run: 771 s of wall time, 4 numba threads, CPU only, under load (load
average 14.4 on 18 cores, with ten builders running). The timing is a busy-machine number.

These results are about this simulator only. They say nothing about real aircraft.

## Headline

**Evolution beats the shipped hand-designed glider, and the win is physics, not integrator error.
But most of the held-out win is the span bound, not evolution** (reviewer finding, see
"Baseline fairness"): the same textbook glider simply stretched to the 50 cm span
bound and trimmed on the still-air throw alone flies **91.8–94.7 m** held-out mean, level with or
above CMA-ES best (89.8 m; 5-seed mean 89.6 m). CMA-ES still clearly wins on the training launches
it optimised (118.1 m against 104–107 m for the stretched baselines).

| design | still-air range (m) | training mean, 6 launches (m) | held-out mean (m) | held-out p10 (m) | held-out worst (m) | wins vs baseline on held-out |
|---|---|---|---|---|---|---|
| hand-designed baseline | 85.7 | 78.6 | 68.1 | 44.8 | 6.1 | – |
| random search best (5 × 4000 evals) | 113.4 | 100.9 | 77.3 | 37.2 | 19.3 | 45/64 (70 %) |
| **CMA-ES best (5 × 4000 evals)** | **122.1** | **118.1** | **89.8** | **49.3** | 13.3 | **58/64 (91 %)** |
| MAP-Elites best (120k evals) | 120.1 | 115.3 | 87.7 | 13.1 | **−145.0** (flew backwards, see below) | 54/64 (84 %) |
| CMA-ES on cheap Euler sim (5 × 4000) | 120.5 | 114.8 (sim said 117.3) | 89.2 | 33.6 | 14.2 | 51/64 (80 %) |

The rows are the single best-on-training seed of each arm (selection used training launches only;
the 64 held-out launches come from a separate generator, `sim.heldout_conditions(seed=12345)`).
**Averaged over all 5 seeds, held-out means are CMA-ES 89.6 m, cheap-sim CMA-ES 88.7 m, random
search 81.2 m** (`eval.json: *_seeds.heldout_mean`). So CMA-ES beats matched-budget random search on
held-out launches by about 8 m, not the 12.5 m the best-seed rows suggest, and the seed ranges
overlap (random seed 0 reaches 90.4 m).

Win rates are paired comparisons on the same 64 held-out launches (best-seed designs). Medians of the per-launch
range ratio against the baseline (best-seed designs): CMA-ES 1.37×, MAP-Elites 1.45×, random 1.16×.

Across seeds (`eval.json: cma_seeds`, `random_seeds`), CMA-ES best training range was
115.3–118.1 m against random search's 94.4–100.9 m. Every CMA seed beat every random seed on
training. Held-out means overlap more: CMA-ES 80.9–95.7 m, random 75.1–90.4 m. Learning
curves are in `media/search_and_archive.png` (left). CMA-ES passes the baseline after about 150
designs; random search never reaches CMA-ES's level at 4000.

## Physical sanity checks (run before any "beats the baseline" claim)

Tolerances were fixed in advance (`evaluate.py`): halving the step from 5 ms to 2.5 ms must
give the same range within **0.5 % (or 5 cm)** on every training **and** held-out launch. Also,
in still air no RK4 step may raise mechanical energy (½mv² + ½Iq² + mgz) by more than
**1e-9 × launch energy**. Gusts and wind can add energy legitimately, so the energy audit
re-flies every launch with wind and gusts off.

| design | max range change from halving dt (held-out) | max per-step energy rise, still air | passes |
|---|---|---|---|
| baseline | 4.4e-6 relative (0.2 mm) | 0 (never rises) | yes |
| CMA-ES best | 7.2e-6 (0.3 mm) | 0 | yes |
| random best | 3.6e-5 (3.9 mm) | 0 | yes |
| MAP-Elites best | 2.4e-4 (1.3 mm) | 0 | yes |
| cheap-sim best | 2.3e-5 (2.1 mm) | 0 | yes |
| MAP-Elites archive, all 399 elites | 396/399 pass dt-halving | 399/399 pass energy | see below |

The three elites that fail are low-fitness looping or tumbling designs (training range 15–37 m,
pitch-rate RMS 4–6 rad/s). Their landing points move by 0.25–2 % between 5 ms and 2.5 ms, and
by at most 0.4 % between 2.5 ms and 1.25 ms (within tolerance). That is sensitive chaotic flight resolved slightly too coarsely, not an
exploit: they are far from the top of the archive. See `media/sanity_checks.png`: energy
falls monotonically, and the 5 ms and 2.5 ms flight paths sit on top of each other.

## Did evolution exploit numerical instability? No, and we tried to make it.

The "cheap" arm ran CMA-ES on explicit forward Euler at 20 ms, 16× cheaper than the RK4 sim.
Euler does gain energy on oscillatory modes: random designs gained up to 0.8 % of launch
energy in a single step. Even so, the cheap-sim winners overstate their true (RK4) training
range by only 0.2–2.4 m (0.2–2 %; `eval.json: cheap_seeds`, search fitness minus RK4 training mean). The same design flies 120.6 m under Euler and 120.5 m under
RK4 in still air (`media/sanity_checks.png`, right). A side probe outside the main run pushed
Euler to 30 ms and 50 ms with 1600-eval CMA-ES. It still found no exploit (Euler 112.1 m against
RK4 110.7 m, and Euler 109.0 m against RK4 107.6 m). *(Reviewer: this side probe's output was not
saved to disk and was not reproduced; treat these four numbers as unverified.)* The physics is too strongly damped:
tail pitch damping and drag swamp Euler's anti-damping on the slow phugoid. Designs stiff
enough to go unstable under Euler simply diverge and crash, so evolution steers away from them.
**Not tested:** an integrator or objective that rewards divergence, such as no crash on blow-up.

## The good failure: a flying wing that glides home backwards

The MAP-Elites champion is nearly a flying wing: 50 cm span, a tail of only 8 % of wing area on
a 5 cm arm, and 18.8 g of ballast sitting on the wing's quarter chord. It is the second-best
design in still air (120.1 m). On held-out launch #10 (8.8 m/s throw, 1.2 m/s headwind,
gusts) a gust pitches it up past vertical and it tumbles. It then settles into a **stable
tail-first glide in the opposite direction and lands 145 m behind the launch point**. The
CMA-ES design flies that same launch +154.8 m forwards. Clip: `media/tumble_backwards.gif`.

This is not numerical: halving dt gives −145.02 m both times, and energy never rises in still
air. It is a **loophole in the simplified aerodynamics**. Past stall the model uses a
fore-aft symmetric flat-plate law (CL = 1.1 sin 2α, CD = CD0 + 1.8 sin²α) with no induced drag,
so a flat sheet flying backwards still gets a glide ratio of about 4–5. A thin flat sheet
really is roughly fore-aft symmetric, so this is not absurd. Still, the model never charges
the reversed wing its induced drag, and a real tail-first glider would not keep this stable
trim. Fitness on held-out launches was signed downrange distance, so this counts as −145 m. A
"distance travelled" metric would have scored it as a huge success. We left the metric alone.

## Stability (linearised around the steady still-air glide)

| design | glide ratio L/D | glide speed (m/s) | slowest eigenvalue real part | phugoid ζ / period | short-period ζ | static margin (formula) | held-out pitch-rate RMS |
|---|---|---|---|---|---|---|---|
| baseline | 8.2 | 5.03 | −0.176 | 0.087 / 3.1 s | 0.73 | +0.29 | 0.57 rad/s |
| CMA-ES best | 11.9 | 5.57 | −0.094 | 0.049 / 3.3 s | 0.66 | −0.01 | 0.38 rad/s |
| random best | 10.7 | 4.69 | −0.220 | 0.194 / 5.6 s | 0.93 | +0.03 | 0.17 rad/s |
| MAP-Elites best | 11.8 | 5.65 | −0.141 | 0.063 / 2.8 s | 0.46 | −0.01 | 0.65 rad/s |
| archive "smooth" | 10.2 | 6.08 | −0.195 | 0.249 / 8.3 s | overdamped | +0.25 | 0.11 rad/s |
| archive "porpoise" | 10.0 | 4.63 | −0.047 | 0.019 / 2.5 s | 0.64 | +0.16 | 0.98 rad/s |

What evolution did, and it is the textbook result: it maxed the span (50 cm bound, aspect ratio
about 13), shrank the tail to near its lower bound, piled on nose ballast (16–19 g), and moved
the CG to **near-neutral stability**. That buys L/D 11.9 against 8.2, but halves the phugoid
damping (0.049 against 0.087). The cost shows up in robustness: held-out range spread is
36 m for CMA-ES against 22 m for the baseline. The evolved glider's range on held-out launches
is 74 % of its still-air range; the baseline's is 79 %. All evolved designs are still
dynamically stable (every eigenvalue has negative real part), even though the static-margin
formula calls them very slightly unstable (−0.01). The formula is wrong here because it uses
the *linear* wing lift slope. Every design trims on the soft shoulder of the stall model: the
wing sits at about 7° angle of attack (stall centre 10°), where the stall blend is already
13–18 % active. The local slope dCL/dα there is 3.06 /rad for CMA-ES best, against 4.90
linear (−38 %), and 3.00 against 4.04 for the baseline (−26 %). A softer wing slope moves the
true neutral point aft, so the real margin is positive. The same fact explains part of the
held-out fragility: the evolved designs trim closer to stall (CL 0.55–0.57 against 0.45 for
the baseline), so a gust adding a few degrees of α half-stalls them. That fits the nose-dive
on held-out #56 and the tumble on #10. Trust the eigenvalues, not the formula (numbers from
`sim.glide_eigen` + `sim._coeffs` at the trim state).

## Diversity across the archive

MAP-Elites, 120k evaluations. Descriptors: still-air flight time (0–24 s, 24 bins) × log10
pitch-rate RMS (−3 to 1, 20 bins).
- **Coverage:** 399/480 cells (83 %). **QD score** (sum of positive elite fitness):
  28,054 m.
- **163 elites beat the baseline's training range** (78.6 m).
- Mean pairwise gene-space distance between elites is 1.01 (8-D unit cube). Every gene keeps a
  std of 0.21–0.38 across the archive, and span covers the full 10–50 cm range.
- Representative "species" (`media/planforms.png`): a **floater** (23 s flights, long 35 cm
  tail arm, 114.8 m still air), a **smooth dart** (porpoising RMS 0.001 rad/s, 27.6 g, glide
  speed 6.1 m/s), and a **porpoiser** (phugoid ζ 0.019). The heatmap is in
  `media/search_and_archive.png` (right). The best region is 17–23 s flights with low
  pitch-rate RMS, and range falls off toward both "porpoising" and "short flight".
- MAP-Elites' best (115.3 m) came close to CMA-ES (118.1 m) but used 6× CMA-ES's total
  evaluations. The budgets are **not** matched between MAP-Elites and CMA-ES/random; they are
  matched between CMA-ES and random search (4000 per seed, 5 seeds).

## Baseline fairness

The baseline is a textbook hand design (AR 5, tail-volume 0.75 by `sim.decode`'s formula, CG ≈ 30 % chord, 3°
decalage). Its ballast and tail incidence were chosen from 18 hand-trim test flights in still
air, roughly what a person test-throwing a glider would do. No automatic search touched it. The
random-search baseline (same simulator, same 6 training launches, same 4000-eval budget per
seed as CMA-ES) is the matched-compute comparison.

**Reviewer check: the hand baseline is a fair-looking glider but leaves easy range on the table**
(`review/baseline_probe.py` → `results/review_baseline_probe.json`, `results/review_baseline_probe_stdout.txt`).
The probe keeps the baseline's textbook tail (25 % area, 18 cm arm) and re-trims only ballast,
ballast position and the two incidences on a 972-point grid scored on the **still-air throw only**,
never on held-out launches. The "18 hand-trim flights" behind the shipped trim are not on disk.

| hand design (still-air trim) | still air (m) | training mean (m) | held-out mean (m) | held-out p10 / worst (m) | CMA-ES best wins on held-out |
|---|---|---|---|---|---|
| shipped baseline, 30 cm, AR 5 | 85.7 | 78.6 | 68.1 | 44.8 / 6.1 | 58/64 |
| same 30 cm planform, re-trimmed | 92.4 | 89.1 | 68.8 | 25.1 / 14.6 | 55/64 |
| stretched to 50 cm span, 6 cm chord (AR 8.3) | 112.7 | 104.1 | **91.8** | 60.5 / 17.5 | 36/64 |
| 50 cm span, 5 cm chord (AR 10) | 114.8 | 106.1 | **92.2** | 57.8 / 16.2 | 34/64 |
| 50 cm span, 4 cm chord (AR 12.5) | 115.8 | 106.5 | **94.7** | 61.9 / 17.5 | 31/64 |
| *CMA-ES best, for reference* | 122.1 | 118.1 | 89.8 | 49.3 / 13.3 | – |

Read: the shipped baseline is not a strawman shape, but it uses 30 cm of a 50 cm span allowance,
and the gene bounds make span free (no spar mass). Using the bound is something any human designer
told "up to 50 cm, longer is better" would do. Once a hand design uses it, CMA-ES's training-set
advantage (118 m against about 106 m) does **not** carry over to held-out launches: the stretched
textbook gliders, with a real static margin and a big tail, match or beat it on held-out mean and
p10, and CMA-ES wins only about half the paired launches. That fits the stability section: evolution
bought training range with near-neutral trim close to stall, which is what loses on unseen gusts.
The grid is itself a small search (972 still-air flights, more than the 18 claimed for the shipped
trim), so these are "a careful hand designer who uses the span bound", not a zero-effort baseline.

## What worked
- A rigid-body 3-DOF model with forces at each surface's own airspeed gives realistic
  behaviour: CG behind the neutral point crashes, nose ballast stabilises, phugoid and
  short-period modes come out right, and still-air energy is provably non-increasing.
- Numba keeps it cheap: a design evaluation (6 flights, about 25k RK4 steps) averaged 4.7 ms
  over the main run's 160k RK4 evaluations, on 4 threads under load (load average about 14).
  The whole study took 13 minutes of wall time.
- CMA-ES was reliable: 5 of 5 seeds landed within 3 m of each other.

## What didn't / surprises
- **No numerical exploit**, even with a deliberately bad integrator. The physics damps too
  well. The exploit we did find is in the aerodynamic *model* (reverse flight), not in the
  integrator.
- Evolution converges on the gene bounds (max span, min tail, max ballast region). The ranking
  depends on those bounds, and a structural or mass model for long spans would change it.
- Optimising for range trades away stability margin, so the held-out worst case and spread got
  worse even though the mean improved a lot.
- The original `results/search_log.json` was overwritten by a smoke run before I made smoke logs
  go elsewhere. It was rebuilt from `checkpoints/search.npz` and the original stdout; the file
  notes this. `results/search_stdout.txt` is the untouched original.

## Not measured
- Turning radius and any lateral or directional dynamics (the model is 2D).
- Robust (worst-case or CVaR) optimisation over gusts, the "Go deeper" step.
- Surrogate models.
- Multiple MAP-Elites seeds.
- Idle-machine timing.
- Real paper planes.

## Next steps
1. Fix the reverse-flight loophole (an asymmetric post-stall model plus induced drag at all
   α), then re-run MAP-Elites and see whether the tailless champions survive.
2. Optimise a robust objective (held-out-style gust ensemble, CVaR of range) and measure the
   range-vs-stability-margin Pareto front.
3. Add a span-dependent spar mass so "max span" stops being free.
4. Add roll and yaw (light 3D) so turning radius can become a MAP-Elites descriptor.

## 12. Round-3 follow-up: span with a structural cost (2026-10-22, 20:04–20:45)

> **Assumptions first:** the strength limit is an assumed allowable stress (σ = 0.25 MPa headline; 0.5 and
> 0.125 MPa sensitivity) on an assumed 3 mm flat foam sheet. None of these is a measurement, and the conclusions
> depend on σ (see Sensitivity). **Cross-review (20:30):** the break check skips the release state for every
> design, and that gap, not threshold sensitivity, causes every dt-halving failure below. The review audit
> (`struct_release_audit.py` → `results/struct/release_audit_all.txt`) applies one release correction to every
> design and seed. The conclusions survive it. Corrected numbers are given in "Review audit" at the end of this section.

Why: the reviewer found span was free (no structure in the model), so every evolved design sat at
the 50 cm bound, and a textbook glider stretched to 50 cm matched CMA-ES on held-out launches.
All numbers below come from `results/struct/eval.json` and `results/struct/stdout.txt`
(`struct_followup.py`, one main run of 665 s, 4 numba threads, through the slot lock, under load:
load average 10.9 / 13.5 / 9.9 at the end). Plot: `media/struct_followup.png`. Winners:
`checkpoints/struct/designs.json`; every seed, the CMA state and the whole hand grid:
`checkpoints/struct/search_s<sigma>.npz`.

**Model variant (`sim_struct.py`; `sim.py` is unchanged and the earlier sections reproduce).** The
wing is the flat 3 mm foam sheet the original mass model already implies (0.12 kg/m² = 3 mm × 40 kg/m³).
Its root can carry a bending moment M_cap = σ·c·t²/6 (flat-plate section modulus). In flight the root
moment is |L_wing|·b/(3π)·(1 − m_wing/m) (elliptic lift, so half the lift acts 4(b/2)/(3π) from the root,
with relief from the wing's own weight). If it exceeds M_cap, the wing folds. The flight ends there and
scores its downrange x at the break. This makes the safe load factor fall with span, aspect ratio and
total weight. Why a strength limit and not just spar mass: from a fixed height, glide range depends on
L/D, not on weight, so adding mass would barely cost anything. With `sigma=1e12` the variant
reproduces `sim.py` bit for bit (`tests/test_struct.py`).

**Constants, fixed before the run:** t = 3 mm. The **headline allowable stress is σ = 0.25 MPa**,
which is 0.5 MPa (roughly the flexural strength of extruded-polystyrene foam sheet) with a safety factor of 2.
Sensitivity levels are 0.5 MPa (no safety factor) and 0.125 MPa (weak or damp sheet). All three
are assumptions, not measurements. A probe before the run showed that 0.5 MPa never binds on the six training
launches for the old CMA-ES winner (peak load 0.59 of capacity).

**Arms, per level, all on the same 6 training launches:** CMA-ES and random search, each 5 seeds ×
4000 evals. The hand baseline is the textbook shape (tail 25 % of wing area on an 18 cm arm, as shipped),
with span {30…50 cm} × chord {4, 6, 8 cm} × ballast × ballast position × both incidences picked from a
4050-design grid scored on the **training mean**. That is about one CMA seed's budget, and it is more trimming than
the reviewer's still-air probe. The held-out set is the same 64 launches (`heldout_conditions(seed=12345)`).

### Headline (σ = 0.25 MPa): span is still free on the training launches, so evolution ignores the limit and loses held-out

| design | span × chord, AR, mass | safe load factor | training mean (m) | held-out mean / median (m) | held-out wing breaks | CMA-ES best wins (of 64) |
|---|---|---|---|---|---|---|
| **CMA-ES best** | 50 × 3.8 cm, 13.3, 20.2 g | 1.51 | **118.0** | 57.9 / 64.8 | 27/64 | – |
| random-search best | 49.7 × 5.7 cm, 8.7, 20.0 g | 2.51 | 100.9 | 68.1 / 73.4 | 10/64 | 34 |
| **hand design, re-trimmed on training** | 50 × 8.0 cm, 6.2, 29.6 g | 2.33 | 104.0 | **69.0** / 70.2 | 11/64 | 31 |
| shipped hand baseline (30 cm) | 30 × 6 cm, 5.0, 13.0 g | 6.66 | 78.6 | 68.1 / 64.1 | **0/64** | 35 |
| reviewer's stretched baseline (50 × 6 cm) | 50 × 6 cm, 8.3, 28.0 g | 1.77 | 104.1 | **69.8** / 82.3 | 19/64 | 29 |
| old CMA-ES winner (original physics) | 50 × 3.9 cm, 12.9, 21.5 g | 1.46 | 101.1 | 51.5 | 31/64 | 38 |

Across seeds, held-out means average **CMA-ES 57.9 m** (44.5–65.4, 16–33 breaks per seed) against **random
search 70.5 m** (65.9–78.0, 4–12 breaks). Under the headline physics, matched-budget random search beats
CMA-ES on held-out launches.

What happened: none of the six training launches (4.5–8 m/s, mild gusts) loads any winner past its limit.
The CMA-ES winner peaks at 0.89 of capacity on training, the hand design at 0.91. So the limit only
trims the training landscape slightly (the hand grid's best per span still rises monotonically:
87.4 m at 30 cm → 104.0 m at 50 cm), and CMA-ES converges on the same design as before: 50 cm span, AR 13, minimum tail,
near-neutral static margin. It rides just under the break line of the training loads. The held-out set
throws up to 9 m/s into stronger gusts, so the AR-13 wing folds on 27 of 64 launches (often at release:
range 0, p10 0 for every 50 cm design). **Evolution did not find anything the hand design misses here.
It found the edge of the training distribution.**

### Sensitivity

| σ | CMA-ES best: span × chord, mass, n_safe | train CMA / hand / random (m) | held-out mean CMA / hand / random (m) | 5-seed held-out mean CMA / random | held-out breaks CMA / hand |
|---|---|---|---|---|---|
| 0.5 MPa | 50 × 3.9 cm, 24.3 g, 2.57 | 117.7 / 105.4 / 100.9 | 88.1 / 88.9 / 77.3 | 86.3 / 79.6 | 6 / 7 |
| **0.25 MPa** | 50 × 3.8 cm, 20.2 g, 1.51 | 118.0 / 104.0 / 100.9 | 57.9 / 69.0 / 68.1 | 57.9 / 70.5 | 27 / 11 |
| 0.125 MPa | 50 × 4.5 cm, **15.7 g**, 1.23 | 109.9 / 95.8 / 92.8 | 43.2 / 43.3 / 41.6 | 40.5 / 55.2 | 38 / 26 |

- **0.5 MPa:** the limit is nearly irrelevant. The result is the round-2 one again: CMA-ES +12 m on training,
  a tie with the hand design on held-out (88.1 against 88.9 m; the reviewer's 50 × 4 cm glider gets 91.7 m).
- **0.125 MPa is the only level where the trade-off is real on the training launches.** The hand designer's
  optimum span moves inside the bounds (best training range per span: 87.4 / 90.8 / **95.8** / 93.0 / 89.8 m at
  30 / 35 / 40 / 45 / 50 cm; hand pick 40 × 6 cm, 23 g), and the old 50 cm winners fold on all six training launches
  (training mean 0.3–1.0 m). CMA-ES kept the full 50 cm span with a light glider (15.7 g, 10.5 g ballast, 4.5 cm
  chord, full-size tail on an 8 cm arm) and scored 109.9 m on training, 14 m above the hand grid. **Most of that
  edge is a loophole in my break check (the good kind of failure).** `sim_struct.simulate` checks the load after
  every step but never on the release state itself. The CMA-ES winner is already over its limit at release on 2 of
  the 6 training launches (load ratio 1.02 and 1.19), and the load drops back under 1 within the first 5 ms. With those two
  launches scored as folded (0 m), its training mean is **78.0 m, below the hand design's 95.8 m and random search's
  92.8 m** (post-run audit, `results/struct/release_check_audit.txt`). Its training dt-halving failure is the same
  gap: at 2.5 ms the first check comes earlier and launch 6 breaks. The hand and random winners do not
  use the gap on training. Held-out (corrected): CMA-ES 35.2 m, hand 42.1 m, random 40.0 m. The code and checkpoints
  were left as run. Fix for next time: check the release state, or require a load margin.
- At every level the **shipped 30 cm baseline** (safe load factor 3.3–13) has the best held-out
  p10 and the fewest breaks (0, 0 and 3). At 0.25 MPa its held-out mean (68.1 m) is within 2 m of the best
  50 cm designs (69.0–69.8 m). At 0.125 MPa it is the best design by 23 m (66.3 against at most 43.3 m). Under a real
  strength limit, the boring short wing is the robust choice.

### Sanity checks for the new winners (same tolerances as before: 0.5 % or 5 cm on halving dt, 1e-9 energy)
- **Energy:** every design at every level passes. The maximum per-step energy rise in still air is 0. The
  break only ends a flight, so it can't create energy.
- **dt-halving:** passes on training and held-out launches for the 0.5 MPa CMA-ES winner, for the hand designs at
  all levels, and for the shipped baseline. It fails on one to four launches for: CMA-ES 0.25 MPa (1 held-out),
  random 0.25 MPa (1), CMA-ES 0.125 MPa (1 training + 3 held-out) and random 0.125 MPa (2 held-out).
  *[Corrected by cross-review; the builder wrote "threshold sensitivity, not an integrator exploit".]* The review
  audit shows **every** dt-halving failure, for every design and seed at every σ (2 + 15, 0 + 3 and 7 + 33
  training + held-out launch failures at 0.25 / 0.5 / 0.125 MPa), is a launch whose **release state already
  exceeds the limit**. At dt/2 the first check comes earlier and the wing folds at about 0.01–0.02 m. At dt it
  can recover first. So the failures come from the release-check gap (a bug in the break check), not from
  in-flight breaks near the threshold. With the release correction applied, there are **0 dt-halving failures**
  at every level.
- Every winner is still dynamically stable (all eigenvalues negative; L/D 10.1–11.9).

### Backwards-glide loophole
The aerodynamic loophole is **still in the model**: the post-stall flat-plate law is unchanged. It
**no longer shows up**, though: 0 of 64 held-out launches land behind the start for any design at any σ, and
this was also true for the new winners under the original physics (`negative_original_physics` = 0). The old
MAP-Elites flying wing that glided back 145 m on held-out #10 (8.8 m/s throw) now **folds its wing at
release** at all three σ (range 0, t = 0). A strength limit masks the loophole, because the tumble needs a
wing that survives the gust, but it does not fix it. MAP-Elites was not rerun under the struct model.

### Caveats / not measured
- The break check skips the release state (see the 0.125 MPa bullet). The post-run audit changes nothing on training at
  0.25 / 0.5 MPa for the three winners and moves held-out means by 0–4 m (`results/struct/release_check_audit.txt`; the
  all-design review audit `results/struct/release_audit_all.txt` agrees within 0.4 m, see "Review audit"). The searches were not rerun
  with the fix.
- The hand grid has 4050 evaluations, against CMA-ES's 20,000 in total with the best of 5 seeds chosen, so the training-set
  comparison slightly favours CMA-ES. That does not affect the held-out conclusion at 0.25 MPa, where the hand design already wins.
- σ and t are assumed, not measured, and there is no stiffness or aeroelastic model: a 3 mm foam wing of AR 13 would
  also bend a lot (deflection not modelled), no spar option, and no thickness gene.
- MAP-Elites and the cheap-Euler arm were not rerun under the struct model. There is no new GIF; the demo is the PNG.
- Timing is a busy-machine number (load average 10–14).

### Answer to the follow-up question
Adding a structural cost does not by itself give evolution an edge that transfers. When the limit doesn't bind on
the training launches (0.5 and 0.25 MPa), CMA-ES lands on the same 50 cm AR-13 design and rides
just under the training loads. At 0.25 MPa that design then breaks on 42 % of held-out launches and loses to
the re-trimmed hand design (57.9 against 69.0 m) and to random search. When the limit does bind on training
(0.125 MPa), the "non-hand" answer CMA-ES found (keep 50 cm, go light) gets most of its +14 m training edge by
exploiting the unchecked release step. With the gap closed it scores 78.0 m against the hand design's 95.8 m. So at no
level does evolution find a design that beats the fairly re-trimmed hand design on held-out launches. What it reliably
finds is the edge of whatever the training objective checks: the training load cases at 0.25 MPa, and the release
step at 0.125 MPa. For a structural cost to shape designs that hold up, it has to be
in the objective's load cases: harder training throws, or a robust or CVaR objective over held-out-style
gusts. That is the obvious next run (not measured).

### Review audit (cross-review, 20:30): one release correction for every design
The builder's audit (`release_check_audit.txt`, an inline snippet that was not saved) covered only the three winners. It also
used a proxy: "max load > 1 but no break". That proxy misses launches that were over the limit at release and then
broke again later, which were scored at the later break's x. `struct_release_audit.py` computes the
release-state load ratio directly. It scores any launch with a ratio above 1 as 0 m, for every design in `eval.json` and every
per-seed winner, at both dt. Release exceedances are common on the held-out launches. At 0.25 MPa: CMA-ES 17/64, hand 9,
random 12, stretched baseline 12, shipped baseline 0. Most of these launches were already counted as breaks, so the means move little:

| σ | held-out mean, corrected: CMA-ES / hand / random / shipped 30 cm / stretched 50×6 | 5-seed held-out mean, corrected: CMA / random | training, corrected: CMA / hand / random |
|---|---|---|---|
| 0.5 MPa | 88.1 / 88.9 / 76.9 / 68.1 / 91.6 | 86.3 / 79.5 | 117.7 / 105.4 / 100.9 |
| 0.25 MPa | 56.5 / 68.0 / 64.0 / 68.1 / 69.7 | 56.3 / 66.7 | 118.0 / 104.0 / 100.9 |
| 0.125 MPa | 35.2 / 42.1 / 39.9 / 66.3 / 1.2 | 34.4 / 52.6 | 78.0 / 95.8 / 92.8 |

- The headline holds. At 0.25 MPa no training launch overloads any of the three winners, including at release. Held-out, CMA-ES
  (56.5 m) loses to the hand design (68.0 m) and to random search. With the correction, the shipped 30 cm baseline
  (68.1 m, 0 breaks) ties the hand re-trim and the 50 × 6 cm stretched design (69.7 m) is nominally best.
  Unselected seeds do use the gap on training: CMA-ES seed 0 at 0.25 MPa (109.9 → 94.9 m) and every CMA-ES seed at 0.125 MPa (−17 to −32 m).
- At 0.125 MPa, the random-search **seed** winners (held-out 58.8 / 66.1 / 39.9 / 44.6 / 53.4 m, mean 52.6) beat the hand pick
  (42.1 m) on held-out in 4 of 5 seeds. Selecting the best training score picked the worst of them. So "nothing beats the
  re-trimmed hand design held-out" is true for designs chosen on training, which is the fair comparison. It does not mean the
  hand design is robust at 0.125 MPa. Only the shipped short wing is (66.3 m, 3 breaks).
- Selection: CMA-ES and random-search winners are the argmax of training fitness over the seeds, and the hand pick is the argmax of
  training mean over the grid. The held-out launches are never used for selection (checked in `struct_followup.py`).

