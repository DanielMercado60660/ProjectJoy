# 12 — Paper Plane Darwin

Evolve small hand-launched gliders in a deliberately simple 2D aerodynamic simulator, compare
them with a hand-designed glider, and check that the "better" designs are better physics, not
better at fooling the integrator.

- **Simulator** (`sim.py`, numba): rigid body in the vertical plane (x, z, pitch). Wing and tail
  each get lift perpendicular to, and drag opposite to, the air velocity *at that surface*, so
  pitch rate damps through the tail. Lift slope from aspect ratio, smooth stall into flat-plate
  post-stall coefficients, low-Reynolds skin friction (CD0 grows as chord shrinks), induced drag,
  wing downwash on the tail, fuselage drag. Mass and pitch inertia come from the geometry
  (sheet wings, a stick fuselage, a nose ballast blob). Frozen gust fields and head/tail wind.
  RK4 at 5 ms; explicit Euler at 20 ms as a deliberately "cheap" alternative.
  In still air the exact dynamics cannot gain energy (lift does no work, drag only removes it,
  and there is no airfoil Cm0 couple), so any energy gain is integration error.
- **Genome** (8 genes in [0, 1]): span 10–50 cm, chord 3–15 cm, tail area 5–60 % of wing,
  tail arm 5–35 cm, nose ballast 0–20 g, ballast position 0–20 cm ahead of the wing,
  wing incidence −4…+8°, tail incidence −8…+4°.
- **Objective**: mean range over 6 fixed training launches (10 m balcony, 6 m/s level throw in
  still air, plus five mild variations with seeded gusts and wind).
- **Search** (`search.py`): CMA-ES (written from scratch), random search at the same budget
  (the cheap baseline), a MAP-Elites archive (descriptors: still-air flight time × log
  pitch-rate RMS, i.e. "floaters vs darts" × "smooth vs porpoising"), and CMA-ES on the cheap
  Euler simulator to see whether evolution exploits integrator error.
- **Hand baseline** (`evaluate.py: BASELINE_UNITS`): a textbook glider (AR 5, tail-volume 0.75 by the code's formula,
  CG ≈ 30 % chord, 3° decalage), with nose ballast and tail incidence chosen from 18 hand-trim
  test flights.
- **Audit** (`evaluate.py`): for each candidate, halve the step (range must agree within 0.5 %
  or 5 cm on all training and held-out launches), check per-step mechanical energy never rises
  in still air, fly 64 held-out launches (different heights, speeds, angles, wind, stronger
  gusts, ±15 % lift slope and ±30 % parasite drag), and linearise around the steady glide for
  stability modes.

## Commands

Run from this folder.

```bash
# tests + tiny end-to-end pipeline (~15 s)
./smoke.sh

# main run (≈ 13 min under load; CPU-only, 4 threads, through the shared slot lock)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python search.py

# audit + held-out evaluation (~1 min) -> results/eval.json, checkpoints/designs.json
NUMBA_NUM_THREADS=4 ../.venv/bin/python evaluate.py

# demo media (~1 min) -> media/*.png, media/flight.gif, media/tumble_backwards.gif
NUMBA_NUM_THREADS=4 ../.venv/bin/python render.py
```

Round-3 follow-up (structural cost, `sim_struct.py`; the original `sim.py` is unchanged and still the default):

```bash
# CMA-ES + random search (5 x 4000 each) + 4050-design hand-trim grid per stress level, then held-out
# evaluation, dt-halving/energy checks and the backwards-glide check (~11 min under load, 4 threads, slot lock)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python struct_followup.py | tee results/struct/stdout.txt
# plot -> media/struct_followup.png
../.venv/bin/python render_struct.py
```

Re-fly a struct-model design: `json.load(open("checkpoints/struct/designs.json"))["0.25MPa"]["cma_best"]["genes"]`
→ `sim_struct.fly(genes, sim.NOMINAL, np.zeros((4, 5)), sigma=0.25e6)`.

To re-fly a saved design: `json.load(open("checkpoints/designs.json"))["cma_best"]["genes"]`
→ `sim.fly(genes, sim.NOMINAL, np.zeros((4, 5)))`.

## Files

| Path | What |
|---|---|
| `sim.py` | glider model, integrators, launch/gust conditions, stability linearisation |
| `search.py` | CMA-ES, random search, MAP-Elites, cheap-sim CMA-ES (main run) |
| `evaluate.py` | sanity checks, held-out robustness, stability, archive diversity |
| `render.py` | plots and the GIF |
| `tests/test_physics.py`, `smoke.sh` | physics tests and the end-to-end smoke check |
| `checkpoints/search.npz` | every arm's best designs, CMA state (mean, sigma, C), learning curves, full MAP-Elites archive |
| `checkpoints/designs.json` | named designs (genes + physical units) used by the demo |
| `results/search_log.json`, `results/search_stdout.txt` | main-run log and timings |
| `results/eval.json` | every number in RESULTS.md except the reviewer's baseline probe |
| `review/baseline_probe.py`, `results/review_baseline_probe*.{json,txt}` | reviewer's check: hand baseline re-trimmed and stretched to the 50 cm span bound (~4 min, 4 threads) |
| `sim_struct.py` | round-3 model variant: wing root bending-strength limit (wing folds, flight ends); `sigma=1e12` reproduces `sim.py` exactly |
| `struct_followup.py`, `render_struct.py` | round-3 follow-up: searches, hand re-trim, evaluation under the struct model; its plot |
| `tests/test_struct.py` | struct model: reduces to the original when unbreakable, energy-safe, folds long thin wings on hard throws |
| `checkpoints/struct/` | `search_s<sigma>.npz` (all seeds' bests, CMA state, hand grid and scores), `designs.json` (winners per level) |
| `results/struct/eval.json`, `results/struct/stdout.txt`, `results/struct/release_check_audit.txt` | every number in the RESULTS.md round-3 follow-up section |
| `struct_release_audit.py` → `results/struct/release_audit_all.txt` | cross-review audit: release-state load check applied to every struct design and seed, and the cause of every dt-halving failure (~20 s: `NUMBA_NUM_THREADS=4 ../.venv/bin/python struct_release_audit.py`) |
| `media/` | `flight.gif` (baseline vs evolved), `tumble_backwards.gif` (the good failure), `flight_paths.png`, `planforms.png`, `heldout_ecdf.png`, `search_and_archive.png`, `sanity_checks.png` |
