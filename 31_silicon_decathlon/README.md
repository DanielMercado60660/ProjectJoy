# 31 · Silicon Decathlon: race the shelf's own models across Apple Silicon

Ten trained models from the ProjectJoy shelf, each one an **event**, raced across every engine this
M5 Max offers, each one an **athlete**: NumPy fp32 on the CPU (the baseline), MLX on the CPU, MLX on the GPU
(fp32 reference, fp16, `mx.compile`, and int8/int4 `nn.quantize` for Linear-heavy models), and Core ML
(fp16 `.mlpackage`) on the CPU, GPU and Neural Engine. For each event and athlete we record batch-1 latency
(median, p90), throughput at the event's stated batch, peak MLX memory, agreement with the MLX fp32
reference, a task-level check (does the car still finish, does the net pick the same move, does the
tracker still follow the hidden ball), Core ML's compute-plan op placement, and a **qualifies** verdict.
A fast athlete that gives wrong answers is not allowed a medal.

Results and findings: [RESULTS.md](RESULTS.md). Full tables: [results/summary.md](results/summary.md).
Demo: open `media/demo.html` (offline leaderboard: pick an event and batch, watch the race, toggle
"qualifying only"). Charts: `media/leaderboard.png`, `media/crossover.png`.

## How to run
From this folder, with the shared venv. Dev runs are written as `results/<event>_provisional.json`;
only the scored pass writes `results/<event>.json`.

```bash
./smoke.sh                                    # < 1 min: harness + report tests, e15 parity, e09 + e15 at b=1, report
../.venv/bin/python -m pytest -q              # every test (harness, report, per-event parity)

# dev run of one event (light: shared machine)
../.venv/bin/python run_event.py e15_object_permanence --repeats 3 --budget 0.1
../.venv/bin/python run_event.py e15_object_permanence --true-fp32 --repeats 3 --budget 0.1   # true-fp32 MLX reference
../.venv/bin/python run_event.py e09_alphazero --no-task --batches 1 --repeats 1 --budget 0.02 --out logs/smoke

# the scored pass (orchestrator only; waits for an idle machine; forces true-fp32 MLX and aborts if it isn't)
# (exactly as run 18:02-18:12 on 2026-10-22; NumPy/MLX CPU BLAS capped at 4 threads by the shelf's sharing rule)
MLX_ENABLE_TF32=0 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python run_event.py all --final
# optional: MLX GPU with its default TF32 matmuls, vs the fp32 refs (results/refs/) the final pass saved.
# These athletes are shown as exhibition rows only (separate process, no task check, no medals).
VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python run_event.py all --tf32-pass

# the exhibit (prefers results/<event>.json over *_provisional.json and labels which it used)
../.venv/bin/python make_report.py

# Core ML batch-1 placement probe (RESULTS.md, finding 1)
../.venv/bin/python probe_coreml_batch1.py

# reviewer probe: does the 4-thread BLAS cap handicap NumPy? (~2.5 min; RESULTS.md, "How to read ×NumPy")
../bin/gpu_run results/probes/numpy_threads/run_probe.sh
```

Test status at review (2026-10-22): `./smoke.sh` OK in 12 s; `pytest -q` 42 passed in ~7 s.

`run_event.py` flags: `--final` (scored pass), `--repeats N --budget S` (rounds and seconds per athlete per
round), `--only a b` (subset; `mlx_gpu` always runs), `--no-task` (skip task checks), `--batches 1`
(subset of the event's batches), `--out DIR` (write elsewhere than `results/`), `--true-fp32`,
`--tf32-pass`. `make_report.py --results DIR --out DIR` builds the report from another folder (smoke).

## How timing works
`bench.race` warms every athlete up, then runs rounds; in each round every athlete runs back to back for
`budget` seconds, in a shuffled order, so drift hits everyone alike. Calls are host to host (NumPy in,
NumPy out), so each engine pays its own transfer and sync. A file lock (`.bench.lock`) keeps timed
sections of different processes from overlapping. Every result records load average, power source,
thermal state, busy processes and a fixed noise probe before and after. MLX GPU precision is measured
(`mlx_gpu_matmul_rel_err`) and stored with every result.

## File layout
| path | what |
|---|---|
| `bench.py` | timing race, peak MLX memory, environment snapshot, noise probe, MLX matmul precision probe |
| `run_event.py` | runs one event or all: agreement, race, placement, task checks, `qualifies`, writes JSON |
| `coreml_util.py` | torch-trace -> Core ML conversion (cached `.mlpackage`), per-unit loading, compute-plan placement |
| `events/eNN_*.py` | one module per event: `EVENT`, `BATCHES`, `make_inputs`, `athletes`, `SKIPS`, `agreement`, `task_check`, `qualifies`, `coreml_packages` |
| `models/*.py` | minimal copies of each sibling project's model (MLX reference, NumPy baseline, torch frontend for Core ML); weights are read from `../NN_project/checkpoints/` |
| `tests/test_eNN_*.py` | per-event parity against the sibling's own model code (verification-only import) |
| `tests/test_harness.py`, `tests/check_demo.js` | harness, report and demo-page tests |
| `make_report.py`, `demo_template.html` | builds `results/summary.md`, `media/leaderboard.png`, `media/crossover.png`, `media/demo.html` |
| `probe_coreml_batch1.py` | Core ML compute-plan / timing / CPU-time probe at batch 1 -> `results/probes/coreml_batch1.json` |
| `results/` | `<event>.json` (final), `<event>_provisional.json` (dev), `<event>_tf32.json` (TF32 pass), `refs/` (true-fp32 reference outputs), `probes/`, `summary.md` |
| `checkpoints/coreml/` | exported Core ML packages (fp16, one per event and batch; **gitignored**, 148 MB for the 21 event packages + 189 MB for the 6 `probe_*` packages; rebuilt from the sibling checkpoints on first use by `run_event.py <event>`, and the probe packages by `probe_coreml_batch1.py`) |
| `media/` | demo page and charts |
| `smoke.sh` | end-to-end smoke check |

No new models are trained here. torch is used only as a Core ML conversion frontend. The "saved
checkpoint" of this project is the set of Core ML packages in `checkpoints/coreml/` (gitignored, 337 MB in
all, too big to commit; any `run_event.py <event>` call regenerates that event's packages from the sibling
checkpoint, deterministically from the same weights) plus the sibling checkpoints each `models/*.py` names,
plus `results/refs/` (27 MB of true-fp32 reference outputs, committed; `--tf32-pass` needs them and
`--final` rewrites them). `make_report.py` reproduces every chart and the demo from `results/`.

## Round 5: observation-to-action control loops (measured)

[DESIGN_R5.md](DESIGN_R5.md) specifies the two-event experiment.
All 288 planned episodes completed on AC. [Measured findings](RESULTS.md#round-5-benchmark-the-whole-observation-to-action-loop)
and [full receipt-linked tables](results/r5/analysis_turn2.md) cover every engine, policy, load and delay.
At 10 Hz (car) and 30 Hz (tracker), every engine acts within 1.2–3.3 ms of the frame with zero
missed deadlines, and all cars finish. Engine choice barely matters: Core ML CPU's lowest tracker
error is a ~2% latency effect. With an explicit 50 ms service delay, newest-frame processing reduces
pursuit error 49.86% versus FIFO, improving all 48 matched pairs across six seeds. The reviewer's
corrections: the age R² is design-driven, the round-3 ranking
match holds for the tracker but not the car, and the loaded condition is a light GPU keep-alive.

Replay the [nominal](media/r5_nominal.html), [stress](media/r5_stress.html), and
[old-secondary](media/r5_old_secondary.html) runs offline. The old comparison contains one old seed.
New work uses `r5_*.py`, `analyze_r5.py`, `review_r5_checks.py`, `tests/test_r5_numpy.py`, `tests/test_r5_analysis.py`, `results/r5/`, `logs/r5/`,
`checkpoints/r5_selected/` and `media/r5_*.html`. Earlier exhibit commands and artifacts are unchanged.

Commands as run on 2026-10-24 (AC power), from this folder. The run commands are verbatim from the
`gpu_run` log headers. The plan invocations are reconstructed from the original runbook and the plan JSONs'
targets. All were run after
`export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MLX_ENABLE_TF32=0 PYTHONDONTWRITEBYTECODE=1`:

```sh
# 1. smoke: NumPy tests, all 7 engine paths, parity, concurrent load, real kill/resume, report (~1 min)
../bin/gpu_run ./smoke_r5.sh 2>&1 | tee logs/r5/smoke.log
# 2. plans from the measured pilot (<1 s)
../.venv/bin/python r5_run.py plan --target-minutes 23 --out results/r5/nominal_plan.json
../.venv/bin/python r5_run.py plan --mode stress --target-minutes 5 --seed-start 920031 --out results/r5/stress_plan.json
../.venv/bin/python r5_run.py plan --split old --seed-start 0 --target-minutes 12 --out results/r5/old_plan.json
# 3. runs: nominal 14:23-14:45 (64 episodes), stress 14:45-14:50 (192), old_secondary 14:50-15:01 (32)
../bin/gpu_run ../.venv/bin/python r5_run.py run --plan results/r5/nominal_plan.json --name nominal --max-minutes 25 2>&1 | tee logs/r5/nominal.log
../bin/gpu_run ../.venv/bin/python r5_run.py run --plan results/r5/stress_plan.json --name stress --max-minutes 6 2>&1 | tee logs/r5/stress.log
../bin/gpu_run ../.venv/bin/python r5_run.py run --plan results/r5/old_plan.json --name old_secondary --max-minutes 15 2>&1 | tee logs/r5/old_secondary.log
# 4. offline replays (<5 s each)
../.venv/bin/python r5_report.py --run nominal --out media/r5_nominal.html
../.venv/bin/python r5_report.py --run stress --out media/r5_stress.html
../.venv/bin/python r5_report.py --run old_secondary --out media/r5_old_secondary.html
# 5. review (no MLX, a few seconds): tests, receipt re-audit, reviewer checks
../.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_r5_analysis.py tests/test_r5_numpy.py --basetemp=logs/r5/pytest_turn2_review 2>&1 | tee logs/r5/numpy_tests_turn2_review.log
../.venv/bin/python analyze_r5.py --out-prefix results/r5/analysis_turn2_review 2>&1 | tee logs/r5/analysis_turn2_review.log
../.venv/bin/python review_r5_checks.py | tee results/r5/review_r5_checks.txt
```

The models stay frozen. NumPy uses the same weights as MLX and Core ML. Calibration copies one
selected weight generation and records unavailable engines; no round-5 training or conversion occurs.
The car's finer integration and the tracker's pursuit controller are new task conditions, so round-3
success rates are context, not round-5 results.

`analyze_r5.py` audits plans/source hashes and frame accounting, writes pooled timing/stage tables,
matches scheduling and load comparisons by seed, and quantifies age versus task error using
leave-one-seed-out regressions. `tests/test_r5_analysis.py` checks those statistics without MLX.
The existing runner, checkpoints, receipts, logs and replays were retained during turn 2.
