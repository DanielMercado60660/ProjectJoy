#!/bin/bash
# Full experiment, resumable: each stage is skipped if its output already exists.
# Run under the shared GPU lock:   ../bin/gpu_run bash pipeline.sh > logs/pipeline.out 2>&1
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${PY:-../.venv/bin/python}
T0=$(date +%s)
elapsed() { echo $(( ($(date +%s) - T0) / 60 )); }
stage() { budget; echo "=== [$(date +%H:%M:%S) +$(elapsed)m] $*"; }
BUDGET=${BUDGET:-36}   # minutes; no stage is longer than ~8 min, so the slot is released before 45
budget() { if [ "$(elapsed)" -ge "$BUDGET" ]; then echo "=== budget of ${BUDGET}m reached; rerun to resume"; exit 0; fi; }

V1_STEPS=${V1_STEPS:-12000000}
FT_STEPS=${FT_STEPS:-6000000}
GENS=${GENS:-100}

# 1. solver on the training families
if [ ! -f checkpoints/solver_v1/DONE ]; then
  stage "train solver_v1"
  $PY train_solver.py --name solver_v1 --steps $V1_STEPS --envs 512 --minibatches 2 --eval-every 50
  touch checkpoints/solver_v1/DONE
fi

# 2. designer vs v1: seed 0 -> training archive, seed 1 -> held-out designer test levels
for s in 0 1; do
  if [ ! -f results/designer_v1_s$s.json ]; then
    stage "designer vs solver_v1 seed $s"
    $PY evolve_designer.py --solver checkpoints/solver_v1 --out v1_s$s --seed $s --gens $GENS
  fi
done

# 3. matched-compute fine-tunes from v1: curriculum (50% designer levels) vs control (families only)
if [ ! -f checkpoints/solver_cur1/DONE ]; then
  stage "fine-tune solver_cur1 (families + designer archive)"
  $PY make_curriculum.py results/curriculum_r1.npz v1_s0
  $PY train_solver.py --name solver_cur1 --init checkpoints/solver_v1 --extra results/curriculum_r1.npz \
      --extra-prob 0.5 --steps $FT_STEPS --envs 512 --minibatches 2 --eval-every 50 --seed 1
  touch checkpoints/solver_cur1/DONE
fi
if [ ! -f checkpoints/solver_ctl1/DONE ]; then
  stage "fine-tune solver_ctl1 (families only)"
  $PY train_solver.py --name solver_ctl1 --init checkpoints/solver_v1 --steps $FT_STEPS --envs 512 \
      --minibatches 2 --eval-every 50 --seed 1
  touch checkpoints/solver_ctl1/DONE
fi

# 4. fresh designer runs (same seed/budget) against each fine-tuned solver: how exploitable is it now?
for m in cur1 ctl1; do
  if [ ! -f results/designer_${m}_s2.json ]; then
    stage "designer vs solver_$m seed 2"
    $PY evolve_designer.py --solver checkpoints/solver_$m --out ${m}_s2 --seed 2 --gens $GENS
  fi
done

stage "evaluate (core)"
$PY evaluate.py $EVAL_ARGS

# 5. designer vs scripted baselines (CPU, quick) — what does each solver's nightmare look like?
for b in wall_follow tremaux; do
  if [ ! -f results/designer_${b}_s0.json ]; then
    stage "designer vs $b"
    $PY evolve_designer.py --solver $b --out ${b}_s0 --seed 0 --gens $GENS
  fi
done

# 6. unconstrained ablation (no solvability check) — short
if [ ! -f results/designer_v1_nocon.json ]; then
  stage "designer vs v1 WITHOUT solvability constraint"
  $PY evolve_designer.py --solver checkpoints/solver_v1 --out v1_nocon --seed 0 --gens 30 --no-constraint
fi
# neutral-drift control (same constraint, no selection)
if [ ! -f results/designer_v1_drift.json ]; then
  stage "neutral drift control vs v1"
  $PY evolve_designer.py --solver checkpoints/solver_v1 --out v1_drift --seed 0 --gens $GENS --no-select
fi

# 7. evaluation tables (again, now including the optional runs)
stage "evaluate (all)"
$PY evaluate.py $EVAL_ARGS
stage "pipeline done"
