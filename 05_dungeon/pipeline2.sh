#!/bin/bash
# Round 2+ experiments (run after pipeline.sh). Resumable: finished stages are skipped.
#   ../bin/gpu_run bash pipeline2.sh > logs/pipeline2.out 2>&1
#  A. memoryless ablation (GRU state zeroed every step), same budget as solver_v1
#  B. seed replicates of the round-1 curriculum / control fine-tunes
#  C. curriculum rounds 2 and 3 (designer vs latest curriculum solver -> growing archive) + matched controls
#  D. fresh designer runs against every new solver (exploitability)
#  E. designer with a longer minimum start-exit distance (12) against v1
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${PY:-../.venv/bin/python}
T0=$(date +%s)
elapsed() { echo $(( ($(date +%s) - T0) / 60 )); }
BUDGET=${BUDGET:-36}
stage() { if [ "$(elapsed)" -ge "$BUDGET" ]; then echo "=== budget reached; rerun to resume"; exit 0; fi; echo "=== [$(date +%H:%M:%S) +$(elapsed)m] $*"; }
V1_STEPS=${V1_STEPS:-12000000}
FT_STEPS=${FT_STEPS:-6000000}
GENS=${GENS:-100}
TR="--envs 512 --minibatches 2 --eval-every 50"

train() {  # name, extra args...
  local n=$1; shift
  if [ ! -f checkpoints/$n/DONE ]; then
    stage "train $n $*"
    $PY train_solver.py --name $n $TR "$@"
    touch checkpoints/$n/DONE
  fi
}
design() {  # solver-name out seed [extra]
  local sv=$1 out=$2 sd=$3; shift 3
  if [ ! -f results/designer_$out.json ]; then
    stage "designer vs $sv -> $out (seed $sd) $*"
    $PY evolve_designer.py --solver checkpoints/$sv --out $out --seed $sd --gens $GENS "$@"
  fi
}

# A
train solver_ff --steps $V1_STEPS --memoryless
# B
for s in 11 12; do
  train solver_cur1_r$s --init checkpoints/solver_v1 --extra results/curriculum_r1.npz --extra-prob 0.5 --steps $FT_STEPS --seed $s
  train solver_ctl1_r$s --init checkpoints/solver_v1 --steps $FT_STEPS --seed $s
done
# C round 2
design solver_cur1 cur1_s3 3
[ -f results/curriculum_r2.npz ] || $PY make_curriculum.py results/curriculum_r2.npz v1_s0 cur1_s3
train solver_cur2 --init checkpoints/solver_cur1 --extra results/curriculum_r2.npz --extra-prob 0.5 --steps $FT_STEPS --seed 2
train solver_ctl2 --init checkpoints/solver_ctl1 --steps $FT_STEPS --seed 2
# C round 3
design solver_cur2 cur2_s5 5
[ -f results/curriculum_r3.npz ] || $PY make_curriculum.py results/curriculum_r3.npz v1_s0 cur1_s3 cur2_s5
train solver_cur3 --init checkpoints/solver_cur2 --extra results/curriculum_r3.npz --extra-prob 0.5 --steps $FT_STEPS --seed 3
train solver_ctl3 --init checkpoints/solver_ctl2 --steps $FT_STEPS --seed 3
# D exploitability: same designer seed (8) and budget against every solver
for m in v1 ctl1 cur1 ctl2 cur2 ctl3 cur3 ff; do
  design solver_$m expl_${m}_s8 8
done
# E
design solver_v1 v1_md12_s7 7 --min-dist 12

stage "evaluate (round 2+)"
$PY evaluate.py --out eval_r2 \
  --solvers solver_v1 solver_ctl1 solver_cur1 solver_ctl1_r11 solver_cur1_r11 solver_ctl1_r12 solver_cur1_r12 \
            solver_ctl2 solver_cur2 solver_ctl3 solver_cur3 solver_ff \
  --designer-runs v1_s1 ctl1_s2 cur1_s2 v1_md12_s7 expl_ctl3_s8 expl_cur3_s8 --n-arch 372 $EVAL_ARGS
stage "pipeline2 done"
