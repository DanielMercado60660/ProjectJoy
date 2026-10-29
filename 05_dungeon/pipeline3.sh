#!/bin/bash
# Extra training seeds for curriculum rounds 2-3 (the review asked for more than one seed per arm).
# Repeats pipeline2.sh stage C for one seed chain S, starting from that seed's round-1 fine-tunes
# (solver_cur1_rS / solver_ctl1_rS; trained here if missing, exactly like pipeline2.sh stage B).
#   ../bin/gpu_run bash pipeline3.sh 11 > logs/pipeline3_r11.out 2>&1
# Same budget (6M requested = 5,996,544 env steps per fine-tune), hyperparameters and level mix as
# pipeline2.sh. The designer attacks *this chain's* curriculum solver (designer seeds 3 and 5, as
# in the seed-0 chain), so each chain grows its own archive. Fine-tune seeds: round 1 = S,
# round k>1 = 10*S + k (both arms share the seed within a round). Resumable; writes only new paths.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${PY:-../.venv/bin/python}
S=${1:?usage: pipeline3.sh SEED}
T0=$(date +%s)
elapsed() { echo $(( ($(date +%s) - T0) / 60 )); }
BUDGET=${BUDGET:-40}
stage() { if [ "$(elapsed)" -ge "$BUDGET" ]; then echo "=== budget reached; rerun to resume"; exit 0; fi; echo "=== [$(date +%H:%M:%S) +$(elapsed)m] $*"; }
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

# round 1 (only runs if this seed has no round-1 replicate yet)
train solver_cur1_r$S --init checkpoints/solver_v1 --extra results/curriculum_r1.npz --extra-prob 0.5 --steps $FT_STEPS --seed $S
train solver_ctl1_r$S --init checkpoints/solver_v1 --steps $FT_STEPS --seed $S
# round 2
design solver_cur1_r$S cur1_r${S}_s3 3
[ -f results/curriculum_r2_r$S.npz ] || $PY make_curriculum.py results/curriculum_r2_r$S.npz v1_s0 cur1_r${S}_s3
train solver_cur2_r$S --init checkpoints/solver_cur1_r$S --extra results/curriculum_r2_r$S.npz --extra-prob 0.5 --steps $FT_STEPS --seed $((10*S+2))
train solver_ctl2_r$S --init checkpoints/solver_ctl1_r$S --steps $FT_STEPS --seed $((10*S+2))
# round 3
design solver_cur2_r$S cur2_r${S}_s5 5
[ -f results/curriculum_r3_r$S.npz ] || $PY make_curriculum.py results/curriculum_r3_r$S.npz v1_s0 cur1_r${S}_s3 cur2_r${S}_s5
train solver_cur3_r$S --init checkpoints/solver_cur2_r$S --extra results/curriculum_r3_r$S.npz --extra-prob 0.5 --steps $FT_STEPS --seed $((10*S+3))
train solver_ctl3_r$S --init checkpoints/solver_ctl2_r$S --steps $FT_STEPS --seed $((10*S+3))
stage "pipeline3 seed $S done"
