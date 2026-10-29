#!/bin/bash
# One seed, one gpu_run slot, one BLAS thread. Invoke through ../bin/gpu_run.
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in 0|1|2) r5_seed=$1 ;; *) echo 'Usage: ../bin/gpu_run ./run_r5_t2.sh {0|1|2}'; exit 2 ;; esac
if ! pmset -g batt | grep -q 'AC Power'; then
  echo 'R5_T2_REQUIRES_AC: this enlarged parallel experiment is AC-only.'
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=../.venv/bin/python
r5_tag=t2_s$r5_seed
r5_stamp=$(date '+%Y%m%d_%H%M%S')
mkdir -p logs/r5
trap 'r5_exit_code=$?; printf "%s R5 turn 2 %s invocation %s exit=%s; logs/r5/%s_*_%s.log\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "$r5_tag" "$r5_stamp" "$r5_exit_code" "$r5_tag" "$r5_stamp" >> STATUS.md' EXIT
printf '%s R5 turn 2 %s running enlarged training; fresh test bank reserved until validation qualification.\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$r5_tag" >> STATUS.md
"$PY" r5.py train --profile t2 --tag "$r5_tag" --seed "$r5_seed" --max-minutes 40 2>&1 | tee "logs/r5/${r5_tag}_train_${r5_stamp}.log"
"$PY" r5.py cpg --profile t2 --tag "$r5_tag" --seed "$r5_seed" --reuse-cpg "main_s$r5_seed" --max-minutes 0.25 2>&1 | tee "logs/r5/${r5_tag}_cpg_${r5_stamp}.log"
"$PY" r5.py evaluate --profile t2 --tag "$r5_tag" --seed "$r5_seed" --max-minutes 2 2>&1 | tee "logs/r5/${r5_tag}_eval_${r5_stamp}.log"
"$PY" r5.py replay-demo --profile t2 --tag "$r5_tag" --seed "$r5_seed" --max-minutes 0.5 2>&1 | tee "logs/r5/${r5_tag}_reload_${r5_stamp}.log"
echo "R5_T2_JOB_COMPLETE $r5_tag"
