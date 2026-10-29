#!/bin/bash
# CPU only, no downloads/MLX. Preserves each smoke's artifacts in fresh R5 paths.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export NUMBA_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
PY=../.venv/bin/python
mkdir -p results/r5 logs/r5 checkpoints media
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$(mktemp -d "results/r5/smoke_${STAMP}_XXXX")
TAG=$(basename "$WORK")
CKPT="checkpoints/r5_${TAG}"
mkdir "$CKPT"
main() {
echo "R5 smoke artifacts: $WORK; checkpoints: $CKPT"
"$PY" -m pytest -q tests/test_r5.py tests/test_r5_turn2.py tests/test_pipeline.py -k 'not sbi' --basetemp "$WORK/pytest"
"$PY" run_r5.py tune --worlds 2 --seed 51001 --grid-size 801 \
  --checkpoint "$CKPT/tune.json" --out "$WORK/tuning.json" --max-minutes 0.4
# Boundary pause + resume is exercised here, in addition to pytest's real SIGKILL test.
"$PY" run_r5.py evaluate --worlds 2 --seed 51002 --grid-size 801 \
  --particles 16 --fantasies 8 --design-stride 5 --tuning "$WORK/tuning.json" \
  --checkpoint "$CKPT/eval.json" --out "$WORK/summary.json" \
  --stop-after-worlds 1 --max-minutes 0.5
"$PY" run_r5.py evaluate --worlds 2 --seed 51002 --grid-size 801 \
  --particles 16 --fantasies 8 --design-stride 5 --tuning "$WORK/tuning.json" \
  --checkpoint "$CKPT/eval.json" --out "$WORK/summary.json" --resume --max-minutes 0.5
"$PY" r5_report.py --checkpoint "$CKPT/eval.json" \
  --summary "$WORK/report_summary.json" --out "media/r5_${TAG}.html" --max-minutes 0.1
"$PY" run_r5_turn2.py tune --worlds 1 --seed 73001 --baseline-tuning "$WORK/tuning.json" \
  --mu-size 301 --gain-size 21 --base-grid-size 401 --base-particles 8 --base-fantasies 4 --base-stride 25 \
  --checkpoint "$CKPT/turn2_tune.json" --out "$WORK/turn2_tuning.json" --max-minutes 0.2
"$PY" run_r5_turn2.py evaluate --worlds 2 --seed 73002 --baseline-tuning "$WORK/tuning.json" \
  --tuning "$WORK/turn2_tuning.json" \
  --mu-size 301 --gain-size 21 --base-grid-size 401 --base-particles 8 --base-fantasies 4 --base-stride 25 \
  --checkpoint "$CKPT/turn2_eval.json" --out "$WORK/turn2_summary.json" --max-minutes 0.2
"$PY" r5_report_turn2.py --checkpoint "$CKPT/turn2_eval.json" \
  --summary "$WORK/turn2_report_summary.json" --out "media/r5_turn2_${TAG}.html" --max-minutes 0.1
echo "SMOKE_R5_OK: $WORK ; media/r5_${TAG}.html"
}
main 2>&1 | tee "logs/r5/${TAG}.log"
