#!/bin/bash
# End-to-end smoke test at tiny size (< 2 min): train -> merge -> adapt -> report -> CPG -> render.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=1
for st in train merge adapt; do
  $PY pipeline.py --stage $st --tiny --seeds 0 --workers 1 --root $T/ck --results $T/res 2>&1 | grep -v -i warn | tail -2
done
$PY cpg.py --iters 2 --pop 4 --out $T/res/cpg.json 2>&1 | grep -v -i warn
REPORT_RES=$T/res REPORT_MEDIA=$T/media $PY report.py > $T/report.txt 2>&1
ls $T/media/interpolation.png $T/media/adaptation.png > /dev/null
$PY render_demo.py --seed 0 --cond pre --steps 40 --every 4 --ckroot $T/ck --results $T/res --out $T/media/demo 2>&1 | grep -v -i warn | tail -2
ls $T/media/demo.mp4 $T/ck/demo/s0_pre_best_merge.npz > /dev/null
$PY -m pytest -q tests 2>&1 | tail -2
rm -rf "$T"
echo "SMOKE OK"
