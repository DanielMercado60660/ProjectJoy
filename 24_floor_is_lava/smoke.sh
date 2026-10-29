#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min): env + CPG, baseline tuner, PPO train/save/resume,
# held-out evaluation, demo rendering. Writes only into logs/smoke_* and checkpoints/smoke_*.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=1
mkdir -p logs/smoke_media
rm -f logs/smoke_train.csv logs/smoke_level.csv checkpoints/smoke_latest.npz
$PY -m pytest -q tests 2>&1 | tail -2
$PY tune_cpg.py --tiny --out logs/smoke_cpg.json | grep final
$PY tune_cpg.py --tiny --planner --out logs/smoke_planner.json | grep final
$PY ppo.py --tiny --run smoke --cpg logs/smoke_cpg.json | tail -1
$PY ppo.py --tiny --run smoke --cpg logs/smoke_cpg.json --max_iters 5 --resume checkpoints/smoke_latest.npz | grep -E "resumed|done"
$PY evaluate.py --policy checkpoints/smoke_latest.npz --cpg logs/smoke_cpg.json --planner logs/smoke_planner.json \
    --n 4 --workers 2 --out logs/smoke_eval.json | grep -c paired
$PY render_demo.py --policy checkpoints/smoke_latest.npz --cpg logs/smoke_cpg.json --planner logs/smoke_planner.json \
    --index 0 --out logs/smoke_media --tag smoke --width 240 --height 160 2>&1 | grep "held-out"
echo "SMOKE OK"
