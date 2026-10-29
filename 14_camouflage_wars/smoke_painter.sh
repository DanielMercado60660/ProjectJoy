#!/bin/bash
# Smoke check for the painter follow-up (~20 s): tiny painter run + analysis + demo + 3-step fine-tune,
# all into a temp dir. Needs the shipped checkpoints/unet*, checkpoints/unet_seed1* and data/train_meta.pkl.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
$PY painter.py --n_per_delta 2 --k 4 --workers 2 --out "$T/p" > "$T/p.log"
grep -q "painted by unet0" "$T/p/summary.md"
$PY painter_analysis.py "$T/p" > /dev/null
mkdir -p "$T/media"; $PY painter_demo.py --res "$T/p" --out "$T/media" > /dev/null 2>&1
test -s "$T/media/painter_best.png"
$PY painter.py --families train --no_baselines --n_per_delta 2 --k 4 --workers 2 --seed_base 8000000 --out "$T/pt" > /dev/null
$PY painter_finetune.py --painted "$T/pt" --steps 3 --out "$T/ck/unet_ft" > /dev/null
$PY -c "import model as M; M.load('$T/ck/unet_ft'); print('reload ok')"
rm -rf "$T"; echo "PAINTER SMOKE OK"
