#!/bin/bash
# End-to-end smoke check at tiny size (< 2 minutes). Run from this folder.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
S=$(mktemp -d)
echo "== unit tests"; $PY -m pytest -q tests
echo "== active vs random (6 worlds x 4 pushes)"; $PY run_active.py --worlds 6 --steps 4 --procs 2 --out $S/active.npz
echo "== world 2 (2 worlds x 2 pushes)"; $PY -c "
import sys; sys.argv=['x']; import deeper; deeper.run_world2(2, 2, 0, '$S/w2.npz', procs=2)"
echo "== world 3 (ridge)"; $PY -c "import deeper; deeper.run_world3('$S/w3.npy')" > $S/w3.log; head -3 $S/w3.log
echo "== SBI: 200 training steps + eval"; $PY sbi.py --steps 200 --ckpt $S/sbi > /dev/null
$PY eval_sbi.py --ckpt $S/sbi --active $S/active.npz --n_fresh 100 --out $S/sbi_eval.json > $S/eval.log; head -2 $S/eval.log
echo "SMOKE OK ($S)"
