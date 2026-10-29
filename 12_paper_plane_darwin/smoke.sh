#!/usr/bin/env bash
# End-to-end smoke check at tiny size (~30-60 s): physics sanity tests, tiny search, tiny audit.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4
TMP=$(mktemp -d)
$PY -m pytest -q tests
$PY search.py --budget 160 --seeds 1 --me-evals 1200 --out "$TMP/search.npz"
$PY evaluate.py --ckpt "$TMP/search.npz" --out "$TMP/eval.json" --designs "$TMP/designs.json" --n-heldout 8
echo "smoke OK ($TMP)"
