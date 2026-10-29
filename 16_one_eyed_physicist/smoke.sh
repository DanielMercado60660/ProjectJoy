#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min): physics sanity, baseline, 15 s training, eval, demo.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
OUT=$(mktemp -d)
$PY - <<'PYEOF'
import numpy as np, scene, baseline
# physics: apex height above contact after bounce k scales by e^2 per bounce
p = scene.sample_params(np.random.default_rng(0), 4)
tb, ys, vs = scene.bounce_table(p["y0"], p["vy0"], p["e"])
assert np.allclose(vs[:, 2] / vs[:, 1], p["e"]), "velocity ratio != e"
t = np.linspace(0, 3, 3001)[None].repeat(4, 0); y = scene.heights(p, t)
assert (y >= scene.R - 1e-9).all()
# baseline on clean-ish clips recovers e
X, q = scene.make_set(5, 16)
e, sd, ok = baseline.estimate_many(X)
mae = np.abs(e - q["e"]).mean(); print("baseline MAE on 16 clips", mae); assert mae < 0.05
PYEOF
$PY train.py --minutes 0.25 --warmup_mse 50 --val_n 64 --out "$OUT/ck"
$PY evaluate.py --ckpt "$OUT/ck" --n_main 64 --n_hard 32 --out "$OUT/r.json" --media "$OUT"
$PY demo.py --ckpt "$OUT/ck" --out "$OUT/demo.gif"
ls "$OUT"; rm -rf "$OUT"
echo SMOKE OK
