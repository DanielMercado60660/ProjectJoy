#!/bin/bash
# End-to-end smoke check at tiny size (< 2 min): data -> classifier -> 4 generators (+resume) -> metrics -> sampling.
# Uses a temp dir for everything; does not touch data/, checkpoints/, runs/ or results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
T=$(mktemp -d)
export MONSTER_DATA=$T/m.npz MONSTER_CKPT_DIR=$T/ck
echo "[smoke] tmp=$T"
$PY data.py --n_train 300 --n_test 200 --out $T/m.npz
$PY train_classifier.py --epochs 1 --min_epochs 0 --ckpt_dir $T/ck --results_dir $T/res | tail -n 3
for m in vae gan ddpm flow; do
  $PY train_gen.py --model $m --run $T/r_$m --budget_pflop 0.004 --budget_min 0.1 --n_snaps 2 --n_ckpts 2 --batch 32 --max_steps 6 | tail -n 1
  $PY train_gen.py --model $m --run $T/r_$m --resume --max_steps 0 | tail -n 1   # resume to completion
done
$PY - <<EOF
import numpy as np, mlx.core as mx
import evaluate as E, train_gen as TG
for m in ["vae", "gan", "ddpm", "flow"]:
    model, g, cfg = TG.load_generator("$T/r_" + m)
    x = TG.to_uint8(TG.sample(model, g, 64, cfg, mx.random.key(0), steps=4))
    assert x.shape == (64, 32, 32, 3)
    r = E.strip(E.metrics(x))
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "attr_tv_per"})
print("SMOKE OK")
EOF
rm -rf "$T"
