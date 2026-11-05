"""R8 Gate C, post-hoc diagnostic (NOT pre-registered): the pre-registered 64 px lamp CNN (seed 0) did not fit
(train accuracy 0.517 = the majority class). Same recipe, data and steps with other init seeds, to tell an
optimisation failure from missing signal.   ../.venv/bin/python r8_rep/rep_cnn_seeds.py 1 2
-> appends to results/r8/probes/cnn_seed_diag.jsonl"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rep_probes as R  # noqa: E402

it = R.load_items()
f96 = np.load(R.PS / "frames96.npy")
s = it["task"] == 0
X64 = np.stack([np.asarray(Image.fromarray(x).resize((64, 64), Image.Resampling.BOX)) for x in f96[it["frame"][s]]])
y, tr = it["y"][s], it["cfg"][s] < 270
for seed in map(int, sys.argv[1:]):
    orig = mx.random.seed
    mx.random.seed = lambda _s, _o=orig, _k=seed: _o(_k)        # cnn_probe seeds MLX with 0: swap in this seed
    try:
        rec = {"seed": seed, "lamp_cnn64": R.cnn_probe(X64, y, tr, sub=it["px"][s] >= 64)}
    finally:
        mx.random.seed = orig
    print(json.dumps(rec), flush=True)
    with open(R.RES / "cnn_seed_diag.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
