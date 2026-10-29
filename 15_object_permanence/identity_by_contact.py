"""Identity switches split by whether two balls ever touch (centre distance < 2R) in the sequence.
Reproduces results/identity_by_contact.json (the 51% vs 5% "swapped at end" numbers in RESULTS.md).
Added by the cross-reviewer: the JSON existed but the script that produced it was not in the folder.

  ../.venv/bin/python identity_by_contact.py
"""
import json

import numpy as np

import kalman
import metrics
import sim
from evaluate import run_model
from train import load_model

kc = json.load(open("results/kalman_tuning_hard.json"))["best"]["bounce=True"]
model = load_model("checkpoints/hard/final")[0]
out = {}
for split in ["abl_ident", "abl_ident_coll"]:
    d = sim.load_split(split)
    idx = np.arange(1000)
    frames = sim.render_split(d, idx)
    preds = {"gru": run_model(model, frames)[0],
             "kalman_bounce": kalman.track_split(d, idx, frames=frames, q=kc["q"], r=kc["r"], bounce=True)[0]}
    pos, pres = d["pos"][idx].astype(np.float64), d["present"][idx]
    touch = np.zeros(len(idx), bool)
    for i in range(len(idx)):
        p = pos[i][:, pres[i]]
        dd = np.linalg.norm(p[:, :, None] - p[:, None, :], axis=-1) + np.eye(p.shape[1]) * 99
        touch[i] = (dd < 2 * sim.R).any()
    for name, mean in preds.items():
        for t in [True, False]:
            s = touch == t
            r = dict(n_seq=int(s.sum()), **metrics.id_switches(mean[s], pos[s], pres[s]))
            out[f"{split}|{name}|touching={t}"] = r
            print(split, name, f"touching={t}", r, flush=True)
with open("results/identity_by_contact.json", "w") as f:
    json.dump(out, f, indent=1)
