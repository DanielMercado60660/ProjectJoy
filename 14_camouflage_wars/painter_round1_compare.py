"""Retraining round comparison. All runs score the SAME 400 held-out scenes x 64 candidate skins
(deterministic seeds), so per-candidate scores line up. Writes results/painter_round1/compare.json.
  orig      : frozen seed-0 U-Net (results/painter)
  ft        : fine-tuned on painted training-family creatures (results/painter_round1)
  ft_ctrl   : identical fine-tune on the same scenes with random skins (results/painter_round1_control)
For each model: IoU on random skins, on the original painter's picks (painted vs orig), and on a
fresh painter's picks against itself (the re-attack). unet1 on the same picks is the 'any other
model' reference for the stale-picks column."""
import json, os
import numpy as np

runs = {"orig": "results/painter", "ft": "results/painter_round1", "ft_ctrl": "results/painter_round1_control"}
S = {k: np.load(f"{v}/scores.npz") for k, v in runs.items() if os.path.exists(f"{v}/scores.npz")}
o = S["orig"]; r = np.arange(len(o["delta"]))
old_pick = o["unet0_soft"].argmin(1)
out = {}
for name, s in S.items():
    assert np.allclose(s["unet1"], o["unet1"]), "candidates differ"
    new_pick = s["unet0_soft"].argmin(1)
    for bn, sel in [("all", np.ones(len(r), bool))] + [(f"delta={d}", o["delta"] == d) for d in (0.25, 0.5, 0.75, 1.0)]:
        out[f"{name}/{bn}"] = dict(
            random=float(s["unet0"][sel].mean()),
            on_orig_painter_picks=float(s["unet0"][r, old_pick][sel].mean()),
            reattack=float(s["unet0"][r, new_pick][sel].mean()),
            reattack_det=float((s["unet0"][r, new_pick][sel] >= 0.5).mean()),
            unet1_on_reattack_picks=float(o["unet1"][r, new_pick][sel].mean()),
            base_er_on_reattack_picks=float(o["base_er"][r, new_pick][sel].mean()))
for name in S:
    lg = {"orig": None, "ft": "checkpoints/painter_round1/unet_ft_log.json",
          "ft_ctrl": "checkpoints/painter_round1/unet_ft_control_log.json"}[name]
    if lg and os.path.exists(lg):
        d = json.load(open(lg)); out[f"{name}/val_iou_seen"] = [d["val_iou_before"], d["val_iou_after"]]
json.dump(out, open("results/painter_round1/compare.json", "w"), indent=1)
for k, v in out.items():
    print(k, {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else [round(x, 4) for x in v])
