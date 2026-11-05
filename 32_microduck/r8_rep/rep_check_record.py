"""R8 Gate C: check the 224 px recorder against a stored M4 episode (record.py conventions).

    ../.venv/bin/python r8_rep/rep_check_record.py [--config 1] [--ticks 60]   -> results/r8/probes/record_check.json

1. Re-records the stored M4 set_lamp episode train-<config> k=1 (clean bot, no perturbation) with
   rep_record.record_rep_episode (extra 224 px + segmentation renderers) and checks that its stored-style
   96 px frames and executed commands equal the stored episode's, tick for tick (the extra renderers must
   not change what the bot sees or does).
2. Compares the 224 px frames, box-downsampled to 96 px, with the stored 96 px frames (which come from the
   128 px render): mean / p99 / max absolute grey-level difference over the first --ticks ticks.
3. Sanity of the truth labels: correlation between the plate's bearing label and the column of its
   visible pixels in the upright frame (they must agree in sign: + bearing = left = smaller column).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from rep_record import record_rep_episode  # noqa: E402


def box96(f):
    return np.asarray(Image.fromarray(f).resize((96, 96), Image.Resampling.BOX))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=60)
    a = ap.parse_args()
    cfgs = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]
    cfg = cfgs[a.config]
    stored = np.load(ROOT / f"episodes/set_lamp/train_{cfg['id']}_k1.npz")
    hdr = json.loads(str(stored["header"]))
    assert hdr["perturbation"]["type"] is None
    row = record_rep_episode(cfg, "set_lamp", 1, noisy=False, keep=False)
    T = min(len(stored["frames"]), row["ticks"])
    f96, f224 = row["_frames96"], row["_frames224"]
    same_len = len(stored["frames"]) == row["ticks"]
    exact96 = bool(np.array_equal(f96[:T], stored["frames"][:T]))
    exact_cmd = bool(np.array_equal(row["_cmd_exec"][:T], stored["cmd_exec"][:T]))
    n = min(a.ticks, T)
    d = np.stack([np.abs(box96(f224[t]).astype(np.int16) - stored["frames"][t].astype(np.int16)) for t in range(n)])
    # where are the large differences? edge = the stored frame's local gradient (max over channels of the
    # 3x3 neighbourhood range) in the top 10% of that frame's pixels
    big = d.max(-1) > 16
    edge = np.zeros_like(big)
    for t in range(n):
        g = stored["frames"][t].astype(np.int16)
        pad = np.pad(g, ((1, 1), (1, 1), (0, 0)), mode="edge")
        win = np.stack([pad[i:i + 96, j:j + 96] for i in range(3) for j in range(3)])
        rng_ = (win.max(0) - win.min(0)).max(-1)
        edge[t] = rng_ >= np.percentile(rng_, 90)
    L = row["_labels"]
    px = np.array(L["plate_px"])
    rc = np.array(L["plate_rc"])
    pb = np.array(L["plate_bearing"])
    vis = px >= 16
    corr = float(np.corrcoef(pb[vis], rc[vis, 1])[0, 1]) if vis.sum() > 3 else None
    out = {"episode": f"set_lamp {cfg['id']} k=1 (M4 stored, clean)", "ticks_rerecorded": row["ticks"],
           "ticks_stored": int(len(stored["frames"])), "same_length": same_len,
           "frames96_bit_identical": exact96, "cmd_exec_identical": exact_cmd,
           "f224_to_96_vs_stored96": {"ticks": n, "mean_abs": round(float(d.mean()), 3),
                                      "p99_abs": float(np.percentile(d, 99)), "max_abs": int(d.max()),
                                      "share_px_within_4": round(float((d <= 4).mean()), 4),
                                      "px_with_diff_over_16": int(big.sum()),
                                      "share_of_those_on_edges_top10pct_gradient": round(float(edge[big].mean()), 4) if big.any() else None},
           "plate_visible_ticks": int(vis.sum()), "lamp_visible_ticks": int((np.array(L["lamp_px"]) >= 16).sum()),
           "corr_plate_bearing_vs_column": corr, "wall_s": row["wall_s"],
           "ticks_per_s": round(row["ticks"] / max(row["wall_s"], 1e-6), 1)}
    print(json.dumps(out, indent=1))
    res = ROOT / "results" / "r8" / "probes"
    res.mkdir(parents=True, exist_ok=True)
    (res / "record_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
