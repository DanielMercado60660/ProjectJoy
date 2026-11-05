"""R8 Gate C, POST HOC (not pre-registered): the bake-off's probes, fit exactly as in the protocol (fit frames of
configs 0000-0269), scored on DEV-family rooms instead (the 80 p2_dev rooms re-recorded at 224 px: other floor /
wall hues, light tilt, wall_EW partition). Same frame-sampling rules as rep_probeset.py.

    ./r8_job.sh rep-probe-dev ../.venv/bin/python r8_rep/rep_probes_dev.py   -> results/r8/probes/dev_family.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import rep_data as D  # noqa: E402
import rep_probes as R  # noqa: E402
from rep_features import ARMS_224, ARMS_64, featurize64, featurize224  # noqa: E402
from rep_probeset import CAP, LAMP_PX, PLATE_PX  # noqa: E402


def dev_items():
    eps = D.episode_list("dev")
    rng = {"lamp": np.random.default_rng(0), "plate": np.random.default_rng(1), "bearing": np.random.default_rng(2)}
    items, F224, F96 = [], [], []
    for r in eps:
        m = np.load(ROOT / r["path"] / "meta.npz")
        fr = np.load(ROOT / r["path"] / "frames224.npy", mmap_mode="r")
        lpx, ppx, fpx = m["lamp_px"], m["plate_px"], m["pframe_px"]
        sel = []
        lt = np.nonzero(lpx >= LAMP_PX)[0]
        if len(lt) > CAP["lamp"]:
            lt = np.sort(rng["lamp"].choice(lt, CAP["lamp"], replace=False))
        sel += [(t, 0, float(m["lamp_on"][t]), int(lpx[t])) for t in lt]
        pos = np.nonzero(ppx >= PLATE_PX)[0]
        neg = np.nonzero((ppx == 0) & (fpx == 0))[0]
        for lst, lab in ((pos, 1.0), (neg, 0.0)):
            if len(lst) > CAP["plate"]:
                lst = np.sort(rng["plate"].choice(lst, CAP["plate"], replace=False))
            sel += [(t, 1, lab, int(ppx[t])) for t in lst]
        bt = pos
        if len(bt) > CAP["bearing"]:
            bt = np.sort(rng["bearing"].choice(bt, CAP["bearing"], replace=False))
        sel += [(t, 2, float(m["plate_bearing"][t]), int(ppx[t])) for t in bt]
        for t, task, y, px in sel:
            items.append((task, y, px))
            F224.append(np.asarray(fr[t]))
            F96.append(m["frames96"][t])
    it = np.array(items, np.float64)
    return {"task": it[:, 0].astype(np.int8), "y": it[:, 1].astype(np.float32), "px": it[:, 2].astype(np.int32)}, \
        np.stack(F224), np.stack(F96), len(eps)


def main():
    dv, f224, f96, n_eps = dev_items()
    main_it = R.load_items()
    fit = main_it["cfg"] < 270
    out = {"note": "POST HOC, not pre-registered: probes fit on the protocol's fit frames, scored on dev-family rooms",
           "dev_episodes": n_eps, "counts": {n: int((dv["task"] == t).sum()) for t, n in R.TASKS}}
    arms = list(ARMS_224) + list(ARMS_64)
    for arm in arms:
        zf = np.load(R.FEAT / f"{arm}.npz")
        zd = featurize64(arm, f96) if arm in ARMS_64 else featurize224(arm, f224)
        rec = {}
        for t, name in R.TASKS:
            sf = (main_it["task"] == t) & fit
            sd = dv["task"] == t
            y = np.concatenate([main_it["y"][sf], dv["y"][sd]])
            if t < 2:
                y = (y > 0.5).astype(np.float32)
            tr = np.concatenate([np.ones(sf.sum(), bool), np.zeros(sd.sum(), bool)])
            sub = np.concatenate([main_it["px"][sf], dv["px"][sd]]) >= 64 if t == 0 else None
            rec[name] = {"linear": R.linear_probe(np.concatenate([zf["glob"][main_it["frame"][sf]], zd["glob"][sd]]), y, tr,
                                                  regress=(t == 2), sub=sub)}
            if "proj" in zf.files:
                rec[name]["linear_proj"] = R.linear_probe(np.concatenate([zf["proj"][main_it["frame"][sf]], zd["proj"][sd]]),
                                                          y, tr, regress=(t == 2), sub=sub)
            if "grid" in zf.files:
                G = np.concatenate([zf["grid"][main_it["frame"][sf]], zd["grid"][sd]])
                rec[name]["attentive"] = R.attentive_probe(G, y, tr, regress=(t == 2), sub=sub)
        out[arm] = rec
        print(arm, json.dumps({k: {kk: vv.get("test_acc", vv.get("test_auc", vv.get("test_mae_deg"))) for kk, vv in v.items()}
                               for k, v in rec.items()}), flush=True)
    (R.RES / "dev_family.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
