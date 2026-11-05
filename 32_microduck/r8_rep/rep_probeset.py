"""R8 Gate C: build the probe set (results/r8/probes/protocol.json) from episodes/r8_rep.

    ../.venv/bin/python r8_rep/rep_probeset.py        -> data/r8_rep/probeset/{frames224.npy, frames96.npy, items.npz}
                                                         + results/r8/probes/probeset.json (counts, episode list)
Items: one row per (task, frame): task 0 = lamp (label lamp on), 1 = plate in view (label in view),
2 = plate bearing (target rad). A frame used by several tasks is stored once. Labels are simulator truth
(labels only). Fit = configs 0000-0269 (set_lamp k 8, 9); test = configs 0270-0299 (set_lamp k 8-11).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EP = ROOT / "episodes" / "r8_rep"
OUT = ROOT / "data" / "r8_rep" / "probeset"
LAMP_PX, PLATE_PX = 16, 16
CAP = {"lamp": 12, "plate": 6, "bearing": 8}


def rows(fit_k=(8, 9), test_k=(8, 9, 10, 11)):
    out = []
    sys.path.insert(0, str(HERE))
    from rep_data import read_index
    for r in read_index():
        if "error" in r or r["task"] != "set_lamp" or r.get("split", "train") != "train":
            continue
        if (r["cfg"] < 270 and r["k"] in fit_k) or (r["cfg"] >= 270 and r["k"] in test_k):
            out.append(r)
    return sorted(out, key=lambda r: r["name"])


def main():
    eps = rows()
    rng = {"lamp": np.random.default_rng(0), "plate": np.random.default_rng(1), "bearing": np.random.default_rng(2)}
    items = []                                    # (episode index, tick, task, label/target)
    for ei, r in enumerate(eps):
        m = np.load(ROOT / r["path"] / "meta.npz")
        lpx, ppx, fpx = m["lamp_px"], m["plate_px"], m["pframe_px"]
        lt = np.nonzero(lpx >= LAMP_PX)[0]
        if len(lt) > CAP["lamp"]:
            lt = np.sort(rng["lamp"].choice(lt, CAP["lamp"], replace=False))
        items += [(ei, int(t), 0, float(m["lamp_on"][t]), int(lpx[t])) for t in lt]
        pos = np.nonzero(ppx >= PLATE_PX)[0]
        neg = np.nonzero((ppx == 0) & (fpx == 0))[0]
        for lst, lab in ((pos, 1.0), (neg, 0.0)):
            if len(lst) > CAP["plate"]:
                lst = np.sort(rng["plate"].choice(lst, CAP["plate"], replace=False))
            items += [(ei, int(t), 1, lab, int(ppx[t])) for t in lst]
        bt = pos
        if len(bt) > CAP["bearing"]:
            bt = np.sort(rng["bearing"].choice(bt, CAP["bearing"], replace=False))
        items += [(ei, int(t), 2, float(m["plate_bearing"][t]), int(ppx[t])) for t in bt]
    frames = sorted({(it_[0], it_[1]) for it_ in items})
    fidx = {ft: i for i, ft in enumerate(frames)}
    OUT.mkdir(parents=True, exist_ok=True)
    F224 = np.lib.format.open_memmap(OUT / "frames224.npy", "w+", np.uint8, (len(frames), 224, 224, 3))
    F96 = np.empty((len(frames), 96, 96, 3), np.uint8)
    by_ep = {}
    for i, (e, t) in enumerate(frames):
        by_ep.setdefault(e, []).append((i, t))
    for e, lst in by_ep.items():
        fr = np.load(ROOT / eps[e]["path"] / "frames224.npy", mmap_mode="r")
        f96 = np.load(ROOT / eps[e]["path"] / "meta.npz")["frames96"]
        for i, t in lst:
            F224[i] = fr[t]
            F96[i] = f96[t]
    F224.flush()
    np.save(OUT / "frames96.npy", F96)
    it = np.array(items, np.float64)
    cfg = np.array([eps[int(e)]["cfg"] for e in it[:, 0]], np.int32)
    np.savez(OUT / "items.npz", frame=np.array([fidx[(int(x[0]), int(x[1]))] for x in items], np.int64),
             px=it[:, 4].astype(np.int32),
             task=it[:, 2].astype(np.int8), y=it[:, 3].astype(np.float32), cfg=cfg,
             ep=it[:, 0].astype(np.int32), tick=it[:, 1].astype(np.int32))
    summ = {"episodes": len(eps), "fit_episodes": sum(r["cfg"] < 270 for r in eps),
            "test_episodes": sum(r["cfg"] >= 270 for r in eps), "frames": len(frames)}
    for task, name in ((0, "lamp"), (1, "plate"), (2, "bearing")):
        sel = it[:, 2] == task
        fit = sel & (cfg < 270)
        tst = sel & (cfg >= 270)
        summ[name] = {"fit": int(fit.sum()), "test": int(tst.sum()),
                      "mean_y_fit": round(float(it[fit, 3].mean()), 4), "mean_y_test": round(float(it[tst, 3].mean()), 4)}
    summ["episode_names"] = [r["name"] for r in eps]
    res = ROOT / "results" / "r8" / "probes"
    res.mkdir(parents=True, exist_ok=True)
    (res / "probeset.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "episode_names"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
