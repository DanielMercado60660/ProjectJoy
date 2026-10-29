"""M3: the lamp detectors on purpose-made views (truth only scores). In train/appearances-dev/
distractor-dev configs the duck stands at a free spot 0.8-2.2 m from the lamp, facing it within
+-15 deg, lamp off and then on, 6 ticks each. Records whether lamp_off / lamp_on is detected within
8 deg of the true bearing, the off-detector's distance estimate error, and wrong-state detections.
Writes results/m3/lamp_<tag>.json."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bodymodel import BodyModel  # noqa: E402
from configs import _point_free, footprints  # noqa: E402
from interface import Command, Interface  # noqa: E402
from perception import CameraPose, detect  # noqa: E402
from room import World  # noqa: E402


def run(args):
    cfg, seed = args
    rng = np.random.default_rng(seed)
    w = World(cfg)
    L = w.model.geom_pos[w.lamp_geom][:2].copy()
    X, Y, ob = footprints(cfg["layout"], pad=0.2)
    out = []
    for _ in range(3):
        for _ in range(200):
            ang = rng.uniform(-np.pi, np.pi)
            d = rng.uniform(0.8, 2.2)
            p = L + d * np.array([np.cos(ang), np.sin(ang)])
            if _point_free(X, Y, ob, p):
                break
        else:
            continue
        yaw = float(np.arctan2(L[1] - p[1], L[0] - p[0]) + rng.uniform(-0.26, 0.26))
        for state in (False, True):
            w.reset(start={"pos": p.round(3).tolist(), "yaw": round(yaw, 4)})
            w.lamp_on = state
            w._apply_visuals()
            it = Interface(w)
            for _ in range(6):
                obs = it.tick(Command())
            cam = CameraPose(BodyModel(), obs, (float(w.duck.pos()[0]), float(w.duck.pos()[1]), w.duck.yaw()))
            dets = detect(obs.frame(), cam)
            tb = np.arctan2(L[1] - cam.pos[1], L[0] - cam.pos[0])
            match = lambda c: [x for x in dets if x["cls"] == c and abs((x["bearing"] - tb + np.pi) % (2 * np.pi) - np.pi) < np.radians(8)]  # noqa: E731
            mo, mn = match("lamp_off"), match("lamp_on")
            out.append({"config": cfg["id"], "state_on": state, "dist": float(np.linalg.norm(L - cam.pos[:2])),
                        "off_hit": bool(mo), "on_hit": bool(mn),
                        "off_dist_err": float(mo[0]["dist"] - np.linalg.norm(L - cam.pos[:2])) if mo else None,
                        "other_off": sum(x["cls"] == "lamp_off" for x in dets) - len(mo),
                        "other_on": sum(x["cls"] == "lamp_on" for x in dets) - len(mn)})
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v0"
    S = HERE / "splits"
    cfgs = (json.loads((S / "train.json").read_text())["configs"][:30]
            + json.loads((S / "appearances.json").read_text())["dev"][:20]
            + json.loads((S / "distractor.json").read_text())["dev"][:20])
    with Pool(4) as pool:
        rows = [r for rs in pool.map(run, [(c, 300 + i) for i, c in enumerate(cfgs)]) for r in rs]
    summ = {}
    for fam in ("train", "appearances-dev", "distractor-dev"):
        rs = [r for r in rows if r["config"].startswith(fam.split("-dev")[0]) and (fam == "train" or "dev" in r["config"])]
        off = [r for r in rs if not r["state_on"]]
        on = [r for r in rs if r["state_on"]]
        errs = [abs(r["off_dist_err"]) for r in off if r["off_dist_err"] is not None]
        summ[fam] = {"views": len(off), "off_recall": float(np.mean([r["off_hit"] for r in off])),
                     "on_recall": float(np.mean([r["on_hit"] for r in on])),
                     "on_seen_when_off": int(sum(r["on_hit"] for r in off)),
                     "off_seen_when_on": int(sum(r["off_hit"] for r in on)),
                     "off_dist_err_median": float(np.median(errs)) if errs else None,
                     "spurious_off_elsewhere": int(sum(r["other_off"] for r in rs)),
                     "spurious_on_elsewhere": int(sum(r["other_on"] for r in rs))}
    (HERE / "results" / "m3" / f"lamp_{tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=0))
    print(json.dumps(summ, indent=1))
