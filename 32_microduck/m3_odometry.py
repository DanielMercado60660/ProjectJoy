"""M3 stage 1: odometry candidates measured against truth (truth used only to score, never by the bot).

40 training configs x 60 s of M0-style mixed commands (stand, forward, backward, turn, forward+turn,
head moves) through the frozen interface. Observations are recorded once, then every candidate is run
offline on the same stream. Error = distance between the estimated and true position, both in the
duck's start frame. Writes results/m3/odometry.json.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interface import Command, Interface  # noqa: E402
from m0_gate import BOUNDS, make_sequence  # noqa: E402
from odometry import Odometry  # noqa: E402
from room import World  # noqa: E402

OUT = HERE / "results" / "m3"
CANDIDATES = {
    "cmd": dict(mode="cmd"),
    "leg_lower": dict(mode="leg", stance_rule="lower"),
    "leg_switch_mean": dict(mode="leg", stance_rule="switch_mean"),
    **{f"leg_switch_mean_x{sc}": dict(mode="leg", stance_rule="switch_mean", leg_scale=sc)
       for sc in (1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5)},
}
TUNE, CHECK = range(0, 20), range(20, 40)   # scale chosen on configs 0-19, checked on 20-39


def record(args):
    cfg, i = args
    rng = np.random.default_rng(5000 + i)
    seq = make_sequence(rng, BOUNDS["chosen"], total_s=60.0)
    w = World(cfg)
    it = Interface(w)
    x0, y0, _ = w.duck.pos()
    yaw0 = w.duck.yaw()
    c0, s0 = np.cos(-yaw0), np.sin(-yaw0)
    obs_list, truth = [], []
    for seg in seq:
        for _ in range(int(round(seg["dur"] / 0.1))):
            obs = it.tick(Command(vx=seg["cmd"][0], wz=seg["cmd"][2], head=tuple(seg["head"])))
            p = w.duck.pos()
            dx, dy = p[0] - x0, p[1] - y0
            truth.append([c0 * dx - s0 * dy, s0 * dx + c0 * dy, (w.duck.yaw() - yaw0 + np.pi) % (2 * np.pi) - np.pi])
            obs_list.append(obs)
    out = {"config": cfg["id"], "ticks": len(obs_list)}
    T = np.array(truth)
    for name, kw in CANDIDATES.items():
        od = Odometry(**kw)
        est = np.array([od.update(o) for o in obs_list])
        err = np.hypot(*(est[:, :2] - T[:, :2]).T)
        yerr = np.abs((est[:, 2] - T[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        path = float(np.sum(np.hypot(*np.diff(T[:, :2], axis=0).T)))
        out[name] = {"err_30s": float(err[299]), "err_60s": float(err[-1]), "err_max": float(err.max()),
                     "yaw_err_60s": float(yerr[-1]), "path_m": path,
                     "err_per_m_path": float(err[-1] / max(path, 1e-6))}
    return out


def main():
    t0 = time.time()
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][:40]
    with Pool(4) as pool:
        rows = pool.map(record, [(c, i) for i, c in enumerate(cfgs)])
    summ = {}
    for part, idx in (("all", range(40)), ("tune_0_19", TUNE), ("check_20_39", CHECK)):
        summ[part] = {}
        for name in CANDIDATES:
            v = {k: np.array([rows[i][name][k] for i in idx]) for k in ("err_30s", "err_60s", "yaw_err_60s", "err_per_m_path")}
            summ[part][name] = {k: {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                                    "max": float(a.max())} for k, a in v.items()}
    leg = [n for n in CANDIDATES if n.startswith("leg_switch_mean")]
    chosen = min(leg, key=lambda n: summ["tune_0_19"][n]["err_60s"]["median"])
    rep = {"summary": summ, "chosen_on_tune_split": chosen,
           "path_m_median": float(np.median([r["leg_lower"]["path_m"] for r in rows])),
           "rows": rows, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "odometry.json").write_text(json.dumps(rep, indent=1))
    for part in ("tune_0_19", "check_20_39"):
        for n in CANDIDATES:
            v = summ[part][n]
            print(f"{part:12s} {n:24s} err30 {v['err_30s']['median']:.3f}  err60 {v['err_60s']['median']:.3f} (p90 {v['err_60s']['p90']:.3f})  per-m {v['err_per_m_path']['median']:.3f}")
    print("chosen:", chosen, "wall", rep["wall_s"])


if __name__ == "__main__":
    main()
