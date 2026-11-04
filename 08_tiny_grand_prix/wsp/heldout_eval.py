"""Final scoring on FRESH held-out tracks that no editor ever saw (chosen by the runner after all
Astra turns finished; eval.py's hardcoded seeds are not reused).

    python wsp/heldout_eval.py --task hack --start 90000 --n 200 --hard_n 100 \
        --policy name=path.npz ... [--ray]

For each policy: strict and soft-wall one-lap trials on `heldout` seeds start..start+n-1 and
`hard` seeds start..start+hard_n-1 (grip 1.0 for task=hack; grips 1.0 and 0.5 strict for task=grip),
plus weight-change statistics vs the start policy and distance to checkpoints/es_best.npz.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import ROOT, WSP, WeightsController, dict_to_theta, load_weights  # noqa: E402
from racing.controllers import get_controller  # noqa: E402
from racing.evaluate import run_lap, summarize  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402


def wstats(th, th0, th_ref):
    d = th - th0
    return dict(frac_changed=float((np.abs(d) > 1e-6).mean()), l2_change=float(np.linalg.norm(d)),
                rel_change=float(np.linalg.norm(d) / np.linalg.norm(th0)),
                dist_to_es_best=float(np.linalg.norm(th - th_ref)),
                cos_change_vs_es_best_direction=float(d @ (th_ref - th0) / (np.linalg.norm(d) * np.linalg.norm(th_ref - th0) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hack")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--hard_n", type=int, default=100)
    ap.add_argument("--policy", nargs="+", default=[], help="name=path.npz")
    ap.add_argument("--named", nargs="*", default=[], help="controllers from racing.controllers (ray, es, ...)")
    ap.add_argument("--start_policy", default=os.path.join(ROOT, "checkpoints", "es_soft_best.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    banks = {"heldout": pack(make_tracks("heldout", a.n, start=a.start)),
             "hard": pack(make_tracks("hard", a.hard_n, start=a.start))}
    th0 = dict_to_theta(load_weights(a.start_policy))
    th_ref = dict_to_theta(load_weights(os.path.join(ROOT, "checkpoints", "es_best.npz")))
    ctrls = {}
    for spec in a.policy:
        name, path = spec.split("=", 1)
        d = load_weights(path)
        ctrls[name] = (WeightsController(d), wstats(dict_to_theta(d), th0, th_ref))
    for name in a.named:
        ctrls[name] = (get_controller(name), None)
    runs = [("strict", 1.0), ("soft", 1.0)] if a.task == "hack" else [("strict", 0.5), ("strict", 1.0)]
    out = dict(task=a.task, seeds=dict(heldout=[a.start, a.start + a.n - 1], hard=[a.start, a.start + a.hard_n - 1]),
               results={})
    per_track = {}
    for name, (c, ws) in ctrls.items():
        r = dict(weights=ws)
        for split, bank in banks.items():
            for mode, g in runs:
                res = run_lap(c, bank, grip=g, mode=mode)
                r[f"{split}_{mode}_g{g}"] = summarize(res)
                per_track[f"{name}|{split}_{mode}_g{g}"] = np.stack([res["finished"].astype(float), res["lap_time"],
                                                                     res["progress_frac"], res["collisions"]], 1)
        out["results"][name] = r
        h = r[f"heldout_{runs[0][0]}_g{runs[0][1]}"]
        print(f"{name:28s} heldout {runs[0][0]} g{runs[0][1]}: done={h['completion']:.3f} lap={h['mean_lap_time']:.2f} "
              f"prog={h['mean_progress']:.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    np.savez(a.out.replace(".json", "_per_track.npz"), **{k.replace("|", "__"): v for k, v in per_track.items()})


if __name__ == "__main__":
    main()
