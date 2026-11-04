"""R6 post-freeze failure diagnosis (no tracker changes), from saved tracks of eval_tracker_r6.py --fresh.
  ../.venv/bin/python diagnose_r6.py -> results/r6_tracker_failure_diagnosis.json
1. all2/light2: per-scene lost rate vs haze strength and background kind (city/autumn), both trackers.
2. old 'light': why the selected classical (v4) fails: frames with no candidate, how often its score
   map's maximum is on the duck, parked scenes, lost estimates in the terrain band.
3. Lost frames on a distractor (within 6 px of its centre) for distract/distract_f (old generator).
"""
import json
from multiprocessing import Pool
import numpy as np
import scene_r6 as R
import track_classical_r6 as C
import diagnose as D

PRM = json.load(open("checkpoints/r6_tracker_classical.json"))
TR = ["cnn_r6_kf", "classical_r6"]


def lost_of(T, nm):
    return (np.linalg.norm(T[nm] - T["gt"], axis=-1) > 5) & T["vis"]


def classical_internals(x):
    pre = C.precompute(x)
    return [len(C.detect(pre, k, PRM)) == 0 for k in range(len(x))], [C.score_map(pre, k, PRM) for k in range(len(x))]


def main():
    out = {}
    for cond in ["all2", "light2"]:
        T = np.load(f"results/r6_tracker_eval_fresh_tracks_{cond}.npz")
        scs = [R.sample_fresh(np.random.default_rng(R.FRESH_SEED[cond] + i), cond) for i in range(len(T["gt"]))]
        hz = np.array([s["haze"][0] for s in scs])
        for nm in TR:
            L = lost_of(T, nm); per = L.sum(1) / T["vis"].sum(1)
            d = dict(lost_haze_below_035=float(per[hz < 0.35].mean()), lost_haze_035_up=float(per[hz >= 0.35].mean()),
                     n_scenes_haze_below_035=int((hz < 0.35).sum()), corr_haze_vs_scene_lost=float(np.corrcoef(hz, per)[0, 1]))
            if cond == "all2":
                kinds = np.array([s["bg2"] for s in scs])
                d.update(lost_city=float(per[kinds == "city"].mean()), lost_autumn=float(per[kinds == "autumn"].mean()))
            out[f"{cond}_{nm}"] = d
    X, GT, VIS = R.heldout("light")
    T = np.load("results/r6_tracker_eval_fresh_tracks_light.npz")
    with Pool(4) as p:
        r = p.map(classical_internals, list(X[:12]))
    on = [np.hypot(*(np.array(np.unravel_index(sm.argmax(), sm.shape)[::-1]) - GT[i, k])) <= 5
          for i, (_, sms) in enumerate(r) for k, sm in enumerate(sms) if VIS[i, k]]
    est = T["classical_r6"]; L = lost_of(T, "classical_r6")
    out["light_classical_v4"] = dict(no_candidate_frac=float(np.mean([v for a, _ in r for v in a])),
                                     score_max_on_duck_frac=float(np.mean(on)), n_scenes_internals=12,
                                     parked_scene_frac=float((est.std(1).max(-1) < 3).mean()),
                                     lost_in_terrain_band_frac=float((L & (est[..., 1] > 62)).sum() / L.sum()))
    for cond, seed in [("distract", R.S.COND_SEED["distract"]), ("distract_f", R.FRESH_SEED["distract_f"])]:
        T = np.load(f"results/r6_tracker_eval_fresh_tracks_{cond}.npz")
        for nm in TR + ["classical_r6_v3"]:
            L = lost_of(T, nm); kinds = {}; on_d = 0
            for i in range(len(L)):
                dx = D.distractor_xy(seed + i, "distract", L.shape[1])
                for f in np.nonzero(L[i])[0]:
                    hit = [k for (x, y, k) in dx[f] if np.hypot(T[nm][i, f, 0] - x, T[nm][i, f, 1] - y) < 6]
                    if hit:
                        on_d += 1; kinds[str(hit[0])] = kinds.get(str(hit[0]), 0) + 1
            out[f"{cond}_{nm}"] = dict(lost_frames=int(L.sum()), on_distractor=on_d, kinds=kinds)
    json.dump(out, open("results/r6_tracker_failure_diagnosis.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
