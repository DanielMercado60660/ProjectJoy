"""Reviewer check: the aug-tuned classical tracker hit the top of its `thr` range (0.9).
Widen thr to (0.05, 0.99) and run a local refinement from the aug best, on the SAME 40 jittered
training scenes (jitter seed 4241) and with the same objective as track_classical.tune; then
evaluate the classical tracker only, on the same 60 x 40-frame scenes per condition as
eval_tracker.py. The classical tracker gets extra search here (400 + n_local configs vs the aug
CNN's 250 Kalman configs), which only favours the baseline.

  ../bin/gpu_run ../.venv/bin/python retune_classical_wide.py
  -> checkpoints/classical_tracker_params_aug_wide.json, results/classical_tracker_tuning_aug_wide.json,
     results/tracker_eval_aug_wide.json
"""
import json, sys, time
import numpy as np
import scene as S
import track_classical as TC


def main(n_local=150, seed=1):
    t0 = time.time()
    TC.RANGES["thr"] = (0.05, 0.99)
    X, GT, VIS = S.make_set("train", 40)
    X = S.jitter_set(X, 4241)
    pres = [TC.precompute(x) for x in X]
    start = json.load(open("checkpoints/classical_tracker_params_aug.json"))
    rng = np.random.default_rng(seed)
    cur = dict(start); cur_m, _ = TC.eval_set(pres, GT, VIS, cur); cur_s = cur_m["clipped_err"]
    start_s = cur_s
    for i in range(n_local):
        p = TC.perturb(rng, cur, 0.12 * (1 - i / n_local) + 0.02)
        m, _ = TC.eval_set(pres, GT, VIS, p)
        if m["clipped_err"] < cur_s:
            cur_s, cur = m["clipped_err"], p
            print("local", i, round(cur_s, 3), m, flush=True)
    m, _ = TC.eval_set(pres, GT, VIS, cur)
    out = dict(params=cur, start_train_clipped=start_s, train_metrics=m, n_local=n_local,
               thr_range=TC.RANGES["thr"], wall_s=time.time() - t0)
    json.dump(cur, open("checkpoints/classical_tracker_params_aug_wide.json", "w"), indent=1)
    json.dump(out, open("results/classical_tracker_tuning_aug_wide.json", "w"), indent=1)
    print(out, flush=True)
    res = dict(note="classical only; same scenes as tracker_eval_aug.json", params=cur)
    for c in ["iid", "bg", "light", "distract", "all"]:
        Xc, GTc, VISc = S.make_set(c, 60)
        est = np.stack([TC.track_scene(TC.precompute(x), cur) for x in Xc])
        res[c] = dict(classical_aug_wide=TC.metrics(est, GTc, VISc))
        print(c, res[c], f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open("results/tracker_eval_aug_wide.json", "w"), indent=1)


if __name__ == "__main__":
    main(n_local=int(sys.argv[1]) if len(sys.argv) > 1 else 150)
