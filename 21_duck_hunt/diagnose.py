"""Why do the trackers fail on held-out scenes? Uses results/tracks_<cond>.npz from eval_tracker.py.
  ../.venv/bin/python diagnose.py [tag]  -> results/failure_diagnosis[_tag].json
For each condition and tracker: how often the estimate sits on a distractor (within 5 px), in the
terrain band (y > 62), or is 'parked' (scene-level std of the estimate < 3 px). For the classical
tracker also: fraction of frames with no blob candidate at all, and how many duck pixels survive
its luminance gate."""
import json, sys
import numpy as np
import scene as S
import track_classical as TC


def distractor_xy(seed, cond, nf):
    sc = S.sample_scene(np.random.default_rng(seed), cond)
    out = []
    for f in range(nf):
        pts = []
        for d in sc["distractors"]:
            x = ((d["x"] + d["vx"] * f + 30) % (S.WS + 60) - 30) / S.SS - 0.25
            y = (d["y"] + d["vy"] * f) / S.SS - 0.25
            pts.append((x, y, d["kind"]))
        out.append(pts)
    return out


def main(tag=""):
    sfx = f"_{tag}" if tag else ""
    cprm = json.load(open(f"checkpoints/classical_tracker_params{sfx}.json"))
    res = {}
    for cond in ["iid", "bg", "light", "distract", "all"]:
        T = np.load(f"results/tracks{sfx}_{cond}.npz")
        gt, vis = T["gt"], T["vis"]
        n, nf = gt.shape[:2]
        D = [distractor_xy(S.COND_SEED[cond] + i, cond, nf) for i in range(n)]
        res[cond] = {}
        for name in ["classical", "cnn_raw", "cnn_kf"]:
            est = T[name]
            err = np.linalg.norm(est - gt, axis=-1)
            lost = (err > TC.LOST_PX) & vis
            on_d = np.zeros_like(lost); kinds = {}
            for i in range(n):
                for f in range(nf):
                    for (x, y, k) in D[i][f]:
                        if lost[i, f] and np.hypot(est[i, f, 0] - x, est[i, f, 1] - y) < 5:
                            on_d[i, f] = True; kinds[k] = kinds.get(k, 0) + 1
            parked = est.std(1).max(-1) < 3.0
            res[cond][name] = dict(lost_frames=int(lost.sum()),
                                   lost_on_distractor_frac=float(on_d.sum() / max(lost.sum(), 1)),
                                   distractor_kinds=kinds,
                                   lost_in_terrain_band_frac=float((lost & (est[..., 1] > 62)).sum() / max(lost.sum(), 1)),
                                   parked_scene_frac=float(parked.mean()))
        # classical internals on 10 scenes
        X, GT, VIS = S.make_set(cond, 10)
        nocand, duck_pass = [], []
        for x, g in zip(X, GT):
            pre = TC.precompute(x)
            for f in range(len(x)):
                nocand.append(len(TC.detect(pre, f, cprm)) == 0)
                gx, gy = np.clip(np.round(g[f]).astype(int), 0, 79)
                duck_pass.append(pre["lum"][f, max(gy - 1, 0):gy + 2, max(gx - 2, 0):gx + 3].mean() > cprm["min_lum"])
        res[cond]["classical_internals"] = dict(no_candidate_frac=float(np.mean(nocand)),
                                                duck_passes_lum_gate_frac=float(np.mean(duck_pass)))
        print(cond, json.dumps(res[cond]))
    json.dump(res, open(f"results/failure_diagnosis{sfx}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
