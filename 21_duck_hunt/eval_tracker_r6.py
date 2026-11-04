"""R6 tracker evaluation. Every tracker sees exactly the same (cached) frames.
Trackers:
  classical_aug    previous best classical (photometric-jitter tuned, R5 follow-up), reference
  cnn_aug_raw      previous aug DuckNet, argmax per frame, reference
  classical_r6     R6 classical (track_classical_r6.py, tuned on DR scenes)
  classical_r6_oldcues  ablation: the original cue set, only retuned on the DR scenes
  cnn_r6_raw       R6 DuckNet (frame stack, DR-trained), argmax per frame
  cnn_r6_kf        R6 DuckNet top-3 peaks -> the same filter as classical_r6 (run_filter_r6), its
                   params random-searched (400 configs) on DR val scenes
  (+ extra CNN checkpoints with --extra_cnn name=path (raw only) and extra R6 classical parameter
   sets with --extra_classical name=params.json)
Conditions: dr val (in-distribution for R6), the existing held-out sets (iid/bg/light/distract/all,
same seeds as eval_tracker.py) and, with --fresh, the fresh held-out sets (scene_r6.FRESH_CONDS).
95% CIs: bootstrap over scenes (frames within a scene are correlated), 2000 resamples.

  ../bin/gpu_run ../.venv/bin/python eval_tracker_r6.py [--fresh] [--tag r6_tracker_eval]
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
import scene as S
import scene_r6 as R
import track_classical as TC
import track_classical_r6 as C
import train_tracker as TT
import train_tracker_r6 as T6
import eval_tracker as ET

KF_RANGES = dict(q=(0.05, 8), r=(0.1, 8), gate=(2, 15), min_p=(0.0, 0.5), log_switch=(0.2, 3.5))


def boot_ci(err, vis, n=2000, seed=0):
    """err, vis (N,F) -> dict with point estimates and scene-bootstrap 95% CIs for lost rate and mean error."""
    lost_s = ((err > TC.LOST_PX) & vis).sum(1); err_s = np.where(vis, err, 0).sum(1); n_s = vis.sum(1)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(err), (n, len(err)))
    L = lost_s[idx].sum(1) / n_s[idx].sum(1); E = err_s[idx].sum(1) / n_s[idx].sum(1)
    e = err[vis]
    return dict(lost_rate=float((e > TC.LOST_PX).mean()), lost_ci=[float(np.percentile(L, 2.5)), float(np.percentile(L, 97.5))],
                mean_err_px=float(e.mean()), mean_err_ci=[float(np.percentile(E, 2.5)), float(np.percentile(E, 97.5))],
                median_err_px=float(np.median(e)), n_scenes=int(len(err)), n_frames=int(vis.sum()))


def kf_track(pk, prm):
    cands = [[c for c in cs if c[2] >= prm["min_p"]] for cs in pk]
    return C.run_filter_r6(cands, prm)


def tune_kf(model, n_cfg=400, seed=0, val="drval"):
    Xv, GTv, VISv = R.dr2_set("dr2val", 40) if val == "dr2val" else R.dr_set("drval", 40)
    pks = [T6.peaks(model, x) for x in Xv]
    rng = np.random.default_rng(seed); best = (1e9, None)
    for i in range(n_cfg):
        p = {k: float(rng.uniform(*v)) for k, v in KF_RANGES.items()}
        if rng.random() < 0.25: p["log_switch"] = 99.0
        p["switch"] = float(np.exp(p["log_switch"])) if p["log_switch"] < 50 else float("inf")
        p["reinit"] = int(rng.integers(1, 8)); p["use_kf"] = True
        m = TC.metrics(np.stack([kf_track(pk, p) for pk in pks]), GTv, VISv)
        if m["clipped_err"] < best[0]:
            best = (m["clipped_err"], p)
    return best[1], best[0]


_CP = None


def _init(cp):
    global _CP
    _CP = cp


def _classical(frames):
    out = {}
    if "classical_aug" in _CP:
        out["classical_aug"] = TC.track_scene(TC.precompute(frames), _CP["classical_aug"])
    pre = C.precompute(frames)
    for k, prm in _CP.items():
        if k != "classical_aug":
            out[k] = C.track_scene(pre, prm)
    return out


def sets(fresh, fresh3=False):
    yield "drval", R.dr_set("drval", 40)
    if fresh3:
        yield "dr2val", R.dr2_set("dr2val", 40)
    for c in R.OLD_CONDS:
        yield c, R.heldout(c)
    if fresh:
        for c in R.FRESH_CONDS:
            yield c, R.fresh_set(c)
    if fresh3:
        for c in R.FRESH3_CONDS:
            yield c, R.fresh3_set(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--fresh3", action="store_true", help="round 2: also DR v2 val and the round-2 held-out sets")
    ap.add_argument("--kf_val", default="drval", choices=["drval", "dr2val"], help="val set for the CNN filter search")
    ap.add_argument("--tag", default="r6_tracker_eval")
    ap.add_argument("--cnn", default="checkpoints/r6_tracker_cnn_nf3")
    ap.add_argument("--classical", default="checkpoints/r6_tracker_classical.json")
    ap.add_argument("--extra_cnn", nargs="*", default=[], help="name=path, evaluated raw only")
    ap.add_argument("--extra_classical", nargs="*", default=[], help="name=params.json, R6 classical variants")
    ap.add_argument("--kf_params", default=None, help="reuse frozen CNN filter params instead of re-tuning")
    a = ap.parse_args()
    t0 = time.time()
    cp = {"classical_aug": json.load(open("checkpoints/classical_tracker_params_aug.json")),
          "classical_r6": json.load(open(a.classical))}
    if os.path.exists("checkpoints/r6_tracker_classical_oldcues.json"):
        cp["classical_r6_oldcues"] = json.load(open("checkpoints/r6_tracker_classical_oldcues.json"))
    for kv in a.extra_classical:
        cp[kv.split("=")[0]] = json.load(open(kv.split("=")[1]))
    for k in cp:
        cp[k].setdefault("switch", float("inf"))
    cnn = T6.load(a.cnn); cnn_aug = ET.load_net("checkpoints/ducknet_aug")
    extra = {kv.split("=")[0]: T6.load(kv.split("=")[1]) for kv in a.extra_cnn}
    if a.kf_params:
        kprm, kval = json.load(open(a.kf_params)), None
    else:
        kprm, kval = tune_kf(cnn, val=a.kf_val)
        json.dump(kprm, open(a.cnn + "_kf_params.json", "w"), indent=1)
    print("cnn kf params", kprm, kval, f"{time.time()-t0:.0f}s", flush=True)
    res = dict(cnn=a.cnn, classical=a.classical, cnn_kf_params=kprm, cnn_kf_val_clipped=kval, flight_sha=R.flight_sha(),
               lost_threshold_px=TC.LOST_PX, ci="95% scene bootstrap, 2000 resamples", conds={})
    with Pool(4, initializer=_init, initargs=(cp,)) as pool:
        for cond, (X, GT, VIS) in sets(a.fresh, a.fresh3):
            cl = pool.map(_classical, list(X))
            est = {k: np.stack([c[k] for c in cl]) for k in cp}
            est["cnn_aug_raw"] = np.stack([TT.predict(cnn_aug, x)[0] for x in X])
            est["cnn_r6_raw"] = np.stack([T6.predict(cnn, x)[0] for x in X])
            est["cnn_r6_kf"] = np.stack([kf_track(T6.peaks(cnn, x), kprm) for x in X])
            for nm, m in extra.items():
                est[nm] = np.stack([T6.predict(m, x)[0] for x in X])
            res["conds"][cond] = {k: boot_ci(np.linalg.norm(v - GT, axis=-1), VIS) for k, v in est.items()}
            np.savez_compressed(f"results/{a.tag}_tracks_{cond}.npz", gt=GT, vis=VIS, **est)
            print(cond, f"{time.time()-t0:.0f}s", " | ".join(f"{k} {v['lost_rate']*100:.1f}% {v['mean_err_px']:.2f}px"
                                                             for k, v in res["conds"][cond].items()), flush=True)
            json.dump(res, open(f"results/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
