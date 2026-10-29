"""Evaluate trackers on in-distribution and HELD-OUT scene conditions.
Trackers (all see exactly the same frames):
  classical   bg-subtraction + colour + blob + Kalman, tuned on 40 train scenes
  cnn_raw     DuckNet argmax per frame (single frame, no temporal info)
  cnn_kf      DuckNet top-3 peaks -> same gated Kalman tracker, its 5 params tuned on 40 val scenes
  ../.venv/bin/python eval_tracker.py   -> results/tracker_eval.json
"""
import json, sys, time
import numpy as np
import mlx.core as mx
import scene as S
import track_classical as TC
import train_tracker as TT

KF_RANGES = dict(q=(0.05, 8), r=(0.1, 8), gate=(2, 15), min_p=(0.0, 0.5))


def load_net(path="checkpoints/ducknet"):
    m = TT.DuckNet(); m.load_weights(path + ".safetensors"); return m


def net_peaks(model, frames, k=3):
    """frames (F,H,W,3) -> per-frame list of top-k (x, y, p) cells, at least 2 cells apart."""
    out = []
    for s in range(0, len(frames), 256):
        o = np.array(model(TT.prep(frames[s:s + 256])))
        B = o.shape[0]
        lg = o[..., 0].reshape(B, -1)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        for b in range(B):
            order = np.argsort(-p[b]); cs = []
            for i in order[:20]:
                gy, gx = divmod(int(i), TT.G)
                if any(abs(gx - c[3]) < 2 and abs(gy - c[4]) < 2 for c in cs):
                    continue
                dx, dy = o[b, gy, gx, 1:]
                cs.append(((gx + 0.5 + dx) * TT.CELL - 0.5, (gy + 0.5 + dy) * TT.CELL - 0.5, p[b, i], gx, gy))
                if len(cs) == k:
                    break
            out.append([c[:3] for c in cs])
    return out


def cnn_kf_track(peaks_scene, prm):
    cands = [[c for c in cs if c[2] >= prm["min_p"]] for cs in peaks_scene]
    return TC.run_filter(cands, dict(prm, reinit=prm["reinit"], use_kf=True))


def tune_cnn_kf(model, n_scenes=40, n_cfg=400, seed=0, aug=False):
    X, GT, VIS = S.make_set("val", n_scenes)
    if aug:
        X = S.jitter_set(X, 4242)
    peaks = [net_peaks(model, x) for x in X]
    rng = np.random.default_rng(seed)
    best = (1e9, None)
    for i in range(n_cfg):
        p = {k: float(rng.uniform(*v)) for k, v in KF_RANGES.items()}; p["reinit"] = int(rng.integers(1, 8))
        est = np.stack([cnn_kf_track(pk, p) for pk in peaks])
        m = TC.metrics(est, GT, VIS)
        if m["clipped_err"] < best[0]:
            best = (m["clipped_err"], p)
    return best[1], best[0]


def main(n_scenes=60, conds=("iid", "bg", "light", "distract", "all"), tag="", n_cfg=400):
    """tag='' -> primary models; tag='aug' -> photometric-jitter follow-up models (separate files)."""
    t0 = time.time()
    sfx = f"_{tag}" if tag else ""
    model = load_net(f"checkpoints/ducknet{sfx}")
    cprm = json.load(open(f"checkpoints/classical_tracker_params{sfx}.json"))
    kprm, kval = tune_cnn_kf(model, n_cfg=n_cfg, aug=(tag == "aug"))
    json.dump(kprm, open(f"checkpoints/cnn_kf_params{sfx}.json", "w"), indent=1)
    print("cnn_kf params", kprm, "val clipped", kval, flush=True)
    res = dict(tag=tag, n_cnn_kf_configs=n_cfg, n_scenes_per_cond=n_scenes, frames_per_scene=S.N_FRAMES, lost_threshold_px=TC.LOST_PX,
               cnn_kf_params=kprm, cnn_kf_val_clipped=kval)
    for c in conds:
        X, GT, VIS = S.make_set(c, n_scenes)
        est_c = np.stack([TC.track_scene(TC.precompute(x), cprm) for x in X])
        raw = np.stack([TT.predict(model, x)[0] for x in X])
        peaks = [net_peaks(model, x) for x in X]
        est_k = np.stack([cnn_kf_track(pk, kprm) for pk in peaks])
        res[c] = dict(classical=TC.metrics(est_c, GT, VIS), cnn_raw=TC.metrics(raw, GT, VIS),
                      cnn_kf=TC.metrics(est_k, GT, VIS))
        np.savez_compressed(f"results/tracks{sfx}_{c}.npz", classical=est_c, cnn_raw=raw, cnn_kf=est_k, gt=GT, vis=VIS)
        print(c, json.dumps(res[c]), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"results/tracker_eval{sfx}.json", "w"), indent=1)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    main(tag=tag, n_cfg=250 if tag == "aug" else 400)
