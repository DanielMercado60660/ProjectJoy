"""Zero-shot check of the published ViNT / NoMaD checkpoints on duck episodes (protocol: results/r8/nav/protocol.json).

    ./r8_job.sh nav-zs ../.venv/bin/python r8_nav/zeroshot.py [--stages primary,wide,secondary] [--limit N]
        -> results/r8/nav/zeroshot.json (rewritten after every stage), runs/r8_nav/preds_<cond>.npz

Per episode and camera: every nav frame is encoded once per model (obs tokens are context-independent), goal tokens
are computed once per (query frame, goal image) and shared by the context spacings, then the heads run per condition.
A condition = (model, camera, goal, spacing[, goal mask]). Truth (ORACLE, scoring only): trunk pose, camera axis,
head joints, plate centre. Frame of reference: origin = trunk xy, x = camera optical axis projected on the floor.
MLX in float32 with TF32 off (the parity setting); no torch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402  (r8_nav.mlx_models sets MLX_ENABLE_TF32=0 before the first matmul)
import numpy as np  # noqa: E402
from scipy.stats import binomtest, spearmanr  # noqa: E402

from r8_nav import mlx_models as M  # noqa: E402
from r8_nav.frames import heading_of_axis, resize, to_ground_frame, wrap  # noqa: E402

EPI = HERE / "episodes" / "r8_nav"
OUT = HERE / "results" / "r8" / "nav" / "zeroshot.json"
PRED_DIR = HERE / "runs" / "r8_nav"
CAMERAS = ("normal", "wide")
GOALS = ("G1", "G2", "wrong")
SPACINGS = (1, 2, 4)                 # grid steps: 0.25, 0.5, 1.0 s
T_MIN_K, WINDOW_K, HORIZON_S = 20, 160, 5.0   # grid frames: t >= 5.0 s, within 40 s of the goal frame
NEAR_M, NEAR_SLICE_M = 0.5, 0.15
MOTION_STEPS, MOTION_MIN_M, HEAD_YAW_MAX = 4, 0.05, 0.2
N_SAMPLES = 8
SEED = 20261105


# ------------------------------------------------------------------------------------------- data
def load_episodes(limit=None):
    metas = [json.loads(line) for line in (EPI / "index.jsonl").read_text().splitlines()]
    eps = []
    for i, m in enumerate(metas):
        if m["g1_index"] is None:
            continue
        z = np.load(EPI / f"{m['expert']}_{m['config_id']}.npz")
        e = {k: z[k] for k in z.files if k != "meta"}
        e["meta"] = m
        eps.append(e)
    if limit:
        kept, count = [], {}
        for e in eps:
            x = e["meta"]["expert"]
            count[x] = count.get(x, 0) + 1
            if count[x] <= limit:
                kept.append(e)
        eps = kept
    return metas, eps


def wrong_partner(eps):
    """Index of the next episode (circular) with the same expert: its G1 is the wrong-room goal."""
    out = []
    for i, e in enumerate(eps):
        same = [j for j in range(len(eps)) if eps[j]["meta"]["expert"] == e["meta"]["expert"]]
        k = same.index(i)
        j = same[(k + 1) % len(same)]
        assert eps[j]["meta"]["config_id"] != e["meta"]["config_id"]
        out.append(j)
    return out


def truth(e):
    """Per-frame ORACLE quantities and the query set."""
    m = e["meta"]
    t = e["time"]
    g = m["g1_index"]
    tg = t[g]
    plate = np.array(m["plate_xy"])
    k = np.arange(len(t))                        # frame index = grid index (frames are captured in grid order)
    q = np.flatnonzero((k >= T_MIN_K) & (k < g) & (k >= g - WINDOW_K))
    head = heading_of_axis(e["cam_axis"])
    pg = to_ground_frame(plate[None], e["trunk_xy"], head)               # (N, 2)
    n = len(t)
    nxt = np.minimum(np.arange(n) + MOTION_STEPS, n - 1)
    disp = e["trunk_xy"][nxt] - e["trunk_xy"]
    dg = np.stack([np.cos(head) * disp[:, 0] + np.sin(head) * disp[:, 1],
                   -np.sin(head) * disp[:, 0] + np.cos(head) * disp[:, 1]], axis=-1)
    has_next = np.arange(n) + MOTION_STEPS < n
    return {"q": q, "remaining": tg - t, "dist": np.hypot(*(e["trunk_xy"] - plate).T),
            "bearing_plate": np.arctan2(pg[:, 1], pg[:, 0]), "bearing_motion": np.arctan2(dg[:, 1], dg[:, 0]),
            "moved": np.hypot(*disp.T) * has_next, "head_yaw": e["head"][:, 2], "t_goal": float(tg)}


# ------------------------------------------------------------------------------------------- models
class Runner:
    def __init__(self):
        self.vint = M.ViNT()
        self.nomad = M.NoMaD()
        self.cache = {}

    @staticmethod
    def _norm(frames, size):
        return mx.array(M.preprocess(resize(frames, size), size))

    def tokens(self, key, model, frames, batch=256):
        if key not in self.cache:
            net = self.vint if model == "vint" else self.nomad
            x = self._norm(frames, net.size)
            outs = [net.encode_obs(x[i:i + batch]) for i in range(0, x.shape[0], batch)]
            self.cache[key] = mx.concatenate(outs, axis=0)
            mx.eval(self.cache[key])
        return self.cache[key]

    def goal_tokens(self, key, model, cur_frames, goal_frames, batch=256):
        if key not in self.cache:
            net = self.vint if model == "vint" else self.nomad
            c = self._norm(cur_frames, net.size)
            g = self._norm(goal_frames, net.size)
            outs = [net.encode_goal(c[i:i + batch], g[i:i + batch]) for i in range(0, c.shape[0], batch)]
            self.cache[key] = mx.concatenate(outs, axis=0)
            mx.eval(self.cache[key])
        return self.cache[key]


def context_index(q, ctx, s):
    """(Q, ctx + 1) frame indices, oldest first."""
    return q[:, None] + s * np.arange(-ctx, 1)[None, :]


def run_condition(R, e, ei, cam, model, goal, s, eps, partner, mask=False, defer=False):
    """Predictions for one episode under one condition: dict(dist (Q,), wp (Q, n, 2|4), [samples]).
    defer=True (NoMaD): return the context vectors and this episode's noise instead of sampling, so that
    sample_deferred() can run the DDPM for all episodes in a few large batches (same per-episode noise draws)."""
    tr = e["_truth"]
    q = tr["q"]
    frames = e[cam]
    net = R.vint if model == "vint" else R.nomad
    tok = R.tokens((ei, cam, model), model, frames)
    idx = mx.array(context_index(q, net.context, s).reshape(-1))
    obs_tok = tok[idx].reshape(len(q), net.context + 1, -1)
    if goal == "G1":
        gimg = frames[e["meta"]["g1_index"]]
    elif goal == "G2":
        gimg = e["g2_" + cam]
    else:
        p = eps[partner[ei]]
        gimg = p[cam][p["meta"]["g1_index"]]
    gkey = (ei, cam, model, "masked" if mask else goal)
    gt = R.goal_tokens(gkey, model, frames[q], np.repeat(gimg[None], len(q), axis=0))
    if model == "vint":
        d, wp = net.head(obs_tok, gt)
        mx.eval(d, wp)
        return {"dist": np.array(d), "wp": np.array(wp)}
    ctxv = net.context_vector(obs_tok, gt, mx.full((len(q),), bool(mask)))
    d = net.distance(ctxv)
    gi = 3 if mask else GOALS.index(goal)
    rng = np.random.default_rng([SEED, ei, CAMERAS.index(cam), gi, SPACINGS.index(s)])
    noise_np = rng.standard_normal((net.n_diffusion + 1, len(q) * N_SAMPLES, net.n_way, 2)).astype(np.float32)
    if defer:
        mx.eval(ctxv, d)
        return {"dist": np.array(d), "_ctx": ctxv, "_noise": noise_np, "_Q": len(q)}
    nd = net.sample_actions(mx.repeat(ctxv, N_SAMPLES, axis=0), mx.array(noise_np))
    mx.eval(d, nd)
    wp = M.NoMaD.to_waypoints(np.array(nd)).reshape(len(q), N_SAMPLES, net.n_way, 2)
    return {"dist": np.array(d), "wp": wp[:, 0], "samples": wp}


def sample_deferred(R, items, chunk=8192):
    """Run NoMaD's DDPM for every deferred episode at once (chunks of <= `chunk` sequences); fills wp / samples."""
    net = R.nomad
    ctx = mx.concatenate([mx.repeat(it["_ctx"], N_SAMPLES, axis=0) for it in items], axis=0)
    noise = np.concatenate([it["_noise"] for it in items], axis=1)
    outs = []
    for i in range(0, ctx.shape[0], chunk):
        nd = net.sample_actions(ctx[i:i + chunk], mx.array(noise[:, i:i + chunk]))
        mx.eval(nd)
        outs.append(np.array(nd))
    wp_all = M.NoMaD.to_waypoints(np.concatenate(outs))
    o = 0
    for it in items:
        n = it["_Q"] * N_SAMPLES
        wp = wp_all[o:o + n].reshape(it["_Q"], N_SAMPLES, net.n_way, 2)
        o += n
        it["wp"], it["samples"] = wp[:, 0], wp
        for k in ("_ctx", "_noise", "_Q"):
            del it[k]


# ------------------------------------------------------------------------------------------- metrics
def deg(a):
    return np.degrees(np.abs(wrap(a)))


def rho(a, b):
    if len(a) < 5 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def episode_metrics(tr, pred):
    q = tr["q"]
    rem, dist = tr["remaining"][q], tr["dist"][q]
    out = {"n_q": int(len(q)), "rho": rho(pred["dist"], rem), "rho_vs_metric_dist": rho(pred["dist"], dist)}
    hz = rem <= HORIZON_S
    out["rho_last5s"] = rho(pred["dist"][hz], rem[hz]) if hz.sum() >= 5 else None
    th0 = np.arctan2(pred["wp"][:, 0, 1], pred["wp"][:, 0, 0])
    th2 = np.arctan2(pred["wp"][:, 2, 1], pred["wp"][:, 2, 0])
    far = dist >= NEAR_M
    bp = tr["bearing_plate"][q]
    out["err_plate"] = deg(th0[far] - bp[far])
    out["err_plate_wp2"] = deg(th2[far] - bp[far])
    mv = far & (tr["moved"][q] >= MOTION_MIN_M) & (np.abs(tr["head_yaw"][q]) < HEAD_YAW_MAX)
    out["err_motion"] = deg(th0[mv] - tr["bearing_motion"][q][mv])
    sl = hz & (dist >= NEAR_SLICE_M)
    out["err_plate_last5s"] = deg(th0[sl] - bp[sl])
    out["err_straight"] = deg(bp[far])                      # control C3 on the same frames
    if "samples" in pred:
        mean0 = pred["samples"][:, :, 0, :].mean(axis=1)
        out["err_plate_sample_mean"] = deg(np.arctan2(mean0[far, 1], mean0[far, 0]) - bp[far])
    out["dist_pred_median"] = float(np.median(pred["dist"]))
    return out


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def sign_test(diffs):
    d = [v for v in diffs if v is not None and np.isfinite(v) and v != 0]
    if not d:
        return {"n": 0, "pos": 0, "p": None}
    k = int(sum(v > 0 for v in d))
    return {"n": len(d), "pos": k, "p": float(binomtest(k, len(d), 0.5).pvalue)}


def summarise(per_ep):
    """per_ep: list of episode_metrics dicts (one condition)."""
    pooled = lambda k: np.concatenate([m[k] for m in per_ep]) if per_ep else np.zeros(0)  # noqa: E731
    s = {"n_episodes": len(per_ep), "n_query_frames": int(sum(m["n_q"] for m in per_ep)),
         "rho_median": med([m["rho"] for m in per_ep]),
         "rho_last5s_median": med([m["rho_last5s"] for m in per_ep]),
         "rho_vs_metric_dist_median": med([m["rho_vs_metric_dist"] for m in per_ep]),
         "dist_pred_median": med([m["dist_pred_median"] for m in per_ep])}
    for k in ("err_plate", "err_plate_wp2", "err_motion", "err_plate_last5s", "err_straight", "err_plate_sample_mean"):
        if per_ep and k in per_ep[0]:
            p = pooled(k)
            s[k + "_pooled_median"] = float(np.median(p)) if len(p) else None
            s[k + "_n_frames"] = int(len(p))
            s[k + "_episode_median"] = med([float(np.median(m[k])) if len(m[k]) else None for m in per_ep])
    pr = np.concatenate([m["rho_frames"] for m in per_ep]) if per_ep and "rho_frames" in per_ep[0] else None
    if pr is not None:
        s["rho_pooled"] = float(spearmanr(pr[:, 0], pr[:, 1]).statistic)
    return s


def paired(per_a, per_b, key, sign):
    """Per-episode differences (sign * (a - b)) of rho or of per-episode median errors."""
    out = []
    for a, b in zip(per_a, per_b):
        if key == "rho":
            va, vb = a["rho"], b["rho"]
        else:
            va = float(np.median(a[key])) if len(a[key]) else None
            vb = float(np.median(b[key])) if len(b[key]) else None
        out.append(None if va is None or vb is None else sign * (va - vb))
    return {"median_diff": med(out), "sign_test": sign_test(out)}


def gates(model, res, cam):
    """The pre-registered gates for one model on one camera (G1, 0.25 s), from the per-episode metrics."""
    real = res[(model, cam, "G1", 1, False)]
    wrong = res[(model, cam, "wrong", 1, False)]
    s = summarise(real)
    g = {"rho_median": s["rho_median"], "err_plate_pooled_median": s["err_plate_pooled_median"]}
    g["G_a"] = bool(s["rho_median"] is not None and s["rho_median"] >= 0.5)
    g["G_b"] = bool(s["err_plate_pooled_median"] is not None and s["err_plate_pooled_median"] <= 30.0)
    dr = paired(real, wrong, "rho", +1)
    g["vs_wrong_rho"] = dr
    g["G_c_distance"] = bool(dr["median_diff"] is not None and dr["median_diff"] >= 0.2 and
                             dr["sign_test"]["p"] is not None and dr["sign_test"]["p"] < 0.05)
    de = paired(wrong, real, "err_plate", +1)               # err_wrong - err_real
    g["vs_wrong_err"] = de
    ok_b = de["median_diff"] is not None and de["median_diff"] >= 10 and de["sign_test"]["p"] is not None \
        and de["sign_test"]["p"] < 0.05
    if model == "nomad":
        dm = paired(res[(model, cam, "G1", 1, True)], real, "err_plate", +1)
        g["vs_masked_err"] = dm
        ok_b = ok_b and dm["median_diff"] is not None and dm["median_diff"] >= 10 and \
            dm["sign_test"]["p"] is not None and dm["sign_test"]["p"] < 0.05
    g["G_c_bearing"] = bool(ok_b)
    st = []
    for m in real:
        if len(m["err_plate"]):
            st.append(float(np.median(m["err_straight"])) - float(np.median(m["err_plate"])))
        else:
            st.append(None)
    g["vs_straight_err"] = {"median_diff": med(st), "sign_test": sign_test(st)}
    g["straight_ahead_reading"] = bool(med(st) is not None and med(st) > 0 and g["vs_straight_err"]["sign_test"]["p"]
                                       is not None and g["vs_straight_err"]["sign_test"]["p"] < 0.05)
    g["all"] = g["G_a"] and g["G_b"] and g["G_c_distance"] and g["G_c_bearing"] and g["straight_ahead_reading"]
    g["distance_part"] = g["G_a"] and g["G_c_distance"]
    g["waypoint_part"] = g["G_b"] and g["G_c_bearing"] and g["straight_ahead_reading"]
    return g


def verdict(gs):
    if gs.get("normal", {}).get("all") or gs.get("wide", {}).get("all"):
        return "transfers"
    if any(gs.get(c, {}).get("distance_part") or gs.get(c, {}).get("waypoint_part") for c in CAMERAS):
        return "partly"
    return "does_not_transfer"


# ------------------------------------------------------------------------------------------- driver
STAGES = {
    "primary": [(m, "normal", g, 1, False) for m in ("vint", "nomad") for g in ("G1", "wrong")] +
               [("nomad", "normal", "G1", 1, True)],
    "wide": [(m, "wide", g, 1, False) for m in ("vint", "nomad") for g in ("G1", "wrong")] +
            [("nomad", "wide", "G1", 1, True)],
    "secondary": [(m, c, "G2", 1, False) for m in ("vint", "nomad") for c in CAMERAS] +
                 [(m, c, g, s, False) for s in (2, 4) for m in ("vint", "nomad") for c in CAMERAS for g in ("G1", "wrong")] +
                 [("nomad", c, "G1", s, True) for s in (2, 4) for c in CAMERAS],
}


def cond_name(c):
    m, cam, g, s, mask = c
    return f"{m}|{cam}|{'masked' if mask else g}|{0.25 * s:g}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="primary,wide,secondary")
    ap.add_argument("--limit", type=int, default=None, help="episodes per expert (smoke runs)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--merge", action="store_true", help="keep the conditions / gates already in --out")
    a = ap.parse_args()
    out_path = Path(a.out)
    t0 = time.time()
    metas, eps = load_episodes(a.limit)
    for e in eps:
        e["_truth"] = truth(e)
    few = [f"{e['meta']['expert']}_{e['meta']['config_id']}" for e in eps if len(e["_truth"]["q"]) < 5]
    eps = [e for e in eps if len(e["_truth"]["q"]) >= 5]       # guard: a goal frame before t = 6.25 s (none expected)
    partner = wrong_partner(eps)
    R = Runner()
    res = {}
    report = {"protocol": "results/r8/nav/protocol.json", "mlx_device": str(mx.default_device()),
              "episodes": {"recorded": len(metas), "with_goal_frame": len(eps),
                           "excluded_no_entry": [f"{m['expert']}_{m['config_id']}" for m in metas if m["g1_index"] is None],
                           "excluded_fewer_than_5_query_frames": few,
                           "per_expert": {x: sum(e["meta"]["expert"] == x for e in eps) for x in ("servo", "bot")}},
              "conditions": {}, "gates": {}, "timing_s": {}}
    if a.merge and out_path.exists():
        old = json.loads(out_path.read_text())
        for k in ("conditions", "gates", "timing_s"):
            report[k] = old.get(k, {})
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for stage in a.stages.split(","):
        ts = time.time()
        for c in STAGES[stage]:
            model, cam, goal, s, mask = c
            per_ep, preds = [], {}
            ps = [run_condition(R, e, ei, cam, model, goal, s, eps, partner, mask, defer=(model == "nomad"))
                  for ei, e in enumerate(eps)]
            if model == "nomad":
                sample_deferred(R, ps)
            for ei, (e, p) in enumerate(zip(eps, ps)):
                m = episode_metrics(e["_truth"], p)
                tr = e["_truth"]
                m["rho_frames"] = np.stack([p["dist"], tr["remaining"][tr["q"]]], axis=1)
                m["expert"] = e["meta"]["expert"]
                per_ep.append(m)
                preds[f"{ei}_dist"] = p["dist"].astype(np.float32)
                preds[f"{ei}_wp"] = p["wp"].astype(np.float32)
                if "samples" in p:
                    preds[f"{ei}_samples"] = p["samples"].astype(np.float32)
                preds[f"{ei}_q"] = e["_truth"]["q"]
            res[c] = per_ep
            np.savez_compressed(PRED_DIR / f"preds_{cond_name(c).replace('|', '_')}.npz", **preds)
            summ = {"all": summarise(per_ep)}
            for x in ("servo", "bot"):
                summ[x] = summarise([m for m in per_ep if m["expert"] == x])
            summ["per_episode_rho"] = [None if m["rho"] is None else round(m["rho"], 3) for m in per_ep]
            report["conditions"][cond_name(c)] = summ
            print(f"[{time.time() - t0:6.0f}s] {cond_name(c):28s} rho_med {summ['all']['rho_median']}  "
                  f"err_plate {summ['all'].get('err_plate_pooled_median')}  straight {summ['all'].get('err_straight_pooled_median')}",
                  flush=True)
        for model in ("vint", "nomad"):
            for cam in CAMERAS:
                if (model, cam, "G1", 1, False) in res and (model, cam, "wrong", 1, False) in res and \
                        (model != "nomad" or (model, cam, "G1", 1, True) in res):
                    report["gates"].setdefault(model, {})[cam] = gates(model, res, cam)
            if model in report["gates"]:
                report["gates"][model]["verdict"] = verdict(report["gates"][model])
        report["timing_s"][stage] = round(time.time() - ts, 1)
        report["episode_index"] = [{"i": i, "expert": e["meta"]["expert"], "config": e["meta"]["config_id"],
                                    "n_q": int(len(e["_truth"]["q"])), "t_goal": e["_truth"]["t_goal"],
                                    "wrong_partner": partner[i]} for i, e in enumerate(eps)]
        out_path.write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"stage {stage} written ({time.time() - ts:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
