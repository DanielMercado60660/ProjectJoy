"""Evaluate controllers on the clip under matched conditions (held-out seeds 1000+).

Controllers: pd_only (tuned PD gains, no feed-forward/balance = the RL policy's zero action),
baseline (tuned PD + feed-forward + balance), baseline_push (same, tuned under training pushes; pass
--ckpt baseline_push=results/baseline_tuned_push.json) and PPO checkpoints given with --ckpt name=path.
Conditions: nominal, pushes (0.1 s horizontal force on the torso, random sign,
spaced U(0.3, 1.0) loops apart, i.e. ~1.5 pushes per loop),
tempo x0.9 / x1.1, floor+foot friction x0.3, all link masses x1.3. Nothing but 'nominal' (and the
push sizes inside the training range, if a checkpoint trained with pushes) was seen in training/tuning.

  ../bin/gpu_run ../.venv/bin/python eval.py --ckpt ppo_pd=checkpoints/ppo_pd.pkl --out results/eval_main.json
"""
import argparse, json, os, pickle, time
import numpy as np
from multiprocessing import Pool

CONDITIONS = {
    "nominal": {}, "push40": dict(push=40), "push80": dict(push=80), "push120": dict(push=120),
    "tempo0.9": dict(tempo=0.9), "tempo1.1": dict(tempo=1.1), "friction0.3": dict(friction=0.3),
    "mass1.3": dict(mass=1.3),
    # push-tolerance sweep (not in the default list)
    "push50": dict(push=50), "push60": dict(push=60), "push70": dict(push=70), "push100": dict(push=100),
}
DEFAULT_CONDS = ["nominal", "push40", "push80", "push120", "tempo0.9", "tempo1.1", "friction0.3", "mass1.3"]
_cache = {}


def make_ctrl(name, ckpts, base_params):
    from dance_env import BaselineController
    from ppo import PolicyController
    base = BaselineController(base_params)
    if name == "baseline":
        return base
    if name.startswith("baseline_"):   # any tuned-baseline params json passed via --ckpt
        js = json.load(open(ckpts[name]))
        if js.get("controller") == "anchor":   # review addition: baseline_anchor.py
            from baseline_anchor import AnchorBaselineController
            return AnchorBaselineController(js["params"])
        return BaselineController(js["params"])
    if name == "pd_only":
        return BaselineController(dict(base_params, ff=0.0, k_com=0, k_comv=0, k_pitch=0, k_pitchv=0, lead=1.0))
    ck = pickle.load(open(ckpts[name], "rb"))
    return PolicyController(ck, BaselineController(ck["base_params"]))


def job(a):
    name, cond, seed, loops, ckpts, base_params = a
    from dance_env import DanceEnv, run_clip, metrics
    from clip import Clip
    if "clip" not in _cache:
        _cache["clip"] = Clip()
    key = (name,)
    if key not in _cache:
        _cache[key] = make_ctrl(name, ckpts, base_params)
    ctrl = _cache[key]
    env = DanceEnv(clip=_cache["clip"], seed=seed, **CONDITIONS[cond])
    phase = np.random.default_rng(seed).uniform()
    log = run_clip(env, ctrl, loops=loops, phase=phase)
    m = metrics(log, env.clip, env.tempo)
    return name, cond, seed, m


def aggregate(ms):
    out = {}
    for k in ["joint_rmse_deg", "foot_pos_err_cm", "root_z_err_cm", "root_x_err_cm", "pitch_err_deg",
              "mean_reward", "contact_agreement", "slip_cm_per_loop", "beat_kick_mean_abs_ms",
              "beat_dip_mean_abs_ms", "beat_kick_within_50ms", "beat_dip_within_50ms"]:
        v = [m[k] for m in ms if m[k] is not None]
        out[k] = float(np.mean(v)) if v else None
        out[k + "_sd"] = float(np.std(v)) if v else None
    out["falls"] = int(sum(m["falls"] for m in ms))
    out["falls_per_loop"] = float(np.mean([m["falls_per_loop"] for m in ms]))
    out["seeds_with_fall"] = int(sum(m["falls"] > 0 for m in ms))
    for ev in ["touchdown_err_ms", "liftoff_err_ms"]:
        v = [m[ev]["mean_ms"] for m in ms if m[ev]["mean_ms"] is not None]
        out[ev] = float(np.mean(v)) if v else None
        out[ev.replace("_ms", "_misses")] = int(sum(m[ev]["misses"] for m in ms))
        out[ev.replace("_ms", "_events")] = int(sum(m[ev]["n"] for m in ms))
    out["n_runs"] = len(ms)
    out["joint_rmse_deg_per_joint"] = np.mean([m["joint_rmse_deg_per_joint"] for m in ms], 0).round(2).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=path")
    ap.add_argument("--controllers", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=DEFAULT_CONDS)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--baseline", default="results/baseline_tuned.json")
    ap.add_argument("--out", default="results/eval_main.json")
    args = ap.parse_args()
    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    names = args.controllers or (["pd_only", "baseline"] + list(ckpts))
    # --ckpt baseline_push=results/baseline_tuned_push.json adds the push-tuned baseline (a params json)
    base_params = json.load(open(args.baseline))["params"]
    seeds = list(range(1000, 1000 + args.seeds))
    jobs = [(n, c, s, args.loops, ckpts, base_params) for n in names for c in args.conditions for s in seeds]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    table = {}
    for n in names:
        table[n] = {c: aggregate([m for (nn, cc, s, m) in res if nn == n and cc == c]) for c in args.conditions}
    out = dict(table=table, seeds=seeds, loops_per_seed=args.loops, ckpts=ckpts, wall_s=time.time() - t0,
               uptime=os.popen("uptime").read().strip(), conditions={c: CONDITIONS[c] for c in args.conditions})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    cols = ["falls_per_loop", "joint_rmse_deg", "foot_pos_err_cm", "root_x_err_cm", "mean_reward",
            "contact_agreement", "slip_cm_per_loop", "touchdown_err_ms", "beat_kick_mean_abs_ms", "beat_dip_mean_abs_ms"]
    print("| controller | condition | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for c in args.conditions:
        for n in names:
            r = table[n][c]
            print(f"| {n} | {c} | " + " | ".join("n/a" if r[k] is None else f"{r[k]:.3g}" for k in cols) + " |")
    print(f"wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
