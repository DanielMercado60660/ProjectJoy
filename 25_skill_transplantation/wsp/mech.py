"""Mechanistic check of the WSP networks (runner: Claude). Numpy only, ~1 min.

States: each specialist's own trajectories under its own label on 5 held-out resets (31000-31004),
5000 states per task. For every network (per seed) it reports
  label_effect   mean |pi(s,fwd) - pi(s,bwd)| over all 10k states (the average/merges are ~0 here by design)
  mse_fwd/bwd    action MSE vs the fwd (bwd) specialist on the fwd (bwd) specialist's states, own label
  act1/act2      per label, number of hidden units that are 'live' (|pre-activation| < 2 on >5% of states)
                 and overlap of the live sets between the labels
  sat1/sat2      per label, units saturated (|pre| > 4) on >95% of states
  frac_changed   fraction of actor weights that differ (>1e-6) from the average / theta0 / fwd / bwd
  nearest_ckpt   L2 distance to the nearest actor in checkpoints/s<k>/*.npz (copy detection)

  ../.venv/bin/python wsp/mech.py      -> wsp/results/mech.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import numpy as np  # noqa: E402
from common import load_params, make_env, policy_input  # noqa: E402
import merge as M  # noqa: E402

AK = ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "a3.weight", "a3.bias"]


def fwdpass(p, x):
    z1 = x @ p["a1.weight"].T + p["a1.bias"]
    h1 = np.tanh(z1)
    z2 = h1 @ p["a2.weight"].T + p["a2.bias"]
    h2 = np.tanh(z2)
    return np.tanh(h2 @ p["a3.weight"].T + p["a3.bias"]), z1, z2


def states(p, norm, tid, seeds=range(31000, 31005)):
    S = []
    for sd in seeds:
        e = make_env()
        o, _ = e.reset(seed=sd)
        for _ in range(1000):
            S.append(o.copy())
            a = fwdpass(p, policy_input(norm, o[None], np.array([tid])))[0][0]
            o, *_ = e.step(a)
        e.close()
    return np.array(S)


def main():
    out = {}
    for seed in range(4):
        d = os.path.join(PROJ, "checkpoints", f"s{seed}")
        t0, norm = load_params(os.path.join(d, "base.npz"))
        A, _ = load_params(os.path.join(d, "pre_fwd.npz"))
        B, _ = load_params(os.path.join(d, "pre_bwd.npz"))
        Sf, Sb = states(A, norm, 0), states(B, norm, 1)
        X = {(st, lab): policy_input(norm, S, np.full(len(S), lab)) for st, S in (("f", Sf), ("b", Sb)) for lab in (0, 1)}
        ref_f, ref_b = fwdpass(A, X["f", 0])[0], fwdpass(B, X["b", 1])[0]
        ck = {os.path.basename(f): load_params(f)[0] for f in glob.glob(os.path.join(d, "*.npz"))}
        nets = {"average": M.interpolate(A, B, 0.5), "theta0": t0, "fwd_spec_both": A}
        for f in sorted(glob.glob(os.path.join(HERE, "attempts", "r6wsp_*", f"merged_s{seed}.npz"))):
            z = np.load(f)
            nets[os.path.basename(os.path.dirname(f))] = {k: z[k].astype(np.float32) for k in AK}
        for f in sorted(glob.glob(os.path.join(HERE, "baselines", f"*_s{seed}.npz"))):
            nets[os.path.basename(f)[:-len(f"_s{seed}.npz")]] = {k: np.load(f)[k].astype(np.float32) for k in
                                                               ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "a3.weight", "a3.bias"]}
        avg = nets["average"]
        R = {}
        for name, p in nets.items():
            r = {}
            allx = np.concatenate([X["f", 0][:, :17], X["b", 0][:, :17]])
            xf = np.concatenate([allx, np.tile([1, 0], (len(allx), 1))], 1).astype(np.float32)
            xb = np.concatenate([allx, np.tile([0, 1], (len(allx), 1))], 1).astype(np.float32)
            af, z1f, z2f = fwdpass(p, xf)
            ab, z1b, z2b = fwdpass(p, xb)
            r["label_effect"] = float(np.abs(af - ab).mean())
            r["mse_fwd"] = float(((fwdpass(p, X["f", 0])[0] - ref_f) ** 2).mean())
            r["mse_bwd"] = float(((fwdpass(p, X["b", 1])[0] - ref_b) ** 2).mean())
            for li, (zf, zb) in enumerate(((z1f, z1b), (z2f, z2b)), 1):
                lf = (np.abs(zf) < 2).mean(0) > 0.05
                lb = (np.abs(zb) < 2).mean(0) > 0.05
                r[f"live{li}_fwd"], r[f"live{li}_bwd"], r[f"live{li}_both"] = int(lf.sum()), int(lb.sum()), int((lf & lb).sum())
                r[f"sat{li}_fwd"] = int(((np.abs(zf) > 4).mean(0) > 0.95).sum())
                r[f"sat{li}_bwd"] = int(((np.abs(zb) > 4).mean(0) > 0.95).sum())
            flat = np.concatenate([p[k].ravel() for k in AK])
            for ref_name, q in (("average", avg), ("theta0", t0), ("fwd", A), ("bwd", B)):
                r[f"frac_changed_vs_{ref_name}"] = float(np.mean(np.abs(flat - np.concatenate([q[k].ravel() for k in AK])) > 1e-6))
            dists = {k: float(np.linalg.norm(flat - np.concatenate([q[kk].ravel() for kk in AK]))) for k, q in ck.items()}
            nk = min(dists, key=dists.get)
            r["nearest_ckpt"], r["nearest_ckpt_dist"] = nk, dists[nk]
            r["max_abs_weight"] = float(np.abs(flat).max())
            R[name] = r
        out[str(seed)] = R
        print(seed, {k: (round(v["label_effect"], 3), round(v["mse_fwd"], 3), round(v["mse_bwd"], 3)) for k, v in R.items()}, flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "mech.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
