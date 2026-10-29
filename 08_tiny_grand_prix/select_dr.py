"""Offline checkpoint (re-)selection for the grip-randomized runs.

The trainers pick `<tag>_best.npz` on grip-1.0 validation (128 validation tracks,
train-generator seeds 5000-5127). For a policy trained with grip ~ U(0.5, 1.1) that
is a mismatch. This script reads `<tag>_snapshots.npz` (written with --snapshots,
one snapshot per validation step) and scores every snapshot twice:

  val_g1  the trainers' own validation (60 s drives on the 128 validation tracks,
          fixed start points, grip 1.0), read from logs/<tag>_history.jsonl; self-check:
          its argmax snapshot must be bit-identical to the trainer's <tag>_best.npz
  val_rg  the same 128 validation tracks and start points, each driven at K grips
          drawn once from U(grip_lo, grip_hi) with a fixed RNG (same draws for every
          snapshot and every seed, so the comparison is paired)

It writes `<tag>_rg_best.npz` (the val_rg argmax; load with `eval.py --controllers
<tag>_rg`) and `results/dr_seeds/selection.json`. Held-out and hard tracks are not
touched here.

    ../bin/gpu_run ../.venv/bin/python select_dr.py ppo_dr_s1 ppo_dr_s2 ... [--extra ppo_dr es_dr]
"""
import argparse
import json
import os

import numpy as np

from racing.policy import MLPController, forward_pop, load_policy, save_policy
from racing.rollout import episodes, score
from racing.track import make_tracks, pack

VAL_START, N_VAL, K_GRIPS, GRIP_SEED = 5000, 128, 4, 777
CK = "checkpoints"


def val_sets(grip_lo=0.5, grip_hi=1.1):
    bank = pack(make_tracks("train", N_VAL, start=VAL_START))
    starts = np.random.default_rng(12345).random(N_VAL)  # same start points as the trainers
    grips = np.random.default_rng(GRIP_SEED).uniform(grip_lo, grip_hi, (K_GRIPS, N_VAL))
    return bank, starts, grips


def val_g1(theta, bank, starts):
    prog, cr = episodes(MLPController(theta), bank, np.arange(N_VAL), starts, 600, grip=1.0)
    return float(score(prog, cr).mean())


def val_rg(theta, bank, starts, grips):
    tids = np.tile(np.arange(N_VAL), K_GRIPS)
    prog, cr = episodes(MLPController(theta), bank, tids, np.tile(starts, K_GRIPS), 600, grip=grips.ravel())
    return float(score(prog, cr).mean()), float(cr.mean())


def val_rg_batch(thetas, bank, starts, grips, chunk=24):
    """val_rg for many snapshots at once (population forward pass; identical protocol per snapshot)."""
    from racing.sim import OBS_DIM, VecRace
    K = grips.shape[0] * N_VAL
    tids = np.tile(np.arange(N_VAL), grips.shape[0])
    st = np.tile(starts, grips.shape[0])
    out = []
    for c in range(0, len(thetas), chunk):
        th = np.asarray(thetas[c:c + chunk], np.float32)
        M = len(th)
        env = VecRace(bank, M * K)
        obs = env.reset(np.tile(tids, M), start_frac=np.tile(st, M), grip=np.tile(grips.ravel(), M))
        for _ in range(600):
            obs, _, done = env.step(forward_pop(th, obs.reshape(M, K, OBS_DIM)).reshape(M * K, 2))
            if done.all():
                break
        s = score(env.prog, env.crashed).reshape(M, K)
        out += [(float(a), float(b)) for a, b in zip(s.mean(1), env.crashed.reshape(M, K).mean(1))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="runs with <tag>_snapshots.npz")
    ap.add_argument("--extra", nargs="*", default=[], help="existing <name>_best.npz to score on val_rg only")
    ap.add_argument("--out", default="results/dr_seeds/selection.json")
    a = ap.parse_args()
    bank, starts, grips = val_sets()
    out = dict(protocol=dict(val_tracks="train generator seeds 5000-5127", starts="rng(12345)",
                             grips=f"U(0.5, 1.1), {K_GRIPS} draws per track, rng({GRIP_SEED})",
                             episodes_val_rg=K_GRIPS * N_VAL, steps=600), runs={}, extra={})
    for tag in a.tags:
        sn = np.load(os.path.join(CK, f"{tag}_snapshots.npz"))
        step_key = "update" if "update" in sn else "gen"
        logged = {r[step_key]: r["val"] for r in map(json.loads, open(f"logs/{tag}_history.jsonl")) if "val" in r}
        rgs = val_rg_batch(sn["theta"], bank, starts, grips)
        rows = []
        for i, (rg, rgc) in enumerate(rgs):
            rows.append(dict(i=i, step=int(sn[step_key][i]), env_steps=int(sn["env_steps"][i]),
                             val_g1=float(logged[int(sn[step_key][i])]), val_rg=rg, val_rg_crash=rgc))
        b1 = max(rows, key=lambda r: r["val_g1"])
        brg = max(rows, key=lambda r: r["val_rg"])
        th_best, meta_best = load_policy(os.path.join(CK, f"{tag}_best.npz"))
        same = bool(np.array_equal(th_best, sn["theta"][b1["i"]]))
        save_policy(os.path.join(CK, f"{tag}_rg_best.npz"), sn["theta"][brg["i"]],
                    dict(method=meta_best["method"], selection="val_rg", **{step_key: brg["step"]},
                         env_steps=brg["env_steps"], val_rg=brg["val_rg"], val_g1=brg["val_g1"],
                         cfg=meta_best.get("cfg")))
        out["runs"][tag] = dict(step_key=step_key, n_snapshots=len(rows), best_g1=b1, best_rg=brg,
                                g1_argmax_matches_trainer_best=same, trainer_best_val=meta_best.get("val"),
                                history=rows)
        print(f"{tag}: {len(rows)} snapshots | g1 rule: {step_key} {b1['step']} ({b1['env_steps']/1e6:.1f} M) "
              f"val_g1 {b1['val_g1']:.1f} val_rg {b1['val_rg']:.1f} [matches trainer best: {same}] | "
              f"rg rule: {step_key} {brg['step']} ({brg['env_steps']/1e6:.1f} M) val_g1 {brg['val_g1']:.1f} "
              f"val_rg {brg['val_rg']:.1f}", flush=True)
    for name in a.extra:
        th, meta = load_policy(os.path.join(CK, f"{name}_best.npz"))
        (rg, rgc), = val_rg_batch(th[None], bank, starts, grips)
        out["extra"][name] = dict(val_g1=val_g1(th, bank, starts), val_rg=rg, val_rg_crash=rgc,
                                  meta={k: v for k, v in meta.items() if k != "cfg"})
        print(f"{name}: val_g1 {out['extra'][name]['val_g1']:.1f} val_rg {rg:.1f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prev.setdefault("runs", {}).update(out["runs"])
    prev.setdefault("extra", {}).update(out["extra"])
    prev["protocol"] = out["protocol"]
    json.dump(prev, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
