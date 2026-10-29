"""Evaluation: efficiency, survival, route-block recovery, held-out layouts, pheromone ablation.

  ../bin/gpu_run ../.venv/bin/python evaluate.py --ppo checkpoints/ppo1 --out results/eval.json

Sets (seeds never used in training; training seeds start at 1,000,000):
  indist  : "train" layout family, seeds 0..31
  heldout : "heldout" family (food 24-30 cells away vs 10-22 in training, long thin walls
            instead of rocks), seeds 50,000..50,031
All colonies get the same worlds, the same 64 ants, 1000 steps, and sample their actions from
the world's own RNG stream.
"""
import argparse, json, os, time
import numpy as np
import sim
import handwritten as hw

SETS = {"indist": (np.arange(0, 32), "train"), "heldout": (np.arange(50_000, 50_032), "heldout")}
N_ANTS, T_EP = 64, 1000


def load_policy(path, kind="mlx"):
    from policy import Policy, load_weights, probs, NumpyActor, mask_obs
    cfg = json.load(open(os.path.join(path, "config.json")))
    nc = cfg.get("no_compass", False)
    m = Policy(hid=cfg.get("hid", 128))
    # PPO: the validation-selected checkpoint (seeds 900-915) if it exists, else the last one
    w = os.path.join(path, "model_best.safetensors")
    if not os.path.exists(w) or os.environ.get("ANTFARM_LAST"):
        w = os.path.join(path, "model.safetensors")
    load_weights(m, w)
    if kind == "numpy":
        na = NumpyActor(m)
        return lambda obs: na(mask_obs(obs, nc))
    return lambda obs: probs(m, mask_obs(obs, nc))


def run(world, pol, steps, pheromones_on=True, on_step=None):
    per_step = np.zeros((steps, world.B), np.float32)
    for t in range(steps):
        if on_step is not None:
            on_step(t, world)
        pm, pd = pol(sim.observe(world))
        m, d = sim.sample_actions(world, pm, pd)
        ev = sim.step(world, m, d, pheromones_on=pheromones_on)
        per_step[t] = ev.deliver.sum(1)
    return per_step


def summarize(w, per_step):
    d = w.delivered
    return {"delivered_mean": float(d.mean()), "delivered_std": float(d.std()),
            "food_per_energy": float(d.sum() / w.spent.sum()),
            "energy_mean": float(w.spent.mean()),
            "ants_alive_frac": float(w.alive.mean()),
            "colonies_survived": float((w.alive.mean(1) >= 0.5).mean()),
            "deaths_mean": float(w.deaths.mean()),
            "first_delivery_step_median": float(np.median([np.argmax(c > 0) if c.any() else len(c) for c in per_step.T]))}


def efficiency_eval(pols, sets=("indist", "heldout"), ablate=()):
    out = {}
    for sname in sets:
        seeds, fam = SETS[sname]
        for pname, pol in pols.items():
            for phero in ([True, False] if pname in ablate else [True]):
                w = sim.make_worlds(seeds, N_ANTS, fam)
                t0 = time.time()
                ps = run(w, pol, T_EP, pheromones_on=phero)
                key = f"{sname}/{pname}" + ("" if phero else "/no_pheromone")
                out[key] = summarize(w, ps)
                out[key]["delivery_curve_100"] = ps.sum(1).reshape(-1, 100).sum(1).tolist()
                print(key, {k: round(v, 3) for k, v in out[key].items() if not isinstance(v, list)},
                      f"{time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- route-block recovery
BLOCK_AT, REC_STEPS, PRE_WIN, SMOOTH = 500, 1000, 200, 100


def recovery_eval(pols, seeds=np.arange(0, 32), family="train", food_per_cell=250):
    """Let the colony establish a trail for BLOCK_AT steps, then drop a wall across the
    route to the source that supplied the most pick-ups in the last 200 steps (decided from
    the food grid, so it is the same rule for every colony). Food is made abundant
    (food_per_cell) so depletion can't masquerade as a blocked route.
    Recovery time = steps after the block until the 100-step moving delivery rate is back to
    >=50% / >=80% of the pre-block rate (averaged over worlds). The detour is longer, so 100%
    may be unreachable; switching to the other source counts as recovery."""
    out = {}
    for pname, pol in pols.items():
        w = sim.make_worlds(seeds, N_ANTS, family)
        w.food[w.food > 0] = food_per_cell
        snap = {}

        def hook(t, world):
            if t == BLOCK_AT - PRE_WIN:
                snap["food"] = world.food.copy()
            if t == BLOCK_AT:
                for b in range(world.B):
                    taken = []
                    for (fy, fx) in sim.food_sources(world, b):
                        m = sim._disc(fy, fx, sim.FOOD_R + 1)
                        taken.append(int(snap["food"][b][m].sum() - world.food[b][m].sum()))
                    src = sim.food_sources(world, b)[int(np.argmax(taken))]
                    sim.block_route(world, b, food_xy=src)
        ps = run(w, pol, BLOCK_AT + REC_STEPS, on_step=hook)
        res = recovery_stats(ps)
        res["delivered_total_mean"] = float(w.delivered.mean())
        res["ants_alive_frac"] = float(w.alive.mean())
        out[pname] = res
        print("recovery", pname, {k: v for k, v in res.items() if not isinstance(v, list)}, flush=True)
    return out


def recovery_stats(ps):
    """ps: [steps, B] deliveries per step."""
    k = np.ones(SMOOTH) / SMOOTH
    mean_curve = ps.mean(1)
    pre = mean_curve[BLOCK_AT - PRE_WIN:BLOCK_AT].mean()
    sm = np.convolve(mean_curve, k, mode="full")[:len(mean_curve)]  # trailing window
    post = sm[BLOCK_AT:]
    # first time the trailing window lies entirely after the block and rate >= x*pre
    def t_rec(x):
        ok = np.flatnonzero(post[SMOOTH:] >= x * pre)
        return int(ok[0] + SMOOTH) if len(ok) else None
    dip = float(post[SMOOTH] / max(pre, 1e-9))
    # per-world recovery (50%), for a distribution
    per_world = []
    for b in range(ps.shape[1]):
        c = ps[:, b]
        pre_b = c[BLOCK_AT - PRE_WIN:BLOCK_AT].mean()
        if pre_b <= 0:
            continue
        smb = np.convolve(c, k, mode="full")[:len(c)][BLOCK_AT + SMOOTH:]
        ok = np.flatnonzero(smb >= 0.5 * pre_b)
        per_world.append(int(ok[0] + SMOOTH) if len(ok) else -1)
    pw = np.array(per_world)
    return {"pre_block_rate": float(pre), "rate_first_100_after_block_rel": dip,
            "rate_last_200_rel": float(mean_curve[-200:].mean() / max(pre, 1e-9)),
            "t_recover_50": t_rec(0.5), "t_recover_80": t_rec(0.8),
            "worlds_recovered_50_frac": float((pw >= 0).mean()) if len(pw) else 0.0,
            "worlds_t50_median": float(np.median(pw[pw >= 0])) if (pw >= 0).any() else None,
            "curve": mean_curve.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", nargs="*", default=[], help="checkpoint dirs to evaluate")
    ap.add_argument("--es", nargs="*", default=[])
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--skip_eff", action="store_true")
    ap.add_argument("--skip_rec", action="store_true")
    ap.add_argument("--baselines", nargs="*", default=["random", "classic", "compass"])
    a = ap.parse_args()
    pols = {k: hw.POLICIES[k] for k in a.baselines}
    for p in a.ppo + a.es:
        pols[os.path.basename(p.rstrip("/"))] = load_policy(p)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if not a.skip_eff:
        res.setdefault("efficiency", {}).update(efficiency_eval(pols, ablate=[k for k in pols if k != "random"]))
        json.dump(res, open(a.out, "w"), indent=1)
    if not a.skip_rec:
        res.setdefault("recovery", {}).update(recovery_eval({k: v for k, v in pols.items() if k != "random"}))
        json.dump(res, open(a.out, "w"), indent=1)
