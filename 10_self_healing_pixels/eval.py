"""Damage / recovery / long-rollout evaluation for trained NCAs (same protocol, seeds and masks for every model).

Protocol (per damage type, K trials batched):
  1. grow from the seed for GROW steps (stochastic updates, fixed MLX seed)
  2. erase cells with a damage mask (identical masks across models; numpy seed fixed per type)
  3. run REGEN more steps, recording RGBA MSE vs the target after every step
Recovery time = first step after damage where MSE <= tau, tau = pre + 0.2 * (post - pre), i.e. 80% of the
damage-induced excess error has been removed (per trial, so each model is judged against its own undamaged
quality). "Stable" additionally requires MSE <= tau at the end of the 400-step window. Non-recoveries are counted.
Long rollout: from the seed, no damage, MSE at steps 100/200/500/1000/2000 (and max over the run).

Usage: ../.venv/bin/python eval.py --models lizard_persist lizard_regen [--trials 16]
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx

import nca
import damage

GROW, REGEN = 200, 400
HEAL_FRAC = 0.8  # recovered = 80% of the damage-induced excess error removed
LONG = 2000
LONG_MARKS = [100, 200, 500, 1000, 2000]


def mse(x, target):
    x = np.clip(np.nan_to_num(np.asarray(x, np.float64)[..., :4], nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    return ((x - target) ** 2).mean(axis=(1, 2, 3))


def run_damage(model, target, dname, K, seed, keep_states=False):
    H, W = target.shape[:2]
    mx.random.seed(seed)
    x = mx.array(nca.make_seed(H, W, K))
    for _ in range(GROW):
        x = model(x)
    mx.eval(x)
    xn = np.array(x)
    pre = mse(xn, target)
    rng = np.random.default_rng(1000 + seed)
    masks = np.stack([damage.ALL_DAMAGE[dname](H, W, rng) for _ in range(K)])
    for k in range(K):
        xn[k] = damage.apply(xn[k], masks[k])
    curve = [mse(xn, target)]
    x = mx.array(xn)
    states = [xn] if keep_states else None
    for t in range(REGEN):
        x = model(x)
        if t % 10 == 9:
            mx.eval(x)
        xs = np.array(x)
        curve.append(mse(xs, target))
        if keep_states:
            states.append(xs)
    curve = np.stack(curve)  # (REGEN+1, K)
    tau = pre + (1 - HEAL_FRAC) * (curve[0] - pre)
    rec = np.full(K, -1)
    for k in range(K):
        hit = np.where(curve[:, k] <= tau[k])[0]
        if len(hit):
            rec[k] = hit[0]
    erased_alive = float((masks & (target[..., 3] > 0.1)[None]).sum(axis=(1, 2)).mean() / (target[..., 3] > 0.1).sum())
    return dict(pre=pre, curve=curve, rec=rec, tau=tau, frac_body_erased=erased_alive), states


def run_long(model, target, K, seed):
    H, W = target.shape[:2]
    mx.random.seed(seed)
    x = mx.array(nca.make_seed(H, W, K))
    curve = []
    for t in range(1, LONG + 1):
        x = model(x)
        if t % 10 == 0:
            mx.eval(x)
            curve.append(mse(np.array(x), target))
    return np.stack(curve)  # (LONG/10, K), entry i = step 10*(i+1)


def run_repeated(model, target, K, seed, rounds=10, every=150):
    """Grow 200 steps, then every `every` steps apply a random held-out damage (half/stripes/blobs).
    Returns MSE right before each new wound (i.e. how well it healed the previous one), shape (rounds, K)."""
    H, W = target.shape[:2]
    mx.random.seed(seed)
    rng = np.random.default_rng(2000 + seed)
    x = model.rollout(mx.array(nca.make_seed(H, W, K)), GROW)
    fns = list(damage.HELDOUT_DAMAGE.values())
    out = []
    for r in range(rounds):
        xn = np.array(x)
        for k in range(K):
            xn[k] = damage.apply(xn[k], fns[rng.integers(len(fns))](H, W, rng))
        x = mx.array(xn)
        for t in range(every):
            x = model(x)
            if t % 10 == 9:
                mx.eval(x)
        out.append(mse(np.array(x), target))
    return np.stack(out)


def summarise(r, blank):
    rec = r["rec"]
    ok = rec[rec >= 0]
    c = r["curve"]
    grown = r["pre"] < 0.5 * blank  # trials where the seed actually grew a creature before the damage
    g = grown if grown.any() else np.ones_like(grown)
    s = {
        "grown_frac": float(grown.mean()),
        "grown_only": {"pre": float(r["pre"][g].mean()), "post": float(c[0][g].mean()), "mse@400": float(c[-1][g].mean()),
                       "recovered_frac": float((rec[g] >= 0).mean()),
                       "recovered_and_stable_frac": float(((rec >= 0) & (c[-1] <= r["tau"]))[g].mean())},
        "pre_mse_mean": float(r["pre"].mean()),
        "post_damage_mse_mean": float(c[0].mean()),
        "frac_body_erased": r["frac_body_erased"],
        **{f"mse@{t}_mean": float(c[t].mean()) for t in (50, 100, 200, 400)},
        **{f"mse@{t}_std": float(c[t].std()) for t in (100, 400)},
        "recovered_frac": float((rec >= 0).mean()),
        "recovered_and_stable_frac": float(((rec >= 0) & (c[-1] <= r["tau"])).mean()),
        "recovery_steps_median": float(np.median(ok)) if len(ok) else None,
        "recovery_steps_mean": float(ok.mean()) if len(ok) else None,
    }
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="names under checkpoints/")
    p.add_argument("--target", default="lizard")
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    target = nca.load_target(a.target)
    blank = float((target ** 2).mean())  # MSE of an empty canvas
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or a.target
    res, curves = {}, {}
    for path in a.models:
        t0 = time.time()
        d = path if os.path.isdir(path) else os.path.join("checkpoints", path)
        name = os.path.basename(os.path.normpath(d))
        model = nca.load_model(os.path.join(d, "model.safetensors"))
        res[name], curves[name] = {}, {}
        for i, d in enumerate(damage.ALL_DAMAGE):
            r, _ = run_damage(model, target, d, a.trials, a.seed + i)
            res[name][d] = summarise(r, blank)
            curves[name][d] = r["curve"].mean(1).tolist()
        lc = run_long(model, target, a.trials, a.seed + 99)
        res[name]["long"] = {f"mse@{t}_mean": float(lc[t // 10 - 1].mean()) for t in LONG_MARKS}
        res[name]["long"].update({f"mse@{t}_max": float(lc[t // 10 - 1].max()) for t in LONG_MARKS})
        res[name]["long"]["max_over_run"] = float(lc.max())
        res[name]["long"]["grown_frac@200"] = float((lc[19] < 0.5 * blank).mean())
        curves[name]["long"] = lc.mean(1).tolist()
        rp = run_repeated(model, target, a.trials, a.seed + 77)
        curves[name]["repeated"] = rp.mean(1).tolist()
        res[name]["repeated"] = {"mse_after_each_heal_mean": rp.mean(1).tolist(),
                                 "final_mse_mean": float(rp[-1].mean()), "final_mse_max": float(rp[-1].max()),
                                 "frac_trials_final_mse_below_0.01": float((rp[-1] < 0.01).mean())}
        print(f"{name}: {time.time()-t0:.0f}s", flush=True)
        for d, s in res[name].items():
            print(f"  {d:8s} " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()
                                          if not isinstance(v, (list, dict))), flush=True)
    meta = {"blank_canvas_mse": blank, "protocol": {"grow_steps": GROW, "regen_steps": REGEN, "tau": "pre + 0.2*(post_damage - pre), per trial",
                         "trials": a.trials, "seed": a.seed, "train_damage": list(damage.TRAIN_DAMAGE),
                         "heldout_damage": list(damage.HELDOUT_DAMAGE),
                         "repeated": "after 200 growth steps, 10 rounds of (random held-out damage, 150 steps)"}}
    json.dump({**meta, "results": res}, open(os.path.join(a.out, f"eval_{tag}.json"), "w"), indent=1)
    json.dump(curves, open(os.path.join(a.out, f"curves_{tag}.json"), "w"))
    plot(curves, os.path.join("media", f"recovery_curves_{tag}.png"))


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = list(damage.ALL_DAMAGE) + ["long"] + (["repeated"] if "repeated" in next(iter(curves.values())) else [])
    fig, axs = plt.subplots(1, len(ds), figsize=(4 * len(ds), 3.4))
    for ax, d in zip(axs, ds):
        for name, c in curves.items():
            y = np.array(c[d])
            xs = {"long": 10 * (np.arange(len(y)) + 1), "repeated": np.arange(1, len(y) + 1)}.get(d, np.arange(len(y)))
            ax.semilogy(xs, y, label=name, marker="o" if d == "repeated" else None)
        held = "" if d in ("circle", "long", "repeated") else " (held-out)"
        ax.set_title({"long": "long rollout, no damage", "repeated": "repeated held-out wounds"}.get(d, f"damage: {d}{held}"))
        ax.set_xlabel({"long": "steps from seed", "repeated": "wounds so far (MSE 150 steps after each)"}.get(d, "steps after damage"))
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("RGBA MSE vs target (mean of trials)")
    axs[0].legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
