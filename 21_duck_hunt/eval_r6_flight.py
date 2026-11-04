"""R6 flight: final held-out evaluation on a FRESH course set (seeds fixed only after both recipes
and every selection were frozen), learning curves over env steps, paired tests vs PD+.

Two arms, reported separately and never merged:
  pure     : PPO with no classical prior (f_pure_s0..2)  vs  the best 1x-budget PD+ (reviewer PD+)
  residual : learned residual on PD+ (f_res_s0..2)       vs  PD+ with 2x tuning budget (pd_plus_r6 cont)
             (its fair baseline, because it builds on PD+'s budget); vs 1x PD+ for context only.

  ../bin/gpu_run ../.venv/bin/python eval_r6_flight.py
      -> results/r6_flight_eval.json, results/r6_flight_times.npz, media/r6_flight_curve.png

Paired test: per course, time(learned) - time(baseline) on courses both completed; mean, fraction of
courses where the learned flyer is strictly faster / tied / slower, and a 95% bootstrap CI (10,000
resamples of courses). Seed-mean row: each course's learned time is the mean over the 3 seeds
(NaN if any seed failed it, so those courses drop out; completion is reported per seed).
"""
import glob, json, os, re
import numpy as np
import flight as F
import pd_plus as P
import train_flight as TF
import train_flight_r6 as R6

HERE = os.path.dirname(os.path.abspath(__file__))
# Set ONLY after both R6 recipes, the PD+ retunes and every selection were frozen (time in STATUS.md).
FRESH_COURSE_SEED = 1104_1713     # set 2026-11-04 17:13 CST, after both recipes were frozen
FRESH_WIND_SEED = 1713
N_COURSES = 2000
CURVE_N = 500
CONDS = [("wind2", 2.0), ("wind4", 4.0)]
ARMS = {"pure": (["f_pure_s0", "f_pure_s1", "f_pure_s2"], "pd_plus_1x", []),
        "residual": (["f_res_s0", "f_res_s1", "f_res_s2"], "pd_plus_2x", ["pd_plus_1x"])}
BASES = {"pd_plus_1x": "checkpoints/pd_plus_params.json", "pd_plus_2x": "checkpoints/pd_plus_r6_cont_params.json"}


def fresh_courses(n):
    assert FRESH_COURSE_SEED is not None, "fresh seeds are set only after the recipes are frozen"
    return F.make_courses(np.random.default_rng(FRESH_COURSE_SEED), n)


def paired(t_a, t_b, n_boot=10_000, seed=0):
    both = ~np.isnan(t_a) & ~np.isnan(t_b)
    d = t_a[both] - t_b[both]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return dict(n=int(both.sum()), mean_learned=float(t_a[both].mean()), mean_base=float(t_b[both].mean()),
                mean_diff_learned_minus_base=float(d.mean()),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                learned_faster_frac=float((d < -1e-9).mean()), tie_frac=float((np.abs(d) <= 1e-9).mean()),
                base_faster_frac=float((d > 1e-9).mean()))


def run_set(C, fn, wind):
    return F.run_episodes(C, fn, seed=FRESH_WIND_SEED, wind_sigma=wind)


def pdp(path):
    prm = json.load(open(path))
    return lambda e, o: P.pd_plus_action(e, prm)


def main():
    os.chdir(HERE)
    C = fresh_courses(N_COURSES)
    pols = {k: pdp(v) for k, v in BASES.items()}
    pols["ppo_r0_original"] = TF.policy_fn(TF.load("checkpoints/flight_ppo"))
    for runs, _, _ in ARMS.values():
        for rn in runs:
            pols[rn], _ = R6.load_policy(os.path.join("checkpoints", "r6", rn), "best.npz")
    out = dict(course_seed=FRESH_COURSE_SEED, wind_seed=FRESH_WIND_SEED, n_courses=N_COURSES, bases=BASES,
               arms={k: dict(runs=v[0], fair_base=v[1], context_bases=v[2]) for k, v in ARMS.items()},
               best_steps={rn: json.load(open(f"checkpoints/r6/{rn}/best.json"))["step"]
                           for runs, _, _ in ARMS.values() for rn in runs})
    times = {}
    for cond, wind in CONDS:
        res = {}; T = {}
        for name, fn in pols.items():
            r = run_set(C, fn, wind)
            res[name] = F.summarize(r); T[name] = r["t"]
        times[cond] = T
        cmp = {}
        for arm, (runs, base, ctx) in ARMS.items():
            for b in [base] + ctx:
                for rn in runs:
                    cmp[f"{rn}_vs_{b}"] = paired(T[rn], T[b])
                cmp[f"{arm}_seedmean_vs_{b}"] = paired(np.mean([T[rn] for rn in runs], 0), T[b])
        cmp["pd_plus_2x_vs_pd_plus_1x"] = paired(T["pd_plus_2x"], T["pd_plus_1x"])
        cmp["ppo_r0_original_vs_pd_plus_1x"] = paired(T["ppo_r0_original"], T["pd_plus_1x"])
        out[cond] = dict(summary=res, paired=cmp)
        print(cond, json.dumps(out[cond], indent=1), flush=True)
    # learning curves: every snapshot of every run on the first CURVE_N fresh courses, wind sigma 2
    Cc = C[:CURVE_N]
    curve = {}
    for runs, _, _ in ARMS.values():
        for rn in runs:
            curve[rn] = []
            for sp in sorted(glob.glob(f"checkpoints/r6/{rn}/snap_*M.npz")):
                fn, _ = R6.load_policy(os.path.join("checkpoints", "r6", rn), os.path.basename(sp))
                curve[rn].append(dict(steps=int(re.findall(r"snap_(\d+)M", sp)[0]) * 1_000_000,
                                      **F.summarize(run_set(Cc, fn, 2.0))))
            print(rn, [(p["steps"] // 1_000_000, round(p["mean_time_completed"], 3), p["completion"]) for p in curve[rn]], flush=True)
    out["curve_wind2"] = dict(runs=curve, n=CURVE_N,
                              bases={b: F.summarize(run_set(Cc, pols[b], 2.0)) for b in BASES})
    json.dump(out, open("results/r6_flight_eval.json", "w"), indent=1)
    np.savez("results/r6_flight_times.npz", **{f"{c}__{k}": v for c in times for k, v in times[c].items()})
    plot_curve(out)


def plot_curve(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv = out["curve_wind2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = {"pure": "#cc6677", "residual": "#4477aa"}
    for arm, a in out["arms"].items():
        for i, rn in enumerate(a["runs"]):
            pts = cv["runs"][rn]
            s = [p["steps"] / 1e6 for p in pts]
            t = [p["mean_time_completed"] if p["completion"] >= 0.998 else np.nan for p in pts]
            ax.plot(s, t, marker="o", ms=2.5, color=cols[arm], alpha=0.85, label=f"{arm} PPO" if i == 0 else None)
    ax.axhline(cv["bases"]["pd_plus_1x"]["mean_time_completed"], color="k", ls="--", lw=1, label="PD+ 1x budget (≤2.3e8 steps)")
    ax.axhline(cv["bases"]["pd_plus_2x"]["mean_time_completed"], color="k", ls=":", lw=1, label="PD+ 2x budget")
    ax.set_xlabel("PPO env steps (millions)"); ax.set_ylabel("mean time, fresh held-out courses (s)")
    ax.set_title(f"held-out time vs env steps (wind σ=2, {cv['n']} fresh courses; points with <99.8% completion hidden)", fontsize=8)
    ax.set_ylim(6.0, 7.2); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("media/r6_flight_curve.png", dpi=110)


if __name__ == "__main__":
    main()
