"""R6 flight figures from the final fresh held-out evaluation (run after eval_r6_flight.py):
  media/r6_flight_paths.png  : PD+ vs pure PPO vs residual on 3 fresh courses + each arm's worst course
                               (largest learned - PD+ time; the "preserve the failure" clip)
  media/r6_flight_diffs.png  : histogram of per-course time differences (learned - fair baseline), σ=2
  results/r6_flight_style.json : pass style (closest approach, speed at pass, saturation) per flyer

  ../.venv/bin/python r6_flight_figs.py
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flight as F
import analyze_flight as AF
import eval_r6_flight as E
import train_flight_r6 as R6
import pd_plus as P

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(HERE)
    ev = json.load(open("results/r6_flight_eval.json"))
    E.FRESH_COURSE_SEED, E.FRESH_WIND_SEED = ev["course_seed"], ev["wind_seed"]
    C = E.fresh_courses(ev["n_courses"])
    T = np.load("results/r6_flight_times.npz")
    pols = {"PD+ 1x": E.pdp(E.BASES["pd_plus_1x"]), "PD+ 2x": E.pdp(E.BASES["pd_plus_2x"]),
            "pure PPO s0": R6.load_policy("checkpoints/r6/f_pure_s0")[0],
            "residual s0": R6.load_policy("checkpoints/r6/f_res_s0")[0]}
    cols = {"PD+ 1x": "#228833", "PD+ 2x": "#88aa55", "pure PPO s0": "#cc6677", "residual s0": "#4477aa"}
    d_pure = T["wind2__f_pure_s0"] - T["wind2__pd_plus_1x"]
    d_res = T["wind2__f_res_s0"] - T["wind2__pd_plus_2x"]
    worst_pure = int(np.nanargmax(np.where(np.isnan(d_pure), -np.inf, d_pure)))
    worst_res = int(np.nanargmax(np.where(np.isnan(d_res), -np.inf, d_res)))
    show = [0, 1, 2, worst_pure, worst_res]
    titles = ["fresh course 0", "fresh course 1", "fresh course 2",
              f"pure PPO s0 worst vs PD+ (course {worst_pure})", f"residual s0 worst vs PD+ 2x (course {worst_res})"]
    Cs = C[show]
    # the wind RNG is shared across envs, so record the FULL eval set (same wind as the eval) and slice
    recs = {}
    for k, fn in pols.items():
        r = F.run_episodes(C, fn, seed=E.FRESH_WIND_SEED, record=True)
        recs[k] = dict(traj=r["traj"][show], t=r["t"][show], complete=r["complete"][show])
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.4))
    for i, ax in enumerate(axs):
        W = Cs[i]
        for k, r in recs.items():
            ax.plot(r["traj"][i, :, 0], r["traj"][i, :, 1], color=cols[k], lw=1.4,
                    label=f"{k} ({r['t'][i]:.2f}s)" if r["complete"][i] else f"{k} (fail)")
        for j in range(1, F.N_WP + 1):
            ax.add_patch(plt.Circle(W[j], F.WP_R, fill=False, color="k"))
            ax.text(W[j, 0] + 1, W[j, 1] + 0.6, str(j), fontsize=8)
        ax.plot(*W[0], "ks", ms=5)
        ax.set_aspect("equal"); ax.set_xlim(-5, 35); ax.set_ylim(0, 18); ax.legend(fontsize=6.5, loc="upper right")
        ax.set_title(titles[i], fontsize=8.5)
    plt.suptitle("R6 flyers on fresh held-out courses (wind σ=2; each flyer sees the same wind sequence)", fontsize=10)
    plt.tight_layout(); plt.savefig("media/r6_flight_paths.png", dpi=105)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.arange(-1.0, 1.0, 0.05) - 0.025
    ax.hist(d_pure[~np.isnan(d_pure)], bins=bins, alpha=0.6, color="#cc6677", label="pure PPO s0 − PD+ 1x")
    ax.hist(d_res[~np.isnan(d_res)], bins=bins, alpha=0.6, color="#4477aa", label="residual s0 − PD+ 2x")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-course time difference (s); < 0 = learned flyer faster"); ax.set_ylabel("courses")
    ax.set_title(f"paired differences on {ev['n_courses']} fresh courses, wind σ=2 (times are multiples of 0.05 s)", fontsize=8)
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig("media/r6_flight_diffs.png", dpi=110)

    Cst = C[:500]
    style = {k: AF.run(Cst, fn, seed=E.FRESH_WIND_SEED) for k, fn in pols.items()}
    style["worst_courses"] = dict(pure_s0=dict(course=worst_pure, diff_s=float(d_pure[worst_pure])),
                                  residual_s0=dict(course=worst_res, diff_s=float(d_res[worst_res])))
    json.dump(style, open("results/r6_flight_style.json", "w"), indent=1)
    print(json.dumps(style, indent=1))


if __name__ == "__main__":
    main()
