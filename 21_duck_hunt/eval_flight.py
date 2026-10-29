"""Evaluate learned flyer (PPO) vs tuned PD/pursuit baseline on HELD-OUT courses.
  ../.venv/bin/python eval_flight.py            -> results/flight_eval.json, media/flight_heldout.png
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flight as F
import train_flight as TF


def main(n=500, ckpt="checkpoints/flight_ppo", out="results/flight_eval.json"):
    model = TF.load(ckpt)
    prm = json.load(open("checkpoints/pd_params.json"))
    pols = {"ppo": TF.policy_fn(model), "pd_tuned": lambda e, o: F.pd_action(e, prm),
            "pd_default": lambda e, o: F.pd_action(e, F.PD_DEFAULT)}
    C = F.heldout_courses(n)
    res = {}
    for cond, wind in [("heldout_wind2", 2.0), ("heldout_wind4_stress", 4.0), ("heldout_calm", 0.0)]:
        res[cond] = {}
        for name, fn in pols.items():
            r = F.run_episodes(C, fn, seed=99, wind_sigma=wind)
            res[cond][name] = F.summarize(r)
            # paired comparison of time on courses both completed
            res[cond][name]["_t"] = r["t"]
        tp, tb = res[cond]["ppo"].pop("_t"), res[cond]["pd_tuned"].pop("_t")
        res[cond]["pd_default"].pop("_t")
        both = ~np.isnan(tp) & ~np.isnan(tb)
        res[cond]["paired_time_both_completed"] = dict(
            n=int(both.sum()), ppo_mean=float(tp[both].mean()), pd_tuned_mean=float(tb[both].mean()),
            ppo_faster_frac=float((tp[both] < tb[both]).mean()))
        print(cond, json.dumps(res[cond], indent=1))
    Ct = F.train_courses(256)
    res["train_courses_wind2"] = {k: F.summarize(F.run_episodes(Ct, fn, seed=11)) for k, fn in pols.items()}
    json.dump(res, open(out, "w"), indent=1)

    # figure: 4 held-out courses
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    rp = F.run_episodes(C[:4], pols["ppo"], seed=99, record=True)
    rb = F.run_episodes(C[:4], pols["pd_tuned"], seed=99, record=True)
    for i, ax in enumerate(axs):
        W = C[i]
        ax.plot(rb["traj"][i, :, 0], rb["traj"][i, :, 1], color="#4477aa", lw=1.5,
                label=f"PD tuned ({rb['t'][i]:.2f}s)" if rb["complete"][i] else "PD tuned (fail)")
        ax.plot(rp["traj"][i, :, 0], rp["traj"][i, :, 1], color="#cc6677", lw=1.5,
                label=f"PPO ({rp['t'][i]:.2f}s)" if rp["complete"][i] else "PPO (fail)")
        for k in range(1, F.N_WP + 1):
            ax.add_patch(plt.Circle(W[k], F.WP_R, fill=False, color="k"))
            ax.text(W[k, 0] + 1, W[k, 1] + 0.6, str(k), fontsize=8)
        ax.plot(*W[0], "ks", ms=5)
        ax.set_aspect("equal"); ax.set_xlim(-5, 35); ax.set_ylim(0, 18); ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"held-out course {i}", fontsize=9)
    plt.tight_layout(); plt.savefig("media/flight_heldout.png", dpi=110)
    return res


if __name__ == "__main__":
    main()
