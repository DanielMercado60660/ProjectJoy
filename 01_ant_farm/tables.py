"""Print the RESULTS.md tables (markdown) from results/eval*.json, so numbers are never retyped."""
import json, os

R = {}
for f in ["results/eval.json", "results/eval_B.json", "results/eval_C.json", "results/eval_D.json"]:
    if os.path.exists(f):
        for k, v in json.load(open(f)).items():
            R.setdefault(k, {}).update(v)
NAMES = {"random": "random walkers", "classic": "hand-written classic (no compass)",
         "compass": "hand-written + compass", "ppo1": "**PPO** (shared policy)", "ppo1_seed1": "PPO, 2nd seed",
         "ppo_nocompass": "PPO, no compass", "es1": "ES (same network)",
         "ppo2": "PPO continued, 3x starvation penalty"}
ORDER = ["random", "classic", "compass", "ppo1", "ppo1_seed1", "ppo2", "ppo_nocompass", "es1"]
E = R.get("efficiency", {})
for sset, title in [("indist", "Unseen worlds from the training family (seeds 0-31)"),
                    ("heldout", "Held-out family: food 24-30 cells away, long walls (seeds 50000-50031)")]:
    print(f"\n**{title}**, 32 colonies × 64 ants × 1000 steps\n")
    print("| colony | food delivered (mean ± sd) | food / energy | ants alive at end | colonies ≥50% alive | delivered without pheromone |")
    print("|---|---|---|---|---|---|")
    for p in ORDER:
        k = f"{sset}/{p}"
        if k not in E:
            continue
        e = E[k]
        nop = E.get(k + "/no_pheromone")
        nops = f"{nop['delivered_mean']:.0f}" if nop else "n/a"
        print(f"| {NAMES[p]} | {e['delivered_mean']:.0f} ± {e['delivered_std']:.0f} | {e['food_per_energy']:.2f} | "
              f"{e['ants_alive_frac']:.2f} | {e['colonies_survived']:.2f} | {nops} |")
REC = R.get("recovery", {})
if REC:
    print("\n**Route-block recovery** (train-family seeds 0-31, abundant food, wall at step 500)\n")
    print("| colony | pre-block rate (food/step) | rate in first 100 steps after (rel.) | rate in last 200 steps (rel.) | t to 50% | t to 80% | worlds back to 50% | median t50 in those worlds | ants alive at 1500 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in ORDER:
        if p not in REC:
            continue
        r = REC[p]
        f = lambda v: "never (in 1000 steps)" if v is None else str(v)
        print(f"| {NAMES[p]} | {r['pre_block_rate']:.2f} | {r['rate_first_100_after_block_rel']:.2f} | {r['rate_last_200_rel']:.2f} | "
              f"{f(r['t_recover_50'])} | {f(r['t_recover_80'])} | {r['worlds_recovered_50_frac']:.2f} | "
              f"{'n/a' if r['worlds_t50_median'] is None else int(r['worlds_t50_median'])} | {r['ants_alive_frac']:.2f} |")
