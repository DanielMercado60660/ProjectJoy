"""Print markdown tables from results/eval_<tag>.json (used to fill RESULTS.md).
Usage: ../.venv/bin/python report.py [tag]
"""
import json, sys

tag = sys.argv[1] if len(sys.argv) > 1 else "lizard"
d = json.load(open(f"results/eval_{tag}.json"))
R = d["results"]
print("protocol:", d["protocol"], "\n")
names = list(R)


def f(v, p=4):
    return "n/a" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


print("| damage | model | pre-damage MSE | right after damage | +50 | +100 | +400 | excess error left at +400 | recovered (80%) | recovered & stable | median steps to recover |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for dmg in ["circle", "half", "stripes", "blobs"]:
    for n in names:
        s = R[n][dmg]
        lab = dmg + (" (train)" if dmg == "circle" else " (held-out)")
        print(f"| {lab} | {n} | {f(s['pre_mse_mean'])} | {f(s['post_damage_mse_mean'])} | {f(s['mse@50_mean'])} | "
              f"{f(s['mse@100_mean'])} | {f(s['mse@400_mean'])} ± {f(s['mse@400_std'])} | "
              f"{(s['mse@400_mean']-s['pre_mse_mean'])/(s['post_damage_mse_mean']-s['pre_mse_mean'])*100:.0f}% | {s['recovered_frac']*100:.0f}% | "
              f"{s['recovered_and_stable_frac']*100:.0f}% | {f(s['recovery_steps_median'], 0)} |")
print()
print("Only trials where the seed actually grew (pre-damage MSE < half of an empty canvas):\n")
print("| damage | model | grown | pre | right after damage | +400 | excess error left | recovered (80%) | recovered & stable |")
print("|---|---|---|---|---|---|---|---|---|")
for dmg in ["circle", "half", "stripes", "blobs"]:
    for n in names:
        s = R[n][dmg]; g = s["grown_only"]
        print(f"| {dmg} | {n} | {s['grown_frac']*100:.0f}% | {f(g['pre'])} | {f(g['post'])} | {f(g['mse@400'])} | "
              f"{(g['mse@400']-g['pre'])/(g['post']-g['pre'])*100:.0f}% | {g['recovered_frac']*100:.0f}% | {g['recovered_and_stable_frac']*100:.0f}% |")
print()
print("| model | seeds that grew (by step 200) | MSE@100 | @200 | @500 | @1000 | @2000 | worst trial @2000 |")
print("|---|---|---|---|---|---|---|---|")
for n in names:
    s = R[n]["long"]
    print(f"| {n} | {s['grown_frac@200']*100:.0f}% | " + " | ".join(f(s[f'mse@{t}_mean']) for t in [100, 200, 500, 1000, 2000]) + f" | {f(s['mse@2000_max'])} |")
print()
print("| model | MSE after wound 1 | 3 | 5 | 10 | worst trial after 10 | trials < 0.01 after 10 |")
print("|---|---|---|---|---|---|---|")
for n in names:
    s = R[n]["repeated"]
    c = s["mse_after_each_heal_mean"]
    print(f"| {n} | {f(c[0])} | {f(c[2])} | {f(c[4])} | {f(c[9])} | {f(s['final_mse_max'])} | {s['frac_trials_final_mse_below_0.01']*100:.0f}% |")
