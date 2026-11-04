"""Aggregate wsp/results/heldout.json into the retained-return table (runner: Claude).

retained(task) = mean return of the network on that task / mean return of the oracle switch (the
specialist) on the same bank and training seed; mean retained = average over the two tasks; then mean
± std over the 4 training seeds. 'fails' = held-out episodes (of 20 per task per seed, 160 total) with
return below 50% of that seed's specialist mean.

  ../.venv/bin/python wsp/report_wsp.py      -> wsp/results/table.md, wsp/results/table.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["oracle", "dagger_200k", "dagger_299008", "dagger_73728", "dagger_61440", "dagger_16384", "r6wsp_b50_a", "r6wsp_b50_b",
         "r6wsp_b10_a", "r6wsp_b10_b", "r6wsp_os_a", "r6wsp_os_b", "ablate_route_only", "ablate_route_refit_out",
         "synth_distill", "ppo_ft_avg_1M", "ppo_ft_avg_299008", "ppo_ft_avg_73728", "ppo_ft_avg_61440", "ppo_ft_avg_16384", "fwd_spec_both", "bwd_spec_both",
         "theta0", "average", "task_arith", "ties", "dare", "task_arith_ext", "ties_ext", "dare_ext"]


def main(path=os.path.join(HERE, "results", "heldout.json")):
    R = json.load(open(path))["results"]
    seeds = sorted(R)
    names = [n for n in ORDER if n in R[seeds[0]]] + sorted(set(R[seeds[0]]) - set(ORDER))
    out, lines = {}, []
    for bank in ("heldout", "canonical"):
        lines += [f"\n### bank: {bank}\n", "| network | fwd return | bwd return | fwd retained | bwd retained | mean retained (± std over seeds) | per seed | fails <50% |",
                  "|---|---|---|---|---|---|---|---|"]
        for n in names:
            rf, rb, mr, fr, br, fails, tot = [], [], [], [], [], 0, 0
            for s in seeds:
                o = R[s]["oracle"][bank]
                x = R[s][n][bank]
                f = np.mean(x["fwd"]) / np.mean(o["fwd"])
                b = np.mean(x["bwd"]) / np.mean(o["bwd"])
                rf.append(np.mean(x["fwd"]))
                rb.append(np.mean(x["bwd"]))
                fr.append(f)
                br.append(b)
                mr.append((f + b) / 2)
                for t in ("fwd", "bwd"):
                    fails += int(np.sum(np.array(x[t]) < 0.5 * np.mean(o[t])))
                    tot += len(x[t])
            out.setdefault(bank, {})[n] = dict(fwd=float(np.mean(rf)), bwd=float(np.mean(rb)), fwd_ret=float(np.mean(fr)),
                                               bwd_ret=float(np.mean(br)), mean_ret=float(np.mean(mr)),
                                               mean_ret_std=float(np.std(mr)), per_seed=[float(v) for v in mr],
                                               fails=fails, episodes=tot)
            lines.append(f"| {n} | {np.mean(rf):.0f} ± {np.std(rf):.0f} | {np.mean(rb):.0f} ± {np.std(rb):.0f} | "
                         f"{100 * np.mean(fr):.0f}% | {100 * np.mean(br):.0f}% | **{100 * np.mean(mr):.1f}%** ± {100 * np.std(mr):.1f} | "
                         f"{' / '.join(f'{100 * v:.0f}' for v in mr)} | {fails}/{tot} |")
    open(os.path.join(HERE, "results", "table.md"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(HERE, "results", "table.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
