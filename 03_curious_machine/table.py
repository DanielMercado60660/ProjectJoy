"""Print markdown tables for RESULTS.md from runs/eval_*.json and runs/*/episodes.jsonl."""
import glob
import json
import sys

import numpy as np

from env import INTER_NAMES

AG = ["random", "count", "rnd", "rnd_ep", "ep", "ngu", "disagree"]
LAB = {"random": "random walk", "count": "PPO + hashed counts", "rnd": "PPO + RND", "rnd_ep": "PPO + RND × episodic count",
       "ep": "PPO + episodic count only (ablation)", "ngu": "PPO + NGU-style (inverse-dynamics kNN)", "disagree": "PPO + ensemble disagreement"}


def evals(ag, env):
    return [json.load(open(p)) for p in sorted(glob.glob(f"runs/eval_{ag}_{env}_s*.json"))]


def ms(v, d=2):
    v = np.array(v, float)
    return f"{v.mean():.{d}f} ± {v.std():.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def main(env):
    print(f"| agent ({env}) | seeds | rooms /9 | interactions /8 | coverage /233 | unproductive steps /512 | repeat interactions | TV steps /512 |")
    print("|---|---|---|---|---|---|---|---|")
    for ag in AG:
        ev = evals(ag, env)
        if not ev:
            continue
        f = lambda k: ms([e[k]["mean"] for e in ev], 1 if k in ("coverage", "unproductive", "tv_steps") else 2)
        print(f"| {LAB[ag]} | {len(ev)} | {f('rooms')} | {f('interactions')} | {f('coverage')} | {f('unproductive')} | {f('repeat_interactions')} | {f('tv_steps')} |")
    print()
    print(f"| % of eval episodes ({env}) | " + " | ".join(INTER_NAMES) + " |")
    print("|---|" + "---|" * 8)
    for ag in AG:
        ev = evals(ag, env)
        if not ev:
            continue
        r = np.mean([[e["interaction_rate"][n] for n in INTER_NAMES] for e in ev], 0)
        print(f"| {LAB[ag]} | " + " | ".join(f"{100 * x:.0f}" for x in r) + " |")
    print()
    print(f"| whole training budget ({env}, 6 M steps/seed) | episodes | key picked up | door unlocked | button E | max rooms in any episode |")
    print("|---|---|---|---|---|---|")
    for ag in AG:
        runs = []
        for d in sorted(glob.glob(f"runs/{ag}_{env}_s*/episodes.jsonl")):
            runs.append([json.loads(l) for l in open(d)])
        if not runs:
            continue
        it = [np.array([e["inter_t"] for e in eps]) >= 0 for eps in runs]
        tot = sum(len(e) for e in runs)
        g = lambda k: sum(int(x[:, k].sum()) for x in it)
        mr = max(max(e["rooms"] for e in eps) for eps in runs)
        print(f"| {LAB[ag]} ({len(runs)} seeds) | {tot} | {g(3)} | {g(4)} | {g(5)} | {mr} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "base")
