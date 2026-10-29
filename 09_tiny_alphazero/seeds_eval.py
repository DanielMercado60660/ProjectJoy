"""Seed-replication evaluation (RESULTS "Seeds" section).

Same protocol as evaluate.py (98 games = 49 two-ply openings x both colours, net + 100 sims,
no noise, match seed 0), but cached in results/seeds/matches.json so the seed-0 match cache,
elo.json and report.md stay exactly as they were. Seed-0 results already cached by evaluate.py
are read (not recomputed) for the tables.

  ../.venv/bin/python seeds_eval.py suite --runs seed1,seed2 --gens 100,136
  ../.venv/bin/python seeds_eval.py h2h --pairs seed1/170:seed1/136,seed1/170:main/170
  ../.venv/bin/python seeds_eval.py table --runs main,seed1,seed2 --gens 100,136,170
"""
import argparse
import json
import os
import time

import numpy as np

import evaluate as E

E.RES = os.path.join(E.HERE, "results", "seeds")
E.CACHE = os.path.join(E.RES, "matches.json")
MAIN_RES = os.path.join(E.HERE, "results")


def load_all():
    """seed-0 cache (results/matches_*.json, read-only) + this module's cache."""
    out = {}
    for fn in sorted(os.listdir(MAIN_RES)):
        if fn.startswith("matches") and fn.endswith(".json"):
            with open(os.path.join(MAIN_RES, fn)) as f:
                out.update(json.load(f))
    own = {}
    if os.path.exists(E.CACHE):
        with open(E.CACHE) as f:
            own = json.load(f)
    return out, own


class Cache(dict):
    """run_match() reads from everything but only this module's own entries get saved."""
    def __init__(self, main, own):
        super().__init__({**main, **own})
        self.own = dict(own)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.own[k] = v


def save(c):
    os.makedirs(E.RES, exist_ok=True)
    tmp = E.CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c.own, f, indent=0)
    os.replace(tmp, E.CACHE)


def spec(rg, sims=E.EVAL_SIMS):
    run, g = rg.split("/")
    return E.net_spec(run, int(g), sims)


def cmd_suite(a, cache):
    for g in [int(x) for x in a.gens.split(",")]:
        for run in a.runs.split(","):
            for opp in a.opps.split(","):
                E.run_match(cache, E.net_spec(run, g), opp)
                save(cache)


def cmd_h2h(a, cache):
    for p in a.pairs.split(","):
        x, y = p.split(":")
        E.run_match(cache, spec(x), spec(y))
        save(cache)


def se(p, n=98):
    return float(np.sqrt(max(p * (1 - p), 1e-9) / n))


def cmd_table(a, cache):
    opps = a.opps.split(",")
    print("| run | gen | " + " | ".join(opps) + " | mean |")
    print("|---|---|" + "---|" * (len(opps) + 1))
    for g in [int(x) for x in a.gens.split(",")]:
        for run in a.runs.split(","):
            cells, vals = [], []
            for opp in opps:
                r = cache.get(f"{E.net_spec(run, g)} | {opp}")
                if r is None:
                    cells.append("–")
                    continue
                cells.append(f"{r['score']:.2f} ± {se(r['score']):.2f}")
                vals.append(r["score"])
            m = f"{np.mean(vals):.3f}" if len(vals) == len(opps) else "–"
            print(f"| {run} | {g} | " + " | ".join(cells) + f" | {m} |")


def cmd_logs(a, cache):
    """Self-play statistics per training-log window (the collapse signature) and inline evals."""
    wins = [(41, 60), (61, 70), (71, 80), (91, 100), (127, 136), (161, 170)]
    print("| run | " + " | ".join(f"gens {x}–{y}: 1st-win / draws / value loss" for x, y in wins) + " |")
    for run in a.runs.split(","):
        p = os.path.join(E.HERE, "logs", f"{run}.jsonl")
        L = [json.loads(l) for l in open(p)]
        cells = []
        for x, y in wins:
            w = [r for r in L if x <= r["gen"] <= y]
            cells.append("–" if not w else "{:.2f} / {:.2f} / {:.2f}".format(
                *[np.mean([r[k] for r in w]) for k in ("first_win", "draws", "value_loss")]))
        hi = [r["gen"] for r in L if r["gen"] <= 100 and r["first_win"] >= 0.85]
        print(f"| {run} | " + " | ".join(cells) + f" | first gen with 1st-win ≥ 0.85: {hi[0] if hi else 'none'}")
    print()
    for run in a.runs.split(","):
        p = os.path.join(E.HERE, "logs", f"{run}_eval.jsonl")
        if os.path.exists(p):
            L = {json.loads(l)["gen"]: json.loads(l) for l in open(p)}
            print(run, " ".join(f"g{g}:{L[g]['ab4']:.2f}/{L[g]['uct1000']:.2f}" for g in sorted(L) if g in
                                (5, 10, 20, 30, 40, 70, 100, 136, 150, 170)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["suite", "h2h", "table", "logs"])
    ap.add_argument("--runs", default="seed1,seed2")
    ap.add_argument("--gens", default="100")
    ap.add_argument("--opps", default=",".join(E.SUITE))
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    main_c, own = load_all()
    cache = Cache(main_c, own)
    E.save_cache = lambda c: save(cache)
    t = time.time()
    {"suite": cmd_suite, "h2h": cmd_h2h, "table": cmd_table, "logs": cmd_logs}[a.cmd](a, cache)
    if a.cmd in ("suite", "h2h"):
        print(f"done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
