"""M6: the reserved-set evaluation (protocol: results/m6/protocol.json, written before any test run).

    python m6_eval.py <tag> <who>[,<who>...] [--sets train:120,instances-test:120,...] [--policy name=weights ...]
who: bot, oracle, searcher_all, searcher_occluder, or a policy name given with --policy.
Writes results/m6/eval_<tag>.json (every judge result + summary with Wilson 95% intervals).
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle import baseline_for, controller_for  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, TASKS, make_episode, run_episode  # noqa: E402

SETS = {"train": ("train.json", "configs")}
for fam in ("instances", "arrangements", "appearances", "distractor"):
    SETS[f"{fam}-test"] = (f"{fam}.json", "test")
    SETS[f"{fam}-dev"] = (f"{fam}.json", "dev")
OUT = HERE / "results" / "m6"


def load_set(name, n):
    f, part = SETS[name]
    return json.loads((HERE / "splits" / f).read_text())[part][:n]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def one(args):
    cfg, setname, task, who, policies = args
    ep = make_episode(cfg, task, 0)
    w = World(cfg)
    t0 = time.time()
    extra = {}
    if who == "oracle":
        fac = controller_for(w, task)
    elif who == "bot":
        fac, bot = controller(task, BUDGET_TICKS[task])
    elif who.startswith("searcher_"):
        if task != "find_ball":
            return None
        fac = baseline_for(w, "all_hiders" if who == "searcher_all" else "occluder_only")
    else:
        from learner import PolicyController
        pc = PolicyController(policies[who])
        fac = pc.factory
    res, _, it = run_episode(cfg, ep, fac, world=w)
    res = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.items()}
    if who not in ("oracle", "bot") and not who.startswith("searcher_"):
        lat = np.array(pc.latency)
        extra = {"latency_ms_median": float(np.median(lat) * 1000), "latency_ms_p99": float(np.percentile(lat, 99) * 1000),
                 "load_1min": round(os.getloadavg()[0], 2)}
    if who == "bot":
        extra["log"] = bot.log
    res.update(set=setname, who=who, wall_s=round(time.time() - t0, 2), hider=(ep.get("ball_launch") or {}).get("hider"),
               target=ep["instruction"].get("target"), **extra)
    if task == "set_lamp":
        res["press_needed"] = bool(res["needed_presses"])
    return res


def summarise(rows):
    out = {}
    for who in sorted({r["who"] for r in rows}):
        for task in TASKS:
            for s in SETS:
                rs = [r for r in rows if r["who"] == who and r["task"] == task and r["set"] == s]
                if not rs:
                    continue
                k = sum(r["success"] for r in rs)
                succ = [r for r in rs if r["success"]]
                row = {"n": len(rs), "success": k, "rate": round(k / len(rs), 3), "ci95": wilson(k, len(rs)),
                       "median_s_success": float(np.median([r["sim_time"] for r in succ])) if succ else None,
                       "falls": int(sum(r["falls"] for r in rs)), "contact_episodes": int(sum(r["collision_ticks"] > 0 for r in rs)),
                       "declared_done": int(sum(r["declared_done"] for r in rs))}
                if task == "set_lamp":
                    pn = [r for r in rs if r["press_needed"]]
                    row["press_needed"] = [int(sum(r["success"] for r in pn)), len(pn)]
                    row["no_press_needed"] = [int(sum(r["success"] for r in rs if not r["press_needed"])), len(rs) - len(pn)]
                    row["extra_presses"] = int(sum(r["extra_presses"] for r in rs))
                if task == "go_and_return":
                    row["by_colour"] = {c: [int(sum(r["success"] for r in rs if r["target"] == c)), int(sum(r["target"] == c for r in rs))]
                                        for c in ("red", "green", "blue", "yellow")}
                    row["reached_pad"] = int(sum(r["reached_pad_tick"] is not None for r in rs))
                if task == "find_ball":
                    row["by_hider"] = {h: [int(sum(r["success"] for r in rs if r["hider"] == h)), int(sum(r["hider"] == h for r in rs))]
                                       for h in ("occluder", "partition")}
                if "latency_ms_median" in rs[0]:
                    row["latency_ms_median"] = float(np.median([r["latency_ms_median"] for r in rs]))
                    row["load_1min_median"] = float(np.median([r["load_1min"] for r in rs]))
                out[f"{who}/{task}/{s}"] = row
    return out


def main():
    a = sys.argv[1:]
    tag, whos = a[0], a[1].split(",")
    sets = "train:120,instances-test:120,arrangements-test:120,appearances-test:120,distractor-test:120"
    policies = {}
    for x in a[2:]:
        if x.startswith("--sets="):
            sets = x.split("=", 1)[1]
        if x.startswith("--policy="):
            name, path = x.split("=", 1)[1].split(":", 1)
            policies[name] = path
    jobs = []
    for spec in sets.split(","):
        name, n = spec.split(":")
        for cfg in load_set(name, int(n)):
            for task in TASKS:
                for who in whos:
                    jobs.append((cfg, name, task, who, policies))
    t0 = time.time()
    with Pool(int(os.environ.get("M6_PROCS", "4"))) as pool:
        rows = [r for r in pool.map(one, jobs, chunksize=1) if r is not None]
    rep = {"tag": tag, "whos": whos, "sets": sets, "uptime_end": os.popen("uptime").read().strip(),
           "summary": summarise(rows), "rows": rows, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eval_{tag}.json").write_text(json.dumps(rep, indent=0, default=float))
    for k, v in rep["summary"].items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("by_colour",)}))
    print("episodes", len(rows), "wall", rep["wall_s"])


if __name__ == "__main__":
    main()
