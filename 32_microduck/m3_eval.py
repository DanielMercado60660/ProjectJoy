"""M3 evaluation: the reference bot (and the oracle) on the same episodes, train and dev only.

    python m3_eval.py <tag> <set:count> [<set:count> ...] [--oracle] [--tasks a,b]
sets: train, instances-dev, arrangements-dev, appearances-dev, distractor-dev (count = first N configs).
Episode k=0 of each task per config. Writes results/m3/eval_<tag>.json with every judge result, the
bot's event log and memory summary, and a summary table per task x set (success, median sim time of
successes, falls, episodes touching an obstacle, task counts).
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle import controller_for  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, TASKS, make_episode, run_episode  # noqa: E402

OUT = HERE / "results" / "m3"
SETS = {"train": ("train.json", "configs"), "instances-dev": ("instances.json", "dev"),
        "arrangements-dev": ("arrangements.json", "dev"), "appearances-dev": ("appearances.json", "dev"),
        "distractor-dev": ("distractor.json", "dev")}


def load_set(name, n):
    f, part = SETS[name]
    return json.loads((HERE / "splits" / f).read_text())[part][:n]


def one(args):
    cfg, setname, task, who, params = args
    ep = make_episode(cfg, task, 0)
    w = World(cfg)
    t0 = time.time()
    if who == "oracle":
        res, _, _ = run_episode(cfg, ep, controller_for(w, task), world=w)
        log = []
    else:
        fac, bot = controller(task, BUDGET_TICKS[task], params)
        res, _, _ = run_episode(cfg, ep, fac, world=w)
        log = bot.log
    res = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.items()}
    res.update(set=setname, who=who, wall_s=round(time.time() - t0, 2), log=log,
               hider=(ep.get("ball_launch") or {}).get("hider"))
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
                succ = [r for r in rs if r["success"]]
                row = {"n": len(rs), "success": len(succ), "rate": round(len(succ) / len(rs), 3),
                       "median_s_success": float(np.median([r["sim_time"] for r in succ])) if succ else None,
                       "falls": int(sum(r["falls"] for r in rs)),
                       "contact_episodes": int(sum(r["collision_ticks"] > 0 for r in rs))}
                if task == "set_lamp":
                    pn = [r for r in rs if r["press_needed"]]
                    row["press_needed"] = [int(sum(r["success"] for r in pn)), len(pn)]
                    row["extra_presses"] = int(sum(r["extra_presses"] for r in rs))
                    row["visits_without_press"] = int(sum(r["visits_without_press"] for r in rs))
                if task == "go_and_return":
                    row["wrong_pad_episodes"] = int(sum(bool(r["wrong_pad_visits"]) for r in rs))
                    row["reached_pad"] = int(sum(r["reached_pad_tick"] is not None for r in rs))
                if task == "find_ball":
                    row["by_hider"] = {h: [int(sum(r["success"] for r in rs if r["hider"] == h)),
                                           int(sum(r["hider"] == h for r in rs))] for h in ("occluder", "partition")}
                out[f"{who}/{task}/{s}"] = row
    return out


def main():
    args = sys.argv[1:]
    tag = args[0]
    with_oracle = "--oracle" in args
    tasks = TASKS
    params = None
    for a in args:
        if a.startswith("--tasks="):
            tasks = tuple(a.split("=", 1)[1].split(","))
        if a.startswith("--params="):
            params = json.loads(a.split("=", 1)[1])
    sets = [a.split(":") for a in args[1:] if ":" in a and not a.startswith("--")]
    jobs = []
    for name, n in sets:
        for cfg in load_set(name, int(n)):
            for task in tasks:
                jobs.append((cfg, name, task, "bot", params))
                if with_oracle:
                    jobs.append((cfg, name, task, "oracle", None))
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    rep = {"tag": tag, "params": params, "summary": summarise(rows), "rows": rows,
           "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eval_{tag}.json").write_text(json.dumps(rep, indent=0, default=float))
    for k, v in rep["summary"].items():
        print(k, json.dumps(v))
    print("episodes", len(rows), "wall", rep["wall_s"])


if __name__ == "__main__":
    main()
