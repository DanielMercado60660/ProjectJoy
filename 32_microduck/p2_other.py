"""Phase 2, amendment 4: go-and-return and find-ball dev check for Phase 2 policies (reported only; not in
the selection rule).

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_other.py NAME=WEIGHTS [NAME=WEIGHTS ...]
Each policy drives go_and_return and find_ball on instances/arrangements/appearances/distractor-dev
0000-0009 (k=0): 40 episodes per task. A weights file with p2_config.json beside it runs as a Phase 2
policy, otherwise as an M5 policy. Appends one row per policy to results/p2/other_tasks.jsonl and writes
the per-episode rows to results/p2/other_tasks_<name>.json.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_eval import load_set  # noqa: E402
from p2_select import SETS, one_any  # noqa: E402

TASKS2 = ("go_and_return", "find_ball")
KEEP = ("set", "task", "config_id", "success", "declared_done", "falls", "collision_ticks", "sim_time", "stopped",
        "reached_pad_tick", "home_after_pad_tick", "on_mat", "wrong_pad_visits",
        "ball_visible_at_end", "ball_distance", "ball_hidden_seen", "reacquired_tick")


def summarise(rows):
    out = {}
    for t in TASKS2:
        rs = [r for r in rows if r["task"] == t]
        sub = [r for r in rs if r["set"] == "instances-dev"]
        row = {"n": len(rs), "success": int(sum(r["success"] for r in rs)),
               "success_instances_dev_0_9": [int(sum(r["success"] for r in sub)), len(sub)],
               "declared_done": int(sum(r["declared_done"] for r in rs)),
               "falls": int(sum(r["falls"] for r in rs)),
               "contact_episodes": int(sum(r["collision_ticks"] > 0 for r in rs))}
        if t == "go_and_return":
            row["reached_pad"] = int(sum(r.get("reached_pad_tick") is not None for r in rs))
            row["home_after_pad"] = int(sum(r.get("home_after_pad_tick") is not None for r in rs))
            row["wrong_pad_episodes"] = int(sum(bool(r.get("wrong_pad_visits")) for r in rs))
        else:
            row["ball_visible_at_end"] = int(sum(bool(r.get("ball_visible_at_end")) for r in rs))
            row["reacquired"] = int(sum(r.get("reacquired_tick") is not None for r in rs))
        out[t] = row
    return out


def main():
    policies = dict(a.split("=", 1) for a in sys.argv[1:])
    jobs = [(c, s, t, who, policies) for who in policies for t in TASKS2 for s in SETS for c in load_set(s, 10)]
    with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
        rows = [r for r in pool.map(one_any, jobs, chunksize=1) if r is not None]
    for who, w in policies.items():
        rs = [{k: r.get(k) for k in KEEP} for r in rows if r["who"] == who]
        (HERE / "results" / "p2" / f"other_tasks_{who}.json").write_text(json.dumps({"weights": w, "rows": rs}, indent=1))
        line = {"policy": who, "weights": w, **summarise(rs)}
        with open(HERE / "results" / "p2" / "other_tasks.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
