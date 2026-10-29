"""M5: apply the pre-written checkpoint selection rule (results/m5/selection_rule.json).

    python m5_select.py <run_dir>
Every ckpt_* in the run is evaluated closed-loop on instances-dev 0000-0009 x 3 tasks (k=0). The pick:
most successes, then lower median sim time of successes, then the earlier step.
Writes results/m5/selection_<run name>.json.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_eval import load_set, one  # noqa: E402
from tasks import TASKS  # noqa: E402


def main():
    run = HERE / sys.argv[1]
    ckpts = sorted(p for p in run.glob("ckpt_*") if (p / "weights.safetensors").exists())
    policies = {p.name: str(p / "weights.safetensors") for p in ckpts}
    cfgs = load_set("instances-dev", 10)
    jobs = [(c, "instances-dev", t, name, policies) for name in policies for c in cfgs for t in TASKS]
    with Pool(int(os.environ.get("M6_PROCS", "4"))) as pool:
        rows = [r for r in pool.map(one, jobs, chunksize=1) if r is not None]
    table = {}
    for name in policies:
        rs = [r for r in rows if r["who"] == name]
        succ = [r for r in rs if r["success"]]
        table[name] = {"success": len(succ), "n": len(rs),
                       "by_task": {t: int(sum(r["success"] for r in rs if r["task"] == t)) for t in TASKS},
                       "median_s_success": float(np.median([r["sim_time"] for r in succ])) if succ else 1e9,
                       "declared_done": int(sum(r["declared_done"] for r in rs)),
                       "falls": int(sum(r["falls"] for r in rs))}
    pick = sorted(table, key=lambda k: (-table[k]["success"], table[k]["median_s_success"], k))[0]
    out = {"run": run.name, "rule": json.loads((HERE / "results" / "m5" / "selection_rule.json").read_text())["rule"],
           "table": table, "chosen": pick, "weights": policies[pick], "rows": rows}
    (HERE / "results" / "m5" / f"selection_{run.name}.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in table.items():
        print(k, v)
    print("chosen:", pick)


if __name__ == "__main__":
    main()
