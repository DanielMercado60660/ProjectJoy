"""R6 LeWM S1: record noisy-bot episodes on TRAINING configs only (world-model data, TwoRoom-style).

    ../bin/gpu_run ../.venv/bin/python lewm/record_noisy.py --lo 0 --hi 300 --k 5 --index episodes/noisy_index.jsonl --max-min 44

Each config x task gets one episode with record.record_episode(..., noise=NOISE, tag="noisy"):
random command bursts of 3-10 ticks start with p=0.0385 per tick (about 20% of ticks noisy), the bot
recovers closed loop. 4 worker processes, 1 thread each (OMP/NUMBA set by the caller or here).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from record import record_episode  # noqa: E402
from tasks import TASKS  # noqa: E402

NOISE = {"p": 0.0385, "len": [3, 10], "head_frac": 0.6}


def job(args):
    cfg, task, k = args
    try:
        return record_episode(cfg, "train", task, k, noise=NOISE, tag="noisy")
    except Exception as e:  # keep going; report the failure in the index
        return {"config": cfg["id"], "task": task, "k": k, "error": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--index", required=True)
    ap.add_argument("--max-min", type=float, default=44.0, help="stop (terminating in-flight episodes) after this")
    a = ap.parse_args()
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][a.lo:a.hi]
    assert all(c["id"].startswith("train-") for c in cfgs)
    out = HERE / a.index
    done = set()
    if out.exists():   # skip (config, task) pairs already recorded at this k
        done = {(r["config"], r["task"]) for r in map(json.loads, open(out)) if "error" not in r and r.get("k") == a.k}
    jobs = [(c, t, a.k) for c in cfgs for t in TASKS if (c["id"], t) not in done]
    t0 = time.time()
    stopped = False
    with Pool(a.procs) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(job, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (time.time() - t0) / 60 > a.max_min:
                stopped = True
                pool.terminate()
                break
    rows = [json.loads(l) for l in open(out)]
    ok = [r for r in rows if "error" not in r]
    print(json.dumps({"episodes": len(rows), "errors": len(rows) - len(ok), "ticks": sum(r["ticks"] for r in ok),
                      "success": sum(bool(r["success"]) for r in ok), "wall_s": round(time.time() - t0),
                      "stopped_by_time_guard": stopped, "jobs_this_run": len(jobs)}))


if __name__ == "__main__":
    main()
