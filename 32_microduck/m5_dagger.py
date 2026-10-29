"""M5: one DAgger round. The selected BC policy and the reference bot run together on training configs
(train split, configs 0..N-1, episode k=2: not used by the demos (k=1) or M3's scoring (k=0)). Each
tick the executed command is the bot's with probability beta (0.5), else the policy's; the bot runs in
shadow mode on the same observations and its command is always the label (cmd_bot). Every episode is
kept: each tick carries the expert's label, whatever happened.

    python m5_dagger.py <policy weights> [--n 150] [--start 0] [--beta 0.5] [--round 1]
Writes episodes/<task>/*_dagger[N].npz and episodes/dagger[N]_index.jsonl; results/m5/dagger[N]_recording.json
(round 1 keeps the unsuffixed names). Processes: M5_PROCS (default 4); run it through ../bin/gpu_run.

Known flaw in the stored round-1/2 data (REVIEW_M5.md finding 6): when the shadow bot's generator ended
before the episode did, every later tick was labelled with the bot's last command (done, stand still),
wherever the policy then drove. Round 1 (beta 0.5): 39 ticks after the bot's done tick (258 done-label
ticks in all). Round 2 (beta 0): 19,022 after it in 62 episodes (19,084 done-label ticks).
Fixed in record.py after M6 (after_bot_done="stop", the default used here): the episode now ends when the
shadow bot finishes, so its "done" is labelled once. Re-running this script therefore no longer reproduces
the stored episodes, and an episode that ends this way (no "done" executed) counts as a failure in the
recording summary.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from record import EP_DIR, record_episode  # noqa: E402
from tasks import TASKS  # noqa: E402


def job(a):
    cfg, task, weights, beta, tag = a
    try:
        r = record_episode(cfg, "train", task, 2, None, None, policy_weights=weights, beta=beta, tag=tag)
        return r
    except Exception as e:
        return {"config": cfg["id"], "task": task, "error": repr(e)[:300]}


def main():
    weights = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 150
    beta = float(sys.argv[sys.argv.index("--beta") + 1]) if "--beta" in sys.argv else 0.5
    start = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 0
    rnd = int(sys.argv[sys.argv.index("--round") + 1]) if "--round" in sys.argv else 1
    sfx = "" if rnd == 1 else str(rnd)
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][start:start + n]
    jobs = [(c, t, weights, beta, "dagger" + sfx) for c in cfgs for t in TASKS]
    t0 = time.time()
    with Pool(int(os.environ.get("M5_PROCS", "4"))) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    with open(EP_DIR / f"dagger{sfx}_index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if "error" not in r]
    summ = {"policy": weights, "beta": beta, "round": rnd, "configs": [start, start + n], "episodes": len(rows), "errors": len(rows) - len(ok),
            "success": {t: [sum(r["success"] for r in ok if r["task"] == t), sum(r["task"] == t for r in ok)] for t in TASKS},
            "ticks": sum(r["ticks"] for r in ok), "wall_s": round(time.time() - t0, 1)}
    (HERE / "results" / "m5" / f"dagger{sfx}_recording.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
