"""Phase 2: record reference-bot set_lamp episodes on dev configs for the open-loop lamp diagnostic
(results/p2/protocol.json). Never trained on.

    ../bin/gpu_run ../.venv/bin/python p2_devdemos.py      # 80 episodes -> episodes/p2_dev/, episodes/p2_dev/index.jsonl
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_eval import load_set  # noqa: E402

OUT = HERE / "episodes" / "p2_dev"
SETS = (("instances-dev", 10, 30), ("arrangements-dev", 0, 20), ("appearances-dev", 0, 20), ("distractor-dev", 0, 20))


def job(a):
    from record import record_episode
    cfg, split = a
    return record_episode(cfg, split, "set_lamp", 0, out_dir=OUT, cold_renderer=True)


def main():
    jobs = [(c, s) for s, a, b in SETS for c in load_set(s, b)[a:b]]
    with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "episodes;", sum(r["success"] for r in rows), "bot successes")


if __name__ == "__main__":
    main()
