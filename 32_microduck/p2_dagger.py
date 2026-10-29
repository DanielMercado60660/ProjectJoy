"""Phase 2: DAgger round 3 (protocol amendment 3). A Phase 2 policy (P2Controller) drives set_lamp on
train configs with the reference bot in shadow mode; the bot's command is the label (cmd_bot). The
recorder ends each episode at the bot's "done" (after_bot_done="stop"), so its done is labelled once.

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_dagger.py \
        runs/p2_a2/ckpt_016000/weights.safetensors [--start 0] [--n 300] [--k 3] [--beta 0] [--tag dagger3]
Writes episodes/set_lamp/train_<cfg>_k<k>_<tag>.npz, episodes/<tag>_index.jsonl and
results/p2/<tag>_recording.json. Each .npz is saved as its episode finishes; the index is written at the
end, so if a run dies the finished episodes are still on disk.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from record import EP_DIR, record_episode  # noqa: E402

TASK = "set_lamp"


def job(a):
    cfg, k, weights, beta, tag = a
    try:
        return record_episode(cfg, "train", TASK, k, None, None, policy_weights=weights, beta=beta, tag=tag,
                              policy_kind="p2")
    except Exception as e:
        return {"config": cfg["id"], "task": TASK, "error": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--tag", default="dagger3")
    a = ap.parse_args()
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][a.start:a.start + a.n]
    jobs = [(c, a.k, a.weights, a.beta, a.tag) for c in cfgs]
    t0 = time.time()
    with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
        rows = []
        for i, r in enumerate(pool.imap(job, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(json.dumps({"done": i + 1, "of": len(jobs), "wall_s": round(time.time() - t0, 1)}), flush=True)
    with open(EP_DIR / f"{a.tag}_index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if "error" not in r]
    summ = {"policy": a.weights, "policy_kind": "p2", "beta": a.beta, "task": TASK, "k": a.k,
            "configs": [a.start, a.start + a.n], "episodes": len(rows), "errors": len(rows) - len(ok),
            "success": [sum(bool(r["success"]) for r in ok), len(ok)],
            "ticks": sum(r["ticks"] for r in ok), "wall_s": round(time.time() - t0, 1)}
    (HERE / "results" / "p2" / f"{a.tag}_recording.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
