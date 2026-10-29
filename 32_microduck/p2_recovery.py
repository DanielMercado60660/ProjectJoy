"""Phase 2, amendment 6: record recovery episodes. P2-A' drives set_lamp with the bot in shadow; from tick T
the shadow bot's commands are executed (T = (60, 150, 250)[i % 3] for the i-th train config); budget T+900.

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_recovery.py --start 100 --n 100
    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_recovery.py --start 200 --n 100
Episodes: episodes/p2_recovery/set_lamp/train_<cfg>_k4_recovery.npz (gitignored); summary rows appended to
results/p2/recovery_recording.jsonl. The training index is built by p2_recovery_index.py.
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

POLICY = "runs/p2_a2/ckpt_016000/weights.safetensors"
K = 4
TICKS = (60, 150, 250)


def job(a):
    i, cfg = a
    T = TICKS[i % 3]
    try:
        r = record_episode(cfg, "train", "set_lamp", K, out_dir=EP_DIR / "p2_recovery", tag="recovery", cold_renderer=True,
                           policy_weights=POLICY, beta=0.0, policy_kind="p2", takeover_tick=T, budget_ticks=T + 900)
        return dict(r, takeover_tick=T)
    except Exception as e:
        return {"config": cfg["id"], "error": repr(e)[:300], "takeover_tick": T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"]
    jobs = [(i, cfgs[i]) for i in range(a.start, a.start + a.n)]
    t0 = time.time()
    with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
        rows = []
        for j, r in enumerate(pool.imap(job, jobs, chunksize=1)):
            rows.append(r)
            if (j + 1) % 25 == 0:
                print(json.dumps({"done": j + 1, "of": len(jobs), "wall_s": round(time.time() - t0, 1)}), flush=True)
    with open(HERE / "results" / "p2" / "recovery_recording.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if "error" not in r]
    print(json.dumps({"configs": [a.start, a.start + a.n], "episodes": len(rows), "errors": len(rows) - len(ok),
                      "success": sum(bool(r["success"]) for r in ok), "ticks": sum(r["ticks"] for r in ok),
                      "wall_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
