"""M4: record the reference bot's demonstrations on training configs (train split only).

    python m4_record.py [--gaze] [--n N]
Every training config 0..N-1 x 3 tasks, episode k=1 (the M3 evaluation used k=0, so demos are distinct
episodes), plus a perturbed copy for configs with index % 7 == 3 (12.5% of episodes labelled; ~8% actually perturbed, REVIEW_M4.md):
  set_lamp      missed_press if that episode needs a press, else shove
  find_ball     ball_moved
  go_and_return shove
Failures are kept. Writes episodes/<task>/*.npz and episodes/index.jsonl (rewritten each run).
A per-episode wall guard: the task budget caps every episode (<= 1200 ticks).
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from record import EP_DIR, record_episode  # noqa: E402
from tasks import TASKS, make_episode  # noqa: E402


def job(args):
    cfg, task, perturb, params = args
    try:
        return record_episode(cfg, "train", task, 1, perturb, params)
    except Exception as e:                     # keep going; report the failure in the index
        return {"config": cfg["id"], "task": task, "perturbation": {"type": perturb}, "error": repr(e)[:300]}


def main():
    gaze = "--gaze" in sys.argv
    n = 300
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    params = {"gaze": True} if gaze else None
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][:n]
    jobs = []
    for i, c in enumerate(cfgs):
        for task in TASKS:
            jobs.append((c, task, None, params))
            if i % 7 == 3:
                if task == "set_lamp":
                    ep = make_episode(c, task, 1)
                    need = (ep["instruction"]["target"] == "on") != bool(c["layout"]["lamp_initial_on"])
                    pert = "missed_press" if need else "shove"
                else:
                    pert = {"find_ball": "ball_moved", "go_and_return": "shove"}[task]
                jobs.append((c, task, pert, params))
    t0 = time.time()
    EP_DIR.mkdir(exist_ok=True)
    with Pool(4) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    with open(EP_DIR / "index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if "error" not in r]
    summ = {"episodes": len(rows), "errors": len(rows) - len(ok),
            "by_task": {t: {"n": sum(r["task"] == t for r in ok), "success": sum(r["success"] for r in ok if r["task"] == t),
                            "perturbed": sum(bool(r["perturbation"]["type"]) for r in ok if r["task"] == t),
                            "ticks": sum(r["ticks"] for r in ok if r["task"] == t)} for t in TASKS},
            "size_gb": round(sum(r["mb"] for r in ok) / 1e3, 2), "wall_s": round(time.time() - t0, 1), "gaze": gaze}
    (HERE / "results" / "m4").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "m4" / "recording.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
