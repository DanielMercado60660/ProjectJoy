"""Phase 2, amendment 5, check A: can the reference bot recover from states the learned policy drives into?

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_takeover.py --arms bot_alone,takeover_100,takeover_250
    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_takeover.py --arms takeover_400,fresh_250
    ../.venv/bin/python p2_takeover.py --summary          # -> results/p2/takeover.json

set_lamp on train configs 0-99, k=4. Arms:
  bot_alone       the bot drives from tick 0 (budget 900)
  takeover_T      P2-A' drives with the bot in shadow; from tick T the shadow bot's commands are executed
                  (budget T+900, so the bot has 900 ticks after taking over)
  fresh_250       as takeover_250, but a new bot instance starts at tick 250 (no shadow memory)
Episodes go to episodes/p2_takeover/<arm>/ (gitignored); per-episode rows to
results/p2/takeover_rows_<arms>.jsonl.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from record import EP_DIR, record_episode  # noqa: E402

POLICY = "runs/p2_a2/ckpt_016000/weights.safetensors"
K, N = 4, 100


def run(a):
    cfg, arm = a
    out = EP_DIR / "p2_takeover" / arm
    if arm == "bot_alone":
        r = record_episode(cfg, "train", "set_lamp", K, out_dir=out, tag=arm, cold_renderer=True)
        T = None
    else:
        T = int(arm.split("_")[1])
        fresh = arm.startswith("fresh")
        r = record_episode(cfg, "train", "set_lamp", K, out_dir=out, tag=arm, cold_renderer=True,
                           policy_weights=POLICY, beta=0.0, policy_kind="p2", takeover_tick=T,
                           takeover_fresh=fresh, budget_ticks=T + 900,
                           after_bot_done="continue" if fresh else "stop")
    z = np.load(r["path"])
    H = json.loads(str(z["header"]))
    res = H["result"]
    ticks = H["ticks"]
    fin = (H["dagger"] or {}).get("bot_finished_tick")
    ended_before = T is not None and ticks <= T
    return {"arm": arm, "config": cfg["id"], "press_needed": bool(res["needed_presses"]), "success": bool(res["success"]),
            "declared_done": bool(res["declared_done"]), "lamp_right": bool(res["lamp_final"]) == bool(res["target"]),
            "presses": int(res["presses"]), "falls": int(res["falls"]), "ticks": ticks,
            "ended_before_takeover": ended_before,
            "ended_by": None if not ended_before else ("policy_done" if res["declared_done"] else
                                                       "shadow_bot_done" if fin is not None else "other")}


def summary():
    rows = []
    for f in sorted((HERE / "results" / "p2").glob("takeover_rows_*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        a = {}
        for grp, flag in (("press_needed", True), ("no_press", False)):
            rs = [r for r in rows if r["arm"] == arm and r["press_needed"] == flag]
            live = [r for r in rs if not r["ended_before_takeover"]]
            done = [r for r in live if r["declared_done"]]
            a[grp] = {"episodes": len(rs), "ended_before_takeover": len(rs) - len(live),
                      "ended_by": {k: sum(r["ended_by"] == k for r in rs) for k in ("policy_done", "shadow_bot_done", "other")},
                      "success": [sum(r["success"] for r in live), len(live)],
                      "said_done": len(done), "said_done_lamp_wrong": sum(not r["lamp_right"] for r in done),
                      "pressed": sum(r["presses"] > 0 for r in live), "falls": sum(r["falls"] for r in live),
                      "timeouts": sum(not r["declared_done"] for r in live)}
        out[arm] = a
    base = out.get("bot_alone", {}).get("press_needed", {}).get("success")
    gate = {}
    for arm in ("takeover_250", "fresh_250"):
        if arm in out and base and base[1]:
            s = out[arm]["press_needed"]["success"]
            rel = (s[0] / s[1]) / (base[0] / base[1]) if s[1] and base[0] else None
            d = out[arm]["press_needed"]["said_done"] + out[arm]["no_press"]["said_done"]
            wrong = out[arm]["press_needed"]["said_done_lamp_wrong"] + out[arm]["no_press"]["said_done_lamp_wrong"]
            gate[arm] = {"press_needed_success_vs_bot_alone": None if rel is None else round(rel, 3),
                         "wrong_done_share": round(wrong / d, 3) if d else None,
                         "passes": rel is not None and rel >= 0.8 and d > 0 and wrong / d <= 0.05}
    res = {"policy": POLICY, "configs": "train 0-99", "k": K, "arms": out, "gate": gate}
    (HERE / "results" / "p2" / "takeover.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    arms = [x for x in a.arms.split(",") if x]
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][:a.n]
    jobs = [(c, arm) for arm in arms for c in cfgs]
    with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
        rows = []
        for i, r in enumerate(pool.imap(run, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(json.dumps({"done": i + 1, "of": len(jobs)}), flush=True)
    with open(HERE / "results" / "p2" / f"takeover_rows_{'_'.join(arms)}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
