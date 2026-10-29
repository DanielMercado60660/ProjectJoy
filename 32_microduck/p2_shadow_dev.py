"""Phase 2 diagnostic (added after P2-E's dev result; not pre-registered): does the policy follow the bot's
turn and walk commands in its own states? The policy drives the 40 set_lamp dev episodes (same as
p2_select) with the reference bot in shadow mode, and its commands are compared with the bot's.

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_shadow_dev.py NAME=WEIGHTS [...]
Episodes go to episodes/p2_shadow_dev/<name>/ (gitignored); one summary line per policy is appended to
results/p2/shadow_dev.jsonl. The recorder stops when the bot says done, so episode outcomes here are not
evaluation results (p2_select is); only the command comparison is reported.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_eval import load_set  # noqa: E402
from p2_select import SETS  # noqa: E402
from record import EP_DIR, record_episode  # noqa: E402


def job(a):
    cfg, setname, name, w = a
    r = record_episode(cfg, setname, "set_lamp", 0, policy_weights=w, beta=0.0, tag=name, policy_kind="p2",
                       out_dir=EP_DIR / "p2_shadow_dev" / name, cold_renderer=True)
    z = np.load(r["path"])
    H = json.loads(str(z["header"]))
    cb, ce = z["cmd_bot"], z["cmd_exec"]
    ok = np.isfinite(cb).all(1)
    cb, ce = cb[ok], ce[ok]
    turn = np.abs(cb[:, 2]) > 0.1
    return {"set": setname, "config": cfg["id"], "press_needed": bool(H["result"]["needed_presses"]),
            "ticks": int(ok.sum()), "bot_turn": int(turn.sum()),
            "same_dir": int((np.sign(cb[turn, 2]) == np.sign(ce[turn, 2])).sum()),
            "abs_wz_bot": float(np.abs(cb[turn, 2]).sum()), "abs_wz_pol": float(np.abs(ce[turn, 2]).sum()),
            "bot_walk": int((cb[:, 0] > 0.15).sum()),
            "pol_walk_when_bot_walk": int(((cb[:, 0] > 0.15) & (ce[:, 0] >= 0.2)).sum()),
            "bot_finished": H["dagger"]["bot_finished_tick"] is not None}


def main():
    pols = dict(a.split("=", 1) for a in sys.argv[1:])
    jobs = [(c, s, n, w) for n, w in pols.items() for s in SETS for c in load_set(s, 10)]
    with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    for n, w in pols.items():
        line = {"policy": n, "weights": w}
        for grp, f in (("press_needed", True), ("no_press", False)):
            rs = [r for r, j in zip(rows, jobs) if j[2] == n and r["press_needed"] == f]
            S = {k: sum(r[k] for r in rs) for k in ("ticks", "bot_turn", "same_dir", "abs_wz_bot", "abs_wz_pol",
                                                     "bot_walk", "pol_walk_when_bot_walk", "bot_finished")}
            line[grp] = {"episodes": len(rs), "ticks": S["ticks"],
                         "bot_turn_share": round(S["bot_turn"] / max(1, S["ticks"]), 3),
                         "turn_same_direction": round(S["same_dir"] / max(1, S["bot_turn"]), 3),
                         "mean_abs_wz_bot": round(S["abs_wz_bot"] / max(1, S["bot_turn"]), 3),
                         "mean_abs_wz_policy": round(S["abs_wz_pol"] / max(1, S["bot_turn"]), 3),
                         "bot_walk_ticks": S["bot_walk"],
                         "policy_walks_when_bot_walks": round(S["pol_walk_when_bot_walk"] / max(1, S["bot_walk"]), 3),
                         "bot_said_done": S["bot_finished"]}
        with open(HERE / "results" / "p2" / "shadow_dev.jsonl", "a") as fo:
            fo.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
