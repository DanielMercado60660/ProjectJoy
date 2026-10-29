"""Phase 2, amendment 7: set-lamp confirmation on dev rooms never used for selection, plus stage counts.

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_confirm.py NAME=WEIGHTS [NAME=WEIGHTS ...]
    ../.venv/bin/python p2_confirm.py --stages select_p2_r3_s1 [...]   # stage counts from a selection file's pick

Each policy drives set_lamp (k=0) on 80 dev rooms the learner has never been selected or diagnosed on:
instances-dev 0030-0049 and arrangements/appearances/distractor-dev 0020-0039. p2_select drives 0000-0009 of
each family, and p2_diag's bot demos (p2_devdemos.py) use instances-dev 0010-0029 and the other families'
0000-0019 (Astra's catch, amendment 7). The Phase 1 bot was tuned on dev; the learner never was. Appends one row per policy to
results/p2/confirm.jsonl and writes per-episode rows to results/p2/confirm_<name>.json.
Stage counts (press-needed episodes): reached the plate, pressed, lamp right at the end, said done,
said done after a press (split by lamp right / wrong at the done). No-press episodes: said done, and median tick of the done.
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
from p2_select import one_any, summarise  # noqa: E402

ROOMS = (("instances-dev", 30, 50), ("arrangements-dev", 20, 40), ("appearances-dev", 20, 40),
         ("distractor-dev", 20, 40))


def stages(rows):
    pn = [r for r in rows if r["press_needed"]]
    npn = [r for r in rows if not r["press_needed"]]
    lamp_right = lambda r: (r["lamp_final"] is True) == (r["target"] == "on")   # noqa: E731
    done_ticks = [r["ticks"] for r in npn if r["declared_done"]]
    return {"press_needed": {"n": len(pn),
                             "reached_plate": int(sum((r.get("plate_visits") or 0) > 0 for r in pn)),
                             "pressed": int(sum((r.get("presses") or 0) > 0 for r in pn)),
                             "lamp_right_at_end": int(sum(lamp_right(r) for r in pn)),
                             "said_done": int(sum(bool(r["declared_done"]) for r in pn)),
                             "said_done_after_press": int(sum(bool(r["declared_done"]) and (r.get("presses") or 0) > 0
                                                              for r in pn)),
                             "done_after_press_lamp_right": int(sum(bool(r["declared_done"]) and (r.get("presses") or 0) > 0
                                                                    and lamp_right(r) for r in pn)),
                             "done_after_press_lamp_wrong": int(sum(bool(r["declared_done"]) and (r.get("presses") or 0) > 0
                                                                    and not lamp_right(r) for r in pn))},
            "no_press": {"n": len(npn), "said_done": len(done_ticks),
                         "median_done_tick": float(np.median(done_ticks)) if done_ticks else None}}


def main():
    if sys.argv[1] == "--stages":
        for name in sys.argv[2:]:
            d = json.loads((HERE / "results" / "p2" / f"{name}.json").read_text())
            rows = [r for r in d["rows"] if r["who"] == d["chosen"]]
            print(json.dumps({"select": name, "chosen": d["chosen"], **stages(rows)}))
        return
    pols = dict(a.split("=", 1) for a in sys.argv[1:])
    for name, w in pols.items():
        jobs = [(c, s, "set_lamp", name, {name: w}) for s, a, b in ROOMS for c in load_set(s, b)[a:b]]
        with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
            rows = [r for r in pool.map(one_any, jobs, chunksize=1) if r is not None]
        res = {"policy": name, "weights": w, "rooms": {s: [a, b - 1] for s, a, b in ROOMS}, **summarise(rows),
               "stages": stages(rows)}
        print(json.dumps(res), flush=True)
        with open(HERE / "results" / "p2" / "confirm.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        (HERE / "results" / "p2" / f"confirm_{name}.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
