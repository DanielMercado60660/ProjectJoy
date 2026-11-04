"""Matched-harness baseline: random-perturbation hill-climb through the SAME budgeted wrapper
(eval_train.py, same tracks, same calls, same keep-if-better rule) that Astra used. Isolates
"Astra's reasoning" from "N tries of anything".

    python wsp/hillclimb.py --task hack --start ../checkpoints/es_soft_best.npz --budget 25 --sigma 0.04 --seed 0
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import WSP, dict_to_theta, load_weights, save_weights, theta_to_dict  # noqa: E402

PY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".venv", "bin", "python"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hack")
    ap.add_argument("--start", required=True)
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    adir = os.path.join(WSP, "baselines", f"hc_{a.task}_s{a.sigma}_seed{a.seed}_b{a.budget}")
    shutil.rmtree(adir, ignore_errors=True)
    os.makedirs(adir)
    save_weights(os.path.join(adir, "start.npz"), load_weights(a.start))
    json.dump(dict(task=a.task, budget=a.budget, start="start.npz"), open(os.path.join(adir, "budget_config.json"), "w"))
    ev = [PY, os.path.join(WSP, "eval_train.py"), "--attempt_dir", adir]
    subprocess.run(ev + ["--weights", os.path.join(adir, "start.npz"), "--init"], check=True, capture_output=True)
    rng = np.random.default_rng(a.seed)
    cur = dict_to_theta(load_weights(os.path.join(adir, "start.npz")))
    for k in range(a.budget):
        cand = cur + a.sigma * rng.standard_normal(cur.shape).astype(np.float32)
        p = os.path.join(adir, f"cand{k + 1:02d}.npz")
        save_weights(p, theta_to_dict(cand))
        out = json.loads(subprocess.run(ev + ["--weights", p], check=True, capture_output=True, text=True).stdout)
        if out["improved_over_best"]:
            cur = cand
        print(k + 1, out["score_key"], out["improved_over_best"], flush=True)
    print("best", json.load(open(os.path.join(adir, "best.json"))) if os.path.exists(os.path.join(adir, "best.json")) else "start")


if __name__ == "__main__":
    main()
