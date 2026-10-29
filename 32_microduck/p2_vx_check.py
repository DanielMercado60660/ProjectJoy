"""Phase 2: does the forward-speed head hedge between stand (0) and walk (0.3)?

    python p2_vx_check.py <weights> [...]     # -> results/p2/vx_check.jsonl (one line per weights)
Teacher-forced on the 80 dev bot demos (episodes/p2_dev): the policy's decoded vx on ticks where the bot
commands 0 and where it commands 0.3. Regression models decode vx from reg[0]; walk-or-stand models
send 0.3 when the walk logit > 0. Reports mean decoded vx and the share of ticks with vx >= 0.2
(the body's gait start, results/m0/body_frozen.json).
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
mx.set_default_device(mx.cpu)
from learner import body_vec, instr_vec, prep_frame  # noqa: E402
from p2_learner import Policy2, decode, policy_config  # noqa: E402

for w in sys.argv[1:]:
    cfg = policy_config(w)
    m = Policy2(**cfg)
    m.load_weights(w)
    vx = {"stand": [], "walk": []}
    for line in open(HERE / "episodes" / "p2_dev" / "index.jsonl"):
        r = json.loads(line)
        z = np.load(HERE / r["path"])
        H = json.loads(str(z["header"]))
        F = np.stack([prep_frame(f) for f in z["frames"]])
        B = np.stack([body_vec(b) for b in z["body"]])
        I = np.broadcast_to(instr_vec(H["instruction"]), (len(B), 9))
        reg, dl, aux, _ = m(mx.array(F[None].astype(np.float32) / 255.0), mx.array(B[None]), mx.array(I[None]))
        reg, dl = np.array(reg[0]), np.array(dl[0])
        walk = np.array(aux["walk"][0]) if cfg["vx_head"] == "binary" else None
        for t, c in enumerate(z["cmd_bot"]):
            key = "walk" if c[0] > 0.15 else "stand" if abs(c[0]) < 1e-6 else None
            if key:
                vx[key].append(decode(reg[t], dl[t], cfg["vx_head"], None if walk is None else float(walk[t])).vx)
    res = {"weights": w, "vx_head": cfg["vx_head"]}
    for k, v in vx.items():
        v = np.array(v)
        res[k] = {"ticks": int(len(v)), "mean_vx": round(float(v.mean()), 3), "share_ge_0.2": round(float((v >= 0.2).mean()), 3)}
    print(res)
    with open(HERE / "results" / "p2" / "vx_check.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
