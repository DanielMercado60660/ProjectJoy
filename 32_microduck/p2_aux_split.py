"""Phase 2: lamp-head accuracy on the first 40 dev demos (episodes/p2_dev), split into ticks where the bot
sees the lamp now (fresh sighting), remembered ticks, and unknown ticks; plus the confusion matrix.
    python p2_aux_split.py <weights>    # -> results/p2/aux_split_<run>_<ckpt>.json
"""
import json, sys, numpy as np, mlx.core as mx
import os; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
mx.set_default_device(mx.cpu)
from p2_learner import Policy2, policy_config
from learner import body_vec, instr_vec, prep_frame
w = sys.argv[1]; m = Policy2(**policy_config(w)); m.load_weights(w)
H0 = HERE + "/"
fresh = [0, 0]; stale = [0, 0]; unk = [0, 0]; pred_hist = np.zeros(3)
conf = np.zeros((3, 3), int)
for line in list(open(H0 + "episodes/p2_dev/index.jsonl"))[:40]:
    r = json.loads(line); z = np.load(r["path"]); H = json.loads(str(z["header"])); mem = json.loads(str(z["memory"]))
    T = min(len(mem), 400)
    F = np.stack([prep_frame(f) for f in z["frames"][:T]]); B = np.stack([body_vec(b) for b in z["body"][:T]])
    I = np.broadcast_to(instr_vec(H["instruction"]), (T, 9))
    _, _, aux, _ = m(mx.array(F[None].astype(np.float32) / 255), mx.array(B[None]), mx.array(I[None]))
    p = np.array(aux["lamp"][0]).argmax(-1)
    presses = [e["tick"] for e in H["events"] if e["event"] == "press"]
    for t in range(T):
        L = mem[t]["lamp"]; lp = max([q for q in presses if q <= t], default=-1)
        lab = 0 if (L is None or L[1] <= lp) else (1 if L[0] == "on" else 2)
        conf[lab, p[t]] += 1
        if lab == 0: unk[0] += p[t] == 0; unk[1] += 1
        elif L[1] == t: fresh[0] += p[t] == lab; fresh[1] += 1
        else: stale[0] += p[t] == lab; stale[1] += 1
res = {"weights": w, "in_view": [int(x) for x in fresh], "remembered": [int(x) for x in stale], "unknown": [int(x) for x in unk],
       "confusion_label_unk_on_off_by_pred": conf.tolist()}
print(res)
from pathlib import Path
pw = Path(w); json.dump(res, open(H0 + f"results/p2/aux_split_{pw.parent.parent.name}_{pw.parent.name}.json", "w"), indent=1)
