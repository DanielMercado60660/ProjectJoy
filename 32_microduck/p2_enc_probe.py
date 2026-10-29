"""Phase 2: linear probe for lamp on/off on a policy's frozen 192-d image features (fresh-sighting frames
from the train demos, bot-labelled; fit configs 0-199, test 200-299).
    python p2_enc_probe.py <weights> [...]    # -> results/p2/enc_probe.jsonl (appended)
"""
import json, sys, numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import os; HERE = os.path.dirname(os.path.abspath(__file__))
from PIL import Image
sys.path.insert(0, HERE)
from p2_learner import Policy2, policy_config
H0 = HERE + "/"
rows = [json.loads(l) for l in open(H0 + "results/m5/bc_index.jsonl")]
X = []; y = []; cid = []; rng = np.random.default_rng(0)
for r in rows:
    z = np.load(H0 + r["path"]); m = json.loads(str(z["memory"])); fr = z["frames"]
    ts = [t for t in range(len(m)) if m[t]["lamp"] is not None and m[t]["lamp"][1] == t]
    if len(ts) > 6: ts = list(rng.choice(ts, 6, replace=False))
    for t in ts:
        X.append(np.asarray(Image.fromarray(fr[t]).resize((64, 64), Image.Resampling.BOX))); y.append(m[t]["lamp"][0] == "on"); cid.append(int(r["config"].split("-")[1]))
X = np.stack(X).astype(np.float32) / 255; y = np.array(y); tr = np.array(cid) < 200
for w in sys.argv[1:]:
    try:
        pm = Policy2(**policy_config(w))
    except FileNotFoundError:
        from learner import Policy; pm = Policy()
    pm.load_weights(w)
    def feats(A):
        n = len(A)
        z = pm.encode(mx.array(A)[None], mx.zeros((1, n, 34)), mx.zeros((1, n, 9)))
        return np.array(z[0, :, :192])
    Fz = np.concatenate([feats(X[i:i + 256]) for i in range(0, len(X), 256)])
    mu, sd = Fz[tr].mean(0), Fz[tr].std(0) + 1e-6; Fz = (Fz - mu) / sd
    lin = nn.Linear(Fz.shape[1], 2); opt = optim.Adam(learning_rate=1e-2)
    vg = nn.value_and_grad(lin, lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean() + 1e-3 * (m.weight ** 2).sum())
    for _ in range(800):
        l, g = vg(lin, mx.array(Fz[tr]), mx.array(y[tr].astype(np.int32))); opt.update(lin, g); mx.eval(lin.parameters())
    acc = lambda M: float((np.array(mx.argmax(lin(mx.array(Fz[M])), -1)) == y[M]).mean())
    rec = {"weights": w, "test": round(acc(~tr), 3), "train": round(acc(tr), 3), "n_test": int((~tr).sum())}
    print(rec, flush=True)
    open(H0 + "results/p2/enc_probe.jsonl", "a").write(json.dumps(rec) + "\n")
