"""Phase 2, step 1: is the lamp's on/off state readable from the frame at 32/64/96? Samples: ticks where the bot has a fresh
confirmed lamp sighting (memory lamp tick == t), labelled by the bot's belief. Split by config: train 0-199 fit,
200-299 test."""
import json, sys, numpy as np
import os; HERE = os.path.dirname(os.path.abspath(__file__))
from PIL import Image
sys.path.insert(0, HERE)
rows = [json.loads(l) for l in open(HERE + "/results/m5/bc_index.jsonl")]
X = {32: [], 64: [], 96: []}; y = []; cid = []
rng = np.random.default_rng(0)
for r in rows:
    z = np.load(HERE + "/" + r["path"])
    m = json.loads(str(z["memory"])); fr = z["frames"]
    ts = [t for t in range(len(m)) if m[t]["lamp"] is not None and m[t]["lamp"][1] == t]
    if len(ts) > 6: ts = list(rng.choice(ts, 6, replace=False))
    for t in ts:
        f = fr[t]
        X[96].append(f)
        for s in (32, 64):
            X[s].append(np.asarray(Image.fromarray(f).resize((s, s), Image.Resampling.BOX)))
        y.append(m[t]["lamp"][0] == "on"); cid.append(int(r["config"].split("-")[1]))
y = np.array(y); cid = np.array(cid); tr = cid < 200
print("samples", len(y), "on frac", y.mean().round(3), "train/test", tr.sum(), (~tr).sum())
out = {"samples": int(len(y)), "on_frac": float(y.mean()), "train": int(tr.sum()), "test": int((~tr).sum())}
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
class Net(nn.Module):
    def __init__(self, s):
        super().__init__()
        n = {32: 3, 64: 4, 96: 4}[s]
        self.convs = [nn.Conv2d(3, 16, 5, stride=2, padding=2)] + [nn.Conv2d(16 if i == 0 else 32, 32, 3, stride=2, padding=1) for i in range(n - 1)]
        last = s // 2 ** n
        self.fc = nn.Linear(last * last * 32, 2)
    def __call__(self, x):
        for c in self.convs: x = nn.relu(c(x))
        return self.fc(x.reshape(x.shape[0], -1))
for s in (32, 64, 96):
    A = np.stack(X[s]).astype(np.float32) / 255
    mx.random.seed(0); net = Net(s); opt = optim.Adam(learning_rate=1e-3)
    lf = lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean()
    vg = nn.value_and_grad(net, lf); r = np.random.default_rng(0); idx = np.where(tr)[0]
    for step in range(1500):
        b = r.choice(idx, 128)
        l, g = vg(net, mx.array(A[b]), mx.array(y[b].astype(np.int32))); opt.update(net, g); mx.eval(net.parameters())
    pred = lambda M: np.concatenate([np.array(mx.argmax(net(mx.array(A[M][i:i + 512])), -1)) for i in range(0, M.sum(), 512)])
    out[s] = {"test_acc": round(float((pred(~tr) == y[~tr]).mean()), 3), "train_acc": round(float((pred(tr) == y[tr]).mean()), 3)}
    print(s, out[s], flush=True)
json.dump(out, open(HERE + "/results/p2/lamp_probe.json", "w"), indent=1)
