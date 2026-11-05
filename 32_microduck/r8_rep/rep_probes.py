"""R8 Gate C: the probe bake-off (results/r8/probes/protocol.json).

    ./r8_job.sh rep-probe ../.venv/bin/python r8_rep/rep_probes.py [arms...]    (default: every feature file + cnn)
-> results/r8/probes/probes.jsonl (one row per arm), table.json, table.md

Probes (fit = configs 0000-0269, test = 0270-0299; labels are simulator truth, labels only):
  linear (primary)    R6's probe: standardised features, linear softmax (lamp, plate) or linear regression on
                      the standardised target (bearing), Adam 1e-2 x 800 full batch, L2 1e-3 on the weight
  attentive           standardised grid tokens -> 1 learned query, Linear(d -> 64) keys / values, softmax
                      pooling, Linear head; AdamW 1e-3, wd 1e-4, 600 full-batch steps, seed 0
  cnn                 R6/P2 64 px CNN trained per task on the fit frames (1500 steps x 128, Adam 1e-3, seed 0)
"""
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PS = ROOT / "data" / "r8_rep" / "probeset"
FEAT = ROOT / "data" / "r8_rep" / "feats"
RES = ROOT / "results" / "r8" / "probes"
TASKS = ((0, "lamp"), (1, "plate"), (2, "bearing"))


def auc(score, y):
    """Mann-Whitney AUC with average ranks for ties (a constant score gives 0.5)."""
    from scipy.stats import rankdata
    y = y.astype(bool)
    ranks = rankdata(score)
    npos, nneg = y.sum(), len(y) - y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _std(F, tr):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    return (F - mu) / sd


def linear_probe(F, y, tr, regress=False, steps=800, sub=None):
    mx.random.seed(0)
    F = _std(F.astype(np.float32), tr)
    if regress:
        ym, ys = y[tr].mean(), y[tr].std() + 1e-9
        t = (y - ym) / ys
        lin = nn.Linear(F.shape[1], 1)
        lf = lambda m, a, b: ((m(a)[:, 0] - b) ** 2).mean() + 1e-3 * (m.weight ** 2).sum()  # noqa: E731
        Yt = mx.array(t[tr].astype(np.float32))
    else:
        lin = nn.Linear(F.shape[1], 2)
        lf = lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean() + 1e-3 * (m.weight ** 2).sum()  # noqa: E731
        Yt = mx.array(y[tr].astype(np.int32))
    opt = optim.Adam(learning_rate=1e-2)
    vg = nn.value_and_grad(lin, lf)
    Xt = mx.array(F[tr])
    for _ in range(steps):
        _, g = vg(lin, Xt, Yt)
        opt.update(lin, g)
        mx.eval(lin.parameters())
    out = np.array(lin(mx.array(F)))
    return _score(out, y, tr, regress, (ym, ys) if regress else None, sub)


def _score(out, y, tr, regress, norm, sub=None):
    """sub (reported only): a mask; adds the test accuracy on test items inside it (lamp >= 64 px in view)."""
    te = ~tr
    if regress:
        pred = out[:, 0] * norm[1] + norm[0]
        mae = np.degrees(np.abs(pred - y))
        base = np.degrees(np.abs(y[tr].mean() - y))
        return {"test_mae_deg": round(float(mae[te].mean()), 3), "train_mae_deg": round(float(mae[tr].mean()), 3),
                "baseline_mae_deg": round(float(base[te].mean()), 3),
                "test_r": round(float(np.corrcoef(pred[te], y[te])[0, 1]), 4), "n_test": int(te.sum())}
    s = out[:, 1] - out[:, 0]
    yb = y > 0.5
    extra = {}
    if sub is not None and (te & sub).any():
        extra = {"test_acc_px64": round(float(((s[te & sub] > 0) == yb[te & sub]).mean()), 4),
                 "n_test_px64": int((te & sub).sum()),
                 "test_acc_px16_63": round(float(((s[te & ~sub] > 0) == yb[te & ~sub]).mean()), 4) if (te & ~sub).any() else None}
    return {**extra, "test_acc": round(float(((s[te] > 0) == yb[te]).mean()), 4), "train_acc": round(float(((s[tr] > 0) == yb[tr]).mean()), 4),
            "test_auc": round(auc(s[te], yb[te]), 4), "n_test": int(te.sum()), "pos_rate_test": round(float(yb[te].mean()), 3),
            "majority_test_acc": round(float(max(yb[te].mean(), 1 - yb[te].mean())), 4)}


class AttnProbe(nn.Module):
    def __init__(self, d, out, dp=64):
        super().__init__()
        self.q = mx.random.normal((dp,)) * 0.02
        self.k, self.v, self.head = nn.Linear(d, dp), nn.Linear(d, dp), nn.Linear(dp, out)
        self.dp = dp

    def __call__(self, X):
        a = mx.softmax((self.k(X) @ self.q) / math.sqrt(self.dp), axis=-1)
        return self.head((a[..., None] * self.v(X)).sum(1))


def attentive_probe(G, y, tr, regress=False, steps=600, sub=None):
    mx.random.seed(0)
    G = G.astype(np.float32)
    mu = G[tr].reshape(-1, G.shape[-1]).mean(0)
    sd = G[tr].reshape(-1, G.shape[-1]).std(0) + 1e-6
    G = (G - mu) / sd
    if regress:
        ym, ys = y[tr].mean(), y[tr].std() + 1e-9
        Yt = mx.array(((y[tr] - ym) / ys).astype(np.float32))
        net = AttnProbe(G.shape[-1], 1)
        lf = lambda m, a, b: ((m(a)[:, 0] - b) ** 2).mean()  # noqa: E731
    else:
        Yt = mx.array(y[tr].astype(np.int32))
        net = AttnProbe(G.shape[-1], 2)
        lf = lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean()  # noqa: E731
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    vg = nn.value_and_grad(net, lf)
    Xt = mx.array(G[tr])
    for _ in range(steps):
        _, g = vg(net, Xt, Yt)
        opt.update(net, g)
        mx.eval(net.parameters(), opt.state)
    out = np.concatenate([np.array(net(mx.array(G[i:i + 1024]))) for i in range(0, len(G), 1024)])
    return _score(out, y, tr, regress, (ym, ys) if regress else None, sub)


class CNN(nn.Module):
    """R6/P2 lamp CNN (p2_lamp_probe.py, 64 px)."""

    def __init__(self, out):
        super().__init__()
        self.convs = [nn.Conv2d(3, 16, 5, stride=2, padding=2)] + [nn.Conv2d(16 if i == 0 else 32, 32, 3, stride=2, padding=1)
                                                                   for i in range(3)]
        self.fc = nn.Linear(4 * 4 * 32, out)

    def __call__(self, x):
        for c in self.convs:
            x = nn.relu(c(x))
        return self.fc(x.reshape(x.shape[0], -1))


def cnn_probe(X64, y, tr, regress=False, steps=1500, sub=None):
    mx.random.seed(0)
    A = X64.astype(np.float32) / 255
    if regress:
        ym, ys = y[tr].mean(), y[tr].std() + 1e-9
        T = ((y - ym) / ys).astype(np.float32)
        net = CNN(1)
        lf = lambda m, a, b: ((m(a)[:, 0] - b) ** 2).mean()  # noqa: E731
    else:
        T = y.astype(np.int32)
        net = CNN(2)
        lf = lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean()  # noqa: E731
    opt = optim.Adam(learning_rate=1e-3)
    vg = nn.value_and_grad(net, lf)
    r = np.random.default_rng(0)
    idx = np.nonzero(tr)[0]
    for _ in range(steps):
        b = r.choice(idx, 128)
        _, g = vg(net, mx.array(A[b]), mx.array(T[b]))
        opt.update(net, g)
        mx.eval(net.parameters())
    out = np.concatenate([np.array(net(mx.array(A[i:i + 512]))) for i in range(0, len(A), 512)])
    return _score(out, y, tr, regress, (ym, ys) if regress else None, sub)


def load_items():
    z = np.load(PS / "items.npz")
    return {k: z[k] for k in z.files}


def run_arm(arm, it):
    rec = {"arm": arm}
    if arm == "cnn64":
        f96 = np.load(PS / "frames96.npy")
        X64 = np.stack([np.asarray(Image.fromarray(x).resize((64, 64), Image.Resampling.BOX)) for x in f96])
        for t, name in TASKS:
            s = it["task"] == t
            sub = (it["px"][s] >= 64) if t == 0 else None
            rec[name] = {"cnn": cnn_probe(X64[it["frame"][s]], it["y"][s], it["cfg"][s] < 270, regress=(t == 2), sub=sub)}
        return rec
    z = np.load(FEAT / f"{arm}.npz")
    for t, name in TASKS:
        s = it["task"] == t
        fr, y, tr = it["frame"][s], it["y"][s], it["cfg"][s] < 270
        if t < 2:
            y = (y > 0.5).astype(np.float32)
        sub = (it["px"][s] >= 64) if t == 0 else None
        rec[name] = {"linear": linear_probe(z["glob"][fr], y, tr, regress=(t == 2), sub=sub)}
        if "proj" in z.files:
            rec[name]["linear_proj"] = linear_probe(z["proj"][fr], y, tr, regress=(t == 2), sub=sub)
        if "grid" in z.files:
            rec[name]["attentive"] = attentive_probe(z["grid"][fr], y, tr, regress=(t == 2), sub=sub)
    return rec


ORDER = ["dinov2_s", "vjepa21_l", "lewm_tworooms", "lewm_pusht", "lewm_cube", "lewm_reacher", "random_vit",
         "p2a", "r6_lewm", "r6_rand", "cnn64"]


def table(rows):
    lines = ["| arm | input | lamp acc (AUC) linear | plate-in-view AUC linear | bearing MAE deg linear | "
             "lamp acc attentive | plate AUC attentive | bearing MAE attentive | LeWM latent (proj): lamp / plate AUC / bearing |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a in ORDER:
        r = rows.get(a)
        if r is None:
            continue
        inp = "64 px" if a in ("p2a", "r6_lewm", "r6_rand", "cnn64") else "224 px"
        g = lambda t, k, m: r[t].get(k, {}).get(m)  # noqa: E731
        if a == "cnn64":
            lines.append(f"| {a} (supervised) | {inp} | {g('lamp','cnn','test_acc')} ({g('lamp','cnn','test_auc')}) | "
                         f"{g('plate','cnn','test_auc')} | {g('bearing','cnn','test_mae_deg')} | | | | |")
            continue
        pj = (f"{g('lamp','linear_proj','test_acc')} / {g('plate','linear_proj','test_auc')} / "
              f"{g('bearing','linear_proj','test_mae_deg')}") if "linear_proj" in r["lamp"] else ""
        lines.append(f"| {a} | {inp} | {g('lamp','linear','test_acc')} ({g('lamp','linear','test_auc')}) | "
                     f"{g('plate','linear','test_auc')} | {g('bearing','linear','test_mae_deg')} | "
                     f"{g('lamp','attentive','test_acc') or ''} | {g('plate','attentive','test_auc') or ''} | "
                     f"{g('bearing','attentive','test_mae_deg') or ''} | {pj} |")
    return "\n".join(lines)


def main():
    it = load_items()
    arms = sys.argv[1:] or [a for a in ORDER if a == "cnn64" or (FEAT / f"{a}.npz").exists()]
    RES.mkdir(parents=True, exist_ok=True)
    rows = {}
    if (RES / "probes.jsonl").exists():
        for line in open(RES / "probes.jsonl"):
            r = json.loads(line)
            rows[r["arm"]] = r
    for arm in arms:
        rec = run_arm(arm, it)
        rows[arm] = rec
        print(json.dumps(rec), flush=True)
        with open(RES / "probes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    (RES / "table.json").write_text(json.dumps(rows, indent=1))
    md = table(rows)
    (RES / "table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
