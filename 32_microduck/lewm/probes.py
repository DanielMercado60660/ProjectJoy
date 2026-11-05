"""R6 LeWM S1 gate G4: linear probes on frozen image features, same frames and probe for every encoder.

    ../bin/gpu_run ../.venv/bin/python lewm/probes.py lewm:runs/lewm_s1 rand:runs/lewm_rand \
        p2:runs/p2_a2/ckpt_016000/weights.safetensors ...
-> appends to results/lewm/probes.jsonl

Frames and lamp labels exactly as p2_enc_probe.py: the bot's fresh lamp sightings in the M5 BC demos
(results/m5/bc_index.jsonl), up to 6 per episode (rng 0), 96 px frames box-resized to 64 px; fit configs
0-199, test 200-299. Plate (button) in view: ticks where the bot has a confirmed button track; positive when
it was sighted this tick; up to 6 positive + 6 negative per episode (rng 1). Ball in view likewise (rng 2,
reported only). Probe: standardised features, linear 2-way softmax, Adam 1e-2 x 800 full-batch, L2 1e-3.
Labels are the bot's sensors-only beliefs, never simulator truth.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))


def r64(f):
    return np.asarray(Image.fromarray(f).resize((64, 64), Image.Resampling.BOX))


def build_sets():
    rows = [json.loads(l) for l in open(ROOT / "results/m5/bc_index.jsonl")]
    rng0, rng1, rng2 = (np.random.default_rng(s) for s in (0, 1, 2))
    sets = {k: {"X": [], "y": [], "cid": []} for k in ("lamp", "plate", "ball")}
    for r in rows:
        z = np.load(ROOT / r["path"])
        m = json.loads(str(z["memory"]))
        fr = z["frames"]
        cid = int(r["config"].split("-")[1])
        ts = [t for t in range(len(m)) if m[t]["lamp"] is not None and m[t]["lamp"][1] == t]
        if len(ts) > 6:
            ts = list(rng0.choice(ts, 6, replace=False))
        for t in ts:
            sets["lamp"]["X"].append(r64(fr[t])); sets["lamp"]["y"].append(m[t]["lamp"][0] == "on")
            sets["lamp"]["cid"].append(cid)
        for name, cls, rng in (("plate", "button", rng1), ("ball", "ball", rng2)):
            pos, neg = [], []
            for t in range(len(m)):
                tr = [x for x in m[t]["tracks"] if x[0] == cls]
                if tr:
                    (pos if any(x[4] == t for x in tr) else neg).append(t)
            for lst, lab in ((pos, True), (neg, False)):
                if len(lst) > 6:
                    lst = list(rng.choice(lst, 6, replace=False))
                for t in lst:
                    sets[name]["X"].append(r64(fr[t])); sets[name]["y"].append(lab); sets[name]["cid"].append(cid)
    return {k: {"X": np.stack(v["X"]), "y": np.array(v["y"]), "cid": np.array(v["cid"])} for k, v in sets.items()}


def featurizer(spec):
    kind, path = spec.split(":", 1)
    if kind == "p2":
        from p2_learner import Policy2, policy_config
        pm = Policy2(**policy_config(str(ROOT / path)))
        pm.load_weights(str(ROOT / path))

        def f(X):
            n = len(X)
            z = pm.encode(mx.array(X.astype(np.float32) / 255)[None], mx.zeros((1, n, 34)), mx.zeros((1, n, 9)))
            return np.array(z[0, :, :192])
        return f
    from evaluate import load_run
    model, _, _ = load_run(ROOT / path)

    def g(X):
        if kind.endswith("cls"):                    # "lewmcls:" / "randcls:": pre-projector [CLS], reported only
            x = (mx.array(X).astype(mx.float32) / 255.0 - 0.45) / 0.25
            z = model.encoder.cls_features(x)
        else:
            z = model.encode_frames(mx.array(X))
        mx.eval(z)
        return np.array(z)
    return g


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def probe(F, y, tr):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    F = (F - mu) / sd
    lin = nn.Linear(F.shape[1], 2)
    opt = optim.Adam(learning_rate=1e-2)
    vg = nn.value_and_grad(lin, lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean() + 1e-3 * (m.weight ** 2).sum())
    Xt, yt = mx.array(F[tr]), mx.array(y[tr].astype(np.int32))
    for _ in range(800):
        _, g = vg(lin, Xt, yt); opt.update(lin, g); mx.eval(lin.parameters())
    lo = np.array(lin(mx.array(F)))
    s = lo[:, 1] - lo[:, 0]
    acc = lambda M: float(((s[M] > 0) == y[M]).mean())
    return {"test_acc": round(acc(~tr), 4), "train_acc": round(acc(tr), 4), "test_auc": round(auc(s[~tr], y[~tr]), 4),
            "n_test": int((~tr).sum()), "pos_rate_test": round(float(y[~tr].mean()), 3)}


def main():
    sets = build_sets()
    print({k: (len(v["y"]), round(float(v["y"].mean()), 3)) for k, v in sets.items()}, flush=True)
    out = ROOT / "results/lewm/probes.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[1:]:
        mx.random.seed(0)
        f = featurizer(spec)
        rec = {"encoder": spec}
        for name, s in sets.items():
            F = np.concatenate([f(s["X"][i:i + 256]) for i in range(0, len(s["X"]), 256)])
            rec[name] = probe(F, s["y"], s["cid"] < 200)
        print(json.dumps(rec), flush=True)
        with open(out, "a") as g:
            g.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
