"""R6 LeWM S1 gates G1-G3 (results/lewm/protocol.json) for one trained run (LeWM or random-encoder control).

    ../bin/gpu_run ../.venv/bin/python lewm/evaluate.py runs/lewm_s1 [--ckpt final] [--n 4000]
-> results/lewm/eval_<run>.json

G1 collapse: embeddings of held-out-config frames (configs 270-299, eval mode): share of dims with std in
   [0.5, 2], effective rank (exp of the entropy of the normalised covariance spectrum), SIGReg statistic
   from the training log at step 1000 vs the last logged step.
G2 prediction: windows of 4 frames at the run's frameskip; predict z3 from true z0..z2 and action blocks
   0..2; R = mean||zhat3 - z3||^2 / mean||z2 - z3||^2 (copy-last-latent = 1). Held-out configs and p2_dev.
G3 action ranking: same held-out windows; the true block 2 vs 15 blocks taken from windows of other
   episodes; the candidate whose prediction lands nearest the true z3 wins. Chance = 1/16.
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import Part, Windows, heldout_select  # noqa: E402
from train import ROOT, build_model  # noqa: E402

NEG = 15


def load_run(run, ckpt="final"):
    st = json.loads((run / ckpt / "state.json").read_text())
    cfg = st["config"]
    model = build_model(cfg)
    model.load_weights(str(run / ckpt / "weights.safetensors"))
    model.eval()
    return model, cfg, st["step"]


def embed(model, frames, bs=512):
    out = []
    for i in range(0, len(frames), bs):
        z = model.encode_frames(mx.array(frames[i:i + bs]))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def predict_last(model, z, A):
    """z (B, 4, D) true latents, A (B, 4, a) -> zhat for frame 3 (B, D)."""
    zh = model.predictor(mx.array(z[:, :3]), mx.array(A[:, :3]))[:, 2]
    mx.eval(zh)
    return np.array(zh)


def window_latents(model, win, ids):
    F, A, k = win.get(ids)
    B = len(ids)
    z = embed(model, F.reshape(-1, *F.shape[2:])).reshape(B, win.T, -1)
    return z, A, k


def g2(model, win, ids, bs=256):
    se, cl = [], []
    for i in range(0, len(ids), bs):
        z, A, _ = window_latents(model, win, ids[i:i + bs])
        zh = predict_last(model, z, A)
        se.append(((zh - z[:, 3]) ** 2).sum(-1))
        cl.append(((z[:, 2] - z[:, 3]) ** 2).sum(-1))
    se, cl = np.concatenate(se), np.concatenate(cl)
    return {"R": float(se.mean() / cl.mean()), "pred_mse": float(se.mean()), "copy_last_mse": float(cl.mean()),
            "n": int(len(ids))}


def g3(model, win, ids, rng, bs=64):
    ok, n = 0, 0
    for i in range(0, len(ids), bs):
        z, A, k = window_latents(model, win, ids[i:i + bs])
        B = len(z)
        # negatives: action block 2 of windows from other episodes
        neg_ids = rng.integers(0, len(win), (B, NEG * 2))
        _, An, kn = win.get(neg_ids.reshape(-1))
        An, kn = An.reshape(B, NEG * 2, win.T, -1), kn.reshape(B, NEG * 2)
        cands = np.empty((B, NEG + 1, A.shape[-1]), np.float32)
        cands[:, 0] = A[:, 2]
        for b in range(B):
            other = [j for j in range(NEG * 2) if kn[b, j] != k[b]][:NEG]
            assert len(other) == NEG
            cands[b, 1:] = An[b, other, 2]
        zz = np.repeat(z, NEG + 1, axis=0)
        AA = np.repeat(A, NEG + 1, axis=0)
        AA[:, 2] = cands.reshape(-1, A.shape[-1])
        zh = predict_last(model, zz, AA).reshape(B, NEG + 1, -1)
        d = ((zh - z[:, None, 3]) ** 2).sum(-1)
        ok += int((d.argmin(1) == 0).sum())
        n += B
    return {"top1": ok / n, "chance": 1 / (NEG + 1), "n": n}


def horizon(model, parts, fs, select, ids_n, rng, H=5, bs=128):
    """Reported only (not a gate): autoregressive rollout from 3 true latents over H action blocks (history 3,
    sliding); per step h the error ratio ||zhat_{2+h} - z_{2+h}||^2 / ||z_2 - z_{2+h}||^2 (copy-last = 1)."""
    win = Windows(parts, T=3 + H, fs=fs, select=select)
    ids = rng.choice(len(win), min(ids_n, len(win)), replace=False)
    se, cl = np.zeros(H), np.zeros(H)
    for i in range(0, len(ids), bs):
        z, A, _ = window_latents(model, win, ids[i:i + bs])
        ctx = z[:, :3].copy()
        for h in range(H):
            zh = predict_last(model, np.concatenate([ctx, ctx[:, -1:]], 1), A[:, h:h + 4])
            se[h] += ((zh - z[:, 3 + h]) ** 2).sum()
            cl[h] += ((z[:, 2] - z[:, 3 + h]) ** 2).sum()
            ctx = np.concatenate([ctx[:, 1:], zh[:, None]], 1)
    return {"ratio_by_step": [float(x) for x in se / cl], "mse_by_step": [float(x) for x in se / len(ids)],
            "copy_last_mse_by_step": [float(x) for x in cl / len(ids)], "n": int(len(ids)), "block_ticks": fs}


def g1(model, run, parts, rng, n=20000):
    win = Windows(parts, T=1, fs=1, select=heldout_select)
    ids = np.sort(rng.choice(len(win), min(n, len(win)), replace=False))
    F, _, _ = win.get(ids)
    z = embed(model, F[:, 0])
    sd = z.std(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(z.T)), 0, None)
    p = ev / ev.sum()
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    log = [json.loads(l) for l in open(run / "log.jsonl")]
    s1000 = next((r["sigreg"] for r in log if r["step"] == 1000), None)
    return {"frac_std_in_0.5_2": float(((sd >= 0.5) & (sd <= 2)).mean()), "std_median": float(np.median(sd)),
            "std_min": float(sd.min()), "std_max": float(sd.max()), "effective_rank": erank,
            "sigreg_step1000": s1000, "sigreg_last": log[-1]["sigreg"], "last_step": log[-1]["step"], "n_frames": len(z)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n3", type=int, default=2000)
    a = ap.parse_args()
    run = ROOT / a.run
    model, cfg, step = load_run(run, a.ckpt)
    parts = [Part(p) for p in cfg["parts"].split(",")]
    rng = np.random.default_rng(2026)
    out = {"run": a.run, "ckpt": a.ckpt, "step": step, "random_encoder": cfg["random_encoder"], "fs": cfg["fs"]}
    if not cfg["random_encoder"]:
        out["G1"] = g1(model, run, parts, rng)
    hw = Windows(parts, T=cfg["T"], fs=cfg["fs"], select=heldout_select)
    dw = Windows([Part("dev")], T=cfg["T"], fs=cfg["fs"])
    ids_h = rng.choice(len(hw), min(a.n, len(hw)), replace=False)
    ids_d = rng.choice(len(dw), min(a.n, len(dw)), replace=False)
    out["G2_heldout"] = g2(model, hw, ids_h)
    out["G2_p2dev"] = g2(model, dw, ids_d)
    out["G3_heldout"] = g3(model, hw, ids_h[:a.n3], np.random.default_rng(7))
    out["G3_p2dev"] = g3(model, dw, ids_d[:a.n3], np.random.default_rng(7))
    out["horizon_heldout"] = horizon(model, parts, cfg["fs"], heldout_select, 2000, np.random.default_rng(11))
    out["horizon_p2dev"] = horizon(model, [Part("dev")], cfg["fs"], None, 2000, np.random.default_rng(11))
    out["heldout_windows"], out["heldout_episodes"] = len(hw), len(hw.eps)
    print(json.dumps(out, indent=1))
    res = ROOT / "results" / "lewm"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"eval_{Path(a.run).name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
