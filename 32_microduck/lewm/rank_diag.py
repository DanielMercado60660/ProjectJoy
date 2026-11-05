"""R6 LeWM S1 (reported only): effective rank and per-batch SIGReg of the final embedding on training vs held-out
rooms, in eval-mode and train-mode BatchNorm, with a true Gaussian batch as reference.
    ../.venv/bin/python lewm/rank_diag.py runs/lewm_s1   -> results/lewm/rank_diag.json
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import Part, Windows, heldout_select, train_select  # noqa: E402
from evaluate import embed, load_run  # noqa: E402
from model import sigreg  # noqa: E402
from train import ROOT  # noqa: E402


def erank(z):
    ev = np.clip(np.linalg.eigvalsh(np.cov(z.T)), 0, None)
    p = ev / ev.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())), [round(float(x), 2) for x in np.sort(ev)[::-1][:5]]


def main():
    run = sys.argv[1]
    m, cfg, step = load_run(ROOT / run)
    parts = [Part(p) for p in cfg["parts"].split(",")]
    out = {"run": run, "step": step, "frames_per_set": 4096}
    for name, sel in (("train_rooms", train_select), ("heldout_rooms", heldout_select)):
        w = Windows(parts, T=1, fs=1, select=sel)
        F = w.get(np.random.default_rng(0).integers(0, len(w), 4096))[0][:, 0]
        m.eval()
        ze = embed(m, F)
        m.train()
        zt = np.concatenate([np.array(m.encode_frames(mx.array(F[i:i + 128]))) for i in range(0, 4096, 128)])
        m.eval()
        ke = mx.random.key(0)
        out[name] = {"eval_bn": dict(zip(("effective_rank", "top5_eig"), erank(ze))),
                     "train_bn": dict(zip(("effective_rank", "top5_eig"), erank(zt))),
                     "sigreg_batch128_eval_bn": float(sigreg(mx.array(ze[:128]), ke)),
                     "sigreg_batch128_train_bn": float(sigreg(mx.array(zt[:128]), ke))}
    out["sigreg_batch128_true_gaussian"] = float(np.mean([float(sigreg(mx.random.normal((128, 192), key=mx.random.key(s)),
                                                                        mx.random.key(100 + s))) for s in range(20)]))
    print(json.dumps(out, indent=1))
    (ROOT / "results" / "lewm" / "rank_diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
