"""Round 6 validation curves -> media/r6_val_curves.png: val100 UTMOS of scored snapshots (left) and the in-training
24-clip val GPE (right) per arm vs step, with the old vocoder (step 0 = the shared init) as reference.
  ../.venv/bin/python r6_plots.py
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
ARMS = [("recon", "no GAN (recon loss)", "#7a7a7a"), ("gan", "GAN (Vocos recipe)", "#2a6fb0"),
        ("gan_mrstft", "GAN + 10×MR-STFT", "#c2501f"), ("gan_phase", "GAN + phase loss (pilot)", "#8a5bb5"),
        ("recon_phase", "recon + phase loss", "#4d9a5b")]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    old = json.load(open(os.path.join(ROOT, "results_r6", "mos", "val100", "old.json")))["mos"]
    for arm, label, c in ARMS:
        pts = []
        for f in glob.glob(os.path.join(ROOT, "results_r6", "mos", "val100", f"{arm}_s*.json")):
            st = int(os.path.basename(f)[len(arm) + 2:-5])
            pts.append((st, json.load(open(f))["mos"]))
        if pts:
            pts = [(0, old)] + sorted(pts)
            ax[0].plot(*zip(*pts), "o-" if len(pts) > 2 else "o", color=c, label=label + ("" if len(pts) > 2 else " (end only)"))
        log = os.path.join(ROOT, "checkpoints", "r6_runs", arm, "log.jsonl")
        if os.path.exists(log):
            v = [json.loads(l) for l in open(log) if "val_gpe_pct" in l]
            if v:
                ax[1].plot([r["step"] for r in v], [r["val_gpe_pct"] for r in v], "o-", ms=3, color=c, label=label)
    ax[0].axhline(old, color="k", ls=":", lw=1, label="old vocoder (init)")
    ax[0].set_title("val100 UTMOS (real mel → vocoder)"); ax[0].set_xlabel("step"); ax[0].legend(fontsize=8)
    ax[1].set_title("in-training val GPE % (24 clips; lower is better)"); ax[1].set_xlabel("step")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ROOT, "media", "r6_val_curves.png")
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()
