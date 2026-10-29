"""Loss curves -> media/loss_curve.png (pretraining) and media/sft_curve.png (SFT runs)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def ema(x, a=0.9):
    out, m = [], None
    for v in x:
        m = v if m is None else a * m + (1 - a) * v
        out.append(m)
    return np.array(out)


def pretrain(log="checkpoints/pilot/log.jsonl", out="media/loss_curve.png"):
    recs = read(os.path.join(HERE, log))
    if not recs:
        return
    # keep the last record per step (a resumed run may re-log steps)
    by = {}
    for r in recs:
        by[r["step"]] = {**by.get(r["step"], {}), **r}
    recs = [by[s] for s in sorted(by)]
    meta = json.load(open(os.path.join(HERE, os.path.dirname(log), "last", "meta.json")))
    tb = meta["args"]["total_batch"]
    steps = np.array([r["step"] for r in recs])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(steps * tb / 1e6, [r["loss"] for r in recs], color="#bbb", lw=0.6, label="train loss (per step)")
    ax[0].plot(steps * tb / 1e6, ema([r["loss"] for r in recs]), color="#1f5fa8", lw=1.6, label="train loss (EMA)")
    v = [r for r in recs if "val_loss" in r]
    ax[0].plot([r["step"] * tb / 1e6 for r in v], [r["val_loss"] for r in v], "o-", color="#d1495b", ms=3, lw=1.2, label="held-out val loss")
    ax[0].set_xlabel("training tokens (M)")
    ax[0].set_ylabel("cross-entropy (nats/token)")
    ax[0].set_ylim(top=min(10, max(r["val_loss"] for r in v) + 0.3) if v else None)
    ax[0].legend(frameon=False)
    ax[0].set_title(f"Pilot pretraining d{meta['config']['n_layer']} ({meta['config']['n_embd']}-dim), ClimbMix")
    ax[1].plot([r["step"] * tb / 1e6 for r in v], [r["val_bpb"] for r in v], "o-", color="#d1495b", ms=3)
    ax[1].set_xlabel("training tokens (M)")
    ax[1].set_ylabel("held-out bits per byte")
    ax[1].set_title("Validation bpb (tokenizer-independent)")
    if v:
        ax[1].set_ylim(min(r["val_bpb"] for r in v) - 0.05, min(r["val_bpb"] for r in v) + 1.0)
    for a in ax:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, out), dpi=130)
    print("wrote", out)


def sft(out="media/sft_curve.png"):
    """val loss on assistant tokens vs SFT step, pretrained (solid) vs scratch (dashed), per data size."""
    runs = [("checkpoints/sft", "8000 ep", "#1f5fa8"), ("checkpoints/sft_n1000", "1000 ep", "#2a9d8f"),
            ("checkpoints/sft_n300", "300 ep", "#e9a23b")]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    any_ = False
    for d, label, c in runs:
        for suffix, ls, kind in [("", "-", "pretrained"), ("_scratch", "--", "scratch")]:
            p = os.path.join(HERE, d + suffix, "last", "meta.json")
            if not os.path.exists(p):
                continue
            h = json.load(open(p))["history"]
            ax.plot([r["step"] for r in h], [r["val_loss"] for r in h], ls, marker="o", ms=2.5, color=c,
                    label=f"{label}, {kind}")
            any_ = True
    if not any_:
        return
    ax.set_yscale("log")
    ax.set_xlabel("SFT step (32 x 512 tokens)")
    ax.set_ylabel("val loss on assistant tokens (nats)")
    ax.set_title("SFT on Grid Courier: pretrained pilot (solid) vs random init (dashed)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, out), dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    pretrain()
    sft()
