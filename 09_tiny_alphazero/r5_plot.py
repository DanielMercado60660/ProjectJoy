"""Headless summary figure from audited round-5 receipts; no network/model execution."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/r5/mpl_cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    report = json.loads(Path("results/r5/turn2_analysis.json").read_text())
    methods = ("fixed", "learned", "entropy", "uncertainty", "disagreement", "rule", "logistic")
    labels = ("Fixed", "Learned", "Entropy", "Value", "Visits", "Rule", "Logistic")
    colors = {"main":"#7a8ea5", "ac_large":"#137c78"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    fig.suptitle("Learning risk did not improve batched search efficiency", fontsize=17, weight="bold")
    x = np.arange(len(methods))
    for shift, run in ((-.18, "main"), (.18, "ac_large")):
        vals = [100*report[run]["table"]["sim/"+m]["mistakes"]/report[run]["rows"]["test"] for m in methods]
        ax[0,0].bar(x+shift, vals, .34, color=colors[run], label=f"{run} (n={report[run]['rows']['test']:,})")
    ax[0,0].set(title="A  Equal simulations: errors per 100 boards", ylabel="Value-losing moves", xticks=x, xticklabels=labels)
    ax[0,0].legend(frameon=False, fontsize=9)
    vals = [report["ac_large"]["table"]["time/"+m]["avoided_per_ms"] for m in methods]
    ax[0,1].barh(labels[::-1], vals[::-1], color=["#137c78" if m!="fixed" else "#243d59" for m in methods[::-1]])
    ax[0,1].set(title="B  Equal time: net errors avoided / ms (AC)", xlabel="Relative to the 8-simulation probe")
    for method, color in (("learned", "#137c78"), ("rule", "#bd7a28"), ("logistic", "#526e9f")):
        bins = report["ac_large"]["calibration"][method]["all"]["bins"]
        ax[1,0].plot([b["predicted"] for b in bins], [b["observed"] for b in bins], "o-", color=color, label=method)
    ax[1,0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax[1,0].set(title="C  Calibration bins (AC; sparsely populated high risk)", xlabel="Predicted mistake probability", ylabel="Observed mistake rate", xlim=(0,1), ylim=(0,1.02))
    ax[1,0].legend(frameon=False, fontsize=9)
    for i, (run, method) in enumerate((("main","learned"),("main","logistic"),("ac_large","learned"),("ac_large","logistic"))):
        g = report[run]["games"][method]
        lo, hi = g["batch_bootstrap_95ci"]
        ax[1,1].errorbar(g["score"], 3-i, xerr=[[g["score"]-lo],[hi-g["score"]]], fmt="o", capsize=4, color=colors[run])
    ax[1,1].axvline(.5, ls="--", color="grey", linewidth=1)
    ax[1,1].set(title="D  Games vs fixed: score and batch-bootstrap 95% CI", xlabel="Win = 1; draw = ½; loss = 0", yticks=range(4),
                yticklabels=["AC logistic", "AC learned", "Main logistic", "Main learned"], xlim=(.475,.51))
    for a in ax.flat:
        a.spines[["top","right"]].set_visible(False)
        a.tick_params(labelsize=9)
    ax[0,0].tick_params(axis="x", rotation=25)
    fig.supxlabel("Main and AC share nested holdouts and game starts; they are not independent replications. Sources: results/r5/{main,ac_large}/ and turn2_analysis.json", fontsize=8)
    path = Path("media/r5_review.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
