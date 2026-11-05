"""R8 Gate C: training curves of the LeWM arms (log.jsonl every 50 steps, val.jsonl every 1,000 steps).

    ../.venv/bin/python r8_rep/rep_curves.py runs/r8_lewm_ckpt runs/r8_lewm_scratch [runs/r8_lewm_rand]
-> results/r8/lewm_ckpt/curves.png, curves.json (the plotted numbers, 500-step means)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "r8" / "lewm_ckpt"


def read(run):
    log = [json.loads(line) for line in open(run / "log.jsonl")]
    seen, rows = set(), []
    for r in log:                       # a resumed invocation can re-log a step: keep the last copy
        if r["step"] in seen:
            rows = [x for x in rows if x["step"] != r["step"]]
        seen.add(r["step"])
        rows.append(r)
    val = [json.loads(line) for line in open(run / "val.jsonl")] if (run / "val.jsonl").exists() else []
    return sorted(rows, key=lambda r: r["step"]), val


def main():
    runs = [ROOT / r for r in sys.argv[1:]]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    out = {}
    for run in runs:
        rows, val = read(run)
        s = np.array([r["step"] for r in rows])
        name = run.name.replace("r8_lewm_", "")
        for i, key in enumerate(("pred", "sigreg", "loss")):
            y = np.array([r[key] for r in rows])
            ax[i].plot(s, y, label=name, lw=1)
            ax[i].set_title(key)
            ax[i].set_xlabel("step")
        if val:
            ax[3].plot([v["step"] for v in val], [v["heldout_R_256"] for v in val], "o-", label=name)
        ax[3].set_title("held-out one-block R vs copy-last (256 windows)")
        ax[3].axhline(0.70, color="grey", ls="--", lw=0.8)
        ax[3].set_xlabel("step")
        k = 10
        out[name] = {"step": s[k - 1::k].tolist(), "pred_500": [float(np.mean([r["pred"] for r in rows[i:i + k]])) for i in range(0, len(rows) - k + 1, k)],
                     "sigreg_500": [float(np.mean([r["sigreg"] for r in rows[i:i + k]])) for i in range(0, len(rows) - k + 1, k)],
                     "val_R": val, "steps_s_median": float(np.median([r["steps_s"] for r in rows]))}
    ax[0].set_yscale("log")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    RES.mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / "curves.png", dpi=110)
    (RES / "curves.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {"last_pred_500": v["pred_500"][-1] if v["pred_500"] else None, "val_R": v["val_R"][-3:]}
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
