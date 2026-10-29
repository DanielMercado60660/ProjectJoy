"""media/train_curve.png (training loss parts) and media/quality_vs_budget.png (GL iteration sweep vs neural)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
log = [json.loads(l) for l in open(os.path.join(ROOT, "train_log.jsonl"))]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for a, k in zip(ax, ["sc", "logmag", "mel_l1"]):
    a.plot([r["step"] for r in log], [r[k] for r in log], lw=1)
    a.set_title(k + " (train batch)"); a.set_xlabel("step"); a.set_yscale("log")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "train_curve.png"), dpi=100)

s = json.load(open(os.path.join(ROOT, "results", "metrics.json")))["summary"]
p3 = os.path.join(ROOT, "results_3k", "metrics.json")
s3 = json.load(open(p3))["summary"]["neural"] if os.path.exists(p3) else None
gl = sorted([k for k in s if k.startswith("gl")], key=lambda k: int(k[2:]))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, k, lab in zip(ax, ["mel_l1", "mrstft", "transient_pre_onset_excess_db"],
                     ["mel L1", "MR-STFT distance", "pre-onset excess dB (plosive clips)"]):
    a.plot([s[g]["rtf_under_load"] for g in gl], [s[g][k] for g in gl], "o-", label="Griffin-Lim (8..128 it)")
    for g in gl:
        a.annotate(g[2:], (s[g]["rtf_under_load"], s[g][k]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    if "neural" in s:
        a.plot([s["neural"]["rtf_under_load"]], [s["neural"][k]], "s", color="C3", label="neural (GPU)")
    if s3 is not None:
        a.plot([s3["rtf_under_load"]], [s3[k]], "D", color="C2", label="neural, 3000-clip ablation (GPU)")
    a.set_xscale("log"); a.set_xlabel("RTF, single run under load (log; see bench_rtf.json)"); a.set_title(lab)
ax[0].legend(fontsize=8)
fig.suptitle("quality vs inference budget, 24 held-out clips (lower is better on every axis)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "quality_vs_budget.png"), dpi=100)
print("plots written")
