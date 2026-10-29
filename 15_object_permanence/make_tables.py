"""Print the markdown tables used in RESULTS.md from results/eval_<tag>.json."""
import json
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "base"
r = json.load(open(f"results/eval_{tag}.json"))
f = lambda x, k: "–" if x.get("n", 0) == 0 else f"{x[k]:.2f}"
for sp, v in r["splits"].items():
    print(f"\n**{sp}** (1000 sequences; {v['gru']['n_runs']} completed occlusions)\n")
    print("| tracker | visible err | hidden err (mean/median) | hidden NLL | hidden 1σ / 2σ cov | last-hidden err | reappear err | all-frame NLL | all 1σ / 2σ cov | ID switches /seq |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, x in v.items():
        h = x["hidden"]
        print(f"| {n} | {f(x['visible'],'err')} | {h['err']:.2f} / {h['err_med']:.2f} | {h['nll']:.2f} | {h['cov1']:.2f} / {h['cov2']:.2f} | "
              f"{f(x['last_hidden'],'err')} | {f(x['reappear'],'err')} | {x['all']['nll']:.2f} | {x['all']['cov1']:.2f} / {x['all']['cov2']:.2f} | "
              f"{x['identity']['switches_per_seq']:.3f} |")
    print("\nHidden-frame error (px) by occlusion length (frames fully hidden):\n")
    buckets = list(v["gru"]["by_occlusion_len"])
    print("| tracker | " + " | ".join(f"{b} (n={v['gru']['by_occlusion_len'][b]['runs']})" for b in buckets) + " |")
    print("|---|" + "---|" * len(buckets))
    for n, x in v.items():
        print(f"| {n} | " + " | ".join(f"{x['by_occlusion_len'][b]['hidden']['err']:.2f} (2σ {x['by_occlusion_len'][b]['hidden']['cov2']:.2f})" if b in x['by_occlusion_len'] else "–" for b in buckets) + " |")
    if "runs_turned" in v["gru"]:
        print("\nHidden occlusions split by whether the ball secretly changed velocity behind the wall:\n")
        print("| tracker | turned: hidden err | turned: NLL | turned: 2σ cov | turned: reappear err | straight: hidden err | straight: NLL | straight: 2σ cov | straight: reappear err |")
        print("|---|---|---|---|---|---|---|---|---|")
        for n, x in v.items():
            a, b = x["runs_turned"], x["runs_straight"]
            print(f"| {n} | {a['hidden']['err']:.2f} | {a['hidden']['nll']:.2f} | {a['hidden']['cov2']:.2f} | {a['reappear']['err']:.2f} | "
                  f"{b['hidden']['err']:.2f} | {b['hidden']['nll']:.2f} | {b['hidden']['cov2']:.2f} | {b['reappear']['err']:.2f} |")
print("\nKalman configs:", json.dumps({k: {kk: v[kk] for kk in ("q", "r", "bounce")} for k, v in r["kalman"].items()}))
