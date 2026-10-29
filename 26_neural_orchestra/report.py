"""Turn results/eval.json into markdown tables (results/tables.md) for RESULTS.md.

  ../.venv/bin/python report.py
"""
import json
import os
import sys

import data


def f(x, d=2):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main(path="results/eval.json", out="results/tables.md"):
    r = json.load(open(path))
    sets = r["sets"]
    L = []
    L.append(f"Data: {r['data']['train']} train / {r['data']['val']} val / {r['data']['test']} test distinct patterns. "
             f"Each generator: {r['n_per_style']} patterns per style x 8 styles, fixed seeds. "
             f"Style classifier accuracy on real test patterns: {100 * r['classifier_real_test_acc']:.1f}%.\n")
    L.append("### Held-out likelihood of the hit grid (nats per 2-bar pattern, lower is better)\n")
    L.append("| model | val NLL | test NLL |\n|---|---|---|")
    for k in r["nll_test"]:
        L.append(f"| {k} (exact) | {f(r['nll_val'][k])} | {f(r['nll_test'][k])} |")
    for k, v in r["vaes"].items():
        L.append(f"| {k} (IWAE bound, K={r.get('iwae_k', 200)}) | {f(v['nll_val_iwae'])} | {f(v['nll_test_iwae'])} |")
    L.append("\n### Generation: novelty, realism, control, pulse (averaged over styles)\n")
    L.append("| set | C2ST acc ↓ (0.5 = indistinguishable; mean of 3 seeds) | style acc ↑ | NN Hamming to train | exact copies % | within-style pairwise Hamming | unique % | bar-1 vs bar-2 Hamming | beat/off-beat density | kick on beat | cell-freq L1 vs test ↓ | sparse % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in sets.items():
        L.append(f"| {k} | {f(s['c2st_acc'], 3)} | {f(s['style_acc'], 3)} | {f(s['nn_hamming_mean'])} | {f(s['exact_copy_pct'], 1)} | "
                 f"{f(s['pairwise_hamming_within_style'], 1)} | {f(s['unique_pct'], 1)} | {f(s['bar_repeat_hamming'])} | "
                 f"{f(s['beat_offbeat_density_ratio'])} | {f(s['kick_on_beat_frac'])} | {f(s['profile_l1_vs_test'], 4)} | {f(s['empty_or_sparse_pct'], 1)} |")
    parts = r.get("demo_vae", r["selected_vae"]).rstrip("/").split("/")
    sel = parts[-2] if parts[-1] == "best" else parts[-1]
    mode = r.get("demo_mode", "sample")
    L.append(f"\n### Per-style pulse check: beat/off-beat onset density ratio and swing (odd minus even 16th offset), real test vs {sel}:{mode}\n")
    keys = ["real_test", f"{sel}:{mode}", "markov_bar", "bernoulli"]
    keys = [k for k in keys if k in sets]
    L.append("| style | " + " | ".join(f"{k} ratio" for k in keys) + " | " + " | ".join(f"{k} swing" for k in keys[:2]) + " | " + " | ".join(f"{k} style acc" for k in keys) + " |")
    L.append("|---|" + "---|" * (2 * len(keys) + 2))
    for st in data.STYLES:
        row = [f(sets[k]["per_style"][st]["beat_offbeat_density_ratio"]) for k in keys]
        row += [f(sets[k]["per_style"][st]["swing"], 3) for k in keys[:2]]
        row += [f(sets[k]["per_style"][st]["style_acc"], 2) for k in keys]
        L.append(f"| {st} | " + " | ".join(row) + " |")
    L.append("\n### Interpolation (64 test pairs of different styles, 9 latent points each, threshold decode; crossfade = data-space baseline)\n")
    L.append("| method | mean step (cells) | max/mean step | midpoints NN Hamming to train ↓ | midpoints bar-repeat Hamming ↓ | P(target style) non-decreasing % | endpoint error (cells) | sparse points % |")
    L.append("|---|---|---|---|---|---|---|---|")
    rows = [(k, v["interpolation"]["vae"]) for k, v in r["vaes"].items() if "interpolation" in v]
    rows.append(("crossfade (data space)", next(v["interpolation"]["crossfade"] for v in r["vaes"].values() if "interpolation" in v)))
    for k, v in rows:
        L.append(f"| {k} | {f(v['adjacent_hamming_mean'])} | {f(v['max_over_mean_step_ratio'])} | {f(v['midpoints_nn_hamming_to_train'])} | "
                 f"{f(v['midpoints_bar_repeat_hamming'])} | {f(100 * v['p_targetstyle_monotone_frac'], 1)} | {f(v['endpoint_hamming_error'])} | {f(v['sparse_point_pct'], 1)} |")
    L.append("\n### Style swap (encode a real test groove, decode it with each other style label)\n")
    L.append("| model | target-style classifier acc | cells changed | per target |\n|---|---|---|---|")
    for k, v in r["vaes"].items():
        if "style_transfer" in v:
            t = v["style_transfer"]
            L.append(f"| {k} | {100 * t['target_style_acc']:.1f}% | {t['mean_cells_changed']:.1f} | "
                     + ", ".join(f"{s} {100 * x:.0f}%" for s, x in t["per_target_acc"].items()) + " |")
    L.append(f"\nSelected by validation NLL: `{r['selected_vae']}`. Demo model: `{r.get('demo_vae', r['selected_vae'])}`, decode mode `{r.get('demo_mode')}`.")
    for k, v in r["vaes"].items():
        L.append(f"\n- {k}: config {v['config']}; test reconstruction hit-F1 {v['recon_test']['hit_f1']:.3f}, exact reconstructions {v['recon_test']['exact_recon_pct']:.1f}%")
    text = "\n".join(L) + "\n"
    open(out, "w").write(text)
    # splice into RESULTS.md between the markers, if present
    if out == "results/tables.md" and os.path.exists("RESULTS.md"):
        doc = open("RESULTS.md").read()
        a, b = "<!-- tables:start -->", "<!-- tables:end -->"
        if a in doc and b in doc:
            doc = doc[:doc.index(a) + len(a)] + "\n" + text + doc[doc.index(b):]
            open("RESULTS.md", "w").write(doc)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
