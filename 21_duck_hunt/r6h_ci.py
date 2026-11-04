"""Paired bootstrap CIs (same 500 held-out courses, same seeds) for the key hunter-game claims.
  ../.venv/bin/python r6h_ci.py -> results/r6h_ci.json"""
import json, os
import numpy as np
import hunter as H

HERE = os.path.dirname(os.path.abspath(__file__))


def hits(d, h, C):
    rng = np.random.default_rng(0)
    return H.play(C, H.make_agent(d, len(C), rng), H.make_agent(h, len(C), rng), seed=7)["hit"].astype(float)


def ci(a, b, n_boot=10000):
    d = a - b; rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return dict(a=float(a.mean()), b=float(b.mean()), diff=float(d.mean()),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def main():
    os.chdir(HERE)
    arch = json.load(open("results/r6h_selfplay_archive.json"))
    s1 = json.load(open("results/r6h_stage1.json"))
    dF, hF = arch["arch_d"][-1], arch["arch_h"][-1]
    C = H.heldout_courses(500)
    out = {}
    for duck in ["straight", "jink", "dodge", "dodge_jink"]:
        out[f"hit on {duck}: hunter_g8 - lead_tuned"] = ci(hits(duck, hF, C), hits(duck, "lead_tuned", C))
        out[f"hit on {duck}: hunter_s - lead_tuned"] = ci(hits(duck, s1["hunter_s"], C), hits(duck, "lead_tuned", C))
    for hunt in ["lead", "lead_ca", "lead_tuned", hF]:
        hn = hunt if isinstance(hunt, str) else hunt["name"]
        out[f"hit by {hn}: duck_g8 - straight"] = ci(hits(dF, hunt, C), hits("straight", hunt, C))
        out[f"hit by {hn}: duck_s - straight"] = ci(hits(s1["duck_s"], hunt, C), hits("straight", hunt, C))
    out["hit by hunter_g8: duck_g8 - duck_s"] = ci(hits(dF, hF, C), hits(s1["duck_s"], hF, C))
    for k, v in out.items():
        print(f"{k:45s} {v['a']:.3f} vs {v['b']:.3f}  diff {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    json.dump(out, open("results/r6h_ci.json", "w"), indent=1)


if __name__ == "__main__":
    main()
