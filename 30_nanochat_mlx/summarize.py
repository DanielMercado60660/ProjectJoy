"""Aggregate the data-efficiency runs (all seeds) into a markdown table: success on seen / held-out
pairs and final SFT val loss, pretrained vs scratch, for 300 and 1000 training episodes."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_evals():
    res = {}
    for p in ["media/sandbox_eval_extra.json", "media/sandbox_eval_seeds.json"] + sorted(glob.glob(os.path.join(HERE, "media/sandbox_eval_seeds_s*.json"))):
        p = os.path.join(HERE, p)
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                res.setdefault(k, {}).update(v)
    return res


def main():
    ev = load_evals()
    lines = ["| episodes | init | seeds | success seen pairs (per seed) | success held-out pairs (per seed) | mean seen | mean held-out | final SFT val loss (mean) |",
             "|---|---|---|---|---|---|---|---|"]
    for n in (300, 1000):
        for kind in ("pretrained", "scratch"):
            seen, held, vl = [], [], []
            for s in range(0, 10):
                key = f"lm[{kind},n={n}]" if s == 0 else f"lm[{kind},n={n},seed={s}]"
                if key not in ev or "test" not in ev[key]:
                    continue
                seen.append(ev[key]["test_seen"]["success_rate"])
                held.append(ev[key]["test"]["success_rate"])
                d = f"checkpoints/sft_n{n}" + ("_scratch" if kind == "scratch" else "") + ("" if s == 0 else f"_s{s}")
                m = os.path.join(HERE, d, "last", "meta.json")
                if os.path.exists(m):
                    vl.append(json.load(open(m))["history"][-1]["val_loss"])
            if not seen:
                continue
            f = lambda xs: " / ".join(f"{100*x:.0f}" for x in xs)
            lines.append(f"| {n} | {kind} | {len(seen)} | {f(seen)} | {f(held)} | **{100*np.mean(seen):.0f}%** | **{100*np.mean(held):.0f}%** | {np.mean(vl):.3f} |")
    print("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
