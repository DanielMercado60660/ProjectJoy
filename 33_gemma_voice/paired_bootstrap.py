"""Paired bootstrap of per-clip test CER: Gemma init (runs/r3) vs random init (runs/scratch).

  ../.venv/bin/python paired_bootstrap.py      # -> results/test_paired_bootstrap.json
Clip-mean CER (the headline 37.96 / 66.94 are corpus CER = total edits / total reference characters).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(run):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, run, "eval_test", "clips.jsonl")))}


def main():
    g, s = load("runs/r3"), load("runs/scratch")
    ids = sorted(g)
    gc, sc = np.array([g[i]["cer"] for i in ids]), np.array([s[i]["cer"] for i in ids])
    rng = np.random.default_rng(0)

    def boot(x, B=5000):
        m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
        return [round(100 * float(np.percentile(m, 2.5)), 1), round(100 * float(np.percentile(m, 97.5)), 1)]

    out = {"gemma_clipmean": round(100 * float(gc.mean()), 2), "gemma_ci": boot(gc),
           "scratch_clipmean": round(100 * float(sc.mean()), 2), "scratch_ci": boot(sc),
           "paired_diff_clipmean": round(100 * float((sc - gc).mean()), 2), "paired_diff_ci": boot(sc - gc),
           "gemma_better_clips": int((gc < sc).sum()), "ties": int((gc == sc).sum())}
    print(out)
    json.dump(out, open(os.path.join(HERE, "results", "test_paired_bootstrap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
