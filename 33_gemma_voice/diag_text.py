"""Does the model use the text? Teacher-forced val loss per codebook with (a) the correct text,
(b) text swapped from another random val clip, (c) BOS only. If (a) ~ (b), the text is ignored.

  ../bin/gpu_run ../.venv/bin/python diag_text.py --run runs/main --ckpt best     # -> results/diag_text_<run>_<ckpt>.json
"""
import argparse, json, os, sys
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import load_run  # noqa: E402
from train import load_split, val_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--val_n", type=int, default=300)
    a = ap.parse_args()
    run = os.path.join(HERE, a.run)
    model, meta = load_run(run, a.ckpt)
    nq = meta["args"]["nq"]
    val, _ = load_split("val", nq, text_mode=meta["args"].get("text", "subword"))
    val = val[: a.val_n]
    perm = np.random.default_rng(0).permutation(len(val))
    variants = {"correct": val,
                "swapped": [(val[p][0], g) for (_, g), p in zip(val, perm)],
                "bos_only": [([2], g) for _, g in val]}
    out = {"run": a.run, "ckpt": a.ckpt, "step": meta["step"]}
    for k, items in variants.items():
        vl, per = val_loss(model, items, nq, 12000)
        out[k] = {"mean": round(vl, 4), "cb0": round(per[0], 4), "cb1": round(per[1], 4), "cb": [round(x, 3) for x in per]}
        print(k, out[k]["mean"], out[k]["cb0"], out[k]["cb1"], flush=True)
    clips = os.path.join(run, f"eval_val20_{a.ckpt}", "clips.jsonl")
    if os.path.exists(clips):
        r = [json.loads(l) for l in open(clips)]
        out["len_corr_val20"] = round(float(np.corrcoef([x["frames"] for x in r], [x["real_frames"] for x in r])[0, 1]), 3)
        print("length corr (generated vs real frames, val20):", out["len_corr_val20"])
    json.dump(out, open(os.path.join(HERE, "results", f"diag_text_{os.path.basename(a.run)}_{a.ckpt}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
