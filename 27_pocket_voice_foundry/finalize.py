"""Export the trained acoustic model for inference: weights only (fp16), config, mel stats, and the duration
pace / rounding chosen on the *validation* split (LJ041-044) so total length matches real speech on average.

  ../.venv/bin/python finalize.py --src runs/train_ckpt/acoustic_best    # -> checkpoints/acoustic/tts.{safetensors,json}
"""
import argparse, json, os
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
from model import Acoustic, infer

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "runs", "train_ckpt", "acoustic_best"))
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "acoustic", "tts"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--n_val", type=int, default=100)
    a = ap.parse_args()
    meta = json.load(open(a.src + ".json"))
    model = Acoustic(**meta["model"])
    model.load_weights(a.src + ".safetensors")
    model.eval()
    z = np.load(os.path.join(a.cache, "val.npz"))
    from text import encode
    lens = z["lens"][: a.n_val]
    toks = [encode(t) for t in z["texts"][: a.n_val]]
    res = {}
    for rounding in ["round", "ceil"]:
        r = [infer(model, mx.array(np.array(t, np.int32)[None]), 1.0, rounding)[0].shape[0] / l for t, l in zip(toks, lens)]
        res[rounding] = float(np.exp(np.mean(np.log(r))))
    rounding = min(res, key=lambda k: abs(np.log(res[k])))
    pace = 1.0 / res[rounding]
    r2 = [infer(model, mx.array(np.array(t, np.int32)[None]), pace, rounding)[0].shape[0] / l for t, l in zip(toks, lens)]
    after = float(np.exp(np.mean(np.log(r2))))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    model.update(tree_map(lambda p: p.astype(mx.float16), model.parameters()))
    model.save_weights(a.out + ".safetensors")
    cfg = {k: meta[k] for k in ["model", "step", "seed", "params", "mel_mean", "mel_std", "symbols"]}
    cfg.update(pace=pace, rounding=rounding, val_len_ratio_pace1=res, val_len_ratio_after=after, src=os.path.relpath(a.src, ROOT),
               wall_s_trained=meta["wall_s"], val_hist=meta.get("val_hist", []), dtype="float16")
    json.dump(cfg, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({k: cfg[k] for k in ["step", "pace", "rounding", "val_len_ratio_pace1", "val_len_ratio_after"]}))


if __name__ == "__main__":
    main()
