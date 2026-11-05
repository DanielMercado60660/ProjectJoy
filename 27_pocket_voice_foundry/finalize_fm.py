"""Export a flow-matching snapshot for inference: fp16 weights + config + the duration pace / rounding fitted on the
*validation* split (first 100 LJ041-044 clips, the same rule as 27's finalize.py), so total length matches real speech.
Durations are deterministic (no sampling), so the pace does not depend on the ODE settings.

  ../.venv/bin/python finalize_fm.py --src runs/fm/lj/ckpt_snap8000 --out runs/fm/export/lj_s8000
"""
import argparse, json, os, shutil
import numpy as np
import mlx.core as mx
from model_fm import load_fm, cap_memory
from text import encode

ROOT = os.path.dirname(os.path.abspath(__file__))


def lengths(model, toks, pace, rounding):
    out = []
    for t in toks:
        tok = mx.array(np.array(t, np.int32)[None])
        h, _, tm, _ = model.encode(tok, mx.array([tok.shape[1]]), mx.array([0]))
        d = np.exp(np.array(model.dur(h, tm)[0])) * pace
        d = np.maximum(1, np.ceil(d) if rounding == "ceil" else np.round(d))
        out.append(d.sum())
    return np.array(out)


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_val", type=int, default=100)
    a = ap.parse_args()
    model, meta = load_fm(a.src + ".safetensors")
    z = np.load(os.path.join(ROOT, "cache2", "lj", "index.npz"))
    v = np.nonzero(z["splits"] == "val")[0][: a.n_val]
    toks = [encode(str(z["texts"][i])) for i in v]
    real = z["lens"][v]
    res = {r: float(np.exp(np.mean(np.log(lengths(model, toks, 1.0, r) / real)))) for r in ("round", "ceil")}
    rounding = min(res, key=lambda k: abs(np.log(res[k])))
    pace = 1.0 / res[rounding]
    after = float(np.exp(np.mean(np.log(lengths(model, toks, pace, rounding) / real))))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    shutil.copyfile(a.src + ".safetensors", a.out + ".safetensors")
    cfg = dict(meta)
    cfg.update(pace=pace, rounding=rounding, val_len_ratio_pace1=res, val_len_ratio_after=after,
               src=os.path.relpath(a.src, ROOT), dtype="float16")
    json.dump(cfg, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({k: cfg[k] for k in ["step", "pace", "rounding", "val_len_ratio_pace1", "val_len_ratio_after"]}))


if __name__ == "__main__":
    main()
