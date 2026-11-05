"""R8 Gate C: frozen-encoder features of the probe set (results/r8/probes/protocol.json, 'features').

    ./r8_job.sh rep-feat ../.venv/bin/python r8_rep/rep_features.py dinov2_s vjepa21_l lewm_tworooms ... \
        random_vit p2a r6_lewm r6_rand
-> data/r8_rep/feats/<arm>.npz with
   glob (N, d)  primary global feature: [CLS ; mean patch] (V-JEPA: mean patch; 64 px rows: R6's embedding)
   grid (N, n, d) float16  2x2-pooled patch grid (+ CLS first when the model has one); 224 px arms only
   proj (N, 192)  LeWM checkpoints only: the projector output (BatchNorm eval mode)
Also usable on any uint8 frame array through `featurize(arm, frames224, frames96)`.
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import rep_ports as P  # noqa: E402

PS = ROOT / "data" / "r8_rep" / "probeset"
FEAT = ROOT / "data" / "r8_rep" / "feats"
ARMS_224 = ("dinov2_s", "vjepa21_l") + tuple(f"lewm_{n}" for n in P.LEWM_CKPTS) + ("random_vit",)
ARMS_64 = {"p2a": "p2:runs/p2_a2/ckpt_016000/weights.safetensors", "r6_lewm": "lewm:runs/lewm_s1",
           "r6_rand": "rand:runs/lewm_rand"}


def pool2(tok, g):
    """(N, g*g, d) patch tokens -> (N, (g//2)^2, d) by 2x2 average pooling."""
    N, _, d = tok.shape
    t = tok.reshape(N, g // 2, 2, g // 2, 2, d).mean((2, 4))
    return t.reshape(N, -1, d)


def model_for(arm):
    if arm == "dinov2_s":
        m = P.load_dinov2()
        return lambda x: m(x), True, 16
    if arm == "vjepa21_l":
        m = P.load_vjepa21()
        return lambda x: m(x), False, 14
    if arm.startswith("lewm_"):
        m = P.load_lewm_official(arm[5:])
        return m, True, 16
    if arm == "random_vit":
        m = P.RandomViT(seed=0)
        m.eval()
        return lambda x: m(x), True, 16
    raise ValueError(arm)


def featurize224(arm, frames, bs=32):
    f, has_cls, g = model_for(arm)
    glob, grid, proj = [], [], []
    for i in range(0, len(frames), bs):
        x = mx.array(np.ascontiguousarray(frames[i:i + bs]))
        if arm.startswith("lewm_"):
            tok = f.encoder.tokens(P.normalize_u8(x))
            pr = f.projector(tok[:, 0])
            mx.eval(pr)
            proj.append(np.array(pr))
        else:
            tok = f(x)
        patches = tok[:, 1:] if has_cls else tok
        gl = mx.concatenate([tok[:, 0], patches.mean(1)], -1) if has_cls else patches.mean(1)
        gr = pool2(patches, g)
        if has_cls:
            gr = mx.concatenate([tok[:, :1], gr], 1)
        mx.eval(gl, gr)
        glob.append(np.array(gl))
        grid.append(np.array(gr.astype(mx.float16)))
    out = {"glob": np.concatenate(glob), "grid": np.concatenate(grid)}
    if proj:
        out["proj"] = np.concatenate(proj)
    return out


def featurize64(arm, frames96, bs=256):
    sys.path.insert(0, str(ROOT / "lewm"))
    from probes import featurizer      # R6's featurizer (lewm/probes.py): same code as the R6 rows
    mx.random.seed(0)
    f = featurizer(ARMS_64[arm])
    X = np.stack([np.asarray(Image.fromarray(x).resize((64, 64), Image.Resampling.BOX)) for x in frames96])
    return {"glob": np.concatenate([f(X[i:i + bs]) for i in range(0, len(X), bs)])}


def main():
    FEAT.mkdir(parents=True, exist_ok=True)
    f224 = np.load(PS / "frames224.npy", mmap_mode="r")
    f96 = np.load(PS / "frames96.npy")
    log = []
    for arm in sys.argv[1:]:
        t0 = time.time()
        out = featurize64(arm, f96) if arm in ARMS_64 else featurize224(arm, f224)
        np.savez(FEAT / f"{arm}.npz", **out)
        rec = {"arm": arm, "n": len(out["glob"]), "glob_dim": out["glob"].shape[1],
               "grid": list(out["grid"].shape[1:]) if "grid" in out else None, "seconds": round(time.time() - t0, 1),
               "frames_per_s": round(len(out["glob"]) / (time.time() - t0), 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    with open(ROOT / "results" / "r8" / "probes" / "features_log.jsonl", "a") as g:
        for r in log:
            g.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
