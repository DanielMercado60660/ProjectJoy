"""R8 Gate C: parity of the MLX ports (rep_ports.py) against torch reference modules transcribed from the
official sources (rep_torchref.py), with the official weights, on random inputs.

    MLX_ENABLE_TF32=0 ../.venv/bin/python r8_rep/rep_parity.py      -> results/r8/probes/parity.json

Tolerance (stated before the first run): every compared output must agree to max |mlx - torch| <= 1e-3 in
fp32 (TF32 off in MLX, torch fp32 on CPU). Compared outputs:
  DINOv2-S       all 257 normalised tokens of 3 random 224 px frames (+ the bicubic position-grid resize alone)
  V-JEPA 2.1-L   all 196 output tokens of 3 random frames
  LeWM x4        [CLS], the projected embedding (BatchNorm in eval mode) of 3 random frames, and the
                 predictor output (pred_proj(predictor(emb, action_encoder(a))), eval mode) for random
                 (2, 3, 192) embeddings and (2, 3, act_dim) actions; the official state_dict must load into
                 the torch reference with strict=True (all key names and shapes).
Torch is used here only to read published weights and run the reference modules.
"""
import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import rep_ports as P  # noqa: E402
import rep_torchref as R  # noqa: E402

TOL = 1e-3


def stat(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    return {"max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()), "ref_max_abs": float(np.abs(b).max()),
            "rel_err": float(d.max() / (np.abs(b).max() + 1e-12)), "pass": bool(d.max() <= TOL)}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (3, 224, 224, 3), dtype=np.uint8)
    x_nhwc = ((frames.astype(np.float32) / 255.0 - P.IMNET_MEAN) / P.IMNET_STD).astype(np.float32)
    x_nchw = torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2)))
    out = {"tolerance_max_abs": TOL, "mlx": mx.__version__, "torch": torch.__version__,
           "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"), "inputs": "uniform random uint8 frames, seed 0"}

    # ---- DINOv2-S
    t0 = time.time()
    w = {k: np.array(v) for k, v in mx.load(str(P.PRE / "dinov2-small" / "model.safetensors")).items()}
    ref = R.TorchDINOv2().load_hf({k: torch.from_numpy(v) for k, v in w.items()}).eval()
    pos = torch.from_numpy(w["embeddings.position_embeddings"])
    with torch.no_grad():
        ref_tok = ref(x_nchw).numpy()
        M, s = 37, (16 + 0.1) / 37
        tpos = F.interpolate(pos[:, 1:].reshape(1, M, M, -1).permute(0, 3, 1, 2), mode="bicubic", antialias=False,
                             scale_factor=(s, s)).permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1]).numpy()
    m = P.load_dinov2()
    tok = np.array(m.tokens(mx.array(x_nhwc)))
    out["dinov2_s"] = {"tokens": stat(tok, ref_tok), "pos_grid_bicubic": stat(P.dinov2_pos_224(w["embeddings.position_embeddings"])[:, 1:], tpos),
                       "params": P.param_count(m), "seconds": round(time.time() - t0, 1)}
    print("dinov2", json.dumps(out["dinov2_s"]), flush=True)
    del ref, m

    # ---- V-JEPA 2.1-L
    t0 = time.time()
    w = {k: np.array(v) for k, v in mx.load(str(P.PRE / "vjepa-2.1-vitl-mlx" / "encoder.safetensors")).items()}
    ref = R.TorchVJEPA21().load_mlx_release({k: torch.from_numpy(v) for k, v in w.items()}).eval()
    with torch.no_grad():
        ref_tok = ref(x_nchw[:, :, None]).numpy()
    del ref
    m = P.load_vjepa21()
    tok = np.array(m.tokens(mx.array(x_nhwc)))
    out["vjepa21_l"] = {"tokens": stat(tok, ref_tok), "params": P.param_count(m), "seconds": round(time.time() - t0, 1)}
    print("vjepa", json.dumps(out["vjepa21_l"]), flush=True)
    del m, w

    # ---- LeWM x4
    for name in P.LEWM_CKPTS:
        t0 = time.time()
        sd = torch.load(str(P.PRE / f"lewm-{name}" / "weights.pt"), map_location="cpu", weights_only=True)
        A = sd["action_encoder.patch_embed.weight"].shape[1]
        ref = R.TorchLeWM(act_dim=A)
        missing = ref.load_state_dict(sd, strict=True)
        ref.eval()
        emb_in = rng.standard_normal((2, 3, 192)).astype(np.float32)
        act_in = rng.standard_normal((2, 3, A)).astype(np.float32)
        with torch.no_grad():
            rc, re = ref.encode(x_nchw)
            rp = ref.predict(torch.from_numpy(emb_in), torch.from_numpy(act_in))
        m = P.load_lewm_official(name)
        mc = m.encoder(mx.array(x_nhwc))
        me = m.projector(mc)
        mp = m.predict(mx.array(emb_in), mx.array(act_in))
        out[f"lewm_{name}"] = {"strict_load": str(missing), "act_dim": int(A), "cls": stat(np.array(mc), rc.numpy()),
                               "embedding": stat(np.array(me), re.numpy()), "predictor": stat(np.array(mp), rp.numpy()),
                               "params": P.param_count(m), "seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(out[f"lewm_{name}"]), flush=True)
    checks = [v[k]["pass"] for v in out.values() if isinstance(v, dict) for k in v if isinstance(v[k], dict) and "pass" in v[k]]
    out["all_pass"] = bool(all(checks))
    out["n_checks"] = len(checks)
    res = ROOT / "results" / "r8" / "probes"
    res.mkdir(parents=True, exist_ok=True)
    (res / "parity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"all_pass": out["all_pass"], "n_checks": len(checks)}))


if __name__ == "__main__":
    main()
