"""Parity of the MLX ports against torch reference modules (r8_nav/torch_ref.py, written from the official sources)
loaded with the official weights, on random inputs. torch is used here only to read the published weights and run the
reference; the runtime path is MLX.

    ../.venv/bin/python -I r8_nav/parity.py        (from 32_microduck; writes results/r8/nav/parity.json)

Tolerance (fixed before the first run): for every compared output, max|mlx - torch| <= TOL x max(1, max|torch|),
TOL = 1e-3 for single forward passes and 2e-3 for the full 10-step DDPM sampler (the same initial and per-step noise
fed to both). Inputs: random uniform images in [0, 1] normalized with the ImageNet statistics (the models'
preprocessing), at the models' sizes, batch 4, seed 0; NoMaD with goal mask 0 and 1.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLX_ENABLE_TF32", "0")     # the runtime setting (r8_nav/mlx_models.py)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from r8_nav import mlx_models as M  # noqa: E402
from r8_nav import weights as W  # noqa: E402

TOL = 1e-3
TOL_SAMPLER = 2e-3
OUT = HERE / "results" / "r8" / "nav" / "parity.json"


def err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = float(np.max(np.abs(a - b)))
    scale = max(1.0, float(np.max(np.abs(b))))
    return {"max_abs_err": m, "max_abs_ref": float(np.max(np.abs(b))), "rel_err": m / scale}


def images(rng, n, c, size):
    w, h = size
    x = rng.uniform(0, 1, (n, h, w, c)).astype(np.float32)
    mean = np.tile(M.IMAGENET_MEAN, c // 3)
    std = np.tile(M.IMAGENET_STD, c // 3)
    return (x - mean) / std


def to_torch_nchw(x):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def run(batch=4, seed=0):
    import torch
    from r8_nav import torch_ref as R
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = {"tolerance": {"single_pass": TOL, "ddpm_sampler": TOL_SAMPLER,
                         "rule": "max|mlx - torch| <= tol * max(1, max|torch|)"},
           "inputs": f"uniform [0,1] images, ImageNet-normalized, batch {batch}, seed {seed}",
           "mlx_device": str(mx.default_device()), "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
           "torch": torch.__version__, "checks": {}}
    chk = out["checks"]

    # ------------------------------------------------------------------ ViNT
    ck = W.load_vint_checkpoint()
    vsd_t = {k: torch.from_numpy(v) for k, v in W.walk_state_dict(ck["model"]).items()}
    ref = R.ViNT().eval()
    ref.load_state_dict(vsd_t, strict=True)
    out["vint_load"] = {"strict": True, "n_tensors": len(vsd_t)}
    vint = M.ViNT()                                        # reads data/pretrained/vint/vint_weights.npz
    obs = images(rng, batch * 6, 3, M.ViNT.size).reshape(batch, 6, 64, 85, 3)
    goal = images(rng, batch, 3, M.ViNT.size)
    obs_t = torch.cat([to_torch_nchw(obs[:, i]) for i in range(6)], dim=1)    # (B, 18, 64, 85), oldest first
    with torch.no_grad():
        d_ref, a_ref = ref(obs_t, to_torch_nchw(goal))
    t0 = time.time()
    d_mlx, a_mlx = vint(mx.array(obs), mx.array(goal))
    mx.eval(d_mlx, a_mlx)
    out["vint_mlx_forward_s"] = round(time.time() - t0, 3)
    chk["vint_distance"] = err(np.array(d_mlx), d_ref[:, 0].numpy())
    chk["vint_waypoints_xy"] = err(np.array(a_mlx)[..., :2], a_ref[..., :2].numpy())
    chk["vint_waypoints_angle"] = err(np.array(a_mlx)[..., 2:], a_ref[..., 2:].numpy())
    out["vint_sample_output"] = {"distance": d_ref[:, 0].numpy().round(4).tolist(),
                                 "waypoint0": a_ref[:, 0].numpy().round(4).tolist()}

    # ------------------------------------------------------------------ NoMaD
    nsd = W.load_nomad_state()
    nref = R.NoMaD().eval()
    nref.load_state_dict({k: torch.from_numpy(v) for k, v in nsd.items()}, strict=True)
    out["nomad_load"] = {"strict": True, "n_tensors": len(nsd)}
    nomad = M.NoMaD()
    obs = images(rng, batch * 4, 3, M.NoMaD.size).reshape(batch, 4, 96, 96, 3)
    goal = images(rng, batch, 3, M.NoMaD.size)
    obs_t = torch.cat([to_torch_nchw(obs[:, i]) for i in range(4)], dim=1)
    tok = nomad.encode_obs(mx.array(obs.reshape(-1, 96, 96, 3))).reshape(batch, 4, -1)
    gtok = nomad.encode_goal(mx.array(obs[:, -1]), mx.array(goal))
    for mask in (0, 1):
        with torch.no_grad():
            c_ref = nref.vision_encoder(obs_t, to_torch_nchw(goal), torch.full((batch,), mask, dtype=torch.long))
            dd_ref = nref.dist_pred_net(c_ref)
        c_mlx = nomad.context_vector(tok, gtok, mx.full((batch,), bool(mask)))
        chk[f"nomad_context_mask{mask}"] = err(np.array(c_mlx), c_ref.numpy())
        chk[f"nomad_distance_mask{mask}"] = err(np.array(nomad.distance(c_mlx)), dd_ref[:, 0].numpy())
        if mask == 0:
            ctx_ref, ctx_mlx = c_ref, c_mlx
    # UNet on random samples / timesteps / conditions
    smp = rng.standard_normal((batch, 8, 2)).astype(np.float32)
    cond = rng.standard_normal((batch, 256)).astype(np.float32)
    ts = rng.integers(0, 10, batch)
    with torch.no_grad():
        e_ref = torch.cat([nref.noise_pred_net(torch.from_numpy(smp[i:i + 1]), torch.tensor(int(ts[i])),
                                               torch.from_numpy(cond[i:i + 1])) for i in range(batch)])
    e_mlx = nomad.unet(mx.array(smp), mx.array(ts.astype(np.int32)), mx.array(cond))
    chk["nomad_unet_eps"] = err(np.array(e_mlx), e_ref.numpy())
    # the full sampler: 8 samples per context, the same noise for both
    ns = 8
    noise = rng.standard_normal((11, batch * ns, 8, 2)).astype(np.float32)
    sched = R.DDPM(10)
    x = torch.from_numpy(noise[0])
    cr = ctx_ref.repeat_interleave(ns, dim=0)
    with torch.no_grad():
        for k, t in enumerate(sched.timesteps):
            eps = nref.noise_pred_net(x, t, cr)
            x = sched.step(eps, t, x, torch.from_numpy(noise[k + 1]))
    t0 = time.time()
    x_mlx = nomad.sample_actions(mx.repeat(ctx_mlx, ns, axis=0), mx.array(noise))
    mx.eval(x_mlx)
    out["nomad_mlx_sampler_s"] = round(time.time() - t0, 3)
    chk["nomad_sampler_ndeltas"] = err(np.array(x_mlx), x.numpy())
    chk["nomad_sampler_waypoints"] = err(M.NoMaD.to_waypoints(np.array(x_mlx)), M.NoMaD.to_waypoints(x.numpy()))
    chk["nomad_sampler_waypoints"]["tolerance"] = TOL_SAMPLER
    chk["nomad_sampler_ndeltas"]["tolerance"] = TOL_SAMPLER
    # schedule constants
    chk["ddpm_alphas_cumprod"] = err(nomad.alphas_cumprod, sched.alphas_cumprod.numpy())

    for k, v in chk.items():
        tol = v.get("tolerance", TOL)
        v["tolerance"] = tol
        v["pass"] = bool(v["rel_err"] <= tol)
    out["all_pass"] = all(v["pass"] for v in chk.values())
    out["params"] = {"vint_total": int(sum(v.numel() for v in ref.parameters())),
                     "vint_used_excl_fc_and_sa_layer_copy": int(sum(
                         v.numel() for k, v in ref.named_parameters() if "._fc." not in k and "sa_layer." not in k)),
                     "nomad_total": int(sum(v.numel() for v in nref.parameters())),
                     "nomad_used_excl_fc_and_sa_layer_copy": int(sum(
                         v.numel() for k, v in nref.named_parameters() if "._fc." not in k and "sa_layer." not in k))}
    return out


if __name__ == "__main__":
    res = run()
    if "--json-only" in sys.argv:
        print(json.dumps(res))
        sys.exit(0)
    # informational: the same check with MLX's default TF32 matmuls (not the runtime setting)
    env = dict(os.environ, MLX_ENABLE_TF32="1")
    p = subprocess.run([sys.executable, "-I", __file__, "--json-only"], env=env, capture_output=True, text=True)
    try:
        tf = json.loads(p.stdout.strip().splitlines()[-1])
        res["tf32_on_for_reference"] = {"note": "MLX default on this GPU (MLX_ENABLE_TF32 unset = TF32 matmuls); NOT "
                                                "the runtime setting", "all_pass": tf["all_pass"],
                                        "rel_err": {k: v["rel_err"] for k, v in tf["checks"].items()}}
    except Exception as e:  # noqa: BLE001
        res["tf32_on_for_reference"] = {"error": repr(e), "stderr": p.stderr[-500:]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for k, v in res["checks"].items():
        print(f"{k:28s} max_abs_err {v['max_abs_err']:.3e}  ref_max {v['max_abs_ref']:.3g}  rel {v['rel_err']:.3e}  "
              f"tol {v['tolerance']:g}  {'PASS' if v['pass'] else 'FAIL'}")
    print("all_pass", res["all_pass"], "| params", res["params"], "| mlx", res["mlx_device"],
          "| TF32 off; with TF32 on all_pass =", res["tf32_on_for_reference"].get("all_pass"))
