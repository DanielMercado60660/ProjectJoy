"""Train one generator (vae | gan | ddpm | flow) under an explicit, matched budget.

Budget = two caps, identical for every model:
  --budget_pflop : estimated training FLOPs (analytic forward count x 3 for fwd+bwd; GAN counted per pass)
  --budget_min   : training wall-clock minutes (excludes snapshot sampling / checkpoint I/O)
whichever binds first ends the run. One process stops after --session_min (<=40) and can be
resumed with --resume (weights, EMA, optimizer, step, RNG, budget counters, loss log are restored).

    ../bin/gpu_run ../.venv/bin/python train_gen.py --model flow --run runs/flow
"""
import argparse, json, os, time
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

import data as D
import models as M

T_DDPM = 1000


# --------------------------------------------------------------------------- diffusion helpers
def ddpm_schedule():
    betas = np.linspace(1e-4, 0.02, T_DDPM, dtype=np.float64)
    abar = np.cumprod(1 - betas)
    return mx.array(abar.astype(np.float32))


ABAR = None


def get_abar():
    global ABAR
    if ABAR is None:
        ABAR = ddpm_schedule()
    return ABAR


def build(model, cfg):
    res = cfg["res"]
    if model == "vae":
        return {"net": M.VAE(ch=cfg["ch"], zdim=cfg["zdim"], res=res)}
    if model == "gan":
        return {"G": M.Generator(zdim=cfg["zdim"], ch=cfg["ch"], res=res), "D": M.Discriminator(ch=cfg["ch"], res=res)}
    return {"net": M.UNet(ch=cfg["ch"], res=res, cond=cfg["cond"], mults=tuple(cfg["mults"]))}


def gen_module(nets, model):
    return nets["G"] if model == "gan" else nets["net"]


def flops_per_image(nets, model, cfg):
    """Training FLOPs per training image, analytic."""
    r = cfg["res"]
    x = mx.zeros((1, r, r, 3)); z = mx.zeros((1, cfg["zdim"]))
    if model == "vae":
        net = nets["net"]
        return 3 * M.count_flops(lambda a: net.decode(net.encode(a)[0]), x)
    if model == "gan":
        g = M.count_flops(nets["G"], z); d = M.count_flops(nets["D"], x)
        # D-step: G fwd + D fwd/bwd on real and fake; G-step: G fwd/bwd + D fwd/bwd(input grads)
        return g + 3 * d * 2 + 3 * g + 3 * d
    net = nets["net"]
    return 3 * M.count_flops(lambda a, b: net(a, b), x, mx.zeros((1,)))


def sampler_flops(nets, model, cfg):
    """Forward FLOPs of one network evaluation for one image (one NFE)."""
    r = cfg["res"]
    if model == "vae":
        return M.count_flops(nets["net"].decode, mx.zeros((1, cfg["zdim"])))
    if model == "gan":
        return M.count_flops(nets["G"], mx.zeros((1, cfg["zdim"])))
    return M.count_flops(lambda a, b: nets["net"](a, b), mx.zeros((1, r, r, 3)), mx.zeros((1,)))


# --------------------------------------------------------------------------- samplers (shared with eval)
def sample(model, net, n, cfg, key, steps=50, method=None, cond=None, guidance=0.0, batch=500):
    """Returns (n,H,W,3) float in [-1,1]. method: ddim | ddpm (ancestral) for ddpm; euler | heun for flow."""
    outs = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        k = mx.random.split(key, n // batch + 2)[i // batch]
        c = None if cond is None else cond[i:i + b]
        outs.append(_sample_batch(model, net, b, cfg, k, steps, method, c, guidance))
        mx.eval(outs[-1])
    return mx.concatenate(outs)


def _guided(net, x, t, cond, guidance):
    if cond is None:
        if getattr(net, "cond", False):  # conditional net, unconditional request -> all-null tokens
            return net(x, t, mx.full((x.shape[0], 6), -1, mx.int32))
        return net(x, t)
    if guidance == 0.0:
        return net(x, t, cond)
    out = net(mx.concatenate([x, x]), mx.concatenate([t, t]), mx.concatenate([cond, mx.full(cond.shape, -1, mx.int32)]))
    c, u = out[: x.shape[0]], out[x.shape[0]:]
    return u + (1 + guidance) * (c - u)


def _sample_batch(model, net, b, cfg, key, steps, method, cond, guidance):
    r = cfg["res"]
    if model == "vae":
        return net.decode(mx.random.normal((b, cfg["zdim"]), key=key))
    if model == "gan":
        return net(mx.random.normal((b, cfg["zdim"]), key=key))
    k1, k2 = mx.random.split(key)
    x = mx.random.normal((b, r, r, 3), key=k1)
    if model == "flow":
        method = method or "euler"
        ts = np.linspace(0, 1, steps + 1)
        for i in range(steps):
            t0, t1 = float(ts[i]), float(ts[i + 1])
            v = _guided(net, x, mx.full((b,), t0 * 1000.0), cond, guidance)
            if method == "heun" and i < steps - 1:
                xe = x + (t1 - t0) * v
                v2 = _guided(net, xe, mx.full((b,), t1 * 1000.0), cond, guidance)
                x = x + (t1 - t0) * 0.5 * (v + v2)
            else:
                x = x + (t1 - t0) * v
            mx.eval(x)
        return mx.clip(x, -1, 1)
    # ddpm / ddim on a strided subset of timesteps
    method = method or "ddim"
    abar = get_abar()
    ts = np.linspace(T_DDPM - 1, 0, steps).round().astype(int)
    keys = mx.random.split(k2, steps)
    for i, t in enumerate(ts):
        a_t = abar[int(t)]
        a_prev = abar[int(ts[i + 1])] if i + 1 < steps else mx.array(1.0)
        eps = _guided(net, x, mx.full((b,), float(t)), cond, guidance)
        x0 = mx.clip((x - mx.sqrt(1 - a_t) * eps) / mx.sqrt(a_t), -1, 1)
        eps = (x - mx.sqrt(a_t) * x0) / mx.sqrt(1 - a_t)
        if method == "ddim":
            x = mx.sqrt(a_prev) * x0 + mx.sqrt(1 - a_prev) * eps
        else:  # ancestral (DDIM eta=1)
            sig = mx.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)) if i + 1 < steps else mx.array(0.0)
            x = mx.sqrt(a_prev) * x0 + mx.sqrt(mx.maximum(1 - a_prev - sig ** 2, 0)) * eps \
                + sig * mx.random.normal(x.shape, key=keys[i])
        mx.eval(x)
    return mx.clip(x, -1, 1)


def to_uint8(x):
    return np.asarray(((mx.clip(x, -1, 1) + 1) * 127.5).round().astype(mx.uint8))


# --------------------------------------------------------------------------- losses
def make_steps(model, nets, opts, cfg):
    if model == "vae":
        net, opt = nets["net"], opts["net"]
        inv2s2 = 1.0 / (2 * cfg["vae_sigma"] ** 2)

        def lf(m, x, key):
            mu, lv = m.encode(x)
            z = mu + mx.exp(0.5 * lv) * mx.random.normal(mu.shape, key=key)
            xr = m.decode(z)
            rec = ((xr - x) ** 2).sum(axis=(1, 2, 3)) * inv2s2
            kl = 0.5 * (mu ** 2 + mx.exp(lv) - 1 - lv).sum(axis=1)
            return (rec + kl).mean(), (rec.mean(), kl.mean())

        vg = nn.value_and_grad(net, lf)
        state = [net.state, opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(x, cond, key):
            (l, (rec, kl)), g = vg(net, x, key)
            opt.update(net, g)
            return {"loss": l, "rec": rec, "kl": kl}
        return step, [net, opt]

    if model == "gan":
        G, Dn, oG, oD = nets["G"], nets["D"], opts["G"], opts["D"]

        def d_loss(d, real, fake):
            return (nn.softplus(-d(real)) + nn.softplus(d(fake))).mean()

        def g_loss(g, z):
            return nn.softplus(-Dn(g(z))).mean()

        dvg, gvg = nn.value_and_grad(Dn, d_loss), nn.value_and_grad(G, g_loss)

        def step(x, cond, key):
            k1, k2 = mx.random.split(key)
            fake = mx.stop_gradient(G(mx.random.normal((x.shape[0], cfg["zdim"]), key=k1)))
            ld, gd = dvg(Dn, x, fake); oD.update(Dn, gd)
            lg, gg = gvg(G, mx.random.normal((x.shape[0], cfg["zdim"]), key=k2)); oG.update(G, gg)
            return {"loss": lg + ld, "d_loss": ld, "g_loss": lg}
        return step, [G, Dn, oG, oD]

    net, opt = nets["net"], opts["net"]
    abar = get_abar()
    cdrop = cfg["cond_drop"]

    def lf(m, x, cond, key):
        k1, k2, k3 = mx.random.split(key, 3)
        b = x.shape[0]
        noise = mx.random.normal(x.shape, key=k1)
        if cond is not None:
            drop = mx.random.uniform(shape=(b, 1), key=k3) < cdrop
            cond = mx.where(drop, -1, cond)
        if model == "ddpm":
            t = mx.random.randint(0, T_DDPM, (b,), key=k2)
            a = abar[t][:, None, None, None]
            xt = mx.sqrt(a) * x + mx.sqrt(1 - a) * noise
            pred, target = m(xt, t.astype(mx.float32), cond), noise
        else:  # rectified flow: x_t = (1-t) noise + t data, v = data - noise
            t = mx.random.uniform(shape=(b,), key=k2)
            tt = t[:, None, None, None]
            xt = (1 - tt) * noise + tt * x
            pred, target = m(xt, t * 1000.0, cond), x - noise
        return ((pred - target) ** 2).mean()

    vg = nn.value_and_grad(net, lf)
    use_cond = cfg["cond"]
    state = [net.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, cond, key):
        l, g = vg(net, x, cond if use_cond else None, key)
        g, gnorm = optim.clip_grad_norm(g, 1.0)
        opt.update(net, g)
        return {"loss": l, "gnorm": gnorm}
    return step, [net, opt]


# --------------------------------------------------------------------------- checkpointing
def save_ckpt(path, nets, ema, opts, meta):
    os.makedirs(path, exist_ok=True)
    for k, n in nets.items():
        n.save_weights(os.path.join(path, f"{k}.safetensors"))
    mx.save_safetensors(os.path.join(path, "ema.safetensors"), dict(tree_flatten(ema)))
    for k, o in opts.items():
        st = dict(tree_flatten(o.state))
        mx.save_safetensors(os.path.join(path, f"opt_{k}.safetensors"), st)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_ckpt(path, nets, opts):
    for k, n in nets.items():
        n.load_weights(os.path.join(path, f"{k}.safetensors"))
    ema = tree_unflatten(list(mx.load(os.path.join(path, "ema.safetensors")).items()))
    for k, o in opts.items():
        o.state = tree_unflatten(list(mx.load(os.path.join(path, f"opt_{k}.safetensors")).items()))
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    return ema, meta


def load_generator(run_dir, ckpt="final", use_ema=True):
    """For eval/demo: returns (model_name, generator module (eval mode), cfg)."""
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        # clean clone: the whole runs/ tree is gitignored (root .gitignore) -> use checkpoints/<run name>/
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", os.path.basename(run_dir.rstrip("/")))
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    model = cfg["model"]
    nets = build(model, cfg)
    g = gen_module(nets, model)
    p = os.path.join(run_dir, ckpt)
    if os.path.exists(os.path.join(run_dir, "ema.safetensors")):  # flat export, e.g. checkpoints/flow/
        p = run_dir
    elif ckpt == "final" and not os.path.exists(os.path.join(p, "ema.safetensors")):
        # clean clone: run weights are gitignored -> fall back to the exported EMA in checkpoints/<run name>/
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", os.path.basename(run_dir.rstrip("/")))
        if os.path.exists(os.path.join(alt, "ema.safetensors")):
            p = alt
    if use_ema:
        g.update(tree_unflatten(list(mx.load(os.path.join(p, "ema.safetensors")).items())))
    else:
        g.load_weights(os.path.join(p, ("G" if model == "gan" else "net") + ".safetensors"))
    g.eval()
    return model, g, cfg


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vae", "gan", "ddpm", "flow"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--mults", type=str, default="1,2,2")
    ap.add_argument("--zdim", type=int, default=128)
    ap.add_argument("--vae_sigma", type=float, default=0.15)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--cond_drop", type=float, default=0.15)
    ap.add_argument("--budget_pflop", type=float, default=6.0)
    ap.add_argument("--budget_min", type=float, default=40.0)
    ap.add_argument("--session_min", type=float, default=40.0)
    ap.add_argument("--n_snaps", type=int, default=30)
    ap.add_argument("--n_ckpts", type=int, default=4, help="milestone checkpoints for stability eval")
    ap.add_argument("--n_train", type=int, default=0, help="subset of train set (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after this many steps")
    a = ap.parse_args()
    os.makedirs(a.run, exist_ok=True)
    cfg_path = os.path.join(a.run, "config.json")
    if a.resume:
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["session_min"] = a.session_min; cfg["max_steps"] = a.max_steps
    else:
        cfg = vars(a).copy()
        cfg["mults"] = [int(m) for m in a.mults.split(",")]
        if cfg["lr"] is None:
            cfg["lr"] = {"vae": 1e-3, "gan": 2e-4, "ddpm": 2e-4, "flow": 2e-4}[a.model]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=1)
    model = cfg["model"]
    mx.random.seed(cfg["seed"])
    xtr, ytr, _, _ = D.load(cfg["res"])
    if cfg["n_train"]:
        xtr, ytr = xtr[: cfg["n_train"]], ytr[: cfg["n_train"]]
    xtr_mx = mx.array(xtr)  # uint8 on device
    ytr_mx = mx.array(ytr.astype(np.int32))

    nets = build(model, cfg)
    if model == "gan":
        opts = {"G": optim.Adam(cfg["lr"], betas=[0.5, 0.999]), "D": optim.Adam(cfg["lr"], betas=[0.5, 0.999])}
    else:
        warm = optim.linear_schedule(1e-6, cfg["lr"], 500)
        opts = {"net": optim.Adam(optim.join_schedules([warm, lambda s: cfg["lr"]], [500]))}
    fpi = flops_per_image(nets, model, cfg)
    for n in nets.values():
        mx.eval(n.parameters())
    gnet = gen_module(nets, model)

    budget_flops = cfg["budget_pflop"] * 1e15
    fixed_key = mx.random.key(1234)
    fixed_cond = mx.array(D.sample_attrs(64, np.random.default_rng(7)).astype(np.int32)) if cfg["cond"] else None

    if a.resume:
        ema, meta = load_ckpt(os.path.join(a.run, "latest"), nets, opts)
        rng = np.random.default_rng(); rng.bit_generator.state = meta["np_rng"]
        print(f"resumed at step {meta['step']}, {meta['flops']/1e15:.3f} PFLOP, {meta['train_sec']/60:.1f} min")
    else:
        ema = tree_map(lambda p: mx.array(p), gnet.parameters())
        rng = np.random.default_rng(cfg["seed"])
        meta = {"step": 0, "flops": 0.0, "train_sec": 0.0, "images": 0, "log": [], "snaps": [], "ckpts": [],
                "flops_per_image": fpi, "done": False, "stop_reason": None}
    mx.eval(ema)

    step_fn, state = make_steps(model, nets, opts, cfg)


    def frac():
        return max(meta["flops"] / budget_flops, meta["train_sec"] / (cfg["budget_min"] * 60))

    def snapshot():
        saved = gnet.parameters()
        gnet.update(ema); gnet.eval()
        x = sample(model, gnet, 64, cfg, fixed_key, steps=50, cond=fixed_cond)
        gnet.update(saved); gnet.train()
        p = os.path.join(a.run, "snaps", f"{len(meta['snaps']):03d}.png")
        D.save_grid(to_uint8(x), p, nrow=8, scale=2)
        meta["snaps"].append({"step": meta["step"], "frac": frac(), "path": p})

    session_t0 = time.time()
    if not a.resume:
        snapshot()
    ema_decay = cfg["ema"]
    log_acc, t_last_latest = [], time.time()
    while True:
        f = frac()
        if f >= 1.0:
            meta["done"] = True
            meta["stop_reason"] = "flop_budget" if meta["flops"] >= budget_flops else "wall_budget"
            break
        if (time.time() - session_t0) / 60 > cfg["session_min"] or (cfg.get("max_steps") and meta["step"] >= cfg["max_steps"]):
            break
        t0 = time.time()
        idx = mx.array(rng.integers(0, len(xtr), size=cfg["batch"]))
        x = xtr_mx[idx].astype(mx.float32) / 127.5 - 1
        cond = ytr_mx[idx]
        key = mx.random.key(cfg["seed"] * 1_000_003 + meta["step"])
        out = step_fn(x, cond, key)
        # EMA update
        ema = tree_map(lambda e, p: e * ema_decay + p * (1 - ema_decay), ema, gnet.parameters())
        mx.eval(state, ema, out)
        meta["train_sec"] += time.time() - t0
        meta["step"] += 1; meta["images"] += cfg["batch"]; meta["flops"] += fpi * cfg["batch"]
        log_acc.append({k: float(v) for k, v in out.items()})
        if meta["step"] % 50 == 0:
            avg = {k: float(np.mean([d[k] for d in log_acc])) for k in log_acc[0]}
            avg.update(step=meta["step"], pflop=meta["flops"] / 1e15, min=meta["train_sec"] / 60)
            meta["log"].append(avg); log_acc = []
            if meta["step"] % 500 == 0:
                print(json.dumps(avg), flush=True)
        f = frac()
        if f >= (len(meta["snaps"])) / cfg["n_snaps"]:
            snapshot()
        if cfg["n_ckpts"] > 0 and f >= (len(meta["ckpts"]) + 1) / cfg["n_ckpts"] and len(meta["ckpts"]) + 1 < cfg["n_ckpts"]:
            name = f"ckpt_{len(meta['ckpts']) + 1}"
            meta["ckpts"].append({"name": name, "step": meta["step"], "frac": f, "pflop": meta["flops"] / 1e15,
                                  "min": meta["train_sec"] / 60})
            meta["np_rng"] = rng.bit_generator.state
            save_ckpt(os.path.join(a.run, name), nets, ema, opts, meta)
        if time.time() - t_last_latest > 300:
            meta["np_rng"] = rng.bit_generator.state
            save_ckpt(os.path.join(a.run, "latest"), nets, ema, opts, meta); t_last_latest = time.time()

    meta["np_rng"] = rng.bit_generator.state
    save_ckpt(os.path.join(a.run, "latest"), nets, ema, opts, meta)
    if meta["done"]:
        snapshot()
        meta["ckpts"].append({"name": "final", "step": meta["step"], "frac": 1.0, "pflop": meta["flops"] / 1e15,
                              "min": meta["train_sec"] / 60})
        save_ckpt(os.path.join(a.run, "final"), nets, ema, opts, meta)
        save_ckpt(os.path.join(a.run, "latest"), nets, ema, opts, meta)
    print(f"stopped: step={meta['step']} pflop={meta['flops']/1e15:.3f} train_min={meta['train_sec']/60:.1f} "
          f"done={meta['done']} reason={meta['stop_reason']}", flush=True)


if __name__ == "__main__":
    main()
