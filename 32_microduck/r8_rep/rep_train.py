"""R8 Gate C, part 2: LeWM at 224 px on duck frames, checkpoint init vs scratch (results/r8/lewm_ckpt/protocol.json).

    ./r8_job.sh lewm-ckpt  ../.venv/bin/python r8_rep/rep_train.py --run runs/r8_lewm_ckpt --init ckpt:<name> --steps N --batch B
    ./r8_job.sh lewm-scr   ../.venv/bin/python r8_rep/rep_train.py --run runs/r8_lewm_scratch --init scratch --steps N --batch B
    ./r8_job.sh lewm-rand  ../.venv/bin/python r8_rep/rep_train.py --run runs/r8_lewm_rand --init random_encoder --steps N --batch B
    ... --resume   (continues a run; each invocation stops itself before --guard-min minutes)

Model: the official LeWM (rep_ports.LeWMOfficial: ViT-tiny/14 at 224 px, BN projector, ARPredictor with AdaLN
action conditioning, dropout 0.1, pred_proj), action input = 5 ticks x 6 z-scored commands = 30-d, history 3.
Init:
  ckpt:<name>     every weight from the official checkpoint except the action encoder (shape differs)
  scratch         the same architecture freshly initialised (HF ViT init, torch-default Linear init, AdaLN zero,
                  predictor positions ~ N(0, 1)), as the official code builds it untrained
  random_encoder  scratch init, encoder + projector frozen (BN statistics from 200 warm-up batches), only the
                  predictor side trained on the prediction loss (R6's G3 control)
  The action encoder is the same fresh init (seeded separately) in every arm.
Loss (official lejepa_forward): MSE(pred_proj(predictor(emb[:, :3], act[:, :3])), emb[:, 1:]) + lam * SIGReg(emb),
SIGReg = official Epps-Pulley sketch (1024 directions shared over time, 17 knots on [0, 3], weights 2*dt (dt at
the ends) x exp(-t^2/2), times batch size), lam 0.09. AdamW lr 5e-5, wd 1e-3, warm-up then cosine to 0.1x,
grad clip 1.0, fp32.
Determinism / resume: batch ids from default_rng([seed, step]); MLX RNG reseeded from (seed, step) every step
(dropout, SIGReg directions); the episode list and action normalisation are frozen in <run>/data.json at the
first invocation. Checkpoint <run>/ckpt/{weights,opt}.safetensors + state.json; final in <run>/final/.
Logs: <run>/log.jsonl every 50 steps; <run>/val.jsonl every --val-every steps (held-out one-block error
ratio vs copy-last on 256 fixed held-out windows, eval mode).
"""
import argparse
import json
import math
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import rep_data as D  # noqa: E402
import rep_ports as P  # noqa: E402

ACT_DIM = 30


def sigreg_official(proj, key, num_proj=1024, knots=17):
    """proj (T, B, D) -> mean over time and directions of the Epps-Pulley statistic (le-wm module.SIGReg)."""
    _, B, Dm = proj.shape
    A = mx.random.normal((Dm, num_proj), key=key)
    A = A / mx.sqrt((A * A).sum(0, keepdims=True))
    t = mx.linspace(0, 3, knots)
    dt = 3.0 / (knots - 1)
    w = np.full(knots, 2 * dt, np.float32)
    w[[0, -1]] = dt
    window = mx.exp(-(t * t) / 2.0)
    weights = mx.array(w) * window
    x_t = (proj @ A)[..., None] * t
    err = (mx.cos(x_t).mean(-3) - window) ** 2 + mx.sin(x_t).mean(-3) ** 2
    return ((err @ weights) * B).mean()


def lewm_loss(model, F, A, key, lam, use_sig):
    emb = model.encode(F)                                   # (B, T, 192)
    if not use_sig:
        emb = mx.stop_gradient(emb)
    pred = model.predict(emb[:, :3], A[:, :3])
    lp = ((pred - emb[:, 1:]) ** 2).mean()
    if not use_sig:
        return lp, (lp, mx.array(0.0))
    ls = sigreg_official(emb.transpose(1, 0, 2), key)
    return lp + lam * ls, (lp, ls)


def lr_at(step, cfg):
    w = cfg["warmup"]
    if step < w:
        return cfg["lr"] * (step + 1) / w
    p = min(1.0, (step - w) / max(1, cfg["steps"] - w))
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def build_model(cfg):
    mx.random.seed(cfg["seed"])
    model = P.LeWMOfficial(ACT_DIM, dropout=0.1)
    init = cfg["init"]
    if init.startswith("ckpt:"):
        w = P.lewm_weights_from_torch(P.lewm_state_cached(init[5:]), with_action_encoder=False)
        model.load_weights([(k, mx.array(v)) for k, v in w.items()], strict=False)
    elif init in ("scratch", "random_encoder"):
        P.init_scratch_lewm(model, cfg["seed"])
    else:
        raise ValueError(init)
    mx.random.seed(cfg["seed"] + 1000)                       # the shared fresh action encoder
    model.action_encoder = P.Embedder(ACT_DIM)
    mx.eval(model.parameters())
    return model


def set_modes(model, cfg, training):
    model.train(training)
    if cfg["init"] == "random_encoder":
        model.encoder.eval()
        model.projector.eval()


def save_ckpt(run, model, opt, cfg, step, name="ckpt"):
    tmp = run / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_weights(str(tmp / "weights.safetensors"))
    mx.save_safetensors(str(tmp / "opt.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "state.json").write_text(json.dumps({"step": step, "config": cfg}))
    dst = run / name
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def load_ckpt(run, model, opt, name="ckpt"):
    d = run / name
    model.load_weights(str(d / "weights.safetensors"))
    opt.state = tree_unflatten(list(mx.load(str(d / "opt.safetensors")).items()))
    return json.loads((d / "state.json").read_text())


def load_for_eval(run, name="final"):
    st = json.loads((run / name / "state.json").read_text())
    model = P.LeWMOfficial(ACT_DIM, dropout=0.1)
    model.load_weights(str(run / name / "weights.safetensors"))
    model.eval()
    return model, st["config"], st["step"]


def data_for(run, cfg):
    p = run / "data.json"
    if not p.exists():
        eps = D.episode_list("train")
        if cfg.get("max_episodes"):
            eps = eps[:cfg["max_episodes"]]
        mu, sd = D.act_norm(eps)
        held = D.episode_list("heldout")
        p.write_text(json.dumps({"train": eps, "heldout": held, "act_mu": mu.tolist(), "act_sd": sd.tolist()}))
    d = json.loads(p.read_text())
    norm = (np.array(d["act_mu"], np.float32), np.array(d["act_sd"], np.float32))
    return D.Episodes(d["train"], norm), D.Episodes(d["heldout"], norm) if d["heldout"] else None


def val_ratio(model, cfg, vwin, ids, bs=32):
    """One-block error ratio vs copy-last on fixed held-out windows (eval mode)."""
    se = cl = 0.0
    for i in range(0, len(ids), bs):
        F, A, _ = vwin.get(ids[i:i + bs])
        z = model.encode(mx.array(F))
        zh = model.predict(z[:, :3], mx.array(A[:, :3]))[:, 2]
        mx.eval(zh, z)
        zn, zhn = np.array(z), np.array(zh)
        se += float(((zhn - zn[:, 3]) ** 2).sum())
        cl += float(((zn[:, 2] - zn[:, 3]) ** 2).sum())
    return se / cl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", default="scratch")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lam", type=float, default=0.09)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0, help="testing only: cap the training episode list")
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--guard-min", type=float, default=44.0)
    ap.add_argument("--stop-at", type=int, default=None, help="testing: stop (with a save) at this step")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    run = ROOT / a.run
    t_start = time.time()
    if a.resume:
        cfg = json.loads((run / "ckpt" / "state.json").read_text())["config"]
    else:
        cfg = {k: getattr(a, k) for k in ("init", "steps", "batch", "lr", "warmup", "lam", "seed", "max_episodes")}
        cfg.update(fs=5, T=4, act_dim=ACT_DIM)
        if (run / "ckpt").exists():
            raise SystemExit(f"{run} already has a checkpoint; use --resume")
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.json").write_text(json.dumps(cfg, indent=1))
    E, H = data_for(run, cfg)
    win = D.Windows(E, T=cfg["T"], fs=cfg["fs"])
    model = build_model(cfg)
    opt = optim.AdamW(learning_rate=cfg["lr"], weight_decay=1e-3)
    step = 0
    rand_enc = cfg["init"] == "random_encoder"
    if a.resume:
        step = load_ckpt(run, model, opt)["step"]
        print(f"resumed at step {step}", flush=True)
    elif rand_enc:      # BN running statistics of the frozen random projector: 200 forward batches, no grads
        model.train()
        for i in range(200):
            F, _, _ = win.get(np.random.default_rng([cfg["seed"], 10 ** 7 + i]).integers(0, len(win), cfg["batch"]))
            mx.eval(model.encode(mx.array(F)))
    if rand_enc:
        model.encoder.freeze()
        model.projector.freeze()
    set_modes(model, cfg, True)
    vwin = D.Windows(H, T=cfg["T"], fs=cfg["fs"]) if H is not None else None
    vids = np.sort(np.random.default_rng(123).choice(len(vwin), min(256, len(vwin)), replace=False)) if vwin else None
    print(f"windows {len(win)} from {len(win.ep_ids)} episodes; params {P.param_count(model) / 1e6:.2f}M "
          f"trainable {sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6:.2f}M", flush=True)

    q = queue.Queue(maxsize=4)
    stop_flag = threading.Event()

    def producer(s0):
        s = s0
        while not stop_flag.is_set() and s < cfg["steps"]:
            F, A, _ = win.get(np.random.default_rng([cfg["seed"], s]).integers(0, len(win), cfg["batch"]))
            while not stop_flag.is_set():
                try:
                    q.put((s, F, A), timeout=1.0)
                    break
                except queue.Full:
                    continue
            s += 1
    th = threading.Thread(target=producer, args=(step,), daemon=True)
    th.start()

    lossgrad = nn.value_and_grad(model, lambda m, F, A, k: lewm_loss(m, F, A, k, cfg["lam"], not rand_enc))
    acc = {"loss": 0.0, "pred": 0.0, "sig": 0.0, "n": 0}
    t_log = time.time()
    logf = open(run / "log.jsonl", "a")
    stop_reason = "done"
    while step < cfg["steps"]:
        s, F, A = q.get()
        assert s == step
        mx.random.seed(cfg["seed"] * 1_000_003 + step)
        key = mx.random.key(cfg["seed"] * 1_000_003 + step)
        opt.learning_rate = lr_at(step, cfg)
        (loss, (lp, ls)), g = lossgrad(model, mx.array(F), mx.array(A), key)
        g, gn = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        mx.eval(loss, lp, ls, model.parameters(), opt.state)
        if not math.isfinite(float(loss)):
            stop_reason = "nan"
            break
        acc["loss"] += float(loss); acc["pred"] += float(lp); acc["sig"] += float(ls); acc["n"] += 1
        step += 1
        if step % 50 == 0:
            dt = time.time() - t_log
            rec = {"step": step, "loss": acc["loss"] / acc["n"], "pred": acc["pred"] / acc["n"],
                   "sigreg": acc["sig"] / acc["n"], "lr": lr_at(step - 1, cfg), "gnorm": float(gn),
                   "steps_s": round(acc["n"] / dt, 3), "elapsed_min": round((time.time() - t_start) / 60, 2),
                   "load": round(os.getloadavg()[0], 1), "mem_gb": round(mx.get_peak_memory() / 1e9, 2)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(json.dumps(rec), flush=True)
            acc = {"loss": 0.0, "pred": 0.0, "sig": 0.0, "n": 0}
            t_log = time.time()
        if vwin is not None and a.val_every and step % a.val_every == 0:
            set_modes(model, cfg, False)
            r = val_ratio(model, cfg, vwin, vids)
            set_modes(model, cfg, True)
            with open(run / "val.jsonl", "a") as vf:
                vf.write(json.dumps({"step": step, "heldout_R_256": r}) + "\n")
            t_log = time.time()
        if step % a.ckpt_every == 0 or step == cfg["steps"]:
            save_ckpt(run, model, opt, cfg, step)
        if a.stop_at is not None and step >= a.stop_at:
            stop_reason = "stop_at"
            break
        if (time.time() - t_start) / 60 > a.guard_min:
            stop_reason = "guard"
            break
    stop_flag.set()
    if stop_reason != "nan":
        save_ckpt(run, model, opt, cfg, step)
    if step >= cfg["steps"]:
        save_ckpt(run, model, opt, cfg, step, name="final")
    print(json.dumps({"stopped": stop_reason, "step": step, "minutes": round((time.time() - t_start) / 60, 1)}),
          flush=True)
    return 0 if stop_reason in ("done", "stop_at") else (3 if stop_reason == "guard" else 1)


if __name__ == "__main__":
    sys.exit(main())
