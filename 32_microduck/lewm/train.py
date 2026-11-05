"""R6 LeWM S1 trainer (MLX). Run through the GPU lock; each invocation stops itself before 45 min.

    ../bin/gpu_run ../.venv/bin/python lewm/train.py --run runs/lewm_s1 --steps 30000
    ../bin/gpu_run ../.venv/bin/python lewm/train.py --run runs/lewm_s1 --resume
    ../bin/gpu_run ../.venv/bin/python lewm/train.py --run runs/lewm_rand --steps 30000 --random-encoder

Determinism: batch indices come from default_rng([seed, step]); the MLX RNG is reseeded from (seed, step)
every step (dropout, SIGReg directions). So a resumed run draws exactly the batches and noise it would
have drawn, and the checkpoint needs weights, optimiser state, step and config only.
Checkpoint (rolling, atomic): <run>/ckpt/{weights.safetensors, opt.safetensors, state.json}; the final
step also goes to <run>/final/. Log: <run>/log.jsonl every 100 steps.
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
sys.path.insert(0, str(HERE))
from data import Part, Windows, train_select  # noqa: E402
from model import LeWM, lewm_loss  # noqa: E402

ROOT = HERE.parent


def lr_at(step, cfg):
    w = cfg["warmup"]
    if step < w:
        return cfg["lr"] * (step + 1) / w
    p = min(1.0, (step - w) / max(1, cfg["steps"] - w))
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


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
    st = mx.load(str(d / "opt.safetensors"))
    opt.state = tree_unflatten(list(st.items()))
    return json.loads((d / "state.json").read_text())


def build_model(cfg):
    mx.random.seed(cfg["seed"])
    return LeWM(act_dim=cfg["fs"] * 6, img=64, patch=8)


def set_modes(model, cfg, training):
    model.train(training)
    if cfg["random_encoder"]:
        model.encoder.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fs", type=int, default=5)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=0.09)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="base,noisy")
    ap.add_argument("--random-encoder", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--guard-min", type=float, default=44.0)
    ap.add_argument("--stop-at", type=int, default=None, help="testing: stop (with a save) at this step")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    run = ROOT / a.run
    t_start = time.time()
    if a.resume:
        cfg = json.loads((run / "ckpt" / "state.json").read_text())["config"]
    else:
        cfg = {k: getattr(a, k) for k in ("steps", "batch", "fs", "T", "lr", "warmup", "lam", "seed", "parts",
                                           "random_encoder")}
        if run.exists() and (run / "ckpt").exists():
            raise SystemExit(f"{run} already has a checkpoint; use --resume")
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.json").write_text(json.dumps(cfg, indent=1))
    model = build_model(cfg)
    opt = optim.AdamW(learning_rate=cfg["lr"], weight_decay=1e-3)
    parts = [Part(p) for p in cfg["parts"].split(",")]
    win = Windows(parts, T=cfg["T"], fs=cfg["fs"], select=train_select)
    step = 0
    if a.resume:
        step = load_ckpt(run, model, opt)["step"]
        print(f"resumed at step {step}", flush=True)
    elif cfg["random_encoder"]:
        # BN running statistics of the frozen random projector from 200 forward batches, no gradients
        model.train()
        for i in range(200):
            F, _, _ = win.get(np.random.default_rng([cfg["seed"], 10 ** 7 + i]).integers(0, len(win), cfg["batch"]))
            mx.eval(model.encode_frames(mx.array(F)))
        model.encoder.freeze()
    if cfg["random_encoder"]:
        model.encoder.freeze()
    set_modes(model, cfg, True)
    print(f"windows {len(win)} from {len(win.eps)} episodes; params "
          f"{sum(v.size for _, v in tree_flatten(model.parameters())) / 1e6:.2f}M "
          f"trainable {sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6:.2f}M", flush=True)

    # background prefetch of step-indexed batches
    q = queue.Queue(maxsize=6)
    stop_flag = threading.Event()

    def producer(s0):
        s = s0
        while not stop_flag.is_set() and s < cfg["steps"]:
            F, A, _ = win.get(np.random.default_rng([cfg["seed"], s]).integers(0, len(win), cfg["batch"]))
            q.put((s, F, A))
            s += 1
    th = threading.Thread(target=producer, args=(step,), daemon=True)
    th.start()

    use_sig = not cfg["random_encoder"]
    lossgrad = nn.value_and_grad(model, lambda m, F, A, k: lewm_loss(m, F, A, k, lam=cfg["lam"], use_sigreg=use_sig))
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
        if step % 100 == 0:
            dt = time.time() - t_log
            rec = {"step": step, "loss": acc["loss"] / acc["n"], "pred": acc["pred"] / acc["n"],
                   "sigreg": acc["sig"] / acc["n"], "lr": lr_at(step - 1, cfg), "gnorm": float(gn),
                   "steps_s": round(acc["n"] / dt, 3), "elapsed_min": round((time.time() - t_start) / 60, 2),
                   "load": round(os.getloadavg()[0], 1)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(json.dumps(rec), flush=True)
            acc = {"loss": 0.0, "pred": 0.0, "sig": 0.0, "n": 0}
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
