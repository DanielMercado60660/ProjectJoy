"""M5: behaviour cloning (and the DAgger retrain) for the recurrent policy (learner.py).

    ../bin/gpu_run ../.venv/bin/python m5_train.py <run_dir> --index episodes/index.jsonl [--index ...]
        [--steps N] [--init weights.safetensors] [--max-minutes 40] [--resume]
Data: the episodes listed in the index files (train split only); labels are the bot's own commands.
Frames are cached once at 64x64 in episodes/cache/ (gitignored). Windows of --window ticks (default 128)
with a --burn tick burn-in (default 32, no loss) unless the window starts at the episode's first tick;
a fraction --p-start of windows (default 0.2) start at tick 0, where the deployed GRU also starts from
zero. (BC v1 used 64/16/uniform starts: no window ever spanned the bot's ~94-tick opening scan.) Loss: MSE on vx, wz (weight 1)
and the 4 head values (0.5) plus BCE on `done` (positive weight 50: one positive per episode).
Checkpoints every --ckpt-every steps hold weights, optimizer state, step, RNG and config (resumable);
the step-indexed schedule never depends on wall clock (lessons file, item 11). Logs steps/s and
utilisation of the wall budget (item 10).

Review notes (REVIEW_M5.md):
- Resume is bit-exact (tests/test_m5.py::test_resume_continues_bit_exact). After --resume the logged
  steps_per_s is 100 / total elapsed (not a rate), and summary.json's wall_s and frames_seen cover only
  the last segment. Keep --max-minutes at 45 or less (AGENTS.md cap; the M5 runs passed 55).
- The done positive weight 50 assumes one done tick per episode. DAgger round-2 files break that: 19,084
  done ticks in 62 episodes (record.py keeps the finished shadow bot's last command), ~57% of the
  done-loss weight in the round-2 retrain.
- With weight 50, `done_logit > 0` at run time (learner.decode) means p(done) > ~2%.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from learner import IMG, Policy, body_vec, cmd_targets, instr_vec, prep_frame  # noqa: E402

CACHE = HERE / "episodes" / "cache"
REG_W = np.array([1, 1, 0.5, 0.5, 0.5, 0.5], np.float32)


def load_episode(rel):
    p = HERE / rel
    c = CACHE / p.parent.name / (p.stem + f"_{IMG}.npz")     # per task: stems repeat across tasks
    if c.exists():
        z = np.load(c)
        ep = {k: z[k] for k in z.files}
        assert "src" in ep and str(ep["src"]) == rel, (c, rel)
        return ep
    z = np.load(p)
    H = json.loads(str(z["header"]))
    frames = np.stack([prep_frame(f) for f in z["frames"]])
    reg, done = cmd_targets(z["cmd_bot"])
    ep = {"frames": frames, "body": np.stack([body_vec(b) for b in z["body"]]),
          "instr": instr_vec(H["instruction"]), "reg": reg.astype(np.float32), "done": done.astype(np.float32),
          "src": np.array(rel)}
    c.parent.mkdir(parents=True, exist_ok=True)
    np.savez(c, **ep)
    return ep


def load_all(index_files):
    rows = []
    for f in index_files:
        for line in open(HERE / f):
            r = json.loads(line)
            if "error" not in r and r.get("split", "train") == "train":
                rows.append(r)
    eps, kept = [], []
    for r in rows:
        try:
            eps.append(load_episode(r["path"]))
            kept.append(r)
        except (KeyError, OSError, ValueError) as e:      # e.g. a file still being written
            print("skipping", r["path"], repr(e)[:80], flush=True)
    return kept, eps


def batch(eps, lens, rng, B, W=128, BURN=32, p_start=0.2):
    p = lens / lens.sum()
    F = np.zeros((B, W, IMG, IMG, 3), np.uint8)
    Bo = np.zeros((B, W, 34), np.float32)
    In = np.zeros((B, W, 9), np.float32)
    R = np.zeros((B, W, 6), np.float32)
    D = np.zeros((B, W), np.float32)
    M = np.zeros((B, W), np.float32)
    for b in range(B):
        e = eps[rng.choice(len(eps), p=p)]
        T = len(e["frames"])
        s = 0 if rng.random() < p_start else int(rng.integers(0, max(1, T - W + 1)))
        n = min(W, T - s)
        F[b, :n] = e["frames"][s:s + n]
        Bo[b, :n] = e["body"][s:s + n]
        In[b, :] = e["instr"]
        R[b, :n] = e["reg"][s:s + n]
        D[b, :n] = e["done"][s:s + n]
        M[b, (0 if s == 0 else BURN):n] = 1.0
    return F, Bo, In, R, D, M


def loss_fn(model, F, Bo, In, R, D, M):
    reg, dl, _ = model(F.astype(mx.float32) / 255.0, Bo, In)
    mse = (((reg - R) ** 2) * mx.array(REG_W)).sum(-1)
    bce = 50.0 * D * nn.softplus(-dl) + (1 - D) * nn.softplus(dl)
    m = M.sum() + 1e-6
    return ((mse + bce) * M).sum() / m, ((mse * M).sum() / m, (bce * M).sum() / m)


def save(run, step, model, opt, rng, cfg):
    d = run / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(d / "weights.safetensors"))
    mx.save_safetensors(str(d / "opt.safetensors"), dict(tree_flatten(opt.state)))
    (d / "meta.json").write_text(json.dumps({"step": step, "rng": rng.bit_generator.state, "cfg": cfg}))
    (run / "latest.json").write_text(json.dumps({"step": step, "dir": d.name}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--init", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--burn", type=int, default=32)
    ap.add_argument("--p-start", type=float, default=0.2)
    a = ap.parse_args()
    run = HERE / a.run
    run.mkdir(parents=True, exist_ok=True)
    cfg = vars(a)
    t_load = time.time()
    rows, eps = load_all(a.index)
    lens = np.array([len(e["frames"]) for e in eps], np.float64)
    print(f"loaded {len(eps)} episodes, {int(lens.sum())} ticks in {time.time() - t_load:.0f} s", flush=True)
    mx.random.seed(a.seed)
    model = Policy()
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=a.lr)
    rng = np.random.default_rng(a.seed)
    step = 0
    if a.resume and (run / "latest.json").exists():
        lt = json.loads((run / "latest.json").read_text())
        d = run / lt["dir"]
        model.load_weights(str(d / "weights.safetensors"))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(d / "opt.safetensors")).items()))
        meta = json.loads((d / "meta.json").read_text())
        step = meta["step"]
        rng.bit_generator.state = meta["rng"]
        print(f"resumed at step {step}", flush=True)
    elif a.init:
        model.load_weights(a.init)
    vg = nn.value_and_grad(model, loss_fn)
    log = open(run / "train_log.jsonl", "a")
    t0 = time.time()
    seen = 0
    while step < a.steps:
        if (time.time() - t0) / 60 > a.max_minutes:
            print("wall guard reached; saving and stopping", flush=True)
            break
        F, Bo, In, R, D, M = batch(eps, lens, rng, a.batch, a.window, a.burn, a.p_start)
        (loss, (mse, bce)), g = vg(model, mx.array(F), mx.array(Bo), mx.array(In), mx.array(R), mx.array(D), mx.array(M))
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        step += 1
        seen += a.batch * a.window
        if step % 100 == 0:
            el = time.time() - t0
            rec = {"step": step, "loss": float(loss), "mse": float(mse), "bce": float(bce),
                   "steps_per_s": round(step / el if not a.resume else 100 / max(el, 1e-6), 2), "elapsed_s": round(el, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 1000 == 0:
                print(rec, flush=True)
        if step % a.ckpt_every == 0:
            save(run, step, model, opt, rng, cfg)
    save(run, step, model, opt, rng, cfg)
    el = time.time() - t0
    (run / "summary.json").write_text(json.dumps({"final_step": step, "wall_s": round(el, 1), "episodes": len(eps),
                                                  "ticks": int(lens.sum()), "frames_seen": seen,
                                                  "budget_used": round(el / 60 / a.max_minutes, 3)}))
    print("done", step, round(el, 1), "s", flush=True)


if __name__ == "__main__":
    main()
