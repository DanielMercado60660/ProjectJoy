"""Control: gradient-descent distillation on SYNTHETIC inputs only (zero env steps) — the gradient
analogue of the WSP one-shot (runner: Claude).

Student = same 19-128-128-6 actor, initialised at theta0 (as DAgger). Each Adam step draws 1024 fresh
inputs: 17 standard-normal normalized-obs coordinates (clipped to +-10, no environment knowledge) and a
random task one-hot; target = that task's specialist action mean; loss = MSE on the action mean (as
DAgger). 4000 steps, lr 1e-3 (DAgger 200k used ~2.4k steps at lr 1e-3).

  ../bin/gpu_run ../.venv/bin/python wsp/synth_distill.py     -> wsp/baselines/synth_distill_s<seed>.npz
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from common import load_params, np_mean, save_params  # noqa: E402
from ppo import model_from_np, np_from_model  # noqa: E402

STEPS, BATCH, LR = 4000, 1024, 1e-3


def run(seed):
    d = os.path.join(PROJ, "checkpoints", f"s{seed}")
    t0, norm = load_params(os.path.join(d, "base.npz"))
    A, _ = load_params(os.path.join(d, "pre_fwd.npz"))
    B, _ = load_params(os.path.join(d, "pre_bwd.npz"))
    rng = np.random.default_rng(777 + seed)
    mx.random.seed(777 + seed)
    model = model_from_np(t0)
    opt = optim.Adam(learning_rate=LR)

    def loss_fn(m, x, y):
        return ((m.mean(x) - y) ** 2).mean()

    gfn = nn.value_and_grad(model, loss_fn)
    for it in range(STEPS):
        obs = np.clip(rng.standard_normal((BATCH, 17)), -10, 10).astype(np.float32)
        tid = rng.integers(0, 2, BATCH)
        x = np.concatenate([obs, np.eye(2, dtype=np.float32)[tid]], 1)
        y = np.where((tid == 0)[:, None], np_mean(A, x), np_mean(B, x)).astype(np.float32)
        l, g = gfn(model, mx.array(x), mx.array(y))
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
        if it % 1000 == 0 or it == STEPS - 1:
            print(f"s{seed} step {it} loss {float(l):.5f}", flush=True)
    p = np_from_model(model)
    os.makedirs(os.path.join(HERE, "baselines"), exist_ok=True)
    save_params(os.path.join(HERE, "baselines", f"synth_distill_s{seed}.npz"), p, norm, grad_steps=STEPS, env_steps=0)


if __name__ == "__main__":
    for s in range(4):
        run(s)
