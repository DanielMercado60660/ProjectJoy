"""
MuonAdamW for MLX, ported from upstream nanochat/optim.py.

Matrix params (transformer blocks) -> Muon with the upstream extras:
  Nesterov momentum, MuonEq row equilibration, Polar Express orthogonalization (5 steps, bf16),
  Muon+ Frobenius renormalization, NorMuon factored variance reduction, cautious weight decay,
  lr * sqrt(max(1, rows/cols)).
Embedding / lm_head / scalars -> AdamW groups with upstream's betas, eps and weight decays.

Differences from upstream: no parameter stacking / sharding (single device), each matrix is
updated on its own (mx.compile of the whole train step fuses it anyway).
"""
import mlx.core as mx
import mlx.optimizers as optim

POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class Muon(optim.Optimizer):
    def __init__(self, learning_rate, momentum=0.95, beta2=0.9, weight_decay=0.0, ns_steps=5):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        # scheduled scalars live in the optimizer state so a compiled step sees new values
        self._state["momentum"] = mx.array(momentum, dtype=mx.float32)
        self._state["weight_decay"] = mx.array(weight_decay, dtype=mx.float32)
        self.beta2 = beta2
        self.ns_steps = ns_steps

    def init_single(self, parameter, state):
        m, n = parameter.shape
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros((m, 1) if m >= n else (1, n), dtype=parameter.dtype)

    def apply_single(self, gradient, parameter, state):
        m, n = parameter.shape
        mom = self.state["momentum"]
        lr = self.learning_rate.astype(mx.float32) * max(1.0, m / n) ** 0.5
        wd = self.state["weight_decay"]
        # Nesterov momentum
        buf = state["m"] + (1 - mom) * (gradient - state["m"])
        state["m"] = buf
        g = gradient + mom * (buf - gradient)
        X = g.astype(mx.bfloat16)
        # MuonEq row equilibration
        Xf = X.astype(mx.float32)
        target = mx.sqrt(mx.sum(Xf * Xf)) / (m ** 0.5)
        row_norm = mx.maximum(mx.sqrt(mx.sum(Xf * Xf, axis=-1, keepdims=True)), 1e-6)
        X = X * (target / row_norm).astype(X.dtype)
        # Polar Express orthogonalization
        X = X / (mx.sqrt(mx.sum(X.astype(mx.float32) ** 2)) * 1.01 + 1e-6).astype(X.dtype)
        if m > n:
            for a, b, c in POLAR_EXPRESS[: self.ns_steps]:
                A = X.T @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in POLAR_EXPRESS[: self.ns_steps]:
                A = X @ X.T
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X.astype(mx.float32)
        # Muon+ renormalization to sqrt(min(m, n))
        g = g * (min(m, n) ** 0.5 / mx.maximum(mx.sqrt(mx.sum(g * g)), 1e-6))
        # NorMuon variance reduction
        red = -1 if m >= n else -2
        red_size = g.shape[red]
        v_mean = mx.mean(g * g, axis=red, keepdims=True)
        v_norm = mx.sqrt(mx.sum(v_mean) * red_size)
        v = state["v"] + (1 - self.beta2) * (v_mean - state["v"])
        state["v"] = v
        step_size = mx.rsqrt(mx.maximum(v, 1e-10))
        v_norm_new = mx.sqrt(mx.sum(v_mean * red_size * step_size * step_size))
        g = g * step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
        # cautious weight decay + update
        mask = ((g * parameter) >= 0).astype(parameter.dtype)
        return parameter - lr * g - lr * wd * parameter * mask


def build_optimizer(model, matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008, scalar_lr=0.5,
                    weight_decay=0.0, lr_scale=1.0):
    """Mirror of upstream GPT.setup_optimizer (minus value-embed / smear groups).
    Returns (optimizer, base_lrs) where base_lrs lists each sub-optimizer's peak lr."""
    d = model.config.n_embd
    dscale = (d / 768) ** -0.5  # AdamW lrs scale ~ 1/sqrt(d_model), tuned at 768
    groups = [
        ("muon", lambda k, v: k.startswith("h."),
         Muon(matrix_lr * lr_scale, momentum=0.95, beta2=0.9, weight_decay=weight_decay)),
        ("lm_head", lambda k, v: k.startswith("lm_head"),
         optim.AdamW(unembedding_lr * dscale * lr_scale, betas=[0.8, 0.96], eps=1e-10, weight_decay=0.01, bias_correction=True)),
        ("wte", lambda k, v: k.startswith("wte"),
         optim.AdamW(embedding_lr * dscale * lr_scale, betas=[0.8, 0.995], eps=1e-10, weight_decay=0.001, bias_correction=True)),
        ("resid", lambda k, v: k.startswith("resid_lambdas"),
         optim.AdamW(scalar_lr * 0.01 * lr_scale, betas=[0.8, 0.95], eps=1e-10, weight_decay=0.05, bias_correction=True)),
        ("x0", None,  # fallback: x0_lambdas
         optim.AdamW(scalar_lr * lr_scale, betas=[0.96, 0.95], eps=1e-10, weight_decay=0.0, bias_correction=True)),
    ]
    opts = [g[2] for g in groups]
    base_lrs = [float(o.learning_rate) for o in opts]
    opt = optim.MultiOptimizer(opts, filters=[g[1] for g in groups[:-1]])
    return opt, base_lrs


def set_schedule(opt, base_lrs, lr_mult, muon_momentum=None, muon_wd=None):
    for o, lr in zip(opt.optimizers, base_lrs):
        o.learning_rate = lr * lr_mult
    muon = opt.optimizers[0]
    if muon_momentum is not None:
        muon.state["momentum"] = mx.array(muon_momentum, dtype=mx.float32)
    if muon_wd is not None:
        muon.state["weight_decay"] = mx.array(muon_wd, dtype=mx.float32)
