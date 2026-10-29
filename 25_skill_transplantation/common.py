"""Shared pieces for 25 Neural Skill Transplantation (kept in one small file on purpose).

Body: gymnasium HalfCheetah-v5 (17-d obs, 6-d action, 1000-step episodes, no termination).
Two skills on the same body, same obs/action conventions, different rewards:
  fwd: r = +x_velocity - 0.1 |a|^2   (the standard HalfCheetah reward)
  bwd: r = -x_velocity - 0.1 |a|^2   (run backwards)
The policy input is [normalized obs (17), task one-hot (2)] so one network could do both.

Parameters travel as a flat dict of numpy arrays (name -> array). Rollouts and evaluation
use a numpy forward pass on CPU; PPO updates use MLX (see ppo.py).
"""
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import gymnasium as gym  # noqa: E402

# The skill pair is chosen by env SKILL_PAIR (default "fwd_bwd"; the follow-up pair is "fwd_cruise").
TASKS = os.environ.get("SKILL_PAIR", "fwd_bwd").split("_")
CRUISE_CTRL_COST = 1.0
OBS_DIM, ACT_DIM, N_TASK = 17, 6, 2
IN_DIM = OBS_DIM + N_TASK
HID = 128
CTRL_COST = 0.1
EP_LEN = 1000
LOGSTD_MIN, LOGSTD_MAX = -3.0, 0.0

# fixed seed banks: training envs use seed*1000+i; validation and held-out eval never overlap
VAL_SEEDS = list(range(20000, 20005))
EVAL_SEEDS = list(range(10000, 10010))

PARAM_KEYS = ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "a3.weight", "a3.bias",
              "c1.weight", "c1.bias", "c2.weight", "c2.bias", "c3.weight", "c3.bias", "log_std"]
ACTOR_KEYS = [k for k in PARAM_KEYS if k.startswith("a") or k == "log_std"]


def task_reward(task, x_vel, act):
    """fwd: +v - 0.1|a|^2, bwd: -v - 0.1|a|^2, cruise (low-energy forward): +v - 1.0|a|^2."""
    if task == "cruise":
        return x_vel - CRUISE_CTRL_COST * np.sum(np.square(act), axis=-1)
    sign = 1.0 if task == "fwd" else -1.0
    return sign * x_vel - CTRL_COST * np.sum(np.square(act), axis=-1)


def make_env(render=False):
    if render:   # camera tracks the torso so the runner stays in frame
        return gym.make("HalfCheetah-v5", render_mode="rgb_array", width=320, height=240,
                        camera_name="track")
    return gym.make("HalfCheetah-v5")


def init_params(seed):
    """Orthogonal init like standard PPO (gain sqrt2 hidden, 0.01 policy head, 1 value head)."""
    rng = np.random.default_rng(seed)

    def ortho(n_out, n_in, gain):
        a = rng.standard_normal((max(n_out, n_in), min(n_out, n_in)))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))
        if n_out < n_in:
            q = q.T
        return (gain * q[:n_out, :n_in]).astype(np.float32)

    p = {}
    for pre, out in (("a", ACT_DIM), ("c", 1)):
        p[f"{pre}1.weight"] = ortho(HID, IN_DIM, np.sqrt(2))
        p[f"{pre}2.weight"] = ortho(HID, HID, np.sqrt(2))
        p[f"{pre}3.weight"] = ortho(out, HID, 0.01 if pre == "a" else 1.0)
        for i in (1, 2):
            p[f"{pre}{i}.bias"] = np.zeros(HID, np.float32)
        p[f"{pre}3.bias"] = np.zeros(out, np.float32)
    p["log_std"] = np.full(ACT_DIM, -0.5, np.float32)
    return p


def np_mean(p, x):
    h = np.tanh(x @ p["a1.weight"].T + p["a1.bias"])
    h = np.tanh(h @ p["a2.weight"].T + p["a2.bias"])
    return np.tanh(h @ p["a3.weight"].T + p["a3.bias"])


def np_value(p, x):
    h = np.tanh(x @ p["c1.weight"].T + p["c1.bias"])
    h = np.tanh(h @ p["c2.weight"].T + p["c2.bias"])
    return (h @ p["c3.weight"].T + p["c3.bias"])[..., 0]


class Normalizer:
    """Running mean/var of raw observations. Frozen after the shared base phase so that
    every branch sees identical observation conventions."""

    def __init__(self, mean=None, var=None, count=1e-4):
        self.mean = np.zeros(OBS_DIM) if mean is None else np.asarray(mean, np.float64)
        self.var = np.ones(OBS_DIM) if var is None else np.asarray(var, np.float64)
        self.count = count

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @staticmethod
    def from_state(s):
        return Normalizer(s["mean"], s["var"], float(s["count"]))


def policy_input(norm, obs, task_ids):
    oh = np.zeros((obs.shape[0], N_TASK), np.float32)
    oh[np.arange(obs.shape[0]), task_ids] = 1.0
    return np.concatenate([norm(obs), oh], 1)


def evaluate(params, norm, task, seeds=EVAL_SEEDS, policy_for_task=None, return_xvel=False):
    """Deterministic (mean-action) return of `params` on `task` over 1000-step episodes,
    one episode per seed. `policy_for_task` (task -> params) overrides params, used by the
    oracle switching baseline. Returns np.array of returns (and mean x-velocity)."""
    p = policy_for_task(task) if policy_for_task else params
    tid = TASKS.index(task)
    envs = [make_env() for _ in seeds]
    obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(envs, seeds)])
    ret = np.zeros(len(seeds))
    xv = np.zeros(len(seeds))
    ids = np.full(len(seeds), tid)
    for _ in range(EP_LEN):
        a = np_mean(p, policy_input(norm, obs, ids))
        for i, e in enumerate(envs):
            o, _, _, _, info = e.step(a[i])
            obs[i] = o
            ret[i] += task_reward(task, info["x_velocity"], a[i])
            xv[i] += info["x_velocity"] / EP_LEN
    for e in envs:
        e.close()
    return (ret, xv) if return_xvel else ret


def save_params(path, params, norm=None, **meta):
    d = {k: v for k, v in params.items()}
    if norm is not None:
        d.update({"norm_mean": norm.mean, "norm_var": norm.var, "norm_count": np.array(norm.count)})
    for k, v in meta.items():
        d["meta_" + k] = np.array(v)
    np.savez(path, **d)


def load_params(path):
    z = np.load(path)
    p = {k: z[k].astype(np.float32) for k in PARAM_KEYS}
    norm = None
    if "norm_mean" in z:
        norm = Normalizer(z["norm_mean"], z["norm_var"], float(z["norm_count"]))
    return p, norm
