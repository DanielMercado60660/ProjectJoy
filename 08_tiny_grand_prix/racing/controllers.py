"""Load any controller by name: es, ppo (learned MLPs), ray (tuned RayFollower),
ray_default (untuned RayFollower), pursuit (privileged pure pursuit)."""
import json
import os

from .baselines import PurePursuit, RayFollower
from .policy import MLPController, load_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")

LABELS = {
    "es": "ES (neuroevolution)",
    "ppo": "PPO (MLX)",
    "es_dr": "ES + grip randomization",
    "ppo_dr": "PPO + grip randomization",
    "es_soft": "ES trained with soft walls (hack demo)",
    "ray": "Ray follower (tuned)",
    "ray_dr": "Ray follower (tuned w/ grip rand.)",
    "ray_default": "Ray follower (untuned)",
    "pursuit": "Pure pursuit (PRIVILEGED)",
}


def get_controller(name, bank=None, ckpt=None):
    if name == "ray":
        return RayFollower(**json.load(open(os.path.join(CK, "ray_follower.json")))["params"])
    if name == "ray_dr":
        return RayFollower(**json.load(open(os.path.join(CK, "ray_follower_dr.json")))["params"])
    if name == "ray_default":
        return RayFollower()
    if name == "pursuit":
        return PurePursuit(bank)
    path = ckpt or os.path.join(CK, f"{name}_best.npz")
    theta, meta = load_policy(path)
    c = MLPController(theta, name)
    c.meta = meta
    return c
