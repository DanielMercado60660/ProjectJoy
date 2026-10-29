"""Follow-up (round 3): clips of the opponent-pool policy. Same unselected first-4 held-out starts as demo.py.

    ../.venv/bin/python demo_pool.py --ckpt checkpoints/pool/pool_mixed.npz --outdir media/pool
"""
import argparse, os
from demo import grid_clip
from eval import get_policies

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/pool/pool_mixed.npz")
ap.add_argument("--outdir", default="media/pool")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
ppo = get_policies(args.ckpt)["ppo"]
o = lambda name: os.path.join(args.outdir, name)
grid_clip(ppo, "goalie", "heldout", range(4), "pool PPO vs goalie (seen)", o("pool_vs_goalie"))
grid_clip(ppo, "keeper", "heldout", range(4), "pool PPO vs keeper (UNSEEN)", o("pool_vs_keeper"))
grid_clip(ppo, "chaser", "heldout", range(4), "pool PPO vs chaser (seen)", o("pool_vs_chaser"))
