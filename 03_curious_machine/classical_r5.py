"""Charge classical methods the same pretraining env-step budget; no fitted weights.
Maps reset each episode, exactly as at deployment. Logs expose the otherwise unused
pretraining experience. Identical semantic observations permit one policy computation
broadcast over all vector lanes; every lane still executes real simulator steps.
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
from env import VecRooms
from r5_core import Frontier


def train_classical(method, seed, tv, steps, checkpoint, log_dir, frontier_cfg=None,
                    max_minutes=25, n_envs=32, episode_steps=512):
    if method not in ("frontier", "random") or steps % n_envs or not 0 < max_minutes <= 45:
        raise ValueError("Invalid method, budget or wall guard")
    checkpoint = Path(checkpoint); log_dir = Path(log_dir)
    checkpoint.parent.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(method=method, seed=seed, tv=tv, n_envs=n_envs, episode_steps=episode_steps,
               frontier_cfg=frontier_cfg or {})
    env = VecRooms(n_envs, episode_steps, tv, seed)
    rng = np.random.default_rng(seed); f = Frontier(seed, **(frontier_cfg or {})); f.observe(env.obs[0])
    step = 0; episode = 0; totals = dict(episodes=0, rooms=0, interactions=0, tv_steps=0)
    if checkpoint.exists():
        state = pickle.loads(checkpoint.read_bytes())
        if state["cfg"] != cfg:
            raise ValueError("Classical resume configuration mismatch")
        env.set_state(state["env"]); rng.bit_generator.state = state["rng"]
        f, step, episode, totals = (state[k] for k in ("frontier", "steps", "episode", "totals"))
    start = time.monotonic(); deadline = start + max_minutes * 60
    def save():
        state = dict(cfg=cfg, steps=step, episode=episode, frontier=f, env=env.get_state(),
                     rng=rng.bit_generator.state, totals=totals)
        tmp = checkpoint.with_suffix(".tmp"); tmp.write_bytes(pickle.dumps(state)); os.replace(tmp, checkpoint)
    while step < steps and time.monotonic() < deadline - 1:
        actions = np.full(n_envs, f.act()) if method == "frontier" else rng.integers(5, size=n_envs)
        obs, done, eps = env.step(actions); step += n_envs
        if method == "frontier":
            if done[0]:
                episode += 1; f = Frontier(seed + episode, **(frontier_cfg or {})); f.observe(obs[0])
            else:
                f.observe(obs[0], int(actions[0]))
        for e in eps:
            totals["episodes"] += 1
            for field in ("rooms", "interactions", "tv_steps"):
                totals[field] += e[field]
        if step % (n_envs * episode_steps * 4) == 0:
            save()
    save()
    outcome = dict(status="complete" if step == steps else "partial_wall_guard", steps=step,
                   target_steps=steps, invocation_seconds=time.monotonic() - start, totals=totals, cfg=cfg)
    (log_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    print(json.dumps(outcome), flush=True)
    return outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("frontier", "random"), required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--world", choices=("plain", "tv"), required=True)
    p.add_argument("--steps", type=int, required=True); p.add_argument("--max-minutes", type=float, required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--log-dir", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--n-envs", type=int, default=32)
    a = p.parse_args()
    cfg = json.loads(Path(a.selection).read_text())["frontier"]
    train_classical(a.method, a.seed, a.world == "tv", a.steps, a.checkpoint, a.log_dir, cfg, a.max_minutes, a.n_envs)


if __name__ == "__main__":
    main()
