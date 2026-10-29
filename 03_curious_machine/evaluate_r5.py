"""Matched exploration -> surprise task. NumPy-only for frontier/random/replay.
MLX is imported lazily ONLY when a learned checkpoint is requested.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from env import VecRooms, LAMP_ON, EP_FIELDS
from r5_core import Frontier, ReplayGraph, TASKS, TEST_SEEDS, transfer_metrics


class RowRNG:
    """Per-episode noise streams, invariant to batching/order of other episodes."""
    def __init__(self, seeds):
        self.rows = [np.random.default_rng(int(s)) for s in seeds]

    def integers(self, low, high, size):
        assert size[0] == len(self.rows)
        return np.stack([r.integers(low, high, size=size[1:]) for r in self.rows])


def task_success(env, tasks):
    """Privileged scoring only. Never passed to the explorer or the replay planner."""
    out = []
    for i, (kind, room) in enumerate(tasks):
        current = int(env.room[tuple(env.pos[i])])
        if kind == "reach":
            ok = current == room
        elif kind == "key_room":
            ok = current == room and bool(env.has_key[i])
        elif kind == "lamp":
            ok = bool(np.any((env.room == room) & (env.grid[i] == LAMP_ON)))
        else:
            raise ValueError(kind)
        out.append(ok)
    return np.array(out)


def new_env(seeds, budget, tv):
    env = VecRooms(len(seeds), max_steps=budget + 1, noisy_tv=tv, seed=int(seeds[0]))
    env.rng = RowRNG(seeds)
    return env


def frame(env, index=0):
    return (env.grid[index].copy(), env.pos[index].copy(), env.tv_val[index].copy())


def evaluate(method, seeds, tv=False, checkpoint=None, frontier_cfg=None, explore_steps=512,
             task_steps=128, max_expansions=4096, ex=None, deadline=None, record=False,
             planner_type="replay", policy_seed=73019):
    if not seeds or explore_steps < 1 or task_steps < 1:
        raise ValueError("Positive budgets and at least one seed required")
    started = time.monotonic()
    def guard():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Evaluation wall guard; no partial result may enter the table")
    env = new_env(seeds, explore_steps, tv); n = len(seeds)
    obs = env.obs.copy(); observations = [obs.copy()]; actions = []
    rng = RowRNG([s + 200000 for s in seeds])
    if method not in ("frontier", "random"):
        if ex is None:
            from train_r5 import R5Explorer
            ex, _ = R5Explorer.load(checkpoint)
        if ex.cfg["agent"] != method or ex.cfg["noisy_tv"] != tv:
            raise ValueError("Checkpoint method/world mismatch")
        ex.rng = np.random.default_rng(policy_seed)  # common stream, independent of training
    frontiers = [Frontier(s, **(frontier_cfg or {})) for s in seeds] if method == "frontier" else []
    for i, f in enumerate(frontiers):
        f.observe(obs[i])
    explore_frames = [frame(env)] if record else []
    for _ in range(explore_steps):
        guard()
        if method == "random":
            a = rng.integers(0, 5, size=(n,))
        elif method == "frontier":
            a = np.array([f.act() for f in frontiers])
        else:
            a = ex.act(obs)[0]
        obs, done, _ = env.step(a)
        assert not done.any(), "Evaluation must retain the final pre-reset observation"
        actions.append(a.copy()); observations.append(obs.copy())
        for i, f in enumerate(frontiers):
            f.observe(obs[i], int(a[i]))
        if record:
            explore_frames.append(frame(env))
    episodes = []
    for i, seed in enumerate(seeds):
        episodes.append(dict(seed=int(seed), rooms=int(env.rooms[i].sum()), interactions=int(env.inter[i].sum()),
                             coverage=int(env.visited[i].sum()), tv_steps=int(env.tvsteps[i]),
                             unproductive=int(env.unprod[i]), repeat_interactions=int(env.repeat[i]),
                             inter_t=env.inter_t[i].tolist()))
    if planner_type == 'replay':
        graphs = [ReplayGraph([o[i] for o in observations], [a[i] for a in actions]) for i in range(n)]
        knowledge_stats = None
    elif planner_type == 'compositional':
        from planner_r5_t2 import Knowledge, CompositionalPlanner
        knowledge = [Knowledge.from_replay([o[i] for o in observations], [a[i] for a in actions]) for i in range(n)]
        graphs = [CompositionalPlanner(k) for k in knowledge]
        knowledge_stats = [k.stats() for k in knowledge]
    else:
        raise ValueError(planner_type)
    trials = [(i, task) for i in range(n) for task in TASKS]
    plans = []
    for i, task in trials:
        guard()
        plans.append(graphs[i].plan(task, task_steps, max_expansions))
    task_seeds = [int(seeds[i]) + 1000 * (TASKS.index(task) + 1) for i, task in trials]
    task_labels = [task for _, task in trials]
    all_rows = []; task_frames = []; record_index = TASKS.index(("reach", 6))
    # Empty replay uses exactly the same random fallback, task budget and seeds.
    for control in (False, True):
        target_env = new_env(task_seeds, task_steps, tv)
        fallback = RowRNG([s + 300000 for s in task_seeds])
        first = np.full(len(trials), -1, int)
        if record and not control:
            task_frames.append(frame(target_env, record_index))
        for t in range(task_steps):
            guard()
            a = fallback.integers(0, 5, size=(len(trials),))
            if not control:
                for j, (route, _) in enumerate(plans):
                    if route is not None and t < len(route):
                        a[j] = route[t]
            target_env.step(a)
            ok = task_success(target_env, task_labels)
            first[(first < 0) & ok] = t + 1
            if record and not control:
                task_frames.append(frame(target_env, record_index))
        rows = []
        for j, (i, task) in enumerate(trials):
            route, expansions = plans[j]
            rows.append(dict(seed=int(seeds[i]), task=list(task), success=bool(first[j] >= 0),
                             steps=int(first[j]) if first[j] >= 0 else task_steps,
                             planned=not control and route is not None,
                             plan_length=len(route) if route is not None and not control else None,
                             expansions=expansions if not control else 0))
        all_rows.append(rows)
    summary = {f: float(np.mean([e[f] for e in episodes])) for f in EP_FIELDS}
    output = dict(method=method, world="tv" if tv else "plain", checkpoint=str(checkpoint) if checkpoint else None,
                  seeds=list(seeds), policy_rng_seed=policy_seed, explore_steps=explore_steps, task_steps=task_steps,
                  planner_type=planner_type, knowledge_stats=knowledge_stats,
                  max_expansions=max_expansions, transferred="one fresh episode of action/observation replay",
                  pretrain_steps=ex.steps if ex is not None else 0,
                  frontier_cfg=frontier_cfg, exploration=summary, episodes=episodes,
                  transfer=transfer_metrics(all_rows[0], task_steps), tasks=all_rows[0],
                  empty_replay=transfer_metrics(all_rows[1], task_steps), empty_tasks=all_rows[1],
                  per_task={f"{k}_{r}": transfer_metrics([v for v in all_rows[0] if v["task"] == [k, r]], task_steps)
                            for k, r in TASKS}, elapsed_seconds=time.monotonic() - started)
    data = None
    if record:
        data = dict(observations=np.array(observations)[:, 0], actions=np.array(actions)[:, 0],
                    explore_grid=np.array([f[0] for f in explore_frames]),
                    explore_pos=np.array([f[1] for f in explore_frames]),
                    explore_tv=np.array([f[2] for f in explore_frames]),
                    task_grid=np.array([f[0] for f in task_frames]), task_pos=np.array([f[1] for f in task_frames]),
                    task_tv=np.array([f[2] for f in task_frames]), room=env.room,
                    method=np.array(method), world=np.array(output["world"]), seed=np.array(seeds[0]))
    return output, data


def save_eval(out, result, recording=None):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    if recording is not None:
        temp=out.with_suffix('.npz.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, **recording)
        os.replace(temp,out.with_suffix('.npz'))
    temp=out.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    os.replace(temp,out)  # JSON commits only after the complete recording exists.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("lp", "lpm", "rnd_ep", "disagree", "ngu", "frontier", "random"), required=True)
    p.add_argument("--world", choices=("plain", "tv"), default="tv"); p.add_argument("--checkpoint")
    p.add_argument("--out", required=True); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-minutes", type=float, default=5); p.add_argument("--selection")
    p.add_argument("--turn2", action="store_true")
    a = p.parse_args()
    if not 1 <= a.episodes <= len(TEST_SEEDS) or not 0 < a.max_minutes <= 45:
        p.error("episodes must be 1..32 and max-minutes in (0,45]")
    frontier_cfg = json.loads(Path(a.selection).read_text())["frontier"] if a.selection else None
    seeds = tuple(range(85400, 85432)) if a.turn2 else TEST_SEEDS
    result, recording = evaluate(a.method, seeds[:a.episodes], a.world == "tv", a.checkpoint,
                                 frontier_cfg, deadline=time.monotonic() + a.max_minutes * 60, record=True,
                                 planner_type='compositional' if a.turn2 else 'replay',
                                 max_expansions=100000 if a.turn2 else 4096,
                                 policy_seed=83519 if a.turn2 else 73019)
    save_eval(a.out, result, recording)
    print(json.dumps({k: result[k] for k in ("exploration", "transfer", "empty_replay")}))


if __name__ == "__main__":
    main()
