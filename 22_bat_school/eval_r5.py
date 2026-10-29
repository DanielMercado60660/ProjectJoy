"""Full, balanced validation/test sweeps; lazy MLX import for learned agents."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from env_r5 import (make_pool, assert_disjoint, CaveEnv, GROUPS, BUDGETS, MAX_STEPS,
                    ECHO, WAIT, LEFT, RIGHT, DR, DC)
from baseline_r5 import AcousticPlanner
from io_r5 import Deadline, write_json, file_digest


class Behavior:
    """Behavior counts are observations, not evidence of intention.

    On stationary repeat/turn echoes, compare a fixed diagnostic decoder's
    Brier error before/after against CURRENT true traversability. Correct for
    head rotation. This removes gate-motion label drift, but remains a decoder
    proxy, not the neural policy's belief or a causal benefit estimate. For a
    translation, compare only the SAME shared edge, heard from opposite sides.
    """
    def __init__(self, n):
        self.last_position = np.full((n, 2), -100, int)
        self.last_heading = np.zeros(n, int)
        self.last_time = np.full(n, -100, int)
        self.last_estimate = np.full((n, 4), .5)
        self.counts = {k: np.zeros(n, int) for k in
                       ("wait_steps", "wait_listens", "double_listens", "turn_listens",
                        "translate_listens", "repeat_pairs", "turn_pairs", "translate_pairs",
                        "repeat_improved", "turn_improved", "translate_improved")}
        self.sums = {k: np.zeros(n) for k in ("repeat_brier_gain", "turn_brier_gain", "translate_brier_gain")}

    def update(self, env, move, obs, info, active):
        used = info["used"] & active
        pos = np.stack([env.r, env.c], 1)
        same = (self.last_position == pos).all(-1)
        recent = env.t-self.last_time <= 2
        same_heading = self.last_heading == env.hd
        repeat = used & same & recent & same_heading
        turn = used & same & recent & ~same_heading
        self.counts["wait_steps"] += active & (move == WAIT)
        self.counts["wait_listens"] += used & (move == WAIT)
        self.counts["double_listens"] += repeat & (env.t-self.last_time == 1)
        self.counts["turn_listens"] += used & np.isin(move, [LEFT, RIGHT])
        self.counts["translate_listens"] += used & info["moved"]
        near = obs[:, :ECHO].reshape(env.n, 2, 4, 12)[:, 0, :, :2].sum(-1)
        estimate_rel = 1/(1+np.exp(np.clip((near-.36)/.12, -30, 30)))
        absolute = (env.hd[:, None] + np.arange(4)) % 4
        estimate = np.zeros_like(estimate_rel)
        estimate[np.arange(env.n)[:, None], absolute] = estimate_rel
        truth = env.pool.open[env.m, env.r, env.c] & ~env.shut(
            env.m[:, None], env.r[:, None], env.c[:, None], np.arange(4), env.t[:, None])
        gain = ((self.last_estimate-truth)**2).mean(-1) - ((estimate-truth)**2).mean(-1)
        for name, mask in (("repeat", repeat), ("turn", turn)):
            self.counts[name+"_pairs"] += mask
            self.counts[name+"_improved"] += mask & (gain > 0)
            self.sums[name+"_brier_gain"] += gain * mask
        offset = pos-self.last_position
        adjacent = (np.abs(offset).sum(-1) == 1) & used & recent
        direction = (offset[:, 0, None]*DR+offset[:, 1, None]*DC).argmax(-1)
        reverse = (direction+2) % 4
        ids = np.arange(env.n)
        target = truth[ids, reverse]
        shared_gain = (self.last_estimate[ids, direction]-target)**2 - (estimate[ids, reverse]-target)**2
        self.counts["translate_pairs"] += adjacent
        self.counts["translate_improved"] += adjacent & (shared_gain > 0)
        self.sums["translate_brier_gain"] += adjacent*shared_gain
        self.last_position[used] = pos[used]
        self.last_heading[used] = env.hd[used]
        self.last_time[used] = env.t[used]
        self.last_estimate[used] = estimate[used]


def run_budget(pool, factory, budget, deadline=None, trace=False):
    env = CaveEnv(pool, pool.n, seed=77123, budget=budget, fixed=True, auto_reset=False)
    agent = factory(pool.n)
    obs = env.reset()
    behavior = Behavior(pool.n)
    frames = []
    for _ in range(MAX_STEPS):
        if deadline is not None:
            deadline.check()
        active = ~env.finished
        move, pulse = agent.act(obs)
        obs, _, _, info = env.step(move, pulse)
        behavior.update(env, move, obs, info, active)
        if trace:
            frames.append(np.stack([env.r, env.c, env.hd, env.t, env.pulses, info["used"],
                                    info["bump"], move, env.finished], -1))
        if env.finished.all():
            break
    episodes = []
    for i in range(pool.n):
        row = dict(cave=i, layout=pool.hashes[i], group=GROUPS[pool.group[i]], budget=budget,
                   success=bool(env.success[i]), timeout=not bool(env.success[i]),
                   steps=int(env.t[i]), pulses=int(env.pulses[i]), collisions=int(env.collisions[i]))
        row.update({k: int(v[i]) for k, v in behavior.counts.items()})
        row.update({k: float(v[i]) for k, v in behavior.sums.items()})
        episodes.append(row)
    return episodes, np.asarray(frames) if trace else None


def summarize(episodes):
    if not episodes:
        raise ValueError("empty evaluation")
    rows = []
    for budget in sorted({r["budget"] for r in episodes}):
        for group in GROUPS:
            e = [r for r in episodes if r["budget"] == budget and r["group"] == group]
            if not e:
                raise ValueError("incomplete group/budget suite")
            mean = lambda k: float(np.mean([r[k] for r in e]))
            count = lambda k: sum(r[k] for r in e)
            succ = mean("success")
            row = dict(budget=budget, group=group, n=len(e), completion=succ,
                       completion_se=float(np.sqrt(max(succ*(1-succ), 0.)/len(e))),
                       timeout=mean("timeout"), steps=mean("steps"), pulses=mean("pulses"),
                       pulses_on_success=(float(np.mean([r["pulses"] for r in e if r["success"]]))
                                          if any(r["success"] for r in e) else None),
                       collisions=mean("collisions"), collision_rate=count("collisions")/count("steps"),
                       wait_fraction=count("wait_steps")/count("steps"))
            for k in ("wait_listens", "double_listens", "turn_listens", "translate_listens"):
                row[k+"_per_episode"] = mean(k)
                row[k+"_episode_fraction"] = float(np.mean([r[k] > 0 for r in e]))
            for k in ("repeat", "turn", "translate"):
                pairs = count(k+"_pairs")
                row[k+"_pairs"] = pairs
                row[k+"_brier_gain"] = count(k+"_brier_gain")/pairs if pairs else None
                row[k+"_improvement_fraction"] = count(k+"_improved")/pairs if pairs else None
            rows.append(row)
    return rows


def selection_score(rows):
    """Completion first; bounded safety/efficiency term only breaks ties.

    Each full-suite cell has the same n. One additional completion changes the
    primary term by 1/total_n; the ENTIRE tie term is smaller than that. Waiting
    out every episode therefore cannot outrank any policy that completes one.
    """
    if not rows or len({r["n"] for r in rows}) != 1:
        raise ValueError("selection requires balanced full-suite cells")
    epsilon = .25/(sum(r["n"] for r in rows)+1)
    cost = np.mean([(r["collisions"]+.1*r["steps"]+.1*r["pulses"])/(1.2*MAX_STEPS) for r in rows])
    return float(np.mean([r["completion"] for r in rows])-epsilon*np.clip(cost, 0., 1.))


def suite(pool, factory, budgets=BUDGETS, deadline=None):
    start = time.monotonic()
    episodes = []
    for b in budgets:
        e, _ = run_budget(pool, factory, b, deadline)
        episodes.extend(e)
    rows = summarize(episodes)
    return dict(rows=rows, score=selection_score(rows), episodes=episodes,
                seconds=time.monotonic()-start, complete=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--split", choices=["val", "test", "test_turn2", "legacy"], default="test")
    ap.add_argument("--per-group", type=int, default=128)
    ap.add_argument("--train-per-group", type=int, default=256)
    ap.add_argument("--val-per-group", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=3)
    ap.add_argument("--out", default="results/r5/test.json")
    a = ap.parse_args()
    deadline = Deadline(a.max_minutes)
    if Path(a.out).exists():
        raise FileExistsError(f"preserve existing evaluation: {a.out}")
    pool = make_pool(a.split, a.per_group)
    train = make_pool("train", a.train_per_group)
    val = make_pool("val", a.val_per_group)
    assert_disjoint(train, pool)
    if a.split != "val":
        assert_disjoint(val, pool)
    if a.split == "test_turn2":
        assert_disjoint(make_pool("test", 128), pool)
    baseline = json.loads(Path(a.baselines).read_text())
    if not baseline.get("complete") or baseline["manifest"]["hashes"] != val.hashes:
        raise ValueError("baselines must be selected on the same FULL validation set")
    if baseline["config"]["train_per_group"] != a.train_per_group:
        raise ValueError("baseline train audit does not match this experiment")
    factories = {mode: (lambda n, mode=mode: AcousticPlanner(n, mode, baseline["best"][mode]["config"]))
                 for mode in ("event", "uncertainty")}
    # Decoder/flight parameters chosen by validation, no test-time tuning.
    best_mode = max(factories, key=lambda k: baseline["best"][k]["score"])
    factories["every"] = lambda n: AcousticPlanner(n, "every", baseline["best"][best_mode]["config"])
    if a.ckpt:
        from policy_r5 import load_policy, LearnedAgent
        model, saved = load_policy(a.ckpt)
        if saved["config"]["train_per_group"] != a.train_per_group:
            raise ValueError("train pool audit must match checkpoint config")
        factories["learned"] = lambda n: LearnedAgent(model, n)
        factories["learned_every"] = lambda n: LearnedAgent(model, n, every=True)
    out = dict(manifest=pool.manifest(), baseline_file=a.baselines, checkpoint=a.ckpt,
               baseline_sha256=file_digest(a.baselines),
               checkpoint_sha256=file_digest(a.ckpt) if a.ckpt else None,
               selected_update=saved["update"] if a.ckpt else None,
               methods={}, complete=False, scope="R5 physics, including legacy layout comparison")
    try:
        for name, factory in factories.items():
            result = suite(pool, factory, deadline=deadline)
            out["methods"][name] = result
            write_json(a.out, out)
            print(json.dumps(dict(method=name, score=result["score"], seconds=result["seconds"],
                                  rows=len(result["rows"]), complete=True)), flush=True)
        out["complete"] = True
    except TimeoutError:
        out["stop"] = "wall-clock guard; incomplete methods excluded, no headline claim"
    out["seconds"] = deadline.elapsed
    write_json(a.out, out)
    if not out["complete"]:
        raise SystemExit("INCOMPLETE evaluation: reserve more budget or reduce test count before running any method")


if __name__ == "__main__":
    main()
