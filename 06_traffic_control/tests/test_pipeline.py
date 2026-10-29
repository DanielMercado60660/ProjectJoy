import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import sim
from sim import Grid, demand, run_episode
import baselines as bl


def _random_ctrl(seed):
    rng = np.random.default_rng(seed)
    return lambda env: rng.integers(0, 2, (env.B, sim.N_INT))


def test_conservation_and_safety():
    R = np.stack([demand("rush")] * 3)
    env = Grid(3).reset(R, 1)
    ctrl = _random_ctrl(0)
    while not env.done:
        before = env.dep.copy()
        g = env.green()
        env.step_decision(ctrl(env))
        # never two phases at once: per intersection, green lanes all belong to one phase
        g = env.green().reshape(3, sim.N_INT, 4)
        assert not (g[..., :2].any(-1) & g[..., 2:].any(-1)).any()
        # internal links never overfill
        assert (env.cnt[:, sim.DOWN[sim.INTERNAL_SRC]] <= sim.LINK_CAP).all()
        assert (env.dep >= before).all()
    # every vehicle that entered either left or is still on the map
    assert (env.entered == env.exited + env.cnt.sum(1)).all()
    assert (env.ready <= env.cnt).all() and (env.ready >= 0).all()


def test_departures_only_on_green():
    R = np.stack([demand("steady")] * 2)
    env = Grid(2).reset(R, 3)
    ctrl = _random_ctrl(1)
    while env.t < 400:
        env.phase[:] = ctrl(env)  # direct phase set, no yellow, to probe single steps
        g = env.green()
        d0 = env.dep.copy()
        env.sim_step()
        assert ((env.dep - d0) <= g).all()


def test_queue_rule_starves_quiet_street_in_rush():
    """The motivating failure: longest-queue-first lets the quiet street wait far longer than fixed time."""
    R = np.stack([demand("rush")] * 4)
    lqf = run_episode(bl.longest_queue_first, R, 11).metrics()
    ft = run_episode(bl.fixed_time([(3, 2), (4, 1), (3, 2), (4, 1)]), R, 11).metrics()
    assert lqf["quiet_max_wait"].mean() > ft["quiet_max_wait"].mean()


def test_tiny_train_and_reload(tmp_path):
    out = str(tmp_path / "ppo")
    py = sys.executable
    r = subprocess.run([py, "train.py", "--max-iters", "2", "--envs", "2", "--val-every", "1",
                        "--out", out, "--log", str(tmp_path / "log.jsonl")], cwd=HERE,
                       capture_output=True, text=True, timeout=110)
    assert r.returncode == 0, r.stderr[-2000:]
    from policy import load_policy, greedy_controller
    model, meta = load_policy(out + "_last")
    env = run_episode(greedy_controller(model), np.stack([demand("steady")]), 5)
    assert env.metrics()["throughput"][0] > 0
