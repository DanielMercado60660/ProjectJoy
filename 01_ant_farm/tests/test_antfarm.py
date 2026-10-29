import os, shutil, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import sim
import handwritten as hw


def run(w, pol, steps):
    for _ in range(steps):
        pm, pd = pol(sim.observe(w))
        m, d = sim.sample_actions(w, pm, pd)
        sim.step(w, m, d)


def same(a, b):
    da, db = a.state_dict(), b.state_dict()
    return all(np.array_equal(da[k], db[k]) for k in da)


def test_deterministic():
    a = sim.make_worlds(np.arange(3), 16)
    b = sim.make_worlds(np.arange(3), 16)
    run(a, hw.compass, 150)
    run(b, hw.compass, 150)
    assert same(a, b)
    assert a.delivered.sum() >= 0 and a.pher.sum() > 0


def test_save_restore_continues_identically(tmp_path):
    a = sim.make_worlds(np.arange(2), 16)
    run(a, hw.compass, 80)
    path = str(tmp_path / "w.npz")
    a.save(path)
    b = sim.World.load(path)
    run(a, hw.compass, 80)
    run(b, hw.compass, 80)
    assert same(a, b)


def test_fork_world_changes_one_condition():
    a = sim.make_worlds(np.arange(1), 16)
    run(a, hw.compass, 50)
    b = a.copy()
    sim.block_route(b, 0)
    assert (b.obst != a.obst).any() and np.array_equal(a.rng, b.rng)


def test_observation_is_local():
    """Food, rocks and pheromone further than 2 cells from every ant must not change any obs."""
    w = sim.make_worlds(np.arange(2), 8)
    run(w, hw.compass, 30)
    o1 = sim.observe(w).copy()
    w2 = w.copy()
    near = np.zeros((w.B, sim.H, sim.W), bool)
    for b in range(w.B):
        for (y, x) in w.pos[b]:
            near[b, max(0, y - 2):y + 3, max(0, x - 2):x + 3] = True
    w2.food[~near] = 0
    w2.pher[:, 0][~near] = 0
    w2.pher[:, 1][~near] = 0
    w2.obst[~near] = 1 - w2.obst[~near]
    o2 = sim.observe(w2)
    assert np.array_equal(o1, o2)


def test_handwritten_beats_random():
    res = {}
    for k in ("random", "compass"):
        w = sim.make_worlds(np.arange(4), 32)
        run(w, hw.POLICIES[k], 300)
        res[k] = w.delivered.mean()
    assert res["compass"] > 5 * (res["random"] + 1)


def test_route_block_and_metrics():
    import evaluate as E
    ps = np.zeros((E.BLOCK_AT + E.REC_STEPS, 2), np.float32)
    ps[:E.BLOCK_AT] = 1.0
    ps[E.BLOCK_AT + 150:] = 0.9
    r = E.recovery_stats(ps)
    assert r["t_recover_80"] is not None and 150 <= r["t_recover_80"] <= 260


def test_numpy_actor_matches_mlx():
    import policy
    m = policy.Policy(hid=32)
    o = np.random.default_rng(0).random((2, 5, sim.OBS_DIM)).astype(np.float32)
    pm, pd = policy.probs(m, o)
    nm, nd = policy.NumpyActor(m)(o)
    assert np.allclose(pm, nm, atol=1e-5) and np.allclose(pd, nd, atol=1e-5)


def test_train_smoke_and_resume():
    run_name = "pytest_smoke"
    shutil.rmtree(os.path.join(HERE, "checkpoints", run_name), ignore_errors=True)
    cmd = [sys.executable, "train.py", "--run", run_name, "--B", "2", "--N", "16", "--T", "50",
           "--T_EP", "100", "--mb", "512", "--max_updates", "2", "--eval_every", "1000"]
    subprocess.run(cmd, cwd=HERE, check=True, capture_output=True)
    out = subprocess.run(cmd[:-4] + ["--max_updates", "4", "--eval_every", "1000", "--resume"], cwd=HERE,
                         check=True, capture_output=True, text=True).stdout
    assert "resumed from update 2" in out
    shutil.rmtree(os.path.join(HERE, "checkpoints", run_name), ignore_errors=True)
    try:
        os.remove(os.path.join(HERE, "logs", run_name + ".jsonl"))
    except FileNotFoundError:
        pass
