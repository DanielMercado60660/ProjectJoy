"""End-to-end smoke tests at tiny size (run: ../.venv/bin/python -m pytest -q)."""
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

from racing.baselines import PurePursuit, RayFollower  # noqa: E402
from racing.evaluate import run_lap, summarize  # noqa: E402
from racing.policy import forward, init_theta, load_policy, n_params, save_policy  # noqa: E402
from racing.sim import N_RAYS, RAY_LEN, VecRace  # noqa: E402
from racing.track import make_track, make_tracks, pack  # noqa: E402

PY = sys.executable


@pytest.fixture(scope="module")
def bank():
    return pack(make_tracks("heldout", 3))


def test_tracks_deterministic_and_split():
    a, b = make_track(0, "train"), make_track(0, "train")
    assert np.allclose(a.center, b.center)
    h = make_track(0, "heldout")
    assert h.seed != a.seed and not np.allclose(h.center[:5], a.center[:5])
    assert 200 < a.length < 1500


def test_drive_forward_makes_progress(bank):
    env = VecRace(bank, 3)
    env.reset(np.arange(3))
    for _ in range(15):
        env.step(np.tile([0.0, 1.0], (3, 1)))
    assert (env.prog > 5).all() and (env.v > 5).all()


def test_reversing_loses_progress(bank):
    """Reward-hack guard: driving the wrong way must give negative progress."""
    env = VecRace(bank, 3)
    env.reset(np.arange(3))
    env.yaw += np.pi
    for _ in range(15):
        env.step(np.tile([0.0, 0.6], (3, 1)))
    alive = ~env.crashed
    assert (env.prog[alive] < -3).all()


def test_wall_ends_episode(bank):
    env = VecRace(bank, 3)
    env.reset(np.arange(3))
    for _ in range(100):
        env.step(np.tile([1.0, 1.0], (3, 1)))
    assert env.crashed.all() and env.done.all()


def test_side_rays_measure_track_width(bank):
    env = VecRace(bank, 3)
    obs = env.reset(np.arange(3))
    width = (obs[:, 0] + obs[:, N_RAYS - 1]) * RAY_LEN
    assert np.allclose(width, 2 * bank.half_width, rtol=0.15)


def test_baselines_and_eval(bank):
    s = summarize(run_lap(PurePursuit(bank), bank, grip=1.0))
    assert s["completion"] == 1.0  # privileged controller should get round easy tracks
    r = run_lap(RayFollower(), bank, grip=1.0, mode="respawn")
    assert set(["finished", "lap_time", "collisions"]) <= set(r)


def test_policy_save_load(tmp_path):
    th = init_theta(np.random.default_rng(0))
    assert th.size == n_params()
    p = tmp_path / "p.npz"
    save_policy(p, th, {"x": 1})
    th2, meta = load_policy(p)
    assert np.array_equal(th, th2) and meta["x"] == 1
    assert forward(th, np.zeros((4, 13))).shape == (4, 2)


def _cleanup(tag):
    for f in glob.glob(os.path.join(ROOT, "checkpoints", f"{tag}_*")) + glob.glob(os.path.join(ROOT, "logs", f"{tag}_*")):
        os.remove(f)


def test_es_tiny_and_resume():
    tag = "pytest_es"
    _cleanup(tag)
    args = [PY, "train_es.py", "--gens", "10", "--half", "4", "--episodes", "2", "--steps", "40",
            "--n_train_tracks", "4", "--n_val", "4", "--tag", tag]
    subprocess.run(args, cwd=ROOT, check=True, capture_output=True)
    st = np.load(os.path.join(ROOT, "checkpoints", f"{tag}_state.npz"))
    assert int(st["gen"]) == 10
    args[3] = "12"
    out = subprocess.run(args + ["--resume", f"checkpoints/{tag}_state.npz"], cwd=ROOT, check=True,
                         capture_output=True, text=True)
    assert "resumed" in out.stdout
    assert os.path.exists(os.path.join(ROOT, "checkpoints", f"{tag}_best.npz"))
    _cleanup(tag)


def test_ppo_tiny():
    tag = "pytest_ppo"
    _cleanup(tag)
    subprocess.run([PY, "train_ppo.py", "--max_updates", "10", "--envs", "16", "--horizon", "32",
                    "--minibatch", "128", "--n_train_tracks", "4", "--n_val", "4", "--max_steps", "60",
                    "--tag", tag], cwd=ROOT, check=True, capture_output=True)
    th, meta = load_policy(os.path.join(ROOT, "checkpoints", f"{tag}_best.npz"))
    assert th.size == n_params() and meta["method"] == "ppo"
    _cleanup(tag)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_port_matches_python(bank, tmp_path):
    """The JS sim in web/sim.js (used by demo.html) reproduces the Python sim."""
    t = bank.tracks[0]
    th = init_theta(np.random.default_rng(1)).astype(np.float64)
    rf = RayFollower()
    env = VecRace(bank, 1)
    obs = env.reset([0])
    xs = []
    for _ in range(120):
        obs, _, done = env.step(rf(obs))
        xs.append([env.x[0], env.y[0]])
    rng = np.random.default_rng(2)
    o = rng.random((5, 13))
    data = dict(track=dict(center=t.center.tolist(), tangent=t.tangent.tolist(), hw=t.half_width, length=t.length),
                theta=th.tolist(), obs=o.tolist(), params=rf.p)
    (tmp_path / "d.json").write_text(json.dumps(data))
    js = f"""
const S = require({json.dumps(os.path.join(ROOT, 'web', 'sim.js'))});
const d = JSON.parse(require('fs').readFileSync({json.dumps(str(tmp_path / 'd.json'))}));
const tr = S.prepTrack(d.track); const c = S.newCar(tr, 1.0); const xs = [];
for (let i = 0; i < 120; i++) {{ const a = S.rayFollower(d.params, S.observe(tr, c)); S.stepCar(tr, c, a[0], a[1]); xs.push([c.x, c.y]); }}
console.log(JSON.stringify({{xs, mlp: d.obs.map(o => S.mlpForward(d.theta, o))}}));
"""
    out = json.loads(subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True).stdout)
    assert np.allclose(np.array(out["xs"]), np.array(xs), atol=1e-6)
    assert np.allclose(np.array(out["mlp"]), forward(th.astype(np.float32), o), atol=1e-4)
