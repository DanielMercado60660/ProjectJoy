"""End-to-end smoke test at tiny size (< 2 min). Run from the project folder:
   ../.venv/bin/python -m pytest -q tests"""
import json, os, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); os.chdir(HERE)
import flight as F
import scene as S
import track_classical as TC

PY = sys.executable


def test_flight_env_and_pd():
    C = F.make_courses(np.random.default_rng(5), 16)
    r = F.run_episodes(C, lambda e, o: F.pd_action(e, F.PD_DEFAULT), seed=1)
    assert r["complete"].mean() > 0.5
    r0 = F.run_episodes(C, lambda e, o: np.zeros((len(o), 2)), seed=1)
    assert r0["complete"].sum() == 0 and r0["crash"].all()   # no thrust -> falls


def test_ppo_tiny(tmp_path):
    out = subprocess.run([PY, "train_flight.py", "--steps", "65536", "--eval_every", "2",
                          "--out", str(tmp_path / "f"), "--log", str(tmp_path / "f.jsonl")],
                         capture_output=True, text=True, timeout=90)
    assert out.returncode == 0, out.stderr
    L = [json.loads(l) for l in open(tmp_path / "f.jsonl")]
    assert len(L) == 4 and np.isfinite(L[-1]["approx_kl"])
    assert os.path.exists(tmp_path / "f.npz")


def test_scenes_and_classical():
    for c in S.CONDS:
        fr, gt, vis = S.render_scene(S.COND_SEED[c], c, n_frames=6)
        assert fr.shape == (6, 80, 80, 3) and fr.dtype == np.uint8 and gt.shape == (6, 2)
    fr, gt, vis = S.render_scene(123, "train", n_frames=12)
    prm = dict(sigma=0.08, min_lum=0.05, fg_scale=0.1, fg_pow=1.0, cw=0.5, thr=0.3, amin=2, amax=60,
               q=1.0, r=1.0, gate=6, K=3, mode="product", reinit=3, use_kf=True)
    est = TC.track_scene(TC.precompute(fr), prm)
    assert est.shape == (12, 2) and np.isfinite(est).all()


def test_cnn_tiny(tmp_path):
    out = subprocess.run([PY, "train_tracker.py", "--scenes", "8", "--val_scenes", "2", "--steps", "60",
                          "--eval_every", "30", "--out", str(tmp_path / "d"), "--log", str(tmp_path / "d.jsonl")],
                         capture_output=True, text=True, timeout=90)
    assert out.returncode == 0, out.stderr
    L = [json.loads(l) for l in open(tmp_path / "d.jsonl")]
    assert L[-1]["loss"] < 6.0
    assert os.path.exists(tmp_path / "d.safetensors")


@pytest.mark.skipif(not os.path.exists("checkpoints/flight_ppo.npz") or not os.path.exists("checkpoints/ducknet.safetensors"),
                    reason="trained checkpoints not present")
def test_checkpoints_reload():
    import train_flight as TF, train_tracker as TT, eval_tracker as ET
    pol = TF.policy_fn(TF.load("checkpoints/flight_ppo"))
    r = F.run_episodes(F.heldout_courses(8), pol, seed=99)
    assert r["complete"].mean() >= 0.75
    net = ET.load_net()
    fr, gt, vis = S.render_scene(S.COND_SEED["iid"], "iid", n_frames=10)
    xy, _ = TT.predict(net, fr)
    assert np.median(np.linalg.norm(xy - gt, axis=1)) < 3.0


def test_pd_plus_baseline():
    """Reviewer-added PD+ baseline: runs, respects the thrust limit, and completes courses."""
    import pd_plus as P
    C = F.make_courses(np.random.default_rng(5), 16)
    env = F.FlightEnv(16, 1, fixed_courses=C)
    a = P.pd_plus_action(env, P.DEFAULT)
    assert np.all(np.linalg.norm(a, axis=1) <= 1.0 + 1e-9)
    r = F.run_episodes(C, lambda e, o: P.pd_plus_action(e, P.DEFAULT), seed=1)
    assert r["complete"].mean() > 0.5
