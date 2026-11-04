"""R6 smoke tests (flight trainer with resume, PD+ retune helpers, hunter game). Tiny sizes.
   ../.venv/bin/python -m pytest -q tests"""
import json, os, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); os.chdir(HERE)
import flight as F
import hunter as H

PY = sys.executable


def _run(args):
    out = subprocess.run([PY, "train_flight_r6.py"] + args, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]


def test_r6_flight_resume_is_exact():
    names = ["_test_a", "_test_b"]
    try:
        common = ["--envs", "32", "--horizon", "64", "--mb", "1024", "--eval_every", "4096", "--save_every", "4096",
                  "--total", "16384", "--lr_sched", "linear"]
        _run(["--run", names[0], "--steps", "16384"] + common)
        _run(["--run", names[1], "--steps", "8192"] + common)
        _run(["--run", names[1], "--steps", "16384", "--resume"])
        a = np.load(f"checkpoints/r6/{names[0]}/last.npz"); b = np.load(f"checkpoints/r6/{names[1]}/last.npz")
        assert max(float(np.abs(a[k] - b[k]).max()) for k in a.files) == 0.0
    finally:
        for n in names:
            shutil.rmtree(f"checkpoints/r6/{n}", ignore_errors=True)
            if os.path.exists(f"results/r6_train_{n}.jsonl"):
                os.remove(f"results/r6_train_{n}.jsonl")


def test_r6_residual_starts_as_pd_plus():
    import train_flight_r6 as R6, pd_plus as P
    cfg = dict(residual=1, res_scale=0.5, hid=32)
    agent = R6.Agent(cfg)
    m = R6.Policy(agent.obs_dim, 32, -1.6, zero_last=True)
    C = F.make_courses(np.random.default_rng(5), 8)
    prm = json.load(open("checkpoints/pd_plus_params.json"))
    r1 = F.run_episodes(C, R6.make_policy_fn(m, agent), seed=1)
    r2 = F.run_episodes(C, lambda e, o: P.pd_plus_action(e, prm), seed=1)
    assert np.allclose(r1["t"], r2["t"], equal_nan=True)


def test_remaining_path_decreases_along_course():
    import train_flight_r6 as R6
    C = F.make_courses(np.random.default_rng(3), 4)
    env = F.FlightEnv(4, 0, fixed_courses=C)
    L0 = R6.remaining_path(env)
    legs = np.linalg.norm(np.diff(C[:, 1:], axis=1), axis=2).sum(1)
    assert np.all(L0 >= legs - 1e-9)
    env.k[:] = F.N_WP; env.p = C[:, F.N_WP].copy()
    assert np.allclose(R6.remaining_path(env), 0.0)


def test_hunter_scripted_game():
    C = H.train_courses(32)
    rng = np.random.default_rng(0)
    r = H.play(C, H.make_agent("straight", 32, rng), H.make_agent("lead", 32, rng))
    s = H.summarize(r)
    assert abs(s["hit_rate"] + s["survival"] + s["crash"] + s["timeout"] - 1.0) < 1e-9
    assert s["shots"] <= H.AMMO
    r0 = H.play(C, H.make_agent("straight", 32, rng), H.make_agent("naive", 32, rng))
    assert r0["complete"].mean() > 0.5
    # a hunter that never fires cannot hit
    class Hold:
        def reset(self, idx): pass
        def act(self, env): return np.tile([0.0, 0.0, -1.0], (env.n, 1))
    r1 = H.play(C, H.make_agent("straight", 32, rng), Hold())
    assert r1["hit"].sum() == 0 and r1["complete"].mean() > 0.9


def test_hunter_tiny_training():
    import train_hunter as TH
    try:
        m, o = TH.new_learner("hunter", 0)
        spec = TH.train_role("hunter", m, o, ["straight", "dodge"], 8192, "_test_h", 0, envs=32, horizon=128, mb=2048)
        rng = np.random.default_rng(0)
        s = H.summarize(H.play(H.train_courses(16), H.make_agent("jink", 16, rng), H.make_agent(spec, 16, rng)))
        assert 0.0 <= s["hit_rate"] <= 1.0
    finally:
        for p in ["checkpoints/r6h/_test_h.npz", "results/r6h_train__test_h.jsonl"]:
            if os.path.exists(p):
                os.remove(p)
