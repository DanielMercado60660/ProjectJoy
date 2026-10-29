"""M5 checks: the learner's input path, a training step, checkpoint reload, and one closed-loop run.
No recorded episodes needed (episodes/ is gitignored)."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import interface
from learner import IMG, Policy, PolicyController, body_vec, cmd_targets, decode, instr_vec, prep_frame
from m5_train import batch, loss_fn
from record import down
from room import World
from tasks import make_episode, run_episode

HERE = Path(__file__).resolve().parent.parent
CFG = json.loads((HERE / "splits" / "train.json").read_text())["configs"][0]


def test_live_frames_match_recorded_path():
    f = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)
    assert np.array_equal(prep_frame(f), prep_frame(down(f)))        # 128 -> 96 -> 64, as recorded
    assert prep_frame(f).shape == (IMG, IMG, 3)


def test_targets_round_trip():
    cmd = np.array([0.25, 0.0, -0.3, 0.1, -0.2, 0.5, 0.05, 1.0], np.float32)
    reg, done = cmd_targets(cmd)
    c = decode(reg, 5.0 if done > 0.5 else -5.0)
    assert np.allclose([c.vx, c.wz, *c.head], [0.25, -0.3, 0.1, -0.2, 0.5, 0.05], atol=1e-5) and c.done


def test_one_training_step_and_reload(tmp_path):
    rng = np.random.default_rng(0)
    T = 40
    ep = {"frames": (rng.random((T, IMG, IMG, 3)) * 255).astype(np.uint8), "body": rng.normal(size=(T, 34)).astype(np.float32),
          "instr": instr_vec({"task": "set_lamp", "target": "on"}), "reg": rng.uniform(-1, 1, (T, 6)).astype(np.float32),
          "done": np.eye(T, dtype=np.float32)[-1]}
    F, Bo, In, R, D, M = batch([ep], np.array([T], float), rng, 2, W=16, BURN=4)
    assert M.sum() > 0
    mx.random.seed(0)
    model = Policy()
    opt = optim.Adam(learning_rate=1e-3)
    vg = nn.value_and_grad(model, loss_fn)
    args = [mx.array(x) for x in (F, Bo, In, R, D, M)]
    (l0, _), g = vg(model, *args)
    opt.update(model, g)
    (l1, _), _ = vg(model, *args)
    assert float(l1) < float(l0)
    model.save_weights(str(tmp_path / "w.safetensors"))
    m2 = Policy()
    m2.load_weights(str(tmp_path / "w.safetensors"))
    a = model(args[0][:1].astype(mx.float32) / 255.0, args[1][:1], args[2][:1])[0]
    b = m2(args[0][:1].astype(mx.float32) / 255.0, args[1][:1], args[2][:1])[0]
    assert np.allclose(np.array(a), np.array(b))


def test_saved_policy_runs_closed_loop():
    """The checkpoint reported in M6 drives the duck through the frozen interface for 30 ticks."""
    ep = make_episode(CFG, "go_and_return", 0)
    pc = PolicyController(HERE / "checkpoints" / "m5_dagger.safetensors")
    n = {"t": 0}

    def fac(iface, instruction):
        g = pc.factory(iface, instruction)
        cmd = next(g)
        while n["t"] < 30:
            n["t"] += 1
            obs = yield cmd
            cmd = g.send(obs)
        yield interface.Command(vx=0.0, vy=0.0, wz=0.0, head=(0, 0, 0, 0), done=True)
    res, w, it = run_episode(CFG, ep, fac, world=World(CFG))
    assert n["t"] == 30 or res["declared_done"]
    assert len(pc.latency) >= 1 and res["falls"] == 0
    assert body_vec(it.observe().body).shape == (34,)


def test_resume_continues_bit_exact(tmp_path, monkeypatch):
    """REVIEW_M5: 4 steps + save + --resume to 8 gives the same weights as 8 straight steps (AGENTS.md
    asks for checkpoint/resume to be verified; no real run used --resume)."""
    import sys
    import m5_train
    rng = np.random.default_rng(1)
    eps = []
    for T in (30, 45):
        eps.append({"frames": (rng.random((T, IMG, IMG, 3)) * 255).astype(np.uint8),
                    "body": rng.normal(size=(T, 34)).astype(np.float32),
                    "instr": instr_vec({"task": "go_and_return", "target": "red"}),
                    "reg": rng.uniform(-1, 1, (T, 6)).astype(np.float32), "done": np.eye(T, dtype=np.float32)[-1]})
    monkeypatch.setattr(m5_train, "load_all", lambda idx: ([{}, {}], eps))
    common = ["--index", "unused", "--ckpt-every", "4", "--batch", "2", "--window", "16", "--burn", "4"]

    def run(d, steps, *extra):
        monkeypatch.setattr(sys, "argv", ["m5_train.py", str(tmp_path / d), "--steps", str(steps), *common, *extra])
        m5_train.main()

    run("a", 8)
    run("b", 4)
    run("b", 8, "--resume")
    a = mx.load(str(tmp_path / "a" / "ckpt_000008" / "weights.safetensors"))
    b = mx.load(str(tmp_path / "b" / "ckpt_000008" / "weights.safetensors"))
    assert all(np.array_equal(np.array(a[k]), np.array(b[k])) for k in a)
    ma = json.loads((tmp_path / "a" / "ckpt_000008" / "meta.json").read_text())
    mb = json.loads((tmp_path / "b" / "ckpt_000008" / "meta.json").read_text())
    assert ma["rng"] == mb["rng"] and mb["step"] == 8
