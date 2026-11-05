"""Round 7 checks: the belief vector (layout, flags, live tracker vs recorded memory vs offline replay), the
pixel-free / belief-vector / belief-head trainer paths with bit-exact resume, the shuffled-vector control, the
partial encoder init, the pixel-free controller, and the opt-in bot_pose recording field."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import p2_train  # noqa: E402
from belief import (AGE_CAP, CLASSES, DIM, OFF_FLAGS, OFF_LAMP, OFF_POSE, BeliefState, BeliefTracker,  # noqa: E402
                    belief_vector, replay_poses, snapshot_of, vectors_from_recording)
from p2_learner import P2Controller, Policy2, cmd_targets, policy_config  # noqa: E402
from p2_train import batch_r7, belief_loss, loss_fn, onset_weights  # noqa: E402
from test_p2 import synthetic_episode  # noqa: E402

TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]


def test_belief_vector_layout_and_flags():
    st = BeliefState()
    snap = {"tracks": [["button", 1.0, 0.0, 5, 10], ["lamp_on", 0.0, 1.0, 4, 3], ["lamp_off", 0.0, 1.0, 3, 8],
                       ["pad_red", -1.0, 0.0, 3, 10], ["pad_red", -2.0, 0.0, 9, 2]], "lamp": ["off", 8]}
    v, last = belief_vector(snap, (0.0, 0.0, 0.0), 10, st)
    assert v.shape == (DIM,) and v.dtype == np.float32
    b = CLASSES.index("button") * 6
    assert np.allclose(v[b:b + 6], [0.0, 1.0, np.log1p(1.0), 1.0, 0.0, 1.0])          # ahead, 1 m, in view now
    la = CLASSES.index("lamp") * 6
    assert np.isclose(v[la], 1.0) and v[la + 3] == 0 and last[CLASSES.index("lamp")] == 8  # latest lamp track wins
    r = CLASSES.index("pad_red") * 6
    assert np.isclose(v[r + 2], np.log1p(2.0)) and last[CLASSES.index("pad_red")] == 2    # most-sighted pad wins
    m = CLASSES.index("mat") * 6
    assert np.allclose(v[m:m + 6], [0, 0, 0, 0, 1.0, 0])                                  # absent: age 1, present 0
    # the lamp belief is the latest confirmed lamp track's class (lamp_off, last seen at 8), not the fresh-run slot
    assert np.allclose(v[OFF_LAMP:OFF_LAMP + 4], [0, 0, 1, np.log1p(2) / np.log1p(AGE_CAP)])
    assert np.allclose(v[OFF_POSE:OFF_POSE + 4], [0, 0, 0, 1]) and v[OFF_FLAGS] == 0 and v[OFF_FLAGS + 1] == 0
    # crossing the remembered plate after the last lamp sighting raises crossed; a later sighting with the other
    # state clears it and raises lamp_changed; one with the same state clears it without.
    v2, _ = belief_vector(snap, (0.9, 0.05, 0.3), 11, st)
    assert v2[OFF_FLAGS] == 1 and v2[OFF_FLAGS + 1] == 0 and np.allclose(v2[OFF_POSE:OFF_POSE + 2], [0.45, 0.025])
    on20 = dict(snap, tracks=snap["tracks"] + [["lamp_on", 0.0, 1.0, 5, 20]])
    v3, _ = belief_vector(on20, (0.9, 0.05, 0.3), 20, st)
    assert v3[OFF_FLAGS] == 0 and v3[OFF_FLAGS + 1] == 1        # seen from the plate: no crossing on the sighting tick
    assert v3[OFF_LAMP + 1] == 1 and v3[OFF_LAMP + 3] == 0
    v4, _ = belief_vector(on20, (0.95, 0.0, 0.3), 21, st)
    assert v4[OFF_FLAGS] == 1 and v4[OFF_FLAGS + 1] == 0        # still there a tick later: crossed again
    on30 = dict(snap, tracks=snap["tracks"] + [["lamp_on", 0.0, 1.0, 6, 30]])
    v5, _ = belief_vector(on30, (0.95, 0.0, 0.3), 30, st)
    assert v5[OFF_FLAGS] == 0 and v5[OFF_FLAGS + 1] == 0        # same state re-seen: cleared, nothing changed
    e, _ = belief_vector({"tracks": [], "lamp": None}, (0.0, 0.0, 0.0), 0, BeliefState())
    assert e[OFF_LAMP] == 1 and e[[k * 6 + 4 for k in range(8)]].sum() == 8 and e.sum() == 1 + 8 + 1


def _bot_and_tracker_run(cfg, ticks):
    """Drive the bot (anchors off) and the tracker on the same observations; return what record.py would store."""
    from interface import Interface
    from refbot import controller
    from room import World
    from tasks import make_episode, start_episode
    ep = make_episode(cfg, "set_lamp", 1)
    w = World(cfg)
    start_episode(w, ep)
    iface = Interface(w)
    fac, bot = controller("set_lamp", ticks, {"anchor_gain": 0.0})
    gen = fac(iface, ep["instruction"])
    tr = BeliefTracker()
    obs = iface.observe()
    cmd = next(gen)
    body = [np.concatenate([obs.body["gyro"], obs.body["gravity"], obs.body["joint_pos"], obs.body["joint_vel"]]).astype(np.float32)]
    imu = [np.concatenate([obs.body["gyro_substeps"], obs.body["gravity_substeps"]], 1).astype(np.float32)]
    mems, online, bot_mems = [snapshot_of(bot.mem, 3)], [tr.vector()[0]], [snapshot_of(bot.mem, 3)]
    bot_poses, tr_poses = [bot.pose()], [tr.odo.pose()]
    try:
        for _ in range(ticks):
            obs = iface.tick(cmd)
            tr.update(obs)
            cmd = gen.send(obs)                      # the bot senses obs here (not when it finishes: StopIteration)
            body.append(np.concatenate([obs.body["gyro"], obs.body["gravity"], obs.body["joint_pos"], obs.body["joint_vel"]]).astype(np.float32))
            imu.append(np.concatenate([obs.body["gyro_substeps"], obs.body["gravity_substeps"]], 1).astype(np.float32))
            mems.append(tr.snapshot())
            bot_mems.append(snapshot_of(bot.mem, 3))
            bot_poses.append(bot.pose())
            tr_poses.append(tr.odo.pose())
            online.append(tr.vector()[0])
    except StopIteration:
        pass
    finally:
        if iface._renderer is not None:
            iface._renderer.close()
    return np.stack(body), np.stack(imu), mems, bot_mems, np.stack(online), np.array(bot_poses), np.array(tr_poses)


def test_tracker_matches_bot_memory_and_offline_replay():
    """The deployed tracker sees what the bot's memory saw (same detector, odometry with anchors off, same tick
    convention), and the offline builder (recorded memory + replayed poses) gives the same vectors."""
    body, imu, mems, bot_mems, online, bot_poses, tr_poses = _bot_and_tracker_run(TRAIN[1], 140)
    assert len(mems) >= 100 and any(m["tracks"] for m in mems[50:])          # the scan saw things
    assert mems == bot_mems                                                   # identical confirmed memory, every tick
    assert np.allclose(bot_poses, tr_poses, atol=1e-9)                        # identical odometry, every tick
    poses = replay_poses(body, imu)
    assert np.abs(poses - tr_poses).max() < 2e-3                              # float32 storage, not a different path
    V, L = vectors_from_recording(mems, poses)
    assert V.shape == online.shape and np.abs(V - online).max() < 2e-3
    assert (L >= -1).all() and L.max() < len(mems)


def _vec_episode(img, T):
    ep = synthetic_episode(img, T)
    rng = np.random.default_rng(T + 1)
    ep["vec"] = rng.normal(size=(T, DIM)).astype(np.float32)
    ep["vec"][:, 5::6] = 1.0                                                  # every class present
    ep["last_seen"] = np.repeat(np.arange(T)[:, None], 8, 1) - rng.integers(0, 4, (T, 8))
    ep["ticks"] = np.arange(T)
    return ep


@pytest.mark.parametrize("extra", [["--no-pixels", "--stream", "--stream-pool", "3", "--stream-warmup", "2"],
                                   ["--belief-heads", "--onset-weight", "20"],
                                   ["--shuffle-vector"]])
def test_r7_training_paths_resume_bit_exact(tmp_path, monkeypatch, extra):
    eps = [_vec_episode(64, t) for t in (12, 18)]
    monkeypatch.setattr(p2_train, "load_all", lambda *a, **k: ([{}, {}], eps))
    common = ["--index", "unused", "--ckpt-every", "4", "--batch", "2", "--window", "4", "--burn", "1", "--seed", "5",
              "--act", "leaky", "--vx-head", "binary", "--turn-head", "discrete", "--mem-vector", *extra]

    def run(name, steps, resume=False):
        monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / name), "--steps", str(steps),
                                         *(["--resume"] if resume else common)])
        p2_train.main()
    run("straight", 12)
    run("resumed", 6)
    run("resumed", 12, resume=True)
    a, b = tmp_path / "straight" / "ckpt_000012", tmp_path / "resumed" / "ckpt_000012"
    wa, wb = mx.load(str(a / "weights.safetensors")), mx.load(str(b / "weights.safetensors"))
    assert wa.keys() == wb.keys() and all(np.array_equal(np.array(wa[k]), np.array(wb[k])) for k in wa)
    meta = json.loads((b / "meta.json").read_text())["cfg"]
    assert meta["mem_vector"] and meta["no_pixels"] == ("--no-pixels" in extra) and meta["belief_heads"] == ("--belief-heads" in extra)
    assert ("c1.weight" in wa) == ("--no-pixels" not in extra) and "vec.weight" in wa
    assert ("belief.weight" in wa) == ("--belief-heads" in extra)
    log = [json.loads(l) for l in (b / ".." / "train_log.jsonl").read_text().splitlines()] if (b / ".." / "train_log.jsonl").exists() else []
    assert isinstance(log, list)
    cfg = policy_config(b / "weights.safetensors")
    assert cfg["vec_dim"] == DIM and cfg["pixels"] == ("--no-pixels" not in extra)


def test_belief_loss_mask_and_onset_weights():
    B, T = 1, 4
    V = np.zeros((B, T, DIM), np.float32)
    k = CLASSES.index("button")
    V[0, :, k * 6 + 0] = 1.0                       # sin = 1 (bearing +90 deg), cos 0
    V[0, :, k * 6 + 5] = 1.0                       # present every tick
    V[0, :, k * 6 + 3] = [0, 1, 1, 0]
    LS = np.full((B, T, 8), -1, np.int64)
    LS[0, :, k] = [1, 1, 6, 6]                     # sighting inside the window (start 5) only at ticks 2, 3
    WS = np.array([5])
    M = np.array([[1., 1., 1., 0.]])
    pred = mx.zeros((B, T, 32))
    loss, n, deg = belief_loss(pred, mx.array(V), mx.array(LS), mx.array(WS), mx.array(M))
    assert float(n) == 1                           # tick 2 only: inside the window and in the loss mask
    assert np.isclose(float(loss), (1.0 + 0.0 + 0.0) + np.log(2), atol=1e-6)   # (sin-0)^2 + bce(0 logit, in view)
    assert np.isclose(float(deg), 90.0, atol=1e-3)
    cmds = np.zeros((6, 8), np.float32)
    cmds[:, 0] = [0, 0, 0, 0.3, 0.3, 0]
    cmds[:, 2] = [0, 0.55, 0.55, 0, 0, 0.55]
    R = mx.array(cmd_targets(cmds)[0][None])
    w = np.array(onset_weights(R, 20.0))[0]
    assert w.tolist() == [1, 20, 1, 20, 1, 1]     # stand->turn, turn->walk are onsets; holds and walk->turn are not
    assert np.all(np.array(onset_weights(R, 1.0)) == 1)
    cmds2 = np.zeros((3, 8), np.float32)
    cmds2[:, 0] = [0.3, 0, 0]
    cmds2[:, 2] = [0, 0.55, 0.55]
    assert np.array(onset_weights(mx.array(cmd_targets(cmds2)[0][None]), 20.0))[0].tolist() == [1, 1, 1]  # walk->turn is not


def test_shuffle_vector_control_draws_another_episode():
    eps = [_vec_episode(64, t) for t in (30, 40)]
    lens = np.array([30.0, 40.0])
    a = batch_r7(eps, lens, np.random.default_rng(3), 4, W=8, BURN=2, img=64)
    b = batch_r7(eps, lens, np.random.default_rng(3), 4, W=8, BURN=2, img=64, shuffle=True)
    plain = p2_train.batch(eps, lens, np.random.default_rng(3), 4, W=8, BURN=2, img=64)
    assert len(a) == 11 and all(np.array_equal(x, y) for x, y in zip(a[:8], plain))     # the old draw is unchanged
    assert all(np.array_equal(x, y) for x, y in zip(a[:8], b[:8]))                        # shuffle only moves vectors
    V, LS, WS = a[8:]
    for r in range(4):
        e = eps[0] if np.array_equal(a[1][r, 0], eps[0]["body"][WS[r]]) else eps[1]
        n = min(8, len(e["vec"]) - int(WS[r]))
        assert np.array_equal(a[1][r, :n], e["body"][WS[r]:WS[r] + n])
        assert np.array_equal(V[r, :n], e["vec"][WS[r]:WS[r] + n]) and np.array_equal(LS[r, :n], e["last_seen"][WS[r]:WS[r] + n])
        assert not np.array_equal(b[8][r, :n], V[r, :n]) and (b[9][r] == -1).all()
    c = batch_r7(eps, lens, np.random.default_rng(3), 4, W=8, BURN=2, img=64, frames=False)
    assert c[0] is None and all(np.array_equal(x, y) for x, y in zip(c[1:], a[1:]))


def test_pixel_free_controller_runs_the_tracker(tmp_path):
    from room import World
    from tasks import make_episode, run_episode
    cfg = TRAIN[0]
    model = Policy2(act="leaky", vx_head="binary", turn_head="discrete", vec_dim=DIM, pixels=False)
    reg, _ = cmd_targets(np.zeros(8, np.float32))
    model.reg.weight = mx.zeros_like(model.reg.weight)
    model.reg.bias = mx.array(np.arctanh(reg))
    model.done.weight = mx.zeros_like(model.done.weight)
    model.done.bias = mx.array([-20.])
    weights = tmp_path / "weights.safetensors"
    model.save_weights(str(weights))
    (tmp_path / "p2_config.json").write_text(json.dumps({"img": 64, "aux": True, "act": "leaky", "vx_head": "binary",
                                                         "turn_head": "discrete", "mem_vector": True, "no_pixels": True}))
    device, iface = mx.default_device(), None
    try:
        pc = P2Controller(weights)
        assert pc.cfg["vec_dim"] == DIM and not pc.cfg["pixels"] and not hasattr(pc.model, "c1")
        ep = make_episode(cfg, "set_lamp", 0)
        ep["budget_ticks"] = 25
        result, _, iface = run_episode(cfg, ep, pc.factory, world=World(cfg))
        assert iface.tick_count == 25 and result["falls"] == 0
        assert len(pc.tracker_latency) == 25 and pc.tracker.tick == 25      # sensing from tick 1 on, one update per tick
        assert len(pc.tracker.poses) == 26 and np.all(np.array(pc.tracker.poses[0]) == 0)
    finally:
        if iface is not None and iface._renderer is not None:
            iface._renderer.close()
        mx.set_default_device(device)


def test_encoder_only_init_and_freeze(tmp_path, monkeypatch):
    mx.random.seed(11)
    donor = Policy2(act="leaky", vx_head="binary", turn_head="discrete")
    donor_w = tmp_path / "donor.safetensors"
    donor.save_weights(str(donor_w))
    eps = [_vec_episode(64, t) for t in (12, 18)]
    monkeypatch.setattr(p2_train, "load_all", lambda *a, **k: ([{}, {}], eps))
    monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / "run"), "--index", "unused", "--steps", "3",
                                     "--ckpt-every", "3", "--batch", "2", "--window", "4", "--burn", "1", "--act", "leaky",
                                     "--vx-head", "binary", "--turn-head", "discrete", "--mem-vector",
                                     "--init", str(donor_w), "--init-encoder-only", "--freeze-encoder"])
    p2_train.main()
    out = mx.load(str(tmp_path / "run" / "ckpt_000003" / "weights.safetensors"))
    src = mx.load(str(donor_w))
    for name in p2_train.ENCODER_LAYERS:
        for p in ("weight", "bias"):
            assert np.array_equal(np.array(out[f"{name}.{p}"]), np.array(src[f"{name}.{p}"]))   # loaded and frozen
    assert not np.array_equal(np.array(out["reg.weight"]), np.array(src["reg.weight"]))       # heads were fresh
    opt = mx.load(str(tmp_path / "run" / "ckpt_000003" / "opt.safetensors"))
    assert not any(k.split(".")[0] == "c1" for k in opt)                                          # no optimizer state for frozen layers


def test_record_bot_pose_is_opt_in(tmp_path):
    from record import record_episode
    cfg = TRAIN[2]
    r = record_episode(cfg, "train", "set_lamp", 1, out_dir=tmp_path / "a", budget_ticks=30, bot_pose=True)
    z = np.load(r["path"])
    H = json.loads(str(z["header"]))
    assert z["bot_pose"].shape == (H["ticks"], 3) and z["bot_pose"].dtype == np.float32 and H["bot_pose"] is True
    assert np.all(z["bot_pose"][0] == 0) and np.abs(z["bot_pose"][-1]).sum() > 0
    r0 = record_episode(cfg, "train", "set_lamp", 1, out_dir=tmp_path / "b", budget_ticks=30)
    z0 = np.load(r0["path"])
    assert "bot_pose" not in z0.files and "bot_pose" not in json.loads(str(z0["header"]))
    for k in ("frames", "body", "cmd_exec", "truth"):
        assert np.array_equal(z[k], z0[k])
