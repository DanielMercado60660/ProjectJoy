"""M2 checks: the frozen interface, skills on a hand-made room, deterministic episodes, the task 3
launch staying bit-exact through a session, and one oracle episode per task."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import session  # noqa: E402
from configs import hand_config  # noqa: E402
from interface import BOUNDS, Command, Interface, frozen_spec  # noqa: E402
from oracle import controller_for  # noqa: E402
from policy.velstand_np import VelstandPolicy  # noqa: E402
from room import World  # noqa: E402
from interface import Command as Cmd  # noqa: E402
from skills import Ctx, TruePose, go_to, look, plan_path, press_button, run, turn_to  # noqa: E402
from tasks import TASKS, make_episode, run_episode, start_episode  # noqa: E402

TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]


@pytest.fixture(scope="module")
def policy():
    return VelstandPolicy()


def test_interface_is_frozen():
    f = json.loads((ROOT / "results" / "m2" / "interface_frozen.json").read_text())
    assert f["spec"] == json.loads(json.dumps(frozen_spec()))
    assert f["interface_py_sha256"] == hashlib.sha256((ROOT / "interface.py").read_bytes()).hexdigest()


def test_command_clipping_and_tick(policy):
    w = World(hand_config(empty=True), policy=policy)
    it = Interface(w)
    c = Command(vx=2, vy=0.5, wz=-3, head=(5, -5, 5, 5)).clipped()
    assert (c.vx, c.vy, c.wz) == (BOUNDS["vx"][1], 0.0, BOUNDS["wz"][0])
    assert np.allclose(c.head, [BOUNDS["head"][0], -BOUNDS["head"][1], BOUNDS["head"][2], BOUNDS["head"][3]])
    obs = it.tick(Command(vx=0.3))
    assert w.step_count == 5 and obs.tick == 1 and abs(obs.sim_time - 0.1) < 1e-9
    assert obs.frame().shape == (128, 128, 3)
    assert set(obs.body) == {"gyro", "gravity", "joint_pos", "joint_vel", "gyro_substeps", "gravity_substeps"}
    assert obs.body["gyro_substeps"].shape == (5, 3) and np.array_equal(obs.body["gyro_substeps"][-1], obs.body["gyro"])
    assert np.array_equal(obs.body["gravity_substeps"][-1], obs.body["gravity"])
    it.tick(Command(done=True))
    assert it.done


def test_skills_on_hand_room(policy):
    w = World(hand_config(empty=True, button_pos=(0.6, 0.0)), policy=policy)
    ctx = Ctx(Interface(w), TruePose(w), w.cfg["layout"])
    r = run(ctx, turn_to(ctx, 1.2))
    assert r["ok"] and abs(r["achieved"]["error"]) <= 0.2
    r = run(ctx, go_to(ctx, (-0.5, 0.8)))
    assert r["ok"] and r["achieved"]["distance"] < 0.15
    lamp0 = w.lamp_on
    r = run(ctx, press_button(ctx, (0.6, 0.0)))
    assert r["ok"] and w.press_count == 1 and w.lamp_on != lamp0


def test_episodes_are_deterministic():
    for task in TASKS:
        assert make_episode(TRAIN[5], task, 3) == make_episode(TRAIN[5], task, 3)
    assert make_episode(TRAIN[5], "set_lamp", 0)["instruction"]["target"] in ("on", "off")


def test_ball_launch_session_is_bit_exact(policy, tmp_path):
    cfg = TRAIN[7]
    ep = make_episode(cfg, "find_ball", 0)
    a = World(cfg, policy=policy)
    start_episode(a, ep)
    session.save(tmp_path / "s.json", a)          # saved right after the launch
    for _ in range(150):
        a.step()
    b, _ = session.start(tmp_path / "s.json", policy=policy)
    for _ in range(150):
        b.step()
    assert np.array_equal(a.duck.data.qpos, b.duck.data.qpos)
    assert np.array_equal(a.duck.data.qvel, b.duck.data.qvel)


@pytest.mark.parametrize("task", TASKS)
def test_oracle_solves_one_episode(policy, task):
    cfg = TRAIN[1]
    w = World(cfg, policy=policy)
    res, _, _ = run_episode(cfg, make_episode(cfg, task, 0), controller_for(w, task), world=w)
    assert res["success"], res


def test_ctx_obs_is_current_under_run_episode(policy):
    """M2 review: skills discard what their yield returns; ctx.obs must still follow the interface's
    tick when run_episode (not skills.run) drives them, or look/scan/stop read tick-0 sensors."""
    cfg = TRAIN[1]
    w = World(cfg, policy=policy)
    seen = {}

    def ctrl(iface, instruction):
        ctx = Ctx(iface, TruePose(w), None)
        seen["rep"] = yield from look(ctx, (0.0, 0.0, 1.0, 0.0))
        seen["obs_tick"], seen["iface_tick"] = ctx.obs.tick, iface.tick_count
        seen["true_head"] = iface.observe().head_joints()
        yield Cmd(done=True)

    run_episode(cfg, make_episode(cfg, "go_and_return", 0), ctrl, world=w)
    assert seen["obs_tick"] == seen["iface_tick"] > 3
    assert np.allclose(seen["rep"]["achieved"]["head"], seen["true_head"], atol=1e-4)
    assert seen["rep"]["ok"]


def test_plan_path_start_in_goal_cell():
    lay = TRAIN[0]["layout"]
    wps = plan_path(lay, lay["home"]["pos"], lay["home"]["pos"])
    assert len(wps) == 1 and np.allclose(wps[0], lay["home"]["pos"], atol=1e-3)


def test_find_ball_not_passed_by_done_at_once(policy):
    """M2 review: 'done' at tick 1 used to pass find_ball while the launched ball was still in view and
    close. Under the current judge no such episode passes; at least one of these episodes would have
    passed the old judge (ball in view and within 0.8 m at done), so the test isn't vacuous."""
    def done_now(iface, instruction):
        yield Cmd(done=True)
    old_would_pass = 0
    for i in range(12):
        for k in (0, 1):
            c = TRAIN[i]
            w = World(c, policy=policy)
            res, _, _ = run_episode(c, make_episode(c, "find_ball", k), done_now, world=w)
            assert not res["success"]
            old_would_pass += bool(res["ball_visible_at_end"] and res["ball_distance"] <= 0.8)
    assert old_would_pass >= 1