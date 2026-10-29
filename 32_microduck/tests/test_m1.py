"""M1 checks: repeatable episodes, appearance independence, sessions, button/lamp logic, split files."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import session  # noqa: E402
from configs import hand_config  # noqa: E402
from policy.velstand_np import VelstandPolicy  # noqa: E402
from room import World  # noqa: E402

SPLITS = ROOT / "splits"


@pytest.fixture(scope="module")
def policy():
    return VelstandPolicy()


@pytest.fixture(scope="module")
def splits():
    train = json.loads((SPLITS / "train.json").read_text())["configs"]
    app = json.loads((SPLITS / "appearances.json").read_text())
    dist = json.loads((SPLITS / "distractor.json").read_text())
    return train, app, dist


# A fixed command script with walking, turning, head moves and a stop.
SCRIPT = ([((0.3, 0, 0), (0, 0, 0, 0))] * 120 + [((0, 0, 0.6), (0, 0.3, 0.5, 0))] * 60
          + [((0.25, 0, -0.4), (0, 0, 0, 0))] * 80 + [((0, 0, 0), (0.2, 0, -0.4, 0))] * 40)


def run(world, script=SCRIPT):
    traj = []
    for twist, head in script:
        world.duck.set_command(*twist, head=head)
        world.step()
        traj.append(np.concatenate([world.duck.data.qpos, world.duck.data.qvel, [world.lamp_on]]))
    return np.array(traj)


def test_compile_is_deterministic_and_replay_is_exact(policy, splits):
    cfg = splits[0][0]
    a = run(World(cfg, policy=policy))
    b = run(World(cfg, policy=policy))      # fresh build from the same config
    assert np.array_equal(a, b)


def test_reset_is_bit_exact(policy, splits):
    w = World(splits[0][1], policy=policy)
    first = run(w)
    w.reset()
    second = run(w)
    assert np.array_equal(first, second)
    assert w.get_state()["room"]["step_count"] == len(SCRIPT)


def test_appearance_and_distractor_do_not_change_physics(policy, splits):
    train, app, dist = splits
    by_id = {c["id"]: c for c in train}
    for sib in (app["test"][0], app["dev"][0], dist["test"][0]):
        base = by_id[sib["source_train_id"]]
        assert sib["appearance"] != base["appearance"] or sib["distractor"] is not None
        assert np.array_equal(run(World(base, policy=policy)), run(World(sib, policy=policy)))


def test_session_resume_is_bit_exact(policy, splits, tmp_path):
    cfg = splits[0][2]
    cont = World(cfg, policy=policy)
    full = run(cont)
    w, mem = session.new(tmp_path / "life.json", cfg, memory={"seen": []}, policy=policy)
    run(w, SCRIPT[:150])
    session.save(tmp_path / "life.json", w, memory={"seen": ["red"], "steps": 150})
    w2, mem2 = session.start(tmp_path / "life.json", policy=policy)
    assert mem2 == {"seen": ["red"], "steps": 150} and w2.step_count == 150
    rest = run(w2, SCRIPT[150:])
    assert np.array_equal(full[150:], rest)
    assert np.array_equal(cont.duck.data.ctrl, w2.duck.data.ctrl)


def test_button_press_toggles_lamp_once_per_visit(policy):
    cfg = hand_config(button_pos=(0.6, 0.0), empty=True)
    w = World(cfg, policy=policy)
    assert not w.lamp_on
    lamp_off = w.model.geom_rgba[w.lamp_geom].copy()
    from m0_gate import goto_command
    goal = np.array([1.1, 0.0])          # straight over the plate and 0.5 m beyond (true-pose go-to)
    for _ in range(int(20 / 0.02)):
        if np.linalg.norm(w.duck.pos()[:2] - goal) < 0.1:
            break
        w.duck.set_command(*goto_command(w.duck, goal))
        w.step()
    w.duck.set_command(0, 0, 0)
    for _ in range(100):
        w.step()
    assert w.press_count == 1 and w.lamp_on
    assert not np.array_equal(w.model.geom_rgba[w.lamp_geom], lamp_off)
    w.reset()                            # reset restores the lamp's look too
    assert not w.lamp_on and np.array_equal(w.model.geom_rgba[w.lamp_geom], lamp_off)


def test_version_pin_and_state_guard(policy, splits):
    cfg = dict(splits[0][0])
    cfg["mujoco_version"] = "9.9.9"
    with pytest.raises(RuntimeError):
        World(cfg, policy=policy)
    a, b = World(splits[0][0], policy=policy), World(splits[0][1], policy=policy)
    with pytest.raises(ValueError):
        a.set_state(b.get_state())
    from room import FROZEN_BODY   # review M1: a state from another body must not load silently
    c = World(splits[0][0], body=dict(FROZEN_BODY, phys_delay=2), policy=policy)
    with pytest.raises(ValueError):
        a.set_state(c.get_state())


def test_validate_rejects_occluder_on_partition(splits):
    """Review M1: validate() also checks obstacle vs obstacle (44/640 layouts had them interpenetrating)."""
    import copy
    from configs import validate
    cfg = copy.deepcopy(splits[0][0])
    assert validate(cfg) == []
    p = cfg["layout"]["partition"]
    cfg["layout"]["occluder"].update(pos=list(p["pos"]), yaw=p["yaw"])
    assert any("partition" in prob for prob in validate(cfg))


def test_split_files_pass_their_checks():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "check_splits.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
