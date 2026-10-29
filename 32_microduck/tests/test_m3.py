"""M3 checks: the bot's sensors-only contract, its camera model against truth, memory round-trips,
and one reference-bot episode per task on a training config."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bodymodel import BodyModel, rot_z, tilt_rotation  # noqa: E402
from interface import Command, Interface  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, TASKS, make_episode, run_episode  # noqa: E402

BOT_FILES = ("refbot.py", "bodymodel.py", "odometry.py", "perception.py", "botmap.py", "gaze.py")
FORBIDDEN_IMPORTS = {"room", "tasks", "configs", "oracle", "session", "make_splits", "check_splits"}
FORBIDDEN_TEXT = (".world", "_iface", "cfg[", "layout", ".lamp_on", "press_count", "ball_pos", "duck.data")
TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]


def test_bot_contract():
    """results/m3/bot_contract.md: the bot's modules can't reach the world, a config or the judge."""
    for f in BOT_FILES:
        src = (ROOT / f).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert not FORBIDDEN_IMPORTS & set(names), (f, names)
        code = "\n".join(line.split("#")[0] for line in src.splitlines())   # comments may mention them
        for tok in FORBIDDEN_TEXT:
            assert tok not in code, (f, tok)


def test_bot_camera_model_matches_truth():
    w = World(TRAIN[0])
    it = Interface(w)
    bm = BodyModel()
    for k in range(40):
        obs = it.tick(Command(vx=0.3 if k < 20 else 0.0, head=(0.0, 0.3, 0.6 if k % 10 < 5 else -0.4, 0.0)))
    feet, cp, cm = bm.fk(obs.body["joint_pos"])
    R = rot_z(w.duck.yaw()) @ tilt_rotation(obs.body["gravity"])
    cam = w.duck.data.xpos[w.duck.trunk_id] + R @ cp
    assert np.linalg.norm(cam - w.duck.data.cam_xpos[w.duck.cam_id]) < 0.005
    Mt = w.duck.data.cam_xmat[w.duck.cam_id].reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip((np.trace((R @ cm).T @ Mt) - 1) / 2, -1, 1)))
    assert ang < 3.0


@pytest.mark.parametrize("task,idx", [("go_and_return", 0), ("set_lamp", 1), ("find_ball", 1)])
def test_bot_solves_a_training_episode(task, idx):
    cfg = TRAIN[idx]
    w = World(cfg)
    fac, bot = controller(task, BUDGET_TICKS[task])
    res, _, _ = run_episode(cfg, make_episode(cfg, task, 0), fac, world=w)
    assert res["success"], (res, bot.log)
    json.dumps(bot.mem.to_dict())          # memory is plain JSON
