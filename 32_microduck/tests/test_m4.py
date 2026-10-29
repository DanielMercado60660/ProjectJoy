"""M4 checks: the episode format and the perturbation plumbing."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import session  # noqa: E402
from record import record_episode  # noqa: E402
from room import World  # noqa: E402
from tasks import make_episode  # noqa: E402

TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]


def _no_press_episode():
    for i, c in enumerate(TRAIN):
        ep = make_episode(c, "set_lamp", 1)
        if (ep["instruction"]["target"] == "on") == bool(c["layout"]["lamp_initial_on"]):
            return c
    raise AssertionError


def test_episode_format(tmp_path):
    cfg = _no_press_episode()
    r = record_episode(cfg, "train", "set_lamp", 1, out_dir=tmp_path)
    z = np.load(r["path"])
    H = json.loads(str(z["header"]))
    T = H["ticks"]
    assert z["frames"].shape == (T, 96, 96, 3) and z["frames"].dtype == np.uint8
    assert z["body"].shape == (T, 34) and z["imu_sub"].shape == (T, 5, 6)
    assert z["cmd_exec"].shape == (T, 8) and np.array_equal(z["cmd_exec"], z["cmd_bot"])
    assert len(z["observed"]) == T and len(json.loads(str(z["memory"]))) == T
    assert z["truth"].shape == (T, 7) and z["obs_age"].shape == (T,)
    assert z["cmd_exec"][-1, 7] == 1.0                     # the episode ends with the done bit
    for k in ("config_id", "split", "task", "instruction", "controller", "perturbation", "result", "events",
              "skill_log", "skill_vocab", "activity_vocab", "expected_by_skill"):
        assert k in H
    assert H["split"] == "train" and r["success"] == H["result"]["success"]
    # observed labels line up with the skill column: each finished skill's report covers start..end-1
    names = np.array(H["skill_vocab"])[z["skill"]]
    for e in H["skill_log"]:
        if e["end"] > e["start"] and e["start"] < T:
            assert str(z["observed"][e["start"]]).startswith(e["skill"] + ":"), e
            assert names[e["start"]] == e["skill"], e


def test_missed_press_flag_is_state(tmp_path):
    w = World(TRAIN[0])
    w.suppress_presses = 1
    session.save(tmp_path / "s.json", w)
    w2, _ = session.start(tmp_path / "s.json")
    assert w2.suppress_presses == 1
    w2.reset()
    assert w2.suppress_presses == 0
