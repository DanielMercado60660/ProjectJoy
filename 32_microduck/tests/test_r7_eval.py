"""Round 7 evaluator checks: the instrumented runner reproduces tasks.run_episode's judge row exactly (the
instrument only reads frames the policy already rendered), and the intentionality fields follow the protocol."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p2_learner import Policy2, cmd_targets  # noqa: E402
from r7_eval import instrument, run_instrumented  # noqa: E402

TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]


def test_instrumented_runner_matches_run_episode(tmp_path):
    from p2_learner import P2Controller
    from room import World
    from tasks import make_episode, run_episode
    cfg = TRAIN[3]
    mx.random.seed(4)
    model = Policy2(act="leaky", vx_head="binary", turn_head="discrete")
    model.done.weight = mx.zeros_like(model.done.weight)
    model.done.bias = mx.array([-20.])
    weights = tmp_path / "weights.safetensors"
    model.save_weights(str(weights))
    (tmp_path / "p2_config.json").write_text(json.dumps({"img": 64, "aux": True, "act": "leaky", "vx_head": "binary", "turn_head": "discrete"}))
    ep = make_episode(cfg, "set_lamp", 0)
    ep["budget_ticks"] = 60
    device = mx.default_device()
    try:
        ref, _, iface = run_episode(cfg, ep, P2Controller(weights).factory, world=World(cfg))
        if iface._renderer is not None:
            iface._renderer.close()
        row = run_instrumented(cfg, ep, "test", str(weights), 400, 2.5)
    finally:
        mx.set_default_device(device)
    for k in ("ticks", "declared_done", "presses", "plate_visits", "lamp_final", "collision_ticks", "falls", "success", "stopped"):
        assert row[k] == ref[k], k
    assert row["lamp_check_tick"] is None or row["lamp_check_tick"] >= 1
    assert row["min_dist_plate_m"] > 0 and isinstance(row["within_0_5m"], bool)
    assert "intentional_reach" in row and "turn_onset_toward" in row and "verified_done" in row


def _logs(T, cmds, xy, presses, lamp, btn, pose):
    return (cmds, xy, [0.0] * T, presses, lamp, btn, pose)


def test_instrument_fields():
    bpos, half = np.array([1.0, 0.0]), 0.12
    T = 12
    # walk straight from the origin to the plate: lamp confirmed at tick 2, entry at tick 8, press at 9, done at 11 after a fresh sighting
    xy = [(0.125 * t, 0.0) for t in range(T)]
    cmds = [(0.3, 0.0, False)] * 10 + [(0.0, 0.0, True)]
    presses = [0] * 9 + [1] * 3
    lamp = [None, None] + [("on", 2)] * 8 + [("off", 10)] * 2
    btn = [(0.0, 1.0 - 0.125 * t, 1) for t in range(T)]
    pose = [(0.125 * t, 0.0, 0.0) for t in range(T)]
    out = instrument(cmds, xy, [0.0] * T, presses, lamp, btn, pose, bpos, half, 400, 2.5)
    assert out["lamp_check_tick"] == 2 and out["first_entry_tick"] == 8 and out["press_ticks"] == [9] and out["done_tick"] == 10
    assert out["intentional_reach"] is True and out["reach_delay_ticks"] == 6 and np.isclose(out["reach_path_ratio"], 1.0)
    assert out["verified_done"] is True and out["done_after_press"] and out["turn_onset_toward"] is None   # never turned
    # a wandering path (ratio > R) after the check is not intentional; a done with no fresh sighting is unverified
    pose2 = [(0.125 * t, (0.8 if t % 2 else -0.8), 0.0) for t in range(T)]
    lamp2 = [None, None] + [("on", 2)] * 10
    out2 = instrument(cmds, xy, [0.0] * T, presses, lamp2, btn, pose2, bpos, half, 400, 2.5)
    assert out2["intentional_reach"] is False and out2["verified_done"] is False
    # entry long after the check (> K) is not intentional either
    out3 = instrument(cmds, xy, [0.0] * T, presses, lamp, btn, pose, bpos, half, 3, 2.5)
    assert out3["intentional_reach"] is False
    # turn onset: a 6-tick standing turn toward the remembered plate (bearing +60 deg -> left, wz > 0)
    cmds4 = [(0.0, 0.0, False)] * 3 + [(0.0, 0.55, False)] * 6 + [(0.3, 0.0, False)] * 3
    btn4 = [(np.radians(60), 1.0, 1)] * T
    xy4 = [(0.0, 0.0)] * T
    out4 = instrument(cmds4, xy4, [0.0] * T, [0] * T, lamp, btn4, pose, bpos, half, 400, 2.5)
    assert out4["turn_onset_tick"] == 3 and out4["turn_onset_toward"] is True and out4["first_entry_tick"] is None
    cmds5 = [(0.0, 0.0, False)] * 3 + [(0.0, -0.55, False)] * 6 + [(0.3, 0.0, False)] * 3
    assert instrument(cmds5, xy4, [0.0] * T, [0] * T, lamp, btn4, pose, bpos, half, 400, 2.5)["turn_onset_toward"] is False
    cmds6 = [(0.0, 0.0, False)] * 3 + [(0.0, 0.55, False)] * 4 + [(0.0, 0.0, False)] * 5     # too short a run
    assert instrument(cmds6, xy4, [0.0] * T, [0] * T, lamp, btn4, pose, bpos, half, 400, 2.5)["turn_onset_toward"] is None


def test_stage_gate_reader():
    from r7_gate import stage_gate
    sel = {"run": "x", "chosen": "ckpt_016000",
           "table": {"ckpt_016000": {"press_needed": [4, 18], "no_press": [17, 22],
                                     "r7": {"reached_plate": 6, "within_0_5m": 9, "pressed": 8, "done_after_press_lamp_right": 4,
                                            "said_done_without_press": 2, "intentional_reach": 2, "verified_done": 4}}}}
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sel, f)
    g = stage_gate(f.name)
    assert g["pass"] and all(g["components"].values())
    sel["table"]["ckpt_016000"]["r7"]["reached_plate"] = 5
    with open(f.name, "w") as fh:
        json.dump(sel, fh)
    g = stage_gate(f.name)
    assert not g["pass"] and not g["components"]["reached_ge_6"] and g["components"]["within_0_5m_ge_9"]
