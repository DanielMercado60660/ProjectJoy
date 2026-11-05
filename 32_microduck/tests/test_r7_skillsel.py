"""Round 7 Stage 3 checks: macro-skill segmentation of a bot recording, the feature layout, the program rule, the
frequency control, and the adapter driving a short episode through the evaluator."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import p2_skillsel as ss  # noqa: E402
from belief import DIM, OFF_FLAGS, OFF_LAMP  # noqa: E402
from p2_skillsel import APPROACH, DONE, EXPLORE, FEAT, LOOK, PRESS, SCAN, SKILLS, features, macro_segments, program_choice  # noqa: E402

TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]


def test_macro_segments_on_a_stored_press_demo():
    p = ROOT / "episodes" / "set_lamp" / "train_train-0001_k1.npz"
    if not p.exists():
        pytest.skip("M4 episodes not present")
    z = np.load(p, allow_pickle=False)
    H = json.loads(str(z["header"]))
    segs = macro_segments(H, z)
    labels = [SKILLS[s[2]] for s in segs]
    assert labels == ["scan", "approach", "press", "look_lamp", "done"]
    assert segs[0][0] == 0 and segs[1][0] == 105 and segs[2][0] == 235 and segs[-1][0] == 425
    assert segs[2][3] is True                                   # press_button reported ok
    dec = ss.decisions_from_recording(p)
    assert [d["y"] for d in dec] == [SCAN, APPROACH, PRESS, LOOK, DONE]
    assert len(dec[0]["x"]) == FEAT and dec[1]["prev"] == SCAN


def test_features_and_program_rule():
    v = np.zeros(DIM, np.float32)
    v[OFF_LAMP] = 1.0                                           # lamp unknown
    f = features(v, "on", None, False, 0)
    assert f.shape == (FEAT,) and f[DIM] == 1 and f[DIM + 8] == 1 and f[DIM + 11] == 1
    assert program_choice(f) == SCAN
    v[OFF_LAMP] = 0.0
    v[OFF_LAMP + 1] = 1.0                                       # lamp on
    assert program_choice(features(v, "on", SCAN, True, 110)) == DONE          # matches after the scan
    f = features(v, "off", SCAN, True, 110)                                     # mismatch, no button known
    assert f[DIM + 13] == 1 and program_choice(f) == EXPLORE
    v[5] = 1.0                                                                  # button present
    assert program_choice(features(v, "off", SCAN, True, 110)) == APPROACH
    assert program_choice(features(v, "off", APPROACH, True, 200)) == PRESS
    assert program_choice(features(v, "off", PRESS, True, 300)) == LOOK
    v[OFF_FLAGS] = 1.0                                                          # crossed, lamp not re-seen
    assert program_choice(features(v, "off", LOOK, False, 400)) == DONE
    v[OFF_FLAGS] = 0.0
    v[OFF_LAMP + 1], v[OFF_LAMP + 2] = 0.0, 1.0                                 # re-seen off: matches the target
    assert program_choice(features(v, "off", LOOK, True, 400)) == DONE
    v[OFF_LAMP + 1], v[OFF_LAMP + 2] = 1.0, 0.0                                 # still on: another try
    assert program_choice(features(v, "off", LOOK, True, 400)) == APPROACH


def test_freq_control_and_mlp_roundtrip(tmp_path, monkeypatch):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(60):
        x = rng.normal(size=FEAT).astype(np.float32)
        prev = None if i % 2 else SCAN
        x[DIM + 2:DIM + 9] = 0
        x[DIM + 2 + (6 if prev is None else prev)] = 1
        x[DIM + 11:DIM + 14] = 0
        x[DIM + 11 + (i % 3)] = 1
        rows.append({"x": x.tolist(), "y": i % 6, "prev": prev})
    dec = tmp_path / "decisions.json"
    dec.write_text(json.dumps({"rows": rows}))
    monkeypatch.setattr(ss, "OUT", tmp_path)
    table = ss.freq_table(dec)
    assert set(table) == {"start|0", "start|1", "start|2", "0|0", "0|1", "0|2"}
    chooser = ss.load_chooser(f"freq:{tmp_path / 'freq.json'}")
    x = np.array(rows[0]["x"], np.float32)
    assert chooser(x) == table[f"{'start' if rows[0]['prev'] is None else rows[0]['prev']}|{int(np.argmax(x[DIM + 11:DIM + 14]))}"]
    rep = ss.train("learned", 0, decisions=dec, steps=200)
    assert rep["train_accuracy"] > 0.5
    m = ss.MLP.load(tmp_path / "learned_s0.npz")
    X = np.array([r["x"] for r in rows], np.float32)
    assert np.array_equal(m.predict(X), ss.MLP.load(tmp_path / "learned_s0.safetensors").predict(X))


def test_adapter_runs_through_the_evaluator():
    from r7_eval import run_instrumented
    from tasks import make_episode
    cfg = TRAIN[5]
    ep = make_episode(cfg, "set_lamp", 0)
    ep["budget_ticks"] = 130
    row = run_instrumented(cfg, ep, "sel_program", "selector:program", 400, 3.1)
    assert row["ticks"] <= 130 and row["falls"] == 0 and "intentional_reach" in row


def test_adapter_never_hangs_on_a_zero_tick_choice():
    """Amendment 3: a chooser that keeps picking look_lamp with no remembered lamp consumes no tick per decision; the
    adapter must still reach the budget (the shuffled-label control hung on the fresh slice before the guard)."""
    from interface import Interface
    from room import World
    from tasks import make_episode, start_episode
    from p2_skillsel import SelectorBot
    cfg = TRAIN[6]
    ep = make_episode(cfg, "set_lamp", 0)
    w = World(cfg)
    start_episode(w, ep)
    iface = Interface(w)
    sel = SelectorBot(lambda f: LOOK)
    gen = sel.run(iface, ep["instruction"], 40)
    cmd = next(gen)
    n = 0
    try:
        while not iface.done and iface.tick_count < 40 and n < 60:
            obs = iface.tick(cmd)
            cmd = gen.send(obs)
            n += 1
    except StopIteration:
        pass
    finally:
        if iface._renderer is not None:
            iface._renderer.close()
    assert iface.tick_count >= 38 and len(sel.decisions) <= 3 * 40 + 3
