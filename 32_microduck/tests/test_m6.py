"""M6 checks on the recorder: it can follow an evaluation episode exactly, and a DAgger recording
records the shadow bot's "done" label once (not on every tick after the bot finished)."""
import json
from pathlib import Path

import numpy as np

import interface
import record
from refbot import controller
from room import World
from tasks import make_episode, run_episode

HERE = Path(__file__).resolve().parent.parent
TRAIN = {c["id"]: c for c in json.loads((HERE / "splits" / "train.json").read_text())["configs"][:120]}
APP_DEV = {c["id"]: c for c in json.loads((HERE / "splits" / "appearances.json").read_text())["dev"]}


def test_cold_renderer_recording_follows_run_episode(tmp_path, monkeypatch):
    """The first frame of a new renderer differs by 1-2 grey levels; with cold_renderer the recorder
    leaves it to the bot, as run_episode does, and the duck's path is the same tick for tick.
    appearances-dev-0002 find_ball is one of the rare episodes (1 of 60 dev episodes scanned) where the
    default recorder already drifts within a few ticks, so the check has teeth."""
    cfg, task, T = APP_DEV["appearances-dev-0002"], "find_ball", 40
    monkeypatch.setitem(record.BUDGET_TICKS, task, T)
    r = record.record_episode(cfg, "appearances-dev", task, 0, out_dir=tmp_path, cold_renderer=True)
    rec = np.load(r["path"])["truth"][:, :3]
    warm = np.load(record.record_episode(cfg, "appearances-dev", task, 0, out_dir=tmp_path, tag="warm")["path"])["truth"][:, :3]
    ep = make_episode(cfg, task, 0)
    ep["budget_ticks"] = T
    w = World(cfg)
    fac, _ = controller(task, T)
    P = []

    def wrap(iface, instr):
        g = fac(iface, instr)
        cmd = next(g)
        while True:
            P.append([*w.duck.pos()[:2], w.duck.yaw()])
            obs = yield cmd
            cmd = g.send(obs)
    run_episode(cfg, ep, wrap, world=w)
    assert len(rec) == T and len(P) >= T                   # the wrapper also sees the state after the last tick
    assert np.abs(np.array(P[:T]) - rec).max() < 1e-6      # truth is stored as float32
    assert np.abs(warm - rec).max() > 1e-3                 # the default (M4) recorder does not


def test_dagger_recording_ends_when_shadow_bot_finishes(tmp_path, monkeypatch):
    """A stand-in policy copies the bot's command but never says done. When the bot finishes, the
    episode ends: exactly one 'done' label, on the last tick."""
    held = {}

    def spy_controller(task, budget, params=None):
        fac, bot = controller(task, budget, params)
        held["bot"] = bot
        return fac, bot

    class CopyPolicy:
        def __init__(self, weights):
            pass

        def factory(self, iface, instruction):
            while True:
                c = getattr(held["bot"], "proposed", interface.Command())
                yield interface.Command(vx=c.vx, vy=c.vy, wz=c.wz, head=c.head, done=False)

    import learner
    monkeypatch.setattr(record, "controller", spy_controller)
    monkeypatch.setattr(learner, "PolicyController", CopyPolicy)
    r = record.record_episode(TRAIN["train-0027"], "train", "set_lamp", 0, out_dir=tmp_path,
                              policy_weights="unused", beta=0.0, tag="t")
    z = np.load(r["path"])
    done = z["cmd_bot"][:, 7] > 0.5
    H = json.loads(str(z["header"]))
    assert done.sum() == 1 and done[-1]
    assert H["dagger"]["bot_finished_tick"] == len(done) and not H["result"]["declared_done"]
    assert r["ticks"] < record.BUDGET_TICKS["set_lamp"]


def test_default_recorder_still_reproduces_stored_m4_episode(tmp_path):
    """(REVIEW_M6) cold_renderer is off by default so the M4 demos stay reproducible: re-recording a stored
    M4 episode on the default path gives the same frames, body, commands and truth, bit for bit. The
    episodes are gitignored, so this skips on a fresh checkout."""
    import pytest
    stored = HERE / "episodes" / "set_lamp" / "train_train-0000_k1.npz"
    if not stored.exists():
        pytest.skip("M4 episodes not present")
    r = record.record_episode(TRAIN["train-0000"], "train", "set_lamp", 1, out_dir=tmp_path)
    a, b = np.load(stored), np.load(r["path"])
    for k in ("frames", "body", "cmd_exec", "cmd_bot", "truth", "skill", "activity"):
        assert np.array_equal(a[k], b[k]), k
