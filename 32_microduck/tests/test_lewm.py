"""R6 LeWM S1 checks: noisy-bot hook, window/action alignment, model invariants, SIGReg, bit-exact resume."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lewm"))
import data  # noqa: E402
from model import LeWM, sigreg  # noqa: E402
from record import record_episode  # noqa: E402

TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]


def test_noise_hook_off_by_default_and_marks_exec_only(tmp_path):
    r0 = record_episode(TRAIN[0], "train", "go_and_return", 77, out_dir=tmp_path / "a", budget_ticks=40)
    z0 = np.load(ROOT / r0["path"] if not Path(r0["path"]).is_absolute() else r0["path"])
    assert "noise" not in json.loads(str(z0["header"]))
    noise = {"p": 0.5, "len": [3, 4], "head_frac": 0.6}
    r1 = record_episode(TRAIN[0], "train", "go_and_return", 77, out_dir=tmp_path / "b", budget_ticks=40, noise=noise)
    z1 = np.load(r1["path"])
    h = json.loads(str(z1["header"]))
    diff = np.abs(z1["cmd_exec"] - z1["cmd_bot"]).max(1) > 1e-6
    assert h["noise"]["noisy_ticks"] >= 10
    assert diff.sum() <= h["noise"]["noisy_ticks"]          # only burst ticks differ from the bot
    assert diff.sum() >= 0.8 * h["noise"]["noisy_ticks"]
    assert (z1["cmd_exec"][:, 1] == 0).all()                 # vy stays 0
    assert np.abs(z1["cmd_exec"][:, 2]).max() <= 0.6 + 1e-6


class _FakePart:
    def __init__(self, lens, name="fake"):
        n = sum(lens)
        self.frames = np.arange(n, dtype=np.uint8)[:, None, None, None] * np.ones((1, 64, 64, 3), np.uint8)
        self.act = np.repeat(np.arange(n, dtype=np.float32)[:, None], 6, 1)
        self.ep_len = np.array(lens)
        self.ep_start = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.ep_cfg = np.arange(len(lens))
        self.name = name


def test_windows_align_frames_and_action_blocks():
    p = _FakePart([30, 12, 50])
    w = data.Windows([p], T=4, fs=5)
    assert len(w) == (30 - 15) + 0 + (50 - 15)               # the 12-tick episode has no window
    F, A, k = w.get(np.arange(len(w)))
    for b in range(len(w)):
        s = int(F[b, 0, 0, 0, 0])
        assert list(F[b, :, 0, 0, 0]) == [s, s + 5, s + 10, s + 15]
        for i in range(3):                                   # block i = ticks s+5i .. s+5i+4
            assert list(A[b, i, ::6]) == [s + 5 * i + j for j in range(5)]


def test_model_actions_inert_at_init_and_shapes():
    mx.random.seed(0)
    m = LeWM(act_dim=30)
    m.eval()
    fr = mx.array(np.random.default_rng(0).integers(0, 255, (2, 4, 64, 64, 3), dtype=np.uint8))
    z = m.encode_frames(fr)
    assert z.shape == (2, 4, 192)
    a1 = mx.zeros((2, 4, 30))
    a2 = mx.ones((2, 4, 30))
    p1, p2 = m.predictor(z, a1), m.predictor(z, a2)
    assert p1.shape == (2, 4, 192)
    assert float(mx.abs(p1 - p2).max()) == 0.0               # AdaLN-zero: no action effect before training
    # causal: changing the last frame cannot change earlier predictions
    z2 = mx.concatenate([z[:, :3], z[:, 3:] + 1.0], axis=1)
    assert float(mx.abs(m.predictor(z2, a1)[:, :3] - p1[:, :3]).max()) < 1e-5


def test_sigreg_separates_gaussian_from_collapsed():
    key = mx.random.key(0)
    g = mx.random.normal((256, 192), key=mx.random.key(1))
    c = mx.ones((256, 192)) * 0.3
    s_g, s_c = float(sigreg(g, key)), float(sigreg(c, key))
    assert s_g < 2.0 and s_c > 20 * s_g        # null: O(1) after the n scaling


import pytest  # noqa: E402


@pytest.mark.parametrize("extra", [[], ["--random-encoder"]])
def test_train_resume_is_bit_exact(tmp_path, monkeypatch, extra):
    # a tiny fake cache part
    d = tmp_path / "lewm_tiny"
    d.mkdir()
    rng = np.random.default_rng(0)
    lens = [40, 35, 50]
    n = sum(lens)
    rng.integers(0, 255, (n, 64, 64, 3), dtype=np.uint8).tofile(d / "frames.u8")
    np.savez(d / "meta.npz", act=rng.normal(size=(n, 6)).astype(np.float32), yaw=np.zeros(n, np.float32),
             ep_start=np.concatenate([[0], np.cumsum(lens)[:-1]]), ep_len=np.array(lens), ep_cfg=np.array([1, 2, 3]),
             ep_src=np.zeros(3, np.int16), ep_split=np.zeros(3, np.int8), src_names=np.array(["x"]))
    monkeypatch.setattr(data, "CACHE", tmp_path)
    import train
    common = ["--steps", "4", "--batch", "2", "--parts", "tiny", "--warmup", "2", "--ckpt-every", "100"] + extra
    assert train.main(["--run", str(tmp_path / "a")] + common) == 0
    assert train.main(["--run", str(tmp_path / "b"), "--stop-at", "2"] + common) == 0
    assert train.main(["--run", str(tmp_path / "b"), "--resume"]) == 0
    a = mx.load(str(tmp_path / "a" / "final" / "weights.safetensors"))
    b = mx.load(str(tmp_path / "b" / "final" / "weights.safetensors"))
    assert max(float(mx.abs(a[k] - b[k]).max()) for k in a) == 0.0
    if extra:   # control: the encoder stayed at its seeded init, and its BN statistics came from the warm-up
        mx.random.seed(0)
        init = dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(LeWM(act_dim=30).parameters()))
        enc = [k for k in a if k.startswith("encoder.") and "running" not in k]
        assert max(float(mx.abs(a[k] - init[k]).max()) for k in enc) == 0.0
        rm = a["encoder.proj.layers.1.running_mean"]
        assert float(mx.abs(rm).max()) > 0.0
