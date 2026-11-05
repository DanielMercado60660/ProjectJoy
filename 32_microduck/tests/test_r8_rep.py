"""R8 Gate C (representation) checks: 224 px recorder vs a stored episode, port pieces (bicubic, RoPE tables,
LeWM shapes / checkpoint load), window alignment, official SIGReg, probes, bit-exact trainer resume.
Tests that need downloaded weights (data/pretrained) or R8 recordings (episodes/r8_rep) skip when absent."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "r8_rep"))
import rep_data as D  # noqa: E402
import rep_ports as P  # noqa: E402

TRAIN = json.loads((ROOT / "splits" / "train.json").read_text())["configs"]
HAS_LEWM = (P.PRE / "lewm-tworooms" / "weights.pt").exists() or (P.PRE / "lewm-tworooms" / "state_np.npz").exists()
HAS_R8 = (ROOT / "episodes" / "r8_rep" / "index.jsonl").exists()


def test_recorder_reproduces_stored_m4_episode():
    from rep_record import record_rep_episode
    stored = np.load(ROOT / "episodes" / "set_lamp" / "train_train-0000_k1.npz")
    row = record_rep_episode(TRAIN[0], "set_lamp", 1, noisy=False, keep=False)
    assert row["ticks"] == len(stored["frames"])
    assert np.array_equal(row["_frames96"], stored["frames"])          # extra renderers change nothing
    assert np.array_equal(row["_cmd_exec"], stored["cmd_exec"])
    assert row["_frames224"].shape == (row["ticks"], 224, 224, 3)
    lab = row["_labels"]
    assert all(len(v) == row["ticks"] for v in lab.values())
    assert max(lab["lamp_px"]) > 16                                    # the bot looks at the lamp


def test_bicubic_identity_and_ramp():
    a = np.random.default_rng(0).standard_normal((6, 6, 3)).astype(np.float32)
    assert np.allclose(P.bicubic_resize(a, (6, 6)), a, atol=1e-6)
    const = np.full((37, 37, 2), 0.7, np.float32)
    assert np.allclose(P.bicubic_resize(const, (16, 16), scale_factor=(16.1 / 37, 16.1 / 37)), 0.7, atol=1e-6)
    ramp = np.tile(np.arange(37, dtype=np.float32)[:, None, None], (1, 37, 1))
    r = P.bicubic_resize(ramp, (16, 16), scale_factor=(16.1 / 37, 16.1 / 37))[:, 0, 0]
    # torch's kernel (A = -0.75) is only first-order: a ramp stays monotonic with the right mean slope
    # (exact agreement with torch is checked in rep_parity.py: max err 1.2e-7)
    assert (np.diff(r) > 0).all() and abs(np.diff(r[2:-2]).mean() - 37 / 16.1) < 0.02


def test_vjepa_rope_tables():
    cos, sin, seg = P.vjepa_rope_tables(14, 14, 1, 64)
    assert seg == 20 and cos.shape == (196, 60)
    assert np.allclose(cos[:, :20], 1) and np.allclose(sin[:, :20], 0)  # time position 0 for an image
    x = mx.array(np.random.default_rng(1).standard_normal((1, 1, 196, 60)).astype(np.float32))
    y = P.rope_rotate(x, mx.array(cos), mx.array(sin))
    assert np.allclose(np.linalg.norm(np.array(y), axis=-1), np.linalg.norm(np.array(x), axis=-1), atol=1e-4)


@pytest.mark.skipif(not HAS_LEWM, reason="LeWM checkpoint not downloaded")
def test_lewm_official_shapes_and_new_action_encoder():
    m = P.load_lewm_official("tworooms")
    f = mx.array(np.random.default_rng(0).integers(0, 256, (2, 2, 224, 224, 3), dtype=np.uint8))
    z = m.encode(f)
    assert z.shape == (2, 2, 192)
    m30 = P.load_lewm_official("tworooms", act_dim=30, with_action_encoder=False)
    p = m30.predict(mx.array(np.zeros((2, 3, 192), np.float32)), mx.array(np.zeros((2, 3, 30), np.float32)))
    assert p.shape == (2, 3, 192)
    assert np.allclose(np.array(m.encoder.pos), np.array(m30.encoder.pos))


def test_official_sigreg_gaussian_vs_collapsed():
    from rep_train import sigreg_official
    k = mx.random.key(0)
    g = mx.random.normal((4, 256, 192), key=mx.random.key(1))
    c = mx.zeros((4, 256, 192)) + mx.random.normal((4, 256, 1), key=mx.random.key(2)) * 0.01
    assert float(sigreg_official(g, k)) < 2.0 < float(sigreg_official(c, k))


class _FakeE:
    def __init__(self, lens):
        self.act = [np.repeat(np.arange(n, dtype=np.float32)[:, None] + 1000 * i, 6, 1) for i, n in enumerate(lens)]
        self._f = [np.arange(n, dtype=np.uint8)[:, None, None, None] * np.ones((1, 224, 224, 3), np.uint8) for n in lens]

    def frames(self, i):
        return self._f[i]


def test_windows_alignment():
    E = _FakeE([30, 10, 40])
    w = D.Windows(E, T=4, fs=5)
    assert len(w) == (30 - 15) + (40 - 15)                             # the 10-tick episode is too short
    F, A, e = w.get([0, 16])
    assert list(F[0, :, 0, 0, 0]) == [0, 5, 10, 15] and e[0] == 0
    assert A[0, 1].reshape(5, 6)[:, 0].tolist() == [5, 6, 7, 8, 9]     # block 1 = ticks 5..9
    assert e[1] == 2 and F[1, 0, 0, 0, 0] == 1                         # second window: episode 2, start tick 1


def test_linear_and_attentive_probes_separable():
    from rep_probes import attentive_probe, linear_probe
    r = np.random.default_rng(0)
    y = (r.random(400) > 0.5).astype(np.float32)
    X = r.standard_normal((400, 8)).astype(np.float32)
    X[:, 0] += 3 * (2 * y - 1)
    tr = np.arange(400) < 300
    assert linear_probe(X, y, tr)["test_acc"] > 0.95
    G = r.standard_normal((400, 5, 8)).astype(np.float32)
    G[np.arange(400), r.integers(0, 5, 400), 0] += 4 * (2 * y - 1)    # signal in one random token
    assert attentive_probe(G, y, tr)["test_acc"] > 0.85
    t = X[:, 1] * 0.5
    assert linear_probe(X, t, tr, regress=True)["test_mae_deg"] < 3.0


@pytest.mark.skipif(not (HAS_LEWM and HAS_R8), reason="needs the LeWM checkpoint and R8 recordings")
def test_trainer_resume_bit_exact(tmp_path, monkeypatch):
    import rep_train
    monkeypatch.setattr(rep_train, "ROOT", tmp_path)
    args = ["--init", "ckpt:tworooms", "--steps", "50", "--batch", "2", "--max-episodes", "3", "--val-every", "0",
            "--ckpt-every", "1000", "--guard-min", "10"]
    rep_train.main(["--run", "a"] + args + ["--stop-at", "4"])
    rep_train.main(["--run", "b"] + args + ["--stop-at", "2"])
    rep_train.main(["--run", "b", "--resume", "--stop-at", "4", "--val-every", "0", "--ckpt-every", "1000"])
    wa, wb = mx.load(str(tmp_path / "a/ckpt/weights.safetensors")), mx.load(str(tmp_path / "b/ckpt/weights.safetensors"))
    assert max(float(mx.abs(wa[k] - wb[k]).max()) for k in wa) == 0.0
