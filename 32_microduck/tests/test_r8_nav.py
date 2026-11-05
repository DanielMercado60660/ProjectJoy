"""R8 Gate C navigation track (r8_nav/): MLX ports vs the torch reference on random weights, geometry, sampler
constants, the goal mask, the nav-frame recorder's bit-exactness, and (if downloaded) the published weights' shapes.
torch is used here only as the parity reference (as in r8_nav/parity.py); the runtime path is MLX."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx  # noqa: E402

from r8_nav import frames as F  # noqa: E402
from r8_nav import mlx_models as M  # noqa: E402

torch = pytest.importorskip("torch")


def _sd(module):
    return {k: v.detach().numpy().astype(np.float32) if v.dtype.is_floating_point else v.numpy()
            for k, v in module.state_dict().items()}


def _img(rng, n, c, size):
    w, h = size
    return rng.standard_normal((n, h, w, c)).astype(np.float32)


def _nchw(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def test_static_padding_layout_matches_efficientnet_pytorch():
    stem, blocks = M.b0_layout(224)
    assert stem == (0, 1, 0, 1)
    assert len(blocks) == 16
    pads = {(b["k"], b["s"]): b["pad"] for b in blocks}
    assert pads[(3, 2)] == (0, 1, 0, 1) and pads[(5, 2)] == (1, 2, 1, 2)
    assert pads[(3, 1)] == (1, 1, 1, 1) and pads[(5, 1)] == (2, 2, 2, 2)
    assert [b["skip"] for b in blocks].count(True) == 16 - 7          # only repeated blocks skip


def test_vint_port_matches_torch_reference_random_weights():
    from r8_nav import torch_ref as R
    torch.manual_seed(0)
    ref = R.ViNT().eval()
    with torch.no_grad():                         # non-trivial BN statistics
        for m in ref.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.1, 0.1)
                m.running_var.uniform_(0.5, 1.5)
    net = M.ViNT(sd=_sd(ref))
    rng = np.random.default_rng(0)
    obs = _img(rng, 12, 3, M.ViNT.size).reshape(2, 6, 64, 85, 3)
    goal = _img(rng, 2, 3, M.ViNT.size)
    with torch.no_grad():
        d_ref, a_ref = ref(torch.cat([_nchw(obs[:, i]) for i in range(6)], 1), _nchw(goal))
    d, a = net(mx.array(obs), mx.array(goal))
    np.testing.assert_allclose(np.array(d), d_ref[:, 0].numpy(), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(np.array(a), a_ref.numpy(), atol=1e-4, rtol=1e-3)


@pytest.fixture(scope="module")
def nomad_pair():
    from r8_nav import torch_ref as R
    torch.manual_seed(1)
    ref = R.NoMaD().eval()
    return ref, M.NoMaD(sd=_sd(ref))


def test_nomad_port_matches_torch_reference_random_weights(nomad_pair):
    from r8_nav import torch_ref as R
    ref, net = nomad_pair
    rng = np.random.default_rng(1)
    obs = _img(rng, 8, 3, M.NoMaD.size).reshape(2, 4, 96, 96, 3)
    goal = _img(rng, 2, 3, M.NoMaD.size)
    tok = net.encode_obs(mx.array(obs.reshape(-1, 96, 96, 3))).reshape(2, 4, -1)
    gt = net.encode_goal(mx.array(obs[:, -1]), mx.array(goal))
    obs_t = torch.cat([_nchw(obs[:, i]) for i in range(4)], 1)
    for mask in (0, 1):
        with torch.no_grad():
            c_ref = ref.vision_encoder(obs_t, _nchw(goal), torch.full((2,), mask, dtype=torch.long))
        c = net.context_vector(tok, gt, mx.full((2,), bool(mask)))
        np.testing.assert_allclose(np.array(c), c_ref.numpy(), atol=1e-4, rtol=1e-3)
    # the sampler with shared noise
    noise = rng.standard_normal((11, 2, 8, 2)).astype(np.float32)
    sched = R.DDPM(10)
    x = torch.from_numpy(noise[0])
    with torch.no_grad():
        for k, t in enumerate(sched.timesteps):
            x = sched.step(ref.noise_pred_net(x, t, c_ref), t, x, torch.from_numpy(noise[k + 1]))
    xm = net.sample_actions(c, mx.array(noise))
    np.testing.assert_allclose(np.array(xm), x.numpy(), atol=1e-4, rtol=1e-3)


def test_goal_mask_ignores_goal_and_rescales_mean(nomad_pair):
    net = nomad_pair[1]
    rng = np.random.default_rng(2)
    tok = mx.array(rng.standard_normal((3, 4, 256)).astype(np.float32))
    g1 = mx.array(rng.standard_normal((3, 256)).astype(np.float32))
    g2 = mx.array(rng.standard_normal((3, 256)).astype(np.float32))
    m = mx.full((3,), True)
    a, b = net.context_vector(tok, g1, m), net.context_vector(tok, g2, m)
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6)        # masked goal: content irrelevant
    u = net.context_vector(tok, g1, mx.full((3,), False))
    assert np.abs(np.array(u) - np.array(a)).max() > 1e-3                  # unmasked goal matters


def test_ddpm_schedule_constants():
    betas, ac = M.ddpm_schedule(10)
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    assert abs(betas[0] - (1 - f(0.1) / f(0.0))) < 1e-7
    assert betas.max() <= 0.999 and np.all(np.diff(betas) > 0)
    np.testing.assert_allclose(ac, np.cumprod(1 - betas.astype(np.float64)), rtol=1e-5)
    wp = M.NoMaD.to_waypoints(np.zeros((1, 8, 2), np.float32))             # deltas at the stats' midpoint
    np.testing.assert_allclose(wp[0, :, 0], 1.25 * np.arange(1, 9))
    np.testing.assert_allclose(wp[0, :, 1], 0.0)


def test_crop_resize_and_ground_frame():
    sq = np.zeros((160, 160, 3), np.uint8)
    sq[20:140] = 255
    c = F.crop43(sq)
    assert c.shape == (120, 160, 3) and c.min() == 255
    assert F.resize(c[None], (85, 64)).shape == (1, 64, 85, 3)
    assert F.resize(c[None], (96, 96)).shape == (1, 96, 96, 3)
    x = M.preprocess(F.resize(c[None], (85, 64)), (85, 64))
    np.testing.assert_allclose(x[0, 0, 0], (1 - M.IMAGENET_MEAN) / M.IMAGENET_STD, rtol=1e-6)
    # heading pi/2 (facing +y): a point at +y is straight ahead, a point at -x is to the left
    g = F.to_ground_frame(np.array([[0.0, 1.0], [-1.0, 0.0]]), np.zeros(2), np.pi / 2)
    np.testing.assert_allclose(g, [[1.0, 0.0], [0.0, 1.0]], atol=1e-12)
    assert abs(F.heading_of_axis(np.array([0.0, 1.0, -0.3])) - np.pi / 2) < 1e-12


def test_grid_and_nav_rendering_is_bit_exact():
    from interface import Command, Interface
    from r8_nav.episodes import NavCam, grid_step, train_configs
    from room import World
    assert [grid_step(k) for k in range(6)] == [0, 12, 25, 37, 50, 62]
    cfg = train_configs(1)[0]
    states = []
    for nav in (True, False):
        w = World(cfg)
        iface = Interface(w)
        nc = NavCam(w) if nav else None
        if nc:
            nc.attach()
        for i in range(12):
            iface.tick(Command(vx=0.3, wz=0.2 * (-1) ** (i // 4), head=(0.0, 0.3, 0.1, 0.0)))
        states.append(w.get_state()["physics"])
        if nc:
            a = nc.arrays()
            assert list(a["step"]) == [grid_step(k) for k in range(len(a["step"]))] and len(a["step"]) == 5
            assert a["normal"].shape == (5, 120, 160, 3) and a["wide"].shape == (5, 120, 160, 3)
            assert float(w.model.cam_fovy[w.duck.cam_id]) == 45.0                 # restored after the wide render
            nc.close()
    assert np.array_equal(states[0], states[1])


@pytest.mark.skipif(not (M.VINT_NPZ.exists() and M.NOMAD_NPZ.exists()), reason="published weights not downloaded")
def test_published_weights_load_and_run():
    v = M.ViNT()
    d, a = v(mx.zeros((1, 6, 64, 85, 3)), mx.zeros((1, 64, 85, 3)))
    assert np.isfinite(np.array(d)).all() and np.array(a).shape == (1, 5, 4)
    n = M.NoMaD()
    tok = n.encode_obs(mx.zeros((4, 96, 96, 3))).reshape(1, 4, -1)
    c = n.context_vector(tok, n.encode_goal(mx.zeros((1, 96, 96, 3)), mx.zeros((1, 96, 96, 3))), mx.full((1,), False))
    x = n.sample_actions(c, mx.array(np.random.default_rng(0).standard_normal((11, 1, 8, 2)).astype(np.float32)))
    assert np.array(x).shape == (1, 8, 2) and np.isfinite(np.array(x)).all()


def test_stub_loader_runs_nothing():
    from r8_nav import weights as W
    sg = W._vint_safe_globals()
    names = {n for _, n in sg}
    assert set(W.VINT_GLOBALS) <= names
    cls = dict((n, o) for o, n in sg)["vint_train.models.vint.vint.ViNT"]
    obj = cls.__new__(cls, 1, 2)
    obj.__setstate__({"_parameters": {}, "_modules": {}})
    assert obj._args == (1, 2) and W.walk_state_dict(obj) == {}


def test_metrics_perfect_and_straight_predictors():
    """episode_metrics on a synthetic approach: a perfect predictor scores rho 1 / 0 deg; a straight-ahead
    predictor's plate error equals the straight-ahead control's."""
    from r8_nav import zeroshot as Z
    n = 60
    t = 0.25 * np.arange(n)
    plate = np.array([1.0, 0.5])
    xy = np.stack([np.linspace(-1.5, 0.9, n), np.linspace(-1.0, 0.45, n)], 1)
    axis = np.tile([1.0, 0.0, -0.3], (n, 1))                      # camera heading +x
    e = {"time": t, "trunk_xy": xy, "cam_axis": axis, "head": np.zeros((n, 4)),
         "meta": {"g1_index": n - 1, "plate_xy": plate.tolist()}}
    tr = Z.truth(e)
    q = tr["q"]
    assert q[0] == Z.T_MIN_K and q[-1] == n - 2
    bp = tr["bearing_plate"][q]
    perfect = {"dist": tr["remaining"][q] * 4, "wp": np.stack([np.cos(bp), np.sin(bp)], -1)[:, None, :].repeat(5, 1)}
    m = Z.episode_metrics(tr, perfect)
    assert abs(m["rho"] - 1) < 1e-9 and np.max(m["err_plate"]) < 1e-9 and len(m["err_plate"]) > 0
    straight = {"dist": np.zeros(len(q)), "wp": np.tile([[1.0, 0.0]], (len(q), 5, 1))}
    m2 = Z.episode_metrics(tr, straight)
    np.testing.assert_allclose(m2["err_plate"], m2["err_straight"])
    assert m2["rho"] is None                                     # constant prediction: undefined rank correlation
