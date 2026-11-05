"""Round 8 Gate B (r8_stabcam.py): the extra cameras leave physics and the bot's frames untouched, the camera
conventions and the rotation warp agree with direct renders, the target filter and the stance detector behave.
One config, a few dozen ticks (~15 s)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import r8_stabcam as S  # noqa: E402
from interface import Command, Interface  # noqa: E402
from room import World  # noqa: E402

CFG = S.TRAIN[0]


def _walk(world, n, head=(0.0, 0.15, 0.0, 0.0), frames_at=()):
    it = Interface(world)
    qs, frames = [], {}
    for k in range(n):
        obs = it.tick(Command(vx=0.3, wz=0.2 if k > n // 2 else 0.0, head=head))
        qs.append(world.duck.data.qpos.copy())
        if k in frames_at:
            frames[k] = obs.frame().copy()
    return np.array(qs), frames


@pytest.fixture(scope="module")
def walked():
    w = S.StabWorld(CFG)
    _walk(w, 30)
    rig = S.Rig(w)
    rig.sync()
    yield w, rig
    rig.close()


def test_extra_cameras_change_nothing_physical_or_seen():
    a, b = World(CFG), S.StabWorld(CFG)
    assert b.model.ncam == a.model.ncam + 3 and b.model.nbody == a.model.nbody
    qa, fa = _walk(a, 25, frames_at=(0, 12, 24))
    qb, fb = _walk(b, 25, frames_at=(0, 12, 24))
    assert np.array_equal(qa, qb)                               # bit for bit
    for k in fa:                                                # the bot's own frames (bigger offscreen buffer)
        assert np.array_equal(fa[k], fb[k])


def test_virtual_camera_reproduces_head_render(walked):
    """Same pose, upright raster vs the head camera's sideways raster + rot90: equal up to a rasterisation noise
    floor (~1 grey level: texture filtering differs when the raster is turned), and best aligned at zero shift."""
    _, rig = walked
    hp, Rh = rig.head_pose()
    head, virt = rig.render_head().astype(float), rig.render_pose(hp, Rh).astype(float)
    assert np.abs(head - virt).mean() < 1.5
    _assert_aligned(virt, head)


def _assert_aligned(img, ref):
    """Geometry check: no integer shift of img matches ref better than no shift."""
    M = {(i, j): np.abs(np.roll(img, (i, j), (0, 1))[4:-4, 4:-4] - ref[4:-4, 4:-4]).mean()
         for i in (-1, 0, 1) for j in (-1, 0, 1)}
    assert min(M, key=M.get) == (0, 0)


def test_warp_matches_direct_render(walked):
    w, _ = walked
    rig = S.Rig(w, headlight_diffuse=0.0)                       # the headlight turns with the camera: off here
    rig.sync()
    hp, Rh = rig.head_pose()
    a = S.angles(Rh)
    wide = rig.render_pose(hp, Rh, S.WIDE_PX, wide=True)
    img, valid = S.warp(wide, Rh, Rh)
    assert valid.all() and np.abs(S.grey64(img) - S.grey64(rig.render_pose(hp, Rh))).mean() < 1.0
    for dy, dp in ((0.0, 0.0), (0.3, -0.2)):                   # level targets: the roll changes too
        Rt = S.level_matrix(a[0] + dy, a[1] + dp)
        img, valid = S.warp(wide, Rh, Rt)
        ref = rig.render_pose(hp, Rt)
        assert valid.all()
        assert np.abs(S.grey64(img) - S.grey64(ref)).mean() < 1.5
        _assert_aligned(img.astype(float), ref.astype(float))
    rig.close()


def test_geometry_helpers():
    for y, p in ((0.3, -0.2), (-2.9, 0.4)):
        assert np.allclose(S.angles(S.level_matrix(y, p)), [y, p, 0.0], atol=1e-9)
    R = S.level_matrix(0.0, 0.0)
    y, p, c = S.clamp_target(1.2, 0.3, R)
    assert c and S.axis_angle_deg(S.level_matrix(y, p), R) <= S.CLAMP_DEG + 1e-6
    # every output pixel stays inside the wide render at the clamp bound
    src = np.zeros((S.WIDE_PX, S.WIDE_PX, 3), np.uint8)
    for d in (np.radians(S.CLAMP_DEG), -np.radians(S.CLAMP_DEG)):
        _, valid = S.warp(src, R, S.level_matrix(d / np.sqrt(2), d / np.sqrt(2)) @ np.array(
            [[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]]))
        assert valid.all()


def test_grey64_is_the_4a_pipeline():
    from p2_learner import prep_frame
    f = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    assert np.allclose(S.grey64(f), prep_frame(f, 64).astype(np.float32).mean(-1))


def test_filter_notches_the_stride_and_tracks_slowly():
    t = np.arange(600) * S.SUB_DT
    wob = np.radians(10) * np.sin(2 * np.pi * 1.86 * t)
    f = S.TargetFilter()                                        # the frozen ma2
    out = np.array([f.step(w, 0.5 * w)[0] for w in wob])
    assert np.abs(out[100:]).max() < np.radians(0.3)
    f = S.TargetFilter({"kind": "ab", "wn": 2.0})               # type-2: no lag on a steady turn
    ramp = 0.5 * t
    out = np.array([f.step(r, 0.0)[0] for r in S.wrap(ramp)])
    assert abs(out[-1] - ramp[-1]) < 1e-3


def test_stance_events_once_per_stride():
    w = S.StabWorld(CFG)
    pr = S.Probe(w, "off", "none")
    _walk(w, 60)
    ev = np.array(pr.result()["events"])
    left = ev[ev[:, 1] == 0, 0]
    left = left[left > 100]                                     # after the gait has started
    gaps = np.diff(left)
    assert len(gaps) >= 5 and np.all((gaps >= 22) & (gaps <= 32))
