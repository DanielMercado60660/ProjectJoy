"""R6 tracker smoke tests (tiny, ~20 s). Run from the project folder via ./smoke.sh."""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); os.chdir(HERE)
import scene as S
import scene_r6 as R
import track_classical_r6 as C
import eval_tracker_r6 as E

PY = sys.executable


def test_dr_scene_and_template_cue():
    fr, gt, vis = R.render_dr(R.DR_SEED["dr"], n_frames=6)
    assert fr.shape == (6, 80, 80, 3) and fr.dtype == np.uint8 and gt.shape == (6, 2)
    # the NCC duck template peaks on the duck in a clean training-style scene
    fr, gt, vis = S.render_scene(123, "train", n_frames=8)
    tpl = C.ncc_map(fr.astype(np.float32) / 255)
    pk = np.array([np.unravel_index(t.argmax(), t.shape)[::-1] for t in tpl])
    assert np.median(np.linalg.norm(pk - gt, axis=1)[vis]) < 2.0


def test_classical_r6_tracks_and_filter_switch():
    fr, gt, vis = S.render_scene(123, "train", n_frames=12)
    prm = dict(sigma=0.1, min_lum=0.0, fg_scale=0.1, a_col=1.0, a_fg=1.0, a_tpl=1.0, t0=0.2, thr=0.05, amin=1,
               amax=60, q=1.0, r=1.0, gate=6, K=3, norm="gw", reinit=3, use_kf=True, switch=3.0)
    est = C.track_scene(C.precompute(fr), prm)
    assert est.shape == (12, 2) and np.isfinite(est).all()
    # re-detection: a locked-on weak track jumps to a much stronger far candidate
    cands = [[(10, 10, 1.0)]] * 3 + [[(11, 10, 0.1), (60, 60, 5.0)]]
    out = C.run_filter_r6(cands, dict(q=1, r=1, gate=4, reinit=3, use_kf=True, switch=3.0))
    assert np.allclose(out[-1], (60, 60))
    out = C.run_filter_r6(cands, dict(q=1, r=1, gate=4, reinit=3, use_kf=True, switch=float("inf")))
    assert np.linalg.norm(out[-1] - (11, 10)) < 2


def test_cnn_r6_tiny(tmp_path):
    out = subprocess.run([PY, "train_tracker_r6.py", "--scenes", "8", "--val_scenes", "2", "--steps", "40", "--c", "16",
                          "--eval_every", "20", "--no_cache", "--out", str(tmp_path / "d"), "--log", str(tmp_path / "d.jsonl")],
                         capture_output=True, text=True, timeout=90)
    assert out.returncode == 0, out.stderr
    L = [json.loads(l) for l in open(tmp_path / "d.jsonl")]
    assert len(L) == 2 and np.isfinite(L[-1]["loss"])
    import train_tracker_r6 as T6
    m = T6.load(str(tmp_path / "d"))
    fr, gt, vis = R.render_dr(5, n_frames=5)
    xy, conf = T6.predict(m, fr)
    assert xy.shape == (5, 2) and len(T6.peaks(m, fr)) == 5


def test_bootstrap_ci_and_fresh_gate():
    err = np.r_[np.zeros((5, 10)), np.full((5, 10), 9.0)]
    r = E.boot_ci(err, np.ones_like(err, bool), n=200)
    assert r["lost_rate"] == 0.5 and r["lost_ci"][0] < 0.5 < r["lost_ci"][1]
    assert set(R.FRESH_CONDS) >= {"bg2", "light2", "distract2", "all2"}


def test_round2_dr_haze_is_dr_plus_haze():
    """DR v2 = the DR v1 scene of the same seed (same duck track) plus an optional global haze."""
    seeds = range(100000, 100006)
    hazed = 0
    for s in seeds:
        f1, g1, _ = R.render_dr(s, n_frames=4); f2, g2, _ = R.render_dr2(s, n_frames=4)
        assert np.allclose(g1, g2)
        hazed += not np.array_equal(f1, f2)
    assert 0 < hazed < len(seeds)
    assert set(R.FRESH3_CONDS) >= {"bg3", "light3", "distract3", "all3"}
