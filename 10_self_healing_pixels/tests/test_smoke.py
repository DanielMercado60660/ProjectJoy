"""End-to-end smoke tests at tiny size (run: ../.venv/bin/python -m pytest -q, < 2 min)."""
import json, os, shutil, subprocess, sys
import numpy as np
import mlx.core as mx
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import nca, damage, eval as ev  # noqa: E402

PY = sys.executable
TINY = ["--size", "12", "--pad", "4", "--pool", "32", "--batch", "4", "--min_iter", "8", "--max_iter", "12",
        "--log_every", "10", "--ckpt_every", "10"]


def run_train(out, *extra):
    r = subprocess.run([PY, "train.py", "--out", out, *TINY, *extra], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


def test_target_and_model_shapes():
    t = nca.load_target("lizard", 12, 4)
    assert t.shape == (20, 20, 4) and t[..., 3].max() > 0.5
    assert np.all(t[..., :3] <= t[..., 3:4] + 1e-6)  # premultiplied
    m = nca.NCA()
    x = mx.array(nca.make_seed(20, 20, 2))
    y = m(x)
    assert y.shape == (2, 20, 20, 16)
    assert np.allclose(np.array(y), np.array(x))  # zero-init last layer: first step is identity


def test_damage_masks():
    rng = np.random.default_rng(0)
    for name, f in damage.ALL_DAMAGE.items():
        m = f(72, 72, rng)
        assert m.shape == (72, 72) and m.dtype == bool and 0 < m.mean() < 0.7, name
    x = np.ones((72, 72, 16), np.float32)
    m = damage.half(72, 72, rng)
    assert damage.apply(x, m)[m].sum() == 0 and damage.apply(x, m)[~m].min() == 1


def test_train_resume_and_eval(tmp_path):
    out = str(tmp_path / "run")
    run_train(out, "--mode", "regen", "--steps", "40", "--stop_after", "20")
    meta = json.load(open(os.path.join(out, "ckpt", "meta.json")))
    assert meta["step"] == 20 and os.path.exists(os.path.join(out, "ckpt", "pool.npy"))
    log = run_train(out, "--mode", "regen", "--steps", "40", "--resume")
    assert "resumed from step 20" in log
    hist = np.loadtxt(os.path.join(out, "loss.csv"), delimiter=",")
    assert hist.shape[0] == 40 and hist[-1, 0] == 40
    assert hist[-10:, 1].mean() < hist[:5, 1].mean()  # loss goes down
    # eval pieces on the tiny model
    model = nca.load_model(os.path.join(out, "model.safetensors"))
    target = nca.load_target("lizard", 12, 4)
    r, _ = ev.run_damage(model, target, "half", 2, 0)
    assert r["curve"].shape == (ev.REGEN + 1, 2) and np.isfinite(r["curve"]).all()
    s = ev.summarise(r, float((target ** 2).mean()))
    assert 0 <= s["recovered_frac"] <= 1


def test_export_and_js_port(tmp_path):
    m = nca.NCA()
    m.fc2.weight = mx.random.normal(m.fc2.weight.shape) * 0.05  # non-trivial update
    target = nca.load_target("lizard", 12, 4)
    d = nca.export_json(m, target)
    assert len(d["w1"]) == 128 * 48 and len(d["w2"]) == 16 * 128 and len(d["check"]["out"]) == 13 * 13 * 16
    if shutil.which("node") is None:
        pytest.skip("node not installed; JS port not checked")
    d["name"] = "test"
    html = open(os.path.join(HERE, "demo_template.html")).read().replace("/*__NCA_DATA__*/null", json.dumps({"main": d}))
    p = tmp_path / "demo.html"
    p.write_text(html)
    r = subprocess.run(["node", os.path.join(HERE, "tests", "check_demo_js.mjs"), str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
