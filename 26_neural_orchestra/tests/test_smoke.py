"""End-to-end smoke test at tiny size: data -> train (+resume) -> classifier -> eval -> demo build -> JS decoder check.

Run from the project folder:  ../.venv/bin/python -m pytest -q
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
ENV = {**os.environ, "OMP_NUM_THREADS": "4"}

import baselines as B  # noqa: E402
import data  # noqa: E402
import metrics as M  # noqa: E402
import synth  # noqa: E402


def run(*args):
    r = subprocess.run([PY, *args], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("no")
    run("data.py", "--n_per_style", "80", "--out", str(tmp / "d.npz"))
    return tmp


def test_data_splits_disjoint(tiny):
    d = data.load(str(tiny / "d.npz"))
    assert d["hits"].shape[1:] == (32, 9)
    keys = [set(map(bytes, d["hits"][d["split"] == k].reshape(-1, 288))) for k in range(3)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    assert np.all(d["off"][d["hits"] == 0] == 0) and np.abs(d["off"]).max() <= 0.5
    assert len(np.unique(d["style"])) == 8


def test_synth_renders_sound(tiny):
    d = data.load(str(tiny / "d.npz"))
    w = synth.render(d["hits"][0], d["vel"][0], d["off"][0], bpm=120, loops=1)
    assert w.dtype == np.float32 and np.isfinite(w).all() and 0.05 < np.abs(w).max() <= 1.0


def test_baselines_and_metrics(tiny):
    d = data.load(str(tiny / "d.npz"))
    tr = data.subset(d, "train")
    rng = np.random.default_rng(0)
    for m in (B.BernoulliModel(), B.MarkovModel(), B.BarMarkovModel()):
        m.fit(tr)
        h, v, o = m.sample(np.arange(8), rng)
        assert h.shape == (8, 32, 9) and np.isfinite(m.nll(h, np.arange(8))).all()
    h, _, _ = B.CopyModel().fit(tr).sample(np.arange(8), rng)
    assert M.novelty(h, tr["hits"])["exact_copy_pct"] == 100.0


def test_train_resume_eval_demo(tiny):
    dp, vae, vae2, clf = (str(tiny / x) for x in ("d.npz", "vae", "vae2", "clf"))
    common = ["train.py", "--data", dp, "--hidden", "64", "--latent", "8", "--epochs", "4", "--warmup", "2", "--ckpt_every", "2"]
    out_full = run(*common, "--out", vae)
    run(*common, "--out", vae2, "--stop_after", "2")
    out_res = run(*common, "--out", vae2, "--resume")
    # resumed run must land on exactly the same final validation loss as the uninterrupted run
    last = lambda s: [ln for ln in s.splitlines() if ln.startswith("{")][-1].split(" ")[:-1]  # noqa: E731
    assert json.loads(" ".join(last(out_full)))["val_bce"] == json.loads(" ".join(last(out_res)))["val_bce"]
    run("train_classifier.py", "--data", dp, "--out", clf, "--epochs", "2")
    ev = str(tiny / "eval.json")
    run("eval.py", "--data", dp, "--vaes", vae, "--clf", clf, "--n_per_style", "6", "--iwae_k", "4", "--out", ev)
    res = json.load(open(ev))
    assert res["selected_vae"] == vae and "interpolation" in res
    media = str(tiny / "media")
    run("make_demo.py", "--data", dp, "--vae", vae, "--media", media)
    for f in ("demo.html", "pattern_grid.png", "interpolation_grid.png", "interp/rock_to_funk.wav",
              "listening_ab/_answer_key/answer_key.json", "listening_ab/pair01_A.wav"):
        assert os.path.exists(os.path.join(media, f)), f
    if shutil.which("node"):
        r = subprocess.run(["node", "web/check_decoder.js", os.path.join(media, "demo.html")], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


def test_ar_vae_demo_and_js_decoder(tiny):
    dp, ar = str(tiny / "d.npz"), str(tiny / "ar")
    run("train.py", "--arch", "ar", "--data", dp, "--hidden", "64", "--latent", "8", "--rnn", "32", "--epochs", "2",
        "--warmup", "1", "--free_bits", "4", "--out", ar)
    media = str(tiny / "media_ar")
    run("make_demo.py", "--data", dp, "--vae", ar, "--media", media, "--mode", "sample_T0.7")
    assert os.path.exists(os.path.join(media, "demo.html"))
    if shutil.which("node"):
        r = subprocess.run(["node", "web/check_decoder.js", os.path.join(media, "demo.html")], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
