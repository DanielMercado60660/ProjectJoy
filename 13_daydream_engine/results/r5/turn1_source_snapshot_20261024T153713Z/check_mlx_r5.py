"""Claude-only smoke assertions. This file is deliberately not a pytest module."""
import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

import sim
from r5_core import walls_step, pair_features, read_json
from r5_model import HybridEnsemble, Predictor, load_checkpoint, checkpoint_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumed", required=True)
    ap.add_argument("--reference", required=True)
    a = ap.parse_args()
    x = sim.random_init(16, np.random.default_rng(43)).astype(np.float32)
    x[0] = [.4, .5, 1, 0, .54, .5, -1, 0, .8, .8, 0, 0]
    actual = np.array(walls_step(mx.array(x), mx))
    np.testing.assert_allclose(actual, walls_step(x), atol=3e-6, rtol=2e-6)
    f_np, g_np = pair_features(x, walls_step(x))
    f_mx, g_mx = pair_features(mx.array(x), walls_step(mx.array(x), mx), mx)
    np.testing.assert_allclose(np.array(f_mx), f_np, atol=1e-4, rtol=1e-5)
    np.testing.assert_equal(np.array(g_mx), g_np)
    model, meta = load_checkpoint(a.resumed)
    reference, refmeta = load_checkpoint(a.reference)
    assert meta["step"] == refmeta["step"]
    assert meta["numpy_rng_states"] == refmeta["numpy_rng_states"]
    assert meta["bootstrap_scenes"] == refmeta["bootstrap_scenes"]
    for (key, w), (refkey, rw) in zip(tree_flatten(model.parameters()), tree_flatten(reference.parameters())):
        assert key == refkey
        np.testing.assert_allclose(np.array(w), np.array(rw), atol=2e-6, rtol=2e-6)
    left = mx.load(str(checkpoint_path(a.resumed) / "optimizer.safetensors"))
    right = mx.load(str(checkpoint_path(a.reference) / "optimizer.safetensors"))
    assert left.keys() == right.keys()
    assert int(left["step"]) == meta["step"]
    for key in left:
        np.testing.assert_allclose(np.array(left[key]), np.array(right[key]), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose([r["loss"] for r in meta["history"]],
                               [r["loss"] for r in refmeta["history"]], atol=2e-6, rtol=2e-6)
    initial = HybridEnsemble(meta["config"]["members"], meta["config"]["hidden"], meta["config"]["seed"])
    change = max(float(mx.max(mx.abs(w - iw))) for (_, w), (_, iw) in
                 zip(tree_flatten(model.parameters()), tree_flatten(initial.parameters())))
    assert change > 1e-7, "compiled training did not change weights"
    predictor = Predictor(model, chunk=7)
    ensemble = predictor.rollout(x, 3)
    assert ensemble.shape == (model.members, len(x), 4, 12)
    assert np.isfinite(ensemble).all()
    # Learned permutation symmetry, including compile/inference implementation.
    perm = [2, 0, 1]
    swapped = x.reshape(-1, 3, 4)[:, perm].reshape(-1, 12)
    rotated = predictor.rollout(swapped, 1)[:, :, 1].reshape(model.members, -1, 3, 4)
    np.testing.assert_allclose(rotated, ensemble[:, :, 1].reshape(model.members, -1, 3, 4)[:, :, perm], atol=2e-4, rtol=2e-4)
    # Explicitly check compile sees changed model state after compilation.
    state = mx.array(np.broadcast_to(x, (model.members,) + x.shape).copy())
    compiled = mx.compile(lambda s: model(s), inputs=[model.state])
    before = np.array(compiled(state))
    model.weights[-1] = model.weights[-1] * 0
    after = np.array(compiled(state))
    eager = np.array(model(state))
    np.testing.assert_allclose(after, eager, atol=2e-5, rtol=2e-5)
    assert np.max(np.abs(after - before)) > 1e-7, "compiled inference ignored changed model weights"
    # Random scenes might have no gated pair; include a contact example separately.
    contact = np.array([[.4, .5, 1, 0, .54, .5, -1, 0, .8, .8, 0, 0]], np.float32)
    cp = Predictor(reference).rollout(contact, 1)
    assert np.isfinite(cp).all()
    print(f"MLX_CHECKS_OK step={meta['step']} max_weight_change={change:.7g} resume_losses_match=True")


if __name__ == "__main__":
    main()
