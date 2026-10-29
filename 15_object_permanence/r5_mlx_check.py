"""Runner-only MLX checks: warm start, density parity, causal state, reveal renderer."""
import argparse
import pickle
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from model import Tracker
from r5_model import CueTracker, distribution, mixture_nll
from r5_data import read_split, frames, cues, look_flags, view_occ
from r5_metrics import nll
from train import render_mx
import sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn2', action='store_true')
    args = parser.parse_args()
    klass = CueTracker
    if args.turn2:
        from r5_turn2_model import BoundCueTracker
        klass = BoundCueTracker
    mx.random.seed(5)
    original = Tracker(enc_dim=32, hid=16)
    mx.eval(original.parameters())
    warm = {k: np.array(v) for k, v in tree_flatten(original.parameters())}
    d = read_split('hard_val', 1)
    f = mx.array(frames(d, 0)[None, :5])
    cue, pres, occ, flags = [mx.array(v) for v in (cues(d, [0]), d['present'].astype(np.float32),
                                                  d['occ'], look_flags(5)[None])]
    target = (d['pos'][:, :5]-32)/32
    common = None
    for k in (1, 3):
        model = klass(k, enc_dim=32, hid=16)
        model.warm_start(warm)
        if args.turn2:
            params = {key: np.array(value) for key, value in tree_flatten(model.parameters())
                      if not key.startswith('head2.')}
            if common is not None:
                assert common.keys() == params.keys()
                assert all(np.array_equal(common[key], params[key]) for key in common)
            common = params
            from r5_turn2_utils import query_features
            assert np.allclose(np.array(model.query_features(cue, pres, occ)),
                               query_features(np.array(cue), np.array(pres), np.array(occ)))
            # Avoid a trivial zero-head parity test: exercise learned-motion paths.
            model.head2.weight = mx.array(np.random.default_rng(8).normal(0, .02, model.head2.weight.shape).astype(np.float32))
        raw = model(f, cue, pres, occ, flags)
        assert raw.shape == (1, 5, 3, k, 6)
        if k == 1 and not args.turn2:
            assert np.allclose(np.array(raw[..., 0, :5]), np.array(original(f)), atol=1e-5)
        if args.turn2:
            anchor = ((np.array(cue)-32)/32*np.array(pres)[..., None])[:, :, None, :]
            assert np.allclose(np.array(raw[:, 0, ..., :2]), anchor)
            order = [2, 0, 1]
            permuted = model(f, cue[:, order], pres[:, order], occ, flags)
            assert np.allclose(np.array(permuted), np.array(raw)[:, :, order], atol=2e-4)
            import mlx.nn as nn
            from r5_model import training_loss
            _, gradient = nn.value_and_grad(model, training_loss)(model, f, cue, pres, occ, flags, mx.array(target), 1.)
            flat = dict(tree_flatten(gradient))
            for prefix in ('cue_input', 'cue_initial', 'cue_head'):
                norm = sum(float(np.square(np.array(v)).sum()) for key, v in flat.items() if key.startswith(prefix))
                assert np.isfinite(norm) and norm > 1e-12, (prefix, norm)
        expected = nll(distribution(raw), d['pos'][:, :5])
        observed = np.array(mixture_nll(raw, mx.array(target)))+2*np.log(32)
        assert np.allclose(expected, observed, atol=2e-3), np.max(abs(expected-observed))
        h, online = None, []
        for t in range(5):
            out, h = model.step(f[:, t], cue, pres, occ, flags[:, t], h)
            online.append(np.array(out))
        assert np.allclose(np.stack(online, 1), np.array(raw), atol=2e-4)
        assert np.allclose(np.array(model(f[:, :3], cue, pres, occ, flags[:, :3])), np.array(raw[:, :3]), atol=1e-5)
    # Exact static reveal renderer parity, including preservation of the other wall.
    ro = view_occ(d['occ'][0], 0)[None]
    expected = sim.render(d['pos'][:, :2], d['present'], ro, d['colors'], noise=0)
    observed = np.array(render_mx(mx.array(d['pos'][:, :2]), pres, mx.array(ro), mx.array(d['colors']), noise=0))
    assert np.max(abs(expected-observed)) < 1e-5
    # Real pretrained architecture/key compatibility is separate from the tiny run.
    with open('checkpoints/hard/final.state.pkl', 'rb') as fp:
        cfg = pickle.load(fp)['cfg']
    model = klass(3, **{key: cfg[key] for key in ('enc_dim', 'hid', 'layers')})
    with np.load('checkpoints/hard/final.weights.npz') as z:
        model.warm_start(dict(z))
    out = np.array(model(f[:, :1], cue, pres, occ, flags[:, :1]))
    assert out.shape == (1, 1, 3, 3, 6) and np.isfinite(out).all()
    print('MLX_CHECK_OK warm-start + NLL parity + causal GRU + renderer + real checkpoint'
          + (' + shared weights + query equivariance + all cue gradients' if args.turn2 else ''))


if __name__ == '__main__':
    main()
