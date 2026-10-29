"""Fast checks: data splits, side-channel controls, metrics, checkpoint/resume. Runs in well under 2 min."""
import inspect
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alien import (World, Receiver, make_episodes, reshuffle, compositional_code, holistic_code,  # noqa: E402
                   topographic_similarity, positional_disentanglement, message_stats)
from train import default_cfg, train, evaluate, receiver_accuracy, sender_table, load_model  # noqa: E402


def test_split_is_combinations_only():
    w = World()
    assert len(w.train_ids) + len(w.test_ids) == w.n == 128
    for j, d in enumerate(w.dims):  # every attribute value is seen in training
        assert len(np.unique(w.objects[w.train_ids, j])) == d
    c, t = make_episodes(w, 3000, 5, "train", seed=0)
    assert set(np.unique(c)) <= set(w.train_ids.tolist())
    c, t = make_episodes(w, 3000, 5, "test", seed=1)
    assert set(np.unique(c).tolist()) <= set(w.test_ids.tolist())   # all candidates novel
    assert all(len(set(row)) == 5 for row in c.tolist())  # candidates are distinct


def test_receiver_sees_only_message_and_candidates():
    params = list(inspect.signature(Receiver.__call__).parameters)
    assert params == ["self", "msg_onehot", "cands"]


def test_no_structural_side_channel():
    """Target position is uniform, and 'pick the centre of the candidate cluster' is at chance."""
    w = World()
    c, t = make_episodes(w, 20000, 5, "train", seed=3)
    freq = np.bincount(t, minlength=5) / len(t)
    assert np.all(np.abs(freq - 0.2) < 0.02)
    objs = w.objects[c]                                           # (E, K, A)
    ham = (objs[:, :, None, :] != objs[:, None, :, :]).sum(-1).sum(-1)  # (E, K)
    centre = np.argmin(ham + 1e-3 * np.random.RandomState(0).rand(*ham.shape), axis=1)
    assert abs((centre == t).mean() - 0.2) < 0.03
    c2, t2 = reshuffle(c, t, np.random.default_rng(0))            # reshuffle keeps the same target
    assert np.all(c2[np.arange(len(c2)), t2] == c[np.arange(len(c)), t])


def test_metrics_on_reference_codes():
    w = World()
    comp = compositional_code(w, 3, 8)
    hol = holistic_code(w, 3, 8, seed=0)
    assert topographic_similarity(w.objects, comp) > 0.99
    assert positional_disentanglement(w.objects, comp) > 0.99
    assert abs(topographic_similarity(w.objects, hol)) < 0.1
    assert message_stats(w.objects, hol, 8)["unique_messages"] == 128


def test_train_controls_and_resume(tmp_path):
    """Tiny end-to-end run: learns above chance; shuffled messages and no-comm fall to chance;
    a stopped-and-resumed run matches an uninterrupted one."""
    from mlx.utils import tree_flatten
    ck = str(tmp_path)
    cfg = default_cfg("full", "gumbel", 0, P=2, steps=600, eval_every=300, ckpt_every=300, n_eval_episodes=2000)
    model, w, log, _ = train(cfg, ckpt_dir=ck, resume=False, verbose=False)
    assert log[-1]["train_acc"] > 0.5
    cand, tpos = make_episodes(w, 4000, 5, "train", seed=99)
    table = sender_table(model, w, cfg)                         # (P, N, L)
    msgs = table[:, cand[np.arange(len(cand)), tpos]]           # (P, E, L)
    real = receiver_accuracy(model.receiver, w, msgs, cand, tpos, 8)
    perm = np.random.RandomState(0).permutation(len(cand))
    shuffled = receiver_accuracy(model.receiver, w, msgs[:, perm], cand, tpos, 8)
    assert np.all(real > 0.5) and np.all(np.abs(shuffled - 0.2) < 0.04), (real, shuffled)

    nc = default_cfg("full", "nocomm", 0, P=2, steps=300, eval_every=300, n_eval_episodes=2000)
    _, _, nlog, _ = train(nc, ckpt_dir=ck, resume=False, verbose=False)
    assert abs(nlog[-1]["train_acc"] - 0.2) < 0.04
    assert abs(nlog[-1]["heldout_acc"] - 0.2) < 0.04   # no "novelty" side channel on held-out episodes

    # resume: 300 steps, stop, resume to 600 -> same weights as the uninterrupted 600-step run
    ck2 = os.path.join(ck, "resume")
    train(cfg, ckpt_dir=ck2, resume=False, stop_at=300, verbose=False)
    m2, _, log2, path2 = train(cfg, ckpt_dir=ck2, resume=True, verbose=False)
    assert log2[-1]["step"] == 600
    a = mx.concatenate([v.reshape(-1) for _, v in tree_flatten(model.parameters())])
    b = mx.concatenate([v.reshape(-1) for _, v in tree_flatten(m2.parameters())])
    assert float(mx.abs(a - b).max()) < 1e-3
    m3, _, _, _ = load_model(path2)
    acc3, _ = evaluate(m3, w, cfg, {"x": (cand, tpos)})
    acc2, _ = evaluate(m2, w, cfg, {"x": (cand, tpos)})
    assert np.array_equal(acc3["x"], acc2["x"])
