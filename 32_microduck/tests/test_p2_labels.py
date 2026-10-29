"""Pure label tests collect without MLX; the pretrain tests import it only inside their bodies.

Sandbox (no MLX): ../.venv/bin/python -m pytest -q tests/test_p2_labels.py -k 'not pretrain'
Claude/MLX host: OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python -m pytest -q tests/test_p2_labels.py
"""
import json

import numpy as np
import pytest

from p2_labels import INVIEW_CLASSES, inview_labels, observable_mask


def track(cls, last, n=3):
    return [cls, 0.2, -0.1, n, last]


def test_inview_current_confirmed_tracks_multilabel_and_stale():
    memory = [
        {"tracks": [track("pad_red", 0), track("pad_red", 0), track("lamp_on", 0)]},
        {"tracks": [track("pad_red", 0), track("button", 1), track("ball", 1)], "lamp": ["on", 0]},
        {"tracks": [track("lamp_on", 2), track("lamp_off", 2), track("mat", 2)]},
        {"tracks": [track("pad_green", 3), track("pad_blue", 3), track("pad_yellow", 3)]},
        {"tracks": [], "lamp": ["off", 4]},  # lamp memory is NOT the in-view target
    ]
    labels = inview_labels({"result": {"success": True}, "truth": "must not matter"}, memory)
    assert INVIEW_CLASSES == ("pad_red", "pad_green", "pad_blue", "pad_yellow", "button",
                              "ball", "mat", "lamp_on", "lamp_off")
    assert labels.shape == (5, 9) and labels.dtype == np.uint8
    expected = [{"pad_red", "lamp_on"}, {"button", "ball"}, {"lamp_on", "lamp_off", "mat"},
                {"pad_green", "pad_blue", "pad_yellow"}, set()]
    for row, classes in zip(labels, expected):
        assert {INVIEW_CLASSES[i] for i in np.flatnonzero(row)} == classes
    assert np.array_equal(labels, inview_labels({}, memory))


def test_inview_unknown_classes_count_rows_and_never_become_targets():
    memory = [{"tracks": [track("wall", 0), track("wall", 0), track("button", 0)]},
              {"tracks": [track("wall", 0), track("alien", 1)]}]
    counts = {"wall": 5}
    labels = inview_labels({}, memory, unknown_counts=counts)
    assert counts == {"wall": 8, "alien": 1}  # stale rows count too; not a count of unique objects
    assert labels.sum() == 1 and labels[0, INVIEW_CLASSES.index("button")] == 1
    with pytest.warns(UserWarning, match="unknown track classes"):
        assert np.array_equal(labels, inview_labels({}, memory))


def test_inview_confirmation_already_applied_and_empty_episode():
    # A recorder using confirm=1 can legitimately store this row; don't apply a new threshold.
    assert inview_labels({"params": {"confirm": 1}}, [{"tracks": [track("ball", 0, n=1)]}]).sum() == 1
    assert inview_labels({}, []).shape == (0, len(INVIEW_CLASSES))
    assert inview_labels({}, [{}]).sum() == 0
    with pytest.raises(ValueError, match="five fields"):
        inview_labels({}, [{"tracks": [["ball", 0, 0, 3]]}])


def test_observable_mask_boundary_unknown_and_missing_sighting():
    labels = np.array([1, 2, 1, 0, 0, -1, 2], np.int32)
    memory = [{"lamp": x} for x in (["on", 4], ["off", 5], ["on", 8],
                                    None, ["off", 1], ["on", 9], None)]
    got = observable_mask(labels, memory, window_start=5)
    assert got.dtype == np.bool_
    assert got.tolist() == [False, True, True, True, True, False, False]


def test_observable_mask_uses_original_ticks_after_nan_filtering():
    memory = [{"lamp": None} for _ in range(12)]
    memory[5] = {"lamp": ["on", 5]}
    memory[7] = {"lamp": ["on", 1]}       # unknown after a press still passes (label is already 0)
    memory[10] = {"lamp": ["off", 3]}     # old sighting outside this window
    labels, ix = np.array([1, 0, 2]), np.array([5, 7, 10])
    mask = observable_mask(labels, memory, int(ix[0]), ticks=ix)
    assert mask.tolist() == [True, True, False]
    assert np.array_equal(mask, observable_mask(labels, [memory[t] for t in ix], 5))
    # A frame in burn-in counts as observed, but burn-in still contributes no auxiliary loss.
    M = np.array([0., 1., 1.])
    assert (M * mask).tolist() == [0, 1, 0]
    assert observable_mask([], [], 0).shape == (0,)


@pytest.mark.parametrize("labels,memory,start,ticks", [
    ([1], [], 0, None),                    # no implicit guessing of alignment
    ([3], [{}], 0, None),                  # not a lamp label
    ([1], [{}], -1, None),
    ([1], [{}], 0, [0.0]),                 # float index
    ([1], [{}], 0, [1]),                   # outside source
    ([1, 2], [{}, {}], 0, [1, 0]),         # original ticks must increase
    ([1, 2], [{}, {}], 0, np.array([1, 0], dtype=np.uint64)),  # no unsigned-difference wraparound
    ([1], [{}, {}], 1, [0]),               # before window start
])
def test_observable_rejects_misaligned_inputs(labels, memory, start, ticks):
    with pytest.raises(ValueError):
        observable_mask(labels, memory, start, ticks=ticks)


def test_pretrain_metrics_handle_imbalance_ties_and_absent_classes():
    from p2_pretrain import binary_auc, class_metrics
    assert binary_auc(np.array([1., 1., 1., 1.]), [0, 1, 0, 1]) == 0.5
    assert binary_auc(np.array([0., 1., 2., 3.]), [0, 0, 1, 1]) == 1.0
    assert binary_auc(np.array([0., 1., 2., 3.]), [1, 1, 0, 0]) == 0.0
    assert binary_auc(np.zeros(3), [0, 0, 0]) is None
    labels = np.zeros((10, len(INVIEW_CLASSES)), np.uint8)
    labels[0, 7] = 1
    metrics = class_metrics(np.full(labels.shape, -1.), labels)
    assert metrics["lamp_on"]["accuracy"] == 0.9
    assert metrics["lamp_on"]["balanced_accuracy"] == 0.5
    assert metrics["lamp_on"]["recall"] == 0
    assert metrics["lamp_on"]["auc"] == 0.5
    assert metrics["lamp_off"]["balanced_accuracy"] is None


def test_pretrain_20_step_smoke_resume_and_export(tmp_path, monkeypatch):
    """Eight steps + resume to twenty, using real npz/cache paths and an entire held-out config."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import p2_pretrain
    import p2_train
    from p2_learner import Policy2, policy_config

    monkeypatch.setattr(p2_train, "CACHE", tmp_path / "cache")
    records = []
    for cid in (0, 200):
        T = 18
        frames = np.zeros((T, 96, 96, 3), np.uint8)
        memory = []
        for t in range(T):
            on = t % 2 == 0
            frames[t, :, :, 0 if on else 2] = 220
            state = "on" if on else "off"
            memory.append({"tracks": [track("lamp_" + state, t), track("mat", t)], "lamp": [state, t]})
        header = {"split": "train", "config_id": f"train-{cid:04d}",
                  "instruction": {"task": "set_lamp", "target": "on"}, "events": []}
        cmd = np.zeros((T, 8), np.float32)
        cmd[15:, 7] = 1
        cmd[3] = np.nan
        path = tmp_path / f"episode_{cid}.npz"
        np.savez(path, frames=frames, body=np.zeros((T, 34), np.float32), cmd_bot=cmd,
                 header=np.array(json.dumps(header)), memory=np.array(json.dumps(memory)),
                 truth=np.array("not an input"))
        records.append({"path": str(path), "config": header["config_id"], "split": "train"})
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in [*records, records[0]]))
    run = tmp_path / "run"
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        p2_pretrain.main([str(run), "--index", str(index), "--steps", "8", "--batch", "4", "--seed", "7",
                          "--ckpt-every", "8", "--eval-every", "8", "--eval-batch", "8", "--lamp-weight", "2"])
        # All architecture/data/sampler settings must survive omitted flags on resume.
        p2_pretrain.main([str(run), "--resume", "--steps", "20", "--eval-batch", "8"])
        d = run / "ckpt_000020"
        meta = json.loads((d / "meta.json").read_text())
        assert meta["step"] == 20 and meta["cfg"]["lamp_weight"] == 2 and meta["cfg"]["act"] == "leaky"
        assert int(mx.load(str(d / "opt.safetensors"))["step"]) == 20
        data = json.loads((run / "dataset.json").read_text())
        assert data["fit"]["configs"] == [0] and data["heldout"]["configs"] == [200]
        assert data["fit"]["episodes"] == data["heldout"]["episodes"] == 1  # duplicated index row removed
        assert data["fit"]["frames"] == data["heldout"]["frames"] == 15
        report = json.loads((d / "heldout.json").read_text())
        assert report["complete"] and report["frames_evaluated"] == 15
        assert set(report["classes"]) == set(INVIEW_CLASSES)
        assert report["classes"]["lamp_on"]["positive"] == 8
        assert report["classes"]["lamp_off"]["positive"] == 7
        rng = np.random.default_rng(7)
        for _ in range(20):
            rng.integers(0, 15, size=4)
        assert meta["rng"] == rng.bit_generator.state

        initial = mx.load(str(run / "ckpt_000000" / "weights.safetensors"))
        final = mx.load(str(d / "weights.safetensors"))
        enc = set(p2_pretrain.ENCODER_LAYERS)
        assert initial.keys() == final.keys()
        assert any(not np.array_equal(np.array(initial[k]), np.array(final[k])) for k in final if k.split(".")[0] in enc)
        assert all(np.array_equal(np.array(initial[k]), np.array(final[k])) for k in final if k.split(".")[0] not in enc)
        # Exact load method used by p2_train --init; temporary classifier keys cannot leak into this file.
        exported = Policy2(**policy_config(d / "weights.safetensors"))
        exported.load_weights(str(d / "weights.safetensors"))
        assert exported.act == "leaky" and set(dict(tree_flatten(exported.parameters()))) == set(final)

        fit, heldout, _ = p2_pretrain.load_data([str(index)])
        assert fit[0]["ticks"].tolist() == [0, 1, 2, *range(4, 16)]
        restored = p2_pretrain.EncoderPretrainer(meta["cfg"])
        restored.load_weights(str(d / "pretrain.safetensors"))
        before = p2_pretrain.EncoderPretrainer(meta["cfg"])
        before.load_weights(str(run / "ckpt_000000" / "pretrain.safetensors"))
        args = [mx.array(fit[0]["frames"][:]), mx.array(fit[0]["labels"].astype(np.float32))]
        assert float(p2_pretrain.loss_fn(restored, *args)) < float(p2_pretrain.loss_fn(before, *args))
        f = args[0][:2].astype(mx.float32) / 255.0
        image_features = exported.encode(f[:, None], mx.zeros((2, 1, 34)), mx.zeros((2, 1, 9)))[:, 0, :192]
        assert np.allclose(np.array(restored.features(f)), np.array(image_features), atol=1e-6)
        trainable = dict(tree_flatten(restored.trainable_parameters()))
        assert all(k.startswith("classifier.") or k.split(".")[1] in enc for k in trainable)
        # A deadline cannot masquerade as a complete evaluation with high all-negative accuracy.
        partial = p2_pretrain.evaluate(restored, heldout, deadline=0)
        assert not partial["complete"] and partial["frames_evaluated"] == 0
        assert all(v["accuracy"] is None for v in partial["classes"].values())
    finally:
        mx.set_default_device(device)
