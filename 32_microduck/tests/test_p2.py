"""Phase 2 checks, with synthetic recordings; run on Claude's MLX-capable host.

    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python -m pytest -q tests/test_p2.py
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

import p2_train
from p2_learner import P2Controller, Policy2, cmd_targets, decode, instr_vec, policy_config, prep_frame
from p2_train import aux_labels, batch, label_rows, load_all, load_episode, loss_fn, masked_aux, save
from record import down

HERE = Path(__file__).resolve().parent.parent


def belief_episode(task="set_lamp", target="on"):
    # Unsorted events and a future press also exercise causal event lookup.
    header = {"split": "train", "instruction": {"task": task, "target": target},
              "events": [{"event": "press", "tick": 6}, {"event": "press", "tick": 3},
                         {"event": "press", "tick": 20}, {"event": "look", "tick": 5}]}
    memory = [{"lamp": v} for v in (None, ["on", 1], ["on", 1], ["on", 3],
                                    ["off", 4], ["off", 4], ["off", 4], ["on", 7], ["on", 7])]
    return header, memory


@pytest.mark.parametrize("target,expected", [("on", [0, 1, 1, 0, 2, 2, 0, 1, 1]),
                                             ("off", [0, 2, 2, 0, 1, 1, 0, 2, 2])])
def test_belief_labels_press_stales_until_later_sighting(target, expected):
    header, memory = belief_episode(target=target)
    lamp, match = aux_labels(header, memory)
    assert lamp.tolist() == [0, 1, 1, 0, 2, 2, 0, 1, 1]
    assert match.tolist() == expected
    # A sighting at tick zero is known when there has been no press.
    assert aux_labels(dict(header, events=[]), [{"lamp": ["on", 0]}])[0].tolist() == [1]


@pytest.mark.parametrize("task,target", [("find_ball", None), ("go_and_return", "red")])
def test_match_masked_for_other_tasks(task, target):
    lamp, match = aux_labels(*belief_episode(task, target))
    assert lamp.tolist() == [0, 1, 1, 0, 2, 2, 0, 1, 1]
    assert np.all(match == -1)


def test_truncate_first_done_and_ignore_nan_rows():
    cmd = np.zeros((9, 8), np.float32)
    cmd[5:, 7] = 1
    cmd[2, 0] = np.nan
    keep, stats = label_rows(cmd)
    assert keep.tolist() == [0, 1, 3, 4, 5]
    assert stats == {"source_ticks": 9, "cut_ticks": 3, "invalid_ticks": 1}
    assert cmd[keep, 7].sum() == 1
    cmd[:, 7] = 0
    cmd[8] = np.nan
    keep, stats = label_rows(cmd)
    assert keep.tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert stats["cut_ticks"] == 0 and stats["invalid_ticks"] == 2


def test_cache_resolution_cleaning_and_index_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p2_train, "CACHE", tmp_path / "cache")
    header, memory = belief_episode()
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (9, 96, 96, 3), dtype=np.uint8)
    cmd = np.zeros((9, 8), np.float32)
    cmd[5:, 7] = 1
    cmd[2] = np.nan
    path = tmp_path / "episode.npz"
    # Deliberately unusable replay-only truth: no learner path may need its values or shape.
    np.savez(path, header=np.array(json.dumps(header)), memory=np.array(json.dumps(memory)),
             frames=frames, body=np.zeros((9, 34), np.float32), cmd_bot=cmd, truth=np.array("replay only"))
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"path": str(path), "split": "train"}) + "\n")
    paths = []
    for img in (64, 96):
        rows, eps = load_all([str(index)], img)
        ep = eps[0]
        assert len(rows) == 1 and ep["frames"].shape == (5, img, img, 3)
        assert ep["ticks"].tolist() == [0, 1, 3, 4, 5]
        assert ep["lamp_lbl"].tolist() == [0, 1, 0, 2, 2]  # still uses original tick 3
        assert ep["match_lbl"].tolist() == [0, 1, 0, 2, 2]
        assert ep["done"].tolist() == [0, 0, 0, 0, 1]
        assert np.array_equal(ep["frames"][0], prep_frame(frames[0], img))
        assert np.isfinite(ep["reg"]).all()
        again = load_episode(str(path), img)
        assert np.array_equal(ep["frames"][:], again["frames"][:])
        paths.append(ep["frames"].path)
        # Successful reuse cannot rerun preprocessing.
        with monkeypatch.context() as patch:
            def no_rebuild(*args):
                raise AssertionError("cache should be reused")
            patch.setattr(p2_train, "prep_frame", no_rebuild)
            load_episode(str(path), img)
    assert paths[0] != paths[1]
    assert all(p2_train.CACHE_VERSION in str(p) for p in paths)
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(reports) == 2
    assert all(r["cut_ticks"] == 3 and r["invalid_ticks"] == 1 and r["kept_ticks"] == 5 for r in reports)


@pytest.mark.parametrize("img", [64, 96])
def test_live_frames_match_recorded_path(img):
    frame = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    assert np.array_equal(prep_frame(frame, img), prep_frame(down(frame), img))
    assert prep_frame(frame, img).shape == (img, img, 3)


def synthetic_episode(img, T=12):
    rng = np.random.default_rng(T)
    return {"frames": rng.integers(0, 256, (T, img, img, 3), dtype=np.uint8),
            "body": (0.1 * rng.normal(size=(T, 34))).astype(np.float32),
            "instr": instr_vec({"task": "set_lamp", "target": "on"}),
            "reg": np.zeros((T, 6), np.float32), "done": np.eye(T, dtype=np.float32)[-1],
            "lamp_lbl": (np.arange(T) % 3).astype(np.int32),
            "match_lbl": (np.arange(T) % 3).astype(np.int32)}


@pytest.mark.parametrize("img", [64, 96])
@pytest.mark.parametrize("vx_head", ["regress", "binary"])  # T3: both image sizes and action heads
def test_one_training_step_and_reload(tmp_path, img, vx_head):
    rng = np.random.default_rng(0)
    ep = synthetic_episode(img, 6)
    if vx_head == "binary":
        commands = np.zeros((6, 8), np.float32)
        commands[::2, 0] = 0.3
        ep["reg"] = cmd_targets(commands)[0]
    arrays = batch([ep], np.array([6.0]), rng, 2, W=8, BURN=2, p_start=1, img=img)
    assert arrays[5].sum() == 12 and np.all(arrays[6][:, 6:] == -1)
    args = [mx.array(x) for x in arrays]
    mx.random.seed(0)
    model = Policy2(img=img, done_from_aux=True, vx_head=vx_head)
    opt = optim.Adam(learning_rate=1e-4)
    vg = nn.value_and_grad(model, loss_fn)
    (l0, _), grads = vg(model, *args)
    mx.eval(l0, grads)
    opt.update(model, grads)
    l1, _ = loss_fn(model, *args)
    assert float(l1) < float(l0)
    cfg = {"img": img, "aux_weight": 1.0, "done_from_aux": True, "vx_head": vx_head}
    save(tmp_path, 1, model, opt, rng, cfg)
    weights = tmp_path / "ckpt_000001" / "weights.safetensors"
    restored = Policy2(**policy_config(weights))
    assert policy_config(weights)["vx_head"] == vx_head
    assert json.loads((weights.parent / "meta.json").read_text())["cfg"]["vx_head"] == vx_head
    restored.load_weights(str(weights))
    inputs = (args[0].astype(mx.float32) / 255.0, args[1], args[2])
    a, b = model(*inputs), restored(*inputs)
    for x, y in ((a[0], b[0]), (a[1], b[1]), (a[3], b[3]),
                 (a[2]["lamp"], b[2]["lamp"]), (a[2]["match"], b[2]["match"])):
        assert np.array_equal(np.array(x), np.array(y))
    assert model.img.weight.shape[1] == (img // 16) ** 2 * 64
    if vx_head == "binary":
        assert np.array_equal(np.array(a[2]["walk"]), np.array(b[2]["walk"]))
    else:
        assert "walk" not in a[2]


def test_masked_aux_excludes_burn_padding_and_non_lamp_tasks():
    logits = mx.array([[[3., 0., 0.], [0., 3., 0.], [0., 0., 3.], [0., 3., 0.]]])
    labels, M = mx.array([[2, 1, -1, 0]]), mx.array([[0., 1., 1., 0.]])
    loss, correct, counts = masked_aux(logits, labels, M)
    assert np.array_equal(np.array(counts), [0, 1, 0])
    assert np.array_equal(np.array(correct), [0, 1, 0])
    assert np.isclose(float(loss), np.log(1 + 2 * np.exp(-3)), atol=1e-6)
    grad = mx.grad(lambda x: masked_aux(x, labels, M)[0])(logits)
    assert np.all(np.array(grad)[0, [0, 2, 3]] == 0)
    loss, correct, counts = masked_aux(logits, mx.full((1, 4), -1, dtype=mx.int32), M)
    assert float(loss) == 0 and float(counts.sum()) == 0 and float(correct.sum()) == 0


def test_done_reads_predicted_match_and_gru_steps_match_sequence():
    """On the CPU, where P2Controller runs the policy (the GPU sums in a different order, ~1e-4)."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        _done_reads_predicted_match()
    finally:
        mx.set_default_device(prev)


def _done_reads_predicted_match():
    mx.random.seed(0)
    model = Policy2(done_from_aux=True)
    model.done.weight = mx.concatenate([mx.zeros((1, 128)), mx.array([[0., 0., 1.]])], axis=-1)
    model.done.bias = mx.zeros(1)
    ep = synthetic_episode(64, 4)
    f = mx.array(ep["frames"][None]).astype(mx.float32) / 255.0
    body = mx.array(ep["body"][None])
    ins = mx.array(np.broadcast_to(ep["instr"], (1, 4, 9)).copy())
    reg, done, aux, last = model(f, body, ins)
    assert np.allclose(np.array(done), np.array(mx.softmax(aux["match"], axis=-1)[..., 2]))
    h, rs, ds, ls, ms = None, [], [], [], []
    for t in range(4):
        r, d, a, h = model(f[:, t:t + 1], body[:, t:t + 1], ins[:, t:t + 1], h)
        rs.append(r)
        ds.append(d)
        ls.append(a["lamp"])
        ms.append(a["match"])
    for expected, parts in ((reg, rs), (done, ds), (aux["lamp"], ls), (aux["match"], ms)):
        assert np.allclose(np.array(expected), np.array(mx.concatenate(parts, axis=1)), atol=1e-5)
    assert np.allclose(np.array(last), np.array(h), atol=1e-6)
    plain = Policy2(aux=False)
    assert plain(f, body, ins)[2] == {}
    assert plain.done.weight.shape == (1, 128)
    with pytest.raises(ValueError, match="requires auxiliary"):
        Policy2(aux=False, done_from_aux=True)


@pytest.mark.parametrize("img", [64, 96])
def test_controller_runs_20_ticks(tmp_path, img):
    from room import World
    from tasks import make_episode, run_episode
    cfg = json.loads((HERE / "splits" / "train.json").read_text())["configs"][0]
    model = Policy2(img=img, done_from_aux=True)
    # A neutral untrained checkpoint tests the real controller without random early termination.
    reg, _ = cmd_targets(np.zeros(8, np.float32))
    model.reg.weight = mx.zeros_like(model.reg.weight)
    model.reg.bias = mx.array(np.arctanh(reg))
    model.done.weight = mx.zeros_like(model.done.weight)
    model.done.bias = mx.array([-20.])
    weights = tmp_path / "weights.safetensors"
    model.save_weights(str(weights))
    (tmp_path / "p2_config.json").write_text(json.dumps({"img": img, "aux": True, "done_from_aux": True}))
    device, iface = mx.default_device(), None
    try:
        pc = P2Controller(weights)
        assert pc.cfg["img"] == img and pc.cfg["done_from_aux"]
        ep = make_episode(cfg, "go_and_return", 0)
        ep["budget_ticks"] = 20
        result, _, iface = run_episode(cfg, ep, pc.factory, world=World(cfg))
        assert iface.tick_count == 20 and result["falls"] == 0
        assert len(pc.latency) >= 20 and np.isfinite(pc.latency).all()
    finally:
        if iface is not None and iface._renderer is not None:
            iface._renderer.close()
        mx.set_default_device(device)


@pytest.mark.parametrize("img", [64, 96])
@pytest.mark.parametrize("vx_head", ["regress", "binary"])  # T3: resume must inherit the head too
def test_resume_continues_bit_exact(tmp_path, monkeypatch, img, vx_head):
    eps = [synthetic_episode(img, t) for t in (12, 18)]
    monkeypatch.setattr(p2_train, "load_all", lambda *args, **kwargs: ([{}, {}], eps))
    common = ["--index", "unused", "--ckpt-every", "4", "--batch", "1", "--window", "4", "--burn", "1",
              "--img", str(img), "--done-from-aux", "--aux-weight", "0.7", "--seed", "3", "--vx-head", vx_head]

    def run(name, steps, resume=False):
        # Resume must inherit img/aux/sampler settings even when they aren't passed again.
        monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / name), "--steps", str(steps),
                                         *(["--resume"] if resume else common)])
        p2_train.main()

    run("straight", 8)
    run("resumed", 4)
    run("resumed", 8, resume=True)
    a = tmp_path / "straight" / "ckpt_000008"
    b = tmp_path / "resumed" / "ckpt_000008"
    for name in ("weights.safetensors", "opt.safetensors"):
        wa, wb = mx.load(str(a / name)), mx.load(str(b / name))
        assert wa.keys() == wb.keys()
        assert all(np.array_equal(np.array(wa[k]), np.array(wb[k])) for k in wa)
    ma, mb = json.loads((a / "meta.json").read_text()), json.loads((b / "meta.json").read_text())
    assert ma["rng"] == mb["rng"] and ma["mlx_rng"] == mb["mlx_rng"] and mb["step"] == 8
    assert mb["cfg"]["img"] == img and mb["cfg"]["done_from_aux"] and mb["cfg"]["aux_weight"] == 0.7
    assert mb["cfg"]["vx_head"] == vx_head


@pytest.mark.parametrize("flag,value", [("--img", "96"), ("--vx-head", "binary")])  # T3
def test_resume_rejects_changed_model_config(tmp_path, monkeypatch, flag, value):
    eps = [synthetic_episode(64)]
    monkeypatch.setattr(p2_train, "load_all", lambda *args, **kwargs: ([{}], eps))
    monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path), "--index", "unused", "--steps", "0"])
    p2_train.main()
    monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path), "--resume", flag, value])
    with pytest.raises(SystemExit):
        p2_train.main()


# T3: binary walking is an action head, independent of lamp/match auxiliary heads.
@pytest.mark.parametrize("walk_logit,expected", [(-100., 0.), (-1e-6, 0.), (0., 0.), (1e-6, 0.3), (100., 0.3)])
def test_binary_decode_threshold(walk_logit, expected):
    reg, _ = cmd_targets(np.array([0.4, 0, -0.3, 0.1, -0.2, 0.5, 0.05, 0], np.float32))
    legacy = decode(reg, 1.0)
    for ignored_vx in (-1., 1.):
        reg[0] = ignored_vx
        cmd = decode(reg, 1.0, vx_head="binary", walk_logit=walk_logit)
        assert cmd.vx == expected and cmd.done
        assert cmd.vy == legacy.vy and cmd.wz == legacy.wz and cmd.head == legacy.head
    assert policy_config(cfg={})["vx_head"] == "regress"
    with pytest.raises(ValueError, match="walk logit"):
        decode(reg, 0.0, vx_head="binary")


def test_binary_loss_replaces_vx_mse_and_uses_window_mask():
    commands = np.zeros((4, 8), np.float32)
    commands[:, 0] = [0., 0.15, 0.1501, 0.3]  # strict threshold, including equality
    targets = cmd_targets(commands)[0][None]

    class FixedOutputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.vx_head = "binary"
            predicted = targets.copy()
            predicted[..., 0] = 0.9             # wrong vx regression must carry no loss or gradient
            self.reg_values = mx.array(predicted)   # not "values": nn.Module is a dict
            self.walk_values = mx.array([[1., -1., 2., 100.]])

        def __call__(self, *args):
            return self.reg_values, mx.zeros((1, 4)), {"walk": self.walk_values}, mx.zeros((1, 256))

    model = FixedOutputs()
    args = [mx.zeros((1, 4, 64, 64, 3)), mx.zeros((1, 4, 34)), mx.zeros((1, 4, 9)),
            mx.array(targets), mx.zeros((1, 4)), mx.array([[1., 1., 1., 0.]]),
            mx.full((1, 4), -1, dtype=mx.int32), mx.full((1, 4), -1, dtype=mx.int32)]
    (loss, metrics), grad = nn.value_and_grad(model, loss_fn)(model, *args, aux_weight=0.0)
    # First two targets are stop, third is walk, fourth is masked despite its extreme logit.
    expected_walk = np.logaddexp(0, np.array([1., -1., -2.])).sum() / (3 + 1e-6)
    assert np.isclose(float(metrics["walk_bce"]), expected_walk, atol=1e-6)
    assert float(metrics["mse"]) == 0
    assert np.isclose(float(loss), expected_walk + 3 * np.log(2) / (3 + 1e-6), atol=1e-6)
    assert np.all(np.array(grad["reg_values"])[..., 0] == 0)
    assert float(grad["walk_values"][0, 3]) == 0
    assert float(grad["walk_values"][0, 0]) > 0 and float(grad["walk_values"][0, 2]) < 0
    assert np.isclose(float(loss_fn(model, *args, aux_weight=100.0)[0]), float(loss))
    # Other regression outputs still matter: wz has weight 1, each head value weight 0.5.
    delta = np.zeros_like(targets)
    delta[..., 1:] = 0.1
    model.reg_values = model.reg_values + mx.array(delta)
    assert np.isclose(float(loss_fn(model, *args)[0]) - float(loss), 0.03, atol=1e-6)


def test_binary_controller_emits_stop_and_walk_without_aux(tmp_path):
    """Exercise both decode branches through the real controller; no simulator truth in observations."""
    model = Policy2(aux=False, vx_head="binary")
    weights = tmp_path / "binary.safetensors"
    model.save_weights(str(weights))
    cfg = {"img": 64, "aux": False, "vx_head": "binary"}

    class Observation:
        body = np.zeros(34, np.float32)
        sim_time = 0.0

        def frame(self):
            return np.zeros((128, 128, 3), np.uint8)

    class Interface:
        def observe(self):
            return Observation()

    device = mx.default_device()
    try:
        pc = P2Controller(weights, cfg=cfg)
        pc.model.walk.weight = mx.zeros_like(pc.model.walk.weight)
        gen = pc.factory(Interface(), {"task": "set_lamp", "target": "on"})
        velocities = []
        for i, bias in enumerate((-5., 5., 0.)):
            pc.model.walk.bias = mx.array([bias])
            cmd = next(gen) if i == 0 else gen.send(Observation())
            velocities.append(cmd.vx)
        gen.close()
        assert velocities == [0., 0.3, 0.] and set(velocities) == {0., 0.3}
        assert len(pc.latency) == 3
    finally:
        mx.set_default_device(device)


def test_dagger_recording_with_p2_policy(tmp_path, monkeypatch):
    """Amendment 3: record_episode drives with a Phase 2 policy when policy_kind='p2' (explicit, since
    M5 run directories also hold a meta.json), and stores the kind in the header."""
    import record
    cfg = json.loads((HERE / "splits" / "train.json").read_text())["configs"][0]
    model = Policy2(img=64, act="leaky")
    reg, _ = cmd_targets(np.zeros(8, np.float32))
    model.reg.weight = mx.zeros_like(model.reg.weight)
    model.reg.bias = mx.array(np.arctanh(reg))
    model.done.weight = mx.zeros_like(model.done.weight)
    model.done.bias = mx.array([-20.])
    weights = tmp_path / "weights.safetensors"
    model.save_weights(str(weights))
    (tmp_path / "p2_config.json").write_text(json.dumps({"img": 64, "aux": True, "act": "leaky"}))
    monkeypatch.setitem(record.BUDGET_TICKS, "set_lamp", 20)
    device = mx.default_device()
    try:
        r = record.record_episode(cfg, "train", "set_lamp", 3, out_dir=tmp_path / "eps", policy_weights=str(weights),
                                  beta=0.0, tag="t", policy_kind="p2")
        with pytest.raises(ValueError):
            record.record_episode(cfg, "train", "set_lamp", 3, out_dir=tmp_path / "eps", policy_weights=str(weights),
                                  beta=0.0, tag="u", policy_kind="other")
    finally:
        mx.set_default_device(device)
    z = np.load(r["path"])
    H = json.loads(str(z["header"]))
    assert H["dagger"]["policy_kind"] == "p2" and H["dagger"]["policy_ticks"] == r["ticks"] == 20
    assert np.abs(z["cmd_exec"][:, :3]).max() < 1e-5 and not H["result"]["declared_done"]   # the neutral policy stands
    assert np.isfinite(z["cmd_bot"]).all()


def test_drop_done_tick_removes_only_the_flagged_done(tmp_path, monkeypatch, capsys):
    """Amendment 3: an index row flagged drop_done_tick loses its final 'done' tick (removed, not
    relabelled); the rest of the episode and unflagged rows are unchanged."""
    monkeypatch.setattr(p2_train, "CACHE", tmp_path / "cache")
    header, memory = belief_episode()
    cmd = np.zeros((9, 8), np.float32)
    cmd[5:, 7] = 1
    path = tmp_path / "episode.npz"
    np.savez(path, header=np.array(json.dumps(header)), memory=np.array(json.dumps(memory)),
             frames=np.zeros((9, 96, 96, 3), np.uint8), body=np.zeros((9, 34), np.float32), cmd_bot=cmd)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"path": str(path), "split": "train"}) + "\n"
                     + json.dumps({"path": str(path), "split": "train", "drop_done_tick": True}) + "\n")
    _, (plain, dropped) = load_all([str(index)], 64)
    assert plain["done"].tolist() == [0, 0, 0, 0, 0, 1] and dropped["done"].tolist() == [0] * 5
    assert len(dropped["frames"]) == 5 and dropped["frames"][:].shape[0] == 5
    for k in ("body", "reg", "lamp_lbl", "match_lbl", "ticks"):
        assert np.array_equal(dropped[k], plain[k][:-1])
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["dropped_done"] == 1 and report["kept_ticks"] == 11


@pytest.mark.parametrize("fresh", [False, True])
def test_takeover_hands_control_to_the_bot(tmp_path, monkeypatch, fresh):
    """Amendment 5: the policy drives until takeover_tick, then the bot's commands are executed (the
    shadow bot, or a new bot instance with takeover_fresh)."""
    import record
    cfg = json.loads((HERE / "splits" / "train.json").read_text())["configs"][0]
    model = Policy2(img=64, act="leaky")
    reg, _ = cmd_targets(np.zeros(8, np.float32))
    model.reg.weight = mx.zeros_like(model.reg.weight)
    model.reg.bias = mx.array(np.arctanh(reg))
    model.done.weight = mx.zeros_like(model.done.weight)
    model.done.bias = mx.array([-20.])
    weights = tmp_path / "weights.safetensors"
    model.save_weights(str(weights))
    (tmp_path / "p2_config.json").write_text(json.dumps({"img": 64, "aux": True, "act": "leaky"}))
    device = mx.default_device()
    try:
        r = record.record_episode(cfg, "train", "set_lamp", 3, out_dir=tmp_path / "eps", policy_weights=str(weights),
                                  beta=0.0, tag="t", policy_kind="p2", takeover_tick=10, takeover_fresh=fresh,
                                  budget_ticks=40, after_bot_done="continue" if fresh else "stop")
    finally:
        mx.set_default_device(device)
    z = np.load(r["path"])
    H = json.loads(str(z["header"]))
    assert H["dagger"]["policy_ticks"] == 10 and H["dagger"]["takeover_tick"] == 10 and r["ticks"] == 40
    assert np.abs(z["cmd_exec"][:10, :3]).max() < 1e-5                              # the neutral policy stands
    assert np.array_equal(z["cmd_exec"][10:], z["cmd_bot"][10:])                     # then the bot drives
    assert (H["dagger"]["pre_takeover_events"] is not None) == fresh


# Amendment 6: discrete turn head, recovery sampling group, full resume.
def test_turn_head_decodes_in_place_turns_only_when_standing():
    reg, _ = cmd_targets(np.array([0.0, 0, 0.2, 0, 0, 0, 0, 0], np.float32))
    for logits, wz in (([5, 0, 0], 0.0), ([0, 5, 0], 0.55), ([0, 0, 5], -0.55)):
        assert decode(reg, -1.0, "binary", -1.0, np.array(logits, np.float32)).wz == pytest.approx(wz)
        walking = decode(reg, -1.0, "binary", 1.0, np.array(logits, np.float32))
        assert walking.vx == 0.3 and walking.wz == pytest.approx(0.2, abs=1e-5)   # steering stays regression
    assert decode(reg, -1.0, "binary", -1.0).wz == pytest.approx(0.2, abs=1e-5)  # no turn head: unchanged


def test_turn_loss_uses_standing_ticks_and_the_bot_sign():
    model = Policy2(img=64, act="leaky", vx_head="binary", turn_head="discrete")
    cmds = np.zeros((6, 8), np.float32)
    cmds[:, 2] = [0.55, -0.55, 0.0, 0.05, 0.55, -0.2]
    cmds[4:, 0] = 0.3                                     # last two ticks walk: no turn label
    R = cmd_targets(cmds)[0][None]
    lbl = np.where(R[0, :, 1] > p2_train.TURN_REG_THRESHOLD, 1, np.where(R[0, :, 1] < -p2_train.TURN_REG_THRESHOLD, 2, 0))
    assert lbl[:4].tolist() == [1, 2, 0, 0] and (R[0, :, 0] < p2_train.STAND_REG_THRESHOLD).tolist() == [True] * 4 + [False] * 2
    ep = synthetic_episode(64, 6)
    F = mx.array(ep["frames"][None]); Bo = mx.array(ep["body"][None]); In = mx.array(np.broadcast_to(ep["instr"], (1, 6, 9)).copy())
    M = mx.ones((1, 6)); L = mx.full((1, 6), -1); A = mx.full((1, 6), -1)
    _, m = loss_fn(model, F, Bo, In, mx.array(R), mx.zeros((1, 6)), M, L, A)
    logits = np.array(model(F.astype(mx.float32) / 255.0, Bo, In)[2]["turn"][0])
    ce = -np.log(np.exp(logits) / np.exp(logits).sum(-1, keepdims=True))[np.arange(6), lbl]
    assert float(m["turn_ce"]) == pytest.approx(ce[:4].mean(), rel=1e-4)


def test_recovery_windows_start_at_or_after_focus_and_rec_n_zero_is_unchanged():
    eps = [synthetic_episode(64, t) for t in (12, 18)]
    lens = np.array([12.0, 18.0])
    rec = ([synthetic_episode(64, 300)], [200])
    a = batch(eps, lens, np.random.default_rng(5), 3, W=8, BURN=2, img=64)
    b = batch(eps, lens, np.random.default_rng(5), 3, W=8, BURN=2, img=64, rec=rec, rec_n=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    starts = set()
    for seed in range(40):
        F = batch(eps, lens, np.random.default_rng(seed), 2, W=8, BURN=2, img=64, rec=rec, rec_n=1)[0]
        frames = rec[0][0]["frames"]
        s = next(i for i in range(len(frames) - 7) if np.array_equal(frames[i], F[0, 0]))
        starts.add(s)
    assert min(starts) >= 200 and max(starts) <= 292 and len(starts) > 5


def test_resume_keeps_recovery_turn_head_and_ckpt_every(tmp_path, monkeypatch):
    main_eps = [synthetic_episode(64, t) for t in (12, 18)]
    rec_ep = dict(synthetic_episode(64, 20), ticks=np.arange(20))

    def fake_load_all(index, *args, **kwargs):
        return ([{"takeover_tick": 9}], [rec_ep]) if index == ["rec"] else ([{}, {}], main_eps)
    monkeypatch.setattr(p2_train, "load_all", fake_load_all)
    common = ["--index", "unused", "--ckpt-every", "4", "--batch", "2", "--window", "4", "--burn", "1",
              "--aux-weight", "1", "--seed", "2", "--act", "leaky", "--vx-head", "binary", "--turn-head", "discrete",
              "--recovery-index", "rec", "--recovery-frac", "0.5", "--recovery-lead", "3"]

    def run(name, steps, resume=False):
        monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / name), "--steps", str(steps),
                                         *(["--resume"] if resume else common)])
        p2_train.main()
    run("straight", 12)
    run("resumed", 4)
    run("resumed", 12, resume=True)
    assert (tmp_path / "resumed" / "ckpt_000008").exists()           # --ckpt-every 4 survived the resume
    a, b = tmp_path / "straight" / "ckpt_000012", tmp_path / "resumed" / "ckpt_000012"
    wa, wb = mx.load(str(a / "weights.safetensors")), mx.load(str(b / "weights.safetensors"))
    assert all(np.array_equal(np.array(wa[k]), np.array(wb[k])) for k in wa) and "turn.weight" in wa
    cfg = json.loads((b / "meta.json").read_text())["cfg"]
    assert cfg["turn_head"] == "discrete" and cfg["recovery_index"] == ["rec"] and cfg["recovery_frac"] == 0.5
    assert policy_config(str(b / "weights.safetensors"))["turn_head"] == "discrete"


def test_controller_with_discrete_heads_turns_at_the_bot_rate(tmp_path):
    from room import World
    from tasks import make_episode, run_episode
    cfg = json.loads((HERE / "splits" / "train.json").read_text())["configs"][0]
    model = Policy2(img=64, act="leaky", vx_head="binary", turn_head="discrete")
    model.walk.weight = mx.zeros_like(model.walk.weight)
    model.walk.bias = mx.array([-5.])                         # stand
    model.turn.weight = mx.zeros_like(model.turn.weight)
    model.turn.bias = mx.array([0., 5., 0.])                  # turn left in place
    model.done.weight = mx.zeros_like(model.done.weight)
    model.done.bias = mx.array([-20.])
    weights = tmp_path / "weights.safetensors"
    model.save_weights(str(weights))
    (tmp_path / "p2_config.json").write_text(json.dumps({"img": 64, "aux": True, "act": "leaky",
                                                         "vx_head": "binary", "turn_head": "discrete"}))
    device, iface, sent = mx.default_device(), None, []
    try:
        pc = P2Controller(weights)

        def fac(i, instr):
            g = pc.factory(i, instr)
            c = next(g)
            while True:
                sent.append(c)
                obs = yield c
                c = g.send(obs)
        ep = make_episode(cfg, "set_lamp", 0)
        ep["budget_ticks"] = 10
        _, _, iface = run_episode(cfg, ep, fac, world=World(cfg))
    finally:
        if iface is not None and iface._renderer is not None:
            iface._renderer.close()
        mx.set_default_device(device)
    assert len(sent) >= 10 and all(c.vx == 0.0 and c.wz == pytest.approx(0.55) for c in sent)


def test_recovery_training_runs_past_the_first_progress_log(tmp_path, monkeypatch):
    """Regression: the progress-log dict once overwrote the recovery group at step 100."""
    main_eps = [synthetic_episode(64, t) for t in (12, 18)]
    rec_ep = dict(synthetic_episode(64, 20), ticks=np.arange(20))
    monkeypatch.setattr(p2_train, "load_all", lambda index, *a, **k: ([{"takeover_tick": 9}], [rec_ep]) if index == ["rec"] else ([{}, {}], main_eps))
    monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / "r"), "--index", "unused", "--steps", "102",
                                     "--ckpt-every", "1000", "--batch", "2", "--window", "4", "--burn", "1", "--act", "leaky",
                                     "--recovery-index", "rec", "--recovery-frac", "0.5"])
    p2_train.main()
    assert json.loads((tmp_path / "r" / "latest.json").read_text())["step"] == 102


def test_p_end_windows_end_on_the_last_tick_and_zero_is_unchanged():
    """Amendment 7: p_end windows end on the episode's final (done) tick; p_end=0 keeps the old draws."""
    eps = [synthetic_episode(64, t) for t in (30, 40)]
    lens = np.array([30.0, 40.0])
    a = batch(eps, lens, np.random.default_rng(3), 6, W=8, BURN=2, img=64)
    b = batch(eps, lens, np.random.default_rng(3), 6, W=8, BURN=2, img=64, p_end=0.0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    D, M = batch(eps, lens, np.random.default_rng(3), 50, W=8, BURN=2, img=64, p_start=0.0, p_end=1.0)[4:6]
    assert np.all(D[:, -1] == 1.0) and np.all(M[:, -1] == 1.0)     # every window shows its done tick
    D0 = batch(eps, lens, np.random.default_rng(3), 50, W=8, BURN=2, img=64, p_start=0.0)[4]
    assert D0.sum() < 10                                              # uniform starts rarely do


def test_streams_play_episodes_in_order_and_reset_memory():
    eps = [synthetic_episode(64, t) for t in (10, 7)]
    st = p2_train.Streams(2, 4)
    rng = np.random.default_rng(0)
    seen = {0: [], 1: []}
    for step in range(12):
        (F, Bo, In, R, D, M, L, A), h0 = st.next(eps, rng, 4)
        for b in range(2):
            e = eps[st.ep[b]]
            s = int(st.pos[b]) - 4
            n = int(M[b].sum())
            assert n == min(4, len(e["frames"]) - s) and np.all(M[b, n:] == 0)
            assert np.array_equal(F[b, :n], e["frames"][s:s + n])         # chunks follow the episode in order
            if s == 0:
                assert np.all(h0[b] == 0)                                 # fresh episode, fresh memory
            else:
                assert np.all(h0[b] == step + 1)                          # carried from the previous chunk
            seen[b].append((int(st.ep[b]), s))
        st.update(np.full((2, 4), step + 2, np.float32))                  # stand-in for the model's h_last
    for b in seen:                                                        # every tick, done included, once per pass
        for (e0, s0), (e1, s1) in zip(seen[b], seen[b][1:]):
            assert (e1 == e0 and s1 == s0 + 4) or (s1 == 0 and s0 + 4 >= len(eps[e0]["frames"]))


def test_stream_training_resumes_bit_exact_and_carries_memory(tmp_path, monkeypatch):
    main_eps = [synthetic_episode(64, t) for t in (12, 18)]
    monkeypatch.setattr(p2_train, "load_all", lambda index, *a, **k: ([{}, {}], main_eps))
    common = ["--index", "unused", "--ckpt-every", "4", "--batch", "2", "--window", "4", "--aux-weight", "1",
              "--burn", "1", "--seed", "2", "--act", "leaky", "--vx-head", "binary", "--turn-head", "discrete", "--stream",
              "--stream-pool", "3", "--stream-warmup", "2"]

    def run(name, steps, resume=False):
        monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / name), "--steps", str(steps),
                                         *(["--resume"] if resume else common)])
        p2_train.main()
    run("straight", 12)
    run("resumed", 6)
    run("resumed", 12, resume=True)
    a, b = tmp_path / "straight" / "ckpt_000012", tmp_path / "resumed" / "ckpt_000012"
    wa, wb = mx.load(str(a / "weights.safetensors")), mx.load(str(b / "weights.safetensors"))
    assert all(np.array_equal(np.array(wa[k]), np.array(wb[k])) for k in wa)
    za, zb = np.load(a / "streams.npz"), np.load(b / "streams.npz")
    assert all(np.array_equal(za[k], zb[k]) for k in ("ep", "pos", "h"))
    assert np.abs(za["h"]).sum() > 0 and json.loads((b / "meta.json").read_text())["cfg"]["stream"] is True


@pytest.mark.parametrize("extra", [["--stream", "--p-end", "0.1"], ["--p-end", "0.9"],
                                   ["--stream", "--recovery-index", "r", "--recovery-frac", "0.5"]])
def test_stream_and_p_end_argument_checks(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / "x"), "--index", "unused", *extra])
    with pytest.raises(SystemExit):
        p2_train.main()


def test_chunked_stream_outputs_match_one_full_pass():
    """Amendment 7: with frozen weights, 4-tick chunks with carried memory (incl. a padded last chunk)
    give the same outputs as running the whole episode at once."""
    mx.random.seed(0)
    model = Policy2(img=64, act="leaky", vx_head="binary", turn_head="discrete")
    ep = synthetic_episode(64, 10)
    full = model(mx.array(ep["frames"][None]).astype(mx.float32) / 255.0, mx.array(ep["body"][None]),
                 mx.array(np.repeat(ep["instr"][None, None], 10, 1)))
    st = p2_train.Streams(1, model.gru.hidden_size)
    outs = []
    for _ in range(3):                                   # 4 + 4 + 2 real ticks
        (F, Bo, In, R, D, M, L, A), h0 = st.next([ep], np.random.default_rng(0), 4)
        reg, dl, aux, h_last = model(mx.array(F).astype(mx.float32) / 255.0, mx.array(Bo), mx.array(In), mx.array(h0))
        n = int(M[0].sum())
        outs.append((np.array(reg[0, :n]), np.array(dl[0, :n]), np.array(aux["turn"][0, :n])))
        st.update(np.array(h_last))
    for k in range(3):
        got = np.concatenate([o[k] for o in outs])
        want = np.array([full[0], full[1], full[2]["turn"]][k][0])
        assert got.shape[0] == 10 and np.allclose(got, want, atol=1e-5)


def test_stream_reset_zeroes_memory_but_keeps_the_order(tmp_path, monkeypatch):
    main_eps = [synthetic_episode(64, t) for t in (12, 18)]
    monkeypatch.setattr(p2_train, "load_all", lambda index, *a, **k: ([{}, {}], main_eps))
    seen = []
    real = p2_train.loss_fn

    def spy(model, *arrays, aux_weight=1.0, h0=None):
        seen.append(None if h0 is None else float(mx.abs(h0).sum()))
        return real(model, *arrays, aux_weight=aux_weight, h0=h0)
    monkeypatch.setattr(p2_train, "loss_fn", spy)
    monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / "r"), "--index", "unused", "--steps", "6",
                                     "--batch", "2", "--window", "4", "--burn", "1", "--act", "leaky",
                                     "--stream", "--stream-reset", "--stream-pool", "2",
                                     "--stream-warmup", "0"])
    p2_train.main()
    assert seen and all(v == 0.0 for v in seen)
    summ = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert summ["loss_ticks"] > 0 and summ["done_ticks"] >= 1


def test_stream_pool_draws_rows_at_random_and_keeps_each_stream_in_order():
    """Amendment 7a: pool > batch; each drawn stream continues where it stopped, with its own memory."""
    eps = [synthetic_episode(64, t) for t in (9, 13, 21)]
    st = p2_train.Streams(2, 3, pool=6)
    rng = np.random.default_rng(1)
    where = {}
    drawn = set()
    for step in range(40):
        (F, Bo, In, R, D, M, L, A), h0 = st.next(eps, rng, 4)
        for b, k in enumerate(st.last):
            drawn.add(int(k))
            s = int(st.pos[k]) - 4
            e = eps[st.ep[k]]
            assert np.array_equal(F[b, :int(M[b].sum())], e["frames"][s:s + int(M[b].sum())])
            if s == 0:
                assert np.all(h0[b] == 0)
            else:
                assert where[int(k)] == (int(st.ep[k]), s - 4) and np.all(h0[b] == 100 + k)
            where[int(k)] = (int(st.ep[k]), s)
        st.update(np.stack([np.full(3, 100 + k, np.float32) for k in st.last]))
    assert drawn == set(range(6))


def test_confirm_stage_counts_handle_both_targets_and_split_done_after_press():
    """Review finding 13: p2_confirm.stages, incl. a target "off" row (a bool("off") slip would pass it)."""
    from p2_confirm import stages
    row = lambda **k: dict(dict(press_needed=True, declared_done=False, presses=0, plate_visits=0, ticks=50,   # noqa: E731
                                lamp_final=False, target="on"), **k)
    rows = [row(presses=1, plate_visits=1, lamp_final=True, declared_done=True),          # on: pressed, right, done
            row(target="off", presses=1, lamp_final=True, declared_done=True),            # off: pressed, wrong, done
            row(target="off", lamp_final=False),                                          # off: right, no press
            row(declared_done=True),                                                      # said done, no press
            row(press_needed=False, declared_done=True, ticks=40, lamp_final=True),
            row(press_needed=False, declared_done=True, ticks=60, lamp_final=True)]
    st = stages(rows)
    pn = st["press_needed"]
    assert (pn["n"], pn["reached_plate"], pn["pressed"], pn["lamp_right_at_end"]) == (4, 1, 2, 2)
    assert (pn["said_done"], pn["said_done_after_press"]) == (3, 2)
    assert (pn["done_after_press_lamp_right"], pn["done_after_press_lamp_wrong"]) == (1, 1)
    assert st["no_press"] == {"n": 2, "said_done": 2, "median_done_tick": 50.0}


@pytest.mark.parametrize("target,lamp_at_done,expected", [("on", 1.0, False), ("on", 0.0, True),
                                                          ("off", 0.0, False), ("off", 1.0, True)])
def test_done_label_filter_flags_only_a_done_with_the_lamp_off_target(tmp_path, target, lamp_at_done, expected):
    """Review finding 13: the filter reads truth[:, 5] (the lamp) at the bot's first done, and only flags."""
    import p2_recovery_index
    T = 6
    cb = np.zeros((T, 8), np.float32)
    cb[4, 7] = 1.0                                      # the bot's first done at tick 4
    truth = np.zeros((T, 7), np.float32)
    truth[:, 5] = 1.0 - lamp_at_done                    # the lamp is the other way everywhere else
    truth[4, 5] = lamp_at_done
    truth[:, 6] = 9.0                                   # presses column: must not be read as the lamp
    f = tmp_path / "ep.npz"
    np.savez(f, cmd_bot=cb, truth=truth,
             header=json.dumps({"task": "set_lamp", "instruction": {"task": "set_lamp", "target": target}}))
    assert p2_recovery_index.done_lamp_wrong(f) is expected
    np.savez(f, cmd_bot=np.zeros((T, 8), np.float32), truth=truth,
             header=json.dumps({"task": "set_lamp", "instruction": {"task": "set_lamp", "target": target}}))
    assert p2_recovery_index.done_lamp_wrong(f) is False             # no done: nothing to flag


def test_selection_takes_scheduled_checkpoints_only(tmp_path):
    """Review finding 13: guard-stop and stray saves are not candidates; ckpt_every comes from the first meta."""
    from p2_select import scheduled_checkpoints
    for name in ("ckpt_004000", "ckpt_005123", "ckpt_008000", "ckpt_011827", "ckpt_014000", "ckpt_016000"):
        d = tmp_path / name
        d.mkdir()
        (d / "weights.safetensors").write_bytes(b"x")
        (d / "meta.json").write_text(json.dumps({"cfg": {"ckpt_every": 4000}}))
    assert list(scheduled_checkpoints(tmp_path)) == ["ckpt_004000", "ckpt_008000", "ckpt_016000"]


def test_stream_warmup_uses_windows_first_then_streams(tmp_path, monkeypatch):
    """Review finding 13: steps before --stream-warmup use the window sampler, later steps the streams."""
    main_eps = [synthetic_episode(64, t) for t in (12, 18)]
    monkeypatch.setattr(p2_train, "load_all", lambda index, *a, **k: ([{}, {}], main_eps))
    calls = []
    real_batch, real_next = p2_train.batch, p2_train.Streams.next
    monkeypatch.setattr(p2_train, "batch", lambda *a, **k: (calls.append("window"), real_batch(*a, **k))[1])
    monkeypatch.setattr(p2_train.Streams, "next", lambda self, *a, **k: (calls.append("stream"), real_next(self, *a, **k))[1])
    monkeypatch.setattr(sys, "argv", ["p2_train.py", str(tmp_path / "r"), "--index", "unused", "--steps", "7",
                                     "--batch", "2", "--window", "4", "--burn", "1", "--act", "leaky",
                                     "--stream", "--stream-pool", "3", "--stream-warmup", "4"])
    p2_train.main()
    assert calls == ["window"] * 4 + ["stream"] * 3
