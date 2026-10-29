"""Claude-only MLX smoke. Parent bounds the entire GPU portion to 90 seconds."""
import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import numpy as np


def assert_tree(a, b, path="state"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_tree(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tree(x, y, path + f"/{i}")
    elif isinstance(a, np.ndarray):
        assert np.isfinite(a).all() and np.isfinite(b).all(), path
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5, err_msg=path)
    else:
        assert a == b, path


def worker(tag):
    import mlx.core as mx
    from train_r5 import CONFIG, METHODS, R5Explorer, Progress, safe_clip, train_r5
    from evaluate_r5 import evaluate, save_eval
    from classical_r5 import train_classical
    from render_r5 import render
    from r5_core import VAL_SEEDS
    root = Path("logs/r5") / tag; root.mkdir(parents=True, exist_ok=True)
    check = Path("checkpoints") / f"r5_{tag}"; check.mkdir(parents=True, exist_ok=True)
    results = Path("results/r5") / tag; results.mkdir(parents=True, exist_ok=True)
    grad = safe_clip({"zero": mx.zeros((4,))})
    mx.eval(grad)
    assert np.isfinite(np.array(grad["zero"])).all()
    derivative = mx.grad(lambda x: mx.sum(safe_clip({"zero": x})["zero"]))(mx.zeros((4,)))
    mx.eval(derivative)
    assert np.isfinite(np.array(derivative)).all(), "zero-norm backward pass"
    from env import OBS_DIM
    progress = Progress(); zero = mx.zeros((2, OBS_DIM)); actions = mx.array([0, 1])
    np.testing.assert_allclose(np.array(progress.errors(zero, actions, zero)),
                               np.array(progress.errors(zero, actions, zero, old=True)), atol=0)
    for method in METHODS:
        cfg = dict(CONFIG, agent=method, seed=510, noisy_tv=True, n_envs=4, rollout=8, hidden=32,
                   epochs=1, minibatches=2, max_steps=32, ckpt_every=1, total_steps=128, max_minutes=1,
                   lp_lag=2)
        if method == 'lpm':
            cfg.update(lpm_window=1,total_steps=256)
        full = check / f"{method}_full.pkl"; part = check / f"{method}_part.pkl"
        train_r5(cfg, full, root / f"{method}_full")
        if method in ("lp", "lpm"):
            cmd = [sys.executable, "train_r5.py", "--method", method, "--seed", "510", "--world", "tv",
                   "--steps", str(cfg['total_steps']), "--max-minutes", "1", "--tiny", "--lp-lag", "2", "--barrier-update", "2",
                   "--checkpoint", str(part), "--log-dir", str(root / f"{method}_part")]
            with (root / f"kill_child_{method}.log").open("w") as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                try:
                    end = time.monotonic() + 35
                    while not Path(str(part) + ".ready").exists():
                        if child.poll() is not None:
                            raise RuntimeError(f"Checkpoint child died; see kill_child_{method}.log")
                        if time.monotonic() > end:
                            raise TimeoutError("Checkpoint barrier not reached")
                        time.sleep(.05)
                    child.kill(); child.wait(timeout=5)
                    assert child.returncode != 0
                finally:
                    if child.poll() is None:
                        child.kill(); child.wait()
        else:
            train_r5(cfg, part, root / f"{method}_part", stop_after_updates=2)
        train_r5(cfg, part, root / f"{method}_part", resume=True)
        a, b = (pickle.loads(p.read_bytes()) for p in (full, part))
        for k in a:
            if k not in ("cfg",):
                assert_tree(a[k], b[k], k)
        for world in (False, True):
            # Policy weights can be exercised in both env variants; match evaluation metadata.
            ex, _ = R5Explorer.load(part); ex.cfg["noisy_tv"] = world
            result, recording = evaluate(method, VAL_SEEDS[:2], tv=world, checkpoint=part, ex=ex,
                                         explore_steps=32, task_steps=16, max_expansions=64, record=True,
                                         planner_type='compositional' if method=='lpm' else 'replay')
            path = results / f"{method}_{world}.json"; save_eval(path, result, recording)
            if method == "lp" and world:
                render(path, Path("media") / f"r5_{tag}_lp.gif")
                again, _ = evaluate(method, VAL_SEEDS[:2], tv=True, checkpoint=part,
                                    explore_steps=32, task_steps=16, max_expansions=64)
                for key in ("episodes", "tasks", "transfer", "empty_replay"):
                    assert result[key] == again[key], f"reload reproduction: {key}"
        print(f"SMOKE_METHOD_OK {method}: finite, save/stop/resume matches uninterrupted", flush=True)
        if method=='lpm':
            logs=[json.loads(x) for x in (root/'lpm_part'/'updates.jsonl').read_text().splitlines()]
            assert logs[-1]['progress_ready'] and logs[-1]['queue_count']>0
            assert np.isfinite(logs[-1]['calibration_rmse'])
            assert 0 <= logs[-1]['policy_entropy'] <= np.log(5) + 1e-5
    for method in ("frontier", "random"):
        cfg = dict(method=method, seed=510, tv=True, checkpoint=check / f"{method}.pkl",
                   log_dir=root / method, max_minutes=.5, n_envs=2, episode_steps=16)
        train_classical(**cfg, steps=32)
        state = train_classical(**cfg, steps=64)
        assert state["steps"] == 64
        result, recording = evaluate(method, VAL_SEEDS[:2], tv=True, explore_steps=32, task_steps=16, record=True)
        path = results / f"{method}.json"; save_eval(path, result, recording)
        if method == "frontier":
            render(path, Path("media") / f"r5_{tag}_frontier.gif")
    print(f"SMOKE_R5_OK tag={tag}", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--worker", action="store_true"); p.add_argument("--tag", required=True)
    a = p.parse_args()
    if a.worker:
        worker(a.tag)
    else:
        proc = subprocess.Popen([sys.executable, __file__, "--worker", "--tag", a.tag], start_new_session=True)
        try:
            code = proc.wait(timeout=90)
            if code:
                raise RuntimeError(f"GPU smoke failed ({code}); keep the smoke artifacts")
        except BaseException:
            # Kill only this smoke's own process group, including the checkpoint child.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise


if __name__ == "__main__":
    main()
