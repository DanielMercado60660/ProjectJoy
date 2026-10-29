"""Actual MLX smoke + kill/resume equivalence gate; Claude runs this inside gpu_run."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    started = time.monotonic()
    deadline = started + 110
    tag = time.strftime("smoke_%Y%m%d_%H%M%S")
    out = Path("results/r5") / tag
    ck = Path("checkpoints") / f"r5_{tag}"
    logs = Path("logs/r5") / tag
    for p in (out, ck, logs):
        p.mkdir(parents=True, exist_ok=False)
    processes = []

    def launch(name, script, flags):
        log = open(logs / f"{name}.log", "w")
        p = subprocess.Popen([sys.executable, script] + list(map(str, flags)),
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        processes.append(p)
        return p

    def wait(p):
        rc = p.wait(timeout=max(.01, deadline - time.monotonic()))
        if rc:
            raise RuntimeError(f"smoke subprocess rc={rc}; see {logs}")

    def run(name, script, flags):
        wait(launch(name, script, flags))

    try:
        run("prepare", "r5_data.py", ["--out", out / "data", "--episodes", 16,
            "--val-episodes", 4, "--seq-len", 384, "--max-minutes", .4])
        common = ["--data", out / "data", "--seed", 91, "--batch", 2, "--max-minutes", .5]
        tiny = common + ["--kind", "lm", "--init", "scratch", "--tiny", "--steps", 4]
        run("full", "r5_train.py", tiny + ["--out", ck / "full"])
        killed = launch("killed", "r5_train.py", tiny + ["--out", ck / "split",
                         "--save-steps", 2, "--pause-after-save-step", 2])
        while time.monotonic() < deadline:
            pointer = ck / "split/current.json"
            if pointer.exists() and json.loads(pointer.read_text())["step"] == 2:
                os.killpg(killed.pid, signal.SIGKILL)
                killed.wait(timeout=5)
                break
            if killed.poll() is not None:
                raise RuntimeError("kill gate exited before publishing step 2")
            time.sleep(.02)
        else:
            raise TimeoutError("kill gate")
        run("resumed", "r5_train.py", tiny + ["--out", ck / "split", "--resume"])
        import mlx.core as mx
        import numpy as np
        diffs = {}
        for filename in ("model.safetensors", "optim.safetensors", "rng.safetensors"):
            full = mx.load(str(ck / "full/selected" / filename))
            resumed = mx.load(str(ck / "split/selected" / filename))
            assert full.keys() == resumed.keys()
            max_diff = 0.
            for key in full:
                x, y = np.array(full[key]), np.array(resumed[key])
                np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6, err_msg=f"{filename}:{key}")
                max_diff = max(max_diff, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
            diffs[filename] = max_diff
        before = json.loads((ck / "split/resume_a/meta.json").read_text())
        after = json.loads((ck / "split/selected/meta.json").read_text())
        assert before["step"] == 2 and np.isfinite(before["last_loss"])
        assert after["step"] == after["optimizer_step"] == 4
        assert np.isfinite(after["last_loss"])
        # Cover the runner's MLX 0.32 RNG fix, including the next actual draw.
        saved_rng = mx.load(str(ck / "split/selected/rng.safetensors"))
        def next_draw():
            for key in range(len(saved_rng)):
                mx.random.state[key][:] = saved_rng[str(key)]
            return np.array(mx.random.normal((16,)))
        np.testing.assert_array_equal(next_draw(), next_draw())
        run("decode_tiny", "r5_decode_check.py", ["--checkpoint", ck / "full/selected",
            "--data", out / "data", "--out", out / "decode_tiny.json", "--examples", 4])
        # Use retained trained weights when available; no updates to the old run.
        trained = Path("checkpoints/r5_battery_s51/pretrained/selected")
        if trained.exists():
            run("decode_trained", "r5_decode_check.py", ["--checkpoint", trained,
                "--data", "results/r5/battery_s51/data", "--out", out / "decode_trained.json"])
        # Real pilot initialization, factored MLP and all output heads.
        run("pretrained", "r5_train.py", common + ["--out", ck / "pretrained", "--steps", 1])
        run("mlp", "r5_train.py", common + ["--kind", "mlp", "--init", "scratch",
            "--out", ck / "mlp", "--steps", 4])
        # Actual model rollouts, including the legitimate zero-success path.
        run("collect", "r5_eval.py", ["collect", "--checkpoint", ck / "full/selected",
            "--out", out / "reward", "--episodes", 2, "--max-minutes", .3])
        # Explicitly labelled smoke fixture exercises replay mixing even if the tiny
        # untrained policy earns zero rewards. It is never used for experiment results.
        run("replay_fixture", "r5_train.py", common + ["--from-checkpoint", ck / "full/selected",
            "--out", ck / "replay_fixture", "--replay", out / "data/train.npz", "--steps", 2])
        run("control", "r5_train.py", common + ["--from-checkpoint", ck / "full/selected",
            "--out", ck / "control", "--steps", 2])
        run("eval", "r5_eval.py", ["evaluate", "--data", out / "data", "--methods",
            f"pretrained={ck}/pretrained/selected", f"scratch={ck}/full/selected",
            f"mlp={ck}/mlp/selected", f"replay_fixture={ck}/replay_fixture/selected",
            f"control={ck}/control/selected", "--out", out / "eval.json",
            "--demo", f"media/r5_{tag}.html", "--smoke-one", "--max-minutes", .5])
        ev = json.loads((out / "eval.json").read_text())
        assert ev["complete"] and ev["paired_scenarios"] == 1
        from r5_data import json_write
        json_write(out / "smoke.json", dict(elapsed_s=time.monotonic() - started,
                   kill_resume_max_differences=diffs, resumed_step=after["step"],
                   final_loss=after["last_loss"], rng_next_draw_exact=True,
                   decode_gate="passed", result="R5 SMOKE OK"))
        print(f"R5 SMOKE OK ({time.monotonic()-started:.1f}s): {out}", flush=True)
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


if __name__ == "__main__":
    main()
