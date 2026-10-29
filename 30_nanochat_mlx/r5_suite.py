"""One-slot sequential battery-first experiment. All children/logs belong to this run.

Invoke through ../bin/gpu_run. No fixed training-step caps. The outer guard includes
preparation, LR selection, reward collection, updates and evaluation.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from r5_data import json_write, TEST_START, SECONDARY_START


def power_status():
    p = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="battery_s51")
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--max-minutes", type=float, default=32.5)
    ap.add_argument("--sft-minutes", type=float, default=7)
    ap.add_argument("--eval-minutes", type=float, default=4)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--test-seed-start", type=int, default=TEST_START)
    ap.add_argument("--secondary-seed-start", type=int, default=SECONDARY_START)
    ap.add_argument("--secondary-episodes", type=int, default=1)
    ap.add_argument("--ac", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.tag):
        raise ValueError("tag must be a simple directory suffix")
    power = power_status()
    on_battery = "Battery Power" in power
    if on_battery:
        charge = re.search(r"(\d+)%;", power)
        if charge is None or int(charge[1]) < 50:
            raise RuntimeError(f"no new battery job below 50%: {power}")
        if a.ac or a.max_minutes > 33 or a.sft_minutes > 7:
            raise ValueError("AC extension requested on battery")
    if not 0 < a.max_minutes <= 45:
        raise ValueError("run must be <=45 min")
    if 2 + 3 * a.sft_minutes + 5 + a.eval_minutes > a.max_minutes:
        raise ValueError("stage allowances exceed total guard")
    start = time.monotonic()
    deadline = start + a.max_minutes * 60
    result = Path("results/r5") / a.tag
    logdir = Path("logs/r5") / a.tag
    root = Path("checkpoints") / f"r5_{a.tag}"
    if result.exists() or root.exists() or logdir.exists():
        raise FileExistsError("tag already exists; preserve receipts, use a new --tag")
    for p in (result, logdir, root):
        p.mkdir(parents=True)
    commands = []
    child = [None]

    def stop_child():
        if child[0] is not None and child[0].poll() is None:
            os.killpg(child[0].pid, signal.SIGTERM)
            try:
                child[0].wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child[0].pid, signal.SIGKILL)
                child[0].wait()

    def interrupted(signum, frame):
        stop_child()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    def run(name, script, flags, allowance):
        now = time.monotonic()
        if now >= deadline - 10:
            raise TimeoutError("outer wall guard; no new child")
        if "Battery Power" in power_status():
            # No new heavy stage below the brief's 50% threshold.
            charge = re.search(r"(\d+)%;", power_status())
            if charge is None or int(charge[1]) < 50:
                raise RuntimeError("battery below 50%; no new stage")
        cmd = [sys.executable, script] + list(map(str, flags))
        record = dict(name=name, command=cmd, allowance_s=allowance * 60,
                      started_s=now - start, log=str(logdir / f"{name}.log"))
        commands.append(record)
        json_write(result / "commands.json", dict(power=power, args=vars(a), commands=commands))
        print(f"START {name}: {' '.join(cmd)}", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   VECLIB_MAXIMUM_THREADS="4", RAYON_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        with open(record["log"], "w") as f:
            child[0] = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            child_deadline = min(deadline - 8, now + allowance * 60 + 8)
            while child[0].poll() is None and time.monotonic() < child_deadline:
                time.sleep(.2)
            timed_out = child[0].poll() is None
            if timed_out:
                stop_child()
            record.update(elapsed_s=time.monotonic() - now, returncode=child[0].returncode,
                          watchdog_timeout=timed_out)
        json_write(result / "commands.json", dict(power=power, args=vars(a), commands=commands))
        print(f"END {name}: {record['elapsed_s']:.1f}s, rc={record['returncode']}", flush=True)
        if timed_out or record["returncode"]:
            raise RuntimeError(f"{name} did not finish normally; inspect {record['log']}")

    data = result / "data"
    run("prepare", "r5_data.py", ["--out", data, "--max-minutes", 2], 2)
    common = ["--data", data, "--seed", a.seed]
    for name, init in (("pretrained", "pretrained"), ("scratch", "scratch")):
        run(name, "r5_train.py", common + ["--kind", "lm", "--init", init,
            "--out", root / name, "--max-minutes", a.sft_minutes], a.sft_minutes)
    # Tune MLP as seriously as the LM: three LRs on the same fixed val data. Its full
    # selection + final fit gets exactly the same wall allowance as each SFT arm.
    mlp_start = time.monotonic()
    trials = []
    for k, lr in enumerate((3e-4, 1e-3, 3e-3)):
        path = root / f"mlp_trial{k}"
        run(f"mlp_trial{k}", "r5_train.py", common + ["--kind", "mlp", "--init", "scratch",
            "--lr", lr, "--out", path, "--max-minutes", 1 / 3], 1 / 3)
        meta = json.loads((path / "selected/meta.json").read_text())
        trials.append(dict(lr=lr, path=str(path), val_loss=meta["val_loss"]))
    best = min(trials, key=lambda t: t["val_loss"])
    json_write(result / "mlp_tuning.json", dict(trials=trials, selected=best,
               selection_metric="teacher-forced validation joint action NLL"))
    remaining = a.sft_minutes - (time.monotonic() - mlp_start) / 60
    run("mlp_final", "r5_train.py", common + ["--kind", "mlp", "--init", "scratch",
        "--lr", best["lr"], "--out", best["path"], "--resume", "--max-minutes", remaining], remaining)
    mlp_path = Path(best["path"]) / "selected"
    pre_path = root / "pretrained/selected"
    reward_start = time.monotonic()
    run("reward_collect", "r5_eval.py", ["collect", "--checkpoint", pre_path,
        "--out", result / "reward", "--seed", a.seed, "--max-minutes", 1.25], 1.25)
    reward = json.loads((result / "reward/manifest.json").read_text())
    reward_path = pre_path
    if reward["replay"]:
        remaining = 2.5 - (time.monotonic() - reward_start) / 60
        run("reward_fit", "r5_train.py", common + ["--kind", "lm", "--from-checkpoint", pre_path,
            "--replay", reward["replay"], "--lr", 1e-4, "--out", root / "reward",
            "--max-minutes", remaining], remaining)
        reward_path = root / "reward/selected"
    else:
        print("NO SUCCESSFUL ROLLOUTS: reward arm unchanged; this is a measured failed attempt.", flush=True)
    run("sft_control", "r5_train.py", common + ["--kind", "lm", "--from-checkpoint", pre_path,
        "--lr", 1e-4, "--out", root / "control", "--max-minutes", 2.5], 2.5)
    methods = dict(pretrained=str(pre_path), scratch=str(root / "scratch/selected"),
                   mlp=str(mlp_path), reward=str(reward_path), control=str(root / "control/selected"))
    json_write(result / "selected.json", dict(methods=methods, reward_updated=bool(reward["replay"]),
               reward_note="extra train-distribution rollouts; compare against equal-time continued SFT",
               pretraining_note="pretrained arm has extra ClimbMix data; not matched total data/compute"))
    eval_flags = ["evaluate", "--data", data, "--methods"] + [f"{k}={v}" for k, v in methods.items()]
    run("evaluate", "r5_eval.py", eval_flags + ["--out", result / "eval.json", "--demo",
        f"media/r5_{a.tag}.html", "--episodes", a.eval_episodes, "--seed-start", a.test_seed_start,
        "--max-minutes", a.eval_minutes * .8], a.eval_minutes * .8)
    run("secondary", "r5_eval.py", eval_flags + ["--out", result / "secondary.json", "--demo",
        f"media/r5_{a.tag}_secondary.html", "--episodes", a.secondary_episodes, "--secondary",
        "--seed-start", a.secondary_seed_start,
        "--max-minutes", a.eval_minutes * .2], a.eval_minutes * .2)
    evaluation = json.loads((result / "eval.json").read_text())
    json_write(result / "finished.json", dict(elapsed_s=time.monotonic() - start,
               evaluation_complete=evaluation["complete"], power_end=power_status(),
               selected=str(result / "selected.json")))
    print("SUITE FINISHED", result, "evaluation_complete", evaluation["complete"], flush=True)


if __name__ == "__main__":
    main()
