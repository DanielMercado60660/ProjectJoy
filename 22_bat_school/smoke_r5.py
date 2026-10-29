"""Claude-only subprocess smoke with a hard overall 114-second guard."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from io_r5 import Deadline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=1.9)
    a = ap.parse_args()
    deadline = Deadline(a.max_minutes)
    tag = "smoke_"+datetime.now().strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}"
    work = Path("logs/r5")/tag; work.mkdir(parents=True)
    python = sys.executable
    def run(*args):
        deadline.check()
        subprocess.run([python, *map(str, args)], check=True, timeout=max(.1, deadline.remaining))
    run("-m", "pytest", "-q", "tests_r5")
    run("check_mlx_r5.py")
    baseline = Path("results/r5")/(tag+"_baselines.json")
    run("tune_r5.py", "--per-group", 1, "--train-per-group", 2, "--max-candidates", 2,
        "--max-minutes", .4, "--out", baseline)
    out = Path("checkpoints")/("r5_"+tag+".npz")
    train_work = work/"kill"
    command = [python, "train_r5.py", "--max-minutes", ".5", "--envs", "8", "--horizon", "8", "--hidden", "16",
               "--epochs", "1", "--minibatches", "1", "--train-per-group", "2", "--val-per-group", "1",
               "--validate-updates", "1", "--save-every-update", "--work", str(train_work), "--out", str(out)]
    latest = train_work/"latest.npz"
    with (work/"kill_stdout.txt").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while not latest.exists():
                deadline.check()
                if proc.poll() is not None:
                    raise RuntimeError(f"train exited before checkpoint; see {log.name}")
                time.sleep(.05)
            # Force-kill only the child this smoke owns AFTER an atomic saved
            # generation exists. Two independent resumes replay that generation.
            proc.kill()
            proc.wait(timeout=max(.1, min(30., deadline.remaining)))
            if proc.returncode != -9:
                raise RuntimeError(f"signal checkpoint failed ({proc.returncode}); see {log.name}")
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    before = work/"before.npz"; shutil.copyfile(latest, before)
    for name in ("resume_a", "resume_b"):
        run("train_r5.py", "--resume", before, "--updates", 2, "--max-minutes", .4,
            "--work", work/name, "--out", work/(name+"_selected.npz"))
    run("check_mlx_r5.py", "--before", before, "--after", work/"resume_a/latest.npz", "--replay", work/"resume_b/latest.npz")
    selected = work/"resume_a_selected.npz"
    evaluation = Path("results/r5")/(tag+"_eval.json")
    run("eval_r5.py", "--ckpt", selected, "--baselines", baseline, "--per-group", 1,
        "--train-per-group", 2, "--val-per-group", 1, "--max-minutes", .4, "--out", evaluation)
    data = json.loads(evaluation.read_text())
    assert data["complete"] and len(data["methods"]) == 5
    assert all(len(m["rows"]) == 16 for m in data["methods"].values())
    for name in ("resume_a", "resume_b"):
        summary = json.loads((work/name/"summary.json").read_text())
        records = [json.loads(x) for x in (work/name/"train.jsonl").read_text().splitlines()]
        assert summary["target_reached"] and summary["last_validation_update"] == summary["update"]
        assert len(records) == 2 and all("reward_mean" in r and len(r["action_fractions"]) == 5 for r in records)
    fresh_evaluation = Path("results/r5")/(tag+"_turn2_eval.json")
    run("eval_r5.py", "--ckpt", selected, "--baselines", baseline, "--split", "test_turn2", "--per-group", 1,
        "--train-per-group", 2, "--val-per-group", 1, "--max-minutes", .4, "--out", fresh_evaluation)
    fresh = json.loads(fresh_evaluation.read_text())
    assert fresh["complete"] and not (set(fresh["manifest"]["hashes"]) & set(data["manifest"]["hashes"]))
    run("demo_r5.py", "--ckpt", selected, "--per-group", 1, "--max-minutes", .3,
        "--prefix", "media/r5_"+tag)
    print(f"SMOKE R5 OK in {deadline.elapsed:.1f}s; owned child saved/killed/resumed; {work}")


if __name__ == "__main__":
    main()
